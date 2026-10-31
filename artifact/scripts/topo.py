#!/usr/bin/env python3
"""Read the CPU topology from sysfs and choose pinned CPU sets.

The three allocations of the evaluation are 16C32T, 8C16T and 1T. They are
built from SMT siblings, L3 domains (`index3/shared_cpu_list`, one core
complex on AMD) and NUMA nodes. A machine that is too small keeps the names it
can fill and labels the sets with the cores and threads it actually has.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import cpu_model, write_json  # noqa: E402

SYS = Path("/sys/devices/system/cpu")


def parse_list(text):
    """Parse a kernel CPU list such as `0-3,8,10-11`."""
    cpus = set()
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return cpus


def format_list(cpus):
    cpus = sorted(cpus)
    ranges = []
    start = prev = cpus[0]
    for cpu in cpus[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = cpu
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def read(path, default=None):
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return default


def topology():
    online_text = read(SYS / "online", "0")
    online = sorted(parse_list(online_text))
    cores = {}
    l3_of_cpu = {}
    node_of_cpu = {}

    for cpu in online:
        base = SYS / f"cpu{cpu}"
        core_id = read(base / "topology" / "core_id")
        package = read(base / "topology" / "physical_package_id", "0")
        if core_id is None:
            core_id = str(cpu)
        siblings = read(base / "topology" / "thread_siblings_list", str(cpu))
        key = (package, int(core_id))
        cores.setdefault(key, set()).update(parse_list(siblings) & set(online))
        shared = read(base / "cache" / "index3" / "shared_cpu_list")
        if shared is None:
            shared = online_text
        l3_of_cpu[cpu] = frozenset(parse_list(shared) & set(online))
        nodes = sorted(base.glob("node[0-9]*"))
        node_of_cpu[cpu] = int(nodes[0].name[4:]) if nodes else 0

    core_list = sorted(cores.items(), key=lambda item: min(item[1]))
    domain_cpus = sorted({frozenset(cpus) for cpus in l3_of_cpu.values()}, key=min)
    domains = []
    for index, cpus in enumerate(domain_cpus):
        domain_cores = [
            (key, core_cpus)
            for key, core_cpus in core_list
            if min(core_cpus) in cpus
        ]
        domains.append(
            {
                "id": index,
                "cpus": sorted(cpus),
                "cores": [key for key, _ in domain_cores],
                "numa_node": node_of_cpu[min(cpus)],
            }
        )

    smt = any(len(cpus) > 1 for cpus in cores.values())
    numa_nodes = sorted(set(node_of_cpu.values()))
    return online, core_list, domains, smt, numa_nodes, node_of_cpu


def cores_for_domain(domain, core_list, avoid_zero=True):
    selected = [
        (key, cpus) for key, cpus in core_list if min(cpus) in set(domain["cpus"])
    ]
    if avoid_zero:
        selected.sort(key=lambda item: (0 in item[1], min(item[1])))
    else:
        selected.sort(key=lambda item: min(item[1]))
    return selected


def make_config(name, selected_cores, domains, smt, numa_node, note=None):
    cpus = set()
    for _, core_cpus in selected_cores:
        cpus.update(core_cpus)
    ncores = len(selected_cores)
    nthreads = len(cpus)
    label = name
    if name in ("8C16T", "16C32T"):
        expected = 8 if name == "8C16T" else 16
        if ncores < expected:
            label = f"{ncores}C{nthreads}T"
    elif name == "1T":
        label = f"{ncores}C{nthreads}T"
    used_domains = sorted(
        {
            domain["id"]
            for domain in domains
            for _, core_cpus in selected_cores
            if min(core_cpus) in set(domain["cpus"])
        }
    )
    result = {
        "available": ncores > 0,
        "label": label,
        "cpus": sorted(cpus),
        "cpuset": format_list(cpus),
        "cores": ncores,
        "threads": nthreads,
        "domains": used_domains,
        "numa_node": numa_node,
    }
    if note:
        result["note"] = note
    return result


def choose(online, core_list, domains, smt, numa_nodes, node_of_cpu, avoid_cpu0):
    configs = {}
    node = numa_nodes[0]
    node_cores = [
        (key, cpus) for key, cpus in core_list if node_of_cpu[min(cpus)] == node
    ]
    node_domains = [domain for domain in domains if domain["numa_node"] == node]

    # 8C16T: one L3 domain with eight cores when one exists.
    chosen = []
    exact = [
        domain
        for domain in node_domains
        if len(cores_for_domain(domain, node_cores, avoid_cpu0)) >= 8
    ]
    exact.sort(
        key=lambda domain: abs(len(cores_for_domain(domain, node_cores, avoid_cpu0)) - 8)
    )
    if exact:
        chosen = cores_for_domain(exact[0], node_cores, avoid_cpu0)[:8]
    else:
        for domain in sorted(node_domains, key=lambda d: -len(d["cpus"])):
            for item in cores_for_domain(domain, node_cores, avoid_cpu0):
                if item not in chosen and len(chosen) < 8:
                    chosen.append(item)
    configs["8C16T"] = make_config(
        "8C16T", chosen, node_domains, smt, node,
        None if len(chosen) == 8 else "all cores on the node",
    )

    # 16C32T: two eight-core domains, or every core when the machine has fewer
    # than sixteen and the user asked for the largest set.
    wide = []
    if len(node_cores) >= 16:
        for domain in sorted(
            node_domains,
            key=lambda d: -len(cores_for_domain(d, node_cores, avoid_cpu0)),
        ):
            if len(wide) >= 16:
                break
            for item in cores_for_domain(domain, node_cores, avoid_cpu0):
                if item not in wide and len(wide) < 16:
                    wide.append(item)
        configs["16C32T"] = make_config("16C32T", wide, node_domains, smt, node)
    else:
        configs["16C32T"] = {
            "available": False,
            "label": "n/a",
            "reason": f"{len(node_cores)} cores on the node, 16 required",
            "cores": 0,
            "threads": 0,
            "cpus": [],
            "cpuset": "",
            "domains": [],
            "numa_node": node,
        }

    # 1T: one thread of a core in the 8C16T set, sibling left idle.
    base = chosen if chosen else node_cores
    if base:
        pick_core = base[1] if len(base) > 1 and avoid_cpu0 else base[0]
        cpus = sorted(pick_core[1])
        one = cpus[-1] if cpus[0] == 0 else cpus[0]
        configs["1T"] = make_config("1T", [(pick_core[0], {one})], node_domains, smt, node)
    else:
        configs["1T"] = {
            "available": False,
            "label": "n/a",
            "reason": "no online CPU",
            "cores": 0,
            "threads": 0,
            "cpus": [],
            "cpuset": "",
            "domains": [],
            "numa_node": node,
        }

    configs["all"] = make_config("all", node_cores, node_domains, smt, node)
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="write cpusets.json here")
    parser.add_argument(
        "--no-avoid-cpu0", action="store_true", help="allow CPU 0 in every set"
    )
    args = parser.parse_args()
    online, core_list, domains, smt, numa_nodes, node_of_cpu = topology()
    configs = choose(
        online, core_list, domains, smt, numa_nodes, node_of_cpu, not args.no_avoid_cpu0
    )
    document = {
        "cpu_model": cpu_model(),
        "online_cpus": online,
        "smt": smt,
        "numa_nodes": numa_nodes,
        "domains": [
            {
                "id": domain["id"],
                "cpus": domain["cpus"],
                "cpuset": format_list(domain["cpus"]),
                "cores": len(domain["cores"]),
                "numa_node": domain["numa_node"],
            }
            for domain in domains
        ],
        "configs": configs,
    }
    if args.out:
        write_json(args.out, document)
    else:
        print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
