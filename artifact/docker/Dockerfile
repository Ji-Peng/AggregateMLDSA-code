# Functional reproduction in a container.
#
# Build from the repository root:
#   docker build -f artifact/docker/Dockerfile -t aggmldsa-ae .
#   docker run --rm --network none --memory=48g \
#     --cpuset-cpus=0-15 --cpuset-mems=0 \
#     -v "$PWD/artifact/results:/work/artifact/results" \
#     aggmldsa-ae ./artifact/run.sh check
#
# Timing caveat: the container shares the host kernel, so THP, the frequency
# governor, boost and co-tenancy are host properties. The probe records them
# but cannot change them. Do not use --cpus: the quota throttles CPU-bound
# loops. CPU-bound timings are expected to match native runs, but that is an
# expectation, not a measurement; see README "Docker".
FROM rust:1.99.1-bookworm@sha256:0097cc72a939f26665ab11380e995aa9476735ec9516867e14ff5dfa8318b068

ENV DEBIAN_FRONTEND=noninteractive \
    CARGO_BUILD_JOBS=6 \
    RUSTUP_TOOLCHAIN=1.99.1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        numactl \
        time \
        util-linux \
        python3 \
    && rm -rf /var/lib/apt/lists/*

# Aerie pins its own toolchain.
RUN rustup toolchain install 1.95.0 --profile minimal

WORKDIR /work
COPY . /work

# The build is intentionally part of the image so that `run.sh check` in a
# container has the same binaries as a native run.
RUN ./artifact/run.sh fetch --no-cargo && ./artifact/run.sh build

ENTRYPOINT ["/work/artifact/run.sh"]
CMD ["check"]

# Nix packaging is future work: a flake would pin gcc and both Rust
# toolchains, but the timing caveats are the same as for Docker.
