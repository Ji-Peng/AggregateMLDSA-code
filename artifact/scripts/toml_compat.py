"""A small TOML reader for Python 3.10 and older.

The evaluator's Python 3.11 has `tomllib`; this fallback keeps the scripts
usable on older distributions. It supports the subset used by the files in
`artifact/config`: tables, arrays of tables, basic and literal strings,
integers, floats, booleans, arrays and inline tables.
"""

import json
import re


def _strip_comment(line):
    quote = None
    escaped = False
    for i, ch in enumerate(line):
        if quote is not None:
            if escaped:
                escaped = False
            elif ch == "\\" and quote == '"':
                escaped = True
            elif ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#":
            return line[:i]
    return line


def _split_top(text, separator=","):
    parts = []
    depth = 0
    quote = None
    escaped = False
    start = 0
    for i, ch in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif ch == "\\" and quote == '"':
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        elif ch == separator and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


def _find_equals(text):
    quote = None
    escaped = False
    for i, ch in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif ch == "\\" and quote == '"':
                escaped = True
            elif ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "=":
            return i
    return -1


def _key(text):
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        return json.loads(text)
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    return text


def _balanced(text):
    depth = 0
    quote = None
    escaped = False
    for ch in text:
        if quote is not None:
            if escaped:
                escaped = False
            elif ch == "\\" and quote == '"':
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
    return depth <= 0


def _value(text):
    text = text.strip()
    if not text:
        raise ValueError("empty value")
    if text.startswith('"'):
        return json.loads(text)
    if text.startswith("'"):
        if not text.endswith("'"):
            raise ValueError("unterminated literal string")
        return text[1:-1]
    if text == "true":
        return True
    if text == "false":
        return False
    if text.startswith("["):
        if not text.endswith("]"):
            raise ValueError("unterminated array")
        inner = text[1:-1].strip()
        return [] if not inner else [_value(p) for p in _split_top(inner)]
    if text.startswith("{"):
        if not text.endswith("}"):
            raise ValueError("unterminated inline table")
        table = {}
        inner = text[1:-1].strip()
        if inner:
            for item in _split_top(inner):
                eq = _find_equals(item)
                if eq < 0:
                    raise ValueError(f"bad inline table entry: {item!r}")
                table[_key(item[:eq])] = _value(item[eq + 1 :])
        return table
    number = text.replace("_", "")
    if re.fullmatch(r"[+-]?\d+", number):
        return int(number)
    if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?", number):
        return float(number)
    raise ValueError(f"unsupported value: {text!r}")


def loads(text):
    root = {}
    current = root
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = _strip_comment(lines[index]).strip()
        index += 1
        if not line:
            continue
        if line.startswith("[[") and line.endswith("]]"):
            path = [_key(p) for p in line[2:-2].split(".")]
            parent = root
            for part in path[:-1]:
                parent = parent.setdefault(part, {})
            entries = parent.setdefault(path[-1], [])
            if not isinstance(entries, list):
                raise ValueError(f"{line!r} conflicts with a table")
            current = {}
            entries.append(current)
            continue
        if line.startswith("[") and line.endswith("]"):
            path = [_key(p) for p in line[1:-1].split(".")]
            current = root
            for part in path:
                current = current.setdefault(part, {})
            continue
        eq = _find_equals(line)
        if eq < 0:
            raise ValueError(f"expected key = value: {line!r}")
        value_text = line[eq + 1 :]
        while not _balanced(value_text):
            if index >= len(lines):
                raise ValueError(f"unterminated value: {line!r}")
            value_text += " " + _strip_comment(lines[index]).strip()
            index += 1
        current[_key(line[:eq])] = _value(value_text)
    return root
