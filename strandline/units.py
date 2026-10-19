"""Parse and format Nextflow memory and duration strings."""
from __future__ import annotations

import math
import re

_MEM = {"b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3, "tb": 1024**4}
_DUR = {"ms": 1, "s": 1000, "m": 60_000, "h": 3_600_000, "d": 86_400_000}


def parse_memory(text: str | None) -> int | None:
    """'15.8 GB' -> bytes. Returns None for '-', '' or anything unparseable."""
    if not text:
        return None
    m = re.fullmatch(r"\s*([\d.]+)\s*([kmgt]?b)\s*", text.strip(), re.I)
    if not m:
        return None
    return int(float(m.group(1)) * _MEM[m.group(2).lower()])


def parse_duration(text: str | None) -> int | None:
    """'1h 30m 2s' or '6h' -> milliseconds."""
    if not text or text.strip() == "-":
        return None
    parts = re.findall(r"([\d.]+)\s*(ms|s|m|h|d)\b", text.strip())
    if not parts:
        return None
    return int(sum(float(n) * _DUR[u] for n, u in parts))


def gb(n_bytes: int) -> float:
    return n_bytes / 1024**3


def ceil_gb(n_bytes: int, step: int = 2) -> int:
    """Round up to a whole number of GB, in steps (default even numbers)."""
    return max(step, step * math.ceil(gb(n_bytes) / step))


def ceil_hours(ms: int) -> int:
    return max(1, math.ceil(ms / 3_600_000))
