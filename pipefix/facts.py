"""Deterministic facts the agent looks up instead of inferring (the crew's 'A1').

  memory_need(script, params)       what the task's own command asks for (--buffer-gb, -Xmx, sort -S, samtools -m...)
  instance_for(cpus, memory_gib)    which HealthOmics instance a request lands on, and the usable memory after the
                                    5% HealthOmics keeps for its agents
  size_memory(need_gib, cpus)       the playbook's memory sizing: smallest whole-GiB request whose usable memory
                                    covers the need plus headroom, and the instance it lands on

The HealthOmics table is from docs.aws.amazon.com/omics/latest/dev/memory-and-compute-tasks.html (checked
2026-10-18): containers get exactly the requested vCPUs and memory (hard limits); HealthOmics picks the smallest
instance that accommodates both; ~5% of memory is reserved for management and logging agents; fractional units
are rounded up.
"""
from __future__ import annotations

import math
import re

RESERVED = 0.05          # HealthOmics: memory kept for management and logging agents
HEADROOM = 0.15          # playbook headroom over what the command needs (interpreter, buffers, page cache)

# (name, vCPUs, memory GiB), from the HealthOmics docs; GPU types are left out on purpose
HEALTHOMICS = [(f"omics.{fam}.{size}", v, m) for fam, sizes in {
    "c": [("large", 2, 4), ("xlarge", 4, 8), ("2xlarge", 8, 16), ("4xlarge", 16, 32), ("8xlarge", 32, 64),
          ("12xlarge", 48, 96), ("16xlarge", 64, 128), ("24xlarge", 96, 192), ("32xlarge", 128, 256),
          ("48xlarge", 192, 384)],
    "m": [("large", 2, 8), ("xlarge", 4, 16), ("2xlarge", 8, 32), ("4xlarge", 16, 64), ("8xlarge", 32, 128),
          ("12xlarge", 48, 192), ("16xlarge", 64, 256), ("24xlarge", 96, 384), ("32xlarge", 128, 512),
          ("48xlarge", 192, 768)],
    "r": [("large", 2, 16), ("xlarge", 4, 32), ("2xlarge", 8, 64), ("4xlarge", 16, 128), ("8xlarge", 32, 256),
          ("12xlarge", 48, 384), ("16xlarge", 64, 512), ("24xlarge", 96, 768), ("32xlarge", 128, 1024),
          ("48xlarge", 192, 1536)]}.items() for size, v, m in sizes]


def instance_for(cpus: float, memory_gib: float) -> dict:
    """The smallest HealthOmics instance for this request (fewest vCPUs, then least memory), and usable memory."""
    cpus, mem = math.ceil(max(cpus, 1)), math.ceil(max(memory_gib, 1))
    fits = sorted((x for x in HEALTHOMICS if x[1] >= cpus and x[2] >= mem), key=lambda x: (x[1], x[2]))
    out = {"requested_cpus": cpus, "requested_memory_gib": mem,
           "usable_memory_gib": round(mem * (1 - RESERVED), 2), "reserved_pct": int(RESERVED * 100)}
    if not fits:
        return {**out, "instance": None, "fits": False,
                "note": "no HealthOmics instance has this many vCPUs and this much memory; the run fails validation"}
    name, v, m = fits[0]
    return {**out, "instance": name, "instance_vcpus": v, "instance_memory_gib": m, "fits": True}


def size_memory(need_gib: float, cpus: float = 1, headroom: float = HEADROOM) -> dict:
    """Playbook sizing: request = ceil(need * (1 + headroom) / (1 - reserved)) GiB."""
    request = math.ceil(need_gib * (1 + headroom) / (1 - RESERVED))
    inst = instance_for(cpus, request)
    return {"need_gib": need_gib, "headroom_pct": int(headroom * 100), "reserved_pct": int(RESERVED * 100),
            "formula": "ceil(need_gib * (1 + headroom) / (1 - reserved))", "request_gib": request, **inst}


# ---------- what the command itself asks for ----------

_UNIT = {"": 1 / 1024**3, "b": 1 / 1024**3, "k": 1 / 1024**2, "m": 1 / 1024, "g": 1.0, "t": 1024.0}
_PATTERNS = [  # (name, regex with groups num + unit, default unit, note)
    ("--buffer-gb", r"--buffer-gb[ =]+(?P<num>\d+(?:\.\d+)?)()", "g", "sort/collect buffer in GiB"),
    ("-Xmx", r"-Xmx(?P<num>\d+(?:\.\d+)?)(?P<unit>[kKmMgGtT]?)", "", "Java heap"),
    ("--java-options -Xmx", r"--java-options\s+['\"]?-Xmx(?P<num>\d+(?:\.\d+)?)(?P<unit>[kKmMgGtT]?)", "", "GATK heap"),
    ("sort -S", r"\bsort\b[^|;\n]*?\s-S\s*(?P<num>\d+(?:\.\d+)?)(?P<unit>[kKmMgGtT%]?)", "k", "GNU sort buffer"),
    ("samtools sort -m", r"samtools\s+sort\b[^|;\n]*?\s-m\s*(?P<num>\d+(?:\.\d+)?)(?P<unit>[kKmMgG]?)", "", "per thread"),
    ("--limitBAMsortRAM", r"--limitBAMsortRAM\s+(?P<num>\d+)()", "", "STAR, bytes"),
    ("--mem", r"--mem(?:ory)?[ =]+(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>[kKmMgGtT]?)[bB]?\b", "g", "tool memory flag"),
    ("--max-memory", r"--max[-_]memory[ =]+(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>[kKmMgGtT]?)[bB]?\b", "g", "tool memory flag"),
]
_THREADS = re.compile(r"\s(?:-@|--threads)\s*(\d+)")


def resolve_params(script: str, params: dict) -> str:
    """${params.x} / $params.x -> value from nextflow.config params (strings unquoted); unknown ones stay."""
    def sub(m):
        v = params.get(m.group(1) or m.group(2))
        return str(v).strip("'\"") if v is not None else m.group(0)
    return re.sub(r"\$\{params\.(\w+)\}|\$params\.(\w+)", sub, script)


def memory_need(script: str, params: dict | None = None) -> dict:
    """Memory the task's own command asks for, from tool flags (after resolving params.*)."""
    text = resolve_params(script or "", params or {})
    found = []
    for name, rx, default_unit, note in _PATTERNS:
        for m in re.finditer(rx, text):
            unit = (m.groupdict().get("unit") or default_unit).lower()
            if unit == "%":
                continue
            gib = float(m.group("num")) * _UNIT.get(unit, 1.0)
            if name == "samtools sort -m":
                t = _THREADS.search(text)
                if t:
                    gib *= int(t.group(1))
                    note = f"per thread x {t.group(1)} threads"
            found.append({"flag": name, "text": m.group(0).strip(), "gib": round(gib, 3), "note": note})
    need = max((f["gib"] for f in found), default=None)
    return {"flags": found, "need_gib": need,
            "note": "largest explicit memory request in the command; add interpreter and buffer overhead"
            if found else "no memory flag found in the command; size from the allocation and peak usage instead"}
