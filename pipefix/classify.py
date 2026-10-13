"""Classify a failed task into a known failure type, with the evidence that led there.

Rules are ordered and deterministic: every diagnosis can be traced to a specific exit code, number or log line.
An LLM can be layered on top for explanations, but the decision itself stays auditable.
"""
from __future__ import annotations

import re

from .models import Diagnosis, FailureClass, TaskFailure
from .units import gb

# (class, pattern, weight). Patterns match error lines and AWS status reasons, case-insensitive.
PATTERNS: list[tuple[FailureClass, str, float]] = [
    (FailureClass.SPOT_INTERRUPTION, r"Host EC2 \(instance [^)]+\) terminated", 0.95),
    (FailureClass.SPOT_INTERRUPTION, r"spot.?(instance)?.?interrupt|SpotInterruption|capacity.?rebalanc", 0.9),
    (FailureClass.OUT_OF_MEMORY, r"OutOfMemoryError: Container killed due to memory usage", 0.97),
    (FailureClass.OUT_OF_MEMORY, r"java\.lang\.OutOfMemoryError|GC overhead limit exceeded", 0.92),
    (FailureClass.OUT_OF_MEMORY, r"std::bad_alloc|Cannot allocate memory|MemoryError|out of memory|oom-kill", 0.88),
    (FailureClass.TIMEOUT, r"exceeded running time limit|DUE TO TIME LIMIT|walltime|TIMEOUT", 0.9),
    (FailureClass.DISK_FULL, r"No space left on device|Disk quota exceeded|ENOSPC", 0.95),
    (FailureClass.CONTAINER, r"CannotPullContainerError|manifest unknown|pull access denied|"
                             r"repository does not exist|ImagePull", 0.95),
    (FailureClass.CONTAINER, r"command not found|exec format error|GLIBC_[\d.]+' not found", 0.8),
    (FailureClass.REFERENCE, r"Fasta (index|dict) file .* (does not exist|not found)|\[E::fai_load\]|"
                             r"could not (load|open) .*\.(fai|dict|bwt|amb|ann|pac|sa)\b|"
                             r"Genome (index|directory) .* not (found|exist)|EXITING because of FATAL ERROR.*genome", 0.9),
    (FailureClass.REFERENCE, r"contig .* (not found|not present) in (the )?(reference|sequence dictionary)|"
                             r"Input files reference and .* have incompatible contigs|incompatible contigs", 0.9),
]

RECOMMEND = {
    FailureClass.OUT_OF_MEMORY: "Raise memory for this process only and retry with escalating memory on OOM exit codes.",
    FailureClass.SPOT_INTERRUPTION: "Retry automatically on Spot reclamation; long tasks can fall back to an On-Demand queue.",
    FailureClass.TIMEOUT: "Raise the time limit for this process and scale it with each retry.",
    FailureClass.DISK_FULL: "Give the task more scratch space: enlarge the Batch compute environment's EBS volume "
                            "(launch template) or enable Fusion/ephemeral NVMe; for HealthOmics, raise the run's "
                            "storage capacity or use DYNAMIC storage.",
    FailureClass.CONTAINER: "Check the process's container image and tag exist and are reachable from the compute "
                            "environment (ECR permissions, VPC endpoints), and that the tool is inside the image.",
    FailureClass.REFERENCE: "Check the reference genome and its index files match each other and the inputs "
                            "(same build, same contig names, e.g. 'chr1' vs '1'); regenerate missing .fai/.dict/index.",
    FailureClass.UNKNOWN: "No known signature matched. Inspect .command.err in the task's work directory.",
}


def _text(f: TaskFailure) -> list[str]:
    return [f.status_reason] + f.error_lines if f.status_reason else list(f.error_lines)


def classify(f: TaskFailure) -> Diagnosis:
    scores: dict[FailureClass, float] = {}
    evidence: dict[FailureClass, list[str]] = {}

    def add(cls: FailureClass, score: float, why: str):
        scores[cls] = max(scores.get(cls, 0.0), score)
        evidence.setdefault(cls, []).append(why)

    for line in _text(f):
        for cls, pat, w in PATTERNS:
            if re.search(pat, line, re.I):
                add(cls, w, f"log: `{line.strip()[:200]}`")
                break

    # Numeric signals
    near_limit = (f.memory_bytes and f.peak_rss_bytes and f.peak_rss_bytes >= 0.9 * f.memory_bytes)
    if f.exit_code in (137, 247) or (f.exit_code == 1 and near_limit):
        # 137 = SIGKILL. On its own it can also be Spot reclamation or an external kill, so weigh with memory use.
        if near_limit:
            add(FailureClass.OUT_OF_MEMORY, 0.95,
                f"exit {f.exit_code}; peak memory {gb(f.peak_rss_bytes):.1f} GB of {gb(f.memory_bytes):.1f} GB requested")
        elif FailureClass.SPOT_INTERRUPTION not in scores:
            add(FailureClass.OUT_OF_MEMORY, 0.7, f"exit {f.exit_code} (process killed, typical of out-of-memory)")
    elif near_limit:
        add(FailureClass.OUT_OF_MEMORY, 0.6,
            f"peak memory {gb(f.peak_rss_bytes):.1f} GB of {gb(f.memory_bytes):.1f} GB requested")
    if f.exit_code == 140:
        add(FailureClass.TIMEOUT, 0.75, "exit 140 (killed by the scheduler, usually a time limit)")
    if f.time_ms and f.realtime_ms and f.realtime_ms >= 0.98 * f.time_ms:
        add(FailureClass.TIMEOUT, 0.9, "task ran for its full time limit")
    if f.exit_code == 127:
        add(FailureClass.CONTAINER, 0.85, "exit 127 (command not found in the container)")

    if not scores:
        return Diagnosis(f, FailureClass.UNKNOWN, 0.0,
                         [f"exit {f.exit_code}" if f.exit_code is not None else "no exit code"],
                         "No known failure signature matched.", RECOMMEND[FailureClass.UNKNOWN])

    # A Spot reclamation explains a SIGKILL better than memory, unless memory was clearly at the limit.
    if FailureClass.SPOT_INTERRUPTION in scores and FailureClass.OUT_OF_MEMORY in scores and not near_limit:
        scores[FailureClass.OUT_OF_MEMORY] = min(scores[FailureClass.OUT_OF_MEMORY], 0.5)

    best = max(scores, key=scores.get)
    others = sorted((c for c in scores if c != best), key=scores.get, reverse=True)
    expl = f"{best.label} in {f.simple_name}" + (f" ({f.tag})" if f.tag else "") + "."
    if others:
        expl += " Also possible: " + ", ".join(f"{c.label} ({scores[c]:.0%})" for c in others) + "."
    return Diagnosis(f, best, round(scores[best], 2), evidence[best], expl, RECOMMEND[best])


def classify_all(failures: list[TaskFailure]) -> list[Diagnosis]:
    return [classify(f) for f in failures]
