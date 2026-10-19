"""Classify a failed task into a known failure type, with the evidence that led there.

Rules are ordered and deterministic: every diagnosis can be traced to a specific exit code, number or log line.
Sources of signal: the tool's own stderr, Nextflow's log, AWS Batch status reasons, AWS HealthOmics failure
reason codes (documented list), and resource numbers (requested vs peak memory, time limit vs run time).
"""
from __future__ import annotations

import re

from .models import Diagnosis, FailureClass, TaskFailure
from .units import gb

F = FailureClass
# (class, pattern, weight). First match per line wins, so more specific patterns come first.
PATTERNS: list[tuple[FailureClass, str, float]] = [
    # AWS HealthOmics failure reason codes (docs: workflows-run-errors)
    (F.OUT_OF_MEMORY, r"\bOUT_OF_MEMORY_ERROR\b", 0.97),
    (F.TIMEOUT, r"\bTASK_TIMED_OUT\b", 0.95),
    (F.TIMEOUT, r"\bRUN_TIMED_OUT\b", 0.9),
    (F.DISK_FULL, r"\bFILE_SYSTEM_OUT_OF_SPACE\b", 0.97),
    (F.PERMISSION, r"\b(ECR_PERMISSION_ERROR|ASSUME_ROLE_FAILED|EXPORT_FAILED)\b", 0.95),
    (F.CONTAINER, r"\b(CANNOT_START_CONTAINER_ERROR|CANNOT_START_CONTAINER_SIZE_ERROR|INVALID_ECR_IMAGE_URI|"
                  r"IMAGE_VERIFICATION_FAILURE)\b", 0.95),
    (F.CAPACITY, r"\b(INVALID_TASK_RESOURCE_VALUE|INSTANCE_RESERVATION_FAILED|UNSUPPORTED_GPU_INSTANCE_TYPE)\b", 0.95),
    (F.REFERENCE, r"\b(INPUT_URI_NOT_FOUND|INVALID_URI_INPUT|IMPORT_FAILED|INACTIVE_OMICS_STORAGE_RESOURCE)\b", 0.9),
    # AWS Batch status reasons
    (F.SPOT_INTERRUPTION, r"Host EC2 \(instance [^)]+\) terminated", 0.95),
    (F.SPOT_INTERRUPTION, r"spot.?(instance)?.?interrupt|SpotInterruption|capacity.?rebalanc", 0.9),
    (F.OUT_OF_MEMORY, r"OutOfMemoryError: Container killed due to memory usage", 0.97),
    (F.TIMEOUT, r"Job attempt duration exceeded timeout", 0.95),
    (F.CAPACITY, r"MISCONFIGURATION:JOB_RESOURCE_REQUIREMENT|resource requirement .* higher than", 0.95),
    # Permissions before container errors: "CannotPullContainerError: ... AccessDenied" is a permission problem
    (F.PERMISSION, r"AccessDenied|Access Denied|not authorized to perform|is not authorized|403 Forbidden|"
                   r"ResourceInitializationError: unable to pull secrets", 0.9),
    # Tool stderr
    (F.OUT_OF_MEMORY, r"java\.lang\.OutOfMemoryError|GC overhead limit exceeded", 0.92),
    (F.OUT_OF_MEMORY, r"std::bad_alloc|(cannot|can't|couldn't|could not|failed to|unable to) allocate( \w+)? memory|"
                      r"MemoryError|out of memory|oom-kill|Killed process .* total-vm", 0.88),
    (F.TIMEOUT, r"exceeded running time limit|DUE TO TIME LIMIT|walltime", 0.9),
    (F.DISK_FULL, r"No space left on device|Disk quota exceeded|ENOSPC", 0.95),
    (F.CONTAINER, r"CannotPullContainerError|manifest unknown|pull access denied|repository does not exist|"
                  r"ImagePull", 0.95),
    (F.CONTAINER, r"command not found|exec format error|GLIBC_[\d.]+' not found", 0.8),
    (F.REFERENCE, r"Fasta (index|dict) file .* (does not exist|not found)|\[E::fai_load\]|"
                  r"could not (load|open) .*\.(fai|dict|bwt|amb|ann|pac|sa)\b|"
                  r"Genome (index|directory) .* not (found|exist)|EXITING because of FATAL ERROR.*genome", 0.9),
    (F.REFERENCE, r"contig .* (not found|not present) in (the )?(reference|sequence dictionary)|"
                  r"incompatible contigs", 0.9),
    (F.REFERENCE, r"NoSuchKey|The specified key does not exist", 0.75),
]

RECOMMEND = {
    F.OUT_OF_MEMORY: "Raise memory for this process only, and retry with escalating memory on kill/OOM exit codes.",
    F.SPOT_INTERRUPTION: "Let AWS Batch retry Spot reclamations (aws.batch.maxSpotAttempts); tasks reclaimed "
                         "repeatedly or that run for hours belong on an On-Demand queue.",
    F.TIMEOUT: "Raise the time limit for this process and scale it with each retry.",
    F.DISK_FULL: "Give the task more scratch space: a larger EBS volume in the Batch compute environment's launch "
                 "template (or Fusion / instance NVMe). For HealthOmics, start the run with DYNAMIC storage or a "
                 "larger storage capacity.",
    F.CONTAINER: "Check the container image and tag exist and are reachable from the compute environment, and that "
                 "the tool is inside the image.",
    F.REFERENCE: "Check the reference genome, its index files and the inputs match (same build, same contig names, "
                 "e.g. 'chr1' vs '1') and exist at the given paths; regenerate missing .fai/.dict/index.",
    F.PERMISSION: "Grant the run's IAM role the denied action on the resource named in the evidence (least "
                  "privilege: that action, that resource). This lives in your IAM/IaC, not the pipeline.",
    F.CAPACITY: "The task asks for more CPU/memory than any instance the queue (or HealthOmics) can provide: "
                "lower the request or add a larger instance type to the compute environment.",
    F.UNKNOWN: "No known signature matched. Inspect .command.err in the task's work directory.",
}

KILL_CODES = (137, 247)


def _text(f: TaskFailure) -> list[str]:
    return ([f.status_reason] if f.status_reason else []) + list(f.error_lines)


def classify(f: TaskFailure) -> Diagnosis:
    scores: dict[FailureClass, float] = {}
    evidence: dict[FailureClass, list[str]] = {}

    def add(cls: FailureClass, score: float, why: str):
        scores[cls] = max(scores.get(cls, 0.0), score)
        if why not in evidence.setdefault(cls, []):
            evidence[cls].append(why)

    for line in _text(f):
        line = re.sub(r"^\[[0-9a-f-]{4,12}\]\s*", "", line.strip())   # drop the log-stream prefix
        for cls, pat, w in PATTERNS:
            if re.search(pat, line, re.I):
                add(cls, w, f"log: `{line[:200]}`")
                break

    near_limit = bool(f.memory_bytes and f.peak_rss_bytes and f.peak_rss_bytes >= 0.9 * f.memory_bytes)
    used = (f"peak memory {gb(f.peak_rss_bytes):.1f} GB of {gb(f.memory_bytes):.1f} GB requested"
            if f.memory_bytes and f.peak_rss_bytes else "")
    if f.exit_code in KILL_CODES:
        if near_limit:
            add(F.OUT_OF_MEMORY, 0.95, f"exit {f.exit_code} (killed); {used}")
        elif F.SPOT_INTERRUPTION not in scores:
            add(F.OUT_OF_MEMORY, 0.7, f"exit {f.exit_code} (process killed, typical of out-of-memory)")
    elif near_limit:
        add(F.OUT_OF_MEMORY, 0.6, used)
    if f.spot_reclaims:
        add(F.SPOT_INTERRUPTION, 0.97, f"{f.spot_reclaims} of {f.attempts} attempt(s) lost to Spot reclamation")
    if f.time_ms and f.realtime_ms and f.realtime_ms >= 0.98 * f.time_ms:
        add(F.TIMEOUT, 0.9, "task ran for its full time limit")
    if f.exit_code == 140:  # SIGUSR2; some schedulers use it for a run-time limit. Weak on its own.
        add(F.TIMEOUT, 0.5, "exit 140 (some schedulers use this for a time limit)")
    if f.exit_code == 127:
        add(F.CONTAINER, 0.85, "exit 127 (command not found in the container)")

    if not scores:
        return Diagnosis(f, F.UNKNOWN, 0.0, [f"exit {f.exit_code}" if f.exit_code is not None else "no exit code"],
                         "No known failure signature matched.", RECOMMEND[F.UNKNOWN])

    # A Spot reclamation explains a kill better than memory, unless memory was clearly at the limit.
    if F.SPOT_INTERRUPTION in scores and F.OUT_OF_MEMORY in scores and not near_limit:
        scores[F.OUT_OF_MEMORY] = min(scores[F.OUT_OF_MEMORY], 0.5)

    best = max(scores, key=scores.get)
    others = sorted((c for c in scores if c != best), key=scores.get, reverse=True)
    expl = f"{best.label} in {f.display}."
    if others:
        expl += " Also possible: " + ", ".join(f"{c.label} ({scores[c]:.0%})" for c in others) + "."
    return Diagnosis(f, best, round(scores[best], 2), evidence[best], expl, RECOMMEND[best])


def classify_all(failures: list[TaskFailure]) -> list[Diagnosis]:
    return [classify(f) for f in failures]
