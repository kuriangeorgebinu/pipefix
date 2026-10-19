"""Plain data types shared by the collectors, the classifier and the patcher."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FailureClass(str, Enum):
    OUT_OF_MEMORY = "out_of_memory"
    SPOT_INTERRUPTION = "spot_interruption"
    TIMEOUT = "timeout"
    DISK_FULL = "disk_full"
    CONTAINER = "container_error"
    REFERENCE = "reference_error"
    PERMISSION = "permission_denied"
    CAPACITY = "capacity"
    UNKNOWN = "unknown"

    @property
    def label(self) -> str:
        return {
            "out_of_memory": "Out of memory",
            "spot_interruption": "Spot instance interruption",
            "timeout": "Time limit exceeded",
            "disk_full": "Disk full",
            "container_error": "Container / tool error",
            "reference_error": "Reference or input problem",
            "permission_denied": "Permission denied (IAM)",
            "capacity": "No capacity / resources too large",
            "unknown": "Unknown",
        }[self.value]

    @property
    def auto_fixable(self) -> bool:
        """Classes where a pipeline config change is reliably the right fix."""
        return self in (FailureClass.OUT_OF_MEMORY, FailureClass.SPOT_INTERRUPTION, FailureClass.TIMEOUT)


@dataclass
class TaskFailure:
    """One failed task, normalised from whatever source reported it."""
    process: str                       # fully qualified, e.g. NFCORE_SAREK:SAREK:GATK4_HAPLOTYPECALLER
    tag: str = ""                      # sample / chunk tag, e.g. "sample1"
    exit_code: int | None = None
    memory_bytes: int | None = None    # requested
    peak_rss_bytes: int | None = None  # used
    time_ms: int | None = None         # requested
    realtime_ms: int | None = None     # used
    cpus: int | None = None
    status_reason: str = ""            # AWS Batch / HealthOmics status reason or failure reason code
    error_lines: list[str] = field(default_factory=list)
    source: str = "nextflow"
    hash: str = ""                     # Nextflow task hash, e.g. e5/f6a7b8
    work_dir: str = ""                 # local path or s3:// URI
    attempts: int = 1                  # how many times the platform ran it
    spot_reclaims: int = 0             # attempts lost to Spot reclamation
    job_ref: str = ""                  # Batch job id / HealthOmics run:task id
    image: str = ""                    # container image
    context: dict = field(default_factory=dict)  # extra facts from the platform (queue, instance, storage...)

    @property
    def simple_name(self) -> str:
        """GATK4_HAPLOTYPECALLER from NFCORE_SAREK:SAREK:...:GATK4_HAPLOTYPECALLER."""
        return self.process.split(":")[-1].strip()

    @property
    def display(self) -> str:
        return self.simple_name + (f" ({self.tag})" if self.tag else "")


@dataclass
class Diagnosis:
    failure: TaskFailure
    cls: FailureClass
    confidence: float                  # 0..1
    evidence: list[str]
    explanation: str
    recommendation: str
