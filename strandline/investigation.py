"""The result of investigating one failed run: failed tasks, platform facts, access denials, and what was read."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .models import TaskFailure


@dataclass
class Finding:
    """A fact about the platform or config that matters for the fix (shown in the report and the MR)."""
    title: str
    detail: str
    severity: str = "info"     # info | warn | block
    process: str = ""          # simple process name it applies to; "" = the whole run
    data: dict = field(default_factory=dict)


@dataclass
class Investigation:
    platform: str                                   # aws-batch | healthomics | files
    run_ref: str
    failures: list[TaskFailure] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    denials: list[dict] = field(default_factory=list)   # CloudTrail access denials in the run window
    calls: list[str] = field(default_factory=list)      # every AWS API call made (audit trail)
    errors: list[str] = field(default_factory=list)     # calls that were refused or found nothing
    start: datetime | None = None
    end: datetime | None = None
    nextflow_version: str = ""
    facts: dict = field(default_factory=dict)           # machine-readable facts the patcher uses

    def add(self, title: str, detail: str, severity: str = "info", process: str = "", **data) -> None:
        self.findings.append(Finding(title, detail, severity, process, data))
