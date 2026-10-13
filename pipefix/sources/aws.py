"""Offline AWS inputs: saved output of `aws batch describe-jobs` or `aws omics get-run` / `get-run-task`.

For live investigation through the SDK, see pipefix.aws (used by `pipefix investigate`).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..aws.batch import job_to_failure
from ..aws.omics import task_to_failure
from ..models import TaskFailure


def batch_failures_from_json(doc: dict) -> list[TaskFailure]:
    return [job_to_failure(j) for j in doc.get("jobs", []) if j.get("status") == "FAILED"]


def omics_failures_from_json(run: dict, tasks: list[dict] | None = None) -> list[TaskFailure]:
    out = [task_to_failure(t, run) for t in tasks or [] if t.get("status") == "FAILED"]
    if not out and run.get("status") == "FAILED":
        out.append(TaskFailure(process="(run)", status_reason=" | ".join(
            x for x in (run.get("failureReason"), run.get("statusMessage")) if x), source="healthomics"))
    return out


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
