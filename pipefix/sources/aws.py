"""Read failures straight from AWS: AWS Batch jobs (with their CloudWatch logs) and AWS HealthOmics runs.

Needs boto3 (`pip install pipefix[aws]`) and read-only permissions:
    batch:DescribeJobs, logs:GetLogEvents, omics:GetRun, omics:ListRunTasks, omics:GetRunTask
The functions also accept the JSON that `aws batch describe-jobs` / `aws omics get-run` print, so they can be
tested offline and used without credentials.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..models import TaskFailure
from ..units import parse_memory


def _client(service: str, region: str | None):
    try:
        import boto3
    except ImportError as e:  # pragma: no cover - depends on the install
        raise SystemExit("AWS access needs boto3: pip install 'pipefix[aws]'") from e
    return boto3.client(service, region_name=region)


# ---------- AWS Batch ----------

def batch_failures_from_json(doc: dict, log_lines: dict[str, list[str]] | None = None) -> list[TaskFailure]:
    """Failed jobs from a `describe-jobs` response. Nextflow names Batch jobs after the process."""
    out = []
    for job in doc.get("jobs", []):
        if job.get("status") != "FAILED":
            continue
        cont = job.get("container") or {}
        attempts = job.get("attempts") or [{}]
        last = (attempts[-1] or {}).get("container") or {}
        mem_mib = next((int(r["value"]) for r in cont.get("resourceRequirements", []) if r.get("type") == "MEMORY"),
                       cont.get("memory"))
        name = job.get("jobName", "")
        # Nextflow job names look like nf-NFCORE_SAREK_SAREK_GATK4_HAPLOTYPECALLER_sample1 ; keep it readable
        process = (job.get("tags") or {}).get("nextflow.io/processName") or name.removeprefix("nf-")
        reasons = [x for x in (job.get("statusReason"), last.get("reason") or cont.get("reason")) if x]
        reason = " | ".join(dict.fromkeys(reasons))  # drop duplicates, keep order
        stream = last.get("logStreamName") or cont.get("logStreamName") or ""
        out.append(TaskFailure(
            process=process,
            tag=(job.get("tags") or {}).get("nextflow.io/taskName", ""),
            exit_code=last.get("exitCode", cont.get("exitCode")),
            memory_bytes=int(mem_mib) * 1024**2 if mem_mib else None,
            status_reason=reason,
            error_lines=(log_lines or {}).get(stream, [])[-40:],
            source="aws-batch",
        ))
    return out


def fetch_batch(job_ids: list[str], region: str | None = None) -> list[TaskFailure]:
    batch, logs = _client("batch", region), _client("logs", region)
    doc = batch.describe_jobs(jobs=job_ids)
    lines: dict[str, list[str]] = {}
    for job in doc.get("jobs", []):
        stream = ((job.get("attempts") or [{}])[-1].get("container") or {}).get("logStreamName") \
            or (job.get("container") or {}).get("logStreamName")
        if stream:
            ev = logs.get_log_events(logGroupName="/aws/batch/job", logStreamName=stream, limit=200,
                                     startFromHead=False)
            lines[stream] = [e["message"] for e in ev.get("events", [])]
    return batch_failures_from_json(doc, lines)


# ---------- AWS HealthOmics ----------

def omics_failures_from_json(run: dict, tasks: list[dict] | None = None) -> list[TaskFailure]:
    """Failed tasks from `get-run` (+ `get-run-task` for each failed task)."""
    out = []
    for t in tasks or []:
        if t.get("status") != "FAILED":
            continue
        out.append(TaskFailure(
            process=t.get("name", "").split(" (")[0],
            tag=(t.get("name", "").split(" (")[1].rstrip(")") if " (" in t.get("name", "") else ""),
            memory_bytes=int(t["memory"]) * 1024**3 if t.get("memory") else None,
            status_reason=t.get("statusMessage") or t.get("failureReason") or "",
            source="healthomics",
        ))
    if not out and run.get("status") == "FAILED":
        out.append(TaskFailure(process="(run)", status_reason=run.get("failureReason", "") + " "
                               + run.get("statusMessage", ""), source="healthomics"))
    return out


def fetch_omics(run_id: str, region: str | None = None) -> list[TaskFailure]:
    omics = _client("omics", region)
    run = omics.get_run(id=run_id)
    tasks = []
    for page in omics.get_paginator("list_run_tasks").paginate(id=run_id, status="FAILED"):
        for item in page.get("items", []):
            tasks.append(omics.get_run_task(id=run_id, taskId=item["taskId"]))
    return omics_failures_from_json(run, tasks)


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = ["batch_failures_from_json", "fetch_batch", "omics_failures_from_json", "fetch_omics", "load_json",
           "parse_memory"]
