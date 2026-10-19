"""Investigate a failed AWS HealthOmics run.

Reads the run (failure reason, storage, engine version, role), every failed task (memory in GiB, CPUs,
instance type, failure reason, image), each failed task's CloudWatch log, and the engine log, which for
Nextflow workflows contains Nextflow's own error blocks ("Error executing process > ..."; Command error).
Log group /aws/omics/WorkflowLog; streams run/{runId}/engine and run/{runId}/task/{taskId}.
"""
from __future__ import annotations

from ..investigation import Investigation
from ..models import TaskFailure
from ..sources.nextflow import parse_log_text, split_name
from .session import Clients, paginate

LOG_GROUP = "/aws/omics/WorkflowLog"
RUN_LEVEL_ONLY = {"RUN_TIMED_OUT", "ASSUME_ROLE_FAILED", "EXPORT_FAILED", "IMPORT_FAILED", "INPUT_URI_NOT_FOUND",
                  "INVALID_URI_INPUT", "UNSUPPORTED_INPUT_SIZE", "WORKFLOW_VER_VALIDATION_FAILED",
                  "INACTIVE_OMICS_STORAGE_RESOURCE", "MODIFIED_INPUT_RESOURCE", "SERVICE_ERROR",
                  "FILE_SYSTEM_OUT_OF_SPACE"}


def _log_tail(clients: Clients, stream: str, n: int) -> list[str]:
    stream = stream.split(":log-stream:")[-1]  # GetRunTask may return an ARN
    ev = clients.call("logs", "get_log_events", logGroupName=LOG_GROUP, logStreamName=stream, limit=n,
                      startFromHead=False)
    return [e.get("message", "") for e in ev.get("events", [])]


def task_to_failure(task: dict, run: dict) -> TaskFailure:
    proc, tag = split_name(task.get("name", ""))
    reason = task.get("failureReason") or ""
    msg = task.get("statusMessage") or ""
    start, stop = task.get("startTime"), task.get("stopTime")
    realtime = None
    if hasattr(start, "timestamp") and hasattr(stop, "timestamp"):
        realtime = int((stop.timestamp() - start.timestamp()) * 1000)
    return TaskFailure(
        process=proc, tag=tag, memory_bytes=int(task["memory"]) * 1024**3 if task.get("memory") else None,
        cpus=task.get("cpus"), realtime_ms=realtime,
        status_reason=" | ".join(x for x in (reason, msg) if x), source="healthomics",
        job_ref=f"{run.get('id', '')}:{task.get('taskId', '')}",
        image=(task.get("imageDetails") or {}).get("image", ""),
        context={"instance_type": task.get("instanceType"), "log_stream": task.get("logStream") or
                 f"run/{run.get('id')}/task/{task.get('taskId')}", "process_name_source": "HealthOmics task name"})


def investigate_omics(clients: Clients, run_id: str) -> Investigation:
    run = clients.call("omics", "get_run", id=run_id)
    if "__error__" in run:
        raise RuntimeError(f"Cannot read HealthOmics run {run_id}: {run['__error__']}")
    inv = Investigation("healthomics", f"HealthOmics run {run_id}")
    inv.start, inv.end = run.get("startTime"), run.get("stopTime")
    inv.nextflow_version = run.get("engineVersion", "") or ""
    items = paginate(clients, "omics", "list_run_tasks", "items", token_in="startingToken", token_out="nextToken",
                     id=run_id, status="FAILED")
    for item in items[:50]:
        task = clients.call("omics", "get_run_task", id=run_id, taskId=item["taskId"])
        if "__error__" in task:
            task = item
        f = task_to_failure(task, run)
        f.error_lines = _log_tail(clients, f.context["log_stream"], 80)[-40:]
        inv.failures.append(f)

    # The engine log is Nextflow's own log: match its error blocks to the failed tasks
    engine = _log_tail(clients, (run.get("logLocation") or {}).get("engineLogStream") or f"run/{run_id}/engine", 1000)
    blocks = parse_log_text("\n".join(engine))
    for b in blocks:
        f = next((x for x in inv.failures if x.process == b.process and x.tag == b.tag), None) \
            or next((x for x in inv.failures if x.simple_name == b.simple_name), None)
        if f:
            f.exit_code = f.exit_code if f.exit_code is not None else b.exit_code
            f.error_lines = list(dict.fromkeys(f.error_lines + b.error_lines))[-60:]
        elif not inv.failures or all(x.process.startswith("(") for x in inv.failures):
            b.source = "healthomics-engine"
            inv.failures.append(b)

    reason = run.get("failureReason") or ""
    if not inv.failures and run.get("status") == "FAILED":
        inv.failures.append(TaskFailure(process="(run)", status_reason=" | ".join(
            x for x in (reason, run.get("statusMessage")) if x), source="healthomics", job_ref=run_id,
            error_lines=engine[-40:]))
    elif reason and len(inv.failures) == 1 and reason not in inv.failures[0].status_reason:
        inv.failures[0].status_reason = " | ".join(x for x in (inv.failures[0].status_reason, reason) if x)
    if reason in RUN_LEVEL_ONLY and not any(f.process == "(run)" for f in inv.failures):
        inv.failures.append(TaskFailure(process="(run)", status_reason=reason, source="healthomics", job_ref=run_id))

    storage = run.get("storageType") or "STATIC"
    cap = run.get("storageCapacity")
    inv.facts.update(storage_type=storage, storage_capacity_gib=cap, workflow_id=run.get("workflowId"),
                     workflow_version=run.get("workflowVersionName"), roles=[run["roleArn"]] if run.get("roleArn") else [],
                     healthomics=True)
    inv.add("Run", f"status {run.get('status')}, failure reason {reason or '-'}, workflow {run.get('workflowId')}"
            + (f" version {run.get('workflowVersionName')}" if run.get("workflowVersionName") else "")
            + f", Nextflow {inv.nextflow_version or '?'}, storage {storage}" + (f" {cap} GiB" if cap else ""))
    if reason == "FILE_SYSTEM_OUT_OF_SPACE":
        new = f"{max(1200, int((cap or 1200) * 2))} GiB" if storage == "STATIC" else "a larger maximum"
        inv.add("Run storage is full", f"The run's {storage} file system filled up. Start the next run with "
                f"`--storage-type DYNAMIC`, or with `--storage-capacity` of {new}. This is a start-run setting, "
                "not pipeline code.", severity="block")
    if reason == "RUN_TIMED_OUT":
        inv.add("Run hit its maximum run time", "Raise maxRunTime on the run group (`aws omics update-run-group`), "
                "or find the hung task. This is not pipeline code.", severity="block")
    return inv
