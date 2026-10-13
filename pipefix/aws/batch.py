"""Investigate failed Nextflow tasks on AWS Batch.

For each failed job: every attempt (exit code, status reason, Spot reclamation), the requested resources, the
CloudWatch log of each attempt, and the task's S3 work dir (.command.run header for the real process name,
.command.err, .command.trace for peak memory and run time, .exitcode).
Then the platform the job ran on: job queue -> compute environments (Spot or On-Demand, instance types,
max vCPUs) -> launch template (root/scratch EBS size) -> the largest instance memory available, and whether
an On-Demand queue exists to fall back to.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from ..investigation import Investigation
from ..models import TaskFailure
from ..sources.nextflow import split_name
from .session import Clients, paginate
from .workdir import read_work_dir, work_dir_from_command

SPOT_REASON = re.compile(r"Host EC2 \(instance [^)]+\) terminated", re.I)
DEFAULT_LOG_GROUP = "/aws/batch/job"
# AWS Batch's "optimal" instance type picks from these families (C, M, R); newer regions may use newer generations.
OPTIMAL_FAMILIES = ["c4", "m4", "r4", "c5", "m5", "r5", "c6i", "m6i", "r6i"]


def _ms(ts) -> datetime | None:
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc) if isinstance(ts, (int, float)) and ts else None


def _resources(container: dict) -> tuple[int | None, int | None]:
    """-> (memory MiB, vCPUs) from resourceRequirements or the legacy fields."""
    mem = cpu = None
    for r in container.get("resourceRequirements") or []:
        if r.get("type") == "MEMORY":
            mem = int(float(r["value"]))
        elif r.get("type") == "VCPU":
            cpu = int(float(r["value"]))
    return mem or container.get("memory"), cpu or container.get("vcpus")


def find_failed_jobs(clients: Clients, queue: str, since: datetime, name_prefix: str = "", limit: int = 50) -> list[str]:
    """Failed jobs in a queue created after `since`. ListJobs ignores jobStatus when a filter is set, so we
    filter by status ourselves (documented AWS Batch behaviour)."""
    after = str(int(since.timestamp() * 1000))
    n_err = len(clients.errors)
    jobs = paginate(clients, "batch", "list_jobs", "jobSummaryList", jobQueue=queue, maxResults=100,
                    filters=[{"name": "AFTER_CREATED_AT", "values": [after]}], limit=2000)
    if len(clients.errors) > n_err:
        raise RuntimeError(f"Could not list jobs in queue {queue}: {clients.errors[-1]}")
    failed = [j for j in jobs if j.get("status") == "FAILED"
              and (not name_prefix or j.get("jobName", "").lower().startswith(name_prefix.lower()))]
    return [j["jobId"] for j in failed[:limit]]


def describe_jobs(clients: Clients, job_ids: list[str]) -> list[dict]:
    out = []
    for i in range(0, len(job_ids), 100):
        page = clients.call("batch", "describe_jobs", jobs=job_ids[i:i + 100])
        if "__error__" in page:
            raise RuntimeError(f"Could not describe Batch jobs: {page['__error__']}")
        out += page.get("jobs", [])
    missing = set(job_ids) - {j.get("jobId") for j in out}
    if missing and len(missing) == len(job_ids):
        raise RuntimeError(f"No such Batch job(s) in this account/region: {', '.join(sorted(missing))}")
    return out


def _log_tail(clients: Clients, group: str, stream: str, n: int = 100) -> list[str]:
    ev = clients.call("logs", "get_log_events", logGroupName=group, logStreamName=stream, limit=n,
                      startFromHead=False)
    return [e.get("message", "") for e in ev.get("events", [])]


def job_to_failure(job: dict, clients: Clients | None = None) -> TaskFailure:
    """One FAILED Batch job -> TaskFailure. With `clients`, also reads its logs and S3 work dir."""
    cont = job.get("container") or {}
    attempts = job.get("attempts") or []
    last = (attempts[-1].get("container") if attempts else None) or {}
    mem_mib, vcpus = _resources(cont)
    reasons = [job.get("statusReason")] + [a.get("statusReason") for a in attempts] + \
              [(a.get("container") or {}).get("reason") for a in attempts] + [cont.get("reason")]
    reasons = list(dict.fromkeys(r for r in reasons if r))
    reclaims = sum(1 for a in attempts if SPOT_REASON.search(
        " ".join(filter(None, [a.get("statusReason"), (a.get("container") or {}).get("reason")]))))
    exit_code = last.get("exitCode", cont.get("exitCode"))
    proc, tag = split_name(job.get("jobName", ""))
    timeout = (job.get("timeout") or {}).get("attemptDurationSeconds")
    f = TaskFailure(
        process=proc, tag=tag, exit_code=exit_code,
        memory_bytes=int(mem_mib) * 1024**2 if mem_mib else None, cpus=vcpus,
        time_ms=int(timeout) * 1000 if timeout else None,
        status_reason=" | ".join(reasons), source="aws-batch", attempts=max(1, len(attempts)),
        spot_reclaims=reclaims, job_ref=job.get("jobId", ""), image=cont.get("image", ""),
        work_dir=work_dir_from_command(cont.get("command")),
        context={"job_name": job.get("jobName", ""), "job_queue": job.get("jobQueue", ""),
                 "job_definition": job.get("jobDefinition", ""), "job_role": cont.get("jobRoleArn", ""),
                 "execution_role": cont.get("executionRoleArn", ""), "started": job.get("startedAt"),
                 "stopped": job.get("stoppedAt"), "process_name_source": "job name (approximate)"})
    if job.get("startedAt") and job.get("stoppedAt"):
        f.realtime_ms = int(job["stoppedAt"] - job["startedAt"])
    if clients is None:
        return f
    group = ((cont.get("logConfiguration") or {}).get("options") or {}).get("awslogs-group", DEFAULT_LOG_GROUP)
    streams = list(dict.fromkeys(filter(None, [(a.get("container") or {}).get("logStreamName") for a in attempts]
                                        + [cont.get("logStreamName")])))
    for s in streams[-2:]:  # the last two attempts are enough to see what changed
        f.error_lines += [f"[{s.split('/')[-1][:8]}] {ln}" for ln in _log_tail(clients, group, s)][-40:]
    if f.work_dir:
        wd = read_work_dir(clients, f.work_dir)
        if wd.get("name"):
            f.process, f.tag = split_name(wd["name"])
            f.context["process_name_source"] = ".command.run header"
        f.image = wd.get("container") or f.image
        f.error_lines += wd.get("err", [])[-40:]
        if not wd.get("err"):
            f.error_lines += wd.get("log", [])[-40:]
        tr = wd.get("trace") or {}
        f.peak_rss_bytes = tr.get("peak_rss") or f.peak_rss_bytes
        f.realtime_ms = tr.get("realtime") or f.realtime_ms
        if f.exit_code is None and "exitcode" in wd:
            f.exit_code = wd["exitcode"]
    return f


# ---------- the platform: queue -> compute environments -> launch template -> instance capacity ----------

def _family_patterns(instance_types: list[str]) -> list[str]:
    pats = []
    for t in instance_types:
        if t == "optimal":
            pats += [f"{fam}.*" for fam in OPTIMAL_FAMILIES]
        elif "." in t:
            pats.append(t)
        else:
            pats.append(f"{t}.*")
    return list(dict.fromkeys(pats))


def max_instance_memory_mib(clients: Clients, instance_types: list[str]) -> tuple[int | None, str]:
    pats = _family_patterns(instance_types)
    exact = [p for p in pats if "*" not in p]
    wild = [p for p in pats if "*" in p]
    sizes: dict[str, int] = {}
    if exact:
        for it in clients.call("ec2", "describe_instance_types", InstanceTypes=exact).get("InstanceTypes", []):
            sizes[it["InstanceType"]] = it["MemoryInfo"]["SizeInMiB"]
    if wild:
        for it in paginate(clients, "ec2", "describe_instance_types", "InstanceTypes", token_in="NextToken",
                           token_out="NextToken", Filters=[{"Name": "instance-type", "Values": wild}]):
            sizes[it["InstanceType"]] = it["MemoryInfo"]["SizeInMiB"]
    if not sizes:
        return None, ""
    best = max(sizes, key=sizes.get)
    return sizes[best], best


def _launch_template_disks(clients: Clients, lt: dict) -> list[int]:
    if not lt:
        return []
    args = {"Versions": [str(lt.get("version") or "$Default")]}
    if lt.get("launchTemplateId"):
        args["LaunchTemplateId"] = lt["launchTemplateId"]
    elif lt.get("launchTemplateName"):
        args["LaunchTemplateName"] = lt["launchTemplateName"]
    else:
        return []
    vers = clients.call("ec2", "describe_launch_template_versions", **args).get("LaunchTemplateVersions", [])
    sizes = []
    for v in vers[:1]:
        for bd in (v.get("LaunchTemplateData") or {}).get("BlockDeviceMappings", []) or []:
            if (bd.get("Ebs") or {}).get("VolumeSize"):
                sizes.append(int(bd["Ebs"]["VolumeSize"]))
    return sizes


def trace_platform(clients: Clients, inv: Investigation, queue_arns: list[str]) -> None:
    """Describe the queues the failed jobs used and record facts the fix depends on."""
    queues = clients.call("batch", "describe_job_queues", jobQueues=queue_arns).get("jobQueues", [])
    ce_arns = [ce["computeEnvironment"] for q in queues for ce in q.get("computeEnvironmentOrder", [])]
    ces = clients.call("batch", "describe_compute_environments",
                       computeEnvironments=list(dict.fromkeys(ce_arns))).get("computeEnvironments", []) if ce_arns else []
    by_arn = {ce.get("computeEnvironmentArn"): ce for ce in ces} | {ce.get("computeEnvironmentName"): ce for ce in ces}
    max_mem, spot_any = 0, False
    for q in queues:
        for order in q.get("computeEnvironmentOrder", []):
            ce = by_arn.get(order["computeEnvironment"]) or {}
            cr = ce.get("computeResources") or {}
            ctype = cr.get("type", ce.get("type", "?"))
            spot_any |= ctype in ("SPOT", "FARGATE_SPOT")
            types = cr.get("instanceTypes") or []
            mem_mib, biggest = max_instance_memory_mib(clients, types) if types else (None, "")
            disks = _launch_template_disks(clients, cr.get("launchTemplate") or {})
            if mem_mib:
                max_mem = max(max_mem, mem_mib)
            inv.add(f"Queue {q.get('jobQueueName')} -> {ce.get('computeEnvironmentName', order['computeEnvironment'])}",
                    f"{ctype}, allocation {cr.get('allocationStrategy', '-')}, instance types {', '.join(types) or '-'}, "
                    f"max vCPUs {cr.get('maxvCpus', '-')}"
                    + (f", largest instance {biggest} ({mem_mib / 1024:.0f} GiB)" if mem_mib else "")
                    + (f", launch template EBS volumes {', '.join(f'{d} GiB' for d in disks)}" if disks else
                       ", no EBS size set in a launch template (AMI default, often 30 GiB)"),
                    type=ctype, instance_types=types, max_memory_mib=mem_mib, ebs_gib=disks,
                    state=ce.get("state"), status=ce.get("status"))
            if ce.get("status") == "INVALID" or ce.get("state") == "DISABLED":
                inv.add("Compute environment not usable",
                        f"{ce.get('computeEnvironmentName')}: state {ce.get('state')}, status {ce.get('status')} "
                        f"({ce.get('statusReason', '')})", severity="block")
    if max_mem:
        # ECS keeps some memory for the agent and OS; leave ~5% (min 512 MiB) headroom.
        inv.facts["max_task_memory_mib"] = int(max_mem - max(512, max_mem * 0.05))
    inv.facts["spot"] = spot_any
    # Is there an On-Demand queue to fall back to for tasks Spot keeps killing?
    all_queues = paginate(clients, "batch", "describe_job_queues", "jobQueues", maxResults=100, limit=200)
    all_ce_arns = list(dict.fromkeys(o["computeEnvironment"] for q in all_queues
                                     for o in q.get("computeEnvironmentOrder", [])))
    known = {**by_arn}
    missing = [a for a in all_ce_arns if a not in known]
    if missing:
        for ce in clients.call("batch", "describe_compute_environments",
                               computeEnvironments=missing[:100]).get("computeEnvironments", []):
            known[ce.get("computeEnvironmentArn")] = ce
            known[ce.get("computeEnvironmentName")] = ce
    for q in all_queues:
        if q.get("state") != "ENABLED" or q.get("status") not in (None, "VALID"):
            continue
        types = [((known.get(o["computeEnvironment"]) or {}).get("computeResources") or {}).get("type")
                 for o in q.get("computeEnvironmentOrder", [])]
        if types and all(t == "EC2" for t in types):
            inv.facts.setdefault("ondemand_queues", []).append(q["jobQueueName"])


def investigate_batch(clients: Clients, job_ids: list[str] | None = None, queue: str | None = None,
                      since_hours: float = 24, name_prefix: str = "", limit: int = 50) -> Investigation:
    if not job_ids:
        if not queue:
            raise ValueError("give --batch-job ids or a --batch-queue to search")
        since = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        job_ids = find_failed_jobs(clients, queue, since, name_prefix, limit)
    inv = Investigation("aws-batch", ", ".join(job_ids[:5]) + (" ..." if len(job_ids) > 5 else ""))
    jobs = [j for j in describe_jobs(clients, job_ids) if j.get("status") == "FAILED"]
    inv.failures = [job_to_failure(j, clients) for j in jobs]
    starts = [_ms(j.get("startedAt") or j.get("createdAt")) for j in jobs]
    ends = [_ms(j.get("stoppedAt")) for j in jobs]
    inv.start = min((s for s in starts if s), default=None)
    inv.end = max((e for e in ends if e), default=None)
    queues = list(dict.fromkeys(j.get("jobQueue") for j in jobs if j.get("jobQueue")))
    if queues:
        trace_platform(clients, inv, queues)
    for f in inv.failures:
        if f.context.get("process_name_source") != ".command.run header":
            inv.add("Process name is approximate", f"Could not read {f.work_dir or 'the task work dir'}/.command.run "
                    f"for job {f.job_ref}; using the job name `{f.context.get('job_name')}`. pipefix matches it "
                    "against the processes defined in the repo.", severity="warn", process=f.simple_name)
    roles = {r for f in inv.failures for r in (f.context.get("job_role"), f.context.get("execution_role")) if r}
    inv.facts["roles"] = sorted(roles)
    return inv
