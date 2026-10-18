"""Accuracy benchmark for the agents: realistic failures with known-correct fixes, scored three ways.

Each case is a recorded AWS account (replies to whatever read-only API calls the agents choose to make) plus a
small pipeline repo. The agents run for real (real LLM, real documentation search) against that recording,
without opening anything. Scoring keeps three things separate so one good number can't hide a bad one:
  value      the key number lands in the known-correct range (e.g. memory within one instance step)
  justified  nothing outside the case's allowed settings changed (no unjustified cpus/time/retry edits)
  decision   the right kind of outcome (agreed config change, or correctly no config change) and retry choice
Run:  pipefix bench [--case ID] [--env-file .env] [--out results.json]
"""
from __future__ import annotations

import io
import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .agent import Agent, first_number
from .aws.session import Clients

ACCOUNT = "111122223333"
BASE_CONFIG = """process {{
    cpus   = {{ 1 * task.attempt }}
    memory = {{ 2.GB * task.attempt }}
    time   = {{ 1.h * task.attempt }}
    errorStrategy = {{ task.exitStatus in ((130..145) + 104) ? 'retry' : 'finish' }}
    maxRetries    = 1
    withLabel: '{label}' {{
        cpus   = {{ {cpus} * task.attempt }}
        memory = {{ {mem}.GB * task.attempt }}
        time   = {{ {hours}.h * task.attempt }}
    }}
}}
"""


@dataclass
class Case:
    id: str
    title: str
    process: str
    label: str
    cpus: int
    mem: int
    script: str
    params: dict
    exit_code: int
    log: list[str]
    status_message: str = ""
    platform: str = "healthomics"          # or "batch"
    hours: int = 4
    extra: dict = field(default_factory=dict)  # case-specific replies: (service, op) -> reply
    expect: dict = field(default_factory=dict)  # kind, value {setting: [lo, hi]}, allow [...], retry "none"|"any"


SPOT = {"statusReason": "Host EC2 (instance i-0abc123) terminated.",
        "attempts": [{"statusReason": "Host EC2 (instance i-0abc123) terminated.", "container": {"exitCode": None}},
                     {"statusReason": "Host EC2 (instance i-0def456) terminated.", "container": {"exitCode": None}}]}

CASES = [
    Case("sort-buffer-deterministic", "Python sort buffer larger than the task's memory (the live SORT_READS case)",
         "SORT_READS", "process_low", 2, 2, "sort_reads.py ${reads} out.fastq --buffer-gb ${params.sort_buffer_gb}",
         {"sort_buffer_gb": 12}, 1, ["[sort_reads] allocating a 12 GB sort buffer", "Traceback (most recent call last):",
                                     "MemoryError"],
         expect={"kind": "config", "value": {"memory": [14, 16]}, "allow": ["memory", "resourceLimits"],
                 "retry": "none"}),
    Case("java-heap-oom-kill", "GATK with a 30 GB Java heap killed by the OOM killer in a 16 GB task",
         "GATK4_HAPLOTYPECALLER", "process_medium", 4, 16,
         "gatk --java-options \"-Xmx${params.gatk_heap}g\" HaplotypeCaller -R ref.fa -I in.bam -O out.vcf.gz",
         {"gatk_heap": 30}, 137, ["Using GATK jar /gatk/gatk.jar", "Running: java -Xmx30g -jar gatk.jar HaplotypeCaller",
                                  "Killed"],
         expect={"kind": "config", "value": {"memory": [32, 48]},
                 "allow": ["memory", "resourceLimits", "errorStrategy", "maxRetries"], "retry": "any"}),
    Case("samtools-threads", "samtools sort with 4 threads x 2 GB each in a 4 GB task",
         "SAMTOOLS_SORT", "process_medium", 4, 4, "samtools sort -@ ${task.cpus} -m 2G -o sorted.bam in.bam",
         {}, 137, ["[bam_sort_core] merging from 12 files and 4 in-memory blocks...", "Killed"],
         expect={"kind": "config", "value": {"memory": [10, 16]},
                 "allow": ["memory", "resourceLimits", "errorStrategy", "maxRetries"], "retry": "any"}),
    Case("container-missing", "Container image tag that does not exist (config cannot fix it)",
         "FASTQC", "process_single", 1, 2, "fastqc --threads ${task.cpus} ${reads}", {}, 1,
         ["CannotPullContainerError: pull image manifest has been retried 5 time(s): failed to resolve ref "
          f"{ACCOUNT}.dkr.ecr.us-east-1.amazonaws.com/tools:9.9: not found"],
         status_message="Container image not found",
         extra={("ecr", "describe_images"): {"__error__": "ImageNotFoundException",
                                             "message": "The image with imageId {imageTag:'9.9'} does not exist"}},
         expect={"kind": "no_config"}),
    Case("oom-at-ceiling", "STAR genome generation asks for more memory than the largest HealthOmics instance has",
         "STAR_GENOMEGENERATE", "process_high", 16, 128,
         "STAR --runMode genomeGenerate --runThreadN ${task.cpus} --limitGenomeGenerateRAM ${params.star_ram_bytes} "
         "--genomeDir star/ --genomeFastaFiles genome.fa", {"star_ram_bytes": 1000000000000}, 1,
         ["EXITING because of FATAL PARAMETER ERROR: limitGenomeGenerateRAM=1000000000000 is too large for the "
          "available memory"],
         expect={"kind": "no_config"}),
    Case("timeout-full-limit", "BWA alignment killed at its 4 h time limit when half done (AWS Batch)",
         "BWA_MEM", "process_high", 8, 16, "bwa mem -t ${task.cpus} ref.fa r1.fq r2.fq | samtools view -b -o out.bam",
         {}, 140, ["[M::mem_process_seqs] Processed 400000 reads in 3.1 CPU sec", "[progress] 50% of reads aligned "
                                                                                "after 4h 0m",
                   "Process exceeded running time limit (4h)"],
         platform="batch", hours=4,
         extra={"status": "Job attempt duration exceeded timeout"},
         expect={"kind": "config", "value": {"time": [8, 10]}, "allow": ["time", "errorStrategy", "maxRetries"],
                 "retry": "any"}),
    Case("spot-twice", "Spot instance reclaimed twice (AWS Batch with a Spot and an On-Demand queue)",
         "MARKDUP", "process_medium", 4, 8, "gatk MarkDuplicates -I in.bam -O md.bam -M metrics.txt", {}, 0,
         ["Host EC2 (instance i-0def456) terminated."], platform="batch",
         extra={"spot": True},
         expect={"kind": "config", "value": {}, "allow": ["queue", "errorStrategy", "maxRetries"], "require": ["queue"],
                 "retry": "any"}),
    Case("disk-full", "Task ran out of local disk (No space left on device)", "FASTP", "process_medium", 4, 8,
         "fastp -i r1.fq -I r2.fq -o o1.fq -O o2.fq --thread ${task.cpus}", {}, 1,
         ["ERROR: write failed: No space left on device"], status_message="Task failed: No space left on device",
         expect={"kind": "no_config"}),
    Case("permission-kms", "The run role cannot decrypt the KMS-encrypted input bucket", "ALIGN", "process_medium", 4,
         8, "bwa mem ref.fa ${reads}", {}, 1,
         ["download failed: s3://inputs/sample1.fq to ./sample1.fq An error occurred (AccessDenied) when calling the "
          "GetObject operation: User: arn:aws:sts::<account-id>:assumed-role/omics-run/x is not authorized to perform: "
          "kms:Decrypt on resource: arn:aws:kms:us-east-1:<account-id>:key/abcd"],
         expect={"kind": "no_config"}),
    Case("oom-need-unknown", "A tool killed for memory with no memory flag in its command", "VARIANT_FILTER",
         "process_low", 2, 4, "variant_filter --in calls.vcf.gz --out filtered.vcf.gz", {}, 137,
         ["[variant_filter] loading 2.1M records", "Killed"],
         expect={"kind": "config", "value": {"memory": [5, 16]},
                 "allow": ["memory", "resourceLimits", "errorStrategy", "maxRetries"], "retry": "any"}),
]


class _Aws:
    """A recorded account: replies to whatever read-only calls the agents make for one case."""

    def __init__(self, case: Case):
        self.case = case

    def factory(self, service):
        case = self.case

        class C:
            def __getattr__(self, op):
                def call(**kw):
                    return _reply(case, service, op, kw)
                return call
        return C()


def _reply(c: Case, service: str, op: str, kw: dict) -> dict:
    if (service, op) in c.extra:
        return c.extra[(service, op)]
    grp = "arn:aws:logs:us-east-1:%s:log-group:/aws/omics/WorkflowLog" % ACCOUNT
    task = {"taskId": "7001", "name": f"PIPELINE:{c.process} (sample1)", "status": "FAILED", "cpus": c.cpus,
            "memory": c.mem, "instanceType": "omics.c.large" if c.mem <= 4 else "omics.m.xlarge",
            "statusMessage": c.status_message or f"Task failed with exit code {c.exit_code}",
            "logStream": f"{grp}:log-stream:run/9001/task/7001",
            "imageDetails": {"image": f"{ACCOUNT}.dkr.ecr.us-east-1.amazonaws.com/tools:"
                                      + ("9.9" if c.id == "container-missing" else "1.0")}}
    if service == "omics":
        if op == "get_run":
            return {"id": "9001", "status": "FAILED", "failureReason": "RUN_TASK_FAILED", "workflowId": "4242",
                    "engine": "NEXTFLOW", "engineVersion": "25.10.0", "storageType": "DYNAMIC",
                    "roleArn": f"arn:aws:iam::{ACCOUNT}:role/omics-run",
                    "logLocation": {"engineLogStream": f"{grp}:log-stream:run/9001/engine",
                                    "runLogStream": f"{grp}:log-stream:run/9001"},
                    "statusMessage": f"Run failed due to task: PIPELINE:{c.process} (sample1), id: 7001, failure."}
        if op == "list_run_tasks":
            return {"items": [{k: task[k] for k in ("taskId", "name", "status", "cpus", "memory")}]}
        if op == "get_run_task":
            return task
        return {"items": []} if op.startswith("list_") else {}
    if service == "batch":
        jq = "arn:aws:batch:us-east-1:%s:job-queue/" % ACCOUNT
        if op in ("describe_jobs", "list_jobs"):
            job = {"jobId": "a1b2c3", "jobName": f"nf-PIPELINE_{c.process}_sample1", "status": "FAILED",
                   "jobQueue": jq + "spot-queue", "statusReason": c.extra.get("status") or "Essential container in "
                                                                                          "task exited",
                   "attempts": [{"container": {"exitCode": c.exit_code, "logStreamName": "nf/default/a1b2c3"},
                                 "statusReason": c.extra.get("status") or "Essential container in task exited"}],
                   "container": {"vcpus": c.cpus, "memory": c.mem * 1024, "exitCode": c.exit_code,
                                 "logStreamName": "nf/default/a1b2c3"}, "timeout": {"attemptDurationSeconds": c.hours * 3600}}
            if c.extra.get("spot"):
                job.update(SPOT)
                job["container"].pop("exitCode", None)
            return {"jobs": [job]} if op == "describe_jobs" else {"jobSummaryList": [
                {"jobId": job["jobId"], "jobName": job["jobName"], "status": "FAILED", "statusReason": job["statusReason"]}]}
        if op == "describe_job_queues":
            return {"jobQueues": [{"jobQueueName": "spot-queue", "state": "ENABLED", "computeEnvironmentOrder": [
                {"order": 1, "computeEnvironment": "ce-spot"}]},
                {"jobQueueName": "ondemand-queue", "state": "ENABLED",
                 "computeEnvironmentOrder": [{"order": 1, "computeEnvironment": "ce-ondemand"}]}]}
        if op == "describe_compute_environments":
            return {"computeEnvironments": [
                {"computeEnvironmentName": "ce-spot", "computeResources": {"type": "SPOT", "maxvCpus": 256,
                                                                           "instanceTypes": ["optimal"]}},
                {"computeEnvironmentName": "ce-ondemand", "computeResources": {"type": "EC2", "maxvCpus": 256,
                                                                               "instanceTypes": ["optimal"]}}]}
        return {}
    if service == "logs":
        if op in ("get_log_events", "filter_log_events"):
            name = str(kw.get("logStreamName") or kw.get("logStreamNamePrefix") or kw.get("logStreamNames") or "")
            if "engine" in name:
                lines = [f"ERROR ~ Error executing process > 'PIPELINE:{c.process} (sample1)'",
                         f"Command exit status:\n  {c.exit_code}", "Command executed:", f"  {c.script}"]
            else:
                lines = c.log
            return {"events": [{"timestamp": 1760000000000 + i, "message": m} for i, m in enumerate(lines)]}
        if op == "describe_log_streams":
            return {"logStreams": [{"logStreamName": "run/9001/engine"}, {"logStreamName": "run/9001/task/7001"},
                                   {"logStreamName": "nf/default/a1b2c3"}]}
        return {}
    if service == "s3" and op == "get_object":
        return {"Body": io.BytesIO(c.script.encode())}
    if service == "ecr" and op == "describe_images":
        return {"imageDetails": [{"imageTags": ["1.0"]}]}
    if service == "cloudtrail" and op == "lookup_events" and c.id == "permission-kms":
        return {"Events": [{"EventName": "Decrypt", "EventSource": "kms.amazonaws.com",
                            "CloudTrailEvent": json.dumps({"errorCode": "AccessDenied", "errorMessage":
                                                           "not authorized to perform: kms:Decrypt"})}]}
    return {}


def make_repo(c: Case, root: Path) -> Path:
    (root / "modules").mkdir(parents=True)
    (root / "modules" / f"{c.process.lower()}.nf").write_text(
        f"process {c.process} {{\n    label '{c.label}'\n    input:\n    path reads\n    script:\n"
        f"    \"\"\"\n    {c.script}\n    \"\"\"\n}}\n")
    (root / "conf").mkdir()
    (root / "conf" / "base.config").write_text(BASE_CONFIG.format(label=c.label, cpus=c.cpus, mem=c.mem, hours=c.hours))
    params = "\n".join(f"    {k} = {v}" for k, v in c.params.items())
    extra = "\nprocess.queue = 'spot-queue'\n" if c.platform == "batch" else ""
    (root / "nextflow.config").write_text(f"params {{\n{params}\n}}\nincludeConfig 'conf/base.config'\n{extra}")
    return root


def score(c: Case, res) -> dict:
    e = c.expect
    chosen = next((x for x in res.candidates if x["id"] == res.decision.get("chosen")), None)
    settings = chosen["settings"] if chosen else {}
    rounds = res.review.get("rounds", [])
    out = {"case": c.id, "chosen": settings, "error": res.error[:200], "calls": len(res.steps),
           "refused": res.refused, "agreed": res.review.get("agreed"), "rounds": len(rounds),
           "tokens": res.usage["prompt"] + res.usage["completion"]}
    if e["kind"] == "no_config":
        out["value"] = None
        out["justified"] = not settings
        out["decision"] = (res.patch is None and not settings and chosen is not None and chosen.get("declared") == 0
                           and bool(res.review.get("agreed")) and "not a pipeline config change" in res.error)
        return out
    vals = []
    for k, (lo, hi) in (e.get("value") or {}).items():
        v = first_number(k, settings.get(k, "")) if k in settings else None
        vals.append(v is not None and lo <= v <= hi)
    out["value"] = all(vals) if vals else None
    out["justified"] = all(k in e.get("allow", []) for k in settings) and all(k in settings for k in e.get("require", []))
    retry_ok = e.get("retry") == "any" or "errorStrategy" not in settings
    out["decision"] = res.patch is not None and retry_ok and bool(res.review.get("agreed"))
    return out


def run_bench(case_ids: list[str] | None = None, samples: int = 0, on_progress=None) -> list[dict]:
    rows = []
    for c in CASES:
        if case_ids and c.id not in case_ids:
            continue
        t0 = time.time()
        with tempfile.TemporaryDirectory() as d:
            repo = make_repo(c, Path(d))
            goal = {"omics_run": "9001"} if c.platform == "healthomics" else {"batch_jobs": ["a1b2c3"]}
            agent = Agent(Clients(_Aws(c).factory, region="us-east-1"), repo, goal)
            try:
                res = agent.run()
                row = score(c, res) | {"seconds": round(time.time() - t0)}
            except Exception as ex:  # noqa: BLE001 - one broken case doesn't stop the bench
                row = {"case": c.id, "value": False, "justified": False, "decision": False, "chosen": {},
                       "error": f"{type(ex).__name__}: {ex}"[:200], "calls": 0, "seconds": round(time.time() - t0)}
        rows.append(row)
        if on_progress:
            on_progress(row)
    return rows


def summary(rows: list[dict]) -> str:
    def rate(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return f"{sum(vals)}/{len(vals)}" if vals else "-"
    lines = [f"{'case':28} {'value':6} {'justified':10} {'decision':9} {'rounds':6} chosen"]
    for r in rows:
        mark = lambda v: "-" if v is None else ("ok" if v else "FAIL")  # noqa: E731
        lines.append(f"{r['case']:28} {mark(r['value']):6} {mark(r['justified']):10} {mark(r['decision']):9} "
                     f"{str(r.get('rounds', '-')):6} {json.dumps(r['chosen'])[:70]}"
                     + (f"  [{r['error'][:70]}]" if r["error"] else ""))
    lines.append(f"\nvalue {rate('value')}   justified {rate('justified')}   decision {rate('decision')}")
    return "\n".join(lines)
