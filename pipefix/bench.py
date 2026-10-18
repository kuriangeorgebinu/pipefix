"""Accuracy benchmark for the agent: realistic failures with known-correct fixes, scored three ways.

Each case is a recorded AWS account (the replies to the read-only calls the agent may make) plus a small pipeline
repo. The agent runs for real (real LLM, real documentation search) against that recording, without opening
anything. Scoring, per the review crew's recommendation, keeps three things separate so one good number can't hide
a bad one:
  value      the key number lands in the known-correct range (e.g. memory within one instance step)
  justified  nothing else changed (no unjustified cpus/time/retry edits)
  decision   the right kind of outcome (config change, or correctly no config change) and the right retry choice
Run:  pipefix bench [--case ID] [--samples 3] [--env-file .env] [--out results.json]
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

BASE_CONFIG = """process {{
    cpus   = {{ 1 * task.attempt }}
    memory = {{ 2.GB * task.attempt }}
    time   = {{ 1.h * task.attempt }}
    errorStrategy = {{ task.exitStatus in ((130..145) + 104) ? 'retry' : 'finish' }}
    maxRetries    = 1
    withLabel: '{label}' {{
        cpus   = {{ {cpus} * task.attempt }}
        memory = {{ {mem}.GB * task.attempt }}
        time   = {{ 4.h * task.attempt }}
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
    expect: dict = field(default_factory=dict)   # kind, memory [lo, hi], forbid [...], retry "none"|"any"


CASES = [
    Case("sort-buffer-deterministic", "Python sort buffer larger than the task's memory (the live SORT_READS case)",
         "SORT_READS", "process_low", 2, 2, "sort_reads.py ${reads} out.fastq --buffer-gb ${params.sort_buffer_gb}",
         {"sort_buffer_gb": 12}, 1, ["[sort_reads] allocating a 12 GB sort buffer", "Traceback (most recent call last):",
                                     "MemoryError"],
         expect={"kind": "config", "memory": [14, 16], "forbid": ["cpus", "time", "errorStrategy", "maxRetries"],
                 "retry": "none"}),
    Case("java-heap-oom-kill", "GATK with a 30 GB Java heap killed by the OOM killer in a 16 GB task",
         "GATK4_HAPLOTYPECALLER", "process_medium", 4, 16,
         "gatk --java-options \"-Xmx${params.gatk_heap}g\" HaplotypeCaller -R ref.fa -I in.bam -O out.vcf.gz",
         {"gatk_heap": 30}, 137, ["Using GATK jar /gatk/gatk.jar", "Running: java -Xmx30g -jar gatk.jar HaplotypeCaller",
                                  "Killed"],
         expect={"kind": "config", "memory": [32, 48], "forbid": ["cpus", "time"], "retry": "any"}),
    Case("samtools-threads", "samtools sort with 4 threads x 2 GB each in a 4 GB task",
         "SAMTOOLS_SORT", "process_medium", 4, 4, "samtools sort -@ ${task.cpus} -m 2G -o sorted.bam in.bam",
         {}, 137, ["[bam_sort_core] merging from 12 files and 4 in-memory blocks...", "Killed"],
         expect={"kind": "config", "memory": [10, 16], "forbid": ["cpus", "time"], "retry": "any"}),
    Case("container-missing", "Container image tag that does not exist (config cannot fix it)",
         "FASTQC", "process_single", 1, 2, "fastqc --threads ${task.cpus} ${reads}", {}, 1,
         ["CannotPullContainerError: pull image manifest has been retried 5 time(s): failed to resolve ref "
          "<account-id>.dkr.ecr.us-east-1.amazonaws.com/tools:9.9: not found"],
         status_message="Container image not found",
         expect={"kind": "no_config"}),
]


class _Aws:
    """A recorded account: replies to the read-only calls for one case."""

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
    task = {"taskId": "7001", "name": f"PIPELINE:{c.process} (sample1)", "status": "FAILED", "cpus": c.cpus,
            "memory": c.mem, "instanceType": "omics.c.large", "statusMessage": c.status_message or
            f"Task failed with exit code {c.exit_code}", "logStream": "run/9001/task/7001",
            "imageDetails": {"image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/tools:1.0"}}
    if (service, op) == ("omics", "get_run"):
        return {"id": "9001", "status": "FAILED", "failureReason": "RUN_TASK_FAILED", "workflowId": "4242",
                "engine": "NEXTFLOW", "engineVersion": "25.10.0", "storageType": "DYNAMIC",
                "statusMessage": f"Run failed due to task: PIPELINE:{c.process} (sample1), id: 7001, failure."}
    if (service, op) == ("omics", "list_run_tasks"):
        return {"items": [{k: task[k] for k in ("taskId", "name", "status", "cpus", "memory")}]}
    if (service, op) == ("omics", "get_run_task"):
        return task
    if (service, op) == ("omics", "list_runs"):
        return {"items": []}
    if (service, op) == ("logs", "get_log_events"):
        if "engine" in str(kw.get("logStreamName")):
            return {"events": [{"message": f"ERROR ~ Error executing process > 'PIPELINE:{c.process} (sample1)'"},
                               {"message": f"Command exit status:\n  {c.exit_code}"}]}
        return {"events": [{"message": m} for m in c.log]}
    if (service, op) == ("logs", "describe_log_streams"):
        return {"logStreams": [{"logStreamName": "run/9001/engine"}, {"logStreamName": "run/9001/task/7001"}]}
    if (service, op) == ("s3", "get_object"):
        return {"Body": io.BytesIO(c.script.encode())}
    if (service, op) == ("ecr", "describe_images"):
        return {"__error__": "ImageNotFoundException"} if c.id == "container-missing" else {"imageDetails": [{}]}
    return {}


def make_repo(c: Case, root: Path) -> Path:
    (root / "modules").mkdir(parents=True)
    (root / "modules" / f"{c.process.lower()}.nf").write_text(
        f"process {c.process} {{\n    label '{c.label}'\n    input:\n    path reads\n    script:\n"
        f"    \"\"\"\n    {c.script}\n    \"\"\"\n}}\n")
    (root / "conf").mkdir()
    (root / "conf" / "base.config").write_text(BASE_CONFIG.format(label=c.label, cpus=c.cpus, mem=c.mem))
    params = "\n".join(f"    {k} = {v}" for k, v in c.params.items())
    (root / "nextflow.config").write_text(f"params {{\n{params}\n}}\nincludeConfig 'conf/base.config'\n")
    return root


def score(c: Case, res) -> dict:
    e = c.expect
    chosen = next((x for x in res.candidates if x["id"] == res.decision.get("chosen")), None)
    settings = chosen["settings"] if chosen else {}
    out = {"case": c.id, "chosen": settings, "error": res.error[:200], "calls": len(res.steps),
           "refused": res.refused, "votes": res.decision.get("votes")}
    if e["kind"] == "no_config":
        out["value"] = None
        out["justified"] = not settings
        out["decision"] = res.patch is None and not settings and "not a pipeline config change" in res.error
        return out
    mem = first_number("memory", settings.get("memory", "")) if "memory" in settings else None
    lo, hi = e["memory"]
    out["value"] = mem is not None and lo <= mem <= hi
    out["justified"] = not any(k in settings for k in e.get("forbid", []))
    retry_ok = e.get("retry") == "any" or "errorStrategy" not in settings
    out["decision"] = res.patch is not None and retry_ok
    return out


def run_bench(case_ids: list[str] | None = None, samples: int = 3, on_progress=None) -> list[dict]:
    rows = []
    for c in CASES:
        if case_ids and c.id not in case_ids:
            continue
        t0 = time.time()
        with tempfile.TemporaryDirectory() as d:
            repo = make_repo(c, Path(d))
            agent = Agent(Clients(_Aws(c).factory, region="us-east-1"), repo, {"omics_run": "9001"},
                          decision_samples=samples)
            res = agent.run()
            row = score(c, res) | {"seconds": round(time.time() - t0)}
        rows.append(row)
        if on_progress:
            on_progress(row)
    return rows


def summary(rows: list[dict]) -> str:
    def rate(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return f"{sum(vals)}/{len(vals)}" if vals else "-"
    lines = [f"{'case':28} {'value':6} {'justified':10} {'decision':9} chosen"]
    for r in rows:
        mark = lambda v: "-" if v is None else ("ok" if v else "FAIL")  # noqa: E731
        lines.append(f"{r['case']:28} {mark(r['value']):6} {mark(r['justified']):10} {mark(r['decision']):9} "
                     f"{json.dumps(r['chosen'])[:80]}" + (f"  [{r['error'][:60]}]" if r["error"] else ""))
    lines.append(f"\nvalue {rate('value')}   justified {rate('justified')}   decision {rate('decision')}")
    return "\n".join(lines)
