"""pipefix agent: the AI investigates, researches, decides, implements and opens the MR itself, inside hard rules.

The model drives every step through tool calls, but pipefix holds the steering wheel:

  1. DEBUG      the model calls read-only AWS commands and repo readers from a fixed dictionary (COMMANDS) and
                must finish with a diagnosis whose evidence quotes the real output of its own calls.
  2. RESEARCH   web search (Serper, restricted to AWS / Nextflow / nf-core / Terraform docs) and page reading;
                must finish with at least two candidate solutions, every setting backed by a verbatim quote
                from a page it read (checked against the downloaded page).
  3. DECIDE     no tools; must pick one candidate and say why each other candidate lost.
  4. IMPLEMENT  applies the chosen candidate; pipefix validates every value (allowed directives, strict grammar,
                failed process only, platform limits) and builds the change; the model sees the diff.
  5. MR         opens a draft merge request with the change, or stops at the diff when not allowed.

Hard rules enforced by the harness (not just the prompt): phases run in this order and cannot be skipped; only
the current phase's commands can be called (anything else is refused and counted); every phase must end with its
finish_* call, whose output is checked and sent back with the problems if it fails; per-phase and total call
budgets; AWS stays read-only (dictionary allow-list plus the session's read-only guard); S3 reads only Nextflow
task files (.command.*, .exitcode); everything sent to the model is redacted (account IDs, tokens) and the
account ID is put back only inside AWS call parameters. Every step is recorded and shown in the console and MR.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import keys
from .aws.session import AwsUnavailable, Clients
from .configtrace import RepoConfig, load_repo
from .investigation import Investigation
from .models import Diagnosis, FailureClass, TaskFailure
from .report import redact
from .research import (AIFix, AIResult, Http, SITES, _allowed_host, _clean_query, _json, _norm, _page_text,
                       check, excerpt, verify_quote)

PHASES = ("debug", "research", "decide", "implement", "mr")
PHASE_TITLE = {"debug": "1. Debug", "research": "2. Research solutions", "decide": "3. Pick the best solution",
               "implement": "4. Implement it", "mr": "5. Open the merge request"}
BUDGET = {"debug": 24, "research": 16, "decide": 3, "implement": 4, "mr": 3}
MAX_FINISH_TRIES = 4
OUT_CHARS = 6000
LOG_GROUPS = ("/aws/omics/WorkflowLog", "/aws/batch/job")
S3_TASK_FILE = re.compile(r"(^|/)\.(command\.(run|sh|log|err|out|trace|begin)|exitcode)$")
_ACCOUNT = re.compile(r"(?<!\d)\d{12}(?!\d)")


# ---------- the command dictionary: the agent's skills ----------
# Each AWS command: service + boto3 operation (must be read-only) + the parameters the agent may pass.

def _p(type_: str, desc: str, **extra) -> dict:
    return {"type": type_, "description": desc, **extra}


COMMANDS: dict[str, dict] = {
    # ---- 1. debug: AWS (read-only) ----
    "omics_get_run": {"phase": "debug", "aws": ("omics", "get_run"),
                      "desc": "HealthOmics run: status, failureReason, engine and version, storage, role, logLocation.",
                      "params": {"id": _p("string", "run id")}, "required": ["id"]},
    "omics_list_run_tasks": {"phase": "debug", "aws": ("omics", "list_run_tasks"),
                             "desc": "Tasks of a run (taskId, name, status, cpus, memory GiB). Use status FAILED.",
                             "params": {"id": _p("string", "run id"),
                                        "status": _p("string", "task status filter", enum=["FAILED", "COMPLETED",
                                                                                           "RUNNING", "CANCELLED"])},
                             "required": ["id"]},
    "omics_get_run_task": {"phase": "debug", "aws": ("omics", "get_run_task"),
                           "desc": "One task: cpus, memory, instanceType, failureReason, statusMessage, logStream, image.",
                           "params": {"id": _p("string", "run id"), "taskId": _p("string", "task id")},
                           "required": ["id", "taskId"]},
    "logs_get_log_events": {"phase": "debug", "aws": ("logs", "get_log_events"),
                            "desc": "Read a CloudWatch log stream. HealthOmics: group /aws/omics/WorkflowLog, streams "
                                    "run/{runId}/engine (Nextflow log) and run/{runId}/task/{taskId}. Batch: /aws/batch/job.",
                            "params": {"logGroupName": _p("string", "log group", enum=list(LOG_GROUPS)),
                                       "logStreamName": _p("string", "log stream name"),
                                       "limit": _p("integer", "max events (<= 200)"),
                                       "startFromHead": _p("boolean", "true = from the start, false = the tail")},
                            "required": ["logGroupName", "logStreamName"]},
    "logs_describe_log_streams": {"phase": "debug", "aws": ("logs", "describe_log_streams"),
                                  "desc": "List log streams by prefix, e.g. run/{runId}/ .",
                                  "params": {"logGroupName": _p("string", "log group", enum=list(LOG_GROUPS)),
                                             "logStreamNamePrefix": _p("string", "stream name prefix")},
                                  "required": ["logGroupName"]},
    "cloudtrail_lookup_events": {"phase": "debug", "aws": ("cloudtrail", "lookup_events"),
                                 "desc": "CloudTrail events (e.g. AccessDenied) by event source or name, in a time window.",
                                 "params": {"attribute": _p("string", "lookup attribute", enum=["EventSource", "EventName",
                                                                                                "Username", "ResourceName"]),
                                            "value": _p("string", "attribute value, e.g. s3.amazonaws.com"),
                                            "start": _p("string", "ISO time, e.g. 2026-10-17T19:30:00Z"),
                                            "end": _p("string", "ISO time"),
                                            "max": _p("integer", "max events (<= 50)")},
                                 "required": ["attribute", "value"]},
    "batch_describe_jobs": {"phase": "debug", "aws": ("batch", "describe_jobs"),
                            "desc": "AWS Batch jobs: status reason, attempts, container exit codes, resources, log stream.",
                            "params": {"jobs": _p("array", "job ids", items={"type": "string"})}, "required": ["jobs"]},
    "batch_list_jobs": {"phase": "debug", "aws": ("batch", "list_jobs"),
                        "desc": "Jobs in a Batch queue by status.",
                        "params": {"jobQueue": _p("string", "queue name"),
                                   "jobStatus": _p("string", "status", enum=["FAILED", "SUCCEEDED", "RUNNING"])},
                        "required": ["jobQueue"]},
    "batch_describe_job_queues": {"phase": "debug", "aws": ("batch", "describe_job_queues"),
                                  "desc": "Batch queues and their compute environments.",
                                  "params": {"jobQueues": _p("array", "queue names", items={"type": "string"})},
                                  "required": []},
    "batch_describe_compute_environments": {"phase": "debug", "aws": ("batch", "describe_compute_environments"),
                                            "desc": "Compute environments: instance types, Spot/On-Demand, max vCPUs.",
                                            "params": {"computeEnvironments": _p("array", "names or ARNs",
                                                                                 items={"type": "string"})},
                                            "required": []},
    "ec2_describe_instance_types": {"phase": "debug", "aws": ("ec2", "describe_instance_types"),
                                    "desc": "vCPUs and memory of EC2 instance types (for Batch sizing limits).",
                                    "params": {"InstanceTypes": _p("array", "instance types", items={"type": "string"})},
                                    "required": ["InstanceTypes"]},
    "ecr_describe_images": {"phase": "debug", "aws": ("ecr", "describe_images"),
                            "desc": "Check a container image tag or digest exists in ECR.",
                            "params": {"repositoryName": _p("string", "ECR repository name"),
                                       "imageTag": _p("string", "tag"), "imageDigest": _p("string", "sha256:... digest")},
                            "required": ["repositoryName"]},
    "s3_get_task_file": {"phase": "debug", "aws": ("s3", "get_object"),
                         "desc": "Read a Nextflow task file from its work dir: .command.run/.sh/.log/.err/.out/.trace or "
                                 ".exitcode ONLY (never data).",
                         "params": {"Bucket": _p("string", "bucket"), "Key": _p("string", "key ending in a task file")},
                         "required": ["Bucket", "Key"]},
    # ---- 1. debug: the pipeline repo ----
    "repo_list_files": {"phase": "debug", "desc": "List config and workflow files in the pipeline repo.",
                        "params": {"glob": _p("string", "glob, e.g. **/*.config or modules/**/*.nf")}, "required": []},
    "repo_read_file": {"phase": "debug", "desc": "Read a file in the pipeline repo (up to 300 lines).",
                       "params": {"path": _p("string", "path relative to the repo"),
                                  "start": _p("integer", "first line (1-based)")}, "required": ["path"]},
    "repo_trace_setting": {"phase": "debug",
                           "desc": "Every config line that sets memory/time/cpus/errorStrategy/maxRetries/queue for a "
                                   "process, with file:line, selector and which one wins (Nextflow precedence).",
                           "params": {"process": _p("string", "process name, e.g. SORT_READS")}, "required": ["process"]},
    "finish_debug": {"phase": "debug", "finish": True,
                     "desc": "End step 1 with the diagnosis. Evidence must quote the output of calls you made.",
                     "params": {"process": _p("string", "failed process (simple name)"),
                                "tag": _p("string", "task tag / sample, if any"),
                                "failure_class": _p("string", "class", enum=[c.value for c in FailureClass]),
                                "exit_code": _p("integer", "exit code of the failed task, if known"),
                                "memory_requested_gb": _p("number", "memory the task had (GiB), if known"),
                                "cpus": _p("integer", "vCPUs the task had, if known"),
                                "root_cause": _p("string", "the root cause in 2-4 sentences, with the numbers"),
                                "evidence": _p("array", "facts from your calls", items={
                                    "type": "object", "properties": {
                                        "call": {"type": "integer", "description": "the #number of the call"},
                                        "excerpt": {"type": "string", "description": "verbatim text from its output"}},
                                    "required": ["call", "excerpt"]})},
                     "required": ["process", "failure_class", "root_cause", "evidence"]},
    # ---- 2. research ----
    "web_search": {"phase": "research", "desc": "Google search (Serper), restricted to one documentation family.",
                   "params": {"query": _p("string", "short keyword query, no paths/IDs/sample names"),
                              "site": _p("string", "where", enum=list(SITES))}, "required": ["query", "site"]},
    "web_read_page": {"phase": "research", "desc": "Read a search result. Returns the relevant passages of the page.",
                      "params": {"source": _p("integer", "source number from web_search")}, "required": ["source"]},
    "finish_research": {"phase": "research", "finish": True,
                        "desc": "End step 2 with 2-4 candidate solutions. Every setting needs a verbatim quote from a "
                                "page you READ with web_read_page.",
                        "params": {"candidates": _p("array", "candidate solutions", items={
                            "type": "object", "properties": {
                                "id": {"type": "string", "description": "short id, e.g. A"},
                                "title": {"type": "string"},
                                "process": {"type": "string"},
                                "settings": {"type": "object", "description": "directive -> value (may be {} for a "
                                                                              "non-config solution)"},
                                "rationale": {"type": "string"},
                                "pros": {"type": "string"}, "cons": {"type": "string"},
                                "evidence": {"type": "array", "items": {"type": "object", "properties": {
                                    "setting": {"type": "string"}, "source": {"type": "integer"},
                                    "quote": {"type": "string"}}, "required": ["source", "quote"]}}},
                            "required": ["id", "title", "settings", "rationale", "evidence"]})},
                        "required": ["candidates"]},
    # ---- 3. decide ----
    "finish_decision": {"phase": "decide", "finish": True,
                        "desc": "Pick the best candidate. Explain why it wins and why each other candidate lost.",
                        "params": {"chosen": _p("string", "candidate id"),
                                   "reasons": _p("string", "why this one is best for THIS failure"),
                                   "why_not": _p("object", "candidate id -> why it lost (every other candidate)"),
                                   "risks": _p("string", "what could still go wrong")},
                        "required": ["chosen", "reasons", "why_not"]},
    # ---- 4. implement ----
    "implement_apply": {"phase": "implement",
                        "desc": "Apply the chosen solution to the pipeline config. Returns the diff, or the problems.",
                        "params": {}, "required": []},
    "finish_implement": {"phase": "implement", "finish": True, "desc": "End step 4 once the diff is right.",
                         "params": {"summary": _p("string", "what the change does, in 1-2 sentences")},
                         "required": ["summary"]},
    # ---- 5. merge request ----
    "mr_open": {"phase": "mr", "desc": "Open the merge request (draft) with the change and the full evidence.",
                "params": {}, "required": []},
    "finish_mr": {"phase": "mr", "finish": True, "desc": "End step 5.",
                  "params": {"summary": _p("string", "one line for the person reading the result")},
                  "required": ["summary"]},
}

RULES = """You are pipefix, an autonomous agent that fixes failed Nextflow genomics pipeline runs on AWS
(AWS HealthOmics or AWS Batch). You work in FIVE STEPS, in this order, and the harness enforces them:
  1. DEBUG: find out exactly why the run failed, using the read-only AWS commands and the repo readers.
  2. RESEARCH SOLUTIONS: search the AWS, Nextflow/nf-core and Terraform docs and READ the pages.
  3. PICK THE BEST SOLUTION: choose one candidate and justify it against the others.
  4. IMPLEMENT THE BEST SOLUTION: apply it as a Nextflow config change.
  5. MERGE REQUEST: open it.
HARD RULES (breaking one gets your call refused):
- Only the current step's commands work. Each step ends ONLY by calling its finish_* command.
- Base every claim on evidence: in step 1 quote the output of your own calls (by call #); in step 2 quote, word
  for word, pages you read with web_read_page. Never invent numbers, IDs, file names or quotes.
- AWS access is read-only. Never ask for data files; S3 is only for Nextflow task files (.command.*, .exitcode).
- Fix only the process that failed. pipefix writes every candidate as a withName block for that process in
  conf/pipefix.config, so candidates differ in their VALUES, not in where they are written. Put in "settings" only
  the directives you CHANGE. Config values must use these forms: memory "{ N.GB * task.attempt }",
  time "{ N.h * task.attempt }", cpus "N", maxRetries "N", errorStrategy
  "{ task.exitStatus in <codes> ? 'retry' : 'finish' }" (codes: integers, a..b ranges, + and parentheses).
- Size resources from what the task actually needs (read the process script and params for tool flags such as
  --buffer-gb, -Xmx, sort -S, --mem) and leave headroom. Only widen a retry rule with an exit code when the
  failure is memory-related and the evidence shows it.
- Be efficient: a few targeted calls per step. Account IDs appear as <account-id>; pass them as is."""

PHASE_BRIEF = {
    "debug": "STEP 1, DEBUG. Start from the run you were given. Read the run, its failed tasks, the task and engine "
             "logs, then the pipeline repo (process script, params, the config lines that set its resources). Finish "
             "with finish_debug: process, failure_class, exit_code, root_cause with numbers, and evidence quoting "
             "your calls by #.",
    "research": "STEP 2, RESEARCH SOLUTIONS. Search the docs (aws, nextflow, terraform) for how to fix this failure, "
                "READ the useful results with web_read_page, then finish_research with 2-4 genuinely different "
                "candidates (e.g. raise memory vs lower the tool's buffer; with or without retry changes). Each "
                "setting needs a verbatim quote from a page you read. Cite pages by their SOURCE number (the "
                "'source' in web_search results, 'cite_as_source' in web_read_page), copy quotes exactly from the "
                "'passages', and only cite pages you have read. Label each quote with the setting it supports "
                "(memory, cpus, time, errorStrategy, maxRetries); a quote only supports a setting it talks about.",
    "decide": "STEP 3, PICK THE BEST SOLUTION. No tools. Compare the candidates on correctness for THIS failure, cost, "
              "and risk, then call finish_decision.",
    "implement": "STEP 4, IMPLEMENT. Call implement_apply, check the diff it returns, then finish_implement.",
    "mr": "STEP 5, MERGE REQUEST. Call mr_open, then finish_mr with one line for the person reading the result.",
}


# ---------- run state ----------

@dataclass
class Step:
    phase: str
    n: int                      # call number (#n) the model refers to
    tool: str
    args: dict
    ok: bool
    summary: str                # one line for the UI / MR
    output: str = ""            # what the model saw (redacted, truncated)


@dataclass
class AgentResult:
    model: str = ""
    steps: list[Step] = field(default_factory=list)
    phase_reached: str = ""
    diagnosis: dict = field(default_factory=dict)
    candidates: list[dict] = field(default_factory=list)
    decision: dict = field(default_factory=dict)
    implement_summary: str = ""
    mr_url: str = ""
    mr_summary: str = ""
    error: str = ""
    inv: Investigation | None = None
    diagnoses: list[Diagnosis] = field(default_factory=list)
    ai: AIResult | None = None
    patch: object = None
    refused: int = 0

    @property
    def done(self) -> bool:
        return not self.error and self.phase_reached == "mr"


def _loose(text: str) -> str:
    """For checking debug evidence against JSON tool output: ignore JSON punctuation/escapes, case and spacing."""
    text = re.sub(r'\\[nrt"/\\]', " ", text or "")
    return " ".join(re.sub(r'[{}\[\]",:]', " ", _norm(text)).split())


# what a quote must talk about to support a value for each setting
SUBJECT = {"memory": r"memory|\bGB\b|GiB|RAM|out.of.memory|\bOOM\b", "cpus": r"\bcpus?\b|vCPU",
           "time": r"\btime\b|hours?|wall|timeout", "maxRetries": r"maxRetries|retr(y|ies)|attempt",
           "errorStrategy": r"errorStrategy|retr(y|ies)|exit (code|status)|exitStatus", "queue": r"\bqueues?\b",
           "disk": r"\bdisk\b|storage"}


class _Refused(Exception):
    pass


class _Stop(Exception):
    """A legitimate end before the MR (e.g. the best solution is not a config change)."""


def _tool_schema(name: str) -> dict:
    c = COMMANDS[name]
    return {"type": "function", "function": {
        "name": name, "description": c["desc"],
        "parameters": {"type": "object", "properties": c["params"], "required": c.get("required", [])}}}


def _short(obj, limit: int = OUT_CHARS) -> str:
    text = obj if isinstance(obj, str) else json.dumps(obj, default=str, indent=None)
    text = redact(text)
    return text if len(text) <= limit else text[:limit] + f" ...[truncated, {len(text)} chars]"


class Agent:
    def __init__(self, clients: Clients | None, repo: str | Path | None, goal: dict, http: Http | None = None,
                 allow_mr: bool = False, on_step=None, mr_opener=None, max_seconds: int = 900,
                 transcript: str | Path | None = None):
        self.clients, self.goal, self.http = clients, goal, http or Http()
        self.repo = Path(repo).resolve() if repo else None
        self.cfg: RepoConfig | None = load_repo(self.repo) if self.repo and (self.repo / "nextflow.config").is_file() \
            else None
        self.allow_mr, self.on_step, self.mr_opener = allow_mr, on_step or (lambda s: None), mr_opener
        self.deadline = time.time() + max_seconds
        self.transcript = Path(transcript) if transcript else None
        key, self.base, self.model = keys.llm_endpoint()
        self.llm_key = key
        self.serper_key, _ = keys.get("serper")
        self.res = AgentResult(model=self.model)
        self.outputs: dict[int, str] = {}          # call # -> full redacted output (for evidence checks)
        self.sources: list[dict] = []              # research sources {n, title, url, site, snippet, text, read}
        self.read_by_call: dict[int, int] = {}     # call # of a web_read_page -> the source it read
        self.account: str | None = None
        self.n = 0
        platform = "healthomics" if goal.get("omics_run") else "aws-batch"
        ref = f"HealthOmics run {goal['omics_run']}" if goal.get("omics_run") else \
            f"Batch {'queue ' + goal['batch_queue'] if goal.get('batch_queue') else 'jobs'}"
        self.inv = Investigation(platform, ref, facts={"healthomics": platform == "healthomics"})
        self.res.inv = self.inv

    # ----- the LLM -----
    def _chat(self, messages: list[dict], tools: list[dict]) -> dict:
        payload = {"model": self.model, "temperature": 0, "messages": messages}
        if tools:
            payload["tools"], payload["tool_choice"] = tools, "required"
        out = self.http.post_json(f"{self.base}/chat/completions", payload, {"Authorization": f"Bearer {self.llm_key}"},
                                  timeout=180)
        return out["choices"][0]["message"]

    # ----- record -----
    def _record(self, phase: str, tool: str, args: dict, ok: bool, summary: str, output: str) -> Step:
        st = Step(phase, self.n, tool, json.loads(redact(json.dumps(args, default=str))), ok, redact(summary)[:300],
                  output)
        self.res.steps.append(st)
        self.on_step(st)
        return st

    # ----- AWS -----
    def _unredact(self, value):
        if isinstance(value, str) and self.account:
            return value.replace("<account-id>", self.account)
        if isinstance(value, list):
            return [self._unredact(v) for v in value]
        return value

    def _aws(self, name: str, args: dict):
        if not self.clients:
            raise _Refused("AWS is not available in this run.")
        service, op = COMMANDS[name]["aws"]
        allowed = COMMANDS[name]["params"]
        bad = [k for k in args if k not in allowed]
        if bad:
            raise _Refused(f"unknown parameter(s) {bad} for {name}; allowed: {list(allowed)}")
        a = {k: self._unredact(v) for k, v in args.items()}
        if name == "logs_get_log_events":
            a["limit"] = max(1, min(int(a.get("limit") or 100), 200))
            a.setdefault("startFromHead", False)
            a["logStreamName"] = str(a["logStreamName"]).split(":log-stream:")[-1]
        if name.startswith("logs_") and a.get("logGroupName") not in LOG_GROUPS:
            raise _Refused(f"log group must be one of {LOG_GROUPS}")
        if name == "omics_list_run_tasks":
            a["maxResults"] = 100
        if name == "cloudtrail_lookup_events":
            from datetime import datetime, timedelta, timezone
            def ts(s, default):
                try:
                    return datetime.fromisoformat(str(s).replace("Z", "+00:00")) if s else default
                except ValueError:
                    raise _Refused(f"bad time {s!r}; use ISO 8601 like 2026-10-17T19:30:00Z")
            end = ts(a.get("end"), datetime.now(timezone.utc))
            a = {"LookupAttributes": [{"AttributeKey": a["attribute"], "AttributeValue": a["value"]}],
                 "StartTime": ts(a.get("start"), end - timedelta(hours=6)), "EndTime": end,
                 "MaxResults": max(1, min(int(a.get("max") or 20), 50))}
        if name == "ecr_describe_images":
            ids = {k: a.pop(k) for k in ("imageTag", "imageDigest") if a.get(k)}
            a["imageIds"] = [ids] if ids else None
            if a["imageIds"] is None:
                a.pop("imageIds")
        if name == "s3_get_task_file":
            if not S3_TASK_FILE.search(str(a["Key"])):
                raise _Refused("S3 is only for Nextflow task files (.command.run/.sh/.log/.err/.out/.trace, .exitcode)")
            a["Range"] = "bytes=-65536"
        out = self.clients.call(service, op, **a)
        self.inv.calls[:] = self.clients.calls
        if name == "s3_get_task_file" and "Body" in out:
            out = {"text": out["Body"].read().decode("utf-8", errors="replace")}
        if isinstance(out, dict):
            out.pop("ResponseMetadata", None)
        if name == "cloudtrail_lookup_events":
            out = {"events": [{"time": e.get("EventTime"), "name": e.get("EventName"), "source": e.get("EventSource"),
                               "user": e.get("Username"),
                               "error": (json.loads(e.get("CloudTrailEvent") or "{}").get("errorCode")),
                               "message": (json.loads(e.get("CloudTrailEvent") or "{}").get("errorMessage"))}
                              for e in out.get("Events", [])]}
        if name == "logs_get_log_events":
            out = {"lines": [e.get("message", "") for e in out.get("events", [])]}
        raw = json.dumps(out, default=str)
        if not self.account:
            m = _ACCOUNT.search(raw)
            self.account = m.group(0) if m else None
        return out

    # ----- repo -----
    def _repo_path(self, rel: str) -> Path:
        if not self.repo:
            raise _Refused("no pipeline repo was given")
        p = (self.repo / rel).resolve()
        if self.repo not in p.parents and p != self.repo:
            raise _Refused("path is outside the pipeline repo")
        if any(part in (".git", ".env", ".venv", "node_modules") for part in p.parts) or p.name.startswith(".env"):
            raise _Refused("that path is not readable")
        return p

    def _repo(self, name: str, args: dict):
        if name == "repo_list_files":
            if not self.repo:
                raise _Refused("no pipeline repo was given")
            pats = [args.get("glob")] if args.get("glob") else ["**/*.config", "**/*.nf", "nextflow.config"]
            files = sorted({p.relative_to(self.repo).as_posix() for g in pats for p in self.repo.glob(g)
                            if p.is_file() and ".git" not in p.parts})
            return {"files": files[:200]}
        if name == "repo_read_file":
            p = self._repo_path(str(args["path"]))
            if not p.is_file():
                raise _Refused(f"{args['path']} does not exist")
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            start = max(1, int(args.get("start") or 1))
            chunk = lines[start - 1:start + 299]
            return {"path": args["path"], "lines": [f"{start + i}: {ln}" for i, ln in enumerate(chunk)],
                    "total_lines": len(lines)}
        if name == "repo_trace_setting":
            if not self.cfg:
                raise _Refused("the repo has no nextflow.config")
            tr = self.cfg.trace(str(args["process"]))
            return {"process": args["process"], "defined_in": tr.defined_in, "labels": tr.labels,
                    "settings": [{"directive": s.directive, "value": s.value.strip(), "where": s.where,
                                  "selector": s.selector, "wins": tr.effective(s.directive) is s}
                                 for s in tr.settings], "params": dict(list(self.cfg.params.items())[:40])}
        raise _Refused(name)

    # ----- research -----
    def _web(self, name: str, args: dict):
        if name == "web_search":
            if not self.serper_key:
                raise _Refused("no Serper key")
            q = _clean_query(str(args.get("query", "")))
            site = args.get("site") if args.get("site") in SITES else "aws"
            if not q:
                raise _Refused("empty query after removing paths and IDs")
            hits = self.http.post_json(keys.SERPER_URL, {"q": f"{q} {SITES[site]}", "num": 6},
                                       {"X-API-KEY": self.serper_key}, timeout=20)
            found = []
            for h in hits.get("organic", [])[:6]:
                url = str(h.get("link", ""))
                old = next((s for s in self.sources if s["url"] == url), None)
                if not url.startswith("https://"):
                    continue
                if not old and len(self.sources) < 20:
                    old = {"n": len(self.sources) + 1, "title": str(h.get("title", ""))[:160], "url": url,
                           "snippet": str(h.get("snippet", ""))[:300], "site": site, "read": False}
                    self.sources.append(old)
                if old:
                    found.append({"source": old["n"], "title": old["title"], "url": old["url"], "snippet": old["snippet"]})
            return {"query": q, "results": found}
        if name == "web_read_page":
            n = args.get("source")
            src = next((s for s in self.sources if s["n"] == n), None)
            if not src:
                raise _Refused(f"no source #{n}; search first")
            if not _allowed_host(src["url"]):
                raise _Refused("that host is not on the documentation allow-list")
            if not src.get("text"):
                try:
                    src["text"] = _page_text(self.http.get_text(src["url"]))
                except Exception as e:  # network errors are reported to the agent, not fatal
                    raise _Refused(f"could not read the page ({type(e).__name__})")
                src["read"] = bool(src["text"])
            self.read_by_call[self.n] = n
            return {"cite_as_source": n, "title": src["title"], "url": src["url"], "passages": excerpt(src["text"])}
        raise _Refused(name)

    # ----- finish checks -----
    def _ctx(self) -> dict:
        d = self.res.diagnosis
        return {"failed_tasks": [{"process": d.get("process"), "exit_code": d.get("exit_code")}],
                "platform": {"healthomics": self.inv.platform == "healthomics",
                             "largest_task_memory_gb": self.inv.facts.get("largest_task_memory_gb"),
                             "ondemand_queues": self.inv.facts.get("ondemand_queues", [])}}

    def _finish_debug(self, a: dict) -> str:
        problems = []
        aws_calls = [s for s in self.res.steps if s.phase == "debug" and s.ok and COMMANDS[s.tool].get("aws")]
        if not aws_calls:
            problems.append("you made no AWS calls; debug from the real run, not from assumptions")
        try:
            FailureClass(a.get("failure_class"))
        except ValueError:
            problems.append("failure_class is not one of the allowed values")
        good = []
        for ev in a.get("evidence") or []:
            n, ex = ev.get("call"), str(ev.get("excerpt", ""))
            out = self.outputs.get(n if isinstance(n, int) else -1)
            if out is None:
                problems.append(f"evidence cites call #{n}, which you did not make")
            elif len(_loose(ex)) < 6 or _loose(ex) not in _loose(out):
                problems.append(f"evidence excerpt for #{n} is not in that call's output: {ex[:80]!r}")
            else:
                good.append({"call": n, "tool": next(s.tool for s in self.res.steps if s.n == n), "excerpt": ex})
        if len(good) < 2 or not any(COMMANDS[g["tool"]].get("aws") for g in good):
            problems.append("give at least 2 verified evidence excerpts, at least one from an AWS call")
        if problems:
            raise _Refused("; ".join(problems))
        self.res.diagnosis = {k: a.get(k) for k in ("process", "tag", "failure_class", "exit_code",
                                                    "memory_requested_gb", "cpus", "root_cause")}
        self.res.diagnosis["process"] = str(a["process"]).split(":")[-1].strip()
        self.res.diagnosis["evidence"] = good
        return "Diagnosis accepted."

    def _finish_research(self, a: dict) -> str:
        cands, problems = [], []
        ctx = self._ctx()
        by_n = {s["n"]: s for s in self.sources}
        for c in a.get("candidates") or []:
            cid = str(c.get("id", "")).strip() or f"C{len(cands) + 1}"
            proc = str(c.get("process") or ctx["failed_tasks"][0]["process"]).split(":")[-1].strip()
            cp, notes = [], []
            # settings: strip selector prefixes, drop values identical to what is in effect now
            tr = self.cfg.trace(proc) if self.cfg and proc in self.cfg.processes else None
            settings = {}
            for k, v in (c.get("settings") or {}).items():
                m = re.search(r"(memory|time|cpus|errorStrategy|maxRetries|queue|disk)\s*$", str(k))
                key, val = (m.group(1) if m else str(k).strip()), str(v).strip()
                cur = tr.effective(key) if tr else None
                if cur and _loose(cur.value) == _loose(val):
                    notes.append(f"{key} dropped: same as now ({cur.where})")
                    continue
                settings[key] = val
            # quotes: keep the verbatim ones, list the others (a bad quote alone doesn't sink the candidate)
            evid = []
            for ev in c.get("evidence") or []:
                n = ev.get("source")
                n = int(n) if isinstance(n, (int, float)) or (isinstance(n, str) and str(n).isdigit()) else n
                if not by_n.get(n, {}).get("read") and n in self.read_by_call:
                    n = self.read_by_call[n]     # the model cited the call that read the page: same page
                src = by_n.get(n) if by_n.get(n, {}).get("read") else None
                quote = " ".join(str(ev.get("quote", "")).split())
                why = verify_quote(quote, src)
                if why:
                    notes.append(f"quote from source {n} not used: {why}")
                    continue
                label = str(ev.get("setting", "")).strip()
                lm = re.search(r"(memory|time|cpus|errorStrategy|maxRetries|queue|disk)\s*$", label)
                evid.append({"setting": lm.group(1) if lm else "", "n": n, "quote": redact(quote)})
            # every changed setting needs a verified quote that is about that setting
            for k, v in settings.items():
                why = check(proc, k, v, ctx)
                if why:
                    cp.append(f"{k} = {v}: {why}")
                    continue
                topic = SUBJECT.get(k)
                support = [e for e in evid if e["setting"] in ("", k) and (not topic or re.search(topic, e["quote"], re.I))]
                if not support:
                    cp.append(f"{k} has no verified quote about {k} from a page you read")
                for e in support:
                    e.setdefault("supports", []).append(k)
            for e in evid:   # a quote may support several settings; one that supports none is only context
                e["setting"] = ", ".join(e.pop("supports", [])) or "context"
            if not evid:
                cp.append("no verified quote at all")
            if cp:
                problems.append(f"candidate {cid}: " + "; ".join(cp))
            cands.append({"id": cid, "title": redact(str(c.get("title", "")))[:160], "process": proc,
                          "settings": settings, "rationale": redact(str(c.get("rationale", "")))[:1500],
                          "pros": redact(str(c.get("pros", "")))[:400], "cons": redact(str(c.get("cons", "")))[:400],
                          "evidence": evid, "valid": not cp, "problems": cp, "notes": notes})
        if len(cands) < 2:
            problems.append("give at least 2 genuinely different candidates")
        if not any(c["valid"] for c in cands):
            problems.append("no candidate passed every check")
        fixable = FailureClass(self.res.diagnosis["failure_class"]).auto_fixable
        if fixable and not any(c["valid"] and c["settings"] for c in cands):
            problems.append(f"a {self.res.diagnosis['failure_class']} failure is fixable by config, so at least one "
                            "candidate must change settings and pass every check; fix the problems above "
                            "(remove settings you cannot support with a quote about them)")
        if len(cands) < 2 or not any(c["valid"] for c in cands) or (fixable and not any(c["valid"] and c["settings"]
                                                                                        for c in cands)):
            extra = [f"candidate {c['id']} notes: " + "; ".join(c["notes"]) for c in cands if c["notes"]]
            raise _Refused(" | ".join(problems + extra))
        self.res.candidates = cands
        bad = [c for c in cands if not c["valid"]]
        return "Research accepted." + ("".join(f" Candidate {c['id']} failed checks and cannot be chosen: "
                                               f"{'; '.join(c['problems'])}." for c in bad))

    def _finish_decision(self, a: dict) -> str:
        ids = {c["id"]: c for c in self.res.candidates}
        chosen = str(a.get("chosen", "")).strip()
        if chosen not in ids:
            raise _Refused(f"chosen must be one of {list(ids)}")
        if not ids[chosen]["valid"]:
            raise _Refused(f"candidate {chosen} failed the checks: {ids[chosen]['problems']}")
        why_not = {str(k): str(v) for k, v in (a.get("why_not") or {}).items()}
        missing = [i for i in ids if i != chosen and not why_not.get(i, "").strip()]
        if missing:
            raise _Refused(f"say why each other candidate lost; missing: {missing}")
        if len(str(a.get("reasons", "")).strip()) < 40:
            raise _Refused("reasons are too thin; compare against the evidence")
        self.res.decision = {"chosen": chosen, "reasons": redact(str(a["reasons"]))[:1500],
                             "why_not": {k: redact(v)[:600] for k, v in why_not.items()},
                             "risks": redact(str(a.get("risks", "")))[:800]}
        return f"Decision accepted: {chosen}."

    def _build_ai(self) -> AIResult:
        c = next(c for c in self.res.candidates if c["id"] == self.res.decision["chosen"])
        ai = AIResult(model=self.model, queries=[s.args.get("query", "") for s in self.res.steps if s.tool == "web_search"],
                      sources=[{k: s.get(k) for k in ("n", "title", "url", "snippet", "site", "read")} for s in self.sources])
        why = f"{c['title']}: {c['rationale']}"
        if c["settings"]:
            ai.fixes.append(AIFix(c["process"], dict(c["settings"]), why, sorted({e["n"] for e in c["evidence"]}),
                                  list(c["evidence"])))
        ai.summary = self.res.decision["reasons"]
        for o in self.res.candidates:
            if o["id"] != c["id"]:
                ai.infrastructure.append({"title": f"Alternative {o['id']} (not chosen): {o['title']}",
                                          "detail": self.res.decision["why_not"].get(o["id"], ""),
                                          "sources": sorted({e["n"] for e in o["evidence"]}), "evidence": o["evidence"]})
        return ai

    def _diagnoses(self) -> list[Diagnosis]:
        d = self.res.diagnosis
        f = TaskFailure(d["process"], tag=str(d.get("tag") or ""), exit_code=d.get("exit_code"),
                        memory_bytes=int(float(d["memory_requested_gb"]) * 1024**3) if d.get("memory_requested_gb") else None,
                        cpus=d.get("cpus"), source=self.inv.platform,
                        error_lines=[e["excerpt"] for e in d.get("evidence", [])],
                        context={"process_name_source": "agent diagnosis"})
        ev = [f"#{e['call']} {e['tool']}: {e['excerpt']}" for e in d.get("evidence", [])]
        return [Diagnosis(f, FailureClass(d["failure_class"]), 0.9, ev, d.get("root_cause", ""),
                          "See the agent's chosen solution.")]

    def _implement(self) -> str:
        from .patch import build_patch
        if not self.repo:
            raise _Refused("no pipeline repo to change")
        self.res.ai = self._build_ai()
        self.res.diagnoses = self._diagnoses()
        if not self.res.ai.fixes:
            raise _Refused("the chosen solution has no config settings, so there is nothing to implement here")
        patch = build_patch(self.res.diagnoses, self.repo, 0.0, self.inv, self.cfg or load_repo(self.repo),
                            ai=self.res.ai)
        if patch.empty or not patch.ai_used:
            raise _Refused("the change could not be built (the process may not exist in the repo)")
        self.res.patch = patch
        return patch.diff()

    def _mr(self) -> dict:
        if not self.res.patch:
            raise _Refused("implement first")
        if not self.allow_mr or not self.mr_opener:
            return {"opened": False, "why": "opening merge requests is not enabled for this run; the diff is ready"}
        url = self.mr_opener(self.res)
        self.res.mr_url = url
        return {"opened": True, "url": url}

    # ----- one tool call -----
    def _run_tool(self, phase: str, name: str, args: dict) -> tuple[str, bool]:
        if name not in COMMANDS:
            raise _Refused(f"{name} is not a pipefix command")
        if COMMANDS[name]["phase"] != phase:
            raise _Refused(f"{name} belongs to step {PHASE_TITLE[COMMANDS[name]['phase']]}; you are in "
                           f"{PHASE_TITLE[phase]}. Finish this step first.")
        if COMMANDS[name].get("aws"):
            out = self._aws(name, args)
        elif name.startswith("repo_"):
            out = self._repo(name, args)
        elif name.startswith("web_"):
            out = self._web(name, args)
        elif name == "implement_apply":
            out = {"diff": self._implement()}
        elif name == "mr_open":
            out = self._mr()
        elif name == "finish_debug":
            return self._finish_debug(args), True
        elif name == "finish_research":
            return self._finish_research(args), True
        elif name == "finish_decision":
            return self._finish_decision(args), True
        elif name == "finish_implement":
            if not self.res.patch:
                raise _Refused("call implement_apply first")
            self.res.implement_summary = redact(str(args.get("summary", "")))[:600]
            return "Implementation accepted.", True
        elif name == "finish_mr":
            self.res.mr_summary = redact(str(args.get("summary", "")))[:600]
            return "Done.", True
        else:
            raise _Refused(name)
        return _short(out), False

    def _summary(self, name: str, args: dict, text: str, ok: bool) -> str:
        if not ok:
            return f"refused: {text[:200]}"
        arg = ", ".join(f"{k}={v}" for k, v in args.items() if k not in ("candidates", "evidence", "why_not"))[:160]
        return f"{name}({arg})"

    # ----- the loop -----
    def run(self) -> AgentResult:
        if not self.llm_key:
            self.res.error = "The agent needs an LLM key (console, --env-file, or PIPEFIX_LLM_API_KEY / DEEPSEEK_API_KEY)."
            return self.res
        start = {k: v for k, v in self.goal.items() if v}
        messages = [{"role": "system", "content": RULES},
                    {"role": "user", "content": "Failed run to fix: " + json.dumps(start) + ". Pipeline repo: "
                                                + ("available (use the repo_* commands)" if self.repo else "none")}]
        try:
            for phase in PHASES:
                self.res.phase_reached = phase
                tools = [_tool_schema(n) for n, c in COMMANDS.items() if c["phase"] == phase]
                messages.append({"role": "user", "content": PHASE_BRIEF[phase]})
                calls, finish_fails = 0, 0
                done = False
                while not done:
                    if time.time() > self.deadline:
                        raise RuntimeError("time limit reached")
                    msg = self._chat(messages, tools)
                    tcs = msg.get("tool_calls") or []
                    messages.append({"role": "assistant", "content": msg.get("content") or "",
                                     **({"tool_calls": tcs} if tcs else {})})
                    if not tcs:
                        messages.append({"role": "user", "content": "Call one of this step's commands."})
                        calls += 1
                    for tc in tcs:
                        name = tc["function"]["name"]
                        try:
                            args = json.loads(tc["function"].get("arguments") or "{}")
                        except ValueError:
                            args = {}
                        self.n += 1
                        calls += 1
                        try:
                            text, finished = self._run_tool(phase, name, args)
                            ok = True
                        except _Refused as e:
                            text, finished, ok = f"REFUSED: {e}", False, False
                            self.res.refused += 1
                            if COMMANDS.get(name, {}).get("finish"):
                                finish_fails += 1
                        except AwsUnavailable:
                            raise
                        except Exception as e:  # a failed command is reported to the agent, not fatal
                            text, finished, ok = f"ERROR: {type(e).__name__}: {redact(str(e))[:300]}", False, False
                        self.outputs[self.n] = text if ok else ""
                        shown = text if ok else text
                        self._record(phase, name, args, ok, self._summary(name, args, text, ok), shown[:2000])
                        prefix = f"[call #{self.n}] " if phase == "debug" else ""
                        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": prefix + text})
                        if finished:
                            done = True
                    if done and phase == "decide":
                        chosen = next(c for c in self.res.candidates if c["id"] == self.res.decision["chosen"])
                        if not chosen["settings"]:
                            self.res.ai, self.res.diagnoses = self._build_ai(), self._diagnoses()
                            raise _Stop(f"The chosen solution ({chosen['id']}: {chosen['title']}) is not a pipeline "
                                        "config change, so there is nothing to implement or open; see the decision.")
                    if not done and (calls >= BUDGET[phase] + 4 or finish_fails >= MAX_FINISH_TRIES):
                        raise RuntimeError(f"{PHASE_TITLE[phase]} did not finish within its limits "
                                           f"({calls} calls, {finish_fails} rejected finishes)")
                    if not done and calls >= BUDGET[phase]:
                        messages.append({"role": "user", "content": f"Call budget for this step is used up. Call "
                                                                    f"finish_{phase if phase != 'decide' else 'decision'} now."})
        except _Stop as e:
            self.res.error = str(e)
        except AwsUnavailable as e:
            self.res.error = str(e)
        except Exception as e:  # noqa: BLE001 - reported in the result
            self.res.error = redact(f"Agent stopped in {PHASE_TITLE.get(self.res.phase_reached, '?')}: {e}")
        self.inv.calls[:] = self.clients.calls if self.clients else []
        if self.transcript:  # the full conversation (already redacted), for debugging the agent
            self.transcript.write_text(json.dumps(messages, indent=1, default=str), encoding="utf-8")
        return self.res


def open_mr(repo: str | Path, res: AgentResult, draft: bool = True) -> str:
    """Branch, commit, push and open the draft MR for the agent's change (used as the agent's mr_open)."""
    from . import vcs
    from .report import body, pr_title
    repo = Path(repo)
    patch = res.patch
    patch.agent = res
    remote = vcs.remote_info(repo)
    diff = patch.diff()
    h = vcs.diff_hash(diff)
    if existing := vcs.find_existing(remote, h):
        return existing
    if vcs.git(repo, "status", "--porcelain", "--", *patch.files.keys()):
        raise RuntimeError("the files pipefix changes have uncommitted edits; commit or stash them first")
    written = patch.apply(repo)
    text, title = body(res.diagnoses, patch, res.inv), pr_title(patch)
    branch, base = vcs.commit_branch(repo, written, title, text, h)
    try:
        return vcs.open_request(repo, remote, branch, base, title, text, draft=draft)
    finally:
        vcs.git(repo, "checkout", base)


def to_json(res: AgentResult) -> dict:
    """What the console shows about the agent's run."""
    return {"model": res.model, "error": res.error, "phase_reached": res.phase_reached, "refused": res.refused,
            "phases": [{"id": p, "title": PHASE_TITLE[p],
                        "steps": [{"n": s.n, "tool": s.tool, "ok": s.ok, "summary": s.summary,
                                   "aws": bool(COMMANDS.get(s.tool, {}).get("aws"))}
                                  for s in res.steps if s.phase == p]} for p in PHASES],
            "diagnosis": res.diagnosis, "candidates": res.candidates, "decision": res.decision,
            "implement_summary": res.implement_summary, "mr_url": res.mr_url, "mr_summary": res.mr_summary,
            "aws_calls": sum(1 for s in res.steps if s.ok and COMMANDS.get(s.tool, {}).get("aws"))}
