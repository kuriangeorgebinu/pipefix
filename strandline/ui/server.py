"""`strandline ui`: a local web console for investigating failed runs and approving fixes.

Security model: binds to 127.0.0.1 only; refuses requests whose Host header is not localhost (DNS rebinding);
every state-changing request must carry the per-session token embedded in the page (so other websites in the
same browser cannot trigger an investigation, a write or a merge request). AWS access is read-only as everywhere
in Strandline. GitHub/GitLab tokens and the AI research keys (Serper, LLM) come from the environment or are pasted into the console once: either way they
stay in this process's memory, are never written to disk, and are never sent back to the page (only "set", who
it belongs to and its last four characters).
"""
from __future__ import annotations

import json
import os
import re
import secrets
import threading
import traceback
import uuid
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .. import __version__, keys, vcs
from ..configtrace import load_repo, memory_literal_bytes
from ..investigation import Investigation
from ..models import Diagnosis
from ..patch import Patch, build_patch
from ..report import body, pr_title, redact

STATIC = Path(__file__).resolve().parent / "static"
TOKEN = secrets.token_urlsafe(24)
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()
STATUS: dict = {"aws": {"state": "checking"}}
MAX_DISCUSSIONS = 3

# ---------- turning results into JSON for the page ----------

_GB = 1024**3


def _gb(n) -> float | None:
    return round(n / _GB, 1) if n else None


def _num_gb(value: str | None) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\.GB", value or "")
    return float(m.group(1)) if m else None


def _hours(value: str | None) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\.h", value or "")
    return float(m.group(1)) if m else None


def _trace_rows(tr) -> list[dict]:
    if not tr:
        return []
    rows = []
    for s in tr.settings:
        if s.directive in ("memory", "time", "errorStrategy", "maxRetries", "queue"):
            rows.append({"level": s.level, "selector": redact(s.selector), "directive": s.directive,
                         "value": redact(s.value.strip()), "where": s.where,
                         "wins": tr.effective(s.directive) is s})
    return rows


def serialize(inv: Investigation, diags: list[Diagnosis], patch: Patch, repo: Path, cfg) -> dict:
    fixes = {f.name: f for f in patch.fixes}
    facts = inv.facts
    ceiling = facts.get("max_task_memory_mib")
    cap_bytes = memory_literal_bytes(cfg.caps.get("memory", "")) if cfg and cfg.caps.get("memory") else None
    failures = []
    for i, d in enumerate(diags):
        f = d.failure
        fx = fixes.get(f.simple_name)
        tr = patch.traces.get(f.simple_name)
        cur_mem = tr.effective("memory") if tr else None
        cur_time = tr.effective("time") if tr else None
        failures.append({
            "id": i, "process": f.process, "name": f.simple_name, "tag": f.tag, "display": f.display,
            "cls": d.cls.value, "label": d.cls.label, "confidence": d.confidence,
            "explanation": redact(d.explanation), "recommendation": d.recommendation,
            "evidence": [redact(e) for e in d.evidence],
            "fixed": fx is not None, "ai": f.simple_name in patch.ai_used,
            "fix": {"settings": {k: redact(v) for k, v in fx.settings.items()}, "notes": [redact(n) for n in fx.notes]}
            if fx else None,
            "exit_code": f.exit_code, "attempts": f.attempts, "spot_reclaims": f.spot_reclaims,
            "memory_gb": _gb(f.memory_bytes) or _num_gb(cur_mem.value if cur_mem else None),
            "peak_gb": _gb(f.peak_rss_bytes),
            "new_memory_gb": _num_gb(fx.settings.get("memory")) if fx else None,
            "ceiling_gb": round(ceiling / 1024, 1) if ceiling else None,
            "cap_gb": _gb(cap_bytes), "cap_source": cfg.cap_sources.get("memory", "") if cfg else "",
            "time_h": round(f.time_ms / 3_600_000, 2) if f.time_ms else _hours(cur_time.value if cur_time else None),
            "runtime_h": round(f.realtime_ms / 3_600_000, 2) if f.realtime_ms else None,
            "new_time_h": _hours(fx.settings.get("time")) if fx else None,
            "new_queue": fx.settings.get("queue", "").strip("'\"") if fx else "",
            "job_ref": redact(f.job_ref), "image": redact(f.image), "work_dir": f.work_dir, "source": f.source,
            "name_source": f.context.get("process_name_source", ""),
            "log": [redact(x) for x in f.error_lines[-40:]],
            "defined_in": tr.defined_in if tr else "", "labels": tr.labels if tr else [],
            "trace": _trace_rows(tr),
        })
    findings = [{"severity": fd.severity, "title": redact(fd.title), "detail": redact(fd.detail), "process": fd.process}
                for fd in inv.findings + patch.findings]
    calls: dict[str, int] = {}
    for c in inv.calls:
        calls[c] = calls.get(c, 0) + 1
    return {
        "run": redact(inv.run_ref), "platform": inv.platform, "nextflow": inv.nextflow_version,
        "start": inv.start.isoformat() if inv.start else None, "end": inv.end.isoformat() if inv.end else None,
        "repo": str(repo), "failures": failures, "findings": findings,
        "denials": [{**{k: redact(str(v)) for k, v in dn.items()}} for dn in inv.denials],
        "calls": [{"call": k, "count": v} for k, v in sorted(calls.items())], "errors": sorted(set(inv.errors)),
        "diff": patch.diff(), "files": [rel for rel, (o, n) in patch.files.items() if o != n],
        "title": pr_title(patch) if patch.fixes else "", "description": body(diags, patch, inv),
        "config_files": cfg.files if cfg else [], "ondemand_queues": facts.get("ondemand_queues", []),
        "ai": _ai_json(patch),
    }


def _ai_json(patch: Patch) -> dict | None:
    ai = patch.ai
    if not ai:
        return None
    return {"model": ai.model, "error": ai.error, "summary": ai.summary, "used": patch.ai_used,
            "queries": [redact(q) for q in ai.queries], "rejected": ai.rejected,
            "sources": [{k: s.get(k) for k in ("n", "title", "url", "snippet", "site", "read")} for s in ai.sources],
            "fixes": [{"name": f.name, "settings": f.settings, "rationale": f.rationale, "sources": f.sources,
                       "evidence": f.evidence} for f in ai.fixes],
            "infrastructure": ai.infrastructure}


# ---------- jobs ----------

def _new_job(kind: str, label: str) -> dict:
    job = {"id": uuid.uuid4().hex[:10], "kind": kind, "label": label, "status": "running", "stages": [],
           "result": None, "error": None, "created": datetime.now().isoformat(timespec="seconds"),
           "patch": None, "repo": None}
    with LOCK:
        JOBS[job["id"]] = job
    return job


def _stage(job: dict, name: str) -> None:
    for s in job["stages"]:
        if s["state"] == "running":
            s["state"] = "done"
    job["stages"].append({"name": name, "state": "running"})


def _run_investigation(job: dict, req: dict) -> None:
    try:
        from ..investigate import investigate
        mode = req.get("mode")
        if mode == "demo":
            from ..demo import DEMO_JOBS, demo_clients, demo_repo
            _stage(job, "Preparing the demo account and pipeline")
            clients, repo = demo_clients(), demo_repo()
            kwargs = {"batch_jobs": DEMO_JOBS}
        else:
            from ..aws.session import boto3_clients
            _stage(job, "Connecting to AWS")
            clients = boto3_clients(req.get("profile") or None, req.get("region") or None)
            repo = Path(req.get("repo") or ".").expanduser().resolve()
            value = (req.get("value") or "").strip()
            if not value:
                raise ValueError("Enter a queue name, job ids or a run id.")
            kwargs = {"batch_queue": value} if mode == "batch-queue" else \
                {"batch_jobs": value.replace(",", " ").split()} if mode == "batch-jobs" else {"omics_run": value}
            kwargs["since_hours"] = float(req.get("since") or 24)
            if req.get("fix_mode") == "agent":
                return _run_agent(job, req, clients, repo, kwargs)
        inv, diags = investigate(clients, on_stage=lambda n: _stage(job, n), **kwargs)
        cfg = None
        if (repo / "nextflow.config").is_file():
            _stage(job, "Tracing the pipeline config")
            cfg = load_repo(repo)
            ai = None
            if req.get("ai"):
                from ..research import research
                ai = research(inv, diags, cfg, tuple(req.get("nf_profiles") or ()), on_stage=lambda n: _stage(job, n))
            _stage(job, "Building the fix")
            patch = build_patch(diags, repo, inv=inv, cfg=cfg, profiles=tuple(req.get("nf_profiles") or ()), ai=ai)
        else:
            _stage(job, "No nextflow.config in the repo folder, so only diagnosing")
            patch = Patch({}, [], diags, {})
        job["patch"], job["repo"] = patch, repo
        job["result"] = serialize(inv, diags, patch, repo, cfg)
        _stage(job, "Done")
        job["stages"][-1]["state"] = "done"
        job["status"] = "done"
    except Exception as e:  # shown to the user as-is (redacted)
        for s in job["stages"]:
            if s["state"] == "running":
                s["state"] = "failed"
        job["status"], job["error"] = "failed", redact(f"{e}")
        traceback.print_exc()


def _run_agent(job: dict, req: dict, clients, repo: Path, goal: dict) -> None:
    """The agentic mode: the AI makes the AWS calls, researches, decides, implements and opens the MR."""
    import importlib
    from .. import agent as _agent_mod
    importlib.reload(_agent_mod)   # pick up agent changes without restarting the console (tokens stay in memory)
    from ..agent import PHASE_TITLE, Agent
    job["repo"] = repo
    goal = {k: v for k, v in goal.items() if k != "since_hours"}
    has_repo = (repo / "nextflow.config").is_file()
    # the agents never open the MR themselves here: a person reviews the change and approves it (open-mr)

    def step(st):
        for s in job["stages"]:
            if s["state"] == "running":
                s["state"] = "done"
        job["stages"].append({"name": f"{PHASE_TITLE[st.phase]} · #{st.n} {st.summary}",
                              "state": "done" if st.ok else "failed"})
    job["trace"] = []
    agent = Agent(clients, repo if has_repo else None, goal, allow_mr=False, on_step=step,
                  on_event=job["trace"].append, require_approval=True)
    job["agent"] = agent
    _stage(job, "Agents starting: 1. Debug, 2. Research, 3. Pick, 3b. Expert review, 4. Implement; then you review "
                "and approve the merge request")
    agent.run()
    _agent_result(job, agent)


def _agent_result(job: dict, agent) -> None:
    from ..agent import to_json
    res, repo = agent.res, Path(job["repo"])
    has_repo = bool(agent.repo)
    patch = res.patch
    if patch is not None:
        patch.agent = res
    else:
        patch = Patch({}, [], res.diagnoses, {})
    cfg = load_repo(repo) if has_repo else None
    job["patch"], job["repo"] = (res.patch, repo) if res.patch is not None else (None, repo)
    job["result"] = serialize(res.inv, res.diagnoses, patch, repo, cfg)
    job["result"]["agent"] = to_json(res)
    job["result"]["approval_needed"] = bool(to_json(res)["awaiting_approval"]) and not job.get("mr")
    for s in job["stages"]:
        if s["state"] == "running":
            s["state"] = "done"
    job["status"] = "stopped" if res.error else "done"   # stopped: e.g. the experts did not agree; nothing opened


def _discuss(job: dict, message: str) -> None:
    """A person disagrees with or asks about the agents' result: they verify, answer or revise, experts re-review."""
    agent = job["agent"]
    prev = job["status"]
    try:
        agent.discuss(message)
        job.pop("written", None)          # a revised change has not been written anywhere yet
        _agent_result(job, agent)
    except Exception as e:  # noqa: BLE001
        job["status"] = prev
        job["discuss_error"] = redact(str(e))
        traceback.print_exc()


def _apply(job: dict) -> dict:
    patch: Patch = job["patch"]
    if not patch or patch.empty:
        raise ValueError("There is nothing to write for this investigation.")
    written = patch.apply(job["repo"])
    job["written"] = written
    return {"written": written, "repo": str(job["repo"])}


def _open_mr(job: dict, ready: bool) -> dict:
    from .. import vcs
    patch: Patch = job["patch"]
    if not patch or patch.empty:
        raise ValueError("There is nothing to propose for this investigation.")
    repo = job["repo"]
    remote = vcs.remote_info(repo)
    diff = patch.diff()
    h = vcs.diff_hash(diff)
    if existing := vcs.find_existing(remote, h):
        return {"url": existing, "existing": True}
    if vcs.git(repo, "status", "--porcelain", "--", *patch.files.keys()) and not job.get("written"):
        raise ValueError("The files Strandline changes have uncommitted edits. Commit or stash them first.")
    written = job.get("written") or patch.apply(repo)
    text, title = job["result"]["description"], pr_title(patch)
    if "agent" in job:
        text += ("\n\n---\nA person reviewed this AI-generated change and its evidence in the Strandline console and "
                 "approved opening this merge request.\n")
    branch, base = vcs.commit_branch(repo, written, title, text, h)
    try:
        url = vcs.open_request(repo, remote, branch, base, title, text, draft=not ready)
    finally:
        vcs.git(repo, "checkout", base)
    return {"url": url, "existing": False, "branch": branch}


def _check_aws() -> None:
    try:
        from ..aws.session import boto3_clients
        c = boto3_clients()
        ident = c.call("sts", "get_caller_identity")
        if "__error__" in ident:
            STATUS["aws"] = {"state": "error", "detail": ident["__error__"], "region": c.region}
        else:
            STATUS["aws"] = {"state": "ok", "region": c.region, "identity": redact(ident.get("Arn", ""))}
    except Exception as e:
        STATUS["aws"] = {"state": "missing", "detail": redact(str(e))}


# ---------- HTTP ----------

class Handler(BaseHTTPRequestHandler):
    server_version = "strandline"

    def log_message(self, *args):
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, default=str).encode())

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").split(":")[0]
        return host in ("127.0.0.1", "localhost")

    def do_GET(self):
        if not self._host_ok():
            return self._json(403, {"error": "Strandline only answers on localhost"})
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            html = (STATIC / "index.html").read_text(encoding="utf-8").replace("__STRANDLINE_TOKEN__", TOKEN)
            return self._send(200, html.encode(), "text/html; charset=utf-8")
        if m := re.fullmatch(r"/fonts/([\w.-]+\.woff2)", u.path):
            f = STATIC / "fonts" / m.group(1)
            if f.is_file():
                self.send_response(200)
                data = f.read_bytes()
                self.send_header("Content-Type", "font/woff2")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "max-age=86400")
                self.end_headers()
                return self.wfile.write(data)
        if u.path == "/api/status":
            return self._json(200, {"version": __version__, **STATUS, "cwd": os.getcwd(),
                                    "cwd_is_pipeline": Path("nextflow.config").is_file(),
                                    "tokens": {p: vcs.token_status(p) for p in vcs.PROVIDERS},
                                    "keys": keys.status()})
        if u.path == "/api/jobs":
            return self._json(200, [{k: j[k] for k in ("id", "kind", "label", "status", "created")} |
                                    {"failures": len(j["result"]["failures"]) if j["result"] else None,
                                     "fixed": sum(f["fixed"] for f in j["result"]["failures"]) if j["result"] else None}
                                    for j in sorted(JOBS.values(), key=lambda j: j["created"], reverse=True)])
        if m := re.fullmatch(r"/api/jobs/(\w+)", u.path):
            job = JOBS.get(m.group(1))
            if not job:
                return self._json(404, {"error": "No such investigation."})
            since = int((parse_qs(u.query).get("since") or ["0"])[0] or 0)
            return self._json(200, {k: job[k] for k in ("id", "kind", "label", "status", "stages", "result", "error")}
                              | {"written": job.get("written"), "mr": job.get("mr"),
                                 "discuss_error": job.get("discuss_error"),
                                 "trace": job["trace"][since:] if "trace" in job else None,
                                 "trace_len": len(job.get("trace") or [])})
        self._json(404, {"error": "Not found."})

    def do_POST(self):
        # read the body first, even for requests we refuse: replying early makes Windows reset the connection
        try:
            n = min(int(self.headers.get("Content-Length") or 0), 1_000_000)
        except ValueError:
            n = 0
        raw = self.rfile.read(n) if n else b""
        if not self._host_ok():
            return self._json(403, {"error": "Strandline only answers on localhost"})
        if not secrets.compare_digest(self.headers.get("X-Strandline-Token", ""), TOKEN):
            return self._json(403, {"error": "Missing or wrong session token. Reload the page."})
        try:
            req = json.loads(raw or b"{}")
        except ValueError:
            return self._json(400, {"error": "Invalid JSON."})
        u = urlparse(self.path)
        try:
            if u.path == "/api/token":
                return self._json(200, vcs.set_token(req.get("provider", ""), req.get("token", ""), req.get("host")))
            if u.path == "/api/token/clear":
                return self._json(200, vcs.clear_token(req.get("provider", "")))
            if u.path == "/api/key":
                return self._json(200, keys.set_key(req.get("kind", ""), req.get("key", ""), req.get("base_url", ""),
                                                    req.get("model", "")))
            if u.path == "/api/key/clear":
                return self._json(200, keys.clear(req.get("kind", "")))
            if u.path == "/api/investigate":
                label = {"demo": "Demo run", "batch-queue": f"Queue {req.get('value', '')}",
                         "batch-jobs": "Batch jobs", "omics-run": f"HealthOmics run {req.get('value', '')}"}.get(
                    req.get("mode"), "Investigation")
                job = _new_job(req.get("mode", ""), label)
                threading.Thread(target=_run_investigation, args=(job, req), daemon=True).start()
                return self._json(202, {"id": job["id"]})
            if m := re.fullmatch(r"/api/jobs/(\w+)/discuss", u.path):
                job = JOBS.get(m.group(1))
                if not job or "agent" not in job:
                    return self._json(404, {"error": "Only an AI agent run can be discussed."})
                with LOCK:
                    if job["status"] not in ("done", "stopped"):
                        return self._json(409, {"error": "The agents are still working on this run."})
                    if job.get("mr"):
                        return self._json(409, {"error": "The merge request is open; discuss the change there."})
                    msg = str(req.get("message") or "").strip()
                    if not msg or len(msg) > 2000:
                        return self._json(400, {"error": "Write your feedback (up to 2,000 characters)."})
                    if len(job["agent"].res.discussion) >= MAX_DISCUSSIONS:
                        return self._json(409, {"error": f"This run has had {MAX_DISCUSSIONS} discussions. Start a "
                                                         "new investigation to go on."})
                    job["status"], job["discuss_error"] = "discussing", None
                threading.Thread(target=_discuss, args=(job, msg), daemon=True).start()
                return self._json(202, {"ok": True})
            if m := re.fullmatch(r"/api/jobs/(\w+)/(apply|open-mr)", u.path):
                job = JOBS.get(m.group(1))
                if not job or job["status"] != "done":
                    return self._json(409, {"error": "That investigation isn't finished."})
                if "agent" in job and not job["result"].get("approval_needed") and not job.get("mr"):
                    return self._json(409, {"error": "The expert agents have not agreed on a change to approve."})
                if m.group(2) == "apply":
                    return self._json(200, _apply(job))
                if "agent" in job and not req.get("reviewed"):
                    return self._json(400, {"error": "Confirm that you reviewed the change and its evidence first."})
                job["mr"] = _open_mr(job, bool(req.get("ready")))
                if "agent" in job:
                    job["result"]["approval_needed"] = False
                return self._json(200, job["mr"])
        except Exception as e:
            return self._json(400, {"error": redact(str(e))})
        self._json(404, {"error": "Not found."})


def serve(port: int = 8787, open_browser: bool = True) -> None:
    threading.Thread(target=_check_aws, daemon=True).start()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"strandline {__version__} console: {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
