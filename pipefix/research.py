"""AI-researched fixes (opt-in): search AWS, Nextflow/nf-core and Terraform sources, then let an LLM propose
the config change, and accept only what passes pipefix's own checks.

How it stays safe and auditable:
  * What leaves the machine: process names, exit codes, the task's error lines, the failed process's script and
    the config lines that set its resources, platform facts (instance memory, caps). Account IDs and tokens are
    redacted first. No genomic data, no input files. Search queries are cleaned of paths and S3 URIs.
  * Sources: Serper (Google) restricted to docs.aws.amazon.com, nextflow.io / nf-co.re and Terraform
    (registry + GitHub). The top pages are fetched (allow-listed hosts only) and numbered; the model must cite
    them by number, and every cited source is linked in the MR.
  * The model writes a structured proposal, never config text. Each value is checked against a strict grammar
    (Nextflow closures run code, so only `{ N.GB * task.attempt }`-style values are accepted), against the
    failed processes, and against the platform (memory above the largest instance is refused). Anything that
    fails is listed as rejected, with the reason; nothing unchecked is written.
  * Infrastructure changes (Terraform, compute environments, IAM) are advice in the MR, never applied.
"""
from __future__ import annotations

import html
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import keys
from .configtrace import RepoConfig
from .investigation import Investigation
from .models import Diagnosis
from .report import redact

SITES = {
    "aws": "site:docs.aws.amazon.com",
    "nextflow": "(site:nextflow.io OR site:nf-co.re OR site:training.nextflow.io)",
    "terraform": "terraform (site:registry.terraform.io OR site:github.com)",
}
FETCH_HOSTS = ("docs.aws.amazon.com", "nextflow.io", "www.nextflow.io", "training.nextflow.io", "nf-co.re",
               "registry.terraform.io", "github.com", "raw.githubusercontent.com")
ALLOWED = ("memory", "time", "cpus", "errorStrategy", "maxRetries", "queue", "disk")
MAX_SOURCES, FETCH_TOP, PAGE_CHARS = 12, 4, 3500


@dataclass
class AIFix:
    name: str
    settings: dict[str, str]
    rationale: str
    sources: list[int]


@dataclass
class AIResult:
    model: str = ""
    queries: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)      # {n, title, url, snippet, site}
    fixes: list[AIFix] = field(default_factory=list)
    infrastructure: list[dict] = field(default_factory=list)  # {title, detail, sources}
    rejected: list[str] = field(default_factory=list)
    summary: str = ""
    error: str = ""

    def cite(self, ns: list[int]) -> list[dict]:
        by_n = {s["n"]: s for s in self.sources}
        return [by_n[n] for n in ns if n in by_n]


# ---------- HTTP (injectable for tests) ----------

class Http:
    def post_json(self, url: str, payload: dict, headers: dict, timeout: int = 120) -> dict:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json", **headers})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def get_text(self, url: str, timeout: int = 15) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": "pipefix (+https://github.com/kuriangeorgebinu/pipefix)"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if "html" not in r.headers.get("Content-Type", "html") and "text" not in r.headers.get("Content-Type", ""):
                return ""
            return r.read(400_000).decode("utf-8", errors="replace")


def _page_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|nav|header|footer|svg)\b.*?</\1>", " ", raw)
    m = re.search(r"(?is)<(main|article)\b.*?</\1>", raw)
    raw = m.group(0) if m else raw
    text = html.unescape(re.sub(r"(?s)<[^>]+>", " ", raw))
    return re.sub(r"\s+", " ", text).strip()[:PAGE_CHARS]


def _allowed_host(url: str) -> bool:
    m = re.match(r"https://([^/:]+)", url)
    return bool(m) and any(m.group(1) == h or m.group(1).endswith("." + h) for h in FETCH_HOSTS)


# ---------- what the model is told ----------

def _process_source(cfg: RepoConfig | None, name: str, limit: int = 60) -> str:
    if not cfg or name not in cfg.processes:
        return ""
    where = cfg.processes[name][0]
    rel, _, line = where.rpartition(":")
    try:
        lines = (cfg.root / rel).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    start = int(line) - 1
    out = []
    for i, ln in enumerate(lines[start:start + limit]):
        if i and re.match(r"\s*(process|workflow)\s+\w*\s*\{", ln):
            break
        out.append(ln)
    return "\n".join(out)


def context(inv: Investigation | None, diagnoses: list[Diagnosis], cfg: RepoConfig | None,
            profiles: tuple[str, ...] = ()) -> dict:
    facts = (inv.facts if inv else {}) or {}
    tasks = []
    for d in diagnoses[:6]:
        f = d.failure
        tr = cfg.trace(f.process, profiles) if cfg and not f.process.startswith("(") else None
        tasks.append({
            "process": f.simple_name, "tag": f.tag, "diagnosis": d.cls.value, "confidence": round(d.confidence, 2),
            "exit_code": f.exit_code, "status_reason": f.status_reason, "attempts": f.attempts,
            "memory_requested_gb": round(f.memory_bytes / 1024**3, 2) if f.memory_bytes else None,
            "peak_memory_gb": round(f.peak_rss_bytes / 1024**3, 2) if f.peak_rss_bytes else None,
            "runtime_s": round(f.realtime_ms / 1000) if f.realtime_ms else None,
            "error_lines": f.error_lines[-30:],
            "config_now": [{"directive": s.directive, "value": s.value.strip(), "set_in": s.where,
                            "by": s.selector, "wins": tr.effective(s.directive) is s}
                           for s in (tr.settings if tr else [])
                           if s.directive in ALLOWED],
            "labels": tr.labels if tr else [],
            "process_source": _process_source(cfg, f.simple_name),
        })
    plat = {"platform": inv.platform if inv else "files", "nextflow": inv.nextflow_version if inv else "",
            "healthomics": bool(facts.get("healthomics")),
            "largest_task_memory_gb": round(facts["max_task_memory_mib"] / 1024, 1)
            if facts.get("max_task_memory_mib") else None,
            "ondemand_queues": facts.get("ondemand_queues", []),
            "params": dict(list((cfg.params if cfg else {}).items())[:40]),
            "caps": dict(cfg.caps) if cfg else {}}
    return json.loads(redact(json.dumps({"failed_tasks": tasks, "platform": plat}, default=str)))


PLAN_PROMPT = """You help fix failed Nextflow genomics pipeline runs on AWS (AWS Batch or AWS HealthOmics).
Given the failure context, write web search queries that would find the authoritative fix.
Return JSON: {"queries": [{"site": "aws" | "nextflow" | "terraform", "q": "..."}]} with 3 to 6 queries.
Rules: short keyword queries (under 12 words). Never include sample names, file paths, S3 URIs, account IDs,
or run IDs. Use at least one "aws" query and one "nextflow" query; add "terraform" queries only if the fix may
need infrastructure (compute environment, instance types, IAM, HealthOmics run storage)."""

FIX_PROMPT = """You are a senior bioinformatics platform engineer. Propose the fix for the failed tasks below,
using the numbered sources. Return ONLY JSON:
{"summary": "2-3 sentences: root cause and fix",
 "fixes": [{"process": "<failed process name>", "settings": {"<directive>": "<value>"},
            "rationale": "why these values, with the numbers you used", "sources": [<source numbers>]}],
 "infrastructure": [{"title": "...", "detail": "...", "sources": [<numbers>]}]}
Rules:
- Only processes listed in failed_tasks. Only these directives: memory, time, cpus, errorStrategy, maxRetries,
  queue, disk. Value formats ONLY:
  memory  "{ N.GB * task.attempt }"  or  "N.GB"  or  "{ [N.GB * task.attempt, M.GB].min() }"
  time    "{ N.h * task.attempt }"  or  "N.h"
  cpus    "N"  or  "{ N * task.attempt }"
  errorStrategy  "{ task.exitStatus in <codes> ? 'retry' : 'finish' }" where <codes> uses only integers,
                 ranges (a..b), lists [a, b], parentheses and +;  or one of 'retry' 'finish' 'ignore' 'terminate'
  maxRetries "N"   queue "'queue-name'"   disk "N.GB"
- Size memory from what the task actually needs: read process_source for tool flags that set memory or buffers
  (e.g. --buffer-gb, -Xmx, sort -S, --mem, -m) and resolve params.* from platform.params; leave headroom (~25%).
  Never exceed platform.largest_task_memory_gb when it is given.
- If you rely on retries, make sure errorStrategy actually retries the observed exit_code; if the existing rule
  doesn't cover it, write an errorStrategy that keeps the existing codes and adds it.
- Cite sources by number for every fix. Do not invent sources. If config cannot fix it (bad input, missing
  reference, IAM, container), return no fix for that process and explain under infrastructure.
- Infrastructure changes (Terraform, compute environments, IAM, HealthOmics run storage) go ONLY in
  "infrastructure" as advice; never put them in settings."""


# ---------- the checks every AI value must pass ----------

_NUM = r"(\d+(?:\.\d+)?)"
_MEM = [re.compile(rf"^\{{\s*{_NUM}\s*\.GB\s*\*\s*task\.attempt\s*\}}$"), re.compile(rf"^'?{_NUM}\s*\.?\s*GB'?$"),
        re.compile(rf"^\{{\s*\[\s*{_NUM}\s*\.GB\s*\*\s*task\.attempt\s*,\s*{_NUM}\s*\.GB\s*\]\s*\.min\(\)\s*\}}$")]
_TIME = [re.compile(rf"^\{{\s*{_NUM}\s*\.h\s*\*\s*task\.attempt\s*\}}$"), re.compile(rf"^'?{_NUM}\s*\.?\s*h'?$")]
_CPUS = [re.compile(r"^(\d+)$"), re.compile(r"^\{\s*(\d+)\s*\*\s*task\.attempt\s*\}$")]
_ES_WORD = re.compile(r"^'(retry|finish|ignore|terminate)'$")
_ES = re.compile(r"^\{\s*task\.exitStatus\s+in\s+([\d\s.,+()\[\]]+?)\s*\?\s*'(retry|ignore)'\s*:\s*"
                 r"'(finish|terminate|ignore)'\s*\}$")


def exit_codes(expr: str) -> set[int] | None:
    """The codes a safe errorStrategy set literal covers, or None if it isn't one we can read."""
    if not re.fullmatch(r"[\d\s.,+()\[\]]+", expr) or re.search(r"\.\.\.", expr):
        return None
    codes: set[int] = set()
    for a, b in re.findall(r"(\d+)\s*\.\.\s*(\d+)", expr):
        if int(b) - int(a) > 1000:
            return None
        codes |= set(range(int(a), int(b) + 1))
    codes |= {int(x) for x in re.findall(r"(?<![.\d])(\d+)(?![.\d])", re.sub(r"\d+\s*\.\.\s*\d+", "", expr))}
    return codes


def check(name: str, directive: str, value: str, ctx: dict) -> str | None:
    """None if the AI's value is acceptable, else why not."""
    v = value.strip()
    task = next((t for t in ctx["failed_tasks"] if t["process"] == name), None)
    plat = ctx["platform"]
    if task is None:
        return f"{name} is not one of the failed processes"
    if directive not in ALLOWED:
        return f"{directive} is not a setting pipefix lets the AI change"
    if len(v) > 160:
        return "value too long"
    if directive == "memory":
        m = next((p.match(v) for p in _MEM if p.match(v)), None)
        if not m:
            return f"memory value `{v}` is not in an accepted form"
        first = float(m.group(1))
        top = plat.get("largest_task_memory_gb")
        if not 0.5 <= first <= 2048:
            return f"memory {first} GB is out of range"
        if top and first > top:
            return f"memory {first} GB is more than the largest instance can give a task ({top} GB)"
    elif directive == "time":
        m = next((p.match(v) for p in _TIME if p.match(v)), None)
        if not m or not 0.05 <= float(m.group(1)) <= 168:
            return f"time value `{v}` is not in an accepted form or range (up to 168 h)"
    elif directive == "cpus":
        m = next((p.match(v) for p in _CPUS if p.match(v)), None)
        if not m or not 1 <= int(m.group(1)) <= 96:
            return f"cpus value `{v}` is not an accepted form or range (1-96)"
    elif directive == "maxRetries":
        if not re.fullmatch(r"[0-5]", v):
            return "maxRetries must be 0-5"
    elif directive == "errorStrategy":
        if _ES_WORD.match(v):
            return None
        m = _ES.match(v)
        codes = exit_codes(m.group(1)) if m else None
        if codes is None:
            return f"errorStrategy `{v}` is not in the accepted form (it would run as code, so it must match exactly)"
        if task.get("exit_code") is not None and m.group(2) == "retry" and task["exit_code"] not in codes:
            return f"errorStrategy doesn't retry the exit code the task failed with ({task['exit_code']})"
    elif directive == "queue":
        if plat.get("healthomics"):
            return "HealthOmics has no Batch queues"
        if not re.fullmatch(r"'[A-Za-z0-9_-]{1,128}'", v):
            return f"queue `{v}` is not a quoted queue name"
        known = plat.get("ondemand_queues") or []
        if known and v.strip("'") not in known:
            return f"queue {v} was not found in this account"
    elif directive == "disk":
        m = re.fullmatch(rf"'?{_NUM}\s*\.?\s*GB'?", v)
        if not m or not 1 <= float(m.group(1)) <= 4000:
            return f"disk value `{v}` is not an accepted form or range"
    return None


# ---------- the research ----------

def _clean_query(q: str) -> str:
    q = redact(q)
    q = re.sub(r"s3://\S+|[A-Za-z]:\\\S+|/\S+/\S+|\b\d{6,}\b", " ", q)
    return re.sub(r"\s+", " ", q).strip()[:120]


def _json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


def research(inv: Investigation | None, diagnoses: list[Diagnosis], cfg: RepoConfig | None,
             profiles: tuple[str, ...] = (), on_stage=None, http: Http | None = None) -> AIResult:
    stage = on_stage or (lambda n: None)
    http = http or Http()
    llm_key, base, model = keys.llm_endpoint()
    serper_key, _ = keys.get("serper")
    res = AIResult(model=model)
    if not diagnoses:
        return res
    if not llm_key or not serper_key:
        res.error = ("AI research needs a Serper key and an LLM key: add them in the console, or set "
                     "SERPER_API_KEY and PIPEFIX_LLM_API_KEY (or DEEPSEEK_API_KEY / OPENAI_API_KEY).")
        return res
    ctx = context(inv, diagnoses, cfg, profiles)
    names = {t["process"] for t in ctx["failed_tasks"]}

    def chat(system: str, user: str) -> dict:
        out = http.post_json(f"{base}/chat/completions",
                             {"model": model, "temperature": 0, "response_format": {"type": "json_object"},
                              "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                             {"Authorization": f"Bearer {llm_key}"})
        return _json(out["choices"][0]["message"]["content"])

    try:
        stage("AI: choosing what to search for")
        plan = chat(PLAN_PROMPT, json.dumps(ctx))
        queries = []
        for q in (plan.get("queries") or [])[:6]:
            site = q.get("site") if q.get("site") in SITES else "aws"
            clean = _clean_query(str(q.get("q", "")))
            if clean:
                queries.append((site, clean))
        if not any(s == "aws" for s, _ in queries):  # always look at the AWS docs for the platform
            what = "HealthOmics workflow run task" if ctx["platform"]["healthomics"] else "AWS Batch job"
            queries.append(("aws", f"{what} {diagnoses[0].cls.label} memory"))
        res.queries = [f"{q} {SITES[s]}" for s, q in queries]

        stage("AI: searching AWS, Nextflow and Terraform docs")
        seen = set()
        for (site, _), full in zip(queries, res.queries):
            hits = http.post_json(keys.SERPER_URL, {"q": full, "num": 5}, {"X-API-KEY": serper_key}, timeout=20)
            for h in hits.get("organic", [])[:5]:
                url = str(h.get("link", ""))
                if not url.startswith("https://") or url in seen or len(res.sources) >= MAX_SOURCES:
                    continue
                seen.add(url)
                res.sources.append({"n": len(res.sources) + 1, "title": str(h.get("title", ""))[:160], "url": url,
                                    "snippet": str(h.get("snippet", ""))[:400], "site": site})

        stage("AI: reading the top sources")
        for s in res.sources[:FETCH_TOP]:
            if _allowed_host(s["url"]):
                try:
                    s["text"] = _page_text(http.get_text(s["url"]))
                except (urllib.error.URLError, OSError, ValueError):
                    pass

        stage("AI: proposing the fix")
        prompt = json.dumps({**ctx, "sources": [{k: s.get(k, "") for k in ("n", "title", "url", "snippet", "text")}
                                                for s in res.sources]})
        out = chat(FIX_PROMPT, prompt)
    except (urllib.error.URLError, OSError, KeyError, ValueError, TypeError) as e:
        res.error = redact(f"AI research failed: {e}")
        return res

    res.summary = redact(str(out.get("summary", "")))[:1200]
    valid_n = {s["n"] for s in res.sources}
    stage("AI: checking the proposal")
    for fx in out.get("fixes") or []:
        name = str(fx.get("process", "")).split(":")[-1].strip()
        cites = [n for n in fx.get("sources") or [] if isinstance(n, int) and n in valid_n]
        settings = {}
        for k, v in (fx.get("settings") or {}).items():
            why = check(name, str(k), str(v), ctx) if name in names else f"{name or '?'} is not one of the failed processes"
            if why:
                res.rejected.append(f"{name}.{k} = {str(v)[:80]}: {why}")
            else:
                settings[str(k)] = str(v).strip()
        if settings and not cites:
            res.rejected.append(f"{name}: no valid source cited, so the proposal was not used")
            continue
        if settings:
            why = redact(str(fx.get("rationale", "")))[:600]
            prev = next((p for p in res.fixes if p.name == name), None)
            if prev:  # the model sometimes splits one process over several entries: merge them
                prev.settings.update(settings)
                prev.rationale = (prev.rationale + " " + why).strip()
                prev.sources += [n for n in cites if n not in prev.sources]
            else:
                res.fixes.append(AIFix(name, settings, why, cites))
    for it in out.get("infrastructure") or []:
        res.infrastructure.append({"title": redact(str(it.get("title", "")))[:160],
                                   "detail": redact(str(it.get("detail", "")))[:800],
                                   "sources": [n for n in it.get("sources") or [] if isinstance(n, int) and n in valid_n]})
    for s in res.sources:
        s.pop("text", None)
    return res
