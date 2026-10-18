"""AI-researched fixes (opt-in): search AWS, Nextflow/nf-core and Terraform sources, then let an LLM propose
the config change, and accept only what passes pipefix's own checks.

How it stays safe and auditable:
  * What leaves the machine: process names, exit codes, the task's error lines, the failed process's script and
    the config lines that set its resources, platform facts (instance memory, caps). Account IDs and tokens are
    redacted first. No genomic data, no input files. Search queries are cleaned of paths and S3 URIs.
  * Sources: Serper (Google) restricted to docs.aws.amazon.com, nextflow.io / nf-co.re and Terraform
    (registry + GitHub). Every result page is fetched (allow-listed hosts only) and READ: the model gets the
    relevant passages of each page, not search snippets.
  * Strict grounding rule: every value the model proposes, and every piece of infrastructure advice, must be
    backed by a verbatim quote from a page pipefix fetched. pipefix checks that the quote really is in that
    page's text; a value without a verified quote is rejected, however plausible. The quotes are shown in the
    console and the MR, so the argument for the fix is the sources' own words.
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
ALLOWED = ("memory", "time", "cpus", "errorStrategy", "maxRetries", "queue", "disk", "resourceLimits")
MAX_SOURCES, PAGE_CHARS, EXCERPT_CHARS = 12, 120_000, 5000
QUOTE_MIN, QUOTE_MAX = 30, 400
_KEYWORDS = re.compile(r"memory|GiB|\bGB\b|retr(y|ies)|errorStrategy|exit ?(status|code)|maxRetries|instance|vCPU|"
                       r"cpus|out of memory|OOM|resourceLimits|task\.attempt|reserve|limit|version", re.I)


@dataclass
class AIFix:
    name: str
    settings: dict[str, str]
    rationale: str
    sources: list[int]
    evidence: list[dict] = field(default_factory=list)   # verified quotes: {setting, n, quote}


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
    raw = re.sub(r"(?is)<(script|style|nav|header|footer|svg|noscript)\b.*?</\1>", " ", raw)
    m = re.search(r"(?is)<(main|article)\b.*?</\1>", raw)
    raw = m.group(0) if m else raw
    raw = re.sub(r"(?i)<br\s*/?>|</(p|li|h\d|tr|pre|div|dt|dd)>", "\n", raw)
    text = html.unescape(re.sub(r"(?s)<[^>]+>", " ", raw))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()[:PAGE_CHARS]


def _norm(text: str) -> str:
    """For quote matching: case, whitespace, quote marks and dashes don't matter."""
    text = text.translate(str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-",
                                         "\u2014": "-", "\u00a0": " ", "\u2026": "..."}))
    return re.sub(r"\s+", " ", text).strip().lower()


def excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    """The passages of a page that talk about resources, retries and limits (what the model must read)."""
    lines = [ln.strip() for ln in text.split("\n") if len(ln.strip()) > 25]
    keep, size = [], 0
    for i, ln in enumerate(lines):
        if _KEYWORDS.search(ln):
            for j in (i - 1, i, i + 1):
                if 0 <= j < len(lines) and j not in keep:
                    keep.append(j)
    out = []
    for j in sorted(keep) or range(min(len(lines), 15)):
        if size + len(lines[j]) > limit:
            break
        out.append(lines[j])
        size += len(lines[j]) + 1
    return "\n".join(out)


def verify_quote(quote: str, source: dict | None) -> str | None:
    """None if the quote is verbatim in the page pipefix read, else why not."""
    if not source or not source.get("text"):
        return "cites a page pipefix could not read"
    q = _norm(quote or "")
    if len(q) < QUOTE_MIN:
        return "quote too short to support a claim"
    if len(q) > QUOTE_MAX:
        return "quote too long (quote the sentence, not the page)"
    return None if q in _norm(source["text"]) else f"quote is not in source [{source['n']}]"


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

FIX_PROMPT = """You are a senior bioinformatics platform engineer. Propose the fix for the failed tasks below.
You have READ the numbered sources: each has the relevant passages of the page under "text".

STRICT RULE: your argument must be built on the sources. For EVERY setting you propose, and every piece of
infrastructure advice, give at least one VERBATIM quote (copied exactly, 30-400 characters, one or two
sentences) from a source's "text" that supports it, with the source number. pipefix checks each quote against
the page it downloaded; a setting without a quote that is really in the page is rejected. Do not quote search
snippets, do not paraphrase inside quotes, do not cite a source whose text says nothing about the claim. Your
rationale must reason FROM those quotes (say what the source says, then how it applies to this task's numbers).
If no source supports a change, do not propose it.

Return ONLY JSON:
{"summary": "2-3 sentences: root cause and fix, grounded in the sources",
 "fixes": [{"process": "<failed process name>", "settings": {"<directive>": "<value>"},
            "evidence": [{"setting": "<directive>", "source": <n>, "quote": "<verbatim text from that source>"}],
            "rationale": "argument from the quoted sources to these values, with the numbers you used"}],
 "infrastructure": [{"title": "...", "detail": "...",
                     "evidence": [{"source": <n>, "quote": "<verbatim text>"}]}]}
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
- Every setting needs its own evidence entry (the same quote may support several settings, list each).
  Do not invent sources. If config cannot fix it (bad input, missing
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
        if len(codes) > 32:
            return f"errorStrategy retries {len(codes)} exit codes; a retry rule may cover at most 32 (retrying every " \
                   "failure hides real bugs and multiplies cost)"
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
    elif directive == "resourceLimits":
        part = rf"(memory:\s*{_NUM}\s*\.GB|cpus:\s*\d+|time:\s*{_NUM}\s*\.h)"
        if not re.fullmatch(rf"\[\s*{part}(\s*,\s*{part})*\s*\]", v):
            return f"resourceLimits `{v}` is not in the accepted form, e.g. [ memory: 30.GB ]"
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

        stage(f"AI: reading {len(res.sources)} sources")

        def read(src: dict) -> None:
            if _allowed_host(src["url"]):
                try:
                    src["text"] = _page_text(http.get_text(src["url"]))
                except (urllib.error.URLError, OSError, ValueError):
                    pass
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(read, res.sources))
        for src in res.sources:
            src["read"] = bool(src.get("text"))
        readable = [src for src in res.sources if src["read"]]
        if not readable:
            res.error = "AI research found sources but could not read any of them, so no fix can be grounded."
            return res

        stage("AI: proposing the fix from what the sources say")
        prompt = json.dumps({**ctx, "sources": [{"n": src["n"], "title": src["title"], "url": src["url"],
                                                 "text": excerpt(src["text"])} for src in readable]})
        out = chat(FIX_PROMPT, prompt)
    except (urllib.error.URLError, OSError, KeyError, ValueError, TypeError) as e:
        res.error = redact(f"AI research failed: {e}")
        return res

    res.summary = redact(str(out.get("summary", "")))[:1200]
    by_n = {src["n"]: src for src in res.sources}
    stage("AI: checking every value and every quote against the sources")

    def grounded(items, label: str) -> list[dict]:
        ok = []
        for ev in items or []:
            if not isinstance(ev, dict):
                continue
            n = ev.get("source")
            n = int(n) if isinstance(n, (int, float)) or (isinstance(n, str) and n.isdigit()) else None
            quote = str(ev.get("quote", "")).strip().strip('"').strip()
            why = verify_quote(quote, by_n.get(n))
            if why:
                res.rejected.append(f"{label}: quote from [{n if n is not None else '?'}] rejected, {why}: "
                                    f"\"{quote[:90]}{'...' if len(quote) > 90 else ''}\"")
            else:
                ok.append({"setting": str(ev.get("setting", "")).strip(), "n": n, "quote": redact(" ".join(quote.split()))})
        return ok

    for fx in out.get("fixes") or []:
        name = str(fx.get("process", "")).split(":")[-1].strip()
        evidence = grounded(fx.get("evidence"), name or "?")
        settings, used = {}, []
        for k, v in (fx.get("settings") or {}).items():
            k, v = str(k), str(v)
            why = check(name, k, v, ctx) if name in names else f"{name or '?'} is not one of the failed processes"
            mine = [e for e in evidence if e["setting"] == k]
            if not why and not mine:
                why = "no verified quote from a source supports it (strict grounding rule)"
            if why:
                res.rejected.append(f"{name}.{k} = {v[:80]}: {why}")
            else:
                settings[k] = v.strip()
                used += mine
        if not settings:
            continue
        why = redact(str(fx.get("rationale", "")))[:3000]
        cites = sorted({e["n"] for e in used})
        prev = next((p for p in res.fixes if p.name == name), None)
        if prev:  # the model sometimes splits one process over several entries: merge them
            prev.settings.update(settings)
            prev.rationale = (prev.rationale + " " + why).strip()
            prev.sources = sorted(set(prev.sources) | set(cites))
            prev.evidence += used
        else:
            res.fixes.append(AIFix(name, settings, why, cites, used))
    for it in out.get("infrastructure") or []:
        title = redact(str(it.get("title", "")))[:160]
        ev = grounded(it.get("evidence"), f"advice '{title[:40]}'")
        if not ev:
            res.rejected.append(f"advice '{title[:60]}': no verified quote from a source, so it was not included")
            continue
        res.infrastructure.append({"title": title, "detail": redact(str(it.get("detail", "")))[:800],
                                   "sources": sorted({e["n"] for e in ev}), "evidence": ev})
    for src in res.sources:
        src.pop("text", None)
    return res
