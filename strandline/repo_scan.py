"""Trace a pipeline repo's own configuration for a process.

Answers, for a failed process: what is it called in the code, which labels does it carry, which config
selectors (withName / withLabel) set its memory and time today and in which file, and which global caps
(resourceLimits, params.max_memory / max_time) would silently clip a fix.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .units import parse_duration, parse_memory

SKIP_DIRS = {".git", "work", ".nextflow", "results", "node_modules", ".venv", "__pycache__"}
_PROCESS = re.compile(r"^\s*process\s+([A-Za-z_][\w]*)\s*\{", re.M)
_LABEL = re.compile(r"^\s*label\s+['\"]([^'\"]+)['\"]", re.M)
_SELECTOR = re.compile(r"(withName|withLabel)\s*:\s*(?:'([^']+)'|\"([^\"]+)\"|([\w.*|:^$\[\]()-]+))\s*\{")
_SETTING = re.compile(r"^\s*(memory|time|cpus|errorStrategy|maxRetries)\s*=\s*(.+?)\s*$", re.M)
_INLINE = re.compile(r"\b(memory|time|cpus)\s*=\s*(\{[^}]*\}|'[^']*'|\"[^\"]*\"|[\w.]+)")


@dataclass
class Setting:
    file: str
    selector: str          # e.g. "withLabel:process_high" or "withName:GATK4_HAPLOTYPECALLER" or "process"
    key: str
    value: str


@dataclass
class ProcessTrace:
    name: str
    defined_in: str = ""
    labels: list[str] = field(default_factory=list)
    settings: list[Setting] = field(default_factory=list)   # in the order Nextflow would apply them
    memory_cap_bytes: int | None = None
    time_cap_ms: int | None = None
    cap_source: str = ""

    def current(self, key: str) -> Setting | None:
        """The setting that wins: withName beats withLabel beats plain process scope; later beats earlier."""
        rank = {"process": 0, "withLabel": 1, "withName": 2}
        found = [s for s in self.settings if s.key == key]
        return max(found, key=lambda s: (rank[s.selector.split(":")[0]], found.index(s))) if found else None

    def describe(self) -> list[str]:
        out = []
        if self.defined_in:
            out.append(f"`{self.name}` is defined in `{self.defined_in}`"
                       + (f" with label(s) {', '.join(f'`{l}`' for l in self.labels)}" if self.labels else ""))
        for key in ("memory", "time"):
            s = self.current(key)
            if s:
                out.append(f"current {key}: `{s.value}` from `{s.selector}` in `{s.file}`")
        if self.memory_cap_bytes or self.time_cap_ms:
            caps = []
            if self.memory_cap_bytes:
                caps.append(f"memory ≤ {self.memory_cap_bytes / 1024**3:.0f} GB")
            if self.time_cap_ms:
                caps.append(f"time ≤ {self.time_cap_ms / 3_600_000:.0f} h")
            out.append(f"pipeline caps: {', '.join(caps)} ({self.cap_source})")
        return out


def _files(repo: Path, suffixes: tuple[str, ...]) -> list[Path]:
    out = []
    for p in sorted(repo.rglob("*")):
        if p.is_file() and p.suffix in suffixes and not (set(p.relative_to(repo).parts) & SKIP_DIRS):
            out.append(p)
    return out


def _block(text: str, open_brace: int) -> str:
    """Text between a '{' at open_brace and its matching '}'."""
    depth = 0
    for i in range(open_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1:i]
    return text[open_brace + 1:]


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"(?m)^\s*//.*$", "", text)


class RepoIndex:
    def __init__(self, repo: str | Path):
        self.repo = Path(repo)
        self.processes: dict[str, tuple[str, list[str]]] = {}     # name -> (file, labels)
        self.selectors: list[tuple[str, str, str, list[tuple[str, str]]]] = []  # (file, kind, pattern, settings)
        self.memory_cap: tuple[int, str] | None = None
        self.time_cap: tuple[int, str] | None = None
        self._scan()

    # ---- scanning ----
    def _scan(self):
        for nf in _files(self.repo, (".nf",)):
            text = _strip_comments(nf.read_text(encoding="utf-8", errors="replace"))
            for m in _PROCESS.finditer(text):
                body = _block(text, m.end() - 1)
                head = body.split("script:")[0].split("shell:")[0].split("exec:")[0]
                self.processes[m.group(1)] = (str(nf.relative_to(self.repo)).replace("\\", "/"),
                                              _LABEL.findall(head))
        configs = _files(self.repo, (".config",))
        main = self.repo / "nextflow.config"
        configs.sort(key=lambda p: (p != main, str(p)))   # main first, includes after (approximate order)
        for cfg in configs:
            rel = str(cfg.relative_to(self.repo)).replace("\\", "/")
            text = _strip_comments(cfg.read_text(encoding="utf-8", errors="replace"))
            for m in _SELECTOR.finditer(text):
                pattern = next(g for g in m.groups()[1:] if g)
                body = _block(text, m.end() - 1)
                settings = [(k, v) for k, v in _SETTING.findall(body)] or _INLINE.findall(body)
                if settings:
                    self.selectors.append((rel, m.group(1), pattern.strip(), settings))
            self._caps(text, rel)

    def _caps(self, text: str, rel: str):
        rl = re.search(r"resourceLimits\s*=\s*\[([^\]]*)\]", text)
        if rl:
            mem = re.search(r"memory\s*:\s*['\"]?([\d.]+)\s*\.?\s*(GB|TB|MB)", rl.group(1), re.I)
            tim = re.search(r"time\s*:\s*['\"]?([\d.]+)\s*\.?\s*(h|d)", rl.group(1), re.I)
            if mem:
                self.memory_cap = (parse_memory(f"{mem.group(1)} {mem.group(2)}"), f"resourceLimits in {rel}")
            if tim:
                self.time_cap = (parse_duration(f"{tim.group(1)}{tim.group(2).lower()}"), f"resourceLimits in {rel}")
        mm = re.search(r"max_memory\s*=\s*['\"]([\d.]+)\s*\.?\s*(GB|TB|MB)['\"]", text, re.I)
        if mm and not self.memory_cap:
            self.memory_cap = (parse_memory(f"{mm.group(1)} {mm.group(2)}"), f"params.max_memory in {rel}")
        mt = re.search(r"max_time\s*=\s*['\"]([\d.]+)\s*\.?\s*(h|d)['\"]", text, re.I)
        if mt and not self.time_cap:
            self.time_cap = (parse_duration(f"{mt.group(1)}{mt.group(2).lower()}"), f"params.max_time in {rel}")

    # ---- queries ----
    @staticmethod
    def _selector_matches(kind: str, pattern: str, name: str, labels: list[str]) -> bool:
        if kind == "withLabel":
            return any(re.fullmatch(p, l) for p in pattern.split("|") for l in labels) if labels else False
        neg = pattern.startswith("!")
        pat = pattern[1:] if neg else pattern
        try:
            hit = bool(re.fullmatch(pat, name) or re.fullmatch(pat, f"X:{name}"))
        except re.error:
            hit = pat == name
        return hit != neg

    def trace(self, name: str) -> ProcessTrace:
        file, labels = self.processes.get(name, ("", []))
        t = ProcessTrace(name, file, labels)
        for rel, kind, pattern, settings in self.selectors:
            if self._selector_matches(kind, pattern, name, labels):
                t.settings += [Setting(rel, f"{kind}:{pattern}", k, v) for k, v in settings]
        if self.memory_cap:
            t.memory_cap_bytes, t.cap_source = self.memory_cap
        if self.time_cap:
            t.time_cap_ms = self.time_cap[0]
            t.cap_source = t.cap_source or self.time_cap[1]
        return t

    def match_job_name(self, job_name: str) -> str | None:
        """Recover the process from an AWS Batch job name like nf-NFCORE_RNASEQ_RNASEQ_ALIGN_STAR_STAR_ALIGN_s1.
        Nextflow replaces ':' and other characters with '_', so pick the longest known process name that appears
        as a whole '_'-delimited token run. Returns None when nothing (or more than one equally long name) fits."""
        clean = "_" + re.sub(r"[^A-Za-z0-9]", "_", job_name) + "_"
        hits = [p for p in self.processes if f"_{p}_" in clean]
        if not hits:
            return None
        best = max(len(h) for h in hits)
        top = [h for h in hits if len(h) == best]
        return top[0] if len(top) == 1 else None
