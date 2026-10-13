"""Trace where a process's resources come from in a Nextflow pipeline repo.

Follows nextflow.config and its includeConfig chain, reads every `process` scope setting (generic, withLabel,
withName, dotted `process.x = ...`), the process definitions in *.nf (labels and directives), and resource caps
(`resourceLimits`, nf-core `params.max_*`). Applies Nextflow's precedence, lowest to highest:
    generic process config < directive in the process script < withLabel < withName
(later files and later lines win within a level). Settings inside `profiles { }` are recorded but only count
when that profile is requested.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

DIRECTIVES = ("memory", "time", "cpus", "errorStrategy", "maxRetries", "queue", "disk", "resourceLimits")
LEVEL = {"generic": 0, "script": 1, "withLabel": 2, "withName": 3}


@dataclass
class Setting:
    file: str
    line: int
    level: str                 # generic | script | withLabel | withName
    selector: str              # e.g. "withLabel: 'process_high'"
    directive: str
    value: str
    profile: str = ""
    order: int = 0

    @property
    def where(self) -> str:
        return f"{self.file}:{self.line}"


@dataclass
class ProcessTrace:
    name: str
    defined_in: str = ""
    labels: list[str] = field(default_factory=list)
    settings: list[Setting] = field(default_factory=list)   # applicable settings, lowest precedence first

    def effective(self, directive: str) -> Setting | None:
        cands = [s for s in self.settings if s.directive == directive]
        return cands[-1] if cands else None


@dataclass
class RepoConfig:
    root: Path
    files: list[str] = field(default_factory=list)          # config files read, in include order
    settings: list[Setting] = field(default_factory=list)
    processes: dict[str, tuple[str, list[str], list[Setting]]] = field(default_factory=dict)
    caps: dict[str, str] = field(default_factory=dict)       # memory/time/cpus caps (raw values)
    cap_sources: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    params: dict[str, str] = field(default_factory=dict)

    # ----- queries -----
    def trace(self, process: str, profiles: tuple[str, ...] = ()) -> ProcessTrace:
        simple = process.split(":")[-1]
        defined_in, labels, script_settings = self.processes.get(simple, ("", [], []))
        applicable = []
        for s in self.settings + script_settings:
            if s.profile and s.profile not in profiles:
                continue
            if s.level == "withName" and not _matches(_selector_value(s.selector), [simple, process]):
                continue
            if s.level == "withLabel" and not any(_matches(_selector_value(s.selector), [lb]) for lb in labels):
                continue
            if s.level == "script" and s.selector != simple:
                continue
            applicable.append(s)
        applicable.sort(key=lambda s: (LEVEL[s.level], s.order))
        return ProcessTrace(simple, defined_in, labels, applicable)

    def resolve_process(self, job_name: str) -> tuple[str, str] | None:
        """Batch job names lose ':' and '()'; find the longest known process name inside one -> (name, tag)."""
        norm = re.sub(r"[^A-Za-z0-9_-]", "", job_name.replace(" ", "_"))
        best = max((p for p in self.processes if p in norm), key=len, default=None)
        if not best:
            return None
        rest = norm.split(best, 1)[1].lstrip("_")
        return best, rest


def _selector_value(selector: str) -> str:
    return selector.split(":", 1)[1].strip().strip("'\"")


def _matches(pattern: str, names: list[str]) -> bool:
    neg = pattern.startswith("!")
    pat = pattern[1:] if neg else pattern
    try:
        hit = any(re.fullmatch(pat, n) for n in names if n)
    except re.error:
        hit = any(pat == n for n in names)
    return hit != neg


# ---------- a small, forgiving parser for Nextflow config files ----------

def _strip_comments(text: str) -> str:
    """Blank out // and /* */ comments (keeping offsets and newlines), respecting quotes."""
    out, i, n, q = list(text), 0, len(text), ""
    while i < n:
        c = text[i]
        if q:
            if c == "\\":
                i += 2
                continue
            if c == q:
                q = ""
        elif c in "'\"":
            q = c
        elif text.startswith("//", i) and not (i > 0 and text[i - 1] == ":"):  # keep URLs like https://
            j = text.find("\n", i)
            j = n if j < 0 else j
            out[i:j] = " " * (j - i)
            i = j
            continue
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out[i:j] = [ch if ch == "\n" else " " for ch in text[i:j]]
            i = j
            continue
        i += 1
    return "".join(out)


def _read_value(t: str, i: int) -> int:
    """End index of a value starting at i: stops at newline/';' when not inside (), [], {} or quotes."""
    depth, q, n = 0, "", len(t)
    while i < n:
        c = t[i]
        if q:
            if c == "\\":
                i += 2
                continue
            if c == q:
                q = ""
        elif c in "'\"":
            q = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            if depth == 0:
                return i
            depth -= 1
        elif c in "\n;" and depth == 0:
            # a trailing operator means the expression continues on the next line
            if c == "\n" and t[:i].rstrip().endswith(("+", "*", "?", ":", "&&", "||", ",")):
                i += 1
                continue
            return i
        i += 1
    return n


_SEL = re.compile(r"(withLabel|withName)\s*:\s*(?:'([^']*)'|\"([^\"]*)\"|([^\s{]+))\s*\{")
_SCOPE = re.compile(r"([A-Za-z_][\w.]*)\s*\{")
_ASSIGN = re.compile(r"([A-Za-z_][\w.]*)\s*=\s*")
_INCLUDE = re.compile(r"includeConfig\s+(?:'([^']+)'|\"([^\"]+)\")")


class _Parser:
    def __init__(self, cfg: RepoConfig):
        self.cfg = cfg
        self.order = 0

    def parse_file(self, path: Path, depth: int = 0, profile: str = "") -> None:
        if depth > 8 or not path.is_file():
            return
        rel = str(path.relative_to(self.cfg.root)) if path.is_relative_to(self.cfg.root) else str(path)
        if rel in self.cfg.files:
            return
        self.cfg.files.append(rel)
        raw = path.read_text(encoding="utf-8", errors="replace")
        t = _strip_comments(raw)
        self._block(t, 0, [], rel, path.parent, depth, profile)

    def _line(self, t: str, i: int) -> int:
        return t.count("\n", 0, i) + 1

    def _block(self, t: str, i: int, scope: list[str], rel: str, base: Path, depth: int, profile: str) -> int:
        n = len(t)
        while i < n:
            while i < n and t[i] in " \t\r\n;":
                i += 1
            if i >= n:
                return n
            if t[i] == "}":
                return i + 1
            if m := _INCLUDE.match(t, i):
                target = (m.group(1) or m.group(2))
                self._include(target, scope, base, depth, profile)
                i = m.end()
                continue
            if m := _SEL.match(t, i):
                kind, value = m.group(1), m.group(2) or m.group(3) or m.group(4)
                i = self._block(t, m.end(), scope + [f"{kind}: '{value}'"], rel, base, depth, profile)
                continue
            if (m := _SCOPE.match(t, i)) and not _ASSIGN.match(t, i):
                name = m.group(1)
                prof = profile
                if scope and scope[-1] == "profiles":
                    prof = name
                i = self._block(t, m.end(), scope + name.split("."), rel, base, depth, prof)
                continue
            if m := _ASSIGN.match(t, i):
                end = _read_value(t, m.end())
                self._record(scope + m.group(1).split("."), t[m.end():end].strip(), rel, self._line(t, i), profile)
                i = end
                continue
            # anything else (def, if, println...): skip the statement, balancing brackets
            i = max(_read_value(t, i), i + 1)
        return n

    def _include(self, target: str, scope: list[str], base: Path, depth: int, profile: str) -> None:
        target = re.sub(r"^\$\{?(projectDir|baseDir|launchDir)\}?/", "", target)
        if "${" in target or target.startswith(("http://", "https://", "/dev/null")):
            self.cfg.notes.append(f"Skipped dynamic or remote includeConfig: {target}")
            return
        p = (base / target) if not Path(target).is_absolute() else Path(target)
        if not p.is_file():
            p = self.cfg.root / target
        prof = profile or (scope[1] if len(scope) >= 2 and scope[0] == "profiles" else "")
        self.parse_file(p.resolve(), depth + 1, prof)

    def _record(self, keys: list[str], value: str, rel: str, line: int, profile: str) -> None:
        keys = [k for k in keys if k != "profiles" and k != profile] if profile else keys
        if keys and keys[0] == "params" and len(keys) == 2:
            self.cfg.params[keys[1]] = value
            if keys[1] in ("max_memory", "max_time", "max_cpus") and not profile:
                kind = keys[1][4:]
                self.cfg.caps.setdefault(kind, value)
                self.cfg.cap_sources.setdefault(kind, f"{rel}:{line} params.{keys[1]}")
            return
        if not keys or keys[0] != "process":
            return
        directive = keys[-1]
        sel = next((k for k in reversed(keys[1:-1]) if k.startswith(("withName", "withLabel"))), "")
        if directive == "resourceLimits" and not profile:
            for k, v in re.findall(r"(memory|time|cpus)\s*:\s*([^,\]]+)", value):
                self.cfg.caps[k] = v.strip()
                self.cfg.cap_sources[k] = f"{rel}:{line} resourceLimits"
            return
        if directive not in DIRECTIVES:
            return
        level = "withName" if sel.startswith("withName") else "withLabel" if sel.startswith("withLabel") else "generic"
        self.order += 1
        self.cfg.settings.append(Setting(rel, line, level, sel or "process", directive, value, profile, self.order))


_PROC = re.compile(r"^\s*process\s+([A-Za-z_]\w*)\s*\{", re.M)
_LABEL = re.compile(r"^\s*label\s+['\"]([^'\"]+)['\"]", re.M)
_DIRECTIVE = re.compile(r"^\s*(memory|time|cpus|errorStrategy|maxRetries|queue|disk)\s+(.+?)\s*$", re.M)


def _scan_processes(cfg: RepoConfig) -> None:
    for nf in sorted(cfg.root.rglob("*.nf")):
        if any(part in (".git", "work", ".nextflow", "node_modules") for part in nf.parts):
            continue
        text = _strip_comments(nf.read_text(encoding="utf-8", errors="replace"))
        for m in _PROC.finditer(text):
            name = m.group(1)
            # the directive section: from the opening brace to the first input:/output:/script:/exec:/shell:
            body_end = re.search(r"^\s*(input|output|script|shell|exec|when)\s*:", text[m.end():], re.M)
            head = text[m.end(): m.end() + (body_end.start() if body_end else 3000)]
            rel = str(nf.relative_to(cfg.root))
            line = text.count("\n", 0, m.start()) + 1
            labels = _LABEL.findall(head)
            settings = []
            for d in _DIRECTIVE.finditer(head):
                settings.append(Setting(rel, line + head.count("\n", 0, d.start()), "script", name, d.group(1),
                                        d.group(2), order=0))
            cfg.processes.setdefault(name, (f"{rel}:{line}", labels, settings))


def load_repo(root: str | Path) -> RepoConfig:
    root = Path(root).resolve()
    cfg = RepoConfig(root)
    _Parser(cfg).parse_file(root / "nextflow.config")
    _scan_processes(cfg)
    return cfg


# ---------- reading Nextflow resource literals ----------

_MEM_LIT = re.compile(r"(\d+(?:\.\d+)?)\s*\.?\s*(KB|MB|GB|TB)\b", re.I)
_UNIT = {"KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}


def memory_literal_bytes(value: str) -> int | None:
    """'128.GB', "'72 GB'", '{ 36.GB * task.attempt }' -> bytes of the first-attempt value."""
    m = _MEM_LIT.search(value or "")
    return int(float(m.group(1)) * _UNIT[m.group(2).upper()]) if m else None


def grows_with_attempt(value: str) -> bool:
    return "task.attempt" in (value or "")
