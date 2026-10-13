"""Human-readable output: terminal summary, the investigation report and the pull/merge request description."""
from __future__ import annotations

import re
from collections import Counter

from .investigation import Investigation
from .models import Diagnosis
from .patch import Patch

_ACCOUNT = re.compile(r"(?<!\d)\d{12}(?!\d)")


def redact(text: str) -> str:
    """Hide AWS account IDs (also inside ARNs and ECR registry hosts) before text leaves the machine."""
    return _ACCOUNT.sub("<account-id>", text or "")


def _md(text: str) -> str:
    return redact(text).replace("|", "\\|")


def summary(diagnoses: list[Diagnosis]) -> str:
    if not diagnoses:
        return "No failed tasks found."
    out = []
    for d in diagnoses:
        out.append(f"* {d.failure.display}: {d.cls.label} ({d.confidence:.0%})")
        out += [f"    - {e}" for e in d.evidence[:4]]
        out.append(f"    -> {d.recommendation}")
    return "\n".join(out)


def pr_title(patch: Patch) -> str:
    names = sorted({f.name for f in patch.fixes})
    head = ", ".join(names[:3]) + (f" +{len(names) - 3}" if len(names) > 3 else "")
    return f"pipefix: fix failed pipeline tasks ({head})"


def _trace_rows(patch: Patch) -> list[str]:
    rows = []
    for fix in patch.fixes:
        tr = patch.traces.get(fix.name)
        if not tr:
            continue
        rows.append(f"**{fix.name}**" + (f" (defined in `{tr.defined_in}`" + (f", labels {', '.join(tr.labels)}"
                                        if tr.labels else "") + ")" if tr.defined_in else ""))
        rows.append("")
        rows.append("| Setting | Value now | Set in | By |")
        rows.append("|---|---|---|---|")
        for directive in ("memory", "time", "errorStrategy", "maxRetries", "queue"):
            s = tr.effective(directive)
            if s:
                rows.append(f"| {directive} | `{_md(s.value.strip())[:90]}` | `{s.where}` | {_md(s.selector)} |")
        rows.append("")
    return rows


def body(diagnoses: list[Diagnosis], patch: Patch, inv: Investigation | None = None) -> str:
    """The MR description (also used as the saved investigation report)."""
    lines = ["## What failed", ""]
    if inv:
        lines += [f"Run: {_md(inv.run_ref)} ({inv.platform})" + (f", Nextflow {inv.nextflow_version}"
                                                                if inv.nextflow_version else ""), ""]
    lines += ["| Process | Diagnosis | Confidence | Action |", "|---|---|---|---|"]
    for d in diagnoses:
        action = "config fix" if any(fx.name == d.failure.simple_name for fx in patch.fixes) else "needs a person"
        lines.append(f"| `{_md(d.failure.display)}` | {d.cls.label} | {d.confidence:.0%} | {action} |")
    lines += ["", "## Evidence", ""]
    for d in diagnoses:
        f = d.failure
        meta = [x for x in (f"exit {f.exit_code}" if f.exit_code is not None else "",
                            f"job {f.job_ref}" if f.job_ref else "", f"image `{f.image}`" if f.image else "",
                            f"work dir `{f.work_dir}`" if f.work_dir else "") if x]
        lines.append(f"**{_md(f.display)}**: {_md(d.explanation)}" + (f" ({_md(', '.join(meta))})" if meta else ""))
        lines += [f"- {_md(e)}" for e in d.evidence[:5]]
        lines.append("")
    trace = _trace_rows(patch)
    if trace:
        lines += ["## Where the current settings come from", "",
                  "Traced through `nextflow.config` and its `includeConfig` files, with Nextflow's precedence "
                  "(withName > withLabel > process script > generic `process {}`).", ""] + trace
    found = (inv.findings if inv else []) + patch.findings
    if found:
        lines += ["## Platform and config findings", ""]
        icon = {"block": "**Blocker**", "warn": "**Warning**", "info": "Info"}
        for fd in sorted(found, key=lambda x: {"block": 0, "warn": 1}.get(x.severity, 2)):
            lines.append(f"- {icon.get(fd.severity, 'Info')} · {_md(fd.title)}: {_md(fd.detail)}")
        lines.append("")
    if inv and inv.denials:
        lines += ["## Access denied in CloudTrail during the run", "",
                  "| Time | Action | Resource | Error | Principal |", "|---|---|---|---|---|"]
        for dn in inv.denials[:10]:
            lines.append(f"| {dn['time']} | {dn['source'].split('.')[0]}:{dn['action']} | `{_md(dn['target'])}` | "
                         f"{dn['code']} | `{_md(dn['principal'].split('/')[-2] if '/' in dn['principal'] else dn['principal'])}` |")
        lines.append("")
    if patch.fixes:
        lines += ["## The fix", "",
                  "Only the processes that failed are changed, in `conf/pipefix.config` (included last in "
                  "`nextflow.config`, so it overrides the pipeline defaults). Resources grow with each retry, so a "
                  "second failure escalates instead of stopping the run.", ""]
        for fx in patch.fixes:
            lines.append(f"- **{fx.name}**: " + "; ".join(_md(n) for n in fx.notes))
        lines.append("")
    if patch.advisories:
        lines += ["## Needs a person", "", "Not safely fixable by pipeline config, so nothing was changed for these:", ""]
        for d in patch.advisories:
            lines.append(f"- **{_md(d.failure.display)}** ({d.cls.label}): {_md(d.recommendation)}")
        lines.append("")
    lines += ["## Before merging", "",
              "- [ ] Re-run with `-resume` so only the failed tasks run again",
              "- [ ] Check the new limits fit your compute environment and budget", ""]
    if inv and inv.calls:
        counts = Counter(c.split(":")[0] for c in inv.calls)
        lines += ["<details><summary>What pipefix read (read-only AWS calls)</summary>", "",
                  ", ".join(f"{svc}: {n}" for svc, n in sorted(counts.items())) + "."]
        if inv.errors:
            lines += ["", "Not readable or not found: " + _md(", ".join(sorted(set(inv.errors))[:15])) + "."]
        lines += ["", "</details>", ""]
    lines += ["---", "Opened by [pipefix](https://github.com/kuriangeorgebinu/pipefix). Diagnoses are rule-based "
              "and cite the exact log lines and numbers they used. AWS account IDs are redacted."]
    return "\n".join(lines)


# kept for the older call sites / tests
def pr_body(diagnoses: list[Diagnosis], patch: Patch, run_ref: str = "") -> str:
    inv = Investigation("files", run_ref) if run_ref else None
    return body(diagnoses, patch, inv)
