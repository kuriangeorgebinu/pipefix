"""Human-readable output: terminal summary and the pull/merge request description."""
from __future__ import annotations

from .models import Diagnosis
from .patch import Patch


def _row(d: Diagnosis) -> str:
    name = d.failure.simple_name + (f" ({d.failure.tag})" if d.failure.tag else "")
    return f"| `{name}` | {d.cls.label} | {d.confidence:.0%} | {'config fix' if d.cls.auto_fixable else 'needs a person'} |"


def summary(diagnoses: list[Diagnosis]) -> str:
    if not diagnoses:
        return "No failed tasks found."
    out = []
    for d in diagnoses:
        name = d.failure.simple_name + (f" ({d.failure.tag})" if d.failure.tag else "")
        out.append(f"* {name}: {d.cls.label} ({d.confidence:.0%})")
        out += [f"    - {e}" for e in d.evidence[:4]]
        out.append(f"    -> {d.recommendation}")
    return "\n".join(out)


def pr_title(patch: Patch) -> str:
    names = sorted({f.name for f in patch.fixes})
    head = ", ".join(names[:3]) + (f" +{len(names) - 3}" if len(names) > 3 else "")
    return f"pipefix: fix failed pipeline tasks ({head})"


def pr_body(diagnoses: list[Diagnosis], patch: Patch, run_ref: str = "") -> str:
    lines = ["## What failed", ""]
    if run_ref:
        lines += [f"Run: `{run_ref}`", ""]
    lines += ["| Process | Diagnosis | Confidence | Action |", "|---|---|---|---|"]
    lines += [_row(d) for d in diagnoses]
    lines += ["", "## Evidence", ""]
    for d in diagnoses:
        name = d.failure.simple_name + (f" ({d.failure.tag})" if d.failure.tag else "")
        lines.append(f"**{name}**: {d.explanation}")
        lines += [f"- {e}" for e in d.evidence[:5]]
        lines.append("")
    if patch.fixes:
        lines += ["## The fix", "",
                  "Only the processes that failed are changed, in `conf/pipefix.config` (included last in "
                  "`nextflow.config`, so it overrides the pipeline defaults). Resources grow with each retry, so a "
                  "second failure escalates automatically instead of stopping the run.", ""]
    if patch.advisories:
        lines += ["## Needs a person", "",
                  "These failures are not safely fixable by changing resources, so no config change was made:", ""]
        for d in patch.advisories:
            lines.append(f"- **{d.failure.simple_name}** ({d.cls.label}): {d.recommendation}")
        lines.append("")
    lines += ["## Before merging", "",
              "- [ ] Re-run with `-resume` so only the failed tasks run again",
              "- [ ] Check the new limits fit your compute environment (largest instance type / queue)", ""]
    lines += ["---", "Opened by [pipefix](https://github.com/kuriangeorgebinu/pipefix). Diagnoses are rule-based "
              "and cite the exact log lines and numbers they used."]
    return "\n".join(lines)
