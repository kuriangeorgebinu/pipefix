"""Human-readable output: terminal summary, the investigation report and the pull/merge request description."""
from __future__ import annotations

import re
from collections import Counter

from .investigation import Investigation
from .models import Diagnosis
from .patch import Patch

_ACCOUNT = re.compile(r"(?<!\d)\d{12}(?!\d)")
_SECRET = re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|glpat-[A-Za-z0-9_-]{20,})")


def redact(text: str) -> str:
    """Hide AWS account IDs (also inside ARNs and ECR registry hosts) and GitHub/GitLab tokens before text
    leaves the machine."""
    from .keys import known_keys
    from .vcs import known_tokens
    text = _SECRET.sub("<token>", text or "")
    for t in known_tokens() + known_keys():
        text = text.replace(t, "<token>")
    return _ACCOUNT.sub("<account-id>", text)


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
    kind = " agent" if getattr(patch, "agent", None) else " (AI-researched)" if patch.ai_used else ""
    return f"pipefix{kind}: fix failed pipeline tasks ({head})"


def _ev(e: dict) -> str:
    if e.get("policy"):
        return f"policy ({_md(e['policy'])})"
    where = f"call #{e['call']} `{e.get('tool', '')}`" if e.get("call") is not None else f"source [{e.get('source')}]"
    return f"{where}: \"{_md(e.get('quote') or e.get('excerpt') or '')[:300]}\""


def _agent_section(patch: Patch) -> list[str]:
    res = getattr(patch, "agent", None)
    if not res:
        return []
    from .agent import AGENTS, PHASE_TITLE, PHASES
    aws = [s for s in res.steps if s.ok and s.tool == "aws_call"]
    rounds = res.review.get("rounds", [])
    lines = ["## How the agents worked", "",
             f"Three pipefix agents (`{_md(res.model)}`) worked on this. The primary agent chose and made its own "
             f"read-only AWS calls ({len(aws)} in total across agents, found from the AWS API reference), read the "
             "pipeline repo, researched the docs, compared candidate solutions and proposed one. Two independent "
             f"experts ({', '.join(v for k, v in AGENTS.items() if k != 'primary')}) fact-checked it with their own "
             f"calls and reading" + (f" over {len(rounds)} round(s) until both agreed" if res.review.get("agreed") else "")
             + ". pipefix enforced the steps, the read-only access, the verbatim evidence checks and the config checks"
             + (f" (it refused {res.refused} call(s) that broke a rule)" if res.refused else "") + ".", ""]
    d = res.diagnosis
    if d:
        lines += [f"**1. Debug.** {_md(d.get('root_cause', ''))}", ""]
        lines += [f"- {_ev(e)}" for e in d.get("evidence", [])]
        lines.append("")
    if res.candidates:
        lines += ["**2. Research: candidate solutions**", "", "| | Candidate | Settings | Checks |", "|---|---|---|---|"]
        for c in res.candidates:
            mark = "**chosen**" if c["id"] == res.decision.get("chosen") else c["id"]
            sets = "; ".join(f"`{k} = {_md(v)}`" + (f" = `{_md(c['derivations'][k]['formula'])}`"
                                                    if c.get("derivations", {}).get(k) else "")
                             for k, v in c["settings"].items()) or "no config change"
            lines.append(f"| {mark} | {_md(c['title'])} | {sets} | "
                         f"{'passed' if c['valid'] else _md('failed: ' + '; '.join(c['problems'])[:200])} |")
        lines.append("")
        chosen = next((c for c in res.candidates if c["id"] == res.decision.get("chosen")), None)
        for k, dv in (chosen or {}).get("derivations", {}).items():
            lines.append(f"- `{k}` = `{_md(dv['formula'])}` with "
                         + "; ".join(f"{n} = {i['value']:g} from {_ev(i)}" for n, i in dv["inputs"].items()))
        lines.append("")
    if res.decision:
        lines += [f"**3. Decision.** {_md(res.decision['reasons'])}", ""]
        lines += [f"- Not {_md(k)}: {_md(v)}" for k, v in res.decision.get("why_not", {}).items()]
        if res.decision.get("risks"):
            lines.append(f"- Risks: {_md(res.decision['risks'])}")
        lines.append("")
    if rounds:
        lines += ["**3b. Expert review.** " + ("Both experts agreed." if res.review.get("agreed") else
                                               "The experts did not agree."), ""]
        for r in rounds:
            for who, v in r["reviews"].items():
                lines.append(f"- Round {r['round']}, {_md(v['title'])}: **{v['verdict']}**")
                for ch in v["checks"]:
                    lines.append(f"  - {ch['finding']}: {_md(ch['claim'])[:200]} — {_md(ch['explanation'])[:300]}"
                                 + "".join(f" ({_ev(e)})" for e in ch["evidence"][:2]))
            if r.get("response"):
                lines.append(f"- Primary {r['response']['action']}d: "
                             + "; ".join(_md(a['answer'])[:200] for a in r["response"]["answers"][:4]))
        lines.append("")
    if res.implement_summary:
        lines += [f"**4. Implementation.** {_md(res.implement_summary)}", ""]
    lines += ["<details><summary>Every step the agents took</summary>", ""]
    for p in PHASES:
        steps = [s for s in res.steps if s.phase == p]
        if steps:
            lines.append(f"*{PHASE_TITLE[p]}*")
            lines += [f"- #{s.n} [{AGENTS.get(s.agent, s.agent)}] {'' if s.ok else '(refused) '}{_md(s.summary)}"
                      + (f" — *{_md(s.reason)}*" if s.reason else "") for s in steps]
    return lines + ["", "</details>", ""]


def _ai_section(patch: Patch) -> list[str]:
    ai = patch.ai
    if not ai:
        return []
    lines = ["## AI research", ""]
    if ai.error:
        return lines + [f"Not used: {_md(ai.error)}", ""]
    used = ", ".join(f"`{n}`" for n in patch.ai_used) or "none"
    read = sum(1 for s in ai.sources if s.get("read"))
    lines += [f"pipefix searched AWS, Nextflow/nf-core and Terraform sources, read {read} of the "
              f"{len(ai.sources)} pages, and asked `{_md(ai.model)}` to propose the fix from them. Strict rule: "
              "every value must be backed by a verbatim quote from a page pipefix read, and pipefix verified each "
              "quote against that page before writing anything (plus: accepted forms only, failed processes "
              f"only, within the platform's limits). AI-proposed fix used for: {used}. "
              "**Review these values before merging.**", ""]
    if ai.summary:
        lines += [f"> {_md(ai.summary)}", ""]
    for fx in ai.fixes:
        lines.append(f"- **{_md(fx.name)}**: " + "; ".join(f"`{k} = {_md(v)}`" for k, v in fx.settings.items())
                     + (f". {_md(fx.rationale)}" if fx.rationale else ""))
        for e in fx.evidence:
            lines.append(f"  - `{_md(e['setting'])}`: source [{e['n']}] says: \"{_md(e['quote'])}\"")
    if ai.infrastructure:
        lines += ["", "**Infrastructure advice (not applied)**", ""]
        for it in ai.infrastructure:
            lines.append(f"- {_md(it['title'])}: {_md(it['detail'])}")
            lines += [f"  - source [{e['n']}] says: \"{_md(e['quote'])}\"" for e in it.get("evidence", [])]
    if ai.rejected:
        lines += ["", "**Rejected by pipefix's checks (not written)**", ""] + [f"- {_md(r)}" for r in ai.rejected]
    if ai.sources:
        lines += ["", "**Sources**", ""] + [f"{s['n']}. [{_md(s['title']) or s['url']}]({s['url']})"
                                             + ("" if s.get("read") else " (could not be read, so not citable)")
                                             for s in ai.sources]
    return lines + [""]


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
    lines += _agent_section(patch)
    lines += _ai_section(patch)
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
              "and cite the exact log lines and numbers they used"
              + ("; the fix values marked AI were researched by an LLM from the cited sources and checked by "
                 "pipefix" if patch.ai_used else "") + ". AWS account IDs are redacted."]
    return "\n".join(lines)


# kept for the older call sites / tests
def pr_body(diagnoses: list[Diagnosis], patch: Patch, run_ref: str = "") -> str:
    inv = Investigation("files", run_ref) if run_ref else None
    return body(diagnoses, patch, inv)
