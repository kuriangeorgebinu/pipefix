"""pipefix command line.

Investigate a failed run straight from AWS (read-only), trace the repo's config, and open the fix:
  pipefix investigate --batch-queue my-queue --since 12h --repo .               # find failed jobs, show the diff
  pipefix investigate --batch-job <id> [<id> ...] --repo . --open-pr            # open a PR/MR
  pipefix investigate --omics-run 1234567 --repo . --report report.md
From files you already have (no AWS access needed):
  pipefix diagnose --trace trace.txt --log .nextflow.log
  pipefix fix --trace trace.txt --log .nextflow.log --repo . [--apply | --open-pr]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from . import __version__
from .classify import classify_all
from .configtrace import load_repo
from .investigation import Investigation
from .models import Diagnosis, TaskFailure
from .patch import build_patch
from .report import body, pr_title, redact, summary
from .sources import aws as aws_files
from .sources import nextflow


def _hours(text: str) -> float:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([hd]?)", text.strip())
    if not m:
        raise argparse.ArgumentTypeError("use e.g. 12h or 2d")
    return float(m.group(1)) * (24 if m.group(2) == "d" else 1)


def collect_files(a: argparse.Namespace) -> list[TaskFailure]:
    trace = nextflow.parse_trace(a.trace) if a.trace else []
    log = nextflow.parse_log(a.log) if a.log else []
    failures = nextflow.merge(trace, log)
    for f in failures:
        if f.work_dir and Path(f.work_dir).is_dir():
            f.error_lines += nextflow.read_task_files(f.work_dir)
    if a.batch_json:
        failures += aws_files.batch_failures_from_json(aws_files.load_json(a.batch_json))
    if a.omics_json:
        doc = aws_files.load_json(a.omics_json)
        failures += aws_files.omics_failures_from_json(doc.get("run", doc), doc.get("tasks", []))
    return failures


def _deliver(a: argparse.Namespace, diagnoses: list[Diagnosis], inv: Investigation | None) -> int:
    """Shared tail of `fix` and `investigate`: build the patch, show it, and apply / open the MR."""
    repo = Path(a.repo)
    cfg = load_repo(repo)
    patch = build_patch(diagnoses, repo, a.min_confidence, inv, cfg, tuple(a.nf_profile or []))
    text = body(diagnoses, patch, inv)
    if a.report:
        Path(a.report).write_text(text, encoding="utf-8")
        print(f"Report written to {a.report}")
    for fd in (inv.findings if inv else []) + patch.findings:
        if fd.severity in ("warn", "block"):
            print(f"[{fd.severity}] {fd.title}: {redact(fd.detail)}")
    if patch.empty:
        print("\nNo pipeline config change is safe to make automatically." if diagnoses else "\nNothing to fix.")
        return 0
    diff = patch.diff()
    print("\n" + diff)
    if not (a.apply or a.open_pr):
        print("Dry run. Add --apply to write these changes, or --open-pr to open a PR/MR.")
        return 0
    if a.open_pr:
        from . import vcs
        remote = vcs.remote_info(repo, a.remote, a.provider)          # fails before anything is written
        h = vcs.diff_hash(diff)
        if existing := vcs.find_existing(remote, h):
            print(f"An open pipefix PR/MR already proposes this change: {existing}")
            return 0
        if vcs.git(repo, "status", "--porcelain", "--", *patch.files.keys()):
            raise SystemExit("The files pipefix changes have uncommitted edits; commit or stash them first.")
    written = patch.apply(repo)
    print("Wrote: " + ", ".join(written))
    if a.open_pr:
        title = pr_title(patch)
        branch, base = vcs.commit_branch(repo, written, title, text, h)
        try:
            url = vcs.open_request(repo, remote, branch, base, title, text, draft=not a.ready)
        finally:
            vcs.git(repo, "checkout", base)
        print(f"Opened: {url}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pipefix", description="Diagnose failed Nextflow runs and open the fix.")
    p.add_argument("--version", action="version", version=f"pipefix {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def outputs(s, fix: bool):
        s.add_argument("--json", action="store_true", help="machine-readable diagnoses")
        if fix:
            s.add_argument("--repo", required=True, help="the pipeline's git repo")
            s.add_argument("--apply", action="store_true", help="write the changes")
            s.add_argument("--open-pr", action="store_true", help="branch, commit, push and open a PR/MR")
            s.add_argument("--ready", action="store_true", help="open a ready-for-review PR instead of a draft")
            s.add_argument("--remote", default="origin")
            s.add_argument("--provider", choices=["github", "gitlab"], help="override auto-detection")
            s.add_argument("--min-confidence", type=float, default=0.6)
            s.add_argument("--nf-profile", action="append", help="Nextflow profile(s) the run used")
            s.add_argument("--report", help="also save the full investigation report (markdown)")

    for name in ("diagnose", "fix"):
        s = sub.add_parser(name, help="from files you already have")
        s.add_argument("--trace", help="Nextflow trace.txt")
        s.add_argument("--log", help=".nextflow.log or saved console output")
        s.add_argument("--batch-json", help="output of `aws batch describe-jobs`")
        s.add_argument("--omics-json", help='{"run": get-run output, "tasks": [get-run-task outputs]}')
        outputs(s, name == "fix")

    s = sub.add_parser("investigate", help="read the failed run from AWS (read-only) and propose the fix")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--batch-job", nargs="+", help="AWS Batch job id(s)")
    g.add_argument("--batch-queue", help="find FAILED jobs in this AWS Batch queue")
    g.add_argument("--omics-run", help="AWS HealthOmics run id")
    s.add_argument("--since", type=_hours, default=24.0, help="with --batch-queue: look back this far (12h, 2d)")
    s.add_argument("--name-prefix", default="", help="with --batch-queue: only jobs whose name starts with this")
    s.add_argument("--profile", help="AWS profile (default: your default credentials)")
    s.add_argument("--region")
    s.add_argument("--no-cloudtrail", action="store_true", help="skip the CloudTrail access-denied search")
    s.add_argument("--diagnose-only", action="store_true", help="print the diagnosis; don't look at a repo")
    outputs(s, True)
    s._option_string_actions["--repo"].required = False

    s = sub.add_parser("ui", help="open the pipefix console in your browser")
    s.add_argument("--port", type=int, default=8787)
    s.add_argument("--no-browser", action="store_true")

    a = p.parse_args(argv)
    if a.cmd == "ui":
        from .ui.server import serve
        serve(a.port, not a.no_browser)
        return 0
    inv = None
    if a.cmd == "investigate":
        from .aws.session import AwsUnavailable, boto3_clients
        from .investigate import investigate
        try:
            clients = boto3_clients(a.profile, a.region)
            inv, diagnoses = investigate(clients, batch_jobs=a.batch_job, batch_queue=a.batch_queue,
                                         since_hours=a.since, name_prefix=a.name_prefix, omics_run=a.omics_run,
                                         cloudtrail=not a.no_cloudtrail)
        except (AwsUnavailable, RuntimeError, ValueError) as e:
            print(f"pipefix: {redact(str(e))}", file=sys.stderr)
            return 2
        print(f"Investigated {redact(inv.run_ref)} ({inv.platform}): {len(inv.failures)} failed task(s), "
              f"{len(inv.calls)} read-only AWS calls.")
        for fd in inv.findings:
            if fd.severity == "info":
                print(f"  - {fd.title}: {redact(fd.detail)}")
        if a.diagnose_only or not a.repo:
            print(summary(diagnoses))
            return 1 if diagnoses else 0
    else:
        if not any([a.trace, a.log, a.batch_json, a.omics_json]):
            p.error("give at least one source: --trace, --log, --batch-json or --omics-json")
        diagnoses = classify_all(collect_files(a))

    if a.json:
        print(json.dumps([{"process": d.failure.process, "tag": d.failure.tag, "class": d.cls.value,
                           "confidence": d.confidence, "evidence": [redact(e) for e in d.evidence],
                           "recommendation": d.recommendation} for d in diagnoses], indent=2))
    else:
        print(summary(diagnoses))
    if a.cmd == "diagnose":
        return 1 if diagnoses else 0
    return _deliver(a, diagnoses, inv)


if __name__ == "__main__":
    sys.exit(main())
