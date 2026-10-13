"""pipefix command line.

  pipefix diagnose --trace trace.txt --log .nextflow.log
  pipefix fix      --trace trace.txt --log .nextflow.log --repo ./my-pipeline          # show the diff
  pipefix fix      ... --repo ./my-pipeline --apply                                     # write the files
  pipefix fix      ... --repo ./my-pipeline --open-pr                                   # branch, commit, PR/MR
Sources can be combined: --trace/--log (Nextflow), --batch-json/--batch-job (AWS Batch),
--omics-json/--omics-run (AWS HealthOmics).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .classify import classify_all
from .models import TaskFailure
from .patch import build_patch
from .report import pr_body, pr_title, summary
from .sources import aws, nextflow


def collect(a: argparse.Namespace) -> list[TaskFailure]:
    trace = nextflow.parse_trace(a.trace) if a.trace else []
    log = nextflow.parse_log(a.log) if a.log else []
    failures = nextflow.merge(trace, log)
    for f in failures:  # pull .command.err when the work dir is local
        wd = getattr(f, "work_dir", "")
        if wd and Path(wd).is_dir():
            f.error_lines = f.error_lines + nextflow.read_task_files(wd)
    if a.batch_json:
        failures += aws.batch_failures_from_json(aws.load_json(a.batch_json))
    if a.batch_job:
        failures += aws.fetch_batch(a.batch_job, a.region)
    if a.omics_json:
        doc = aws.load_json(a.omics_json)
        failures += aws.omics_failures_from_json(doc.get("run", doc), doc.get("tasks", []))
    if a.omics_run:
        failures += aws.fetch_omics(a.omics_run, a.region)
    return failures


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pipefix", description="Diagnose failed Nextflow runs and propose the fix.")
    p.add_argument("--version", action="version", version=f"pipefix {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("diagnose", "fix"):
        s = sub.add_parser(name)
        s.add_argument("--trace", help="Nextflow trace.txt")
        s.add_argument("--log", help=".nextflow.log or saved console output")
        s.add_argument("--batch-json", help="output of `aws batch describe-jobs`")
        s.add_argument("--batch-job", nargs="*", default=[], help="AWS Batch job id(s) to fetch")
        s.add_argument("--omics-json", help='{"run": get-run output, "tasks": [get-run-task outputs]}')
        s.add_argument("--omics-run", help="AWS HealthOmics run id to fetch")
        s.add_argument("--region")
        s.add_argument("--json", action="store_true", help="machine-readable output")
        if name == "fix":
            s.add_argument("--repo", required=True, help="the pipeline's git repo")
            s.add_argument("--apply", action="store_true", help="write the changes")
            s.add_argument("--open-pr", action="store_true", help="branch, commit, push and open a draft PR/MR")
            s.add_argument("--remote", default="origin")
            s.add_argument("--min-confidence", type=float, default=0.6)
    a = p.parse_args(argv)

    if not any([a.trace, a.log, a.batch_json, a.batch_job, a.omics_json, a.omics_run]):
        p.error("give at least one source: --trace, --log, --batch-json, --batch-job, --omics-json or --omics-run")
    diagnoses = classify_all(collect(a))

    if a.cmd == "diagnose":
        if a.json:
            print(json.dumps([{"process": d.failure.process, "tag": d.failure.tag, "class": d.cls.value,
                               "confidence": d.confidence, "evidence": d.evidence,
                               "recommendation": d.recommendation} for d in diagnoses], indent=2))
        else:
            print(summary(diagnoses))
        return 1 if diagnoses else 0

    patch = build_patch(diagnoses, a.repo, a.min_confidence)
    print(summary(diagnoses), end="\n\n")
    if patch.empty:
        print("No config change is safe to make automatically." if diagnoses else "Nothing to fix.")
        return 0
    print(patch.diff())
    if not (a.apply or a.open_pr):
        print("Dry run. Add --apply to write these changes, or --open-pr to open a draft PR/MR.")
        return 0
    written = patch.apply(a.repo)
    print("Wrote: " + ", ".join(written))
    if a.open_pr:
        from . import vcs
        title, body = pr_title(patch), pr_body(diagnoses, patch, a.omics_run or ",".join(a.batch_job))
        branch, base = vcs.commit_branch(a.repo, written, title, body)
        try:
            url = vcs.open_request(a.repo, branch, base, title, body, a.remote)
        finally:
            vcs.git(a.repo, "checkout", base)
        print(f"Opened: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
