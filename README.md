# pipefix

**Your genomics pipeline failed at hour 6. pipefix tells you why, and opens the pull request that fixes it.**

pipefix reads a failed Nextflow run (from Nextflow's own trace and log, AWS Batch, or AWS HealthOmics),
classifies each failed task, and writes a config fix scoped to just the processes that failed. It
can open that fix as a draft pull request on GitHub or a merge request on GitLab.

> Status: prototype (v0.1). The diagnosis rules and the config patcher work and are tested. Live AWS fetching
> works through boto3 but has not yet been tested against real accounts.

## What it catches

| Failure | How it's recognised | What pipefix does |
|---|---|---|
| Out of memory | exit 137/247 with peak memory near the limit, Java `OutOfMemoryError`, `bad_alloc`, Batch "Container killed due to memory usage" | Raises memory for that process (≥2× request, ≥1.5× peak), growing with each retry |
| Spot interruption | Batch "Host EC2 (instance …) terminated", Spot interruption notices | Retries instead of failing the run; sets `aws.batch.maxSpotAttempts` |
| Time limit | exit 140, task ran for its full time limit, "exceeded running time limit" | Raises the time limit for that process, growing with each retry |
| Disk full | `No space left on device` | **Explains** (the fix is the compute environment's storage, not the pipeline) |
| Container / tool | `CannotPullContainerError`, `manifest unknown`, exit 127 | **Explains** |
| Reference / index | missing `.fai`/`.dict`/index, `incompatible contigs` (`chr1` vs `1`) | **Explains** |

pipefix only changes config when a config change is reliably the right fix. Every diagnosis cites the exact
exit code, number or log line it was based on.

## Try it

```bash
pip install -e .
pipefix diagnose --trace examples/oom/trace.txt --log examples/oom/nextflow.log
pipefix fix --trace examples/oom/trace.txt --log examples/oom/nextflow.log --repo examples/oom/pipeline
```

Output (dry run):

```diff
+++ b/conf/pipefix.config
+process {
+    withName: 'GATK4_HAPLOTYPECALLER' {
+        // pipefix: out of memory at 15.8 of 16.0 GB; now 32 GB, growing with each retry
+        memory        = { 32.GB * task.attempt }
+        errorStrategy = { task.exitStatus in [104, 134, 137, 139, 140, 143, 247] ? 'retry' : 'finish' }
+        maxRetries    = 2
+    }
+}
+++ b/nextflow.config
+includeConfig 'conf/pipefix.config'
```

More examples: `examples/spot` (AWS Batch Spot reclamation in STAR) and `examples/mixed` (HealthOmics run with
an OOM, a contig mismatch and a missing container image).

## Using it on a real run

```bash
# From Nextflow's own outputs (any executor). Richer trace = better diagnosis:
#   trace { enabled = true; fields = 'task_id,hash,name,process,tag,status,exit,memory,peak_rss,time,realtime' }
pipefix fix --trace results/pipeline_info/execution_trace.txt --log .nextflow.log --repo . --apply

# Straight from AWS (pip install -e '.[aws]'; read-only IAM: batch:DescribeJobs, logs:GetLogEvents,
# omics:GetRun, omics:ListRunTasks, omics:GetRunTask)
pipefix fix --batch-job <job-id> ... --repo .
pipefix fix --omics-run <run-id> --repo .

# Open a draft PR/MR (branch pipefix/fix-<time>, based on your current branch)
GITHUB_TOKEN=...  pipefix fix ... --repo . --open-pr      # or GITLAB_TOKEN for GitLab
```

## How it's built

```
pipefix/
  sources/nextflow.py   trace.txt + .nextflow.log (+ .command.err if the work dir is local)
  sources/aws.py        AWS Batch describe-jobs + CloudWatch logs, HealthOmics get-run / get-run-task
  classify.py           ordered, auditable rules -> failure class, confidence, evidence
  patch.py              per-process withName blocks in conf/pipefix.config, idempotent, included last
  report.py             terminal summary and PR/MR description
  vcs.py                branch, commit, push, GitHub/GitLab REST (standard library only)
```

No runtime dependencies beyond Python 3.10+ (boto3 only for live AWS fetching).

## Roadmap

1. Verify fixes: re-run the failed task with `-resume` on a sandbox queue and attach the result to the PR
2. GitHub App / GitLab webhook: trigger automatically on failed runs (EventBridge for Batch and HealthOmics)
3. Snakemake support (`resources:` in rules and profiles)
4. Tool-aware sizing from history: learn per-tool memory/time from past successful runs
5. Optional LLM explanation layer over the rule-based evidence (rules keep the decision; the LLM explains it)

## Development

```bash
pip install -e '.[dev]' && pytest
```
