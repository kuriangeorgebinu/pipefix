# pipefix

**Your genomics pipeline failed at hour 6. pipefix finds out why from AWS, traces the setting that caused it in
your pipeline repo, and opens the merge request that fixes it.**

```bash
pipefix investigate --batch-queue spot-queue --since 12h --repo . --open-pr
```

1. **Investigates through the AWS SDK (read-only).** It finds the failed AWS Batch jobs or HealthOmics run, reads
   every attempt, the CloudWatch logs, and the task's own files in the S3 work dir (`.command.run` for the real
   process name, `.command.err`, `.command.trace` for peak memory, `.exitcode`). It traces the platform the job ran on
   (job queue → compute environments → Spot or On-Demand, instance types, launch-template disk size, largest
   instance memory), searches CloudTrail for access denials during the run, and checks ECR when an image fails.
2. **Traces the pipeline config.** It follows `nextflow.config` through every `includeConfig`, reads process
   labels in the `*.nf` modules, and works out where each failed process's memory, time, retries and queue come
   from (file and line), using Nextflow's precedence (withName > withLabel > process script > generic) and any caps
   (`resourceLimits`, `params.max_memory`).
3. **Diagnoses with evidence.** Rule-based and auditable: every diagnosis cites the exit code, numbers, log line,
   AWS status reason, HealthOmics failure code or CloudTrail event it used.
4. **Fixes only what config should fix,** scoped to the failed processes, in `conf/pipefix.config`, and opens a
   GitHub PR or GitLab MR with the evidence, the config trace and the platform findings. Account IDs are redacted.

> Status: v0.2. Tested against a fake AWS account built from the documented API shapes (every call and field is
> checked against boto3's service models), and the generated config is run through **real Nextflow 24.10** in the
> test suite. Not yet run against a live AWS account.

## What it catches and what it does

| Failure | Recognised from | Action |
|---|---|---|
| Out of memory | exit 137 + peak memory near the limit (`.command.trace`), Batch "Container killed due to memory usage", HealthOmics `OUT_OF_MEMORY_ERROR`, Java/samtools/C++ allocation errors | Raises memory for that process (≥2× request, ≥1.5× peak), capped at the largest instance the queue can launch; keeps an existing retry rule |
| Spot interruption | Batch "Host EC2 (instance …) terminated" on any attempt | `aws.batch.maxSpotAttempts = 3`; moves tasks reclaimed repeatedly or running for hours to an On-Demand queue found in your account |
| Time limit | Batch "Job attempt duration exceeded timeout", HealthOmics `TASK_TIMED_OUT`, ran its full limit | Raises the time limit for that process |
| Already at the biggest instance | requested memory ≈ largest instance in the queue | **Explains** (needs a larger instance type, not config) |
| Disk full | `No space left on device`, `FILE_SYSTEM_OUT_OF_SPACE` | **Explains** with the launch template's EBS size, or the HealthOmics storage setting to use |
| Permission denied | CloudTrail `AccessDenied` for the job's role, `ECR_PERMISSION_ERROR`, "not authorized to perform" | **Explains** with the exact action, resource and role |
| Container / tool | `CannotPullContainerError`, HealthOmics container errors, exit 127 | **Explains**, with whether the tag exists in ECR and which tags do |
| Reference / input | missing `.fai`/`.dict`/index, `incompatible contigs`, `INPUT_URI_NOT_FOUND` | **Explains** |

## The console

```bash
pipefix ui            # opens http://127.0.0.1:8787 in your browser
```

A local web console for the same flow: pick a Batch queue, job ids or a HealthOmics run, watch the investigation
run, and review the result. Each failed task gets a resource lane (requested, peak used, the new value, the
largest instance the queue can launch, any pipeline cap). Its evidence, the config precedence ladder showing which
file and line sets the value today and what wins after the fix, its log, the diff, and buttons to write the change
or open the merge request. **Try the demo** runs the whole flow against a built-in example account, with no AWS
access needed. The console only listens on localhost and every action needs the page's session token. Fonts ship
with pipefix, so it makes no third-party requests.

## Setup

```bash
pip install -e '.[aws]'
aws sso login            # or `aws configure`; any read-only profile works
```

Permissions: [`iam/pipefix-readonly-policy.json`](iam/pipefix-readonly-policy.json) is the complete list, all
read-only. pipefix refuses to call anything but Describe/Get/List/Lookup APIs.

## Use

```bash
# Find failed jobs in a Batch queue from the last 12 hours and show the proposed change (dry run)
pipefix investigate --batch-queue spot-queue --since 12h --repo ~/src/my-pipeline

# Specific jobs, save the full report, open the PR/MR
GITHUB_TOKEN=... pipefix investigate --batch-job 3f1c2a9e-... --repo . --report pipefix-report.md --open-pr

# A HealthOmics run
pipefix investigate --omics-run 1234567 --repo . --region us-east-1

# Just the diagnosis, no repo
pipefix investigate --omics-run 1234567 --diagnose-only

# Offline, from files you already have
pipefix fix --trace results/pipeline_info/execution_trace.txt --log .nextflow.log --repo .
```

`--open-pr` checks the token and the remote before writing anything, skips if an open pipefix PR/MR already
proposes the same change, and opens a draft (`--ready` for a normal one). GitLab is detected from the remote
(`GITLAB_TOKEN`), or use `--provider gitlab`. Pass `--nf-profile` with the profiles the run used so profile-only
settings are traced too.

## How it's built

```
pipefix/
  aws/session.py      boto3 clients, read-only guard, throttling retries, audit log of every call
  aws/batch.py        failed jobs, attempts, logs, work dir; queue -> compute env -> launch template -> instances
  aws/workdir.py      .command.run / .err / .trace / .exitcode from the task's S3 work dir
  aws/omics.py        HealthOmics run, failed tasks, task + engine logs (/aws/omics/WorkflowLog), storage
  aws/cloudtrail.py   access denials in the run window for the run's roles
  aws/ecr.py          does the image tag exist, which tags do
  configtrace.py      Nextflow config parser + includeConfig chain + precedence + caps
  classify.py         ordered, auditable rules -> failure class, confidence, evidence
  patch.py            per-process withName blocks, platform- and cap-aware, idempotent
  report.py / vcs.py  MR description (redacted) / branch, push, GitHub & GitLab REST
  sources/            offline inputs: Nextflow trace/log, saved AWS JSON
```

## Roadmap

1. Verify the fix by re-running the failed task (`-resume`) on a sandbox queue and attaching the result
2. Trigger automatically: EventBridge rule on Batch job / HealthOmics run failure -> pipefix (Lambda or container)
3. Tool-specific fixes (STAR `--limitBAMsortRAM`, samtools sort `-m`, GATK heap) beyond resource changes
4. Snakemake support; learning per-tool sizes from past runs; optional LLM explanation over the evidence

## Development

```bash
pip install -e '.[dev,aws]'
NEXTFLOW=/path/to/nextflow pytest      # without NEXTFLOW the real-Nextflow test is skipped
```
