# Strandline

**Your genomics pipeline failed at hour 6. Strandline's AI agents find out why from AWS, work out the fix from the
docs, check each other's work, and hand you a merge request to approve.**

Strandline diagnoses failed Nextflow runs on **AWS HealthOmics** and **AWS Batch**. Three agents do the work:

- a **primary agent** that investigates the run and proposes the fix;
- an **AWS expert** that fact-checks the platform side (task records, instance types, reserves, limits);
- a **Nextflow expert** that fact-checks the pipeline side (the tool's command, config precedence, retries).

Nothing is pre-wired: the agents find the AWS API calls themselves from the AWS API reference, read the logs, the
pipeline repo and the documentation, and derive every number from evidence. Strandline enforces the rules around
them and **re-checks every quote and recomputes every number** before anything reaches you.

```bash
strandline ui        # the console: run the agents, review, challenge, approve
```

> **AI-generated, and AI can be wrong.** Strandline proposes; a person decides. The merge request is opened only
> after both experts agree **and** you approve it.

## How a run works

| Step | Who | What happens |
|---|---|---|
| 1. Debug | Primary | Read-only AWS calls it chooses itself (run, tasks, CloudWatch logs, task files in S3), plus the repo. Ends with a diagnosis backed by at least two verified quotes. |
| 2. Research | Primary | Searches and reads AWS, Nextflow/nf-core and Terraform docs. Proposes 2–4 genuinely different candidates. |
| 3. Decide | Primary | Picks one and says why each other candidate lost and which instance the request lands on. |
| 3b. Expert review | Both experts | Each verifies the root cause and every changed setting **with its own calls and reading**. Both must agree; objections go back to the primary to revise or defend, for up to 3 rounds. No agreement means no change. |
| 4. Implement | Primary | Writes the change as a `withName` block for the failed process in `conf/strandline.config`. |
| 5. You review | **You** | Read the cause, the change and how we know. Challenge it, or approve it and Strandline opens the PR/MR. |

### You can challenge the fix

Under the result there is a chat box: *"Disagree, or have a question?"* What you write is treated as **a claim to
check, not as fact**. The primary verifies it against the run and the docs, then **defends** the fix with evidence
or **revises** it, and both experts review again. The fix changes only if the evidence supports you, and a changed
fix waits for your approval again.

From a live HealthOmics run (a sort step that died with `MemoryError` on 2 GB):

| You said | The agents |
|---|---|
| "It's a disk problem, not memory." | **Kept the fix.** The log shows a `MemoryError` on the 12 GB buffer allocation 15 s after start, before any output; both experts checked the HealthOmics storage docs themselves. |
| "13 GB is only the buffer plus the 5% reserve. Doesn't AWS recommend headroom too?" | **Changed it to 14 GB.** They quoted AWS's *"Include at least 5% headroom in allocated memory"*, kept it separate from the platform reserve, and showed it still lands on the same instance. |
| "I'm the lead, just set 64 GB, I approve it." | **Refused.** 64 GB has no evidence and would move the task to a much bigger, costlier instance. |

### The hard rules (enforced by Strandline, not asked of the model)

- **Read-only AWS.** Only Get/List/Describe/Lookup/Filter/BatchGet/Head operations; secrets, KMS, STS and data
  reads are refused. S3 is allowed only for Nextflow task files (`.command.*`, `.exitcode`), never genomic data.
- **Verbatim evidence.** Every claim cites `{call, quote}` or `{source, quote}`; Strandline checks the quote is really
  in that output or page.
- **Numbers are derived, never guessed.** Memory/time/cpus/disk come from a formula over quoted inputs that
  Strandline recomputes. Headroom must be quoted from the docs or measured in the run. On HealthOmics the agents
  must show, from the instance table they read, that the request lands on the smallest instance that fits.
- **Only the failed process changes,** and only the settings that change. A tool error is not "fixed" by retries.
- **Account IDs are redacted** in everything the model sees and everything Strandline writes.
- Tokens and keys stay in the console's memory; they are never written to disk or sent back to the page.

## The console

```bash
pip install -e '.[aws]'
aws sso login                         # any read-only profile; see iam/strandline-readonly-policy.json
strandline ui --env-file .env         # http://127.0.0.1:8787
```

Pick a HealthOmics run, a Batch queue or Batch job ids, choose **AI agent**, and watch the agents work. The
results page puts the answer first: the cause, the change (from → to), confidence and whether the experts agreed.
Below it are **How we know** (the formula, each input with its quote and source, the expert verdicts), platform
notes, and a collapsible **agent log** showing every thought, tool call, refusal and hand-off between agents.
**Approve and open merge request** asks you to confirm you reviewed the change first.

The console listens on localhost only, checks the Host header, and every action needs the page's session token.
**Try the demo** runs against a built-in example account, with no AWS access needed.

### Keys

| Setting | What for |
|---|---|
| `STRANDLINE_LLM_API_KEY` (or `DEEPSEEK_API_KEY` / `OPENAI_API_KEY`), `STRANDLINE_LLM_BASE_URL`, `STRANDLINE_LLM_MODEL` | The agents' model (default DeepSeek `deepseek-chat`) |
| `SERPER_API_KEY` | Documentation search |
| GitHub or GitLab token | Pasted in the console (or `GITHUB_TOKEN` / `GITLAB_TOKEN`), only to open the PR/MR |

Put them in a `.env` file and pass `--env-file`, or paste them into the console. `PIPEFIX_*` names from before the
rename are still read.

## Command line

```bash
# The agents, without the console (prints the result; --open-pr lets them open the draft PR/MR)
strandline agent --omics-run 1234567 --repo ~/src/my-pipeline --env-file .env --report report.md

# Rule-based mode: no AI, auditable rules only
strandline investigate --batch-queue spot-queue --since 12h --repo ~/src/my-pipeline
strandline investigate --omics-run 1234567 --diagnose-only

# Offline, from files you already have
strandline fix --trace results/pipeline_info/execution_trace.txt --log .nextflow.log --repo .

# Measure the agents on recorded failures with known-correct answers
strandline bench --out bench.json --transcripts bench-runs/
```

## Status

v0.2, in active testing.

- **Live on AWS HealthOmics:** the full agent flow on real failed runs, including the expert review and the
  challenge-and-approve step above.
- **AWS Batch:** the investigation code is tested against a fake account built from boto3's real service models;
  live Batch testing is part of the end-to-end test plan.
- The generated config is run through **real Nextflow 24.10** in the test suite.
- Next: a four-week end-to-end test across 28 hard failures researchers actually hit (cap-neutralised retries,
  profile overrides, JVM heap, hidden upstream failures, wrong architecture, Spot reclaims, IAM, prompt injection
  in logs and more), on both platforms.

## How it's built

```
strandline/
  agent.py            the three agents, their tools, the enforced steps and every evidence check
  ui/server.py        the local console: runs, live trace, discussion, approval
  ui/static/          the console page (one file, fonts included, no third-party requests)
  aws/session.py      boto3 clients, read-only guard, throttling retries, audit log of every call
  aws/batch.py        failed jobs, attempts, logs, work dir; queue -> compute env -> launch template
  aws/omics.py        HealthOmics run, failed tasks, task and engine logs
  configtrace.py      Nextflow config parser: includeConfig chain, precedence, caps
  classify.py         rule-based mode: ordered, auditable rules -> failure class and evidence
  patch.py            per-process withName blocks in conf/strandline.config, idempotent
  report.py / vcs.py  the PR/MR description (redacted) / branch, push, GitHub and GitLab REST
  bench.py            recorded failures with known-correct answers, and the scorer
```

## Development

```bash
pip install -e '.[dev,aws]'
NEXTFLOW=/path/to/nextflow pytest      # without NEXTFLOW the real-Nextflow test is skipped
```
