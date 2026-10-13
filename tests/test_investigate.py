"""End-to-end tests of `pipefix investigate` against a fake AWS account and an nf-core-style repo."""
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from fakeaws import (GATK_JOB, GATK_WORKDIR_FILES, STAR_JOB, WORK, batch_account, ct_event, omics_account)
from pipefix.aws.workdir import parse_command_trace, work_dir_from_command
from pipefix.configtrace import load_repo
from pipefix.investigate import investigate
from pipefix.models import FailureClass
from pipefix.patch import CONFIG_REL, build_patch
from pipefix.report import body, redact

EX = Path(__file__).resolve().parent.parent / "examples"
GiB = 1024**3


@pytest.fixture
def repo(tmp_path):
    dst = tmp_path / "sarek-mini"
    shutil.copytree(EX / "sarek-mini", dst)
    return dst


# ---------- building blocks ----------

def test_work_dir_from_launch_command():
    assert work_dir_from_command(GATK_JOB["container"]["command"]) == WORK
    assert work_dir_from_command(["/usr/bin/fusion", "bash", "/fusion/s3/b/work/ab/cd/.command.run"]) == "s3://b/work/ab/cd"
    assert work_dir_from_command(["bash", "-c", "true"]) == ""


def test_command_trace_units():
    tr = parse_command_trace(GATK_WORKDIR_FILES[f"{WORK}/.command.trace"])
    assert tr["peak_rss"] == 16180000 * 1024 and tr["realtime"] == 21070000
    assert parse_command_trace("not a trace") == {}


def test_config_trace(repo):
    cfg = load_repo(repo)
    tr = cfg.trace("NFCORE_SAREK:SAREK:GATK4_HAPLOTYPECALLER")
    assert tr.labels == ["process_medium"]
    assert tr.effective("memory").where == "conf/base.config:16"
    assert "retry" in tr.effective("errorStrategy").value
    # the test profile's 2.GB override only applies when that profile is used
    assert cfg.trace("GATK4_HAPLOTYPECALLER", ("test",)).effective("memory").value == "2.GB"
    assert cfg.trace(".:ALIGN_STAR:STAR_ALIGN".replace(".", "NFCORE_RNASEQ:RNASEQ")).effective("memory").value.startswith("{ 72.GB")
    assert cfg.caps["memory"] == "'128.GB'"
    assert cfg.resolve_process("nf-NFCORE_RNASEQRNASEQALIGN_STARSTAR_ALIGN_sampleB") == ("STAR_ALIGN", "sampleB")


# ---------- AWS Batch ----------

def test_batch_oom_reads_workdir_platform_and_fixes(repo):
    clients = batch_account([GATK_JOB], GATK_WORKDIR_FILES,
                            {"/aws/batch/job|nf-gatk/default/aaaa1111": ["Running HaplotypeCaller", "Killed"]})
    inv, diags = investigate(clients, batch_queue="spot-queue", since_hours=12)
    f = inv.failures[0]
    # real process name came from the .command.run header, peak memory from .command.trace
    assert f.process.endswith(":GATK4_HAPLOTYPECALLER") and f.tag == "sample1"
    assert f.context["process_name_source"] == ".command.run header"
    assert round(f.peak_rss_bytes / GiB, 1) == 15.4 and f.memory_bytes == 16 * GiB
    assert diags[0].cls == FailureClass.OUT_OF_MEMORY and diags[0].confidence >= 0.95
    # platform: largest instance r5.4xlarge / m5.8xlarge = 128 GiB, minus headroom
    assert 120_000 < inv.facts["max_task_memory_mib"] < 131072
    assert inv.facts["ondemand_queues"] == ["ondemand-queue"]
    assert any("EBS volumes 100 GiB" in fd.detail for fd in inv.findings)
    assert all(c.split(":")[1].startswith(("describe_", "get_", "list_", "lookup_")) for c in inv.calls)

    patch = build_patch(diags, repo, inv=inv)
    cfg = patch.files[CONFIG_REL][1]
    assert "withName: 'GATK4_HAPLOTYPECALLER'" in cfg
    assert "{ [32.GB * task.attempt, 121.GB].min() }" in cfg
    assert "errorStrategy" not in cfg            # base.config already retries on 137: kept
    assert "maxRetries = 2" in cfg              # raised from 1 (conf/base.config:7)
    text = body(diags, patch, inv)
    assert "conf/base.config:16" in text and "process_medium" in text
    assert "123456789012" not in text           # account ids redacted


def test_batch_spot_moves_to_ondemand_queue_and_resolves_name(repo):
    clients = batch_account([STAR_JOB])
    inv, diags = investigate(clients, batch_jobs=["job-star-1"])
    d = diags[0]
    assert d.cls == FailureClass.SPOT_INTERRUPTION
    assert "2 of 2 attempt(s) lost to Spot reclamation" in d.evidence
    patch = build_patch(diags, repo, inv=inv)
    cfg = patch.files[CONFIG_REL][1]
    assert "aws.batch.maxSpotAttempts = 3" in cfg
    assert "withName: 'STAR_ALIGN'" in cfg and "queue      = 'ondemand-queue'" in cfg
    assert any(fd.title == "Process name is approximate" for fd in inv.findings)


def test_batch_memory_already_at_instance_limit_is_not_fixed_by_config(repo):
    big = {**GATK_JOB, "container": {**GATK_JOB["container"], "resourceRequirements": [
        {"type": "MEMORY", "value": str(124_000)}]}}
    files = dict(GATK_WORKDIR_FILES)
    files[f"{WORK}/.command.trace"] = "nextflow.trace/v2\nrealtime=100\npeak_rss=126000000\n"
    inv, diags = investigate(batch_account([big], files), batch_jobs=["job-gatk-1"], cloudtrail=False)
    patch = build_patch(diags, repo, inv=inv)
    assert patch.empty
    assert any(fd.title == "Already at the largest instance" for fd in patch.findings)


def test_cloudtrail_denial_explains_unknown_failure(repo):
    job = {**GATK_JOB, "attempts": [{"container": {"exitCode": 1}}],
           "container": {**GATK_JOB["container"], "exitCode": 1, "command": ["bash", "-c", "true"]}}
    trail = [ct_event("kms.amazonaws.com", "Decrypt", "AccessDenied", "NextflowJobRole", {"keyId": "alias/genomics"}),
             ct_event("kms.amazonaws.com", "Decrypt", "AccessDenied", "SomeOtherRole", {"keyId": "alias/other"})]
    inv, diags = investigate(batch_account([job], trail=trail), batch_jobs=["job-gatk-1"])
    assert [d["target"] for d in inv.denials] == ["alias/genomics"]      # only the job's own role
    assert diags[0].cls == FailureClass.PERMISSION
    patch = build_patch(diags, repo, inv=inv)
    assert patch.empty and patch.advisories
    assert "kms:Decrypt" in body(diags, patch, inv)


# ---------- AWS HealthOmics ----------

def _omics():
    run = {"id": "1234567", "status": "FAILED", "failureReason": "RUN_TASK_FAILED", "workflowId": "9876543",
           "engineVersion": "24.10.0", "storageType": "STATIC", "storageCapacity": 1200,
           "roleArn": "arn:aws:iam::123456789012:role/OmicsRunRole",
           "startTime": datetime(2026, 10, 12, 20, tzinfo=timezone.utc), "stopTime": datetime(2026, 10, 12, 23, tzinfo=timezone.utc),
           "logLocation": {"engineLogStream": "run/1234567/engine"}}
    tasks = [
        {"taskId": "111", "name": "NFCORE_SAREK:SAREK:SAMTOOLS_SORT (patient7)", "status": "FAILED", "memory": 8,
         "cpus": 4, "failureReason": "OUT_OF_MEMORY_ERROR", "statusMessage": "Task failed",
         "instanceType": "omics.m.xlarge"},
        {"taskId": "222", "name": "MULTIQC", "status": "FAILED", "memory": 4, "cpus": 1,
         "failureReason": "CANNOT_START_CONTAINER_ERROR",
         "imageDetails": {"image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/multiqc:1.99"}},
    ]
    engine = [
        "N E X T F L O W  ~  version 24.10.0",
        "ERROR ~ Error executing process > 'NFCORE_SAREK:SAREK:SAMTOOLS_SORT (patient7)'",
        "Caused by:",
        "  Process `NFCORE_SAREK:SAREK:SAMTOOLS_SORT (patient7)` terminated with an error exit status (137)",
        "Command error:",
        "  [bam_sort_core] merging from 24 files and 4 in-memory blocks...",
        "  samtools sort: couldn't allocate memory for bam_mem",
    ]
    streams = {"/aws/omics/WorkflowLog|run/1234567/engine": engine,
               "/aws/omics/WorkflowLog|run/1234567/task/111": ["sorting..."],
               "/aws/omics/WorkflowLog|run/1234567/task/222": []}
    return omics_account(run, tasks, streams, ecr_tags=["1.21", "1.22"])


def test_omics_run(repo):
    inv, diags = investigate(_omics(), omics_run="1234567")
    assert inv.nextflow_version == "24.10.0"
    by = {d.failure.simple_name: d for d in diags}
    assert by["SAMTOOLS_SORT"].cls == FailureClass.OUT_OF_MEMORY
    assert by["SAMTOOLS_SORT"].failure.exit_code == 137          # from the engine log
    assert by["MULTIQC"].cls == FailureClass.CONTAINER
    assert any("multiqc:1.99 does not exist" in e for e in by["MULTIQC"].evidence)
    patch = build_patch(diags, repo, inv=inv)
    cfg = patch.files[CONFIG_REL][1]
    assert "withName: 'SAMTOOLS_SORT'" in cfg and "{ 16.GB * task.attempt }" in cfg
    assert "maxSpotAttempts" not in cfg                          # not a Batch run
    text = body(diags, patch, inv)
    assert "create-workflow-version" in text and "Recent tags: 1.22, 1.21" in text   # newest first
    assert any("couldn't allocate memory" in e for e in by["SAMTOOLS_SORT"].evidence) or \
        by["SAMTOOLS_SORT"].confidence >= 0.95


# ---------- the MR path ----------

def test_open_pr_refuses_without_token_and_pushes_nothing(repo, monkeypatch, tmp_path):
    from pipefix import cli
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    for args in (["init", "-q", "-b", "main"], ["add", "-A"], ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
                 ["remote", "add", "origin", "https://github.com/acme/sarek-mini.git"]):
        subprocess.run(["git", *args], cwd=repo, check=True)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr("pipefix.aws.session.boto3_clients", lambda *a, **k: batch_account([GATK_JOB], GATK_WORKDIR_FILES))
    with pytest.raises(RuntimeError, match="GITHUB_TOKEN"):
        cli.main(["investigate", "--batch-job", "job-gatk-1", "--repo", str(repo), "--open-pr", "--no-cloudtrail"])
    assert not (repo / CONFIG_REL).exists()                     # nothing written, nothing pushed
    assert subprocess.run(["git", "branch"], cwd=repo, capture_output=True, text=True).stdout.strip() == "* main"


def test_redact():
    assert redact("arn:aws:iam::123456789012:role/X and 123456789012.dkr.ecr") == \
        "arn:aws:iam::<account-id>:role/X and <account-id>.dkr.ecr"


def test_unexpected_aws_errors_are_recorded_not_fatal():
    from fakeaws import ClientError, fake_clients
    from pipefix.aws.session import AwsUnavailable
    c = fake_clients({"ec2": {"describe_instance_types": lambda **k: (_ for _ in ()).throw(ClientError("InvalidInstanceType"))},
                      "sts": {"get_caller_identity": lambda **k: (_ for _ in ()).throw(ClientError("ExpiredToken"))}})
    assert c.call("ec2", "describe_instance_types", InstanceTypes=["nope.x"]) == {"__error__": "InvalidInstanceType"}
    assert c.errors == ["ec2:describe_instance_types -> InvalidInstanceType"]
    with pytest.raises(AwsUnavailable, match="ExpiredToken"):
        c.call("sts", "get_caller_identity")
    with pytest.raises(ValueError):
        c.call("batch", "terminate_job", jobId="x")          # never anything but read-only calls
