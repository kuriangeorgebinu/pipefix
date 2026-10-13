"""A self-contained demo: a fake AWS account with four failed tasks and an nf-core-style pipeline repo.

Used by `pipefix ui` ("Try the demo") so the whole flow can be seen without AWS access:
  GATK4_HAPLOTYPECALLER  out of memory          -> config fix (memory, capped at the largest instance)
  STAR_ALIGN             Spot reclaimed twice    -> config fix (On-Demand queue, Batch Spot retries)
  MULTIQC                image tag not in ECR    -> needs a person (with the tags that do exist)
  SAMTOOLS_SORT          KMS access denied       -> needs a person (with the CloudTrail event)
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from .fake import (GATK_JOB, GATK_WORKDIR_FILES, LAUNCH, STAR_JOB, T0, WORK, batch_account, ct_event)

PIPELINE = Path(__file__).resolve().parent / "pipeline"

MULTIQC_JOB = {
    "jobId": "job-multiqc-1", "jobName": "nf-NFCORE_SAREKSAREKMULTIQC",
    "jobQueue": "arn:aws:batch:us-east-1:123456789012:job-queue/spot-queue",
    "status": "FAILED",
    "statusReason": "CannotPullContainerError: Error response from daemon: manifest for "
                    "123456789012.dkr.ecr.us-east-1.amazonaws.com/multiqc:1.99 not found: manifest unknown",
    "createdAt": T0, "startedAt": T0 + 6 * 3_600_000, "stoppedAt": T0 + 6 * 3_600_000 + 40_000,
    "container": {"image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/multiqc:1.99", "command": ["bash", "-c", "true"],
                  "resourceRequirements": [{"type": "VCPU", "value": "1"}, {"type": "MEMORY", "value": "6144"}],
                  "jobRoleArn": "arn:aws:iam::123456789012:role/NextflowJobRole"},
    "attempts": [],
}

SORT_WORK = "s3://lab-nf-work/work/3c/9d1e2f4a5b6c7d8e9f0a1b2c3d4e5f6a"
SORT_JOB = {
    "jobId": "job-sort-1", "jobName": "nf-NFCORE_SAREKSAREKSAMTOOLS_SORT_patient7",
    "jobQueue": "arn:aws:batch:us-east-1:123456789012:job-queue/spot-queue",
    "status": "FAILED", "statusReason": "Essential container in task exited",
    "createdAt": T0, "startedAt": T0 + 2 * 3_600_000, "stoppedAt": T0 + 2 * 3_600_000 + 95_000,
    "container": {"image": "quay.io/biocontainers/samtools:1.20--h50ea8bc_0", "exitCode": 1,
                  "command": [c.replace(WORK, SORT_WORK) for c in LAUNCH],
                  "jobRoleArn": "arn:aws:iam::123456789012:role/NextflowJobRole",
                  "resourceRequirements": [{"type": "VCPU", "value": "6"}, {"type": "MEMORY", "value": "8192"}]},
    "attempts": [{"container": {"exitCode": 1}, "statusReason": "Essential container in task exited"}],
}
SORT_FILES = {
    f"{SORT_WORK}/.command.run": "#!/bin/bash\n### ---\n### name: 'NFCORE_SAREK:SAREK:SAMTOOLS_SORT (patient7)'\n"
                                 "### container: 'quay.io/biocontainers/samtools:1.20--h50ea8bc_0'\n### ...\n",
    f"{SORT_WORK}/.command.err": "download failed: s3://lab-genomes-encrypted/patient7/patient7.md.bam to "
                                "./patient7.md.bam An error occurred (AccessDenied) when calling the GetObject "
                                "operation: User is not authorized to perform: kms:Decrypt\n",
    f"{SORT_WORK}/.exitcode": "1\n",
}


def demo_clients():
    trail = [ct_event("kms.amazonaws.com", "Decrypt", "AccessDenied", "NextflowJobRole",
                      {"keyId": "arn:aws:kms:us-east-1:123456789012:key/genomics-cmk"}, "2025-10-12T22:16:41Z")]
    files = {**GATK_WORKDIR_FILES, **SORT_FILES}
    logs = {"/aws/batch/job|nf-gatk/default/aaaa1111": [
        "INFO  HaplotypeCaller - Initializing engine", "INFO  ProgressMeter - chr7:117548628 351.2 min",
        "Exception in thread \"main\" java.lang.OutOfMemoryError: Java heap space"]}
    return batch_account([GATK_JOB, STAR_JOB, MULTIQC_JOB, SORT_JOB], files, logs, trail, ecr_tags=["1.21", "1.22", "1.25"])


DEMO_JOBS = [GATK_JOB["jobId"], STAR_JOB["jobId"], MULTIQC_JOB["jobId"], SORT_JOB["jobId"]]


def demo_repo() -> Path:
    """A fresh git copy of the demo pipeline in a temp folder (so 'Write changes' can be tried safely)."""
    dst = Path(tempfile.mkdtemp(prefix="pipefix-demo-")) / "sarek-mini"
    shutil.copytree(PIPELINE, dst)
    for args in (["init", "-q", "-b", "main"], ["add", "-A"],
                 ["-c", "user.email=demo@pipefix", "-c", "user.name=pipefix demo", "commit", "-qm", "demo pipeline"]):
        subprocess.run(["git", *args], cwd=dst, check=True, capture_output=True)
    return dst
