"""A fake AWS account for tests and the UI demo: canned API responses shaped like boto3's, plus a record of calls.

Responses follow the documented shapes of AWS Batch, EC2, CloudWatch Logs, S3, HealthOmics, CloudTrail and ECR.
"""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone

from strandline.aws.session import Clients


class ClientError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeService:
    def __init__(self, name: str, handlers: dict):
        self.name, self.handlers = name, handlers

    def __getattr__(self, op):
        if op not in self.handlers:
            raise AttributeError(op)

        def call(**kw):
            h = self.handlers[op]
            return h(**kw) if callable(h) else h
        return call


def fake_clients(services: dict[str, dict]) -> Clients:
    return Clients(lambda svc: FakeService(svc, services.get(svc, {})))


def s3_store(files: dict[str, str]):
    def get_object(Bucket, Key, Range=None):
        uri = f"s3://{Bucket}/{Key}"
        if uri not in files:
            raise ClientError("NoSuchKey")
        data = files[uri].encode()
        if Range and Range.startswith("bytes=-"):
            data = data[-int(Range[7:]):]
        return {"Body": io.BytesIO(data)}
    return {"get_object": get_object}


def logs_store(streams: dict[str, list[str]]):
    def get_log_events(logGroupName, logStreamName, limit=100, startFromHead=False):
        lines = streams.get(f"{logGroupName}|{logStreamName}")
        if lines is None:
            raise ClientError("ResourceNotFoundException")
        return {"events": [{"message": m} for m in lines[-limit:]]}
    return {"get_log_events": get_log_events}


T0 = 1760300000000  # ms epoch
WORK = "s3://lab-nf-work/work/e5/f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"
LAUNCH = ["bash", "-o", "pipefail", "-c",
          f"trap \"{{ ret=$?; /home/ec2-user/miniconda/bin/aws s3 cp --only-show-errors .command.log "
          f"{WORK}/.command.log||true; exit $ret; }}\" EXIT; /home/ec2-user/miniconda/bin/aws s3 cp "
          f"--only-show-errors {WORK}/.command.run - | bash > >(tee .command.log) 2>&1 & pid=$!; wait $pid"]

GATK_JOB = {
    "jobId": "job-gatk-1", "jobName": "nf-NFCORE_SAREKSAREKBAM_VARIANT_CALLING_GERMLINE_ALLGATK4_HAPLOTYPECALLER_sample1",
    "jobQueue": "arn:aws:batch:us-east-1:123456789012:job-queue/spot-queue",
    "jobDefinition": "arn:aws:batch:us-east-1:123456789012:job-definition/nf-broadinstitute-gatk-4-5-0-0:1",
    "status": "FAILED", "statusReason": "Essential container in task exited",
    "createdAt": T0, "startedAt": T0 + 60_000, "stoppedAt": T0 + 60_000 + 5 * 3_600_000,
    "container": {
        "image": "broadinstitute/gatk:4.5.0.0", "command": LAUNCH, "exitCode": 137,
        "jobRoleArn": "arn:aws:iam::123456789012:role/NextflowJobRole",
        "resourceRequirements": [{"type": "VCPU", "value": "12"}, {"type": "MEMORY", "value": "16384"}],
        "logStreamName": "nf-gatk/default/aaaa1111",
    },
    "attempts": [{"container": {"exitCode": 137, "reason": "OutOfMemoryError: Container killed due to memory usage",
                                "logStreamName": "nf-gatk/default/aaaa1111"}, "statusReason": "Essential container in task exited"}],
}

STAR_JOB = {
    "jobId": "job-star-1", "jobName": "nf-NFCORE_RNASEQRNASEQALIGN_STARSTAR_ALIGN_sampleB",
    "jobQueue": "arn:aws:batch:us-east-1:123456789012:job-queue/spot-queue",
    "status": "FAILED", "statusReason": "Host EC2 (instance i-0a1b2c3d4e5f67890) terminated.",
    "createdAt": T0, "startedAt": T0 + 60_000, "stoppedAt": T0 + 60_000 + 3 * 3_600_000,
    "container": {"image": "quay.io/biocontainers/star:2.7.11b--h43eeafb_0", "command": ["bash", "-c", "true"],
                  "resourceRequirements": [{"type": "VCPU", "value": "12"}, {"type": "MEMORY", "value": "73728"}]},
    "attempts": [
        {"container": {"reason": "Host EC2 (instance i-0a1b2c3d4e5f67890) terminated."},
         "statusReason": "Host EC2 (instance i-0a1b2c3d4e5f67890) terminated."},
        {"container": {"reason": "Host EC2 (instance i-0f9e8d7c6b5a43210) terminated."},
         "statusReason": "Host EC2 (instance i-0f9e8d7c6b5a43210) terminated."},
    ],
}

QUEUES = [
    {"jobQueueName": "spot-queue", "jobQueueArn": "arn:aws:batch:us-east-1:123456789012:job-queue/spot-queue",
     "state": "ENABLED", "status": "VALID",
     "computeEnvironmentOrder": [{"order": 1, "computeEnvironment": "arn:aws:batch:us-east-1:123456789012:compute-environment/spot-ce"}]},
    {"jobQueueName": "ondemand-queue", "jobQueueArn": "arn:aws:batch:us-east-1:123456789012:job-queue/ondemand-queue",
     "state": "ENABLED", "status": "VALID",
     "computeEnvironmentOrder": [{"order": 1, "computeEnvironment": "arn:aws:batch:us-east-1:123456789012:compute-environment/od-ce"}]},
]
CES = {
    "arn:aws:batch:us-east-1:123456789012:compute-environment/spot-ce": {
        "computeEnvironmentName": "spot-ce", "computeEnvironmentArn": "arn:aws:batch:us-east-1:123456789012:compute-environment/spot-ce",
        "state": "ENABLED", "status": "VALID",
        "computeResources": {"type": "SPOT", "allocationStrategy": "SPOT_PRICE_CAPACITY_OPTIMIZED",
                             "instanceTypes": ["m5", "r5.4xlarge"], "maxvCpus": 256,
                             "launchTemplate": {"launchTemplateName": "nf-lt", "version": "$Latest"}}},
    "arn:aws:batch:us-east-1:123456789012:compute-environment/od-ce": {
        "computeEnvironmentName": "od-ce", "computeEnvironmentArn": "arn:aws:batch:us-east-1:123456789012:compute-environment/od-ce",
        "state": "ENABLED", "status": "VALID",
        "computeResources": {"type": "EC2", "allocationStrategy": "BEST_FIT_PROGRESSIVE", "instanceTypes": ["optimal"],
                             "maxvCpus": 64}},
}
INSTANCE_TYPES = {"m5.large": 8192, "m5.4xlarge": 65536, "m5.8xlarge": 131072, "r5.4xlarge": 131072,
                  "c5.4xlarge": 32768, "r5.8xlarge": 262144}


def _describe_ces(computeEnvironments):
    return {"computeEnvironments": [CES[a] for a in computeEnvironments if a in CES]}


def _describe_instance_types(InstanceTypes=None, Filters=None, NextToken=None):
    import fnmatch
    if InstanceTypes:
        names = [n for n in InstanceTypes if n in INSTANCE_TYPES]
    else:
        pats = Filters[0]["Values"]
        names = [n for n in INSTANCE_TYPES if any(fnmatch.fnmatch(n, p) for p in pats)]
    return {"InstanceTypes": [{"InstanceType": n, "MemoryInfo": {"SizeInMiB": INSTANCE_TYPES[n]}} for n in names]}


def _ecr(tags: list[str] | None):
    def describe_images(repositoryName, registryId=None, imageIds=None, maxResults=None):
        if imageIds:
            if imageIds[0]["imageTag"] in (tags or []):
                return {"imageDetails": [{"imageTags": [imageIds[0]["imageTag"]]}]}
            raise ClientError("ImageNotFoundException")
        return {"imageDetails": [{"imageTags": [t], "imagePushedAt": datetime(2026, 1, i + 1, tzinfo=timezone.utc)}
                                 for i, t in enumerate(tags or [])]}
    return {"describe_images": describe_images}


def batch_account(jobs: list[dict], s3_files: dict[str, str] | None = None, log_streams: dict | None = None,
                  trail: list[dict] | None = None, ecr_tags: list[str] | None = None) -> Clients:
    by_id = {j["jobId"]: j for j in jobs}

    def describe_job_queues(jobQueues=None, maxResults=None, nextToken=None):
        if jobQueues:
            return {"jobQueues": [q for q in QUEUES if q["jobQueueArn"] in jobQueues or q["jobQueueName"] in jobQueues]}
        return {"jobQueues": QUEUES}

    def list_jobs(jobQueue, filters=None, maxResults=100, nextToken=None):
        return {"jobSummaryList": [{"jobId": j["jobId"], "jobName": j["jobName"], "status": j["status"]}
                                   for j in jobs] + [{"jobId": "ok-1", "jobName": "nf-FASTP_x", "status": "SUCCEEDED"}]}

    return fake_clients({
        "batch": {"describe_jobs": lambda jobs: {"jobs": [by_id[j] for j in jobs if j in by_id]},
                  "list_jobs": list_jobs, "describe_job_queues": describe_job_queues,
                  "describe_compute_environments": _describe_ces},
        "ec2": {"describe_instance_types": _describe_instance_types,
                "describe_launch_template_versions": {"LaunchTemplateVersions": [{"LaunchTemplateData": {
                    "BlockDeviceMappings": [{"DeviceName": "/dev/xvda", "Ebs": {"VolumeSize": 100}}]}}]}},
        "logs": logs_store(log_streams or {}),
        "s3": s3_store(s3_files or {}),
        "cloudtrail": {"lookup_events": lambda **kw: {"Events": [e for e in (trail or [])
                       if kw["LookupAttributes"][0]["AttributeValue"] == json.loads(e["CloudTrailEvent"])["eventSource"]]}},
        "ecr": _ecr(ecr_tags),
    })


GATK_WORKDIR_FILES = {
    f"{WORK}/.command.run": "#!/bin/bash\n### ---\n### name: 'NFCORE_SAREK:SAREK:BAM_VARIANT_CALLING_GERMLINE_ALL:"
                           "GATK4_HAPLOTYPECALLER (sample1)'\n### container: 'broadinstitute/gatk:4.5.0.0'\n### ...\nset -e\n",
    f"{WORK}/.command.err": "22:13:58.402 INFO  ProgressMeter - chr7:117548628 351.2 41810000 119058.3\n"
                           "Exception in thread \"main\" java.lang.OutOfMemoryError: Java heap space\n",
    f"{WORK}/.command.trace": "nextflow.trace/v2\nrealtime=21070000\n%cpu=1180\nrchar=1\nwchar=2\n"
                             "peak_rss=16180000\npeak_vmem=17000000\nrss=16000000\nvmem=16900000\n",
    f"{WORK}/.exitcode": "137\n",
}


def ct_event(source: str, name: str, code: str, role: str, target: dict, when: str = "2025-10-12T21:00:00Z") -> dict:
    return {"CloudTrailEvent": json.dumps({
        "eventSource": source, "eventName": name, "errorCode": code, "eventTime": when,
        "errorMessage": f"User: arn:aws:sts::123456789012:assumed-role/{role}/i-0abc is not authorized to perform: {name}",
        "userIdentity": {"arn": f"arn:aws:sts::123456789012:assumed-role/{role}/i-0abc"},
        "requestParameters": target})}


def omics_account(run: dict, tasks: list[dict], streams: dict[str, list[str]], ecr_tags: list[str] | None = None) -> Clients:
    by_id = {t["taskId"]: t for t in tasks}

    def describe_images(repositoryName, registryId=None, imageIds=None, maxResults=None):
        if imageIds:
            tag = imageIds[0]["imageTag"]
            if tag in (ecr_tags or []):
                return {"imageDetails": [{"imageTags": [tag]}]}
            raise ClientError("ImageNotFoundException")
        return {"imageDetails": [{"imageTags": [t], "imagePushedAt": datetime(2026, 1, i + 1, tzinfo=timezone.utc)}
                                 for i, t in enumerate(ecr_tags or [])]}

    return fake_clients({
        "omics": {"get_run": run,
                  "list_run_tasks": {"items": [{"taskId": t["taskId"], "status": t["status"]} for t in tasks]},
                  "get_run_task": lambda id, taskId: by_id[taskId]},
        "logs": logs_store(streams),
        "cloudtrail": {"lookup_events": {"Events": []}},
        "ecr": {"describe_images": describe_images},
    })
