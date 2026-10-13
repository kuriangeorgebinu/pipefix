"""Investigate a failed run end to end: AWS -> failed tasks -> CloudTrail -> diagnosis -> (ECR check)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .aws.batch import investigate_batch
from .aws.cloudtrail import access_denials
from .aws.ecr import check_image
from .aws.omics import investigate_omics
from .aws.session import Clients
from .classify import classify, classify_all
from .investigation import Investigation
from .models import Diagnosis, FailureClass


def investigate(clients: Clients, *, batch_jobs: list[str] | None = None, batch_queue: str | None = None,
                since_hours: float = 24, name_prefix: str = "", omics_run: str | None = None,
                cloudtrail: bool = True) -> tuple[Investigation, list[Diagnosis]]:
    if omics_run:
        inv = investigate_omics(clients, omics_run)
    else:
        inv = investigate_batch(clients, batch_jobs, batch_queue, since_hours, name_prefix)

    diagnoses = classify_all(inv.failures)

    if cloudtrail:
        end = inv.end or datetime.now(timezone.utc)
        start = inv.start or end - timedelta(hours=since_hours)
        inv.denials = access_denials(clients, start, end, inv.facts.get("roles", []))
        if inv.denials:
            # Use denials as evidence only for failures nothing else explains (never override e.g. an OOM)
            lines = [f"CloudTrail: {d['code']} {d['source'].split('.')[0]}:{d['action']} on {d['target'] or '?'}"
                     for d in inv.denials[:5]]
            for i, d in enumerate(diagnoses):
                if d.cls in (FailureClass.UNKNOWN, FailureClass.PERMISSION, FailureClass.CONTAINER):
                    d.failure.error_lines += lines
                    diagnoses[i] = classify(d.failure)

    for d in diagnoses:
        if d.cls in (FailureClass.CONTAINER, FailureClass.PERMISSION) and d.failure.image:
            info = check_image(clients, d.failure.image)
            if info and not info["exists"]:
                tags = ", ".join(info.get("recent_tags") or []) or "none found"
                inv.add("Container image not in ECR", f"`{info['repo']}:{info['tag']}` was not found"
                        + (f" ({info['error']})" if info.get("error") else "") + f". Recent tags: {tags}.",
                        severity="block", process=d.failure.simple_name)
                d.evidence.append(f"ECR: {info['repo']}:{info['tag']} does not exist (recent tags: {tags})")
            elif info:
                d.evidence.append(f"ECR: {info['repo']}:{info['tag']} exists, so check pull permissions/network")

    inv.calls, inv.errors = list(clients.calls), list(clients.errors)
    return inv, diagnoses
