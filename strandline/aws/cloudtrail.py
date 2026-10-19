"""Find access denials around the failed run in CloudTrail (management events: ECR, KMS, STS, S3 bucket-level).

Note: CloudTrail Event history does not contain S3 object-level (data) events such as GetObject, so a denied
read of an input file only shows up here if the trail records data events. The task logs usually show it.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from .session import Clients, paginate

DENIED = ("AccessDenied", "AccessDeniedException", "UnauthorizedOperation", "Client.UnauthorizedOperation",
          "UnauthorizedAccess", "KMS.AccessDeniedException")
SOURCES = ["ecr.amazonaws.com", "kms.amazonaws.com", "s3.amazonaws.com", "sts.amazonaws.com",
           "secretsmanager.amazonaws.com", "ssm.amazonaws.com"]


def access_denials(clients: Clients, start: datetime, end: datetime, roles: list[str], per_source: int = 300) -> list[dict]:
    start, end = start - timedelta(minutes=10), end + timedelta(minutes=5)
    role_names = [r.split("/")[-1] for r in roles if r]
    out = []
    for src in SOURCES:
        events = paginate(clients, "cloudtrail", "lookup_events", "Events", token_in="NextToken", token_out="NextToken",
                          limit=per_source, StartTime=start, EndTime=end, MaxResults=50,
                          LookupAttributes=[{"AttributeKey": "EventSource", "AttributeValue": src}])
        for e in events:
            try:
                rec = json.loads(e.get("CloudTrailEvent") or "{}")
            except ValueError:
                continue
            code = rec.get("errorCode") or ""
            if not code or not any(code.endswith(d) or code == d for d in DENIED):
                continue
            who = (rec.get("userIdentity") or {}).get("arn", "")
            if role_names and not any(f"/{n}/" in who or who.endswith(f"/{n}") for n in role_names):
                continue
            res = rec.get("requestParameters") or {}
            target = res.get("repositoryName") or res.get("bucketName") or res.get("keyId") or res.get("roleArn") or ""
            out.append({"time": rec.get("eventTime"), "source": src, "action": rec.get("eventName"), "code": code,
                        "principal": who, "target": target, "message": (rec.get("errorMessage") or "")[:300]})
    # newest first, de-duplicated by (action, target, principal)
    seen, uniq = set(), []
    for d in sorted(out, key=lambda d: d["time"] or "", reverse=True):
        k = (d["action"], d["target"], d["principal"])
        if k not in seen:
            seen.add(k)
            uniq.append(d)
    return uniq[:20]
