"""Check whether a task's container image exists in Amazon ECR, and list the tags that do."""
from __future__ import annotations

import re

from .session import Clients

ECR = re.compile(r"^(?P<acct>\d{12})\.dkr\.ecr\.(?P<region>[\w-]+)\.amazonaws\.com/(?P<repo>[^:@]+)(?::(?P<tag>[^@]+))?")


def check_image(clients: Clients, image: str) -> dict | None:
    """None if the image is not in ECR. Otherwise {exists, repo, tag, recent_tags[, error]}."""
    m = ECR.match(image or "")
    if not m:
        return None
    repo, tag = m.group("repo"), m.group("tag") or "latest"
    found = clients.call("ecr", "describe_images", repositoryName=repo, registryId=m.group("acct"),
                         imageIds=[{"imageTag": tag}])
    out = {"repo": repo, "tag": tag, "exists": "__error__" not in found and bool(found.get("imageDetails"))}
    if "__error__" in found:
        out["error"] = found["__error__"]
    if not out["exists"]:
        imgs = clients.call("ecr", "describe_images", repositoryName=repo, registryId=m.group("acct"), maxResults=100)
        tagged = sorted((d for d in imgs.get("imageDetails", []) if d.get("imageTags")),
                        key=lambda d: str(d.get("imagePushedAt", "")), reverse=True)
        out["recent_tags"] = [t for d in tagged[:5] for t in d["imageTags"]][:8]
    return out
