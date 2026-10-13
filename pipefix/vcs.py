"""Open the fix as a GitHub pull request or GitLab merge request.

Order matters: the token and the remote are checked first, then existing open pipefix MRs are searched for the
same change (same diff hash), and only then is anything committed or pushed. Uses git and the providers'
REST APIs through the standard library. Tokens: GITHUB_TOKEN / GH_TOKEN (repo scope) or GITLAB_TOKEN (api scope).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def git(repo: str | Path, *args: str) -> str:
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {(r.stderr or r.stdout).strip()}")
    return r.stdout.strip()


@dataclass
class Remote:
    provider: str      # github | gitlab
    host: str
    path: str          # owner/name or group/sub/name
    token: str
    name: str = "origin"

    @property
    def api(self) -> str:
        if self.provider == "github":
            return "https://api.github.com" if self.host == "github.com" else f"https://{self.host}/api/v3"
        return f"https://{self.host}/api/v4"


def remote_info(repo: str | Path, remote: str = "origin", provider: str | None = None) -> Remote:
    url = git(repo, "remote", "get-url", remote)
    m = re.match(r"(?:https?://(?:[^@/]+@)?|ssh://git@|git@)(?P<host>[^/:]+)(?::\d+)?[/:](?P<path>.+?)(?:\.git)?/?$", url)
    if not m:
        raise RuntimeError(f"cannot parse remote URL {url}")
    host = m.group("host")
    prov = provider or ("github" if "github" in host else "gitlab")
    token = (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")) if prov == "github" else \
        os.environ.get("GITLAB_TOKEN")
    if not token:
        raise RuntimeError(f"set {'GITHUB_TOKEN' if prov == 'github' else 'GITLAB_TOKEN'} to open the "
                           f"{'pull request' if prov == 'github' else 'merge request'} (nothing was pushed)")
    return Remote(prov, host, m.group("path"), token, remote)


def diff_hash(diff: str) -> str:
    return hashlib.sha1(diff.encode()).hexdigest()[:8]


def _request(method: str, url: str, remote: Remote, payload: dict | None = None):
    hdr = {"Accept": "application/json", "Content-Type": "application/json"}
    hdr.update({"Authorization": f"Bearer {remote.token}"} if remote.provider == "github" else
               {"PRIVATE-TOKEN": remote.token})
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, method=method, data=data, headers=hdr)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read() or b"null")


def find_existing(remote: Remote, h: str) -> str | None:
    """An open pipefix PR/MR for the same change (its branch ends with the diff hash)."""
    if remote.provider == "github":
        prs = _request("GET", f"{remote.api}/repos/{remote.path}/pulls?state=open&per_page=100", remote) or []
        return next((p["html_url"] for p in prs if p["head"]["ref"].startswith("pipefix/")
                     and p["head"]["ref"].endswith(h)), None)
    pid = urllib.parse.quote(remote.path, safe="")
    mrs = _request("GET", f"{remote.api}/projects/{pid}/merge_requests?state=opened&per_page=100", remote) or []
    return next((m["web_url"] for m in mrs if m["source_branch"].startswith("pipefix/")
                 and m["source_branch"].endswith(h)), None)


def commit_branch(repo: str | Path, files: list[str], title: str, body: str, h: str) -> tuple[str, str]:
    base = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    branch = f"pipefix/fix-{datetime.now():%Y%m%d-%H%M%S}-{h}"
    git(repo, "checkout", "-b", branch)
    git(repo, "add", *files)
    git(repo, "commit", "-m", title, "-m", body[:3000])
    return branch, base


def open_request(repo: str | Path, remote: Remote, branch: str, base: str, title: str, body: str,
                 draft: bool = True) -> str:
    git(repo, "push", "-u", remote.name, branch)
    if remote.provider == "github":
        payload = {"title": title, "head": branch, "base": base, "body": body[:65000], "draft": draft}
        try:
            pr = _request("POST", f"{remote.api}/repos/{remote.path}/pulls", remote, payload)
        except urllib.error.HTTPError as e:
            # Draft PRs aren't available on private repos of free GitHub plans: open a normal PR instead
            if e.code == 422 and draft and "draft" in e.read().decode(errors="replace").lower():
                pr = _request("POST", f"{remote.api}/repos/{remote.path}/pulls", remote, {**payload, "draft": False})
            else:
                raise
        return pr["html_url"]
    pid = urllib.parse.quote(remote.path, safe="")
    mr = _request("POST", f"{remote.api}/projects/{pid}/merge_requests", remote,
                  {"source_branch": branch, "target_branch": base, "description": body[:1_000_000],
                   "title": ("Draft: " if draft else "") + title, "remove_source_branch": True})
    return mr["web_url"]
