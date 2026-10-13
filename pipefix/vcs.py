"""Commit the patch on a new branch and open a GitHub pull request or GitLab merge request.

Uses git on the command line and the providers' REST APIs through the standard library only.
Tokens come from the environment: GITHUB_TOKEN (repo scope) or GITLAB_TOKEN (api scope).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path


def git(repo: str | Path, *args: str) -> str:
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {(r.stderr or r.stdout).strip()}")
    return r.stdout.strip()


def remote_info(repo: str | Path, remote: str = "origin") -> tuple[str, str, str]:
    """-> (provider, host, 'owner/name' or 'group/sub/name') from the remote URL."""
    url = git(repo, "remote", "get-url", remote)
    m = re.match(r"(?:https?://(?:[^@/]+@)?|git@)(?P<host>[^/:]+)[/:](?P<path>.+?)(?:\.git)?/?$", url)
    if not m:
        raise RuntimeError(f"cannot parse remote URL {url}")
    host = m.group("host")
    provider = "github" if "github" in host else "gitlab"
    return provider, host, m.group("path")


def commit_branch(repo: str | Path, files: list[str], title: str, body: str) -> tuple[str, str]:
    """Create pipefix/<timestamp> from the current branch, commit the files, return (branch, base)."""
    base = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    branch = f"pipefix/fix-{datetime.now():%Y%m%d-%H%M%S}"
    git(repo, "checkout", "-b", branch)
    git(repo, "add", *files)
    git(repo, "commit", "-m", title, "-m", body[:3000])
    return branch, base


def _request(method: str, url: str, token_header: tuple[str, str], payload: dict) -> dict:
    req = urllib.request.Request(url, method=method, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Accept": "application/json",
                                          token_header[0]: token_header[1]})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read() or b"{}")


def open_request(repo: str | Path, branch: str, base: str, title: str, body: str, remote: str = "origin",
                 draft: bool = True) -> str:
    """Push the branch and open a PR/MR. Returns its URL."""
    provider, host, path = remote_info(repo, remote)
    git(repo, "push", "-u", remote, branch)
    if provider == "github":
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if not token:
            raise RuntimeError("set GITHUB_TOKEN to open the pull request")
        api = "https://api.github.com" if host == "github.com" else f"https://{host}/api/v3"
        pr = _request("POST", f"{api}/repos/{path}/pulls", ("Authorization", f"Bearer {token}"),
                      {"title": title, "head": branch, "base": base, "body": body, "draft": draft})
        return pr["html_url"]
    token = os.environ.get("GITLAB_TOKEN")
    if not token:
        raise RuntimeError("set GITLAB_TOKEN to open the merge request")
    pid = urllib.parse.quote(path, safe="")
    mr = _request("POST", f"https://{host}/api/v4/projects/{pid}/merge_requests", ("PRIVATE-TOKEN", token),
                  {"source_branch": branch, "target_branch": base, "description": body,
                   "title": ("Draft: " if draft else "") + title, "remove_source_branch": True})
    return mr["web_url"]
