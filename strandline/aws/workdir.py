"""Read a Nextflow task's work directory on S3.

Every Nextflow task leaves these files in its work dir (s3://bucket/work/ab/cdef...):
  .command.run    wrapper script; header has `### name: 'PROC (tag)'` and `### container: '...'`
  .command.err    the tool's stderr       .command.log   stdout + stderr
  .command.trace  resource use (`nextflow.trace/v2`, memory in KB, realtime in ms)
  .exitcode       the exit status
The AWS Batch job's launch command copies `<workDir>/.command.run` from S3, which is how we find the work dir.
"""
from __future__ import annotations

import re

from .session import Clients

_S3_RUN = re.compile(r"(s3://[^\s'\"]+?)/\.command\.(?:run|log|sh)")
_FUSION_RUN = re.compile(r"/fusion/s3/([^\s'\"]+?)/\.command\.(?:run|log|sh)")
_NAME = re.compile(r"^### name: '(?P<name>.+)'\s*$", re.M)
_CONTAINER = re.compile(r"^### container: '(?P<c>.+)'\s*$", re.M)
TAIL_BYTES = 64 * 1024


def work_dir_from_command(command: list[str] | str | None) -> str:
    text = " ".join(command) if isinstance(command, list) else (command or "")
    if m := _S3_RUN.search(text):
        return m.group(1)
    if m := _FUSION_RUN.search(text):
        return "s3://" + m.group(1)
    return ""


def split_s3(uri: str) -> tuple[str, str]:
    path = uri[len("s3://"):] if uri.startswith("s3://") else uri
    bucket, _, key = path.partition("/")
    return bucket, key


def read_s3_text(clients: Clients, uri: str, tail: int | None = TAIL_BYTES) -> str | None:
    bucket, key = split_s3(uri)
    args = {"Bucket": bucket, "Key": key}
    if tail:
        args["Range"] = f"bytes=-{tail}"
    obj = clients.call("s3", "get_object", **args)
    if "__error__" in obj or "Body" not in obj:
        return None
    body = obj["Body"]
    data = body.read() if hasattr(body, "read") else body
    return data.decode("utf-8", errors="replace") if isinstance(data, (bytes, bytearray)) else str(data)


def parse_command_run_header(text: str) -> dict:
    out = {}
    if m := _NAME.search(text or ""):
        out["name"] = m.group("name")
    if m := _CONTAINER.search(text or ""):
        out["container"] = m.group("c")
    return out


def parse_command_trace(text: str) -> dict:
    """`nextflow.trace/v2` key=value lines -> dict. Memory fields come back in bytes, realtime in ms."""
    lines = (text or "").strip().splitlines()
    if not lines or lines[0].strip() != "nextflow.trace/v2":
        return {}
    out: dict = {}
    for ln in lines[1:]:
        k, _, v = ln.partition("=")
        k, v = k.strip(), v.strip()
        if not k or not v:
            continue
        try:
            n = int(v)
        except ValueError:
            continue
        out[k] = n * 1024 if k in ("rss", "vmem", "peak_rss", "peak_vmem") else n
    return out


def read_work_dir(clients: Clients, work_dir: str) -> dict:
    """Everything useful from a task work dir; missing files are simply absent from the result."""
    if not work_dir.startswith("s3://"):
        return {}
    base = work_dir.rstrip("/")
    out: dict = {"work_dir": base}
    run = read_s3_text(clients, f"{base}/.command.run", tail=None)
    if run:
        out.update(parse_command_run_header(run[:4000]))
    for name, key in ((".command.err", "err"), (".command.log", "log")):
        text = read_s3_text(clients, f"{base}/{name}")
        if text:
            out[key] = text.splitlines()[-60:]
    trace = read_s3_text(clients, f"{base}/.command.trace", tail=None)
    if trace:
        out["trace"] = parse_command_trace(trace)
    code = read_s3_text(clients, f"{base}/.exitcode", tail=None)
    if code and code.strip().lstrip("-").isdigit():
        out["exitcode"] = int(code.strip())
    return out
