"""Read failed tasks from Nextflow's own outputs: the trace file and .nextflow.log.

Both are produced on every run, so this works for any executor (AWS Batch, HealthOmics, local, Slurm).
For the most information, run Nextflow with a trace that includes the requested resources:

    trace { enabled = true; fields = 'task_id,hash,name,process,tag,status,exit,memory,peak_rss,time,realtime' }
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from ..models import TaskFailure
from ..units import parse_duration, parse_memory

FAILED = {"FAILED", "ABORTED"}
_NAME_TAG = re.compile(r"^(?P<proc>[^()]+?)\s*(?:\((?P<tag>.*)\))?\s*$")


def _split_name(name: str) -> tuple[str, str]:
    m = _NAME_TAG.match(name or "")
    return (m.group("proc").strip(), (m.group("tag") or "").strip()) if m else (name, "")


def _int(text: str | None) -> int | None:
    try:
        return int(str(text).strip())
    except (TypeError, ValueError):
        return None


def parse_trace(path: str | Path) -> list[TaskFailure]:
    """Failed tasks from a Nextflow trace.txt (tab separated, header row)."""
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if (row.get("status") or "").strip().upper() not in FAILED:
                continue
            proc, tag = _split_name(row.get("name", ""))
            out.append(TaskFailure(
                process=(row.get("process") or proc).strip(),
                tag=(row.get("tag") or tag).strip(),
                exit_code=_int(row.get("exit")),
                memory_bytes=parse_memory(row.get("memory")),
                peak_rss_bytes=parse_memory(row.get("peak_rss")),
                time_ms=parse_duration(row.get("time")),
                realtime_ms=parse_duration(row.get("realtime")),
                source="nextflow-trace",
            ))
            out[-1].hash = (row.get("hash") or "").strip()  # type: ignore[attr-defined]
    return out


_ERR_HEAD = re.compile(r"Error executing process > '(?P<name>[^']+)'")
_EXIT = re.compile(r"terminated with an error exit status \((?P<code>\d+)\)")
_SECTIONS = ("Caused by:", "Command executed:", "Command exit status:", "Command output:", "Command error:",
             "Work dir:", "Tip:", "Container:")


def parse_log(path: str | Path) -> list[TaskFailure]:
    """Failed tasks described in .nextflow.log (or the console output), with their error lines."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[TaskFailure] = []
    i = 0
    while i < len(lines):
        head = _ERR_HEAD.search(lines[i])
        if not head:
            i += 1
            continue
        proc, tag = _split_name(head.group("name"))
        block, i = [], i + 1
        while i < len(lines) and not _ERR_HEAD.search(lines[i]) and not lines[i].startswith(("ERROR ~", "Execution cancelled")):
            block.append(lines[i])
            i += 1
        text = "\n".join(block)
        code = _EXIT.search(text)
        exit_code = int(code.group("code")) if code else None
        if exit_code is None:
            m = re.search(r"Command exit status:\s*\n\s*(-?\d+)", text)
            exit_code = int(m.group(1)) if m else None
        errors, section = [], None
        for ln in block:
            s = ln.strip()
            if s in _SECTIONS:
                section = s
                continue
            if s and section in ("Caused by:", "Command error:", "Command output:"):
                errors.append(s)
        work = re.search(r"Work dir:\s*\n\s*(\S+)", text)
        f = TaskFailure(process=proc, tag=tag, exit_code=exit_code, error_lines=errors[-40:], source="nextflow-log")
        f.work_dir = work.group(1) if work else ""  # type: ignore[attr-defined]
        out.append(f)
    # Lines outside error blocks that matter (e.g. Batch reasons logged by the executor)
    reasons = [ln for ln in lines if re.search(r"Host EC2 .* terminated|SpotInterruption|CannotPullContainer|"
                                                r"OutOfMemoryError: Container killed", ln)]
    if reasons:
        for f in out:
            f.status_reason = f.status_reason or reasons[-1].split("]")[-1].strip()
    return out


def read_task_files(work_dir: str | Path, max_lines: int = 40) -> list[str]:
    """Tail of .command.err / .command.log in a task's work directory, if it is local."""
    d = Path(work_dir)
    lines: list[str] = []
    for name in (".command.err", ".command.log"):
        p = d / name
        if p.is_file():
            lines += p.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:]
    return lines


def merge(trace: list[TaskFailure], log: list[TaskFailure]) -> list[TaskFailure]:
    """Combine trace rows (resources) with log blocks (error text), matched on process + tag."""
    by_key = {(f.process, f.tag): f for f in trace}
    for lf in log:
        tf = by_key.get((lf.process, lf.tag)) or next((t for t in trace if t.process == lf.process), None)
        if tf is None:
            by_key[(lf.process, lf.tag)] = lf
            continue
        tf.error_lines = tf.error_lines or lf.error_lines
        tf.exit_code = tf.exit_code if tf.exit_code is not None else lf.exit_code
        tf.status_reason = tf.status_reason or lf.status_reason
        if getattr(lf, "work_dir", ""):
            tf.work_dir = lf.work_dir  # type: ignore[attr-defined]
    return list(by_key.values())
