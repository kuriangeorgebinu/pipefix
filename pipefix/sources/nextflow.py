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


def split_name(name: str) -> tuple[str, str]:
    """'PROC:SUB (sample1)' -> ('PROC:SUB', 'sample1')."""
    m = _NAME_TAG.match(name or "")
    return (m.group("proc").strip(), (m.group("tag") or "").strip()) if m else (name or "", "")


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
            proc, tag = split_name(row.get("name", ""))
            out.append(TaskFailure(
                process=(row.get("process") or proc).strip(),
                tag=(row.get("tag") or tag).strip(),
                exit_code=_int(row.get("exit")),
                memory_bytes=parse_memory(row.get("memory")),
                peak_rss_bytes=parse_memory(row.get("peak_rss")),
                time_ms=parse_duration(row.get("time")),
                realtime_ms=parse_duration(row.get("realtime")),
                source="nextflow-trace",
                hash=(row.get("hash") or "").strip(),
            ))
    return out


_ERR_HEAD = re.compile(r"Error executing process > '(?P<name>[^']+)'")
_EXIT = re.compile(r"terminated with an error exit status \((?P<code>-?\d+)\)")
_SECTIONS = ("Caused by:", "Command executed:", "Command exit status:", "Command output:", "Command error:",
             "Work dir:", "Tip:", "Container:")
# Specific platform reasons only ("Essential container in task exited" says nothing about why)
_PLATFORM_REASON = re.compile(r"Host EC2 \(instance [^)]+\) terminated|SpotInterruption|CannotPullContainer\w*:?[^\n]*|"
                              r"OutOfMemoryError: Container killed[^\n]*")
_VERSION = re.compile(r"N E X T F L O W\s+~\s+version\s+(\d+\.\d+\.\d+)|Nextflow version\s+(\d+\.\d+\.\d+)")


def nextflow_version(text: str) -> str:
    m = _VERSION.search(text or "")
    return (m.group(1) or m.group(2)) if m else ""


def parse_log_text(text: str) -> list[TaskFailure]:
    """Failed tasks described in Nextflow log text, each with only its OWN error lines and platform reason."""
    lines = text.splitlines()
    out: list[TaskFailure] = []
    i = 0
    while i < len(lines):
        head = _ERR_HEAD.search(lines[i])
        if not head:
            i += 1
            continue
        proc, tag = split_name(head.group("name"))
        start, i = i, i + 1
        while i < len(lines) and not _ERR_HEAD.search(lines[i]) and not lines[i].startswith(("ERROR ~", "Execution cancelled")):
            i += 1
        block = lines[start + 1:i]
        text_block = "\n".join(block)
        code = _EXIT.search(text_block)
        exit_code = int(code.group("code")) if code else None
        if exit_code is None:
            m = re.search(r"Command exit status:\s*\n\s*(-?\d+)", text_block)
            exit_code = int(m.group(1)) if m else None
        errors, section = [], None
        for ln in block:
            s = ln.strip()
            if s in _SECTIONS:
                section = s
                continue
            if s and section in ("Caused by:", "Command error:", "Command output:"):
                errors.append(s)
        work = re.search(r"Work dir:\s*\n\s*(\S+)", text_block)
        # A platform reason belongs to this task only if it is logged for this task: inside the block, or on a
        # line before it that names the same task.
        reason = next((m.group(0) for ln in block if (m := _PLATFORM_REASON.search(ln))), "")
        if not reason:
            name = head.group("name")
            for ln in lines[max(0, start - 200):start]:
                if name in ln and (m := _PLATFORM_REASON.search(ln)):
                    reason = m.group(0)
        out.append(TaskFailure(process=proc, tag=tag, exit_code=exit_code, error_lines=errors[-40:],
                               status_reason=reason, source="nextflow-log", work_dir=work.group(1) if work else ""))
    return out


def parse_log(path: str | Path) -> list[TaskFailure]:
    return parse_log_text(Path(path).read_text(encoding="utf-8", errors="replace"))


def read_task_files(work_dir: str | Path, max_lines: int = 40) -> list[str]:
    """Tail of .command.err / .command.log in a task's work directory, if it is local."""
    d = Path(work_dir)
    lines: list[str] = []
    for name in (".command.err", ".command.log"):
        p = d / name
        if p.is_file():
            with p.open(encoding="utf-8", errors="replace") as fh:
                fh.seek(0, 2)
                fh.seek(max(0, fh.tell() - 64 * 1024))
                lines += fh.read().splitlines()[-max_lines:]
    return lines


def merge(trace: list[TaskFailure], log: list[TaskFailure]) -> list[TaskFailure]:
    """Combine trace rows (resources) with log blocks (error text), matched on process + tag."""
    by_key = {(f.process, f.tag): f for f in trace}
    for lf in log:
        tf = by_key.get((lf.process, lf.tag)) or next((t for t in trace if t.process == lf.process and not t.tag), None)
        if tf is None:
            by_key[(lf.process, lf.tag)] = lf
            continue
        tf.error_lines = list(dict.fromkeys(tf.error_lines + lf.error_lines))
        tf.exit_code = tf.exit_code if tf.exit_code is not None else lf.exit_code
        tf.status_reason = tf.status_reason or lf.status_reason
        tf.work_dir = tf.work_dir or lf.work_dir
    return list(by_key.values())
