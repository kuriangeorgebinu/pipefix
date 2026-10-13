import shutil
import subprocess
from pathlib import Path

import pytest

from pipefix.classify import classify
from pipefix.cli import main
from pipefix.models import FailureClass, TaskFailure
from pipefix.patch import CONFIG_REL, INCLUDE_LINE, build_patch
from pipefix.report import pr_body
from pipefix.sources import aws, nextflow
from pipefix.units import parse_duration, parse_memory

EX = Path(__file__).resolve().parent.parent / "examples"
GB = 1024**3


def test_units():
    assert parse_memory("15.8 GB") == int(15.8 * GB)
    assert parse_memory("512 MB") == 512 * 1024**2
    assert parse_memory("-") is None
    assert parse_duration("1h 30m") == 90 * 60_000
    assert parse_duration("12m 3s") == 723_000


def test_trace_and_log_merge():
    failures = nextflow.merge(nextflow.parse_trace(EX / "oom/trace.txt"), nextflow.parse_log(EX / "oom/nextflow.log"))
    assert len(failures) == 1
    f = failures[0]
    assert f.simple_name == "GATK4_HAPLOTYPECALLER" and f.tag == "sample1"
    assert f.exit_code == 137 and f.memory_bytes == 16 * GB
    assert any("OutOfMemoryError" in ln for ln in f.error_lines)


@pytest.mark.parametrize("failure, expected", [
    (TaskFailure("P", exit_code=137, memory_bytes=16 * GB, peak_rss_bytes=int(15.9 * GB)), FailureClass.OUT_OF_MEMORY),
    (TaskFailure("P", exit_code=137, status_reason="Host EC2 (instance i-1) terminated."),
     FailureClass.SPOT_INTERRUPTION),
    (TaskFailure("P", exit_code=1, error_lines=["write error: No space left on device"]), FailureClass.DISK_FULL),
    (TaskFailure("P", exit_code=127, error_lines=["bash: STAR: command not found"]), FailureClass.CONTAINER),
    (TaskFailure("P", exit_code=1, error_lines=["[E::fai_load] Could not load fai index of ref.fa"]),
     FailureClass.REFERENCE),
    (TaskFailure("P", exit_code=140, time_ms=3_600_000, realtime_ms=3_600_000), FailureClass.TIMEOUT),
    (TaskFailure("P", exit_code=2, error_lines=["something odd"]), FailureClass.UNKNOWN),
])
def test_classify(failure, expected):
    assert classify(failure).cls == expected


def test_spot_beats_sigkill_without_memory_pressure():
    d = classify(TaskFailure("P", exit_code=137, memory_bytes=16 * GB, peak_rss_bytes=4 * GB,
                             status_reason="Host EC2 (instance i-1) terminated."))
    assert d.cls == FailureClass.SPOT_INTERRUPTION


def _copy(tmp_path, name):
    dst = tmp_path / name
    shutil.copytree(EX / name / "pipeline", dst)
    return dst


def test_oom_patch_is_scoped_and_idempotent(tmp_path):
    repo = _copy(tmp_path, "oom")
    failures = nextflow.merge(nextflow.parse_trace(EX / "oom/trace.txt"), nextflow.parse_log(EX / "oom/nextflow.log"))
    patch = build_patch([classify(f) for f in failures], repo)
    patch.apply(repo)
    cfg = (repo / CONFIG_REL).read_text()
    assert "withName: 'GATK4_HAPLOTYPECALLER'" in cfg and "{ 32.GB * task.attempt }" in cfg
    assert (repo / "nextflow.config").read_text().rstrip().endswith(INCLUDE_LINE)
    # running again changes nothing
    again = build_patch([classify(f) for f in failures], repo)
    assert again.empty


def test_spot_patch_sets_batch_retries(tmp_path):
    repo = _copy(tmp_path, "spot")
    diags = [classify(f) for f in aws.batch_failures_from_json(aws.load_json(EX / "spot/batch-jobs.json"))]
    patch = build_patch(diags, repo)
    cfg = patch.files[CONFIG_REL][1]
    assert "aws.batch.maxSpotAttempts = 3" in cfg
    assert "Integer.MAX_VALUE ? 'retry' : 'finish' }\n" in cfg
    assert cfg.count("{") == cfg.count("}")


def test_mixed_only_fixes_what_config_can_fix(tmp_path):
    repo = _copy(tmp_path, "mixed")
    doc = aws.load_json(EX / "mixed/omics.json")
    diags = [classify(f) for f in aws.omics_failures_from_json(doc["run"], doc["tasks"])]
    assert [d.cls for d in diags] == [FailureClass.OUT_OF_MEMORY, FailureClass.REFERENCE, FailureClass.CONTAINER]
    patch = build_patch(diags, repo)
    assert [f.name for f in patch.fixes] == ["SAMTOOLS_SORT"]
    assert {d.failure.simple_name for d in patch.advisories} == {"GATK4_BASERECALIBRATOR", "MULTIQC"}
    body = pr_body(diags, patch)
    assert "Needs a person" in body and "incompatible contigs" in body


def test_cli_apply_and_commit(tmp_path, capsys):
    repo = _copy(tmp_path, "oom")
    for args in (["init", "-q", "-b", "main"], ["add", "-A"],
                 ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"]):
        subprocess.run(["git", *args], cwd=repo, check=True)
    rc = main(["fix", "--trace", str(EX / "oom/trace.txt"), "--log", str(EX / "oom/nextflow.log"),
               "--repo", str(repo), "--apply"])
    assert rc == 0
    assert (repo / CONFIG_REL).exists()
    assert "Wrote:" in capsys.readouterr().out
