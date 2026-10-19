"""Prove the generated config with real Nextflow: it parses, the selectors match the processes (also when they
run inside nested workflows), and each process gets the memory / queue / retries strandline intended.

Skipped unless Nextflow is available (NEXTFLOW=/path/to/nextflow or `nextflow` on PATH). CI installs it.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from fakeaws import GATK_JOB, GATK_WORKDIR_FILES, STAR_JOB, batch_account
from strandline.investigate import investigate
from strandline.patch import build_patch

NXF = os.environ.get("NEXTFLOW") or shutil.which("nextflow")
pytestmark = pytest.mark.skipif(not NXF, reason="Nextflow not installed")
EX = Path(__file__).resolve().parent.parent / "examples"

PROBE = """
process GATK4_HAPLOTYPECALLER {
    label 'process_medium'
    output: stdout
    script: "echo GATK4_HAPLOTYPECALLER mem=${task.memory} retries=${task.maxRetries} strategy=${task.errorStrategy}"
}
process STAR_ALIGN {
    label 'process_high'
    output: stdout
    script: "echo STAR_ALIGN mem=${task.memory} queue=${task.queue} retries=${task.maxRetries}"
}
process MULTIQC {
    label 'process_single'
    output: stdout
    script: "echo MULTIQC mem=${task.memory}"
}
workflow ALIGN_STAR { main: STAR_ALIGN() | view }
workflow SAREK {
    main:
    GATK4_HAPLOTYPECALLER() | view
    ALIGN_STAR()
}
workflow { SAREK(); MULTIQC() | view }
"""


def test_generated_config_with_real_nextflow(tmp_path):
    repo = tmp_path / "pipe"
    shutil.copytree(Path(__file__).resolve().parent.parent / "strandline" / "demo" / "pipeline", repo)
    jobs = [GATK_JOB, STAR_JOB]
    inv, diags = investigate(batch_account(jobs, GATK_WORKDIR_FILES), batch_jobs=[j["jobId"] for j in jobs],
                             cloudtrail=False)
    patch = build_patch(diags, repo, inv=inv)
    patch.apply(repo)
    # run the probe with the repo's own config + Strandline's; let the local executor accept big requests
    (repo / "main.nf").write_text(PROBE)
    (repo / "local.config").write_text("executor { name = 'local'; memory = '2 TB'; cpus = 64 }\n"
                                       "process.executor = 'local'\nprocess.cpus = 1\n")
    env = {**os.environ, "NXF_HOME": str(tmp_path / ".nxf"), "NXF_ANSI_LOG": "false", "NXF_DISABLE_CHECK_LATEST": "true"}
    r = subprocess.run([NXF, "-q", "run", "main.nf", "-c", "local.config"], cwd=repo, capture_output=True,
                       text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    out = dict(re.findall(r"^(\w+) (.+)$", r.stdout, re.M))
    # GATK: 16 GB failed -> 32 GB on attempt 1, capped expression evaluates; base.config retry rule kept
    assert out["GATK4_HAPLOTYPECALLER"].startswith("mem=32 GB retries=2 ")
    # STAR (inside ALIGN_STAR inside SAREK): matched by simple name, moved to the On-Demand queue
    assert out["STAR_ALIGN"] == "mem=72 GB queue=ondemand-queue retries=2"
    # untouched process keeps the pipeline's own settings
    assert out["MULTIQC"] == "mem=6 GB"
    # and the global Spot setting parses
    c = subprocess.run([NXF, "config", "-flat"], cwd=repo, capture_output=True, text=True, env=env, timeout=120)
    assert c.returncode == 0, c.stderr
    assert "aws.batch.maxSpotAttempts = 3" in c.stdout
