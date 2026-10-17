"""AI-researched fixes: search -> sources -> LLM proposal -> pipefix's checks -> patch and MR with citations.
Serper and the LLM are stubbed; nothing leaves the machine in these tests."""
import json

import pytest

from pipefix import keys
from pipefix.configtrace import load_repo
from pipefix.investigation import Investigation
from pipefix.models import Diagnosis, FailureClass, TaskFailure
from pipefix.patch import build_patch
from pipefix.report import body, pr_title
from pipefix.research import Http, check, context, exit_codes, research

GB = 1024**3


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "main.nf").write_text(
        "process SORT_READS {\n    label 'process_low'\n    input:\n    path reads\n    script:\n"
        "    \"\"\"\n    sort_reads.py --buffer-gb ${params.sort_buffer_gb} $reads\n    \"\"\"\n}\n")
    (tmp_path / "conf").mkdir()
    (tmp_path / "conf" / "base.config").write_text(
        "process {\n    errorStrategy = { task.exitStatus in ((130..145) + 104) ? 'retry' : 'finish' }\n"
        "    maxRetries = 1\n    withLabel: 'process_low' {\n        memory = { 2.GB * task.attempt }\n    }\n}\n")
    (tmp_path / "nextflow.config").write_text("params {\n    sort_buffer_gb = 12\n}\nincludeConfig 'conf/base.config'\n")
    return tmp_path


@pytest.fixture
def failure():
    f = TaskFailure("DEMO_GENOMICS:SORT_READS", tag="sample1", exit_code=1, memory_bytes=2 * GB, source="healthomics",
                    error_lines=["Traceback (most recent call last):", "MemoryError",
                                 "s3://pipefix-demo-data-522921672742-us-east-1/runs/x"])
    return Diagnosis(f, FailureClass.OUT_OF_MEMORY, 0.88, ["log: MemoryError"], "Out of memory in SORT_READS.", "Raise memory")


@pytest.fixture
def ai_keys(monkeypatch):
    for v in ("SERPER_API_KEY", "PIPEFIX_LLM_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("SERPER_API_KEY", "serper-test-key-0001")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-llm-key-0002")
    monkeypatch.setattr(keys, "_FILE", {})


class FakeHttp(Http):
    def __init__(self, fix_reply):
        self.fix_reply, self.calls = fix_reply, []

    def post_json(self, url, payload, headers, timeout=120):
        self.calls.append((url, payload, headers))
        if "serper" in url:
            q = payload["q"]
            site = "docs.aws.amazon.com" if "aws" in q else "www.nextflow.io"
            return {"organic": [{"title": f"Doc for {q[:20]}", "link": f"https://{site}/{len(self.calls)}",
                                 "snippet": "memory and retries"}]}
        system = payload["messages"][0]["content"]
        content = ({"queries": [{"site": "aws", "q": "HealthOmics task out of memory s3://secret/path"},
                                {"site": "nextflow", "q": "nextflow errorStrategy exit status retry memory"}]}
                   if "search queries" in system else self.fix_reply)
        return {"choices": [{"message": {"content": json.dumps(content)}}]}

    def get_text(self, url, timeout=15):
        if "docs.aws" in url:
            return ("<html><nav>menu</nav><main><h1>Task memory</h1><p>The task container uses the number of vCPUs "
                    "and the memory size that you specify in your workflow definition file.</p>"
                    "<p>HealthOmics reserves a small amount of memory (5%) for management and logging agents.</p>"
                    "</main></html>")
        return ("<html><main><p>The errorStrategy directive allows you to define how an error condition is managed "
                "by the process.</p><p>Use task.attempt to increase the memory on each retry, together with "
                "maxRetries.</p></main></html>")


EV = [{"setting": "memory", "source": 1, "quote": "The task container uses the number of vCPUs and the memory size that "
                                                    "you specify in your workflow definition file."},
      {"setting": "errorStrategy", "source": 2, "quote": "The errorStrategy directive allows you to define how an error "
                                                           "condition is managed by the process."},
      {"setting": "maxRetries", "source": 2, "quote": "Use task.attempt to increase the memory on each retry, "
                                                        "together with maxRetries."}]
GOOD = {"summary": "SORT_READS asks for a 12 GB buffer but process_low gives 2 GB; exit 1 is not retried.",
        "fixes": [{"process": "SORT_READS",
                   "settings": {"memory": "{ 16.GB * task.attempt }",
                                "errorStrategy": "{ task.exitStatus in ((130..145) + 104 + 1) ? 'retry' : 'finish' }",
                                "maxRetries": "2"},
                   "evidence": EV, "rationale": "--buffer-gb 12 plus 25% headroom"}],
        "infrastructure": [{"title": "Agent memory", "detail": "Leave headroom.",
                            "evidence": [{"source": 1, "quote": "HealthOmics reserves a small amount of memory (5%) "
                                                                "for management and logging agents."}]}]}


def test_ai_fix_is_researched_checked_cited_and_written(repo, failure, ai_keys):
    cfg = load_repo(repo)
    inv = Investigation("healthomics", "HealthOmics run 5472488", facts={"healthomics": True})
    http = FakeHttp(GOOD)
    res = research(inv, [failure], cfg, http=http)
    assert not res.error and not res.rejected
    assert [s["n"] for s in res.sources] == [1, 2] and res.sources[0]["url"].startswith("https://docs.aws.amazon.com")
    assert all("s3://" not in q and "522921672742" not in q for q in res.queries)   # queries cleaned
    llm_payloads = [json.dumps(p) for u, p, h in http.calls if "chat/completions" in u]
    assert "--buffer-gb ${params.sort_buffer_gb}" in llm_payloads[0]                # the model sees the script
    assert "522921672742" not in "".join(llm_payloads)                              # account id redacted
    assert "reserves a small amount of memory" in llm_payloads[1] and "menu" not in llm_payloads[1]  # read pages
    patch = build_patch([failure], repo, inv=inv, cfg=cfg, ai=res)
    new = patch.files["conf/pipefix.config"][1]
    assert "memory        = { 16.GB * task.attempt }" in new and "+ 104 + 1)" in new
    assert "source [1] https://docs.aws.amazon.com/" in new                         # citations in the config
    assert patch.ai_used == ["SORT_READS"]
    assert any(fd.title.startswith("pipefix's rules would have") for fd in patch.findings)
    text = body([failure], patch, inv)
    assert "## AI research" in text and "**Sources**" in text and "1. [Doc for" in text
    assert 'source [1] says: "The task container uses the number of vCPUs' in text          # quotes in the MR
    assert res.fixes[0].sources == [1, 2] and len(res.fixes[0].evidence) == 3
    assert res.infrastructure[0]["evidence"][0]["n"] == 1
    assert pr_title(patch).startswith("pipefix (AI-researched)")


@pytest.mark.parametrize("settings,why", [
    ({"memory": "{ 'rm -rf /'.execute(); 16.GB }"}, "not in an accepted form"),       # code injection
    ({"memory": "{ 64.GB * task.attempt }"}, "more than the largest instance"),
    ({"errorStrategy": "{ task.exitStatus in (130..145) ? 'retry' : 'finish' }"}, "doesn't retry the exit code"),
    ({"errorStrategy": "{ System.exit(1) }"}, "accepted form"),
    ({"queue": "'spot'"}, "HealthOmics has no Batch queues"),
    ({"container": "'evil:latest'"}, "not a setting"),
])
def test_unsafe_ai_values_are_rejected_not_written(repo, failure, ai_keys, settings, why):
    cfg = load_repo(repo)
    inv = Investigation("healthomics", "run", facts={"healthomics": True, "max_task_memory_mib": 32 * 1024})
    ev = [{"setting": k, "source": 1, "quote": EV[0]["quote"]} for k in settings]
    reply = {"summary": "", "fixes": [{"process": "SORT_READS", "settings": settings, "rationale": "", "evidence": ev}]}
    res = research(inv, [failure], cfg, http=FakeHttp(reply))
    assert res.fixes == [] and len(res.rejected) == 1 and why in res.rejected[0]
    patch = build_patch([failure], repo, inv=inv, cfg=cfg, ai=res)
    assert patch.ai_used == [] and "execute" not in patch.diff() and "evil" not in patch.diff()
    assert "Rejected by pipefix's checks" in body([failure], patch, inv)


def test_strict_grounding_rejects_values_without_a_verified_quote(repo, failure, ai_keys):
    reply = {"fixes": [
        {"process": "SORT_READS", "settings": {"maxRetries": "2"}},                                    # no quote
        {"process": "SORT_READS", "settings": {"memory": "{ 16.GB * task.attempt }"},                  # made-up quote
         "evidence": [{"setting": "memory", "source": 1, "quote": "Always give sort tasks at least sixteen gigabytes."}]},
        {"process": "SORT_READS", "settings": {"cpus": "2"},                                           # paraphrase
         "evidence": [{"setting": "cpus", "source": 1, "quote": "The container uses the vCPUs you specify in the file."}]},
        {"process": "SORT_READS", "settings": {"time": "{ 2.h * task.attempt }"},                     # unknown source
         "evidence": [{"setting": "time", "source": 99, "quote": EV[0]["quote"]}]},
        {"process": "OTHER", "settings": {"maxRetries": "2"}, "evidence": EV}],
        "infrastructure": [{"title": "Bigger disks", "detail": "Use gp3.", "evidence": []}]}
    res = research(None, [failure], load_repo(repo), http=FakeHttp(reply))
    assert res.fixes == [] and res.infrastructure == []
    joined = "\n".join(res.rejected)
    assert "maxRetries = 2: no verified quote" in joined
    assert "quote is not in source [1]" in joined                       # made-up and paraphrased quotes
    assert "could not read" in joined                                   # source 99 doesn't exist
    assert "OTHER" in joined and "advice 'Bigger disks'" in joined


def test_quote_matching_ignores_case_whitespace_and_curly_quotes():
    from pipefix.research import verify_quote
    src = {"n": 3, "text": "HealthOmics reserves a small\namount of memory (5%) for management and logging agents."}
    assert verify_quote("healthomics reserves a small amount of memory (5%) for management", src) is None
    assert verify_quote("HealthOmics reserves", src) == "quote too short to support a claim"
    assert verify_quote("\u201cHealthOmics reserves a small amount of memory\u201d", {"n": 3, "text": "\u201cHealthOmics "
                        "reserves a small amount of memory\u201d for agents"}) is None             # curly quotes
    assert verify_quote('"HealthOmics reserves a small amount of memory"', {"n": 3, "text": "\u201cHealthOmics "
                        "reserves a small amount of memory\u201d"}) is None
    assert "could not read" in verify_quote("anything long enough to be a quote here", {"n": 4})


def test_unreadable_sources_cannot_ground_a_fix(repo, failure, ai_keys):
    class NoPages(FakeHttp):
        def get_text(self, url, timeout=15):
            raise OSError("blocked")
    res = research(None, [failure], load_repo(repo), http=NoPages(GOOD))
    assert res.fixes == [] and "could not read any" in res.error


def test_without_keys_falls_back_to_rules(repo, failure, monkeypatch):
    for v in ("SERPER_API_KEY", "PIPEFIX_LLM_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(keys, "_FILE", {})
    res = research(None, [failure], load_repo(repo), http=FakeHttp(GOOD))
    assert "needs a Serper key" in res.error
    patch = build_patch([failure], repo, cfg=load_repo(repo), ai=res)
    assert patch.ai_used == [] and patch.fixes                                       # rule-based fix still made
    assert "Not used:" in body([failure], patch)


def test_split_entries_for_one_process_are_merged(repo, failure, ai_keys):
    # seen live: the model put memory/retries and cpus for SORT_READS in two separate entries
    reply = {"fixes": [{"process": "SORT_READS", "settings": {"memory": "{ 15.GB * task.attempt }", "maxRetries": "2"},
                        "rationale": "12 GB buffer + 25%", "evidence": [EV[0], EV[2]]},
                       {"process": "SORT_READS", "settings": {"cpus": "{ 2 * task.attempt }"},
                        "rationale": "keep 2 cpus", "evidence": [{**EV[1], "setting": "cpus"}]}]}
    cfg = load_repo(repo)
    res = research(None, [failure], cfg, http=FakeHttp(reply))
    assert len(res.fixes) == 1 and res.fixes[0].sources == [1, 2]
    new = build_patch([failure], repo, cfg=cfg, ai=res).files["conf/pipefix.config"][1]
    assert "15.GB" in new and "cpus" in new and "maxRetries" in new


def test_exit_codes_and_check():
    assert exit_codes("((130..145) + 104 + 1)") == set(range(130, 146)) | {104, 1}
    assert exit_codes("[1, 137]") == {1, 137} and exit_codes("x") is None
    ctx = {"failed_tasks": [{"process": "P", "exit_code": 137}], "platform": {}}
    assert check("P", "memory", "{ [16.GB * task.attempt, 30.GB].min() }", ctx) is None
    assert check("P", "time", "{ 4.h * task.attempt }", ctx) is None
    assert "range" in check("P", "time", "{ 400.h * task.attempt }", ctx)


def test_env_file_keys_are_reused_and_redacted(tmp_path, monkeypatch):
    from pipefix.report import redact
    for v in ("SERPER_API_KEY", "PIPEFIX_LLM_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_MODEL"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(keys, "_FILE", {})
    crew = tmp_path / "glpal-review-crew"
    crew.mkdir()
    (crew / ".env").write_text('SERPER_API_KEY=serper-from-crew-123456\nCREW_PORT=8771\n'
                               'DEEPSEEK_API_KEY="sk-from-crew-abcdef123456"\nOTHER_SECRET=nope\n')
    assert keys.load_env_file(str(crew / ".env")) == ["SERPER_API_KEY", "DEEPSEEK_API_KEY"]
    assert keys.get("serper") == ("serper-from-crew-123456", "SERPER_API_KEY in glpal-review-crew/.env")
    key, base, model = keys.llm_endpoint()
    assert key == "sk-from-crew-abcdef123456" and base == "https://api.deepseek.com" and model == "deepseek-chat"
    st = keys.status()
    assert st["serper"]["hint"] == "3456" and "serper-from-crew" not in json.dumps(st)
    assert redact("boom sk-from-crew-abcdef123456") == "boom <token>"
    assert "OTHER_SECRET" not in json.dumps(keys._FILE)
