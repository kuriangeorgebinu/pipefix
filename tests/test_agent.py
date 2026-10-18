"""The agent: five enforced steps, a command dictionary, evidence checks, read-only AWS, redaction.
The model is scripted and AWS is faked; nothing leaves the machine."""
import io
import json

import pytest

from pipefix import keys
from pipefix.agent import COMMANDS, PHASES, Agent, to_json
from pipefix.aws.session import Clients
from pipefix.report import body, pr_title

ACCT = "522921672742"
BUCKET = f"pipefix-demo-data-{ACCT}-us-east-1"


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


@pytest.fixture(autouse=True)
def ai_keys(monkeypatch):
    for v in ("SERPER_API_KEY", "PIPEFIX_LLM_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("SERPER_API_KEY", "serper-test-key-0001")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-llm-key-0002")
    monkeypatch.setattr(keys, "_FILE", {})


class FakeAws:
    def __init__(self):
        self.seen = []

    def factory(self, service):
        aws = self

        class C:
            def __getattr__(self, op):
                def call(**kw):
                    aws.seen.append((service, op, kw))
                    return aws.reply(service, op, kw)
                return call
        return C()

    def reply(self, service, op, kw):
        if (service, op) == ("omics", "get_run"):
            return {"id": kw["id"], "status": "FAILED", "failureReason": "RUN_TASK_FAILED", "engine": "NEXTFLOW",
                    "roleArn": f"arn:aws:iam::{ACCT}:role/pipefix-demo-omics-run", "outputUri": f"s3://{BUCKET}/runs/"}
        if (service, op) == ("omics", "list_run_tasks"):
            return {"items": [{"taskId": "6130366", "name": "DEMO_GENOMICS:SORT_READS (sample1)", "status": "FAILED",
                               "cpus": 2, "memory": 2}]}
        if (service, op) == ("logs", "get_log_events"):
            return {"events": [{"message": "[sort_reads] allocating a 12 GB sort buffer"},
                               {"message": "MemoryError"}]}
        if (service, op) == ("s3", "get_object"):
            return {"Body": io.BytesIO(b"sort_reads.py reads.fq out.fq --buffer-gb 12\n")}
        return {}


DOC_AWS = ("<main><p>Container resource allocations are hard limits. If a task runs out of memory, the task generates "
           "an error log and exits.</p><p>HealthOmics reserves a small amount of memory (5%) for management and "
           "logging agents.</p></main>")
DOC_NF = ("<main><p>The errorStrategy directive allows you to define how an error condition is managed by the "
          "process.</p><p>Use task.attempt to increase the memory on each retry, together with maxRetries.</p></main>")
Q_MEM = "Container resource allocations are hard limits. If a task runs out of memory, the task generates an error log and exits."
Q_ES = "The errorStrategy directive allows you to define how an error condition is managed by the process."
Q_MR = "Use task.attempt to increase the memory on each retry, together with maxRetries."
Q_RES = "HealthOmics reserves a small amount of memory (5%) for management and logging agents."

GOOD_CANDIDATES = [
    {"id": "A", "title": "Size memory to the 12 GB buffer and retry exit 1", "process": "SORT_READS",
     "settings": {"memory": "{ 15.GB * task.attempt }",
                  "errorStrategy": "{ task.exitStatus in ((130..145) + 104 + 1) ? 'retry' : 'finish' }",
                  "maxRetries": "2"},
     "rationale": "12 GB buffer + 25%", "pros": "passes on attempt 1", "cons": "bigger instance",
     "evidence": [{"setting": "memory", "source": 1, "quote": Q_MEM}, {"setting": "errorStrategy", "source": 2, "quote": Q_ES},
                  {"setting": "maxRetries", "source": 2, "quote": Q_MR}]},
    {"id": "B", "title": "Lower params.sort_buffer_gb instead", "process": "SORT_READS", "settings": {},
     "rationale": "smaller buffer fits 2 GB", "pros": "cheap", "cons": "slower sort, changes the tool",
     "evidence": [{"source": 1, "quote": Q_RES}]},
]


def script():
    """The model's turns, phase by phase: (tool name, args)."""
    return [
        [("omics_get_run", {"id": "5472488"})],
        [("omics_list_run_tasks", {"id": "5472488", "status": "FAILED"}),
         ("logs_get_log_events", {"logGroupName": "/aws/omics/WorkflowLog", "logStreamName": "run/5472488/task/6130366"})],
        [("web_search", {"query": "too early", "site": "aws"})],                       # wrong phase: refused
        [("s3_get_task_file", {"Bucket": BUCKET.replace(ACCT, "<account-id>"), "Key": "work/ab/cdef/reads.fq"})],  # data
        [("s3_get_task_file", {"Bucket": BUCKET.replace(ACCT, "<account-id>"), "Key": "work/ab/cdef/.command.sh"})],
        [("repo_read_file", {"path": "main.nf"}), ("repo_trace_setting", {"process": "SORT_READS"})],
        [("finish_debug", {"process": "SORT_READS", "failure_class": "out_of_memory", "exit_code": 1, "root_cause": "x",
                           "evidence": [{"call": 3, "excerpt": "Killed by the OOM killer"}]})],          # made-up: refused
        [("finish_debug", {"process": "SORT_READS", "tag": "sample1", "failure_class": "out_of_memory", "exit_code": 1,
                           "memory_requested_gb": 2, "cpus": 2,
                           "root_cause": "The script allocates a 12 GB buffer but the task had 2 GB; Python raised MemoryError (exit 1).",
                           "evidence": [{"call": 3, "excerpt": '"lines": ["[sort_reads] allocating a 12 GB sort buffer", "MemoryError"'},
                                        {"call": 7, "excerpt": "--buffer-gb ${params.sort_buffer_gb}"}]})],
        [("web_search", {"query": "HealthOmics task out of memory s3://x/y 5472488", "site": "aws"})],
        [("web_search", {"query": "nextflow errorStrategy retry memory", "site": "nextflow"})],
        [("web_read_page", {"source": 1}), ("web_read_page", {"source": 2})],
        [("finish_research", {"candidates": GOOD_CANDIDATES})],
        [("finish_decision", {"chosen": "A", "reasons": "A sizes memory to what the script really allocates (12 GB) with "
                                                        "headroom, so attempt 1 passes; B changes the tool's behaviour.",
                              "why_not": {"B": "changes the tool and slows the sort"}, "risks": "bigger instance"})],
        [("implement_apply", {})],
        [("finish_implement", {"summary": "SORT_READS gets 15 GB per attempt and retries exit 1."})],
        [("mr_open", {})],
        [("finish_mr", {"summary": "Diff ready; MR not enabled in this run."})],
    ]


class ScriptedLLM:
    def __init__(self, turns):
        self.turns, self.requests, self.i, self.searches = turns, [], 0, []

    def post_json(self, url, payload, headers, timeout=120):
        if "serper" in url:
            self.searches.append(payload["q"])
            site = "docs.aws.amazon.com" if "aws" in payload["q"] else "www.nextflow.io"
            return {"organic": [{"title": f"Doc {site}", "link": f"https://{site}/doc", "snippet": "s"}]}
        self.requests.append(payload)
        turn = self.turns[self.i]
        self.i += 1
        return {"choices": [{"message": {"content": "", "tool_calls": [
            {"id": f"t{self.i}-{j}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
            for j, (n, a) in enumerate(turn)]}}]}

    def get_text(self, url, timeout=15):
        return DOC_AWS if "aws" in url else DOC_NF


def run_agent(repo, turns=None, **kw):
    aws = FakeAws()
    llm = ScriptedLLM(turns or script())
    agent = Agent(Clients(aws.factory, region="us-east-1"), repo, {"omics_run": "5472488"}, http=llm, **kw)
    return agent.run(), aws, llm


def test_agent_runs_the_five_steps_in_order_and_builds_the_fix(repo):
    res, aws, llm = run_agent(repo)
    assert res.error == "" and res.done, res.error
    assert [s.phase for s in res.steps] == sorted([s.phase for s in res.steps], key=PHASES.index)  # never goes back
    # the agent made the AWS calls itself
    assert [(s, o) for s, o, _ in aws.seen] == [("omics", "get_run"), ("omics", "list_run_tasks"),
                                                 ("logs", "get_log_events"), ("s3", "get_object")]
    assert res.inv.calls == ["omics:get_run", "omics:list_run_tasks", "logs:get_log_events", "s3:get_object"]
    # the account id never reached the model, but was put back for the S3 call
    assert ACCT not in json.dumps(llm.requests)
    assert aws.seen[3][2]["Bucket"] == BUCKET and aws.seen[3][2]["Range"] == "bytes=-65536"
    # rules: wrong-phase call, data file on S3 and a made-up excerpt were all refused
    refused = [s for s in res.steps if not s.ok]
    assert [s.tool for s in refused] == ["web_search", "s3_get_task_file", "finish_debug"]
    assert res.refused == 3
    # the web query was cleaned of IDs and paths
    assert llm.searches[0] == "HealthOmics task out of memory site:docs.aws.amazon.com"
    # the chosen fix went through pipefix's checks into conf/pipefix.config
    new = res.patch.files["conf/pipefix.config"][1]
    assert "{ 15.GB * task.attempt }" in new and "+ 104 + 1)" in new
    assert res.decision["chosen"] == "A" and [c["valid"] for c in res.candidates] == [True, True]
    res.patch.agent = res
    text = body(res.diagnoses, res.patch, res.inv)
    assert "## How the agent worked" in text and "call #3 `logs_get_log_events`" in text
    assert "**chosen**" in text and "Not B: changes the tool" in text
    assert pr_title(res.patch).startswith("pipefix agent:")
    j = to_json(res)
    assert j["aws_calls"] == 4 and [p["id"] for p in j["phases"]] == list(PHASES)


def test_mr_step_uses_the_opener_only_when_allowed(repo):
    opened = []
    res, _, _ = run_agent(repo, allow_mr=True, mr_opener=lambda r: opened.append(r) or "https://github.com/o/r/pull/9")
    assert res.mr_url == "https://github.com/o/r/pull/9" and len(opened) == 1
    res2, _, _ = run_agent(repo)                       # not allowed: diff only
    assert res2.mr_url == "" and res2.patch is not None


def test_research_must_be_grounded_and_decision_must_pick_a_valid_candidate(repo):
    turns = script()
    i = next(k for k, t in enumerate(turns) if t[0][0] == "finish_research")
    bad = json.loads(json.dumps(GOOD_CANDIDATES))
    bad[0]["evidence"][0]["quote"] = "Always give sort tasks sixteen gigabytes of memory."       # not in the page
    bad[0]["settings"]["memory"] = "{ 'rm -rf /'.execute(); 15.GB }"                             # code injection
    turns[i:i] = [[("finish_research", {"candidates": bad[:1]})],                                # 1 candidate, invalid
                  [("finish_research", {"candidates": bad})]]                                    # A invalid, B valid: accepted
    turns[i + 3] = [("finish_decision", {"chosen": "A", "reasons": "x" * 50, "why_not": {"B": "y"}})]  # A failed checks
    turns.insert(i + 4, [("finish_decision", {"chosen": "B", "reasons": "B is the only candidate that passed every "
                                                                        "check, so it is the best available.",
                                               "why_not": {"A": "failed the checks"}})])
    res, _, _ = run_agent(repo, turns=turns)
    msgs = [s.summary for s in res.steps if not s.ok]
    assert any("give at least 2" in m for m in msgs)
    assert any("candidate A failed the checks" in m for m in msgs)
    assert res.decision.get("chosen") == "B"
    assert "not a pipeline config change" in res.error             # B has no config change: stops before the MR
    assert res.patch is None


def test_command_dictionary_is_read_only_and_phase_scoped():
    from pipefix.aws.session import READ_ONLY_PREFIXES
    for name, c in COMMANDS.items():
        assert c["phase"] in PHASES
        if c.get("aws"):
            assert c["aws"][1].startswith(READ_ONLY_PREFIXES), name
    for p in PHASES:
        assert sum(1 for c in COMMANDS.values() if c["phase"] == p and c.get("finish")) == 1


def test_stops_cleanly_when_a_step_cannot_finish(repo):
    turns = script()[:1] + [[("omics_get_run", {"id": "5472488"})]] * 40
    res, _, _ = run_agent(repo, turns=turns)
    assert "did not finish within its limits" in res.error and res.phase_reached == "debug" and res.patch is None


def test_citing_the_call_that_read_a_page_maps_to_that_source(repo):
    turns = script()
    i = next(k for k, t in enumerate(turns) if t[0][0] == "finish_research")
    cands = json.loads(json.dumps(GOOD_CANDIDATES))
    cands[0]["evidence"][0]["source"] = 13            # call #13 was web_read_page(source=1)
    turns[i] = [("finish_research", {"candidates": cands})]
    res, _, llm = run_agent(repo, turns=turns)
    assert res.done, res.error
    assert res.candidates[0]["evidence"][0]["n"] == 1
    # research outputs carry source numbers, not call numbers
    tool_msgs = [m["content"] for m in llm.requests[-1]["messages"] if m["role"] == "tool"]
    assert any('"cite_as_source": 1' in t for t in tool_msgs)
    assert all(not t.startswith("[call #") for t in tool_msgs if "cite_as_source" in t)


def test_quotes_must_be_about_the_setting_and_no_op_settings_are_dropped(repo):
    turns = script()
    i = next(k for k, t in enumerate(turns) if t[0][0] == "finish_research")
    a = {"id": "A", "title": "unlabelled quotes, a no-op and a selector prefix", "process": "SORT_READS",
         "settings": {"withName: 'SORT_READS' memory": "{ 15.GB * task.attempt }",          # prefix stripped
                      "maxRetries": "2",
                      "errorStrategy": "{ task.exitStatus in ((130..145) + 104) ? 'retry' : 'finish' }"},  # no-op
         "rationale": "r", "evidence": [{"source": 1, "quote": Q_MEM},                       # unlabelled, about memory
                                        {"source": 2, "quote": Q_MR},                        # unlabelled, about retries
                                        {"source": 1, "quote": "a sentence that is not on the page at all"}]}
    off = {"id": "C", "title": "memory backed only by an errorStrategy quote", "process": "SORT_READS",
           "settings": {"memory": "{ 15.GB * task.attempt }"}, "rationale": "r",
           "evidence": [{"setting": "memory", "source": 2, "quote": Q_ES}]}
    turns[i] = [("finish_research", {"candidates": [a, GOOD_CANDIDATES[1], off]})]
    turns[i + 1] = [("finish_decision", {"chosen": "A", "reasons": "A sizes memory to the 12 GB buffer the script "
                                                                   "allocates, with headroom, and retries once more.",
                                         "why_not": {"B": "changes the tool", "C": "failed the checks"}})]
    res, _, _ = run_agent(repo, turns=turns)
    assert res.done, res.error
    ca, cc = res.candidates[0], res.candidates[2]
    assert ca["valid"] and ca["settings"] == {"memory": "{ 15.GB * task.attempt }", "maxRetries": "2"}
    assert any("errorStrategy dropped: same as now" in n for n in ca["notes"])
    assert any("not used" in n for n in ca["notes"])
    assert [e["setting"] for e in ca["evidence"]] == ["memory", "memory, maxRetries"]
    assert not cc["valid"] and "memory has no verified quote about memory" in cc["problems"][0]
