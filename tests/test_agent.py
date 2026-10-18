"""The agent: five enforced steps, a command dictionary, evidence checks, derived numbers, the playbook,
decision voting, the reasoning trace, read-only AWS and redaction. The model is scripted and AWS is faked."""
import io
import json

import pytest

from pipefix import keys
from pipefix.agent import COMMANDS, PHASES, Agent, _in_phase, first_number, safe_eval, to_json
from pipefix.aws.session import Clients
from pipefix.facts import instance_for, memory_need, size_memory
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
        "    maxRetries = 1\n    withLabel: 'process_low' {\n        cpus = { 2 * task.attempt }\n"
        "        memory = { 2.GB * task.attempt }\n    }\n}\n")
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
            return {"id": kw["id"], "status": "FAILED", "failureReason": "RUN_TASK_FAILED", "workflowId": "1348298",
                    "roleArn": f"arn:aws:iam::{ACCT}:role/pipefix-demo-omics-run", "outputUri": f"s3://{BUCKET}/runs/"}
        if (service, op) == ("omics", "list_run_tasks"):
            return {"items": [{"taskId": "6130366", "name": "DEMO_GENOMICS:SORT_READS (sample1)", "status": "FAILED",
                               "cpus": 2, "memory": 2}]}
        if (service, op) == ("omics", "list_runs"):
            return {"items": [{"id": "999", "workflowId": "1348298", "status": "COMPLETED", "name": "ok-run",
                               "stopTime": "2026-10-10T10:00:00Z"}]}
        if (service, op) == ("logs", "get_log_events"):
            return {"events": [{"message": "[sort_reads] allocating a 12 GB sort buffer"}, {"message": "MemoryError"}]}
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

NEED_CALL = 9      # facts_memory_need in the script below
DERIV_15 = {"memory": {"formula": "ceil(need_gib * 1.15 / 0.95)", "inputs": {"need_gib": {"value": 12, "call": NEED_CALL}}}}

GOOD_CANDIDATES = [
    {"id": "A", "title": "Size memory to the 12 GiB buffer", "process": "SORT_READS",
     "settings": {"memory": "{ 15.GB * task.attempt }"}, "derivations": DERIV_15,
     "rationale": "12 GiB need + 15% headroom over the 5% reserve", "pros": "passes on attempt 1", "cons": "r.large",
     "evidence": [{"setting": "memory", "source": 1, "quote": Q_MEM}]},
    {"id": "B", "title": "Lower params.sort_buffer_gb instead", "process": "SORT_READS", "settings": {},
     "rationale": "smaller buffer fits 2 GB", "pros": "cheap", "cons": "slower sort, changes the tool",
     "evidence": [{"source": 1, "quote": Q_RES}]},
]
DECISION_A = {"chosen": "A", "reasons": "A sizes memory to what the script really allocates (12 GiB) with headroom, "
                                        "so attempt 1 passes; B changes the tool's behaviour.",
              "why_not": {"B": "changes the tool and slows the sort"}, "risks": "bigger instance"}


def R(name, args, reason="checking"):
    return (name, {**args, "reason": reason})


def script():
    return [
        [R("omics_get_run", {"id": "5472488"}, "start from the run: status and failure reason")],                     # 1
        [R("omics_list_run_tasks", {"id": "5472488", "status": "FAILED"}),                                              # 2
         R("logs_get_log_events", {"logGroupName": "/aws/omics/WorkflowLog", "logStreamName": "run/5472488/task/6130366"})],  # 3
        [R("web_search", {"query": "too early", "site": "aws"})],                                                       # 4 refused
        [R("s3_get_task_file", {"Bucket": BUCKET.replace(ACCT, "<account-id>"), "Key": "work/ab/cdef/reads.fq"})],    # 5 refused
        [R("s3_get_task_file", {"Bucket": BUCKET.replace(ACCT, "<account-id>"), "Key": "work/ab/cdef/.command.sh"})],  # 6
        [R("repo_read_file", {"path": "main.nf"}), R("repo_trace_setting", {"process": "SORT_READS"}),                 # 7, 8
         R("facts_memory_need", {"process": "SORT_READS"}, "how much memory the command itself asks for")],             # 9
        [R("finish_debug", {"process": "SORT_READS", "failure_class": "out_of_memory", "exit_code": 1, "root_cause": "x",
                             "evidence": [{"call": 3, "excerpt": "Killed by the OOM killer"}]})],                        # 10 refused
        [R("finish_debug", {"process": "SORT_READS", "tag": "sample1", "failure_class": "out_of_memory", "exit_code": 1,
                             "memory_requested_gb": 2, "cpus": 2,
                             "root_cause": "The script allocates a 12 GB buffer but the task had 2 GB; Python raised MemoryError (exit 1).",
                             "evidence": [{"call": 3, "excerpt": '"lines": ["[sort_reads] allocating a 12 GB sort buffer", "MemoryError"'},
                                          {"call": 7, "excerpt": "--buffer-gb ${params.sort_buffer_gb}"}]})],          # 11
        [R("web_search", {"query": "HealthOmics task out of memory s3://x/y 5472488", "site": "aws"})],                # 12
        [R("web_search", {"query": "nextflow errorStrategy retry memory", "site": "nextflow"})],                        # 13
        [R("web_read_page", {"source": 1}), R("web_read_page", {"source": 2}),                                          # 14, 15
         R("facts_size_memory", {"need_gib": 12, "cpus": 2}, "playbook size for a 12 GiB need")],                       # 16
        [R("finish_research", {"candidates": GOOD_CANDIDATES})],                                                       # 17
        [R("finish_decision", DECISION_A)],                                                                             # 18
        [R("implement_apply", {})],
        [R("finish_implement", {"summary": "SORT_READS gets 15 GiB, sized to its 12 GiB buffer."})],
        [R("mr_open", {})],
        [R("finish_mr", {"summary": "Diff ready; MR not enabled in this run."})],
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
        return {"choices": [{"message": {"content": f"Thinking about step {self.i}.", "tool_calls": [
            {"id": f"t{self.i}-{j}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
            for j, (n, a) in enumerate(turn)]}}]}

    def get_text(self, url, timeout=15):
        return DOC_AWS if "aws" in url else DOC_NF


def run_agent(repo, turns=None, samples=1, **kw):
    aws = FakeAws()
    llm = ScriptedLLM(turns or script())
    agent = Agent(Clients(aws.factory, region="us-east-1"), repo, {"omics_run": "5472488"}, http=llm,
                  decision_samples=samples, **kw)
    return agent.run(), aws, llm


def research_turn(turns):
    return next(k for k, t in enumerate(turns) if t[0][0] == "finish_research")


def first_research(res):
    return next(s for s in res.steps if s.tool == "finish_research")


# ---------- the facts ----------

def test_facts():
    assert memory_need("sort_reads.py --buffer-gb ${params.b}", {"b": "12"})["need_gib"] == 12
    assert memory_need("gatk --java-options '-Xmx30g' X")["need_gib"] == 30
    assert memory_need("samtools sort -@ 4 -m 2G in.bam")["need_gib"] == 8
    assert memory_need("echo hi")["need_gib"] is None
    assert instance_for(2, 15)["instance"] == "omics.r.large" and instance_for(2, 15)["usable_memory_gib"] == 14.25
    assert instance_for(4, 16)["instance"] == "omics.m.xlarge"          # more vCPUs moves the instance
    assert instance_for(2, 2000)["fits"] is False
    assert size_memory(12, 2)["request_gib"] == 15
    assert safe_eval("ceil(need * 1.15 / 0.95)", {"need": 12}) == 15
    for bad in ("__import__('os').system('x')", "open('f')", "need.real", "[1][0]"):
        with pytest.raises(ValueError):
            safe_eval(bad, {"need": 12})
    assert first_number("memory", "{ [16.GB * task.attempt, 30.GB].min() }") == 16


# ---------- the full run ----------

def test_agent_runs_the_five_steps_in_order_and_builds_a_derived_fix(repo):
    res, aws, llm = run_agent(repo)
    assert res.error == "" and res.done, res.error
    assert [s.phase for s in res.steps] == sorted([s.phase for s in res.steps], key=PHASES.index)
    assert [(s, o) for s, o, _ in aws.seen] == [("omics", "get_run"), ("omics", "list_run_tasks"),
                                                 ("logs", "get_log_events"), ("s3", "get_object")]
    assert ACCT not in json.dumps(llm.requests)
    assert aws.seen[3][2]["Bucket"] == BUCKET and aws.seen[3][2]["Range"] == "bytes=-65536"
    assert [s.tool for s in res.steps if not s.ok] == ["web_search", "s3_get_task_file", "finish_debug"]
    assert llm.searches[0] == "HealthOmics task out of memory site:docs.aws.amazon.com"
    new = res.patch.files["conf/pipefix.config"][1]
    assert "{ 15.GB * task.attempt }" in new
    assert "resourceLimits = [ memory: 15.GB ]" in new        # no retry of exit 1: capped at the first attempt
    assert "errorStrategy" not in new and "cpus" not in new
    a = res.candidates[0]
    assert a["derivations"]["memory"]["formula"] == "ceil(need_gib * 1.15 / 0.95)"
    assert any("omics.r.large" in n and "14.25" in n for n in a["notes"])
    assert a["evidence"][0]["setting"] == "memory"
    res.patch.agent = res
    text = body(res.diagnoses, res.patch, res.inv)
    assert "## How the agent worked" in text and "ceil(need_gib * 1.15 / 0.95)" in text
    assert pr_title(res.patch).startswith("pipefix agent:")
    assert to_json(res)["aws_calls"] == 4


def test_reasoning_trace_records_thoughts_reasons_and_results(repo):
    events = []
    res, _, llm = run_agent(repo, on_event=events.append)
    kinds = [e["kind"] for e in res.trace]
    assert kinds[0] == "phase" and "thought" in kinds and "call" in kinds and "result" in kinds
    assert events == res.trace                                   # streamed live as well
    call = next(e for e in res.trace if e["kind"] == "call" and e["tool"] == "facts_memory_need")
    assert call["text"] == "how much memory the command itself asks for" and call["n"] == NEED_CALL
    refused = next(e for e in res.trace if e["kind"] == "result" and not e["ok"])
    assert refused["text"].startswith("REFUSED")
    assert all("reason" not in s.args for s in res.steps)       # the reason is not passed to the command
    assert res.steps[0].reason == "start from the run: status and failure reason"
    schema = llm.requests[0]["tools"][0]["function"]["parameters"]
    assert "reason" in schema["required"]
    assert [e["text"] for e in res.trace if e["kind"] == "phase"] == ["1. Debug", "2. Research solutions",
                                                                      "3. Pick the best solution", "4. Implement it",
                                                                      "5. Open the merge request"]


def test_mr_step_uses_the_opener_only_when_allowed(repo):
    opened = []
    res, _, _ = run_agent(repo, allow_mr=True, mr_opener=lambda r: opened.append(r) or "https://github.com/o/r/pull/9")
    assert res.mr_url == "https://github.com/o/r/pull/9" and len(opened) == 1
    res2, _, _ = run_agent(repo)
    assert res2.mr_url == "" and res2.patch is not None


# ---------- derived numbers ----------

@pytest.mark.parametrize("deriv,why", [
    (None, "needs a derivation"),
    ({"formula": "need_gib * 1.15 / 0.95", "inputs": {"need_gib": {"value": 12, "call": NEED_CALL}}}, "the formula gives 14.5"),
    ({"formula": "ceil(need_gib * 1.15 / 0.95)", "inputs": {"need_gib": {"value": 13, "call": NEED_CALL}}}, "not shown in the output"),
    ({"formula": "ceil(need_gib * 1.15 / 0.95)", "inputs": {"need_gib": {"value": 12, "call": 77}}}, "not shown in the output"),
    ({"formula": "__import__('os').getpid()", "inputs": {}}, "cannot be evaluated"),
])
def test_numbers_must_be_derived_from_traced_inputs(repo, deriv, why):
    turns = script()
    i = research_turn(turns)
    a = {**GOOD_CANDIDATES[0], "derivations": {"memory": deriv} if deriv else {}}
    res, _, _ = run_agent(repo, turns=turns[:i] + [[R("finish_research", {"candidates": [a, GOOD_CANDIDATES[1]]})]] * 4)
    first = first_research(res)
    assert not first.ok and why in first.output, first.output


# ---------- the playbook ----------

@pytest.mark.parametrize("settings,derivs,why", [
    ({"memory": "{ 15.GB * task.attempt }",
      "errorStrategy": "{ task.exitStatus in ((130..145) + 104 + 1) ? 'retry' : 'finish' }"}, DERIV_15,
     "exit 1 comes from the tool itself"),                                           # no retry for a deterministic error
    ({"memory": "{ 15.GB * task.attempt }", "maxRetries": "3"}, DERIV_15, "maxRetries has no effect"),
    ({"memory": "{ 15.GB * task.attempt }",
      "errorStrategy": "{ task.exitStatus in (130..140) ? 'retry' : 'finish' }"}, DERIV_15, "drops codes"),
    ({"memory": "{ 13.GB * task.attempt }"},
     {"memory": {"formula": "ceil(need_gib + 1)", "inputs": {"need_gib": {"value": 12, "call": NEED_CALL}}}},
     "would run out again"),                                                          # 13 * 0.95 < 12 * 1.05
    ({"memory": "{ 15.GB * task.attempt }", "cpus": "4"}, DERIV_15, "cpus = 4: a number needs a derivation"),
])
def test_playbook_rules(repo, settings, derivs, why):
    turns = script()
    i = research_turn(turns)
    a = {**GOOD_CANDIDATES[0], "settings": settings, "derivations": derivs}
    res, _, _ = run_agent(repo, turns=turns[:i] + [[R("finish_research", {"candidates": [a, GOOD_CANDIDATES[1]]})]] * 4)
    first = first_research(res)
    assert not first.ok and why in first.output, first.output


def test_retry_is_allowed_for_a_transient_kill(repo):
    turns = script()
    k = next(j for j, t in enumerate(turns) if t[0][0] == "finish_debug" and "tag" in t[0][1])
    turns[k] = [(n, {**a, "exit_code": 137}) for n, a in turns[k]]
    i = research_turn(turns)
    a = {**GOOD_CANDIDATES[0], "settings": {"memory": "{ 15.GB * task.attempt }", "maxRetries": "2"},
         "evidence": [{"setting": "maxRetries", "source": 2, "quote": Q_MR}]}
    turns[i] = [R("finish_research", {"candidates": [a, GOOD_CANDIDATES[1]]})]
    res, _, _ = run_agent(repo, turns=turns)
    assert res.done, res.error
    assert "resourceLimits = [ memory: 45.GB ]" in res.patch.files["conf/pipefix.config"][1]  # 15 * (1 + 2 retries)


# ---------- research and decision rules ----------

def test_research_must_be_grounded_and_decision_must_pick_a_valid_candidate(repo):
    turns = script()
    i = research_turn(turns)
    bad = json.loads(json.dumps(GOOD_CANDIDATES))
    bad[0]["settings"]["memory"] = "{ 'rm -rf /'.execute(); 15.GB }"
    c = json.loads(json.dumps(GOOD_CANDIDATES[0]))
    c["id"], c["settings"]["memory"] = "C", "{ 16.GB * task.attempt }"
    c["derivations"] = {"memory": {"formula": "ceil(need_gib * 1.3)", "inputs": {"need_gib": {"value": 12, "call": NEED_CALL}}}}
    turns[i:i + 2] = [[R("finish_research", {"candidates": bad[:1]})],
                      [R("finish_research", {"candidates": bad})],
                      [R("finish_research", {"candidates": bad + [c]})],
                      [R("finish_decision", {"chosen": "A", "reasons": "x" * 50, "why_not": {"B": "y", "C": "z"}})],
                      [R("finish_decision", {"chosen": "C", "reasons": "C sizes memory to the buffer and is the only "
                                                                        "config candidate that passed every check.",
                                              "why_not": {"A": "failed the checks", "B": "changes the tool"}})]]
    res, _, _ = run_agent(repo, turns=turns)
    msgs = [s.output for s in res.steps if not s.ok]
    assert any("give at least 2" in m for m in msgs)
    assert any("fixable by config" in m for m in msgs)
    assert any("candidate A failed the checks" in m for m in msgs)
    assert res.done and res.decision["chosen"] == "C"
    assert "16.GB" in res.patch.files["conf/pipefix.config"][1]


def test_quotes_no_op_settings_and_prefixes(repo):
    turns = script()
    i = research_turn(turns)
    a = {"id": "A", "title": "prefix, a no-op and unlabelled quotes", "process": "SORT_READS",
         "settings": {"withName: 'SORT_READS' memory": "{ 15.GB * task.attempt }",
                      "errorStrategy": "{ task.exitStatus in ((130..145) + 104) ? 'retry' : 'finish' }"},
         "derivations": DERIV_15, "rationale": "r",
         "evidence": [{"source": 1, "quote": Q_MEM}, {"setting": "memory", "source": 2, "quote": Q_ES},
                      {"source": 1, "quote": "a sentence that is not on the page at all"}]}
    turns[i] = [R("finish_research", {"candidates": [a, GOOD_CANDIDATES[1]]})]
    res, _, _ = run_agent(repo, turns=turns)
    assert res.done, res.error
    ca = res.candidates[0]
    assert ca["valid"] and ca["settings"] == {"memory": "{ 15.GB * task.attempt }"}
    assert any("errorStrategy dropped: same as now" in n for n in ca["notes"])
    assert any("not used" in n for n in ca["notes"])
    assert [e["setting"] for e in ca["evidence"]] == ["memory", "context"]   # Q_ES is not about memory


def test_citing_the_call_that_read_a_page_maps_to_that_source(repo):
    turns = script()
    i = research_turn(turns)
    cands = json.loads(json.dumps(GOOD_CANDIDATES))
    cands[0]["evidence"][0]["source"] = 14            # call #14 was web_read_page(source=1)
    turns[i] = [R("finish_research", {"candidates": cands})]
    res, _, llm = run_agent(repo, turns=turns)
    assert res.done, res.error
    assert res.candidates[0]["evidence"][0]["n"] == 1
    tool_msgs = [m["content"] for m in llm.requests[-1]["messages"] if m["role"] == "tool"]
    assert any('"cite_as_source": 1' in t for t in tool_msgs)


def test_a_non_config_choice_stops_cleanly_for_failures_config_cannot_fix(repo):
    turns = script()
    k = next(j for j, t in enumerate(turns) if t[0][0] == "finish_debug" and "tag" in t[0][1])
    turns[k] = [(n, {**a, "failure_class": "container_error"}) for n, a in turns[k]]
    i = next(j for j, t in enumerate(turns) if t[0][0] == "finish_decision")
    turns[i] = [R("finish_decision", {"chosen": "B", "reasons": "For a container error the image must be fixed; no "
                                                                 "config change helps here.",
                                       "why_not": {"A": "memory is not the problem"}})]
    res, _, _ = run_agent(repo, turns=turns)
    assert "not a pipeline config change" in res.error and res.patch is None


# ---------- voting ----------

def test_decision_votes_majority_wins_and_split_is_flagged(repo):
    b = {"chosen": "B", "reasons": "B keeps the instance small and avoids paying for a memory-optimised instance.",
         "why_not": {"A": "costs more"}}
    turns = script()
    i = next(j for j, t in enumerate(turns) if t[0][0] == "finish_decision")
    turns[i + 1:i + 1] = [[R("finish_decision", b)], [R("finish_decision", b)]]      # two more samples choose B
    res, _, _ = run_agent(repo, turns=turns, samples=3)
    assert res.decision["votes"] == {"A": 1, "B": 2} and res.decision["chosen"] == "B" and not res.decision["unstable"]
    assert "not a pipeline config change" in res.error                                 # B has no config change
    assert any(e["kind"] == "vote" and "B wins by majority" in e["text"] for e in res.trace)

    turns = script()
    turns[i + 1:i + 1] = [[R("finish_decision", b)], [R("finish_decision", {"chosen": "Z"})]]   # A, B, invalid
    res, _, _ = run_agent(repo, turns=turns, samples=3)
    assert res.decision["chosen"] == "A" and res.decision["unstable"] and res.done


def test_command_dictionary_is_read_only_and_phase_scoped():
    from pipefix.aws.session import READ_ONLY_PREFIXES
    for name, c in COMMANDS.items():
        assert c["phase"] in PHASES
        if c.get("aws"):
            assert c["aws"][1].startswith(READ_ONLY_PREFIXES), name
    for p in PHASES:
        assert sum(1 for c in COMMANDS.values() if c["phase"] == p and c.get("finish")) == 1
    assert _in_phase("facts_size_memory", "research") and not _in_phase("omics_get_run", "research")


def test_stops_cleanly_when_a_step_cannot_finish(repo):
    turns = script()[:1] + [[R("omics_get_run", {"id": "5472488"})]] * 40
    res, _, _ = run_agent(repo, turns=turns)
    assert "did not finish within its limits" in res.error and res.phase_reached == "debug" and res.patch is None


def test_benchmark_cases_build_and_score(tmp_path):
    from pipefix.agent import AgentResult
    from pipefix.bench import CASES, _reply, make_repo, score
    from pipefix.configtrace import load_repo
    for c in CASES:
        repo = make_repo(c, tmp_path / c.id)
        cfg = load_repo(repo)
        assert c.process in cfg.processes and cfg.trace(c.process).effective("memory")
        assert _reply(c, "omics", "get_task", {}) == {} and _reply(c, "omics", "list_run_tasks", {})["items"]
    good = AgentResult(decision={"chosen": "A"}, candidates=[{"id": "A", "settings": {"memory": "{ 15.GB * task.attempt }"}}])
    good.patch = object()
    assert score(CASES[0], good) | {} == score(CASES[0], good)
    r = score(CASES[0], good)
    assert r["value"] and r["justified"] and r["decision"]
    bad = AgentResult(decision={"chosen": "A"}, candidates=[{"id": "A", "settings": {"memory": "{ 20.GB * task.attempt }",
                                                                                     "cpus": "4"}}])
    bad.patch = object()
    r = score(CASES[0], bad)
    assert not r["value"] and not r["justified"]
