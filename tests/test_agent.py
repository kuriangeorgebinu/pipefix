"""The agents: steps in order, a generic read-only AWS tool the agent drives itself, verbatim evidence, derived
numbers, hard rules, the two expert reviewers and their agreement, redaction. The models are scripted and AWS is
faked; the AWS API reference is botocore's real one."""
import io
import json

import pytest

from pipefix import keys
from pipefix.agent import (COMMANDS, PHASES, Agent, _excerpt_in, _in_phase, _Refused, _shows, first_number, safe_eval,
                           to_json)
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
    def __init__(self, exit_code=1):
        self.seen, self.exit_code = [], exit_code

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
        if (service, op) == ("logs", "get_log_events"):
            return {"events": [{"message": "[sort_reads] allocating a 12 GB sort buffer"},
                               {"message": "MemoryError" if self.exit_code == 1 else "Killed"},
                               {"message": f"Command exit status: {self.exit_code}"}]}
        if (service, op) == ("s3", "get_object"):
            return {"Body": io.BytesIO(b"sort_reads.py reads.fq out.fq --buffer-gb 12\n")}
        return {}


DOC_AWS = ("<main><p>Container resource allocations are hard limits. If a task runs out of memory, the task generates "
           "an error log and exits.</p><p>HealthOmics reserves a small amount of memory (5%) for management and "
           "logging agents.</p><p>The omics.r.large instance has 2 vCPUs and 16 GiB of memory.</p><p>Include at least 15% "
           "headroom in allocated memory to account for background processes.</p></main>")
DOC_NF = ("<main><p>The errorStrategy directive allows you to define how an error condition is managed by the "
          "process.</p><p>Use task.attempt to increase the memory on each retry, together with maxRetries.</p>"
          "<p>Exit status 137 means the process was killed with SIGKILL, often by the out of memory killer, so a "
          "retry with more memory can pass.</p></main>")
Q_MEM = "Container resource allocations are hard limits. If a task runs out of memory, the task generates an error log and exits."
Q_RES = "HealthOmics reserves a small amount of memory (5%) for management and logging agents."
Q_INST = "The omics.r.large instance has 2 vCPUs and 16 GiB of memory."
Q_ES = "The errorStrategy directive allows you to define how an error condition is managed by the process."
Q_MR = "Use task.attempt to increase the memory on each retry, together with maxRetries."

LOG = 4          # the call that read the task log
HEADROOM = {"value": 0.15, "source": 1, "quote": "Include at least 15% headroom in allocated memory"}
DERIV_15 = {"memory": {"formula": "ceil(need_gib * (1 + headroom) / (1 - reserve))", "inputs": {
    "need_gib": {"value": 12, "call": LOG, "quote": "allocating a 12 GB sort buffer"}, "headroom": HEADROOM,
    "reserve": {"value": 0.05, "source": 1, "quote": "reserves a small amount of memory (5%)"}}}}

GOOD_CANDIDATES = [
    {"id": "A", "title": "Size memory to the 12 GiB buffer", "process": "SORT_READS",
     "settings": {"memory": "{ 15.GB * task.attempt }"}, "derivations": DERIV_15,
     "rationale": "12 GiB need + 15% headroom over the 5% reserve", "pros": "passes on attempt 1", "cons": "r.large",
     "evidence": [{"setting": "memory", "source": 1, "quote": Q_MEM}, {"setting": "memory", "source": 1, "quote": Q_INST}]},
    {"id": "B", "title": "Lower params.sort_buffer_gb instead", "process": "SORT_READS", "settings": {},
     "rationale": "smaller buffer fits 2 GB", "pros": "cheap", "cons": "slower sort, changes the tool",
     "evidence": [{"source": 1, "quote": Q_RES}]},
]
DECISION_A = {"chosen": "A", "reasons": "A sizes memory to what the script really allocates (12 GiB) with headroom, "
                                        "so attempt 1 passes; B changes the tool's behaviour.",
              "why_not": {"B": "changes the tool and slows the sort"}, "risks": "bigger instance"}

CHECK_ROOT = {"covers": "root_cause", "claim": "the 12 GB buffer does not fit 2 GB", "finding": "confirmed",
              "explanation": "the log shows it", "evidence": [{"call": LOG, "quote": "allocating a 12 GB sort buffer"}]}
CHECK_MEM = {"covers": "memory", "claim": "15 GB leaves room after the 5% reserve", "finding": "confirmed",
             "explanation": "the docs give the reserve", "evidence": [{"source": 1, "quote": Q_RES}]}


def R(name, args, reason="checking this"):
    return (name, {**args, "reason": reason})


def aws(service, op, params, reason="reading the run"):
    return R("aws_call", {"service": service, "operation": op, "params": params}, reason)


def script():
    return [
        # ---- primary: debug ----
        [R("aws_describe_api", {"service": "omics", "operation": "GetRun"}, "find the call that describes a run")],  # 1
        [aws("omics", "GetRun", {"id": "5472488"}, "start from the run: status and failure reason")],                  # 2
        [aws("omics", "ListRunTasks", {"id": "5472488", "status": "FAILED"}),                                          # 3
         aws("logs", "GetLogEvents", {"logGroupName": "/aws/omics/WorkflowLog",
                                      "logStreamName": "run/5472488/task/6130366"})],                                  # 4
        [aws("s3", "GetObject", {"Bucket": BUCKET.replace(ACCT, "<account-id>"), "Key": "work/ab/reads.fq"})],         # 5 refused
        [aws("secretsmanager", "GetSecretValue", {"SecretId": "x"})],                                                  # 6 refused
        [aws("omics", "StartRun", {"workflowId": "1"})],                                                               # 7 refused
        [aws("s3", "GetObject", {"Bucket": BUCKET.replace(ACCT, "<account-id>"), "Key": "work/ab/cdef/.command.sh"})], # 8
        [R("repo_read_file", {"path": "main.nf"}), R("repo_resolve_config", {"process": "SORT_READS"})],               # 9, 10
        [R("finish_debug", {"process": "SORT_READS", "failure_class": "out_of_memory", "exit_code": 1, "root_cause": "x",
                             "evidence": [{"call": LOG, "quote": "Killed by the OOM killer"}]})],                        # 11 refused
        [R("finish_debug", {"process": "SORT_READS", "tag": "sample1", "failure_class": "out_of_memory", "exit_code": 1,
                             "memory_requested_gb": 2, "cpus": 2,
                             "root_cause": "The script allocates a 12 GB buffer but the task had 2 GB; Python raised MemoryError (exit 1).",
                             "evidence": [{"call": LOG, "quote": "allocating a 12 GB sort buffer ... Command exit status: 1"},
                                          {"call": 9, "quote": "--buffer-gb ${params.sort_buffer_gb}"}]})],             # 12
        # ---- research ----
        [R("web_search", {"query": "HealthOmics task out of memory s3://x/y 5472488", "site": "aws"})],                # 13
        [R("web_search", {"query": "nextflow errorStrategy retry memory", "site": "nextflow"})],                        # 14
        [R("web_read_page", {"source": 1}), R("web_read_page", {"source": 2}),                                          # 15, 16
         R("calculate", {"expression": "ceil(n * 1.15 / 0.95)", "inputs": {"n": 12}})],                                # 17
        [R("finish_research", {"candidates": GOOD_CANDIDATES})],                                                       # 18
        [R("finish_decision", DECISION_A)],                                                                             # 19
        # ---- review round 1 ----
        [R("inspect_call", {"call": LOG}), R("web_read_page", {"source": 1}, "check the reserve myself")],            # 20, 21 (aws)
        [R("finish_review", {"verdict": "agree", "checks": [CHECK_ROOT, CHECK_MEM]})],                                 # 22
        [R("repo_read_file", {"path": "main.nf"}, "what does the command ask for")],                                   # 23 (nf)
        [R("finish_review", {"verdict": "object", "checks": [
            {**CHECK_ROOT, "evidence": [{"call": 23, "quote": "--buffer-gb ${params.sort_buffer_gb}"}]},
            {"covers": "memory", "claim": "15% headroom", "finding": "unsupported", "explanation": "why 15%?"}],
            "questions": ["Why is 15 GB enough after the platform reserve?"]})],                                         # 24
        [R("finish_response", {"action": "defend", "answers": [
            {"question": "Why is 15 GB enough after the platform reserve?", "answer": "15 GB minus the 5% reserve leaves 14.25 GB, above the 12 GB buffer.",
             "evidence": [{"source": 1, "quote": Q_RES}]}]})],                                                          # 25
        # ---- review round 2 ----
        [R("finish_review", {"verdict": "agree", "checks": [CHECK_ROOT, CHECK_MEM]})],                                 # 26
        [R("finish_review", {"verdict": "agree", "checks": [
            {**CHECK_ROOT, "evidence": [{"call": 23, "quote": "--buffer-gb ${params.sort_buffer_gb}"}]}, CHECK_MEM]})],  # 27
        # ---- implement / mr ----
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
            for j, (n, a) in enumerate(turn)]}}], "usage": {"prompt_tokens": 100, "completion_tokens": 10}}

    def get_text(self, url, timeout=15):
        return DOC_AWS if "aws" in url else DOC_NF


def run_agent(repo, turns=None, fake=None, aws_doc=None, **kw):
    fake = fake or FakeAws()
    llm = ScriptedLLM(turns or script())
    if aws_doc:
        llm.get_text = lambda url, timeout=15: aws_doc if "aws" in url else DOC_NF
    agent = Agent(Clients(fake.factory, region="us-east-1"), repo, {"omics_run": "5472488"}, http=llm, **kw)
    return agent.run(), fake, llm


def turn_of(turns, tool, nth=0):
    return [k for k, t in enumerate(turns) if t[0][0] == tool][nth]


def first(res, tool):
    return next(s for s in res.steps if s.tool == tool)


def with_research(candidates, tries=1):
    """The script up to research, then these candidates (repeated), the decision and the reviews."""
    turns = script()
    i = turn_of(turns, "finish_research")
    return turns[:i] + [[R("finish_research", {"candidates": candidates})]] * tries + turns[i + 1:]


def add_check(turns, check):
    """Every review also checks one more setting."""
    for j, t in enumerate(turns):
        if t[0][0] == "finish_review":
            turns[j] = [(n, {**a, "checks": a["checks"] + [check]}) for n, a in t]
    return turns


# ---------- helpers ----------

def test_arithmetic_numbers_and_quotes():
    assert safe_eval("ceil(need * 1.15 / (1 - r))", {"need": 12, "r": 0.05}) == 15
    for bad in ("__import__('os').system('x')", "open('f')", "need.real", "[1][0]", "True + 1"):
        with pytest.raises(ValueError):
            safe_eval(bad, {"need": 12})
    assert first_number("memory", "{ 15.GB * task.attempt }") == 15 and first_number("cpus", "4") == 4
    assert first_number("memory", "{ [16.GB * task.attempt, 30.GB].min() }") is None    # never a guess
    assert first_number("memory", "{ 2 * task.attempt }") is None
    assert _shows("allocating a 12 GB sort buffer", 12) and _shows("reserves (5%)", 0.05) and _shows("reserves (5%)", 0.95) and _shows("-Xmx30g", 30)
    assert not _shows("allocating a 12 GB sort buffer", 1.2) and not _shows("version 112", 12)


def test_evidence_may_join_verbatim_pieces_in_order_and_close_together():
    out = '{"lines": ["[sort_reads] allocating a 12 GB sort buffer", "buf = bytearray(size)", "MemoryError"]}'
    assert _excerpt_in("[sort_reads] allocating a 12 GB sort buffer ... MemoryError", out)
    assert _excerpt_in("allocating a 12 GB sort buffer … bytearray(size)", out)
    assert not _excerpt_in("MemoryError ... allocating a 12 GB sort buffer", out)        # wrong order
    assert not _excerpt_in("allocating a 16 GB sort buffer ... MemoryError", out)          # not verbatim
    assert not _excerpt_in("allocating ... Me", out)                                       # piece too short
    assert _excerpt_in("-m 2G", "samtools sort -@ 4 -m 2G -o x") and not _excerpt_in("-m", "-m 2G")
    far = '{"a": "allocating a buffer", "pad": "' + "x " * 400 + '", "b": "MemoryError"}'
    assert not _excerpt_in("allocating a buffer ... MemoryError", far)                     # spliced from far apart


def test_aws_policy_is_read_only_and_never_reads_secrets_or_data():
    pol = Agent._policy
    pol("omics", "GetRun", {"id": "1"})
    pol("logs", "FilterLogEvents", {"logGroupName": "g"})
    pol("s3", "GetObject", {"Bucket": "b", "Key": "work/ab/.command.log"})
    for svc, op, params in [("omics", "StartRun", {}), ("batch", "TerminateJob", {}), ("s3", "PutObject", {}),
                            ("secretsmanager", "GetSecretValue", {}), ("kms", "DescribeKey", {}),
                            ("sts", "GetCallerIdentity", {}), ("ecr", "GetAuthorizationToken", {}),
                            ("omics", "GetReadSet", {}), ("ssm", "GetParameter", {}),
                            ("s3", "GetObject", {"Key": "data/sample1.bam"}), ("logs", "GetLogEvents", {"unmask": True})]:
        with pytest.raises(_Refused):
            pol(svc, op, params)


def test_the_api_reference_tool_documents_operations_and_parameters(repo):
    agent = Agent(None, repo, {"omics_run": "1"})
    ops = agent._describe_api({"service": "omics"})["read_only_operations"]
    assert "GetRun" in ops and "GetRunTask" in ops and "StartRun" not in ops and "GetReadSet" not in ops
    d = agent._describe_api({"service": "logs", "operation": "GetLogEvents"})
    assert "logStreamName" in d["input"]["required"] and "limit" in d["input"]["members"]
    with pytest.raises(_Refused, match="did you mean"):
        agent._describe_api({"service": "omic"})
    assert agent._describe_api({"service": "secretsmanager"})["read_only_operations"] == []


# ---------- the full run ----------

def test_three_agents_run_the_steps_and_agree_on_a_derived_fix(repo):
    res, fake, llm = run_agent(repo)
    assert res.error == "" and res.done, res.error
    assert [s.phase for s in res.steps] == sorted([s.phase for s in res.steps], key=PHASES.index)
    assert [(s, o) for s, o, _ in fake.seen] == [("omics", "get_run"), ("omics", "list_run_tasks"),
                                                  ("logs", "get_log_events"), ("s3", "get_object")]
    assert fake.seen[3][2]["Bucket"] == BUCKET and fake.seen[3][2]["Range"] == "bytes=-65536"   # account put back
    assert ACCT not in json.dumps(llm.requests)                                                 # never sent to a model
    refused = [(s.tool, s.args.get("operation")) for s in res.steps if not s.ok]
    assert refused == [("aws_call", "GetObject"), ("aws_call", "GetSecretValue"), ("aws_call", "StartRun"),
                       ("finish_debug", None)]
    assert llm.searches[0] == "HealthOmics task out of memory site:docs.aws.amazon.com"
    new = res.patch.files["conf/pipefix.config"][1]
    assert "{ 15.GB * task.attempt }" in new and "errorStrategy" not in new and "cpus" not in new
    assert "resourceLimits" not in new                                  # nothing the agents did not choose
    a = res.candidates[0]
    assert a["derivations"]["memory"]["inputs"]["reserve"]["url"] == "https://docs.aws.amazon.com/doc"
    assert a["evidence"][0]["setting"] == "memory"
    # the review: round 1 split, the primary defended with evidence, round 2 both agree
    rounds = res.review["rounds"]
    assert res.review["agreed"] and len(rounds) == 2
    assert {k: v["verdict"] for k, v in rounds[0]["reviews"].items()} == {"aws_expert": "agree",
                                                                          "nextflow_expert": "object"}
    assert rounds[0]["response"]["action"] == "defend"
    assert {s.agent for s in res.steps} == {"primary", "aws_expert", "nextflow_expert"}
    assert first(res, "inspect_call").agent == "aws_expert"
    res.patch.agent = res
    text = body(res.diagnoses, res.patch, res.inv)
    assert "## How the agents worked" in text and "Both experts agreed" in text
    assert "ceil(need_gib * (1 + headroom) / (1 - reserve))" in text
    assert pr_title(res.patch).startswith("pipefix agent:")
    j = to_json(res)
    assert j["aws_calls"] == 4 and j["review"]["agreed"] and j["usage"]["requests"] == len(llm.requests)


def test_reasoning_trace_names_the_agent_and_its_reasons(repo):
    events = []
    res, _, llm = run_agent(repo, on_event=events.append)
    kinds = [e["kind"] for e in res.trace]
    assert kinds[0] == "phase" and {"thought", "call", "result", "round", "review", "agree"} <= set(kinds)
    assert events == res.trace
    call = next(e for e in res.trace if e["kind"] == "call" and e["tool"] == "aws_describe_api")
    assert call["text"] == "find the call that describes a run" and call["agent"] == "primary"
    assert any(e["agent"] == "nextflow_expert" and e["kind"] == "review" and e["verdict"] == "object" for e in res.trace)
    assert all("reason" not in s.args for s in res.steps)
    assert "reason" in llm.requests[0]["tools"][0]["function"]["parameters"]["required"]
    assert [e["text"] for e in res.trace if e["kind"] == "phase"] == [
        "1. Debug", "2. Research solutions", "3. Pick the best solution", "3b. Expert review", "4. Implement it",
        "5. Open the merge request"]
    assert res.trace[-1]["tokens"] > 0 and "elapsed" in res.trace[-1]


def test_a_call_without_a_reason_is_refused(repo):
    turns = script()
    turns.insert(1, [("aws_call", {"service": "omics", "operation": "GetRun", "params": {"id": "5472488"}})])
    res, _, _ = run_agent(repo, turns=turns)
    assert "needs a 'reason'" in res.steps[1].output and not res.steps[1].ok


def test_mr_step_uses_the_opener_only_when_allowed(repo):
    opened = []
    res, _, _ = run_agent(repo, allow_mr=True, mr_opener=lambda r: opened.append(r) or "https://github.com/o/r/pull/9")
    assert res.mr_url == "https://github.com/o/r/pull/9" and len(opened) == 1
    res2, _, _ = run_agent(repo)
    assert res2.mr_url == "" and res2.patch is not None


def test_the_agent_composes_parameters_and_bad_ones_are_explained(repo):
    turns = script()
    turns.insert(1, [aws("omics", "GetRun", {"runId": "5472488"})])
    turns.insert(2, [aws("omics", "GetRunz", {"id": "5472488"})])
    res, _, _ = run_agent(repo, turns=turns)
    assert 'Unknown parameter in input: "runId"' in res.steps[1].output
    assert "has no operation 'GetRunz'" in res.steps[2].output and "GetRun" in res.steps[2].output


# ---------- debug ----------

def test_debug_diagnosis_is_cross_checked(repo):
    turns = script()
    k = turn_of(turns, "finish_debug", 1)
    good = turns[k][0][1]
    turns[k:k] = [[R("finish_debug", {**good, "process": "NOT_A_PROCESS"})],
                  [R("finish_debug", {**good, "exit_code": 137})]]
    res, _, _ = run_agent(repo, turns=turns)
    outs = [s.output for s in res.steps if s.tool == "finish_debug" and not s.ok]
    assert any("is not defined in the pipeline repo" in o for o in outs)
    assert any("exit_code 137 must be shown" in o for o in outs)
    assert res.diagnosis["process"] == "SORT_READS" and res.diagnosis["exit_code"] == 1   # the good one was accepted


# ---------- derived numbers ----------

@pytest.mark.parametrize("deriv,why", [
    (None, "needs a derivation"),
    ({"formula": "need_gib * (1 + headroom) / (1 - reserve)", "inputs": DERIV_15["memory"]["inputs"]},
     "the formula gives 14.5"),
    ({"formula": "ceil(need_gib * 1.15 / (1 - reserve))", "inputs": DERIV_15["memory"]["inputs"]},
     "make each an input backed by evidence"),                          # a headroom literal is a guess
    ({"formula": "ceil(need_gib * h / (1 - reserve))", "inputs": {**DERIV_15["memory"]["inputs"],
                                                                   "h": {"value": 1.15, "policy": "safety"}}},
     "has no evidence"),
    ({"formula": "ceil(need_gib * 1.15 / 0.95)", "inputs": {"need_gib": {"value": 13, "call": LOG,
                                                                          "quote": "allocating a 12 GB sort buffer"}}},
     "the quote does not show that number"),
    ({"formula": "ceil(need_gib * 1.15 / 0.95)", "inputs": {"need_gib": {"value": 12, "call": LOG,
                                                                          "quote": "allocating a 12 GB buffer"}}},
     "quote is not in the output of call"),
    ({"formula": "ceil(need_gib * 1.15 / 0.95)", "inputs": {"need_gib": {"value": 12, "call": 77, "quote": "12 GB"}}},
     "was not made"),
    ({"formula": "15", "inputs": {"need_gib": {"value": 12, "call": LOG, "quote": "allocating a 12 GB sort buffer"}}},
     "must use its inputs"),
    ({"formula": "__import__('os').getpid()", "inputs": {"os": {"value": 12, "call": LOG,
                                                                 "quote": "allocating a 12 GB sort buffer"}}},
     "cannot be evaluated"),
    ({"formula": "floor(k * (1 - r))", "inputs": {"k": {"value": 16, "source": 1, "quote": "16 GiB of memory"},
                                                  "r": {"value": 0.05, "source": 1,
                                                   "quote": "reserves a small amount of memory (5%)"}}},
     "derived from what the task needs"),                    # out of memory: the need must come from the run or repo
])
def test_numbers_must_be_derived_from_quoted_inputs(repo, deriv, why):
    a = {**GOOD_CANDIDATES[0], "derivations": {"memory": deriv} if deriv else {}}
    res, _, _ = run_agent(repo, turns=with_research([a, GOOD_CANDIDATES[1]], tries=4))
    st = first(res, "finish_research")
    assert why in st.output and not res.candidates[0]["valid"] if st.ok else why in st.output, st.output


# ---------- hard rules ----------

@pytest.mark.parametrize("settings,extra,why", [
    ({"memory": "{ 15.GB * task.attempt }",
      "errorStrategy": "{ task.exitStatus in ((130..145) + 104 + 1) ? 'retry' : 'finish' }"}, [],
     "retrying exit 1 needs a verified doc quote"),                                  # deterministic: no retry
    ({"memory": "{ 15.GB * task.attempt }", "maxRetries": "3"}, [{"setting": "maxRetries", "source": 2, "quote": Q_MR}],
     "maxRetries has no effect"),
    ({"memory": "{ 15.GB * task.attempt }",
      "errorStrategy": "{ task.exitStatus in (130..140) ? 'retry' : 'finish' }"}, [], "drops codes"),
    ({"memory": "{ 15.GB * task.attempt }", "cpus": "4"}, [], "cpus = 4: a number needs a derivation"),
    ({"memory": "{ [15.GB * task.attempt, 30.GB].min() }"}, [], "accepted form"),
])
def test_hard_rules(repo, settings, extra, why):
    a = {**GOOD_CANDIDATES[0], "settings": settings, "evidence": GOOD_CANDIDATES[0]["evidence"] + extra}
    res, _, _ = run_agent(repo, turns=with_research([a, GOOD_CANDIDATES[1]], tries=4))
    st = first(res, "finish_research")
    assert why in st.output and not res.candidates[0]["valid"] if st.ok else why in st.output, st.output


def test_the_agent_decides_the_headroom_but_must_back_it_and_table_rows_count(repo):
    deriv = {"memory": {"formula": "ceil(need_gib * (1 + h) / (1 - reserve))", "inputs": {
        **{k: v for k, v in DERIV_15["memory"]["inputs"].items() if k != "headroom"},
        "h": {"value": 0.15, "source": 1, "quote": "at least 15% headroom"}}}}
    a = {**GOOD_CANDIDATES[0], "derivations": deriv,
         "evidence": [{"setting": "memory", "source": 1, "quote": Q_MEM}, {"source": 1, "quote": "omics.r.large instance"}]}
    res, _, _ = run_agent(repo, turns=with_research([a, GOOD_CANDIDATES[1]]))
    assert res.done, res.error
    assert res.candidates[0]["derivations"]["memory"]["inputs"]["h"]["url"].startswith("https://docs.aws")


def test_healthomics_sizing_must_cite_the_instance_docs(repo):
    a = {**GOOD_CANDIDATES[0], "evidence": GOOD_CANDIDATES[0]["evidence"][:1]}
    res, _, _ = run_agent(repo, turns=with_research([a, GOOD_CANDIDATES[1]], tries=4))
    assert "must cite the HealthOmics instance-type documentation" in first(res, "finish_research").output


def test_more_retries_are_allowed_when_the_observed_kill_is_retried(repo):
    turns = script()
    k = turn_of(turns, "finish_debug", 1)
    dbg = turns[k][0][1]
    turns[k] = [R("finish_debug", {**dbg, "exit_code": 137, "evidence": [
        {"call": LOG, "quote": "allocating a 12 GB sort buffer ... Command exit status: 137"},
        {"call": 9, "quote": "--buffer-gb ${params.sort_buffer_gb}"}]})]
    a = {**GOOD_CANDIDATES[0], "settings": {"memory": "{ 15.GB * task.attempt }", "maxRetries": "2"},
         "evidence": GOOD_CANDIDATES[0]["evidence"] + [{"setting": "maxRetries", "source": 2, "quote": Q_MR}]}
    turns[turn_of(turns, "finish_research")] = [R("finish_research", {"candidates": [a, GOOD_CANDIDATES[1]]})]
    add_check(turns, {"covers": "maxRetries", "claim": "2 retries", "finding": "confirmed", "explanation": "ok",
                      "evidence": [{"source": 2, "quote": Q_MR}]})
    res, _, _ = run_agent(repo, turns=turns, fake=FakeAws(exit_code=137))
    assert res.done, res.error
    assert "maxRetries = 2" in res.patch.files["conf/pipefix.config"][1]


def test_a_candidate_whose_settings_are_all_no_ops_is_refused(repo):
    noop = {"id": "C", "title": "only widen the retry rule", "process": "SORT_READS", "rationale": "r",
            "settings": {"errorStrategy": "{ task.exitStatus in ((130..145) + 104) ? 'retry' : 'finish' }"},
            "evidence": [{"source": 2, "quote": Q_ES}]}
    turns = with_research(GOOD_CANDIDATES + [noop])
    turns[turn_of(turns, "finish_decision")] = [R("finish_decision", {**DECISION_A, "why_not": {
        "B": "changes the tool", "C": "changes nothing"}})]
    res, _, _ = run_agent(repo, turns=turns)
    c = res.candidates[2]
    assert not c["valid"] and "changes nothing" in c["problems"][0] and c["declared"] == 1


def test_quotes_no_op_settings_and_prefixes(repo):
    a = {**GOOD_CANDIDATES[0], "title": "prefix, a no-op and unlabelled quotes",
         "settings": {"withName: 'SORT_READS' memory": "{ 15.GB * task.attempt }",
                      "errorStrategy": "{ task.exitStatus in ((130..145) + 104) ? 'retry' : 'finish' }"},
         "evidence": [{"source": 1, "quote": Q_MEM}, {"setting": "memory", "source": 2, "quote": Q_ES},
                      {"source": 1, "quote": Q_INST}, {"source": 1, "quote": "a sentence that is not on the page at all"}]}
    res, _, _ = run_agent(repo, turns=with_research([a, GOOD_CANDIDATES[1]]))
    assert res.done, res.error
    ca = res.candidates[0]
    assert ca["valid"] and ca["settings"] == {"memory": "{ 15.GB * task.attempt }"}
    assert any("errorStrategy dropped: same as now" in n for n in ca["notes"])
    assert any("not used" in n for n in ca["notes"])
    assert [e["setting"] for e in ca["evidence"]] == ["memory", "context", "memory"]   # Q_ES is not about memory


def test_citing_the_call_that_read_a_page_maps_to_that_source(repo):
    cands = json.loads(json.dumps(GOOD_CANDIDATES))
    cands[0]["evidence"][0] = {"setting": "memory", "call": 15, "quote": Q_MEM}   # call #15 read source 1
    res, _, _ = run_agent(repo, turns=with_research(cands))
    assert res.done, res.error
    assert res.candidates[0]["evidence"][0]["n"] == 1


# ---------- the expert review ----------

def test_review_rules_are_enforced(repo):
    turns = script()
    k = turn_of(turns, "finish_review")
    turns[k:k] = [
        [R("finish_review", {"verdict": "agree", "checks": [CHECK_ROOT]})],                                  # misses memory
        [R("finish_review", {"verdict": "agree", "checks": [
            CHECK_ROOT, {**CHECK_MEM, "finding": "unsupported"}]})],                                         # agree + objection
        [R("finish_review", {"verdict": "agree", "checks": [
            CHECK_ROOT, {**CHECK_MEM, "evidence": [{"source": 1, "quote": "a quote that is not on that page"}]}]})],
    ]
    res, _, _ = run_agent(repo, turns=turns)
    outs = [s.output for s in res.steps if s.tool == "finish_review" and not s.ok]
    assert "missing ['memory']" in outs[0]
    assert "so the verdict is 'object'" in outs[1]
    assert "needs verified evidence" in outs[2]
    assert any(s.ok and s.tool == "finish_review" and s.agent == "aws_expert" for s in res.steps)   # 4th try passes


def test_a_reviewer_must_bring_its_own_evidence(repo):
    turns = script()
    k = turn_of(turns, "finish_review", 1)       # the Nextflow expert's turns are k-1 (read) and k (review)
    turns.insert(k - 1, [R("finish_review", {"verdict": "agree", "checks": [CHECK_ROOT, CHECK_MEM]})])
    res, _, _ = run_agent(repo, turns=turns)
    st = next(s for s in res.steps if s.tool == "finish_review" and s.agent == "nextflow_expert")
    assert not st.ok and "a call you made or a page you read" in st.output


def test_defending_needs_evidence_and_revising_is_rechecked(repo):
    turns = script()
    k = turn_of(turns, "finish_response")
    rev = {"id": "A", "title": "Size memory to the buffer, 16 GB", "process": "SORT_READS",
           "settings": {"memory": "{ 16.GB * task.attempt }"}, "rationale": "more headroom",
           "derivations": {"memory": {"formula": "ceil((need_gib + 1) * (1 + headroom) / (1 - reserve))",
                                      "inputs": DERIV_15["memory"]["inputs"]}},
           "evidence": GOOD_CANDIDATES[0]["evidence"]}
    turns[k:k + 1] = [
        [R("finish_response", {"action": "defend", "answers": [{"question": "q", "answer": "trust me"}]})],
        [R("finish_response", {"action": "revise", "answers": [{"question": "q", "answer": "a"}],
                               "candidates": [{**rev, "settings": {"memory": "{ 99.GB * task.attempt }"}},
                                              GOOD_CANDIDATES[1]], **DECISION_A})],
        [R("finish_response", {"action": "revise", "answers": [{"question": "Why 15%?", "answer": "use 20% headroom"}],
                               "candidates": [rev, GOOD_CANDIDATES[1]], **DECISION_A})],
    ]
    res, _, _ = run_agent(repo, turns=turns)
    outs = [s.output for s in res.steps if s.tool == "finish_response" and not s.ok]
    assert "answer each question as asked" in outs[0]
    assert "the formula gives 16, but the value says 99" in outs[1]
    assert res.done and "{ 16.GB * task.attempt }" in res.patch.files["conf/pipefix.config"][1]
    assert res.review["rounds"][0]["response"]["action"] == "revise"
    assert res.review["rounds"][1]["settings"] == {"memory": "{ 16.GB * task.attempt }"}


def test_no_agreement_means_no_change_and_no_mr(repo):
    turns = script()
    obj = [R("finish_review", {"verdict": "object", "checks": [
        {**CHECK_ROOT, "evidence": [{"call": 23, "quote": "--buffer-gb ${params.sort_buffer_gb}"}]},
        {"covers": "memory", "claim": "15 GB", "finding": "unsupported", "explanation": "not proven"}],
        "questions": ["prove it"]})]
    defend = [R("finish_response", {"action": "defend", "answers": [
        {"question": "prove it", "answer": "the docs give the reserve", "evidence": [{"source": 1, "quote": Q_RES}]}]})]
    k = turn_of(turns, "finish_review", 2)
    agree_aws = turns[k]
    turns[k:] = [agree_aws, obj, defend, agree_aws, obj]          # rounds 2 and 3: the Nextflow expert still objects
    opened = []
    res, _, _ = run_agent(repo, turns=turns, allow_mr=True, mr_opener=lambda r: opened.append(r) or "x")
    assert "did not agree" in res.error and res.patch is None and not opened
    assert len(res.review["rounds"]) == 3 and not res.review["agreed"]


def test_an_agreed_non_config_answer_stops_cleanly(repo):
    turns = script()
    k = turn_of(turns, "finish_debug", 1)
    turns[k] = [(n, {**a, "failure_class": "container_error"}) for n, a in turns[k]]
    turns[turn_of(turns, "finish_decision")] = [R("finish_decision", {
        "chosen": "B", "reasons": "For a container error the image must be fixed; no config change helps here.",
        "why_not": {"A": "memory is not the problem"}})]
    nochange = {"covers": "no_change", "claim": "no config change", "finding": "confirmed", "explanation": "ok",
                "evidence": [{"source": 1, "quote": Q_RES}]}
    for j in [turn_of(turns, "finish_review", n) for n in range(4)]:
        turns[j] = [(nm, {**a, "verdict": "agree", "checks": [a["checks"][0], nochange], "questions": []})
                    for nm, a in turns[j]]
    r = turn_of(turns, "finish_response")
    del turns[r:r + 3]                                              # round 1 agrees: no response, no round 2
    res, _, _ = run_agent(repo, turns=turns)
    assert res.review["agreed"] and "not a pipeline config change" in res.error and res.patch is None


# ---------- structure ----------

def test_tools_are_phase_scoped_with_one_finish_each():
    for p in ("debug", "research", "decide", "review", "respond", "implement", "mr"):
        n = sum(1 for c in COMMANDS.values() if p in c["phases"] and c.get("finish"))
        assert n == (2 if p == "decide" else 1), p             # decide may also reopen research
    assert _in_phase("aws_call", "review") and not _in_phase("aws_call", "decide")
    assert not _in_phase("inspect_call", "debug")
    assert not any(n.startswith(("omics_", "logs_", "facts_")) for n in COMMANDS)   # nothing pre-wired


def test_stops_cleanly_when_a_step_cannot_finish(repo):
    turns = script()[:1] + [[aws("omics", "GetRun", {"id": "5472488"})]] * 40
    res, _, _ = run_agent(repo, turns=turns)
    assert "did not finish within its limits" in res.error and res.phase_reached == "debug" and res.patch is None


def test_benchmark_cases_build_and_score(tmp_path):
    from pipefix.agent import AgentResult
    from pipefix.bench import CASES, _reply, make_repo, score
    from pipefix.configtrace import load_repo
    assert len(CASES) == 10
    for c in CASES:
        repo = make_repo(c, tmp_path / c.id)
        cfg = load_repo(repo)
        assert c.process in cfg.processes and cfg.trace(c.process).effective("memory")
        assert _reply(c, "omics", "get_run", {})["logLocation"]["engineLogStream"].endswith("run/9001/engine")
    ok = AgentResult(decision={"chosen": "A"}, candidates=[{"id": "A", "settings": {"memory": "{ 15.GB * task.attempt }"}}],
                     review={"rounds": [{}], "agreed": True})
    ok.patch = object()
    r = score(CASES[0], ok)
    assert r["value"] and r["justified"] and r["decision"]
    bad = AgentResult(decision={"chosen": "A"}, candidates=[{"id": "A", "settings": {"memory": "{ 20.GB * task.attempt }",
                                                                                     "time": "{ 2.h * task.attempt }"}}],
                      review={"rounds": [{}], "agreed": True})
    bad.patch = object()
    r = score(CASES[0], bad)
    assert not r["value"] and not r["justified"]                 # time is not allowed for this case
    ok.review = {"rounds": [{}], "agreed": False}
    assert not score(CASES[0], ok)["decision"]                   # no agreement, no pass


def test_page_focus_finds_the_sentence_asked_for():
    from pipefix.agent import _focus
    page = ("HealthOmics memory HealthOmics memory intro. " * 60 + "HealthOmics reserves a small amount of memory (5%) "
            "for management and logging agents. " + "Other text about instances. " * 60)
    assert "reserves a small amount of memory (5%)" in _focus(page, "reserves a small amount of memory 5%")
    assert _focus(page, "zzz qqq") == ""


def test_repeating_an_identical_call_is_refused(repo):
    turns = script()
    turns.insert(2, [aws("omics", "GetRun", {"id": "5472488"}, "read the run again")])
    res, _, _ = run_agent(repo, turns=turns)
    assert "already made this exact call (#2)" in res.steps[2].output


def test_decide_can_reopen_research_once(repo):
    turns = script()
    i = turn_of(turns, "finish_decision")
    turns[i:i] = [[R("reopen_research", {"why": "candidate A needs a better quote"})],
                  [R("finish_research", {"candidates": GOOD_CANDIDATES})]]
    res, _, _ = run_agent(repo, turns=turns)
    assert res.decision["chosen"] == "A" and res.review["rounds"] is not None   # reached the review
    assert [e["text"] for e in res.trace if e["kind"] == "phase"].count("2. Research solutions") == 2


def test_out_of_memory_without_a_config_candidate_needs_a_proven_reason(repo):
    only_b = [GOOD_CANDIDATES[1], {**GOOD_CANDIDATES[1], "id": "C", "title": "rewrite the tool"}]
    res, _, _ = run_agent(repo, turns=with_research(only_b, tries=4))
    assert "usually fixable by config" in first(res, "finish_research").output
    turns = with_research(only_b)
    i = turn_of(turns, "finish_research")
    turns[i] = [R("finish_research", {"candidates": only_b, "no_config_reason": {
        "reason": "the docs say allocations are hard limits and the tool must change", "evidence": [
            {"source": 1, "quote": Q_MEM}]}})]
    res, _, _ = run_agent(repo, turns=turns)
    assert first(res, "finish_research").ok and res.diagnosis["no_config_reason"]["evidence"]


def test_a_reviewer_cannot_confirm_a_number_with_an_unrelated_quote(repo):
    turns = script()
    k = turn_of(turns, "finish_review")
    off = {**CHECK_MEM, "evidence": [{"source": 1, "quote": "Container resource allocations are hard limits"}]}
    turns.insert(k, [R("finish_review", {"verdict": "agree", "checks": [CHECK_ROOT, off]})])
    res, _, _ = run_agent(repo, turns=turns)
    st = next(s for s in res.steps if s.tool == "finish_review")
    assert not st.ok and "needs evidence that is about memory" in st.output


def test_a_claim_found_wrong_with_evidence_must_be_revised_not_defended(repo):
    turns = script()
    k = turn_of(turns, "finish_review", 1)
    turns[k] = [R("finish_review", {"verdict": "object", "checks": [
        {**CHECK_ROOT, "evidence": [{"call": 23, "quote": "--buffer-gb ${params.sort_buffer_gb}"}]},
        {**CHECK_MEM, "finding": "wrong", "claim": "15 GB fits"}], "questions": ["Why is 15 GB enough after the platform reserve?"]})]
    res, _, _ = run_agent(repo, turns=turns)
    st = first(res, "finish_response")
    assert not st.ok and "cannot defend them: revise" in st.output


DOC_TWO_FAMILIES = DOC_AWS.replace("</main>", "<table><tr><td>omics.m.large</td><td>2</td><td>8 GiB</td></tr>"
                                   "<tr><td>omics.m.xlarge</td><td>4</td><td>16 GiB</td></tr>"
                                   "<tr><td>omics.r.large</td><td>2</td><td>16 GiB</td></tr></table></main>")


def test_a_landing_claim_must_show_every_instance_family_on_the_page(repo):
    one = {**GOOD_CANDIDATES[0], "evidence": [{"setting": "memory", "source": 1, "quote": Q_MEM},
                                              {"source": 1, "quote": "omics.m.xlarge 4 16 GiB"}]}
    res, _, _ = run_agent(repo, turns=with_research([one, GOOD_CANDIDATES[1]], tries=4), aws_doc=DOC_TWO_FAMILIES)
    out = first(res, "finish_research").output
    assert "quote a row from each of ['r']" in out, out
    both = {**one, "evidence": one["evidence"] + [{"source": 1, "quote": "omics.r.large 2 16 GiB"}]}
    res, _, _ = run_agent(repo, turns=with_research([both, GOOD_CANDIDATES[1]]), aws_doc=DOC_TWO_FAMILIES)
    assert first(res, "finish_research").ok, first(res, "finish_research").output


def test_a_reviewer_landing_check_must_show_every_family(repo):
    a = {**GOOD_CANDIDATES[0], "evidence": GOOD_CANDIDATES[0]["evidence"] + [
        {"source": 1, "quote": "omics.m.xlarge 4 16 GiB"}, {"source": 1, "quote": "omics.r.large 2 16 GiB"}]}
    turns = with_research([a, GOOD_CANDIDATES[1]])
    k = turn_of(turns, "finish_review")
    land = {"covers": "memory", "claim": "15 GB lands on omics.m.xlarge", "finding": "confirmed",
            "explanation": "table", "evidence": [{"source": 1, "quote": "omics.m.xlarge 4 16 GiB"}]}
    turns.insert(k, [R("finish_review", {"verdict": "agree", "checks": [CHECK_ROOT, CHECK_MEM, land]})])
    res, _, _ = run_agent(repo, turns=turns, aws_doc=DOC_TWO_FAMILIES)
    st = next(s for s in res.steps if s.tool == "finish_review")
    assert not st.ok and "rows from every family" in st.output, st.output


def test_headroom_and_reserve_are_separate_inputs(repo):
    no_headroom = {**GOOD_CANDIDATES[0], "settings": {"memory": "{ 13.GB * task.attempt }"},
                   "derivations": {"memory": {"formula": "ceil(need_gib / (1 - reserve))", "inputs": {
                       k: v for k, v in DERIV_15["memory"]["inputs"].items() if k != "headroom"}}}}
    res, _, _ = run_agent(repo, turns=with_research([no_headroom, GOOD_CANDIDATES[1]], tries=4))
    assert "recommends memory headroom" in first(res, "finish_research").output
    same = {**GOOD_CANDIDATES[0], "settings": {"memory": "{ 14.GB * task.attempt }"},
            "derivations": {"memory": {"formula": "ceil(need_gib * (1 + h) / (1 - reserve))", "inputs": {
                "need_gib": DERIV_15["memory"]["inputs"]["need_gib"],
                "reserve": {"value": 0.05, "source": 1, "quote": "reserves a small amount of memory (5%)"},
                "h": {"value": 0.05, "source": 1, "quote": "reserves a small amount of memory (5%)"}}}}}
    res, _, _ = run_agent(repo, turns=with_research([same, GOOD_CANDIDATES[1]], tries=4))
    out = first(res, "finish_research").output
    assert "quote the same text" in out and "recommends memory headroom" in out, out
