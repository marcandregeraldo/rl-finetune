"""Weave logging against a fake `weave` module: a trace per graded response,
its judge calls as children."""

import asyncio
import contextlib
import contextvars
import itertools
import sys
import types

import pytest

from rubricrl import observe
from rubricrl.data.schema import Judgment
from rubricrl.judge.client import FakeJudge, JudgeConfig, Verdict


class FakeRef:
    def __init__(self, name):
        self.name = name

    def uri(self):
        return f"weave:///p/{self.name}"


class FakeEvaluationLogger:
    def __init__(self, **kw):
        self.kw, self.summary = kw, None
        fake_weave.evals.append(self)

    def log_prediction(self, *a, **kw):
        raise AssertionError("summary-only: no per-item predictions")

    def log_summary(self, summary=None):
        self.summary = summary

    @property
    def ui_url(self):
        return f"https://wandb.ai/eval/{len(fake_weave.evals)}"


fake_weave = types.SimpleNamespace()


@pytest.fixture(autouse=True)
def weave(monkeypatch):
    fake_weave.evals, fake_weave.calls, fake_weave.published = [], [], []
    fake_weave.attributes = {}
    module = types.ModuleType("weave")
    module.EvaluationLogger = FakeEvaluationLogger
    module.StringPrompt = lambda content: ("prompt", content)
    module.Dataset = lambda name, rows: types.SimpleNamespace(name=name, rows=rows)
    module.init = lambda project, settings: fake_weave.published.append(
        ("init", project, settings)
    )

    def publish(obj, name=None, tags=None):
        fake_weave.published.append((obj, name, tags))
        return FakeRef(name or getattr(obj, "name", "?"))

    @contextlib.contextmanager
    def attributes(attrs):
        previous = fake_weave.attributes
        fake_weave.attributes = previous | attrs
        try:
            yield
        finally:
            fake_weave.attributes = previous

    current = contextvars.ContextVar("current", default=None)
    ids = itertools.count()

    def record(call: dict) -> types.SimpleNamespace:
        call = types.SimpleNamespace(id=next(ids), **call)
        fake_weave.calls.append(call)
        return call

    def op(name=None, call_display_name=None, postprocess_inputs=None):
        def wrap(fn):
            async def call(*args):
                inputs = dict(zip(fn.__code__.co_varnames, args, strict=False))
                attrs = dict(fake_weave.attributes)
                c = record(
                    {
                        "op": name,
                        "inputs": postprocess_inputs(inputs)
                        if postprocess_inputs
                        else inputs,
                        "attributes": attrs,
                        "parent": current.get(),
                        "display_name": call_display_name(
                            types.SimpleNamespace(attributes=attrs)
                        ),
                    }
                )
                token = current.set(c.id)
                try:
                    c.output = await fn(*args)
                finally:
                    current.reset(token)
                return c.output

            return call

        return wrap

    def log_call(op, inputs, output, parent=None, attributes=None, **kw):
        return record(
            {
                "op": op, "inputs": inputs, "output": output,
                "attributes": attributes or {},
                "parent": None if parent is None else parent.id,
                "display_name": kw.get("display_name"),
            }
        )  # fmt: skip

    module.publish, module.attributes, module.op = publish, attributes, op
    module.log_call = log_call
    monkeypatch.setitem(sys.modules, "weave", module)
    return fake_weave


def _j(i: int, labels: tuple) -> Judgment:
    msgs = [{"role": "user", "content": f"q{i}"}]
    return Judgment(f"{i}:c", "p", str(i), msgs, "x" * i, f"r{i}", "c", labels, ())


def test_init_and_publish(weave):
    observe.init("proj")
    assert weave.published[0] == (
        "init",
        "proj",
        {"implicitly_patch_integrations": False},
    )
    ref = observe.publish_prompt()
    (prompt, name, tags) = weave.published[1]
    assert prompt == ("prompt", observe.GRADER_TEMPLATE)
    assert name == observe.PROMPT_NAME and tags == [observe.PROMPT_VERSION]
    assert ref.uri().endswith(observe.PROMPT_NAME)
    ds = observe.publish_judgments([_j(1, (True,)), _j(2, (False,))], seed=3)
    assert ds.name == "healthbench-meta-eval-n2-s3" and len(ds.rows) == 2


def test_agreement():
    assert observe.agreement(True, (True, True, False)) == {
        "physician_agreement": pytest.approx(2 / 3),
        "agrees_with_majority": True,
    }
    assert observe.agreement(False, (True, False)) == {"physician_agreement": 0.5}
    assert observe.agreement(None, (True,)) == {}


def _jc(i: int, completion_id: str, labels: tuple) -> Judgment:
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    msgs += [{"role": "user", "content": "q2"}]
    return Judgment(
        f"{i}:c{i}", "p", completion_id, msgs, f"answer {completion_id}",
        f"r{i}", f"c{i}", labels, (),
    )  # fmt: skip


def test_traced_grader_one_trace_per_completion(weave):
    judge = FakeJudge(rule=lambda p: "r1" in p.split("# Rubric item")[1])
    attrs = observe.run_attributes(
        "haiku-low", JudgeConfig.named("haiku"), 3, 0, FakeRef("g")
    )
    grade = observe.traced_grader(judge, attrs)
    js = [_jc(1, "A", (True, True)), _jc(2, "B", (True,)), _jc(3, "A", (True, False))]
    verdicts = asyncio.run(grade(js, 0))
    assert [v.met for v in verdicts] == [True, False, False]  # input order
    assert isinstance(verdicts[0], Verdict) and verdicts[0].explanation == "fake"

    roots = [c for c in weave.calls if c.parent is None]
    children = [c for c in weave.calls if c.parent is not None]
    assert [r.op for r in roots] == [observe.ROOT_OP] * 2 and len(children) == 3
    a = next(r for r in roots if r.attributes["completion_id"] == "A")
    assert a.display_name == "haiku-low · A"
    assert a.attributes["grader_prompt"].endswith("/g")
    chat = a.inputs["messages"]
    assert [m["role"] for m in chat] == ["user", "assistant", "user", "assistant"]
    assert chat[-1]["content"] == "answer A"
    assert [c["id"] for c in a.inputs["criteria"]] == ["1:c1", "3:c3"]
    assert a.output["criteria"] == 2 and a.output["met"] == 1
    assert a.output["agrees_with_majority"] == 1.0  # item 3: physicians tied

    a_children = [c for c in children if c.parent == a.id]
    assert {c.attributes["item"] for c in a_children} == {"1:c1", "3:c3"}
    first = next(c for c in a_children if c.attributes["item"] == "1:c1")
    assert first.op == observe.OP and first.display_name == "c1"
    assert set(first.inputs) == {"rubric_item", "repeat", "physician_labels"}
    assert first.output["met"] is True and first.output["agrees_with_majority"]

    weave.calls.clear()
    asyncio.run(grade(js[:1], 1))  # the retest pass: its own root
    (root, child) = weave.calls
    assert root.attributes["repeat"] == 1 and root.display_name.endswith("retest")
    assert child.parent == root.id


def _result() -> dict:
    o = {"f1_balanced": 0.6, "f1_pos": 0.7, "f1_neg": 0.5}
    return {
        "judge": {"overall": o}, "retest": {"agreement": 0.9, "kappa": 0.7},
        "judge_rate_met": 0.4, "physician_rate_met": 0.8, "failed": 0,
        "physicians_all": {"overall": {"f1_balanced_mean": 0.67}},
        "by_length": [{"chars_min": 1, "chars_max": 9, "f1_balanced": 0.6}],
        "usage": {"cost_usd": 0.0},
    }  # fmt: skip


def test_calibration_summary_only(weave):
    attrs = observe.run_attributes("haiku-low", JudgeConfig(), 3, 0, None)
    url = observe.log_calibration_summary(_result(), attrs, "ds")
    (ev,) = weave.evals
    assert url == ev.ui_url and ev.kw["dataset"] == "ds"
    assert ev.kw["model"]["name"] == "haiku-low"
    assert ev.summary["f1_balanced"] == 0.6
    assert ev.summary["physician_ceiling_f1_balanced"] == 0.67
    assert ev.summary["f1_balanced_by_length"] == {"1-9": 0.6}


def _rec(eval: bool, pid: str = "p", score: float | None = 0.5) -> dict:
    return {"eval": eval, "prompt_id": pid, "score": score, "chars": 10}


def test_split_records_groups_eval_rounds_and_holds_a_growing_one():
    recs = [_rec(False)] * 3 + [_rec(True)] * 2 + [_rec(False)] * 4 + [_rec(True)] * 2
    rounds, traces, consumed, seen = observe.split_records(recs, 0, 2, 1.0, "r")
    assert [(step, len(r)) for step, r in rounds] == [(1, 2)]
    assert [step for step, _ in traces] == [0, 0, 1, 1, 2, 2, 3]
    assert consumed == 9 and seen == 7  # the trailing round waits
    rounds, _, consumed, _ = observe.split_records(recs, 0, 2, 1.0, "r", final=True)
    assert [step for step, _ in rounds] == [1, 3] and consumed == 11
    # Resuming: the held-back round arrives alone, at the right step.
    rounds, traces, consumed, seen = observe.split_records(recs[9:], 7, 2, 1.0, "r")
    assert rounds == [] and consumed == 0 and seen == 7
    _, traces, _, _ = observe.split_records(recs, 0, 2, 0.0, "r")
    assert traces == []


def test_sampling_is_deterministic_at_the_rate():
    picks = [observe.sampled("run", i, 0.1) for i in range(20_000)]
    assert picks == [observe.sampled("run", i, 0.1) for i in range(20_000)]
    assert 0.09 < sum(picks) / len(picks) < 0.11


def test_log_training_one_trace_per_sample(weave):
    crit = [
        {"id": "a:0", "text": "Says A", "points": 5, "met": True, "explanation": "e"},
        {"id": "a:1", "text": "Bad", "points": -3, "met": False, "explanation": "f"},
    ]
    evals = [_rec(True, "a") | {"criteria": crit}, _rec(True, "b", None)]
    train = _rec(False, "c") | {"criteria": crit[:1], "response": "hi"}
    conversations = {"c": [{"role": "user", "content": "hello?"}]}
    logged = observe.log_training("run", [(10, evals)], [(3, train)], conversations)
    assert logged == {
        "responses": 3, "verdicts": 3, "eval_rounds": 1, "urls": [weave.evals[0].ui_url]
    }  # fmt: skip
    (ev,) = weave.evals
    assert ev.kw["model"]["name"] == "run@step0010"
    assert ev.summary["heldout_score_mean"] == 0.5 and ev.summary["n"] == 2
    roots = [c for c in weave.calls if c.parent is None]
    assert [r.op for r in roots] == [observe.ROOT_OP] * 3
    first, _, last = roots
    assert first.attributes["step"] == 10 and first.attributes["eval"] is True
    assert first.output["met"] == 1 and first.output["score"] == 0.5
    kids = [c for c in weave.calls if c.parent == first.id]
    assert [k.inputs["rubric_item"] for k in kids] == ["[5] Says A", "[-3] Bad"]
    assert kids[0].output == {"met": True, "explanation": "e"}
    assert last.inputs["messages"] == [
        {"role": "user", "content": "hello?"},
        {"role": "assistant", "content": "hi"},
    ]
    assert last.attributes["step"] == 3 and not last.attributes["eval"]


def test_wandb_key_from_keychain_never_printed(monkeypatch, capsys):
    monkeypatch.setenv("WANDB_API_KEY", "")  # restored after the test
    monkeypatch.delenv("WANDB_API_KEY")
    monkeypatch.setattr(observe.shutil, "which", lambda name: "/usr/bin/security")
    monkeypatch.setattr(
        observe.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="secret-key\n"),
    )
    observe._ensure_wandb_key()
    assert observe.os.environ["WANDB_API_KEY"] == "secret-key"
    assert "secret-key" not in capsys.readouterr().out
