"""The eval scorer end to end with a fake judge, and the paired comparison."""

import asyncio
import json

import numpy as np
import pytest

from rubricrl.data import healthbench as hb
from rubricrl.evals import run
from rubricrl.judge.client import FakeJudge


def _examples(n: int) -> dict:
    rows = [
        {
            "prompt_id": f"p{i}",
            "prompt": [{"role": "user", "content": f"q{i}"}],
            "rubrics": [
                {"criterion": f"good{k}", "points": 5, "tags": ["axis:accuracy"]}
                for k in range(4)
            ]
            + [{"criterion": "bad", "points": -5, "tags": ["axis:completeness"]}],
            "example_tags": ["theme:hedging"],
        }
        for i in range(n)
    ]
    return {r["prompt_id"]: hb.parse_example(r) for r in rows}


def _item(prompt: str) -> str:
    return prompt.split("# Rubric item\n")[1].split("\n\n# Instructions")[0]


def test_bootstrap_ci_contains_the_mean():
    v = np.random.default_rng(0).normal(0.5, 0.1, 200)
    lo, hi = run.bootstrap_ci(v)
    assert lo < v.mean() < hi and hi - lo < 0.05


def _score(
    tmp_path, monkeypatch, examples, samples, rule, name, judges=("haiku",), **kw
):
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(json.dumps(s) + "\n" for s in samples))
    monkeypatch.setattr(
        run.healthbench, "load_examples", lambda root: list(examples.values())
    )
    configs = []

    def fake(config, cache):
        configs.append(config)
        return FakeJudge(rule=rule)

    monkeypatch.setattr(run, "make_judge", fake)
    monkeypatch.setattr(run, "VerdictCache", lambda *a: None)
    out = asyncio.run(
        run.score_file(
            path, judges, concurrency=4, root=None, out=tmp_path / "evals", **kw
        )
    )
    return out, configs


def test_score_and_compare(tmp_path, monkeypatch):
    examples = _examples(30)
    samples = [
        {"prompt_id": pid, "sample": k, "response": f"answer {pid}"}
        | {"finish_reason": "stop", "completion_tokens": 10}
        for pid in examples
        for k in range(2)
    ]
    base, _ = _score(
        tmp_path, monkeypatch, examples, samples, lambda p: "good0" in _item(p), "base"
    )
    # "new" meets every positive criterion: gains on both halves, no hacking gap
    new, _ = _score(
        tmp_path, monkeypatch, examples, samples, lambda p: "bad" not in _item(p), "new"
    )
    s = json.loads((new / "summary.json").read_text())
    assert s["haiku"]["full"]["mean"] == pytest.approx(1.0)
    assert s["n_conversations"] == 30 and s["length"]["truncated"] == 0
    b = json.loads((base / "summary.json").read_text())
    assert b["haiku"]["full"]["mean"] == pytest.approx(0.25)
    capped, _ = _score(
        tmp_path, monkeypatch, examples, samples, lambda p: True, "capped",
        judges=("haiku", "sonnet:1"),
    )  # fmt: skip
    c = json.loads((capped / "summary.json").read_text())
    assert c["haiku"]["n_responses"] == 60 and c["sonnet"]["n_responses"] == 30
    out = run.compare(base, new)
    assert "| haiku | 0.750" in out
    assert "| Run | Judge |" in run.table([base, new])


def test_parse_judge_specs():
    assert run.parse_judge("haiku@gepa-v1") == ("haiku", "gepa-v1", None)
    assert run.parse_judge("sonnet:1") == ("sonnet", None, 1)
    assert run.parse_judge("haiku@gepa-v1:2") == ("haiku", "gepa-v1", 2)


def _samples(examples):
    return [
        {"prompt_id": pid, "sample": k, "response": f"answer {pid}"}
        | {"finish_reason": "stop", "completion_tokens": 10}
        for pid in examples
        for k in range(2)
    ]


def test_self_hosted_judge_and_merge(tmp_path, monkeypatch):
    """A `base_url` reaches the judge's config; `merge` puts a second judge's
    verdicts next to the first, so `compare` reports the gap between them."""
    examples, samples = _examples(30), _samples(_examples(30))
    url = "http://127.0.0.1:8004"
    for tag, haiku_rule, oss_rule in (
        ("base", lambda p: "good0" in _item(p), lambda p: "good0" in _item(p)),
        ("new", lambda p: "good0" in _item(p), lambda p: "bad" not in _item(p)),
    ):
        _score(tmp_path, monkeypatch, examples, samples, haiku_rule, tag)
        oss, configs = _score(
            tmp_path, monkeypatch, examples, samples, oss_rule, f"{tag}-oss",
            judges=("gpt-oss-120b@gepa-v1",), base_url=url,
        )  # fmt: skip
        assert configs[0].base_url == url and configs[0].model == "gpt-oss-120b"
        assert configs[0].prompt_version == "gepa-v1"
        run.merge(tmp_path / "evals" / tag, oss, root=None)
    s = json.loads((tmp_path / "evals" / "new" / "summary.json").read_text())
    judges = {"haiku", "gpt-oss-120b@gepa-v1"}
    assert judges <= set(s) and set(s["usage"]) == judges
    out = run.compare(tmp_path / "evals" / "base", tmp_path / "evals" / "new")
    # haiku sees no gain, the self-hosted judge sees +0.75: the gap is the signal
    assert "| haiku | 0.000" in out and "| gpt-oss-120b@gepa-v1 | 0.750" in out
    assert "Full-rubric gain under gpt-oss-120b@gepa-v1 − under haiku: 0.750" in out


def test_merge_refuses_different_samples(tmp_path, monkeypatch):
    examples = _examples(4)
    a, _ = _score(tmp_path, monkeypatch, examples, _samples(examples), bool, "a")
    fewer = [s for s in _samples(examples) if s["sample"] == 0]
    b, _ = _score(tmp_path, monkeypatch, examples, fewer, bool, "b")
    with pytest.raises(ValueError, match="different samples"):
        run.merge(a, b, root=None)
