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


def _score(tmp_path, monkeypatch, examples, samples, rule, name, judges=("haiku",)):
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(json.dumps(s) + "\n" for s in samples))
    monkeypatch.setattr(
        run.healthbench, "load_examples", lambda root: list(examples.values())
    )
    monkeypatch.setattr(run, "ClaudeJudge", lambda config, cache: FakeJudge(rule=rule))
    monkeypatch.setattr(run, "VerdictCache", lambda *a: None)
    return asyncio.run(
        run.score_file(path, judges, concurrency=4, root=None, out=tmp_path / "evals")
    )


def test_score_and_compare(tmp_path, monkeypatch):
    examples = _examples(30)
    samples = [
        {"prompt_id": pid, "sample": k, "response": f"answer {pid}"}
        | {"finish_reason": "stop", "completion_tokens": 10}
        for pid in examples
        for k in range(2)
    ]
    base = _score(
        tmp_path, monkeypatch, examples, samples, lambda p: "good0" in _item(p), "base"
    )
    # "new" meets every positive criterion: gains on both halves, no hacking gap
    new = _score(
        tmp_path, monkeypatch, examples, samples, lambda p: "bad" not in _item(p), "new"
    )
    s = json.loads((new / "summary.json").read_text())
    assert s["haiku"]["full"]["mean"] == pytest.approx(1.0)
    assert s["n_conversations"] == 30 and s["length"]["truncated"] == 0
    b = json.loads((base / "summary.json").read_text())
    assert b["haiku"]["full"]["mean"] == pytest.approx(0.25)
    capped = _score(
        tmp_path, monkeypatch, examples, samples, lambda p: True, "capped",
        judges=("haiku", "sonnet:1"),
    )  # fmt: skip
    c = json.loads((capped / "summary.json").read_text())
    assert c["haiku"]["n_responses"] == 60 and c["sonnet"]["n_responses"] == 30
    out = run.compare(base, new)
    assert "| haiku | 0.750" in out
    assert "| Run | Judge |" in run.table([base, new])
