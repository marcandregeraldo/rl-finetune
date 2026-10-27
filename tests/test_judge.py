"""The judge: prompt rendering, the cache, retries, thinking params; no network."""

import asyncio
from types import SimpleNamespace

import pytest

from rubricrl.data.schema import Criterion
from rubricrl.judge import prompt
from rubricrl.judge.client import (
    ClaudeJudge,
    FakeJudge,
    JudgeConfig,
    VerdictCache,
    cache_key,
)

MESSAGES = [{"role": "user", "content": "Is smoking good for me?"}]


def test_render_matches_simple_evals_layout():
    p = prompt.render(MESSAGES, "No.", "[5] Says smoking is harmful.")
    assert (
        "# Conversation\nuser: Is smoking good for me?\n\nassistant: No.\n\n"
        "# Rubric item\n" in p
    )
    assert "[5] Says smoking is harmful.\n\n# Instructions" in p
    assert "<<" not in p


def _message(text: str, stop: str = "end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop,
        usage=SimpleNamespace(
            input_tokens=100, output_tokens=20,
            cache_read_input_tokens=0, cache_creation_input_tokens=None,
        ),
    )  # fmt: skip


class FakeClient:
    """Returns the queued messages in order; records the request kwargs."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []
        self.messages = self

    async def create(self, **kw):
        self.calls.append(kw)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _judge(tmp_path, *replies, **cfg):
    client = FakeClient(*replies)
    config = JudgeConfig(max_attempts=3, **cfg)
    judge = ClaudeJudge(config, VerdictCache(tmp_path / "c.sqlite"), client=client)
    return judge, client


def test_grade_then_cache_hit(tmp_path, monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    judge, client = _judge(
        tmp_path, _message('{"explanation": "ok", "criteria_met": true}')
    )
    v = asyncio.run(judge.grade(MESSAGES, "No.", "[5] x"))
    assert (v.met, v.explanation, v.cached) == (True, "ok", False)
    kw = client.calls[0]
    assert kw["thinking"] == {"type": "disabled"}
    assert kw["output_config"]["effort"] == "low"
    assert kw["output_config"]["format"]["schema"] == prompt.VERDICT_SCHEMA
    assert "temperature" not in kw  # rejected since API version 2026-10-21

    again = asyncio.run(judge.grade(MESSAGES, "No.", "[5] x"))
    assert again.cached and again.met is True and len(client.calls) == 1
    assert judge.usage.calls == 1 and judge.usage.cache_hits == 1
    assert judge.usage.cost("claude-haiku-5-5") == pytest.approx(
        (100 * 0.1 + 20 * 0.5) / 1e6
    )
    # repeat=1 is a fresh sample (test-retest), not a cache hit.
    assert cache_key(judge.config, "p", 0) != cache_key(judge.config, "p", 1)


async def _no_sleep(_):
    return None


def test_retries_refusal_and_truncation_then_gives_up(tmp_path, monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    ok = _message('{"explanation": "", "criteria_met": false}')
    judge, client = _judge(
        tmp_path, _message("", "refusal"), _message('{"expl', "max_tokens"), ok
    )
    v = asyncio.run(judge.grade(MESSAGES, "r", "[1] x"))
    assert v.met is False and len(client.calls) == 3

    judge, client = _judge(tmp_path, *[_message("", "refusal")] * 3)
    v = asyncio.run(judge.grade(MESSAGES, "r2", "[1] x"))
    assert v.met is None and "refusal" in v.error and judge.usage.failures == 1
    assert len(judge.cache) == 1  # failures are not cached


def test_client_errors_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    err = Exception("bad request")
    err.status_code = 400
    judge, _ = _judge(tmp_path, err)
    with pytest.raises(Exception, match="bad request"):
        asyncio.run(judge.grade(MESSAGES, "r", "[1] x"))
    overloaded = Exception("overloaded")
    overloaded.status_code = 529
    judge, client = _judge(
        tmp_path, overloaded, _message('{"explanation": "", "criteria_met": true}')
    )
    assert asyncio.run(judge.grade(MESSAGES, "r3", "[1] x")).met is True


def test_thinking_params():
    assert JudgeConfig.named("haiku").thinking_param() == {"type": "disabled"}
    assert JudgeConfig.named("sonnet").thinking_param() == {"type": "between_tools"}
    assert JudgeConfig.named("haiku", thinking=True).thinking_param() == {
        "type": "adaptive"
    }
    assert JudgeConfig.named("sonnet").model == "claude-sonnet-5-5"


def test_grade_criteria_with_fake_judge():
    criteria = [Criterion("p:0", "Says A", 5), Criterion("p:1", "Says B", -3)]
    judge = FakeJudge(rule=lambda p: "Says A" in p.split("# Rubric item")[1])
    verdicts = asyncio.run(judge.grade_criteria(MESSAGES, "A", criteria))
    assert [v.met for v in verdicts] == [True, False]
    assert "[-3] Says B" in judge.prompts[1]


def test_prompt_versions_pick_the_template():
    cfg = JudgeConfig.named("haiku", prompt_version=prompt.GEPA_V1_VERSION)
    judge = ClaudeJudge(cfg)
    assert judge.template == prompt.GEPA_V1_TEMPLATE
    assert ClaudeJudge(JudgeConfig()).template == prompt.GRADER_TEMPLATE
    with pytest.raises(ValueError, match="unknown prompt version"):
        JudgeConfig(prompt_version="nope")
    # The reward's default is the optimized template.
    assert prompt.REWARD_PROMPT_VERSION == prompt.GEPA_V1_VERSION
    p = prompt.render(MESSAGES, "No.", "[5] Says no.", judge.template)
    assert "physician reviewer" in p and p.endswith("Do not include any other text.")
    assert "user: Is smoking good for me?\n\nassistant: No." in p


def test_reward_judge_uses_the_reward_prompt(monkeypatch):
    from rubricrl.rewards import miles_rm

    monkeypatch.setattr(miles_rm, "_judges", {})
    monkeypatch.setattr(miles_rm, "_cache", object())
    monkeypatch.delenv("RUBRICRL_PROMPT", raising=False)

    async def configs():
        return miles_rm._judge().config

    assert asyncio.run(configs()).prompt_version == prompt.GEPA_V1_VERSION
    monkeypatch.setattr(miles_rm, "_judges", {})
    monkeypatch.setenv("RUBRICRL_PROMPT", prompt.PROMPT_VERSION)
    assert asyncio.run(configs()).prompt_version == prompt.PROMPT_VERSION
