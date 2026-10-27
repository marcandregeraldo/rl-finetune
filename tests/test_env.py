"""The consultation environment over OpenEnv's HTTP/WebSocket server, the Miles
agent loop and reward hook; fake judge and fake patient, no network."""

import asyncio
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("openenv")
from fastapi.testclient import TestClient  # noqa: E402

from rubricrl.data import healthbench as hb  # noqa: E402
from rubricrl.env import miles_agent, server  # noqa: E402
from rubricrl.env.models import ConsultObservation, DoctorAction  # noqa: E402
from rubricrl.judge.client import FakeJudge  # noqa: E402
from rubricrl.sim.patient import PatientTurn  # noqa: E402
from rubricrl.sim.scenario import Fact, Scenario, ScenarioStore  # noqa: E402

OPENING = [{"role": "user", "content": "I have chest pain."}]


def _example():
    row = {
        "prompt_id": "p1",
        "prompt": OPENING,
        "rubrics": [
            {"criterion": f"good{i}", "points": 5, "tags": ["axis:accuracy"]}
            for i in range(4)
        ]
        + [{"criterion": "bad", "points": -5, "tags": ["axis:completeness"]}],
        "example_tags": ["theme:emergency_referrals"],
    }
    return hb.parse_example(row)


def _store() -> ScenarioStore:
    e = _example()
    halves = hb.criteria_split(e)
    single = Scenario.from_miles_row(hb.miles_row(e, halves, False), "test")
    patient = Scenario(
        id="p1",
        mode="patient",
        opening=OPENING,
        train_criteria=halves.train,
        heldout_criteria=halves.heldout,
        split="test",
        patient="A 50-year-old man.",
        hidden_facts=(Fact("f1", "He takes aspirin."),),
        end_condition="Once told whether to go to the ER.",
    )
    store = ScenarioStore()
    store.add(single)
    store.add(patient)
    return store


class FakePatient:
    """Answers each turn; ends after `end_after` doctor turns; reveals f1 when
    the doctor asks about medication."""

    def __init__(self, end_after=2):
        self.end_after, self.calls = end_after, []

    async def __call__(self, scenario, transcript):
        self.calls.append(list(transcript))
        doctor_turns = sum(m["role"] == "assistant" for m in transcript) - sum(
            m["role"] == "assistant" for m in scenario.opening
        )
        asked = "medication" in transcript[-1]["content"]
        return PatientTurn(
            "I take aspirin." if asked else "OK.",
            ["f1"] if asked else [],
            doctor_turns >= self.end_after,
        )


def _resources(rule=lambda p: True, patient=None):
    return server.Resources(
        store=_store(), judge=FakeJudge(rule=rule), patient=patient or FakePatient()
    )


def _item(prompt: str) -> str:
    return prompt.split("# Rubric item\n")[1].split("\n\n# Instructions")[0]


def test_single_mode_matches_the_in_process_reward():
    judge_rule = lambda p: "good0" in _item(p) or "good1" in _item(p)  # noqa: E731
    env = server.ConsultEnvironment(_resources(judge_rule))
    obs = asyncio.run(env.reset_async(scenario_id="p1", mode="single"))
    assert obs.messages == OPENING and not obs.done and obs.reward is None
    obs = asyncio.run(env.step_async(DoctorAction(message="Go to the ER.<|im_end|>")))
    assert obs.done and obs.end_reason == "single" and obs.turn == 1
    assert obs.messages[-1] == {"role": "assistant", "content": "Go to the ER."}
    # The same function as Miles' in-process reward, on the same half.
    store = _store()
    s = store.get("single", "p1")
    md = {
        "prompt_id": "p1",
        "messages": OPENING,
        "train_criteria": [c.to_dict() for c in s.train_criteria],
        "heldout_criteria": [c.to_dict() for c in s.heldout_criteria],
        "eval": False,
    }
    from rubricrl.rewards import miles_rm

    rec = asyncio.run(
        miles_rm.grade_sample(FakeJudge(rule=judge_rule), md, "Go to the ER.")
    )
    assert obs.reward == rec["reward"] and obs.criteria == rec["criteria"]
    with pytest.raises(RuntimeError, match="reset"):
        asyncio.run(env.step_async(DoctorAction(message="again")))


def test_eval_half_grades_the_held_out_criteria():
    seen = []
    env = server.ConsultEnvironment(_resources(lambda p: seen.append(_item(p)) or True))
    asyncio.run(env.reset_async(scenario_id="p1", mode="single", half="eval"))
    obs = asyncio.run(env.step_async(DoctorAction(message="x")))
    heldout = {str(c) for c in _store().get("single", "p1").heldout_criteria}
    assert set(seen) == heldout and {c["id"] for c in obs.criteria} == {
        c.id for c in _store().get("single", "p1").heldout_criteria
    }


def test_patient_mode_runs_turns_then_grades_the_whole_consultation():
    patient = FakePatient(end_after=2)
    judge = FakeJudge(rule=lambda p: True)
    env = server.ConsultEnvironment(
        server.Resources(store=_store(), judge=judge, patient=patient)
    )
    asyncio.run(env.reset_async(scenario_id="p1", mode="patient"))
    obs = asyncio.run(env.step_async(DoctorAction(message="Any medication?")))
    assert not obs.done and obs.reward == 0.0 and obs.revealed_facts == ["f1"]
    assert obs.messages[-1] == {"role": "user", "content": "I take aspirin."}
    obs = asyncio.run(env.step_async(DoctorAction(message="Go to the ER now.")))
    assert obs.done and obs.end_reason == "patient_ended" and obs.turn == 2
    # Graded once, on the whole consultation, with the trajectory prompt.
    assert len(judge.prompts) == len(_store().get("patient", "p1").train_criteria)
    p = judge.prompts[0]
    assert "the consultation starts here" in p and "Go to the ER now." in p
    assert "Any medication?" in p and obs.score is not None
    assert obs.reward == obs.score


def test_patient_mode_turn_limit():
    env = server.ConsultEnvironment(
        _resources(patient=FakePatient(end_after=99)), max_turns=3
    )
    asyncio.run(env.reset_async(scenario_id="p1", mode="patient"))
    for _ in range(2):
        assert not asyncio.run(env.step_async(DoctorAction(message="Hm?"))).done
    obs = asyncio.run(env.step_async(DoctorAction(message="Bye.")))
    assert obs.done and obs.end_reason == "turn_limit" and obs.turn == 3


def test_reset_draws_deterministically_and_validates():
    env = server.ConsultEnvironment(_resources())
    a = asyncio.run(env.reset_async(seed=1, mode="patient"))
    b = asyncio.run(env.reset_async(seed=1, mode="patient"))
    assert a.scenario_id == b.scenario_id == "p1"
    with pytest.raises(ValueError):
        asyncio.run(env.reset_async(scenario_id="p1", mode="nope"))


def test_openenv_server_http_and_websocket():
    client = TestClient(server.make_app(_resources()))
    assert client.get("/health").json()["status"] == "healthy"
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {"type": "reset", "data": {"scenario_id": "p1", "mode": "patient"}}
        )
        reset = ws.receive_json()["data"]
        assert reset["done"] is False and reset["observation"]["messages"] == OPENING
        ws.send_json({"type": "step", "data": {"message": "Any medication?"}})
        assert ws.receive_json()["data"]["done"] is False
        ws.send_json({"type": "step", "data": {"message": "Go to the ER."}})
        final = ws.receive_json()["data"]
    assert final["done"] is True and final["reward"] == final["observation"]["score"]
    obs = ConsultObservation(**final["observation"], reward=final["reward"], done=True)
    assert obs.revealed_facts == ["f1"]


class FakeEnv:
    """The wire client's interface over an in-process environment."""

    def __init__(self, env):
        self.env = env

    async def reset(self, **kw):
        obs = await self.env.reset_async(**kw)
        return SimpleNamespace(observation=obs.model_dump(), done=obs.done)

    async def step(self, message):
        obs = await self.env.step_async(DoctorAction(message=message))
        return SimpleNamespace(
            observation=obs.model_dump(), done=obs.done, reward=obs.reward
        )


def test_agent_episode_keeps_the_policy_history_verbatim():
    env = FakeEnv(server.ConsultEnvironment(_resources()))
    sent = []

    async def policy(history):
        sent.append([dict(m) for m in history])
        return "<think>\n\n</think>\n\nAny medication?" if len(sent) == 1 else "ER now."

    result, turns = asyncio.run(
        miles_agent.episode(env, policy, OPENING, "p1", "patient", "train")
    )
    assert turns == 2 and result.done
    # The policy sees its own raw reply (TITO prefix matching), then the patient's.
    assert sent[1][1]["content"].startswith("<think>")
    assert sent[1][2] == {"role": "user", "content": "I take aspirin."}
    rec = miles_agent.record(result.observation, "p1", False, turns)
    assert rec["turns"] == 2 and rec["revealed_facts"] == ["f1"]
    assert rec["reward"] == result.observation["score"]
    assert rec["response"] == "ER now."
    assert rec["judge"]["calls"] > 0 and "cost_usd" in rec["judge"]


def test_reward_func_reads_the_episode_and_logs_it(tmp_path, monkeypatch):
    log = tmp_path / "rewards.jsonl"
    monkeypatch.setenv("RUBRICRL_LOG", str(log))
    rec = {"reward": 0.4, "score": 0.4, "prompt_id": "p1", "eval": False}
    a = SimpleNamespace(metadata={"env": rec}, response_length=7)
    b = SimpleNamespace(metadata={}, response_length=3)  # the episode failed
    assert asyncio.run(miles_agent.reward_func(None, a)) == 0.4
    assert asyncio.run(miles_agent.reward_func(None, [a, b])) == [0.4, 0.0]
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(lines) == 2 and lines[0]["tokens"] == 7 and lines[0]["score"] == 0.4


def test_agent_run_returns_none_when_the_environment_fails(monkeypatch):
    monkeypatch.setenv("RUBRICRL_ENV_URL", "http://127.0.0.1:9")  # nothing listens
    out = asyncio.run(
        miles_agent.run("http://x", OPENING, {}, {"prompt_id": "p1", "eval": False})
    )
    assert out is None
