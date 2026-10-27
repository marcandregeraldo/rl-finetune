"""Expert data -> simulation: the scenario mapping and the patient simulator.

Fake LLMs, no network.
"""

import asyncio
import json

import pytest

from rubricrl.data import healthbench as hb
from rubricrl.judge import prompt
from rubricrl.llm import FakeLLM
from rubricrl.sim import patient, scenarios
from rubricrl.sim.scenario import Fact, Scenario, ScenarioStore

OPENING = [{"role": "user", "content": "My dad has chest pain."}]


def _example(pid="p1", theme="emergency_referrals", split=None):
    rubrics = [
        {"criterion": "Asks how long the pain has lasted.", "points": 5,
         "tags": ["axis:context_awareness", "level:example"]},
        {"criterion": "Tells them to call emergency services.", "points": 9,
         "tags": ["axis:accuracy", "level:example"]},
        {"criterion": "Response is under 100 words.", "points": 2,
         "tags": ["axis:communication_quality", "level:example"]},
        {"criterion": "Recommends aspirin without asking about allergies.",
         "points": -6, "tags": ["axis:accuracy", "level:example"]},
        {"criterion": "Seeks the most informative context.", "points": 5,
         "tags": ["axis:context_awareness", "level:cluster"]},
    ]  # fmt: skip
    e = hb.parse_example(
        {
            "prompt_id": pid,
            "prompt": OPENING,
            "rubrics": rubrics,
            "example_tags": [f"theme:{theme}"],
        }  # fmt: skip
    )
    return e if split is None else hb.Example(**{**e.__dict__, "split": split})


def _output(e, drop="p1:2"):
    criteria = []
    for c in e.rubric:
        if c.level == "cluster":
            continue
        if c.id == drop:
            criteria.append({"id": c.id, "status": "dropped", "text": "",
                             "reason": "about one reply's length",
                             "met_means": "good"})  # fmt: skip
        elif "emergency" in c.text:
            criteria.append({"id": c.id, "status": "kept", "text": "ignored",
                             "reason": "holds over a consultation",
                             "met_means": "good"})  # fmt: skip
        else:
            text, means = f"By the end: {c.text}", "bad" if c.points < 0 else "good"
            criteria.append({"id": c.id, "status": "rewritten", "text": text,
                             "reason": "one reply", "met_means": means})  # fmt: skip
    return {
        "patient": "A worried daughter, 30.",
        "hidden_facts": [{"id": "x", "text": "He takes warfarin."}],
        "end_condition": "Once she knows whether to call 911.",
        "criteria": criteria,
    }


def test_render_lists_only_example_level_criteria():
    p = scenarios.render(_example())
    assert "[p1:0] (+5, context_awareness) Asks how long" in p
    assert "[p1:3] (-6, accuracy)" in p
    assert "Seeks the most informative context" not in p  # consensus: kept as is
    assert "My dad has chest pain." in p


def test_build_scenario_maps_criteria_with_provenance():
    e = _example()
    s = scenarios.build_scenario(e, _output(e), "test")
    crit = {c.id: c for c in s.train_criteria + s.heldout_criteria}
    # Points, sign and axis come from the physicians, never from the model.
    assert crit["p1:0~t"].points == 5 and crit["p1:0~t"].axis == "context_awareness"
    assert crit["p1:0~t"].text.startswith("By the end:")
    assert (
        "source:p1:0" in crit["p1:0~t"].tags
        and "status:rewritten" in crit["p1:0~t"].tags
    )
    assert crit["p1:3~t"].points == -6  # negatives survive, rewritten
    assert (
        crit["p1:1"].text == "Tells them to call emergency services."
    )  # kept verbatim
    assert crit["p1:4"].text == "Seeks the most informative context."  # consensus
    assert "p1:2" not in crit and "p1:2~t" not in crit  # dropped
    statuses = {m["source"]: m["status"] for m in s.mapping}
    assert statuses == {"p1:0": "rewritten", "p1:1": "kept", "p1:2": "dropped",
                        "p1:3": "rewritten", "p1:4": "kept"}  # fmt: skip
    # Negatives in both halves, positives split, as for v1.
    assert {c.id for c in s.train_criteria if c.points < 0} == {"p1:3~t"}
    assert {c.id for c in s.heldout_criteria if c.points < 0} == {"p1:3~t"}
    assert s.hidden_facts == (Fact("f1", "He takes warfarin."),)  # ids renumbered
    assert s.opening == OPENING and s.mode == "patient"


def test_build_scenario_rejects_incomplete_or_unknown_decisions():
    e = _example()
    out = _output(e)
    out["criteria"] = out["criteria"][1:]
    with pytest.raises(ValueError, match="missing"):
        scenarios.build_scenario(e, out, "test")
    out = _output(e)
    out["criteria"].append({"id": "p1:4", "status": "kept", "text": "", "reason": ""})
    with pytest.raises(ValueError, match="unknown"):
        scenarios.build_scenario(e, out, "test")
    out = _output(e)
    out["criteria"][0]["text"] = " "
    with pytest.raises(ValueError, match="no text"):
        scenarios.build_scenario(e, out, "test")


def test_build_counts_failures_and_writes_miles_rows(tmp_path):
    good, bad = _example("p1"), _example("p2")

    def respond(p, schema):
        e = good if "[p1:0]" in p else bad
        out = _output(e, drop=f"{e.id}:2")
        if e is bad:
            out["criteria"] = []
        return out

    built, failures = asyncio.run(
        scenarios.build([good, bad], "test", FakeLLM(respond))
    )
    assert [s.id for s in built] == ["p1"] and failures == {"ValueError": 1}
    scenarios.save(built, tmp_path / "scenarios" / "test.jsonl")
    rows_path = scenarios.save_miles(built, "test", tmp_path)
    row = json.loads(rows_path.read_text())
    assert row["prompt"] == OPENING and row["metadata"] == {
        "prompt_id": "p1", "eval": True, "mode": "patient",
        "themes": ["emergency_referrals"],
    }  # fmt: skip
    again = scenarios.load(tmp_path / "scenarios" / "test.jsonl")
    assert again == built
    store = ScenarioStore.load(tmp_path)
    assert store.ids("patient", "test") == ["p1"]
    stats = scenarios.stats(built)
    assert stats["statuses"] == {
        "rewritten": 2,
        "kept": 1,
        "dropped": 1,
        "consensus": 1,
    }


def test_select_dialogue_themes_from_the_split():
    exs = [_example(f"p{i}", theme, "train")
           for i, theme in enumerate(["hedging", "health_data_tasks"] * 3)]  # fmt: skip
    picked = scenarios.select(exs, "train", 10)
    assert {e.themes[0] for e in picked} == {"hedging"}
    assert scenarios.select(exs, "val", 10) == []


def test_scenario_round_trips_and_selects_halves():
    e = _example()
    s = scenarios.build_scenario(e, _output(e), "test")
    assert Scenario.from_dict(json.loads(json.dumps(s.to_dict()))) == s
    assert s.criteria("eval") == s.heldout_criteria
    assert s.criteria("train") == s.train_criteria


def test_patient_prompt_and_parsing():
    e = _example()
    s = scenarios.build_scenario(e, _output(e), "test")
    transcript = [*OPENING, {"role": "assistant", "content": "Any medicines?"}]
    seen = []

    def respond(p, schema):
        seen.append(p)
        return {"reply": "He takes warfarin. [END]", "revealed_fact_ids": ["f1", "f9"],
                "end": True}  # fmt: skip

    turn = asyncio.run(patient.PatientSimulator(FakeLLM(respond))(s, transcript))
    assert turn.reply == "He takes warfarin." and turn.end
    assert turn.revealed == ["f1"]  # unknown ids dropped
    p = seen[0]
    assert "A worried daughter, 30." in p and "- f1: He takes warfarin." in p
    assert "assistant: Any medicines?" in p and "Once she knows" in p


def test_trajectory_prompt_marks_where_grading_starts():
    p = prompt.render_transcript(
        OPENING, [{"role": "assistant", "content": "Call 911."}], "[9] Says call 911."
    )
    head, tail = p.rsplit(prompt.TRAJECTORY_START, 1)  # also named in the instructions
    assert "My dad has chest pain." in head and "assistant: Call 911." in tail
    assert "taken together as one consultation" in p
    assert prompt.TRAJECTORY_TEMPLATE.count("<<conversation>>") == 1


def test_polarity_check_drops_flipped_criteria():
    """A negative criterion rewritten as the good behaviour would flip the reward's
    sign for it (seen with scenario-v1): dropped, with the reason recorded."""
    e = _example()
    out = _output(e)
    flipped = next(d for d in out["criteria"] if d["id"] == "p1:3")
    flipped["text"], flipped["met_means"] = "Asks about allergies first.", "good"
    s = scenarios.build_scenario(e, out, "test")
    ids = {c.id for c in s.train_criteria + s.heldout_criteria}
    assert "p1:3~t" not in ids
    m = next(m for m in s.mapping if m["source"] == "p1:3")
    assert m["status"] == "dropped" and m["reason"].startswith("polarity check failed")


def test_fidelity_compares_original_and_mapped_verdicts(monkeypatch):
    from rubricrl.judge.client import FakeJudge
    from rubricrl.sim import checks

    e = _example()
    s = scenarios.build_scenario(e, _output(e), "test")
    monkeypatch.setattr(checks.healthbench, "load_examples", lambda: [e])

    def rule(p):  # the mapped "By the end:" version of p1:0 disagrees
        item = p.rsplit("# Rubric item\n", 1)[1].split("\n\n# Instructions")[0]
        return not item.startswith("[5] By the end: Asks")

    out = asyncio.run(checks.fidelity([s], {"p1": "Call 911 now."}, FakeJudge(rule)))
    assert out["pairs"] == 2 and out["agreement"] == 0.5  # p1:0 and p1:3 rewritten
    (d,) = out["disagreements"]
    assert d["original"].startswith("Asks how long") and d["mapped_met"] is False
    assert out["agreement_by_sign"] == {"positive": 0.0, "negative": 1.0}


def test_doctors_check_plays_full_consultations():
    pytest.importorskip("openenv")
    from rubricrl.env import server
    from rubricrl.judge.client import FakeJudge
    from rubricrl.sim import checks
    from rubricrl.sim.patient import PatientTurn
    from rubricrl.sim.scenario import ScenarioStore

    e = _example()
    s = scenarios.build_scenario(e, _output(e), "test")
    store = ScenarioStore()
    store.add(s)

    async def patient(scenario, transcript):
        asked = "?" in transcript[-1]["content"]
        doctor_turns = sum(m["role"] == "assistant" for m in transcript)
        return PatientTurn("Yes, warfarin." if asked else "OK.",
                           ["f1"] if asked else [], doctor_turns >= 2)  # fmt: skip

    judge = FakeJudge(rule=lambda p: "Any medicines?" in p)  # rewards asking
    resources = server.Resources(store=store, judge=judge, patient=patient)

    async def careful(messages):
        return "Any medicines?" if len(messages) == 1 else "Call 911."

    async def terse(messages):
        return "Rest."

    out = asyncio.run(
        checks.doctors(
            [s],
            lambda: server.ConsultEnvironment(resources),
            {"careful": careful, "terse": terse},
        )  # fmt: skip
    )
    assert (
        out["careful"]["facts_elicited"] == 1.0 and out["terse"]["facts_elicited"] == 0
    )
    assert out["careful"]["unprompted_reveal_rate"] == 0.0  # asked, then revealed
    assert out["careful"]["episodes"][0]["reveals"] == [{"index": 2, "facts": ["f1"]}]
    assert out["careful"]["turns_mean"] == 2 and out["terse"]["end_reasons"] == {
        "patient_ended": 1
    }
    assert out["careful"]["episodes"][0]["messages"][-2]["content"] == "Call 911."
