"""HealthBench consultations as an OpenEnv environment.

    uvicorn rubricrl.env.server:app --port 8003     # the OpenEnv HTTP/WebSocket server

An episode is one consultation (`reset(scenario_id, mode, half)`, then `step`s):

- **single** (v1): the HealthBench conversation as written; one doctor reply,
  graded on the physicians' criteria by the calibrated judge. The score is the
  same function as Miles' in-process reward (`rewards.miles_rm.grade_sample`).
- **patient** (v2): the conversation is the opening; each doctor reply gets the
  simulated patient's answer (`sim/patient.py`) until the patient ends the
  consultation or the turn limit; then the whole consultation is graded on the
  scenario's trajectory-level criteria (`judge/prompt.TRAJECTORY_TEMPLATE`).
  Intermediate steps carry reward 0: the reward is at the end.

`half="eval"` grades the held-out criteria instead of the reward's: the
reward-hacking detector, through the same interface.

The grading is an OpenEnv rubric (`ConsultRubric`): it scores the finished
observation and fills in its per-criterion breakdown. Judge and patient are
configured by environment variables, the same ones Miles' reward reads:
`RUBRICRL_JUDGE`, `RUBRICRL_EFFORT`, `RUBRICRL_THINKING`, `RUBRICRL_PROMPT` (single
mode; patient mode grades with the trajectory prompt), `RUBRICRL_CACHE`, and
`RUBRICRL_PATIENT` (default haiku), `RUBRICRL_MAX_TURNS` (default 6),
`RUBRICRL_DATA` (the Miles exports and their `scenarios/`; default data/miles),
`RUBRICRL_WEAVE` (a W&B project: one trace per finished episode).
"""

from __future__ import annotations

import asyncio
import os
import random
import uuid
from pathlib import Path
from typing import Any

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.rubrics import Rubric

from rubricrl.data.schema import Criterion
from rubricrl.env.models import ConsultObservation, ConsultState, DoctorAction
from rubricrl.judge.client import JudgeConfig
from rubricrl.rewards import miles_rm
from rubricrl.rewards.rubric import axis_scores, score
from rubricrl.sim.patient import PatientSimulator
from rubricrl.sim.scenario import MODES, Scenario, ScenarioStore

MAX_TURNS = int(os.environ.get("RUBRICRL_MAX_TURNS", "6"))


# --- Shared resources (one per server process) ----------------------------------


class Resources:
    """The scenarios, judge and patient every session shares; lazily built."""

    def __init__(self, store=None, judge=None, patient=None, tracer=None):
        self._store, self._judge, self._patient = store, judge, patient
        self.tracer = tracer

    @property
    def store(self) -> ScenarioStore:
        if self._store is None:
            data = Path(os.environ.get("RUBRICRL_DATA", "data/miles"))
            self._store = ScenarioStore.load(data)
        return self._store

    @property
    def judge(self):
        return self._judge if self._judge is not None else miles_rm._judge()

    @property
    def patient(self) -> PatientSimulator:
        if self._patient is None:
            from rubricrl.llm import JSONCache, StructuredLLM

            config = JudgeConfig.named(
                os.environ.get("RUBRICRL_PATIENT", "haiku"),
                max_tokens=2048,
                concurrency=int(os.environ.get("RUBRICRL_CONCURRENCY", "64")),
            )
            self._patient = PatientSimulator(StructuredLLM(config, JSONCache()))
        return self._patient


RESOURCES = Resources()


# --- The rubric -------------------------------------------------------------------


class ConsultRubric(Rubric):
    """The rubric score of a finished consultation; 0 until it is finished.

    Reads the scenario and transcript off the observation, grades the episode's
    half of the criteria, writes the breakdown back onto the observation, and
    returns the reward (the unclipped score; 0.0 when no verdict was usable).
    """

    def __init__(self, resources: Resources):
        super().__init__()
        self.resources = resources

    async def forward(self, action: DoctorAction, obs: ConsultObservation) -> float:
        if not obs.done:
            return 0.0
        scenario = self.resources.store.get(obs.mode, obs.scenario_id)
        criteria = scenario.criteria(obs.half)
        judge = self.resources.judge
        if obs.mode == "single":
            metadata = {
                "prompt_id": scenario.id,
                "messages": scenario.opening,
                "train_criteria": [c.to_dict() for c in scenario.train_criteria],
                "heldout_criteria": [c.to_dict() for c in scenario.heldout_criteria],
                "eval": obs.half == "eval",
            }
            response = obs.messages[-1]["content"]
            record = await miles_rm.grade_sample(judge, metadata, response)
            obs.criteria = record["criteria"]
            obs.score, obs.axes, obs.failed, obs.judge = (
                record["score"],
                record["axes"],
                record["failed"],
                record["judge"],
            )
            return record["reward"]
        turns = obs.messages[len(scenario.opening) :]
        verdicts = await judge.grade_transcript(scenario.opening, turns, criteria)
        met = [v.met for v in verdicts]
        obs.criteria = [_verdict(c, v) for c, v in zip(criteria, verdicts, strict=True)]
        obs.score = score(criteria, met)
        obs.axes = axis_scores(criteria, met)
        obs.failed = sum(m is None for m in met)
        obs.judge = miles_rm.judge_usage(judge, verdicts)
        return 0.0 if obs.score is None else obs.score


def _verdict(c: Criterion, v) -> dict[str, Any]:
    source = next((t[7:] for t in c.tags if t.startswith("source:")), c.id)
    return {
        "id": c.id,
        "source": source,
        "text": c.text,
        "points": c.points,
        "axis": c.axis,
        "met": v.met,
        "explanation": v.explanation,
    }


# --- The environment ------------------------------------------------------------


class ConsultEnvironment(Environment[DoctorAction, ConsultObservation, ConsultState]):
    SUPPORTS_CONCURRENT_SESSIONS = True  # instances share only safe resources

    def __init__(self, resources: Resources | None = None, max_turns: int = MAX_TURNS):
        self.resources = resources or RESOURCES
        super().__init__(rubric=ConsultRubric(self.resources))
        self.max_turns = max_turns
        self._state = ConsultState(episode_id=str(uuid.uuid4()), step_count=0)
        self._scenario: Scenario | None = None
        self._messages: list[dict] = []
        self._revealed: list[str] = []
        self._reveals: list[dict] = []

    @property
    def state(self) -> ConsultState:
        return self._state

    def _observation(self, **kw) -> ConsultObservation:
        s = self._state
        return ConsultObservation(
            scenario_id=s.scenario_id,
            mode=s.mode,
            half=s.half,
            messages=list(self._messages),
            turn=s.turn,
            revealed_facts=list(self._revealed),
            reveals=list(self._reveals),
            **kw,
        )

    async def reset_async(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        scenario_id: str | None = None,
        mode: str = "single",
        half: str = "train",
        split: str | None = None,
        **kwargs: Any,
    ) -> ConsultObservation:
        if mode not in MODES or half not in ("train", "eval"):
            raise ValueError(f"mode in {MODES}, half in ('train', 'eval')")
        store = self.resources.store
        if scenario_id is None:  # a deterministic draw, for demos and smoke tests
            ids = store.ids(mode, split)
            if not ids:
                raise ValueError(f"no {mode} scenarios (split={split})")
            scenario_id = random.Random(seed).choice(ids)
        await self._reset_rubric_async()
        self._scenario = store.get(mode, scenario_id)
        self._messages = list(self._scenario.opening)
        self._revealed, self._reveals = [], []
        self._state = ConsultState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            scenario_id=scenario_id,
            mode=mode,
            half=half,
        )
        return self._observation(done=False, reward=None)

    def reset(self, seed=None, episode_id=None, **kwargs) -> ConsultObservation:
        return asyncio.run(self.reset_async(seed, episode_id, **kwargs))

    async def step_async(
        self, action: DoctorAction, timeout_s: float | None = None, **kwargs: Any
    ) -> ConsultObservation:
        s = self._state
        if self._scenario is None or s.done:
            raise RuntimeError("call reset() before step(), and not after done")
        s.step_count += 1
        s.turn += 1
        reply = miles_rm.clean_response(action.message)
        self._messages.append({"role": "assistant", "content": reply})
        end_reason = ""
        if s.mode == "single":
            end_reason = "single"
        elif s.turn >= self.max_turns:
            end_reason = "turn_limit"
        else:
            turn = await self.resources.patient(self._scenario, self._messages)
            self._messages.append({"role": "user", "content": turn.reply})
            new = [f for f in turn.revealed if f not in self._revealed]
            self._revealed += new
            if new:
                self._reveals.append({"index": len(self._messages) - 1, "facts": new})
            if turn.end:
                end_reason = "patient_ended"
        s.done = bool(end_reason)
        obs = self._observation(done=s.done, reward=0.0, end_reason=end_reason)
        obs.reward = await self._apply_rubric_async(action, obs)
        if s.done and self.resources.tracer is not None:
            self.resources.tracer(self._scenario, obs)
        return obs

    def step(
        self, action: DoctorAction, timeout_s=None, **kwargs
    ) -> ConsultObservation:
        return asyncio.run(self.step_async(action, timeout_s, **kwargs))


def make_app(resources: Resources | None = None, max_concurrent_envs: int = 256):
    """The OpenEnv FastAPI app; one environment instance per session."""
    resources = resources or RESOURCES
    if resources.tracer is None and (project := os.environ.get("RUBRICRL_WEAVE")):
        from rubricrl import observe

        resources.tracer = observe.episode_tracer(project)
    return create_app(
        lambda: ConsultEnvironment(resources),
        DoctorAction,
        ConsultObservation,
        env_name="healthbench_consult",
        max_concurrent_envs=max_concurrent_envs,
        state_cls=ConsultState,
    )


app = make_app()
