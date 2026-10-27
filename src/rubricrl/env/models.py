"""The consultation environment's typed messages (OpenEnv Action / Observation /
State, pydantic models).

The policy plays the doctor (the assistant): each action is one reply. The
observation is the transcript so far; on the last step it also carries the
graded result: the rubric score (the reward), per-axis scores and every
criterion's verdict and explanation, so a trainer or an eval can log why an
episode scored what it did.

Shipped into the Miles image (Python 3.12): keep the syntax 3.10-compatible.
"""

from __future__ import annotations

from typing import Any

from openenv.core.env_server.types import Action, Observation, State


class DoctorAction(Action):
    message: str


class ConsultObservation(Observation):
    scenario_id: str = ""
    mode: str = "single"  # "single" (v1) | "patient" (v2)
    half: str = "train"  # "train": the reward's criteria | "eval": held out
    messages: list[dict[str, Any]] = []  # opening + consultation so far
    turn: int = 0  # doctor replies so far
    # On the last step:
    score: float | None = None  # unclipped rubric score; reward = score or 0.0
    axes: dict[str, float] = {}
    criteria: list[dict[str, Any]] = []  # id, text, points, met, explanation
    failed: int = 0  # verdicts the judge could not give
    judge: dict[str, Any] = {}  # grading cost: calls, cache hits, tokens, cost_usd
    revealed_facts: list[str] = []  # patient mode: hidden facts drawn out
    reveals: list[dict[str, Any]] = []  # per patient turn: message index, fact ids
    end_reason: str = ""  # "single" | "patient_ended" | "turn_limit"


class ConsultState(State):
    scenario_id: str = ""
    mode: str = "single"
    half: str = "train"
    turn: int = 0
    done: bool = False
