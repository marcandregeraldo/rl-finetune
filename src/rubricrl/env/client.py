"""The OpenEnv client for the consultation environment (WebSocket session).

    async with ConsultEnv(base_url="http://localhost:8003") as env:
        result = await env.reset(scenario_id=pid, mode="single", half="train")
        result = await env.step(DoctorAction(message=reply))
        result.reward, result.observation.criteria

Shipped into the Miles image (Python 3.12): keep the syntax 3.10-compatible.
"""

from __future__ import annotations

from typing import Any

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from rubricrl.env.models import ConsultObservation, ConsultState, DoctorAction


class ConsultEnv(EnvClient[DoctorAction, ConsultObservation, ConsultState]):
    def _step_payload(self, action: DoctorAction) -> dict[str, Any]:
        return action.model_dump(exclude={"metadata"})

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[ConsultObservation]:
        obs = ConsultObservation(
            **payload.get("observation", {}),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )
        return StepResult(observation=obs, reward=obs.reward, done=obs.done)

    def _parse_state(self, payload: dict[str, Any]) -> ConsultState:
        return ConsultState(**payload)
