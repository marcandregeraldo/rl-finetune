"""Miles <-> the consultation environment: the agent function and the reward hook.

Miles' agentic rollout (`agentic_tool_call.generate`, `--use-session-server`)
calls `run` once per sample. It plays one episode against the OpenEnv server:
reset to the sample's scenario, then policy reply -> `step` -> (patient reply)
until done. The policy is reached through Miles' session endpoint, so Miles
records the exact tokens and logprobs of every doctor turn (TITO); the patient's
turns are observations, outside the loss. The episode's result is merged into
`sample.metadata`, and `reward_func` (`--custom-rm-path`) reads the reward back,
as in Miles' own OpenEnv recipe (`examples/experimental/openenv`).

    --custom-generate-function-path \
        miles.rollout.generate_hub.agentic_tool_call.generate
    --custom-agent-function-path rubricrl.env.miles_agent.run
    --custom-rm-path rubricrl.env.miles_agent.reward_func
    --use-session-server --tito-model qwen35

It talks to the server with `env.wire` (OpenEnv's WebSocket protocol, no
`openenv` install: its server stack must stay out of Miles' Python).

Environment variables: `RUBRICRL_ENV_URL` (default http://localhost:8003, the
sidecar `modal/miles_rft.py` starts), `RUBRICRL_ENV_MODE` (single | patient;
the sample's `metadata.mode` wins), `RUBRICRL_LOG` (as for `miles_rm`).

Shipped into the Miles image (Python 3.12): keep the syntax 3.10-compatible.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from rubricrl.rewards import miles_rm

logger = logging.getLogger(__name__)
ENV_TIMEOUT_S = 900.0  # one step may run a patient turn plus ~10 judge calls
POLICY_TIMEOUT_S = 900.0


async def chat(base_url: str, messages: list[dict], request_kwargs: dict) -> str:
    """One policy turn through Miles' session server (OpenAI-compatible)."""
    payload = {
        "model": "default",
        "messages": messages,
        **request_kwargs,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with httpx.AsyncClient(timeout=POLICY_TIMEOUT_S) as client:
        r = await client.post(f"{base_url}/v1/chat/completions", json=payload)
        r.raise_for_status()
    return r.json()["choices"][0]["message"].get("content") or ""


async def episode(
    env, policy, prompt: list[dict], scenario_id: str, mode: str, half: str
) -> tuple[Any, int]:
    """Play one consultation; `policy(messages) -> reply`. `env` is a
    `wire.Session` (or anything with the same reset/step). Returns the last
    result and the number of doctor turns.

    The history sent to the policy keeps its own replies verbatim (the session
    server matches its token prefix on them); the environment cleans them for
    grading.
    """
    result = await env.reset(scenario_id=scenario_id, mode=mode, half=half)
    history, turns = list(prompt), 0
    while not result.done:
        reply = await policy(history)
        history.append({"role": "assistant", "content": reply})
        turns += 1
        result = await env.step(reply)
        last = result.observation["messages"][-1]
        if last["role"] == "user":  # the patient's answer
            history.append({"role": "user", "content": last["content"]})
    return result, turns


def record(obs: dict, prompt_id: str, is_eval: bool, turns: int) -> dict[str, Any]:
    """The episode as the reward log's record (the same keys as `miles_rm`)."""
    doctor = [m["content"] for m in obs["messages"] if m["role"] == "assistant"]
    score = obs.get("score")
    return {
        "prompt_id": prompt_id,
        "eval": is_eval,
        "mode": obs.get("mode"),
        "score": score,
        "reward": 0.0 if score is None else float(score),
        "axes": obs.get("axes", {}),
        "criteria": obs.get("criteria", []),
        "failed": obs.get("failed", 0),
        "judge": obs.get("judge", {}),
        "turns": turns,
        "end_reason": obs.get("end_reason", ""),
        "revealed_facts": obs.get("revealed_facts", []),
        "chars": sum(len(d) for d in doctor[-turns:]) if turns else 0,
        "response": doctor[-1] if doctor else "",
    }


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict | None = None,
    metadata: dict | None = None,
    **kwargs: Any,
) -> dict | None:
    """Miles' agent function: one episode; its record lands in sample.metadata.

    An environment failure returns None (no reward recorded) rather than a fake
    0.0, as in Miles' OpenEnv recipe.
    """
    from rubricrl.env.wire import Session

    metadata, request_kwargs = metadata or {}, request_kwargs or {}
    messages = (
        list(prompt)
        if isinstance(prompt, list)
        else [{"role": "user", "content": str(prompt)}]
    )
    is_eval = bool(metadata.get("eval"))
    mode = metadata.get("mode") or os.environ.get("RUBRICRL_ENV_MODE", "single")
    pid = metadata["prompt_id"]
    url = os.environ.get("RUBRICRL_ENV_URL", "http://localhost:8003")

    policy_s = 0.0

    async def policy(history: list[dict]) -> str:
        nonlocal policy_s
        t = time.monotonic()
        try:
            return await chat(base_url, history, request_kwargs)
        finally:
            policy_s += time.monotonic() - t

    t0 = time.monotonic()
    try:
        async with Session(url, timeout_s=ENV_TIMEOUT_S) as env:
            result, turns = await episode(
                env, policy, messages, pid, mode, "eval" if is_eval else "train"
            )
    except Exception as e:
        logger.error(f"consultation {pid} failed: {e}", exc_info=True)
        return None
    # The environment's share (judge, patient) of the episode's wall time, apart
    # from the policy's; Miles reports it as perf/non_generation_time.
    env_s = time.monotonic() - t0 - policy_s
    rec = record(result.observation, pid, is_eval, turns)
    rec |= {"policy_s": round(policy_s, 3), "env_s": round(env_s, 3)}
    return {"env": rec, "agent_metrics": {"total_tool_time": env_s}}


async def reward_func(args, samples, **kwargs) -> float | list[float]:
    """`--custom-rm-path`: the reward the episode recorded; logged like miles_rm."""
    batch = samples if isinstance(samples, list) else [samples]
    rewards = []
    for s in batch:
        rec = (s.metadata or {}).get("env")
        if rec is None:
            logger.warning("no environment result for %s: reward 0.0", s.metadata)
            rewards.append(0.0)
            continue
        rewards.append(rec["reward"])
        miles_rm._log(
            rec
            | {
                "time": time.time(),
                "tokens": getattr(s, "response_length", None),
                "status": str(getattr(s, "status", "")),
            }
        )
    return rewards if isinstance(samples, list) else rewards[0]
