"""A minimal OpenEnv WebSocket client, for processes that can't install `openenv`.

The `openenv` package imports its whole server stack (FastAPI, gradio, fastmcp,
pandas, ...) even for the client. Miles' Python must not get those, so the agent
function speaks OpenEnv's session protocol directly: one WebSocket per episode
at `/ws`; `{"type": "reset" | "step", "data": {...}}` in, `{"type":
"observation", "data": {"observation", "reward", "done"}}` or `{"type": "error"}`
out; `{"type": "close"}` at the end. Results are plain dicts, the same fields as
`env.models.ConsultObservation`. `env.client.ConsultEnv` is the typed client
where `openenv` is installed.

Needs only `websockets`. Shipped into the Miles image (Python 3.12): keep the
syntax 3.10-compatible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Result:
    observation: dict[str, Any] = field(default_factory=dict)
    reward: float | None = None
    done: bool = False


class Session:
    """`async with Session(url) as env: await env.reset(...); await env.step(msg)`"""

    def __init__(self, base_url: str, timeout_s: float = 900.0):
        self.url = base_url.replace("http://", "ws://").replace("https://", "wss://")
        self.url = self.url.rstrip("/") + "/ws"
        self.timeout_s = timeout_s
        self._ws = None

    async def __aenter__(self) -> Session:
        from websockets.asyncio.client import connect

        self._ws = await connect(
            self.url, open_timeout=30, max_size=100 * 2**20, ping_interval=20
        )
        return self

    async def __aexit__(self, *exc) -> None:
        if self._ws is not None:
            try:
                await self._ws.send(json.dumps({"type": "close"}))
            except Exception:
                pass
            await self._ws.close()

    async def _call(self, kind: str, data: dict[str, Any]) -> Result:
        import asyncio

        await self._ws.send(json.dumps({"type": kind, "data": data}))
        msg = json.loads(await asyncio.wait_for(self._ws.recv(), self.timeout_s))
        if msg.get("type") == "error":
            raise RuntimeError(f"environment error: {msg.get('data')}")
        payload = msg.get("data", {})
        return Result(
            observation=payload.get("observation", {}),
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    async def reset(self, **kwargs: Any) -> Result:
        return await self._call("reset", kwargs)

    async def step(self, message: str) -> Result:
        return await self._call("step", {"message": message})
