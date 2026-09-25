from __future__ import annotations

import os

from textarena_env import TextArenaEnv

ENV_URL = os.getenv("OPENENV_URL", "http://localhost:8001")


class SyncWordleEnv:
    """Synchronous adapter around TextArenaEnv for TRL's _SupportsReset protocol."""

    def __init__(self, base_url: str):
        self._sync = TextArenaEnv(base_url=base_url).sync()

    def reset(self, **kwargs) -> str | None:
        result = self._sync.reset(**kwargs)
        obs = result.observation
        return str(obs) if obs is not None else None

    def step(self, action: str) -> str:
        result = self._sync.step(action)
        obs = result.observation
        return str(obs) if obs is not None else ""

    def close(self) -> None:
        """Disconnect the WebSocket session, freeing a server slot."""
        try:
            self._sync.close()
        except Exception:
            pass


def make_wordle_env() -> SyncWordleEnv:
    """Factory function for GRPOTrainer's environment_factory parameter."""
    return SyncWordleEnv(base_url=ENV_URL)
