from __future__ import annotations

import os

from openenv_textarena import WordleEnv

ENV_URL = os.getenv("OPENENV_URL", "http://localhost:8001")


def make_wordle_env() -> WordleEnv:
    """Factory function for GRPOTrainer's environment_factory parameter."""
    return WordleEnv(base_url=ENV_URL)
