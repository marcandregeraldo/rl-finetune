"""Patched OpenEnv server app that reads MAX_CONCURRENT_ENVS from the environment.

The upstream Wordle image's server/app.py calls create_app() without
max_concurrent_envs, defaulting to 1 session.  This file is volume-mounted
over the original at /app/env/server/app.py so the container respects the
MAX_CONCURRENT_ENVS env var.
"""

from __future__ import annotations

import inspect
import os

from openenv.core.env_server.http_server import create_app
from textarena_env.models import TextArenaAction, TextArenaObservation
from textarena_env.server.environment import TextArenaEnvironment
from textarena_env.server.gradio_ui import build_textarena_gradio_app

env_id = os.getenv("TEXTARENA_ENV_ID", "Wordle-v0")
num_players = int(os.getenv("TEXTARENA_NUM_PLAYERS", "1"))
max_turns_env = os.getenv("TEXTARENA_MAX_TURNS")
max_turns = int(max_turns_env) if max_turns_env is not None else None
download_nltk = os.getenv("TEXTARENA_DOWNLOAD_NLTK", "1") in {"1", "true", "True"}
max_concurrent_envs = int(os.getenv("MAX_CONCURRENT_ENVS", "1"))

# Each WebSocket session gets its own TextArenaEnvironment instance via the
# factory function, so concurrent sessions are safe.  The upstream class
# just doesn't declare the flag yet.
TextArenaEnvironment.SUPPORTS_CONCURRENT_SESSIONS = True


def _parse_env_kwargs(prefix: str = "TEXTARENA_KW_") -> dict[str, str]:
    return {
        key[len(prefix) :].lower(): value
        for key, value in os.environ.items()
        if key.startswith(prefix)
    }


extra_kwargs = _parse_env_kwargs()


def create_textarena_environment():
    return TextArenaEnvironment(
        env_id=env_id,
        num_players=num_players,
        max_turns=max_turns,
        download_nltk=download_nltk,
        env_kwargs=extra_kwargs,
    )


_sig = inspect.signature(create_app)
if "gradio_builder" in _sig.parameters:
    app = create_app(
        create_textarena_environment,
        TextArenaAction,
        TextArenaObservation,
        env_name="textarena_env",
        max_concurrent_envs=max_concurrent_envs,
        gradio_builder=build_textarena_gradio_app,
    )
else:
    app = create_app(
        create_textarena_environment,
        TextArenaAction,
        TextArenaObservation,
        env_name="textarena_env",
        max_concurrent_envs=max_concurrent_envs,
    )
