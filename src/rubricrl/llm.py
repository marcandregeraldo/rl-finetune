"""Structured-output Claude calls for the simulation side: the scenario builder
(`sim/scenarios.py`) and the patient simulator (`sim/patient.py`).

The same conventions as the judge (`judge/client.py`): a JSON schema enforced by
the API (`output_config.format`), no `temperature` (removed in API version
2026-10-21), retries on refusal / truncation / transient errors, and an SQLite
cache keyed on sha256(model, effort, thinking, version, prompt, schema, repeat),
so a rerun costs nothing and a simulation is reproducible.

Shipped into the Miles image (Python 3.12): keep the syntax 3.10-compatible.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

from rubricrl.judge.client import JudgeConfig, Usage, _is_client_error, api_key

DEFAULT_CACHE = Path.home() / ".cache" / "rubricrl" / "llm.sqlite"


class JSONCache:
    """SQLite, one row per structured output. Safe across coroutines and threads."""

    def __init__(self, path: Path | str = DEFAULT_CACHE):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS outputs (key TEXT PRIMARY KEY,"
                " model TEXT, output TEXT, input_tokens INTEGER,"
                " output_tokens INTEGER, created REAL)"
            )
            self._db.commit()

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._db.execute(
                "SELECT output FROM outputs WHERE key = ?", (key,)
            ).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, key: str, model: str, output: dict, tokens: tuple[int, int]):
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO outputs VALUES (?, ?, ?, ?, ?, ?)",
                (key, model, json.dumps(output), *tokens, time.time()),
            )
            self._db.commit()

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM outputs").fetchone()[0]

    def merge(self, other: Path | str) -> int:
        """Add the outputs of another cache file that this one lacks; how many."""
        before = len(self)
        with self._lock:
            self._db.execute("ATTACH DATABASE ? AS other", (str(other),))
            self._db.execute(
                "INSERT OR IGNORE INTO outputs SELECT * FROM other.outputs"
            )
            self._db.commit()
            self._db.execute("DETACH DATABASE other")
        return len(self) - before

    def snapshot(self, path: Path | str) -> None:
        """A consistent single-file copy (WAL included), for moving between hosts."""
        dst = sqlite3.connect(path)
        try:
            with self._lock:
                self._db.backup(dst)
        finally:
            dst.close()


class StructuredLLM:
    """`await llm(prompt, schema, version)` -> the parsed JSON object."""

    def __init__(
        self,
        config: JudgeConfig,
        cache: JSONCache | None = None,
        client=None,
    ):
        self.config = config
        self.cache = cache
        self.usage = Usage()
        self._client = client
        self._sem: asyncio.Semaphore | None = None

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=api_key(), max_retries=6)
        return self._client

    def key(self, prompt: str, schema: dict, version: str, repeat: int = 0) -> str:
        cfg = self.config
        parts = [cfg.model, cfg.effort, cfg.thinking, version, prompt, schema, repeat]
        return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()

    async def __call__(
        self, prompt: str, schema: dict, version: str, repeat: int = 0
    ) -> dict:
        key = self.key(prompt, schema, version, repeat)
        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            self.usage.cache_hits += 1
            return hit
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.config.concurrency)
        cfg, error = self.config, None
        for attempt in range(cfg.max_attempts):
            if attempt:
                await asyncio.sleep(2**attempt)
            async with self._sem:
                try:
                    msg = await self.client.messages.create(
                        model=cfg.model,
                        max_tokens=cfg.max_tokens,
                        messages=[{"role": "user", "content": prompt}],
                        thinking=cfg.thinking_param(),
                        output_config={
                            "effort": cfg.effort,
                            "format": {"type": "json_schema", "schema": schema},
                        },
                    )
                except Exception as e:  # anthropic errors, after the SDK's retries
                    if _is_client_error(e):
                        raise
                    error = f"{type(e).__name__}: {e}"
                    continue
            u = msg.usage
            tokens = (
                u.input_tokens
                + (u.cache_read_input_tokens or 0)
                + (u.cache_creation_input_tokens or 0),
                u.output_tokens,
            )
            self.usage.calls += 1
            self.usage.input_tokens += tokens[0]
            self.usage.output_tokens += tokens[1]
            if msg.stop_reason != "end_turn":
                error = f"stop_reason={msg.stop_reason}"
                continue
            text = "".join(b.text for b in msg.content if b.type == "text")
            try:
                output = json.loads(text)
            except json.JSONDecodeError as e:
                error = f"bad output: {e}"
                continue
            if self.cache is not None:
                self.cache.put(key, cfg.model, output, tokens)
            return output
        self.usage.failures += 1
        raise RuntimeError(f"{cfg.model}: no valid output after retries ({error})")


class FakeLLM:
    """For tests: `respond(prompt, schema) -> dict`; records every prompt."""

    def __init__(self, respond):
        self.respond = respond
        self.prompts: list[str] = []
        self.usage = Usage()

    async def __call__(self, prompt, schema, version, repeat=0) -> dict:
        self.prompts.append(prompt)
        self.usage.calls += 1
        return self.respond(prompt, schema)
