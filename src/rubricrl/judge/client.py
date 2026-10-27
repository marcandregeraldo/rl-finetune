"""The rubric judge: Claude grades one (conversation, response, criterion) per call.

- **Verdicts are hard booleans.** The API has no logprobs and no `temperature`
  (removed in API version 2026-10-21), so a verdict is a sample: test-retest
  (`repeat`) measures the noise.
- **Structured output** (`output_config.format`) guarantees the
  `{explanation, criteria_met}` JSON, so there is no parse-and-retry loop on bad
  JSON; the retry loop here covers refusals, truncation and exhausted API retries.
  The SDK itself retries 429 / 5xx / connection errors with backoff (`max_retries`).
- **Thinking off, effort low** by default: the "explanation" field, written before
  the verdict, is the reasoning (as in simple-evals). Sonnet 5.5 rejects
  `{"type": "disabled"}`; it turns thinking off with `{"type": "between_tools"}`.
- **Every verdict is cached** in SQLite, keyed on sha256(model, effort, thinking,
  prompt version, the full grader prompt, repeat). A rerun costs nothing.
- **API key:** `ANTHROPIC_API_KEY`, else (macOS) the Keychain item
  `anthropic-api-key`. Never logged or written anywhere.

Shipped into the Miles image (Python 3.12): keep the syntax 3.10-compatible.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from rubricrl.data.schema import Criterion
from rubricrl.judge.prompt import (
    GRADER_TEMPLATE,
    PROMPT_VERSION,
    TEMPLATES,
    VERDICT_SCHEMA,
    render,
    render_transcript,
)

MODELS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5"}
# $ per million tokens (input, output), prompts under 100K tokens.
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00)}
DEFAULT_CACHE = Path.home() / ".cache" / "rubricrl" / "judge.sqlite"


def api_key() -> str:
    if key := os.environ.get("ANTHROPIC_API_KEY"):
        return key
    if shutil.which("security"):
        r = subprocess.run(
            ["security", "find-generic-password", "-s", "anthropic-api-key", "-w"],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    raise RuntimeError(
        "No Anthropic API key: set ANTHROPIC_API_KEY or add the Keychain item "
        "'anthropic-api-key'"
    )


@dataclass(frozen=True)
class JudgeConfig:
    model: str = MODELS["haiku"]
    effort: str = "low"
    thinking: bool = False
    max_tokens: int = 4096  # a cap; thinking needs the room
    concurrency: int = 32  # requests in flight
    max_attempts: int = 3  # per verdict, on refusal / truncation / API errors
    prompt_version: str = PROMPT_VERSION  # the response template (prompt.TEMPLATES)

    def __post_init__(self):
        if self.prompt_version not in TEMPLATES:
            raise ValueError(
                f"unknown prompt version {self.prompt_version!r}: {sorted(TEMPLATES)}"
            )

    @property
    def template(self) -> str:
        return TEMPLATES[self.prompt_version]

    @classmethod
    def named(cls, name: str, **kw) -> JudgeConfig:
        """'haiku' | 'sonnet' | a model id."""
        return cls(model=MODELS.get(name, name), **kw)

    def thinking_param(self) -> dict:
        if self.thinking:
            return {"type": "adaptive"}
        if self.model.startswith("claude-sonnet-5-5"):
            return {"type": "between_tools"}  # no tools, so no thinking
        return {"type": "disabled"}


@dataclass
class Verdict:
    met: bool | None  # None: no verdict after max_attempts (see `error`)
    explanation: str = ""
    cached: bool = False
    error: str | None = None
    input_tokens: int = 0  # of the API call that produced it (cached: the original)
    output_tokens: int = 0


@dataclass
class Usage:
    calls: int = 0  # API calls that returned
    cache_hits: int = 0
    failures: int = 0  # verdicts given up on
    input_tokens: int = 0
    output_tokens: int = 0

    def cost(self, model: str) -> float:
        p_in, p_out = PRICES.get(model, (0.0, 0.0))
        return (self.input_tokens * p_in + self.output_tokens * p_out) / 1e6


class VerdictCache:
    """SQLite, one row per verdict. Safe to share between coroutines and threads."""

    def __init__(self, path: Path | str = DEFAULT_CACHE):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS verdicts (key TEXT PRIMARY KEY, model TEXT,"
                " met INTEGER, explanation TEXT, input_tokens INTEGER,"
                " output_tokens INTEGER, created REAL)"
            )
            self._db.commit()

    def get(self, key: str) -> Verdict | None:
        with self._lock:
            row = self._db.execute(
                "SELECT met, explanation, input_tokens, output_tokens FROM verdicts"
                " WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        met, explanation, tokens_in, tokens_out = row
        return Verdict(
            bool(met), explanation, cached=True,
            input_tokens=tokens_in or 0, output_tokens=tokens_out or 0,
        )  # fmt: skip

    def put(self, key: str, model: str, v: Verdict, tokens: tuple[int, int]) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO verdicts (key, model, met, explanation,"
                " input_tokens, output_tokens, created) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (key, model, int(bool(v.met)), v.explanation, *tokens, time.time()),
            )
            self._db.commit()

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM verdicts").fetchone()[0]

    def merge(self, other: Path | str) -> int:
        """Add the verdicts of another cache file that this one lacks; how many."""
        before = len(self)
        with self._lock:
            self._db.execute("ATTACH DATABASE ? AS other", (str(other),))
            self._db.execute(
                "INSERT OR IGNORE INTO verdicts SELECT key, model, met, explanation,"
                " input_tokens, output_tokens, created FROM other.verdicts"
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


def cache_key(config: JudgeConfig, prompt: str, repeat: int = 0) -> str:
    """The full prompt is in the key, so the template is too. PROMPT_VERSION stays
    a constant here: keys made before templates had versions still match."""
    parts = [
        config.model,
        config.effort,
        config.thinking,
        PROMPT_VERSION,
        prompt,
        repeat,
    ]
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


class _Grader:
    """What every judge offers on top of `grade_prompt`."""

    template: str = GRADER_TEMPLATE  # the response template `grade` renders

    async def grade_prompt(self, prompt: str, repeat: int = 0) -> Verdict:
        raise NotImplementedError

    async def grade(
        self, messages: list[dict], response: str, rubric_item: str, repeat: int = 0
    ) -> Verdict:
        prompt = render(messages, response, rubric_item, self.template)
        return await self.grade_prompt(prompt, repeat)

    async def grade_criteria(
        self,
        messages: list[dict],
        response: str,
        criteria: Sequence[Criterion],
        repeat: int = 0,
    ) -> list[Verdict]:
        return list(
            await asyncio.gather(
                *(self.grade(messages, response, str(c), repeat) for c in criteria)
            )
        )

    async def grade_transcript(
        self,
        opening: list[dict],
        turns: list[dict],
        criteria: Sequence[Criterion],
        repeat: int = 0,
    ) -> list[Verdict]:
        """Each criterion over a whole simulated consultation (trajectory-v1)."""
        return list(
            await asyncio.gather(
                *(
                    self.grade_prompt(render_transcript(opening, turns, str(c)), repeat)
                    for c in criteria
                )
            )
        )


class ClaudeJudge(_Grader):
    def __init__(
        self,
        config: JudgeConfig | None = None,
        cache: VerdictCache | None = None,
        client=None,
    ):
        self.config = config or JudgeConfig()
        self.template = self.config.template
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

    async def grade_prompt(self, prompt: str, repeat: int = 0) -> Verdict:
        key = cache_key(self.config, prompt, repeat)
        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            self.usage.cache_hits += 1
            return hit
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.config.concurrency)
        error = None
        for attempt in range(self.config.max_attempts):
            if attempt:
                await asyncio.sleep(2**attempt)
            async with self._sem:
                try:
                    verdict, tokens = await self._call(prompt)
                except Exception as e:  # anthropic errors, after the SDK's retries
                    if _is_client_error(e):
                        raise  # a 4xx is a bug in the request: fail loudly
                    error = f"{type(e).__name__}: {e}"
                    continue
            self.usage.calls += 1
            self.usage.input_tokens += tokens[0]
            self.usage.output_tokens += tokens[1]
            if verdict.met is None:
                error = verdict.error
                continue
            verdict.input_tokens, verdict.output_tokens = tokens
            if self.cache is not None:
                self.cache.put(key, self.config.model, verdict, tokens)
            return verdict
        self.usage.failures += 1
        return Verdict(None, error=error)

    async def _call(self, prompt: str) -> tuple[Verdict, tuple[int, int]]:
        cfg = self.config
        msg = await self.client.messages.create(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            thinking=cfg.thinking_param(),
            output_config={
                "effort": cfg.effort,
                "format": {"type": "json_schema", "schema": VERDICT_SCHEMA},
            },
        )
        u = msg.usage
        tokens = (
            u.input_tokens
            + (u.cache_read_input_tokens or 0)
            + (u.cache_creation_input_tokens or 0),
            u.output_tokens,
        )
        if msg.stop_reason != "end_turn":
            return Verdict(None, error=f"stop_reason={msg.stop_reason}"), tokens
        text = "".join(b.text for b in msg.content if b.type == "text")
        try:
            d = json.loads(text)
            return Verdict(bool(d["criteria_met"]), d.get("explanation", "")), tokens
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return Verdict(None, error=f"bad output: {e}"), tokens


def _is_client_error(e: Exception) -> bool:
    status = getattr(e, "status_code", None)
    return status is not None and 400 <= status < 500 and status not in (408, 409, 429)


@dataclass
class FakeJudge(_Grader):
    """For tests: `rule(prompt) -> bool | None`; records every prompt it sees."""

    rule: Callable[[str], bool | None] = lambda prompt: True
    prompts: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    async def grade_prompt(self, prompt: str, repeat: int = 0) -> Verdict:
        self.prompts.append(prompt)
        self.usage.calls += 1
        met = self.rule(prompt)
        return Verdict(met, "fake", error=None if met is not None else "fake failure")
