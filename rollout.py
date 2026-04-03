from __future__ import annotations

import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_URL = "http://localhost:8080"


def _build_rollout_func(
    server_url: str = _DEFAULT_URL,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 1024,
    num_generations: int = 4,
    timeout: float = 120.0,
):
    """Build a rollout_func closure bound to the given llama-server config.

    Args:
        server_url: llama-server base URL.
        temperature: Sampling temperature.
        top_p: Nucleus sampling probability.
        max_tokens: Maximum tokens per completion.
        num_generations: How many completions to generate per prompt (GRPO group size).
        timeout: HTTP request timeout in seconds.

    Returns:
        A rollout_func compatible with GRPOTrainer.
    """

    async def _generate_one(client: httpx.AsyncClient, prompt: str) -> list[dict]:
        """Generate num_generations completions for a single prompt."""
        payload = {
            "prompt": prompt,
            "n": num_generations,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "logprobs": True,
            "echo": False,
        }
        resp = await client.post(
            f"{server_url}/v1/completions",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"]

    async def _generate_batch(prompts: list[str]) -> list[list[dict]]:
        """Generate completions for a batch of prompts concurrently."""
        async with httpx.AsyncClient() as client:
            tasks = [_generate_one(client, p) for p in prompts]
            return await asyncio.gather(*tasks)

    def llamacpp_rollout(prompts: list[str], trainer) -> dict:
        """rollout_func passed to GRPOTrainer.

        Args:
            prompts: List of prompt strings (one per batch item).
            trainer: GRPOTrainer instance (used for tokenizer access).

        Returns:
            Dict with prompt_ids, completion_ids, logprobs for TRL.
        """
        tokenizer = trainer.processing_class
        all_choices = asyncio.run(_generate_batch(prompts))

        prompt_ids_batch = []
        completion_ids_batch = []
        logprobs_batch = []

        for prompt, choices in zip(prompts, all_choices, strict=True):
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)

            for choice in choices:
                text = choice["text"]
                completion_ids = tokenizer.encode(text, add_special_tokens=False)

                # llama-server returns logprobs in choice["logprobs"]["token_logprobs"].
                # Fall back to zeros if not available (importance ratio = 1).
                raw_logprobs = choice.get("logprobs") or {}
                token_logprobs = raw_logprobs.get("token_logprobs") or []

                if len(token_logprobs) >= len(completion_ids):
                    token_logprobs = token_logprobs[: len(completion_ids)]
                else:
                    token_logprobs = token_logprobs + [0.0] * (
                        len(completion_ids) - len(token_logprobs)
                    )

                prompt_ids_batch.append(prompt_ids)
                completion_ids_batch.append(completion_ids)
                logprobs_batch.append(token_logprobs)

        return {
            "prompt_ids": prompt_ids_batch,
            "completion_ids": completion_ids_batch,
            "logprobs": logprobs_batch,
        }

    return llamacpp_rollout


def make_rollout_func(config: dict):
    """Create a rollout_func from a config dict (loaded from config/default.yaml).

    Args:
        config: Training config dict with llama_server_url, temperature, etc.

    Returns:
        A rollout_func ready to pass to GRPOTrainer.
    """
    return _build_rollout_func(
        server_url=config.get("llama_server_url", _DEFAULT_URL),
        temperature=config.get("temperature", 0.8),
        top_p=config.get("top_p", 0.95),
        max_tokens=config.get("max_completion_length", 1024),
        num_generations=config.get("num_generations", 4),
    )
