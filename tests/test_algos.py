"""Spec for nanorft/algos.py. Grows one function at a time; run with
`uv run pytest tests/test_algos.py -q`."""

import math

import pytest

torch = pytest.importorskip("torch")
from nanorft import algos  # noqa: E402


# ---------------------------------------------------------------- token_logprobs


def test_token_logprobs_matches_log_softmax_gather():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 11)
    tokens = torch.randint(0, 11, (2, 5))
    expected = torch.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    out = algos.token_logprobs(logits, tokens)
    assert out.shape == (2, 5)
    torch.testing.assert_close(out, expected)


def test_token_logprobs_uniform_logits_is_minus_log_vocab():
    logits = torch.zeros(1, 3, 8)
    tokens = torch.tensor([[0, 3, 7]])
    out = algos.token_logprobs(logits, tokens)
    torch.testing.assert_close(out, torch.full((1, 3), -math.log(8)))


def test_token_logprobs_is_stable_for_huge_logits():
    logits = torch.tensor([[[1e4, 0.0, -1e4]]])
    out = algos.token_logprobs(logits, torch.tensor([[0]]))
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.zeros(1, 1), atol=1e-4, rtol=0)


def test_token_logprobs_returns_fp32_from_bf16():
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 16).to(torch.bfloat16)
    out = algos.token_logprobs(logits, torch.randint(0, 16, (1, 4)))
    assert out.dtype == torch.float32
