"""Time one full fine-tuning step of a model on this machine (M0's model decision).

1 warm-up step, then the mean of 3 steps (forward + backward + AdamW) at a batch of
4 x 256 random tokens: the local GRPO micro-batch shape (PLAN.md, "Mac memory").
bf16 weights, gradient checkpointing on. Prints s/step and memory next to the
`memory_math.py` estimate for the same shape.

    uv run python scripts/time_model.py --model Qwen/Qwen3.5-0.8B
    uv run python scripts/time_model.py --model Qwen/Qwen2.5-0.5B-Instruct

Rule (PLAN.md, "Models"): over ~5 s/step on the Mac, use the smaller model.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from memory_math import GB, activations_gb, logits_gb  # noqa: E402


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def allocated_gb(dev: torch.device) -> float:
    """Memory the backend holds now (MPS has no peak counter: sampled per phase)."""
    if dev.type == "cuda":
        return torch.cuda.max_memory_allocated() / GB
    if dev.type == "mps":
        return torch.mps.driver_allocated_memory() / GB
    return float("nan")


def text_config(name: str) -> dict:
    """The memory_math fields of a (possibly multimodal) HF config."""
    cfg = AutoConfig.from_pretrained(name)
    cfg = getattr(cfg, "text_config", cfg)
    return {"L": cfg.num_hidden_layers, "h": cfg.hidden_size, "V": cfg.vocab_size}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--steps", type=int, default=3)
    args = p.parse_args()

    dev = device()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-6)
    n_params = sum(p.numel() for p in model.parameters())

    m = text_config(args.model)
    ids = torch.randint(0, m["V"], (args.batch, args.seq), device=dev)

    peak, times = 0.0, []
    for i in range(1 + args.steps):  # step 0 is the warm-up
        sync(dev)
        t0 = time.perf_counter()
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        peak = max(peak, allocated_gb(dev))
        opt.step()
        opt.zero_grad(set_to_none=True)
        sync(dev)
        peak = max(peak, allocated_gb(dev))
        if i:
            times.append(time.perf_counter() - t0)

    tokens = args.batch * args.seq
    # Pure bf16 here: weights, grads and both Adam moments at 2 B each = 8 B/param.
    # memory_math.finetune_gb's 16 B/param is mixed precision (fp32 master + Adam).
    states = n_params * 8 / GB
    predicted = states + activations_gb(m, tokens) + logits_gb(m, tokens)
    print(f"model      {args.model} ({n_params / 1e9:.2f}B params) on {dev.type}")
    print(f"shape      {args.batch} x {args.seq} tokens, bf16, grad checkpointing")
    print(f"s/step     {sum(times) / len(times):.2f} (mean of {len(times)})")
    print(f"memory     {peak:.1f} GB measured, {predicted:.1f} GB predicted")
    print(f"           ({states:.1f} states + activations + fp32 logits)")


if __name__ == "__main__":
    main()
