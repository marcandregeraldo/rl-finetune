# rl-finetune

Lightweight local RL post-training for LLMs on Apple Silicon.

Finetunes **Qwen3-1.7B** with GRPO (Group Relative Policy Optimization) using:
- **TRL v1.0** `GRPOTrainer` for the training loop
- **llama-server** (llama.cpp, Metal-accelerated) for fast rollout generation
- **OpenEnv** Wordle as the initial multi-turn environment
- **Rich Live** dashboard for real-time training monitoring
- **promptfoo** for structured evaluation

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                     rl train                         │
│                                                      │
│  GRPOTrainer (PyTorch MPS)                          │
│    ├── rollout_func ──► llama-server :8080 (Metal)  │
│    │                    (GGUF, fast generation)      │
│    ├── environment ──► OpenEnv Wordle :8001 (Docker)│
│    └── callbacks:                                    │
│         ├── DashboardCallback (Rich Live TUI)        │
│         └── WeightSyncCallback (LoRA → GGUF sync)   │
│                     │                                │
│              every N steps:                          │
│         merge LoRA → GGUF → reload server            │
└──────────────────────────────────────────────────────┘
                     │
              after training:
                     │
┌────────────────────▼─────────────────────────────────┐
│  rl export → models/finetuned.gguf                  │
│  rl serve  → llama-server serves finetuned model    │
│  rl eval   → promptfoo eval suite                   │
└──────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Tool | Install |
|------|---------|
| Python 3.12 | `pyenv install 3.12` or `brew install python@3.12` |
| uv | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| llama.cpp | `brew install llama.cpp` |
| Docker | [docker.com/get-started](https://www.docker.com/get-started/) |
| Node.js + promptfoo | `npm install -g promptfoo` |

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo> rl-finetune && cd rl-finetune
uv sync

# 2. Download base model (Qwen3-1.7B GGUF)
mkdir -p models
huggingface-cli download Qwen/Qwen3-1.7B-GGUF \
  Qwen3-1.7B-Q4_K_M.gguf --local-dir models

# 3. Start services
rl env start wordle          # OpenEnv Wordle in Docker (:8001)
rl serve                     # llama-server with the GGUF (:8080)

# 4. Train (in a new terminal)
rl train --config config/default.yaml

# 5. Evaluate after training
rl export --checkpoint checkpoints/grpo/final
rl serve --model models/finetuned.gguf
rl eval --view
```

---

## CLI Reference

### `rl train`
```
rl train [OPTIONS]

  Launch GRPO training with a config file.

Options:
  -c, --config TEXT      YAML config file  [default: config/default.yaml]
      --max-steps INT    Override max_steps in config
      --env TEXT         Override env in config (e.g. wordle)
      --dry-run          Print resolved config and exit
```

### `rl serve`
```
rl serve [OPTIONS]

  Start llama-server for rollout generation and eval inference.

Options:
  -m, --model TEXT       Path to GGUF model file
  -p, --port INT         Server port  [default: 8080]
      --ctx-size INT     Context window size  [default: 4096]
```

### `rl env`
```
rl env start [wordle|coding]    Start OpenEnv container
rl env stop  [wordle|coding]    Stop OpenEnv container
rl env ls                       List environment status
```

### `rl export`
```
rl export [OPTIONS]

  Convert a LoRA checkpoint to GGUF for llama-server.

Options:
  -k, --checkpoint TEXT  TRL checkpoint directory  [required]
  -o, --output TEXT      Output GGUF path  [default: models/finetuned.gguf]
  -q, --quantize TEXT    Quantization type  [default: q4_k_m]
```

### `rl eval`
```
rl eval [OPTIONS]

  Run promptfoo evaluation suite against llama-server.

Options:
  -c, --config TEXT      Eval config  [default: evals/promptfooconfig.yaml]
      --view             Open promptfoo web UI after eval
```

### `rl status`
```
rl status   Show llama-server health, OpenEnv status, latest checkpoint
```

---

## Configuration

Edit `config/default.yaml` to tune training:

```yaml
model: Qwen/Qwen3-1.7B          # HuggingFace model ID
env: wordle                      # OpenEnv environment
learning_rate: 5.0e-6
num_generations: 4               # GRPO group size
max_completion_length: 1024
sync_every: 50                   # LoRA→GGUF sync frequency (steps)
```

Pass a custom config: `rl train --config my_config.yaml`

---

## Adding a Custom Environment

1. Create `envs/my_env.py` implementing TRL's environment interface:

```python
class MyEnv:
    def __init__(self):
        self.reward = 0.0

    def reset(self, **kwargs) -> str:
        """Return the initial observation."""
        ...

    def my_tool(self, arg: str) -> str:
        """A tool the model can call.
        Args:
            arg: Description of the argument
        Returns:
            The result
        """
        ...
```

2. Register it in `cli.py` under `_ENV_IMAGES` if Docker-backed.

3. Add reward functions in `rewards.py` with signature:
```python
def my_reward(environments, **kwargs) -> list[float]:
    return [env.reward for env in environments]
```

4. Pass to GRPOTrainer in `train.py`:
```python
environment_factory=MyEnv,
reward_funcs=[my_reward],
```

---

## Project Structure

```
rl-finetune/
├── config/default.yaml      Training config
├── envs/wordle.py           OpenEnv Wordle adapter
├── rewards.py               TRL reward functions
├── rollout.py               Custom llama-server rollout_func
├── dashboard.py             Rich Live training dashboard
├── train.py                 Core training logic (GRPOTrainer)
├── export.py                LoRA → GGUF conversion + reload
├── cli.py                   rl CLI entry point
├── scripts/serve.sh         llama-server launcher script
└── evals/promptfooconfig.yaml  Promptfoo eval config
```

---

## Memory Budget (M4 Pro 24GB)

| Component | VRAM |
|-----------|------|
| Qwen3-1.7B bf16 (policy) | ~4 GB |
| Qwen3-1.7B bf16 (reference) | ~4 GB |
| LoRA adapters + optimizer | ~2 GB |
| Generation cache | ~2 GB |
| **Total** | **~12 GB** |

Comfortable on 24GB. If you hit OOM, reduce `num_generations` from 4 → 2 or `max_completion_length` from 1024 → 512.

---

## Notes

- **No AsyncGRPOTrainer**: Requires vLLM + CUDA. The `rollout_func` approach achieves the same decoupled generation/training on Mac.
- **No bitsandbytes**: MPS not supported. Use bf16 + LoRA instead of 4-bit quantization.
- **llama-server logprobs**: Per-token log probs are requested via `logprobs: true`. If unavailable, `rollout.py` falls back to zeros (importance sampling ratio = 1, still valid for GRPO).
- **Weight sync**: Every `sync_every` steps, LoRA is merged → converted to GGUF → llama-server is notified. The server restart takes ~30–60s; `sync_every: 50` is a good balance.
