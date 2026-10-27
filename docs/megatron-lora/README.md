# Plan: Qwen3.6-35B-A3B with LoRA on Miles' Megatron backend

The stretch row of [`PLAN.md`](../../PLAN.md), made concrete on 27 Oct so it can be picked up
on the morning of Wed 28 Oct. The 4B full-fine-tuning path (`--train-backend fsdp`) stays as
it is; this adds a second path next to it.

**Status:** plan only. Nothing below is implemented yet.

## Why a second path, and why Megatron

- **LoRA in Miles v0.1.1 is Megatron-only.** `docs/user-guide/training-backend.md` in Miles
  lists LoRA as "Supported" for Megatron-LM and "Not supported" for FSDP; every example in
  `examples/lora/` is Megatron.
- **Full fine-tuning a 35B model doesn't fit one 4 × H100 node:** about 16 bytes per
  parameter with Adam is about 560 GB, against 320 GB of GPU memory. Only 3B parameters are
  active per token, but all 35B are held in memory.
- **Miles ships this exact setup:** `scripts/run_qwen3_5_35b_a3b_lora.py` (Qwen3.5 and
  Qwen3.6 35B-A3B, GRPO, LoRA, Megatron-Bridge), with the architecture in
  `scripts/models/qwen3.6-35B-A3B_lora.py`. Our path is that script's configuration, plus our
  environment, judge and logging.

## Checked on 27 Oct (in the Miles v0.1.1 source and on Hugging Face)

| Question | Answer | Where |
|---|---|---|
| Megatron-Bridge in Miles' image? | Pinned in the image's Dockerfile; importing it in our image is step 3 | Miles `docker/Dockerfile` |
| Does the session server use the LoRA adapter? | Yes: it sets the served adapter on every request, so rollouts come from the policy | `miles/rollout/session/request_args.py` |
| Chat-template setting (`--tito-model`) | `qwen35` exists; Qwen3.6 has the same architecture, to verify in step 3 | `miles/utils/chat_template_utils/tito_tokenizer.py` |
| Routing replay (R3) with the session server | Allowed, with a warning: large payloads every turn, known SGLang issues | `miles/utils/arguments.py` |
| LoRA checkpoints | `adapter_model.safetensors`; `--save-hf` exports a Hugging Face model | `miles/backends/megatron_utils/` |
| `Qwen/Qwen3.6-35B-A3B` | Public, not gated; image-text-to-text (we use text only) | Hugging Face API |

## Configuration

| | 4B path (today) | 35B path (this plan) |
|---|---|---|
| Model | Qwen3.5-4B | Qwen3.6-35B-A3B (~70 GB) |
| Backend | `--train-backend fsdp`, gradient checkpointing | Megatron, `--megatron-to-hf-mode bridge` (no offline conversion) |
| Fine-tuning | full | LoRA rank 16, alpha 32, dropout 0, `--target-modules all-linear` (vision tower and multi-token-prediction block excluded by the reference script), `--experts-shared-outer-loras`, `--lora-base-cpu-backup`, `--no-gradient-accumulation-fusion` |
| Model split | 4-way sharding | tensor parallel 2 (the model caps it at 2), sequence parallel, expert parallel = GPUs, full recompute, `--qkv-format bshd`, micro-batch 1 |
| Learning rate | 1e-6 | 1e-5 |
| SGLang | 1 GPU per engine | one engine across the GPUs, `--sglang-lora-backend triton`, `--sglang-max-lora-rank 16`, `--sglang-mem-fraction-static 0.4` |
| GPUs | 4 × H100 | **8 × H100** (pending decision 1) |
| Reply cap | 8,192 tokens | **4,096** (pending decision 2) |
| Routing replay | n/a | **off** at first; watch `train/train_rollout_logprob_abs_diff` |
| Unchanged | | environment, judge (`gepa-v1`), rewards, W&B hooks, dashboard and Grafana |

## Decisions pending

1. **8 × H100 (recommended) or 4.** The reference script assumes 8. Four halves the cost
   (about $16 an hour instead of about $32) but makes out-of-memory on the first smoke test likely.
2. **Reply cap 4,096 (recommended for the first run) or 8,192.** The reference caps tokens
   per GPU at 4,096, and this model's sequences can't be packed.
3. **Before or after the interview.** The morning of Wed 28 is reserved for debugging rep 2
   and the Baseten drill, with a hard stop on new material at 11:00. Steps 6–8 below take about
   10 minutes of attention if nothing breaks.

## Steps

### Claude, CPU only (about 2 hours, about $2)

1. **Code, in `modal/miles_rft.py`:**
   - `--backend fsdp|megatron` (default `fsdp`).
   - `qwen3.6-35b-a3b` in `MODELS` and `TITO` (`qwen35`).
   - The Megatron and LoRA arguments from the table above, built from Miles' model-args helper
     for `qwen3.6-35B-A3B_lora`.
   - GPU count and container RAM per backend.
   - `--save-hf` on the last step.
   - **Done when:** the 4B argument list is byte-identical to today's.
2. **Download the model:** `modal run modal/miles_rft.py::download` for the new model, about
   70 GB onto the `rubricrl-data` volume (about 15 minutes, CPU).
3. **CPU checks in Miles' image:**
   - `import megatron.bridge` works.
   - Miles' parser accepts the full argument list (as `log_check` does for the 4B path).
   - `session_check --model qwen3.6-35b-a3b` starts the session server with `qwen35`.
   - **Done when:** all three pass. If one needs more than a config change, stop here: no
     morning run.
4. **Tests and lint:** the new argument builder (no FSDP flag left, LoRA flags present, no
   flag set twice); `pytest` and `ruff` green.

### You, GPU

5. **Smoke test** (about 30 minutes, about $15–20; loading 70 GB of weights is a large part):

   ```bash
   uv run modal run --detach modal/miles_rft.py --backend megatron \
     --model qwen3.6-35b-a3b --reward env --smoke-test
   ```

6. **Go / no-go** (2 minutes in W&B and the Modal logs). **Go** if all of these hold:
   - 2 steps done and a checkpoint saved;
   - no out-of-memory error;
   - `rollout/truncated_ratio` under 10%;
   - `train/train_rollout_logprob_abs_diff` close to the 4B run's;
   - `rollout/rubric/*` present.

   **No-go:** retry once only if the error names the fix (out of memory → lower the reply cap
   or switch to 8 GPUs). Otherwise stop. No debugging after 8:30 on interview day.
7. **The run** (rough estimate: 2–3 hours, $70–100 of GPU plus about $10 of judge calls on
   8 GPUs):

   ```bash
   uv run modal run --detach modal/miles_rft.py --backend megatron \
     --model qwen3.6-35b-a3b --reward env --num-rollout 20
   ```

### After the interview

8. **Eval path:**
   - `modal/sample.py` serves the `--save-hf` export.
   - `evals.run score` and `compare` against the 4B run and the base model, on both rubric
     halves.
9. **Routing replay:** if step 6 showed a wide trainer–SGLang gap, add
   `--use-rollout-routing-replay` and compare.

## Morning timeline, if steps 1–4 are not done the night before

| Time | Claude | You |
|---|---|---|
| 6:00 | Steps 1–4 (approve tool calls as they come) | Fundamentals card, then debugging rep 2 |
| ~8:00 | Report: checks green, or stop | Launch step 5 |
| ~8:30 | | Step 6: go → launch step 7; no-go → stop |
| 11:00 | | Hard stop. The run finishes on its own |
