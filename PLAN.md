# Plan: HealthBench rubric-reward RL with Miles on Modal

**Status (26 Oct 2026):**
- **Done and tested:** the data, the judge, its calibration and the eval pipeline.
- **Next:** baselines and the GRPO run, on Modal (you launch those).

The nanorft trainer is scrapped: in practice nobody writes their own trainer. The
theory comes from the RLHF book and from reading Miles' code (table below). Earlier
plans are in git history (v1: tag `v0-trl-llamacpp`; v2: the nanorft plan, before
this commit). Day-by-day slots live in the resume repo's interview plan.

## The question

> If HealthBench's rubric score is the RL reward, does the model get better at
> health conversations, or better at the grader?

The **Handshake technicals (Wed 28 Oct)** are 2× post-training design and 1×
eval-pipeline debugging. This repo is the worked example for both: a reward you
can defend, two detectors for reward hacking, a calibrated judge, and an eval
harness kept apart from the trainer.

## Milestones

| # | Milestone | Who | Status |
|---|---|---|---|
| M0 | Refocus: nanorft deleted, package `rubricrl`, deps without torch in the base | Claude | ✅ |
| M1 | **Data:** HealthBench pinned by sha256; conversation split and criteria split; Miles export; the eval subset frozen in `data/eval_ids.txt` | Claude | ✅ |
| M2 | **Judge:** simple-evals' grader prompt; Claude via the API with structured output; SQLite cache; the rubric score | Claude | ✅ |
| M3 | **Calibration** on `oss_meta_eval`: Haiku low, Haiku medium + thinking, Sonnet low, against the physician ceiling (README) | Claude | ✅ ~$11 of API |
| M3b | **Calibration on Modal, in Weave:** `modal/calibrate.py --backfill` puts the 4 runs in Weave from the cache ($0); new judge configs run there too. Read the disagreements before M6 | you | ⬜ cents of Modal |
| M4 | **Baselines:** `modal/sample.py`, then `evals/run.py`, for Qwen3.5-0.8B, Qwen3.5-4B and Qwen3.6-35B-A3B. The physician reference is done | you run Modal; scoring is local | ⬜ ~$8 Modal + ~$12 API per model |
| M5 | **Smoke runs:** `miles_rft.py --task math --smoke-test`, then `--smoke-test` (HealthBench + judge). Measure step time here and re-size M6. Then `miles_rft.py::sync_traces --run-id RUN_ID --final`: check the step numbers against Miles' | you | ⬜ ~$10 |
| M6 | **The run:** Qwen3.5-4B, 16 prompts × 8 samples, ≤1,024 tokens, ~60 steps, in-loop eval every 10 | you | ⬜ ~$30–40 Modal + ~$20 API |
| M7 | **Eval and write-up:** export, sample, score, `compare` against the base. The README's numbers are Claude's; the interpretation is yours. Prep notes go in `resume/applications/handshake/` | both | ⬜ |
| Stretch | 35B-A3B LoRA (Megatron backend, TP2 × EP4, `--lora-base-cpu-backup`), only if M6 is done by Tue afternoon and ~$20 is left | | ⬜ |

**Cut order if time runs short:** the stretch, then the 0.8B baseline, then the
per-axis plots, then the run's length. **Never cut calibration + baselines:**
together they already make a complete eval-pipeline story.

## Decisions to make before M6

1. **Training judge: Haiku low or Haiku medium + thinking?**
   - Medium + thinking: balanced F1 0.587 vs 0.558, test-retest κ 0.85 vs 0.61
     (a much less noisy reward).
   - The cost: about 2× the API spend and judge latency.
   - Flags: `--judge-effort medium --judge-thinking`.
   - Sonnet low calibrated *worse* (0.521) and is ~15× the price, so the plan's
     gate ("switch to Sonnet if Haiku is far worse") doesn't fire.
2. **Negatives in both halves weigh harm about 2× in the reward.** Each half
   divides by its own positive points but carries every negative. That's
   conservative and defensible. The alternative is to halve the negative points
   per half.
3. **Eval sampling:** temperature 0.7 / top-p 0.8 / top-k 20 (Qwen's non-thinking
   defaults) and ≤2,048 new tokens, the same for every model. Training samples at
   temperature 1 with ≤1,024 tokens.

## Risks, and what to do

| Risk | Sign | Fallback |
|---|---|---|
| Qwen3.5 breaks under Miles FSDP v0.1.1 (Gated DeltaNet kernels, HF class `Qwen3_5ForConditionalGeneration`) | the math smoke fails in model init or the weight sync | `--model qwen3-4b`: the example's exact model. Nothing else changes |
| Qwen3.6-35B-A3B not in the image's SGLang | `sample.py` fails to load it | Baseline with Qwen3.5-35B-A3B, or the `v0.1.3` image for sampling only |
| Logit memory at vocab 248K | OOM in the train step | lower `--max-tokens-per-gpu` (16384 → 8192) |
| Judge rate limits (~1,000 calls/step) | 429s and slow rollouts | `RUBRICRL_CONCURRENCY` (default 64); check the API tier first |
| Length gaming | median chars rising, held-out flat | it's the expected result to report. A length penalty is a v2 knob (simple-evals has one) |

## Read alongside (Miles v0.1.1)

| Topic | Read |
|---|---|
| The loop, colocated train / rollout | `train.py`, `miles/ray/` |
| Rollouts, reward calls, eval rollouts | `miles/rollout/sglang_rollout.py` (`generate_and_rm`, `eval_rollout_single_dataset`) |
| Custom rewards | `miles/rollout/rm_hub/__init__.py` (`async_rm`, `batched_async_rm`, `--group-rm`) |
| Data | `miles/utils/data.py` (`Dataset`: chat template, metadata, prompt-length filter) |
| GRPO, TIS, the clip | `miles/backends/fsdp_utils/` (actor, loss), and the args under `--advantage-estimator` |
| Theory | the RLHF book: policy gradients, PPO / GRPO, reward models, over-optimisation |

## Budget

| Item | Modal | API |
|---|---|---|
| Calibration (done) | – | ~$11 |
| Baselines × 3 | ~$8 | ~$35 (Sonnet on K = 1) |
| Smoke runs | ~$10 | ~$1 |
| Main run | ~$30–40 | ~$20–40 |
| Final eval | ~$2 | ~$12 |
| Slack / stretch | ~$15 | |

Set a Modal spend limit and an Anthropic API cap first.

## After the interviews

- Arms A / B / C: rubric GRPO vs on-policy distillation vs both.
- MentalHealthBench transfer.
- The 35B-A3B LoRA run.
- A length-penalised reward.
- A frameworks comparison (verl vs slime / Miles vs SkyRL), written up in the
  resume repo's `study/`.
