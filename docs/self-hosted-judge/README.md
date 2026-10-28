# A self-hosted judge

The Anthropic API credit ran out on 27 Oct, and every reward was a Claude Haiku verdict.
This is the open-model replacement: a judge served by SGLang on our own H100s, calibrated
against physicians the same way Haiku was (report §4.2), and wired into the training
container next to Miles.

**Status (28 Oct, morning):** calibrated and wired. Not yet run in training; the smoke test
below is the first run.

## The decision

**Recommended: gpt-oss-120b with `gepa-v1`, on 4 H100s (2 replicas of TP2).** It agreed with
physicians better than any judge so far, on the categories GEPA never saw as well as the ones
it did, and it grades a training step in about 50 s.

| Gate (set on 27 Oct) | gpt-oss-120b, `gepa-v1` | Pass? |
|---|---|---|
| Balanced F1 vs physicians ≥ 0.625 (Haiku `gepa-v1`: 0.655; physicians: 0.672) | **0.714** | yes |
| Says "met" on the longest quartile ≥ 50% (physicians: 81%) | 88% | yes |
| One step's verdicts in ≤ 60 s | 78 s on 2 GPUs, **50 s on 4** | on 4 GPUs |

The cheaper option is gpt-oss-20b on 2 GPUs: 0.674, 43 s per step, but weaker on the held-out
categories (0.615 against 0.719) and less consistent (κ 0.72 against 0.87).

## What was measured

All on Modal, logged to W&B (`job_type=calibration`, tag `self-hosted`, grouped by prompt next
to the Haiku runs) and Weave (a trace per verdict, a `judge-calibration` evaluation per
configuration). Result JSONs in `analysis/out/calibration/`.

1. **Calibration** (`calibrate_judge`): the same 1,000 meta-eval judgments as Haiku (plus 200
   regraded), with simple-evals' prompt and with `gepa-v1`, temperature 0, verdict JSON forced
   by SGLang's constrained decoding. gpt-oss reasons first, at effort low (a probe request in
   each job's log shows it still reasons under the forced JSON); Qwen has thinking off.
2. **Replay** (`--replay RUN_ID`, `rubricrl.analysis.replay`): run D's first ~1,700 training
   verdicts (its policy's own replies, the reward's own criteria) graded again on the exact
   reward path. It gives the time for a real step (a reply's ~8 criteria share their
   conversation, so SGLang's prefix cache helps) and agreement with the Haiku verdicts D was
   trained on. In `analysis/out/calibration/replay/`.
3. **Step bench** (`--step-bench`): 1,700 meta-eval verdicts alone, no Weave in the loop. A
   pessimistic speed (no shared prefixes); in `analysis/out/calibration/bench/`.

| Judge | GPUs | Balanced F1 (`gepa-v1`) | Replay: one step | Replay: agrees with Haiku (κ) | GPU $ per calibration |
|---|---|---|---|---|---|
| gpt-oss-120b | 2 (TP2) / 4 (TP2 × 2) | **0.714** | 78 s / **50 s** | 67% (0.31) | $0.14 |
| gpt-oss-20b | 1 / 2 | 0.674 | 63 s / 43 s | 68% (0.29) | $0.07 |
| Qwen3.6-35B-A3B | 2 (TP2) | 0.671 | 122 s | 61% (0.24) | $0.17 |
| Qwen3.5-4B | 1 | 0.643 | (bench: 76 s) | not run | $0.06 |

Throughput notes:
- One server grades 15–25 verdicts/s on this workload, about as fast for the 4B as for the
  20B: the ~2,800-token prompts dominate. Forcing JSON costs nothing (74.9 s free
  vs 76.6 s forced), raising client concurrency from 256 to 1,024 made it slower (110 s), and a
  second replica helps a lot on real steps (63 s → 43 s), less on the bench.
- gpt-oss-120b does not load on one H100 in this image's SGLang (out of memory at 70 GB), so it
  needs 2.

## The caveat that matters: the reward changes

On D's own replies, every open judge disagrees with Haiku on about a third of verdicts (κ
0.24–0.31), and says "met" more often (37–53% against Haiku's 32%). The open judges agree with
each other more (gpt-oss-20b vs 120b: 81%, κ 0.60). The meta-eval can't say who is right on
these criteria: it has no labels for the reward's conversation-specific criteria or for this
policy's replies. So:

- **A run with this judge optimizes a different reward from A–D.** Compare it with D on the
  offline test set graded by one fixed judge, not on in-run reward curves.
- **Grade held-out evals with Haiku or Sonnet once credit is back**, so the judge the policy
  trained against isn't also the one scoring it.
- **Possible contamination.** gpt-oss-120b beats the physicians' agreement with each other
  (0.714 vs 0.672). A judge that matches the physician majority can do that, but OpenAI built
  HealthBench, and whether the meta-eval was in gpt-oss's training data isn't public.
- **Length.** gpt-oss-120b grows more lenient on longer answers than physicians do ("met" 74%
  → 88% from the shortest to the longest quartile; physicians 74% → 81%). Watch reply length
  against held-out score in the run.
- At temperature 0 the verdicts are nearly deterministic, so the judge adds bias to the reward
  and little noise; gpt-oss is still not bit-exact (retest κ 0.87).

## In the morning (you launch; nothing has trained with it yet)

1. **Smoke test, 2 steps** (8 GPUs × ~15 min ≈ $8; the judge takes ~5 min to load). Check:
   `env_server.log` and `judge_server.log` in the run's Volume directory, 0 failed verdicts in
   `rollout/rubric/*`, scores that vary within groups, the step time.
   ```
   uv run modal run modal/miles_rft.py::main --smoke-test --reward env \
     --judge-model gpt-oss-120b --judge-gpus 4
   ```
2. **The 60-step run** (D's recipe; 8 GPUs at $31.60/h: about $30–40 if a step takes 50–70 s,
   $0 API; D's recipe with Haiku would have cost ~$65 for 60 steps):
   ```
   uv run modal run --detach modal/miles_rft.py::main --num-rollout 60 --reward env --lr 3e-6 \
     --dynamic-sampling --no-std-normalization --save-interval 10 \
     --judge-model gpt-oss-120b --judge-gpus 4
   ```
   The cheaper variant: `--judge-model gpt-oss-20b --judge-gpus 2` (6 GPUs).

## How it's wired

- `src/rubricrl/judge/client.py`: `OpenAICompatJudge` (SGLang's `/v1/chat/completions`, JSON
  schema, gpt-oss `reasoning_effort`, Qwen `enable_thinking`), chosen by `make_judge` when the
  config has a `base_url`. The reward and the environment get it through `RUBRICRL_JUDGE_URL`.
- `modal/miles_rft.py`:
  - `JUDGE_MODELS`: pinned revisions and the GPUs each model needs (its tensor parallelism);
  - `start_judge_server`: SGLang on given GPUs, TP from `JUDGE_MODELS`, data-parallel over the
    rest;
  - `train --judge-model M --judge-gpus N`: the container gets 4 + N GPUs, Miles and Ray see
    GPUs 0–3, the judge the last N; W&B tags `judge:M-low`, `self-hosted-judge:Ngpu`; the GPU
    cost metric counts all 4 + N;
  - `calibrate_judge`: calibration, `--step-bench`, `--replay RUN_ID`, `--gpus N`.
- `src/rubricrl/analysis/calibrate.py`: `--serving` (GPU cost) and a `throughput` block in every
  result; `--unconstrained` parses JSON out of a free reply instead.
- `src/rubricrl/analysis/replay.py`: the replay.
- Figures: `python -m rubricrl.analysis.figures self-hosted`.
