# rl-finetune

**No RL results yet.** The data, the judge, its calibration, the eval pipeline
and an [OpenEnv](https://github.com/huggingface/openenv) environment (single reply,
and a simulated patient) are done and tested. The GRPO runs are next. The write-up,
step by step: [`docs/report.md`](docs/report.md).

The question: HealthBench grades a model's health conversations with an LLM judge
against physician-written rubrics. **If that rubric score is the RL reward, does
the model get better at health conversations, or better at the grader?**

> HealthBench data → rubric reward graded by a calibrated Claude judge → an RL
> environment (v1: one reply; v2: a simulated patient, rubrics mapped to whole
> consultations) → baselines → GRPO with [Miles](https://github.com/radixark/miles)
> on Modal → eval on held-out criteria, with a held-out judge.

![Post-training loop: SGLang rollouts, the rubric reward graded by Claude, GRPO on an FSDP actor, weight sync; in-loop eval on held-out criteria, offline eval under two judges](docs/diagrams/post-training-loop.svg)

## Design

| Decision | Choice | Why |
|---|---|---|
| Reward | Rubric score on the **training half** of each conversation's criteria, simple-evals' formula, unclipped | Comparable to published HealthBench scores. Unclipped, so a group whose answers are all net-harmful still has an advantage signal |
| Hacking detector 1 | **Held-out criteria.** A seeded split per conversation: positives disjoint, negatives in both halves | Gains only on the trained half mean the model is learning the grader, not the task |
| Hacking detector 2 | **Held-out judge.** Haiku 5.5 with the optimized prompt (`gepa-v1`) grades the reward; the eval grades under that judge *and* under Sonnet 5.5 with simple-evals' prompt | Gains that appear only under the training judge mean it's being exploited |
| Judge trust | Calibrated against physicians on `oss_meta_eval`, with physician-vs-physician as the ceiling and test-retest (below) | A reward is only as good as its grader. The API has no `temperature`, so judge noise is measured, not set |
| Algorithm | GRPO, group 8, TIS, ε 0.2 / 0.28, β = 0, lr 1e-6 constant: Modal's tested Miles recipe | TIS corrects the SGLang-vs-FSDP logprob gap. β = 0 keeps KL out of the reward; KL is still logged |
| Model | Qwen3.5-4B, full fine-tune, FSDP, 1 node × 4 H100; Qwen3.6-35B-A3B as a baseline | A tested Miles path that fits the budget. The 35B-A3B baseline puts any gain in context |
| Eval harness | Separate from the trainer: SGLang sampling, then this repo's scorer, for every model and checkpoint | The same code scores everything, independent of Miles internals |

## Judge calibration

On 1,000 physician judgments from `oss_meta_eval`, stratified by rubric category.
The metric is simple-evals': balanced F1 = (F1 on "met" + F1 on "not met") / 2,
over every (judge, physician) label pair. Physicians are scored against each other
the same way, on all 29.5K judgments. Run:
`python -m rubricrl.analysis.calibrate report --n 1000`. In the W&B project
`rl-finetune` (Weave), every judge call is its own `judge_verdict` trace: filter on
the run, the category, `repeat` (1 = retest) or `agrees_with_majority = false` to
read the disagreements. Each run's metrics are a summary-only evaluation.

| Rater | Balanced F1 | F1 met | F1 not met | Says "met" | Test-retest agreement (κ) |
|---|---|---|---|---|---|
| Haiku 5.5, effort low, no thinking | 0.558 | 0.652 | 0.463 | 43% | 0.900 (0.608) |
| Haiku 5.5, effort medium, thinking | 0.587 | 0.701 | 0.473 | 49% | 0.950 (0.847) |
| Sonnet 5.5, effort low, no thinking | 0.521 | 0.593 | 0.449 | 37% | 1.000 (1.000) |
| Haiku 5.5, effort low, `gepa-v1` prompt (**the reward**) | 0.655 | 0.819 | 0.491 | 70% | 0.890 (0.738) |
| Haiku 5.5, effort medium, thinking, `gepa-v1` prompt | 0.662 | 0.811 | 0.512 | 66% | 0.860 (0.693) |
| Physicians vs each other (181) | 0.672 | | | 78% | |

On 2,000 judgments, Haiku 5.5 (effort low) scores 0.570 balanced F1, with
test-retest κ 0.718.

Balanced F1 by completion length (quartiles):

| Completion (chars) | Haiku low | Haiku medium + thinking | Sonnet low |
|---|---|---|---|
| 3–935 | 0.679 | 0.702 | 0.661 |
| 938–1,840 | 0.578 | 0.618 | 0.564 |
| 1,844–3,283 | 0.525 | 0.527 | 0.442 |
| 3,288–11,265 | 0.446 | 0.521 | 0.411 |

<!-- Interpretation: yours. -->

The meta-eval grades cluster-level ("consensus") rubrics; the reward is mostly
example-level criteria. API cost of these runs: about $11.

## Baselines

The 200-conversation eval subset (`data/eval_ids.txt`). Each score is the mean
over conversations with a 95% bootstrap CI; the K samples of a conversation are
averaged first.

| Run | Judge | Full rubric | Training criteria | Held-out criteria | Chars (median) |
|---|---|---|---|---|---|
| Physician ideal completions (172 of 200 conversations) | Haiku 5.5 | 0.211 [0.167, 0.255] | 0.077 [0.012, 0.140] | 0.049 [−0.024, 0.121] | 1,406 |
| Qwen3.5-0.8B | | | | | |
| Qwen3.5-4B | | | | | |
| Qwen3.6-35B-A3B | | | | | |

The two halves carry every negative criterion against half the positive points.
So each half-score sits below the full-rubric score: compare their **gains**
between runs, not their levels (`python -m rubricrl.evals.run compare`).

## Layout

| Path | What |
|---|---|
| `src/rubricrl/data/` | `schema.py` (Example, Criterion, Judgment); `healthbench.py` (download pinned by sha256, splits, Miles export). See [`data/README.md`](data/README.md) |
| `src/rubricrl/judge/` | `prompt.py` (simple-evals' grader prompt, verbatim, and the GEPA-optimized `gepa-v1` the reward uses); `client.py` (Claude judge: structured output, retries, SQLite verdict cache, `FakeJudge`); `optimize.py` (GEPA on the grader template against physician labels, optional extra `optimize`) |
| `src/rubricrl/rewards/` | `rubric.py` (the HealthBench score); `miles_rm.py` (the Miles reward function) |
| `src/rubricrl/analysis/calibrate.py` | Judge vs physicians, physician ceiling, test-retest, agreement by length |
| `src/rubricrl/observe.py` | W&B Weave: calibration evaluations, the grader prompt and data versioned, training judge records as traces (optional extra `observe`) |
| `src/rubricrl/env/` | The OpenEnv environment: `server.py` (`ConsultEnvironment`, single / patient modes, the rubric), `models.py`, `client.py` (typed OpenEnv client), `wire.py` (a dependency-free client for Miles' Python), `miles_agent.py` (Miles' agent function and reward hook), `openenv.yaml` |
| `src/rubricrl/sim/` | Expert data → simulation: `scenarios.py` (physician criteria mapped to whole consultations, persona, hidden facts), `patient.py` (the simulated patient), `checks.py` (rewrite fidelity, careful vs minimal doctor), `scenario.py` |
| `src/rubricrl/llm.py` | Structured-output Claude calls with an SQLite cache (scenario builder, patient) |
| `src/rubricrl/evals/run.py` | Scores samples under each judge: full / training / held-out, per axis, CIs, paired comparison |
| `modal/sample.py` | SGLang sampling of the eval subset, for any base model or exported checkpoint |
| `modal/calibrate.py` | Judge calibration on a Modal CPU container, logged to Weave; the verdict cache is merged both ways with the Mac's |
| `modal/miles_rft.py` | GRPO with Miles on Modal (adapted from Modal's `miles_grpo` example), checkpoint export, and `sync_traces` (the reward log to Weave) |
| `modal/ray_cluster.py` | Ray on Modal (1 node, or 2 clustered nodes with RDMA) and `rubricrl.cluster_check`, a GPU / NCCL check |
| `scripts/` | `time_model.py`, `memory_math.py` |
| `docs/diagrams/` | `.excalidraw` sources and rendered SVGs |

## Setup

```bash
uv sync --extra train --extra modal --extra observe --extra env   # Python 3.14, uv.lock
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
```

- **API keys:** `ANTHROPIC_API_KEY` and `WANDB_API_KEY`, or on macOS the Keychain
  items `anthropic-api-key` and `wandb-api-key`.
- **Modal secrets:** `anthropic`, `wandb`, `huggingface-secret`.
- **Verdict cache:** every verdict is cached in `~/.cache/rubricrl/judge.sqlite`,
  so a rerun costs nothing.

## Running

Local, no GPU:

```bash
uv run python -m rubricrl.data.healthbench download
uv run python -m rubricrl.data.healthbench export-miles
```

Judge calibration (Modal CPU, logged to Weave; results land in
`analysis/out/calibration/`):

```bash
uv run modal run modal/calibrate.py --backfill    # the table above, all cached: $0
uv run modal run modal/calibrate.py --judge haiku --effort medium --thinking --n 1000
uv run python -m rubricrl.analysis.calibrate report --n 1000
```

Judge prompt optimization (GEPA; train/val disjoint from the calibration test
set, 6 categories held out), then the winner on the test set. On Modal, one
detached job does both and logs to W&B; `--gepa-seeds` repeats it on the same data:

```bash
uv run modal run --detach modal/optimize.py --name gepa-v1
uv run modal run --detach modal/optimize.py --name gepa-v1 --gepa-seeds 1,2
uv run modal run modal/optimize.py --pull --name gepa-v1     # results, if detached
uv run modal run --detach modal/calibrate.py --effort low --n 1000 \
    --template analysis/out/optimize/gepa-v1/gepa-v1.txt     # the prompt, another judge
```

Or locally:

```bash
uv run python -m rubricrl.judge.optimize splits
uv run python -m rubricrl.judge.optimize run --name gepa-v1 --wandb rl-finetune   # stops at $8
uv run python -m rubricrl.analysis.calibrate run --judge haiku --effort medium --thinking \
    --n 1000 --template analysis/out/optimize/gepa-v1/gepa-v1.txt --wandb rl-finetune
uv run python -m rubricrl.judge.optimize compare \
    analysis/out/calibration/claude-haiku-5-5-medium-thinking-n1000.json \
    analysis/out/calibration/claude-haiku-5-5-medium-thinking-gepa-v1-n1000.json
```

The environment (local; `--reward env` runs it inside the training container):

```bash
uv run python -m rubricrl.sim.scenarios build --split test --n 100   # v2 scenarios (~$2)
uv run python -m rubricrl.sim.checks fidelity --split test --n 100   # rewrite fidelity
uv run python -m rubricrl.sim.checks doctors --split test --n 20 --wandb rl-finetune
uv run uvicorn rubricrl.env.server:app --port 8003                   # the OpenEnv server
```

Baselines (Modal, then local scoring):

```bash
uv run modal run modal/sample.py --model qwen3.5-4b          # -> runs/samples/qwen3.5-4b.jsonl
uv run python -m rubricrl.evals.run score runs/samples/qwen3.5-4b.jsonl --estimate
uv run python -m rubricrl.evals.run score runs/samples/qwen3.5-4b.jsonl    # haiku, sonnet:1
```

GRPO (Modal):

```bash
uv run modal run modal/miles_rft.py::main --task math --smoke-test    # the pipeline, no judge
uv run modal run modal/miles_rft.py::main --smoke-test                # 2 steps with the judge
uv run modal run modal/miles_rft.py::main --smoke-test --reward env   # the same, through the environment
uv run modal run modal/miles_rft.py::main --smoke-test --reward env --env-mode patient
uv run modal run --detach modal/miles_rft.py::main --num-rollout 60
uv run modal run --detach modal/miles_rft.py::main --reward env --no-dashboard  # without the Ray dashboard and Grafana (on by default)
uv run modal run modal/miles_rft.py::ray_check --hold-s 600            # try the dashboard + Grafana login (CPU, ~$0.05)
uv run modal run modal/miles_rft.py::log_check                         # every key a step logs to W&B (CPU, ~$0.10)
uv run modal run modal/miles_rft.py::sync_traces --run-id RUN_ID    # any time; --final at the end
uv run modal run modal/miles_rft.py::export --run-id RUN_ID
uv run modal run modal/sample.py --model qwen3.5-4b --checkpoint RUN_ID/huggingface/iter_0000060 --name rft-60
uv run python -m rubricrl.evals.run score runs/samples/rft-60.jsonl
uv run python -m rubricrl.evals.run compare runs/evals/qwen3.5-4b runs/evals/rft-60
```

### Costs, measured or estimated

| Item | Cost |
|---|---|
| Haiku 5.5 verdict | ~$0.0003 (~2K input tokens, ~150 output tokens) |
| Sonnet 5.5 verdict | ~$0.005 |
| Eval of one model: Haiku on K = 2 + Sonnet on K = 1 | ~$1.3 + ~$10 |
| Training step: 128 responses × ~7.8 criteria ≈ 1,000 Haiku verdicts | ~$0.30 (effort low), ~$0.60 (medium + thinking) |
| Modal: 4 × H100 node | ~$16/h of GPU, ~$19/h with CPU and memory |

## Ray on Modal

![Ray on Modal: the Mac spawns a 2-node clustered function; rank 0 runs the Ray head and submits the experiment as a Ray Job, rank 1 joins as a worker](docs/diagrams/ray-on-modal.svg)

`modal/ray_cluster.py --entry check` brings up a Ray cluster: one node with 1–8
H100s, or two 8-GPU nodes with RDMA. It runs `rubricrl.cluster_check` on it,
which times an NCCL all-reduce across every GPU and prints versions. Training
doesn't use this app: Miles runs in its own image, with its own single-node Ray
head (`modal/miles_rft.py`).

```bash
uv run modal run modal/ray_cluster.py --entry check --gpus 8
uv run modal run --detach modal/ray_cluster.py --entry check --nodes 2
```

## Diagrams

The sources are `docs/diagrams/*.excalidraw`: open and edit them on
excalidraw.com, in the Virgil font. Re-render the SVG after an edit; the fonts are
embedded, so GitHub shows them:

```bash
npx -y mcp-excalidraw-server@2.1.2 render docs/diagrams/post-training-loop.excalidraw \
    --out docs/diagrams/post-training-loop.svg
```

## Credits

- **HealthBench** data and grader prompt: OpenAI's
  [simple-evals](https://github.com/openai/simple-evals), MIT.
- **The Miles recipe on Modal:**
  [modal-examples `14_clusters/miles_grpo.py`](https://github.com/modal-labs/modal-examples),
  MIT.
