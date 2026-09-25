# Plan v2: HealthBench RFT: nano trainer locally, verl on Modal

Supersedes v1 (kept as `PLAN.v1.md`). What changed: the compute moves from the Mac to Modal GPUs; the
stack is the one you'll talk about in interviews (**Ray + vLLM + PyTorch**); MentalHealthBench joins
as a transfer test; and the experiment gains a second training arm: **on-policy distillation vs
rubric-reward GRPO**. The core question is unchanged.

## The question

> HealthBench grades responses with an LLM judging against physician-written rubrics. If you use that
> rubric score as an RL reward, does the model get better at health conversations, or better at the
> grader? And does on-policy distillation from a stronger open model avoid the gaming that
> rubric-reward RL falls into?

The second half makes this repo a direct talking point for the 2026 post-training recipe
(`resume/applications/study/post-training-2026.md`). It's cheap to add because verl ships OPD.

## Stack decision: verl on a single Modal node

**Use Modal's verl GRPO example as the starting point** (https://modal.com/docs/examples/grpo_verl).
- verl *is* Ray + vLLM + PyTorch/FSDP. Ray orchestrates actor, rollout and reference workers inside
  the container.
- The example runs **single-container Ray** (`gpu="H100:2"`, no multi-node setup), with Modal Volumes
  for data and checkpoints, a custom reward function wired in through `custom_reward_function.path`,
  and a vLLM serving endpoint for the trained model.
- **Don't do multi-node.** Modal's multi-node clusters are beta, need access requested, must use full
  8-GPU nodes, and you'd bootstrap `ray start --head` / workers by hand from
  `modal.experimental.get_cluster_info()`. A 2–4B policy fits on one node. Multi-node is a sentence in
  the README's "what I'd do with more compute", not a setup week.
- **Update the image.** The example pins `verl 0.4.1` / `vLLM 0.8.5` (2025). Qwen3.5 needs a current
  verl image; pin versions in the Modal image and write them in the README.

What stays low-level and yours:
- **`nano/`**, your own verl-shaped trainer in pure PyTorch + Ray. It's the learning artifact and
  the local training path (workstream 2).
- The **rubric reward**: grading prompts, per-criterion scoring, caching, points arithmetic, the
  train/held-out criteria split. Shared by `nano/` and verl.
- The **judge service**: vLLM on its own Modal GPU function, called in batches.

## Data

- **HealthBench** (`openai/healthbench`, MIT). The files and what each is for:
  - `oss_eval.jsonl` (5K conversations): training and evaluation pool.
  - `hard_*.jsonl`: harder eval slice.
  - `consensus_*.jsonl`: criteria physicians agreed on.
  - **`oss_meta_eval.jsonl`: physician judgments of whether real model responses meet each rubric
    criterion.** This is the big find. It calibrates your judge **against physicians, for free**,
    instead of against the `gpt-4.1` grader. Much stronger result, no API spend.
- **MentalHealthBench** (OpenAI, released 23 Sep 2026): 1,215 synthetic conversations reviewed by
  80+ licensed experts across 22 countries. Criteria weighted −10…+10, kept only when ≥2 of 3
  experts agreed. Acuity levels from non-acute to emergency; personas include teens and clinicians.
  Official grader is GPT-5.6 Sol.
  - **Use as a held-out *domain*:** train on HealthBench, evaluate transfer, no training on it.
  - **Before using it, find the official release (repo or dataset) and its licence.** The only Hub
    copy found so far is an unofficial mirror (`MercuriusDream/MentalHealthBench`). Don't build on
    the mirror.
  - Report emergency and high-acuity behaviour **separately**. A model that got "more complete" but
    worse at escalating a crisis is the finding that matters, and it's the one to be careful
    publishing examples of.

## Models

Checked on the Hub, 24 Sep 2026. **Qwen3.6 only comes in 27B (dense) and 35B-A3B (MoE).** There is
no small Qwen3.6. The Qwen3.5 family has the small sizes, with Base checkpoints for each, and shares
the Qwen3.6 architecture style: hybrid **Gated DeltaNet** (linear attention) + gated attention, and
MoE at 35B-A3B.

| Stage | Model | Why |
|---|---|---|
| Local `nano/` (Mac) | **Qwen3.5-0.8B** (instruct; `-Base` exists) | Same family and tokenizer as everything later, so it can also be the OPD student of a bigger Qwen. **Risk:** DeltaNet layers rely on CUDA/Triton kernels; on MPS, transformers may fall back to a slow torch path. If a step takes minutes, switch the *exercise* to a plain-attention model (Qwen2.5-0.5B-Instruct or SmolLM2-360M-Instruct). The loop is what you're learning, not the model. |
| First Modal runs (verl, 2× H100) | **Qwen3.5-2B → 4B** | Dense, cheap, fits FSDP full fine-tune. Where the HealthBench hill-climb and the three arms run first. |
| MoE stage (verl, 8× H100) | **Qwen3.6-35B-A3B** (or Qwen3.5-35B-A3B-Base if you want a base checkpoint) | Learn MoE post-training: expert parallelism, routing stability (MiMo's Routing Replay), train/rollout routing mismatch, and why labs use Megatron over FSDP for MoE. 35B total params rules out full FT on 2 GPUs: plan **LoRA, or a full 8×H100 node with the Megatron backend**. Only after the dense result exists. Confirm verl supports this architecture in your pinned version first (NVIDIA's Megatron-Bridge lists Qwen3.5/3.6 support). |
| OPD teacher | **Qwen3.6-27B** (dense) or Qwen3.5-27B | Same tokenizer family as the students (verify the vocabs match). A dense teacher keeps logprob serving simple. |
| Training judge | An open ~30B-class model from a **different family** than the teacher | Avoids the "judged by a relative of its teacher" confound. |
| Final eval | Local judge (calibrated vs physicians) + official grader on a capped subset | Ties back to published numbers. |

**Instruct vs base:** HealthBench RFT starts from the **instruct** model (it has to hold a
conversation). Base checkpoints are for a later "SFT → RL from base" comparison, not v1.

## Experiment design

1. **Judge calibration (stands alone):** local judge vs physician labels in `oss_meta_eval`.
   Agreement per criterion and per theme/axis (e.g. Cohen's κ or macro-F1). Also report how
   agreement varies with response length.
2. **Split rubric *criteria*, not just conversations:** per conversation, half the criteria form the
   training reward and half are held out. Gains on held-out criteria = learning the task; gains only
   on trained criteria = learning the grader.
3. **Three arms, same prompts, same step budget:**
   - A. **GRPO, rubric reward** (training criteria)
   - B. **OPD from the Qwen3.5 teacher** (no rubric reward at all)
   - C. **OPD + rubric reward** (the MiMo-style combined advantage)
4. **Decompose every delta:** trained vs held-out criteria; with and without the length penalty; per
   theme/axis; response-length drift over training; MentalHealthBench transfer, with emergency cases
   split out.

Prediction to write down *before* running it: A improves trained criteria most and mostly via
completeness and length; B moves held-out criteria and trained criteria together, capped near the
teacher; C sits between them. If it comes out differently, that's the README.

## Workstreams (week of 28 Sep onward)

### Priority by interview (read this first)

The full plan is weeks of work. Interviews come sooner, so cut by what each process tests:

| When | For | Do | Defer |
|---|---|---|---|
| **Now → Baseten technical rounds** | Baseten | Practical Python coding reps; vLLM lab rows 1–5; numbers from the work laptop; behavioural stories | Everything else here |
| **In parallel, evenings** | **Handshake** (likely next technical loop) | `nano/` stage A on rungs 0–1 (loss, advantages, KL, masking **by hand**); HealthBench data + judge-vs-physician calibration; RLHF book chapters; `study/post-training-2026.md` | Stages B/C |
| Once Handshake technical is scheduled | Handshake | `nano/` OPD variant (a teacher logprob as the advantage; one small change once A works); rung 2 | verl |
| If Cohere / Mistral move | Cohere: stage B (Ray actors, weight sync, staleness) + verl reading; Mistral: fine-tuning + eval story, in French | Modal verl runs, Ultra-Scale Playbook | MoE stage |
| After the interviews | The repo as a research artifact | Modal verl runs, three arms, MentalHealthBench, MoE stage, SkyRL/OpenRLHF pages, MLflow on Modal | — |

Five workstreams, in dependency order. 1 and 2 are the local core; 3 and 4 move to Modal; 5 runs
throughout.

### 0. Public-readiness (1 evening)
- Add a LICENSE (MIT).
- Tighten the README intro so it doesn't claim results that don't exist yet.
- Confirm no secrets in history (`.env` has never been tracked; checked 24 Sep).
- **Push public once `nano/` stage A works on rung 0/1** (after the rewrite, not before: the old
  code isn't what you want people to see). Visibility is a GitHub setting only you should flip.

### 1. Reorganise the repo around the plan

**Decision (24 Sep): scrap the old TRL + `llama-server` path and rewrite in `nano/`.** It was a
December prototype, and rewriting the loop yourself is the interview prep. Before deleting, tag the
current state (`git tag v0-trl-llamacpp`) so it's one command away if you ever want it back.

Target layout:

```
rl_finetune/               shared library, no trainer imports (used by nano/ AND verl)
  data/
    healthbench.py         download, normalise, criteria split
    mentalhealthbench.py   same schema; official source TBD
    medqa.py               verifiable healthcare toy task (see 2)
    schema.py              {id, messages, rubric:[{criterion, points, tags}], split, meta}
  rewards/
    rubric.py              score(conv, response, criteria) -> points; batched; cached
    verifiers.py           exact-match / MCQ scoring for toy tasks
  judge/client.py          OpenAI-compatible client, retries, cache (hash of conv+resp+criterion)
  tracking.py              thin wrapper: W&B and/or MLflow, one call site
nano/                      your pure-PyTorch + Ray trainer (workstream 2)
  protocol.py  algos.py  generator.py  learner.py  reference.py  reward_worker.py  driver.py
  configs/
modal/                     remote (workstream 3)
  images.py                pinned verl / vLLM images
  data_prep.py             normalised data -> parquet on a Volume
  judge.py                 vLLM judge as a Modal function
  verl_grpo.py             from Modal's example, updated
  mlflow_server.py         optional (see tracking)
analysis/                  calibrate.py, plots, per-axis breakdowns
notes/                     thought process (workstream 5)
```

Delete: `train.py`, `rollout.py`, `export.py`, `cli.py`, `rewards.py`, `scripts/serve.sh`,
`config/`, `envs/wordle*`, `evals/promptfoo`, `models/`, `README.draft.md` (fold what's useful into
the new README), `PLAN.v1.md` (it's in git history). Keep `dashboard.py` as a reference for a Rich
live view in `nano/driver.py`, or just use W&B. Rewrite `pyproject.toml` (package `rl_finetune`,
extras `[nano]`, `[modal]`; drop `openenv-textarena` and `llama.cpp`-specific bits) and the README
("what this repo is now"). Wordle stops being the smoke test; ladder rung 0 (arithmetic) replaces
it.

### 2. Data + the `nano/` trainer (local, Mac)

**Data first (1 evening).**
- `healthbench.py`: pull `oss_eval`, `hard`, `consensus` and **`oss_meta_eval`** from
  `openai/healthbench`, then normalise to `schema.py`.
- Implement the per-conversation criteria split with a fixed seed. Keep negative-points criteria in
  *both* halves, so neither half loses the "don't do harmful things" signal.
- `mentalhealthbench.py`: find the official release and licence first. Stub it until then.
- Write a tiny `data/README.md`: counts, themes/axes, criteria per conversation, points distribution.
  That's also your first notebook of observations.

**The toy-to-real ladder.** Healthcare data at every rung, and each rung changes one thing:

| Rung | Task | Reward | Why |
|---|---|---|---|
| 0 | Synthetic arithmetic | exact match | Debug the loop with a task you can't get wrong |
| 1 | **MedQA / MedMCQA** (multiple choice; check each dataset's licence) | exact match on the letter | Healthcare source, verifiable, and learnable by a 0.8B model on the Mac |
| 2 | HealthBench, ~50 short conversations | rubric reward via **API judge**, cached (a cheap model, hard spend cap) | The real reward shape, tiny scale. A local judge won't fit next to training in 24GB |
| 3 | Same on Modal with verl | rubric reward via the Modal vLLM judge | Workstream 3 |

**`nano/` design.** Build in stages A → B → C:

*Roles* (each one becomes a Ray actor in stage B):

| Role | Owns | verl analogue |
|---|---|---|
| `Driver` | The loop: sample prompts → rollout → reward → advantages → update → sync. Holds no weights. | single controller / `RayPPOTrainer` |
| `Generator` | Sampling: token ids + sampled-token logprobs | rollout worker (vLLM/SGLang) |
| `Learner` | Policy weights, optimizer, loss, `state_dict` for sync | actor worker (FSDP) |
| `Reference` | Frozen initial policy logprobs for KL (optional; β=0 is common now) | ref worker |
| `Reward` | `score(prompts, completions) -> float` (verifier first, rubric judge later) | reward manager |

*One batch type:* a dict of tensors (`input_ids`, `attention_mask`, `response_mask`,
`old_logprobs`, `ref_logprobs`, `rewards`, `advantages`) plus non-tensor metadata. That's your
`DataProto`. Every role takes and returns it.

*Stages:*
- **A. Single process, pure PyTorch.** Qwen3.5-0.8B (or the plain-attention fallback, see Models),
  HF `generate` for rollouts, ladder rungs 0 → 1. Write by hand:
  - completion-only masking with left-padded prompts
  - sampled-token logprobs via `log_softmax` + `gather`
  - group-normalised advantages
  - clipped ratio loss
  - k3 KL estimator
  - token-mean vs sequence-mean loss aggregation

  *Checks:* the ratio is exactly 1.0 on the first micro-batch after a rollout; reward rises on the
  toy task; entropy and mean length logged every step.

  *Memory budget (M4 Pro, 24GB unified; macOS gives the GPU roughly 70–75% of it by default):*
  - **Full fine-tune of 0.5–0.8B:** at 0.8B, bf16 weights ~1.6GB + grads ~1.6GB + AdamW states in
    fp32 ~6.4GB (+ ~3.2GB for fp32 master weights) → ~10–13GB, plus activations. It fits, but it's
    tight next to a generator; at 0.5B it's ~6–8GB.
  - **1.5B:** full FT won't fit comfortably. Use LoRA (the frozen base doubles as the reference
    model: disable the adapter to get reference logprobs).
  - **Keep activations small:** completions ≤256 tokens, group size 4–8, micro-batches of 1–2
    sequences with gradient accumulation, gradient checkpointing on.
  - **Skip the separate Reference model at first** (β=0), or use the LoRA-disable trick.
  - **No bitsandbytes or 8-bit Adam on MPS.** If memory is tight, drop the model size before
    anything clever.
  - **Pick a task a small model can learn:** ladder rungs 0–1 (arithmetic, MedQA/MedMCQA).
    HealthBench at rung 2 is a pipeline check, not a hill-climb, at this size.
  - **Stage B holds two copies of the weights** (Generator + Learner): budget ~2× the weights, or
    run the Generator on CPU for the exercise.
- **B. Same code, split into local Ray actors** (Ray runs on the Mac). Generator and Learner as
  separate processes, weights synced through the Ray object store every step (fully on-policy).
  Then add a **staleness knob** (generate step k+1 while training on step k) and correct for it with
  truncated importance sampling. That's async RL in miniature.
- **C. Swap the Generator for a real engine.** On Modal: vLLM, with weights pushed through vLLM's
  weight-loading path or NCCL. Locally, the `llama-server` path shows why a quantised generator
  breaks the ratio. Once the loop is understood, real runs move to verl.

*Read verl alongside each stage.* Read the code path that does the thing you just wrote, then
compare. Paths are from recent verl releases; the tree moves, so check the version you pin.

| While building | Read in verl | What to compare |
|---|---|---|
| A. loss and advantages | `verl/trainer/ppo/core_algos.py` (GRPO advantage, policy loss, KL estimators, loss aggregation modes) | Your advantage normalisation, clipping and token-mean vs seq-mean against theirs |
| A. logprobs and update | `verl/workers/actor/dp_actor.py` (`compute_log_prob`, `update_policy`) | Micro-batching, masking, entropy, how old logprobs are reused across PPO epochs |
| A/B. batch type | `verl/protocol.py` (`DataProto`: `batch` TensorDict + `non_tensor_batch` + `meta_info`) | Your batch dict, and why they chunk and concatenate it for dispatch |
| B. the driver | `verl/trainer/ppo/ray_trainer.py` (`RayPPOTrainer.fit`) | The step order: generate → reward → old/ref logprobs → advantages → update → metrics |
| B. roles and dispatch | `verl/single_controller/` (Ray `WorkerGroup`, `@register` dispatch modes) and `verl/workers/fsdp_workers.py` (`ActorRolloutRefWorker`) | How one controller fans a batch out to N workers, and why actor, rollout and ref share one worker ("hybrid engine") |
| C. weight sync | the rollout sharding manager (FSDP → vLLM weight resharding) under `verl/workers/` | Your object-store sync vs in-place resharding on colocated GPUs |
| C. rollout | `verl/workers/rollout/` (vLLM / SGLang rollouts, async mode) | Token ids and logprobs out of the engine; how rollout vs training mismatch is handled |
| Arm B | verl OPD docs (https://verl.readthedocs.io/en/latest/algo/opd.html) | How the teacher logprob becomes the advantage |
| Background | HybridFlow paper (verl's design) and the verl programming guide | Single controller + multi-controller workers: why this split |

*Rules that prevent the classic bugs:*
- **Never re-tokenise generated text.** Carry the token ids end to end.
- **Same temperature and top-p** for sampling and for the logprob recomputation.
- **Mask the prompt and padding** out of the loss; handle EOS explicitly.
- **Log the metrics that show you're wrong:** reward, KL, clip fraction, ratio min/max, entropy,
  response length, and the logprob gap between generator and learner.

### 3. Modal: verl on NVIDIA GPUs + judge + tracking

- **Image:** start from Modal's `grpo_verl` example, but bump to a current verl image whose vLLM
  supports Qwen3.5. Pin exact versions in `modal/images.py`.
- **Smoke test:** GSM8K on Qwen3.5-2B, `H100:2`. Done when the reward curve moves and a checkpoint
  lands on the Volume.
- **Swap the data and reward:** HealthBench parquet from `modal/data_prep.py`, and
  `rl_finetune/rewards/rubric.py` as `custom_reward_function`. The judge is called over HTTP from
  the reward function.
  - verl calls the reward per sample by default. With ~10–40 criteria per conversation, **batch
    the judge calls** (a batch or async reward manager, or a batching client with a cache).
    Otherwise the reward step dominates the step time.
- **Judge:** `modal/judge.py` serves vLLM on its own GPU (`@modal.web_server` or a class with
  `@modal.concurrent`). Scale to zero between runs.
- **GPU sizing:** 2× H100 for 2–4B dense, plus 1 GPU for the judge. A full 8× H100 node only for
  the 35B-A3B MoE stage.

**Experiment tracking: W&B vs MLflow on Modal.**

| | W&B (free) | MLflow on Modal |
|---|---|---|
| Setup | `wandb login`, set `trainer.logger=['console','wandb']` in verl | Deploy `mlflow server` as a Modal web server, sqlite + artifacts on a Volume, point `MLFLOW_TRACKING_URI` at it |
| Cost | Free plan: 5 GB/month storage, **personal projects only** (fine for this repo; not for AlayaCare work) | Pay for the container while it's up |
| Ops | None | Yours: auth (Modal endpoints are public unless you enable proxy auth), sqlite on a network volume, uptime |
| Learning curve | An afternoon | You already know MLflow |
| verl support | Native logger backend | Native logger backend (`mlflow`), via `MLFLOW_TRACKING_URI` |

**Recommendation:** W&B for this project. Zero ops, verl logs to it natively, the free tier covers
personal use, and public W&B reports are easy to link from the README. Both are thin wrappers in
`rl_finetune/tracking.py`, so `nano/` can log to either. Add `modal/mlflow_server.py` later only if
you want the self-hosting practice. Check the verl logger list against your pinned version.

### 4. Experiments (the design above)
Calibration → arm A → arms B/C → MentalHealthBench transfer → MoE stage. Update the README the day
each result exists.

### 5. Documentation: thought process + framework reading

```
notes/
  journal.md        dated entries: what I tried, what I expected, what happened, what I changed
  verl/             one file per component read, matching the reading map in 2
  frameworks.md     verl vs SkyRL vs OpenRLHF
  decisions.md      short ADR-style: model choice, W&B vs MLflow, verl vs own loop, etc.
```

- **Journal rule:** write the prediction *before* the run. That's what makes it read as research
  taste later, and it feeds the README's "what surprised me".
- **Frameworks:** verl is the focus (most used, and Modal's example uses it). For SkyRL and OpenRLHF,
  one page each is enough: the controller model, how rollout and training share or split GPUs,
  weight sync, async support, and MoE support. You already compared them for throughput at work, so
  write the *design* comparison here. That's the interview-grade version.
- Everything in `notes/` is public-safe. No AlayaCare internals.

## Budget and guardrails

- Set a Modal spend limit and an API cap for the official grader before step 4.
- Check Modal's current H100 pricing when you size runs.
- Cache every judge call.
- One seed first; add seeds only for the arm that makes the headline claim.

## Definition of done (public v1)

- [ ] Repo public with LICENSE, and the README honest about current state
- [ ] `modal/` app: train, serve and judge functions, with pinned image versions
- [ ] HealthBench env + rubric reward with caching; criteria split
- [ ] Judge-vs-physician agreement number in the README
- [ ] Arm A: trained vs held-out criteria, length-penalty and per-axis breakdown
- [ ] Arms B/C and MentalHealthBench transfer (v1.1 if time runs out)
- [ ] `nano/` stages A–B working locally, rung 1 (MedQA) reward rising
- [ ] `notes/`: journal, verl component notes, frameworks page

## Where this leads after the week

The HealthBench work is a natural on-ramp to open-source contributions, and OSS is the fastest public
proof of post-training depth available to you. Don't treat it as a separate project — it falls out of
this one if you're paying attention.

Ranked by fit:

**1. veRL.** Built on Ray. Distributed rollout orchestration for RL is *the* hard infra problem in
post-training right now, and it's the same problem you already solve at work — you're just used to
calling it serving rather than rollout generation. This is the target where you contribute from
strength rather than scrambling to keep up, and it's the clearest public demonstration that your
platform background is load-bearing in RL rather than adjacent to it. You've already evaluated it
against OpenRLHF and TRL, which means you have opinions worth writing down.

**2. TRL.** You're already using `GRPOTrainer`. You will hit real gaps building a rubric-graded
multi-turn environment on top of it — reward-function interfaces, multi-turn rollout handling, judge
caching. Fix one and upstream it. HF maintainers are responsive and the post-training world reads that
repo.

**3. OpenEnv.** Environments are a genuine bottleneck in the RL ecosystem. A rubric-graded, LLM-judged
healthcare environment is a contribution nobody has made, and you'd be donating the thing you built
this week rather than writing new code.

**4. vLLM.** You operate it in production. Rollout-generation paths for RL training are actively
developed. Also directly relevant to the Baseten inference roles.

## On communities

Interconnects and similar are worth being in, but lurking pays almost nothing. The value comes from
posting work and asking specific questions, and the HealthBench rubric-gaming result is exactly the
kind of thing those rooms engage with. Post it when you have it. That converts membership into being
known by people who work at the labs you're applying to — which is a different and much better asset
than a Discord login.
