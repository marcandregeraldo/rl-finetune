# README shape — fill in once you have numbers

Your current README opens with prerequisites and a quick start. That's a *user's* README. What you need is
a *reader's* README: someone spending ninety seconds deciding whether you think well. Setup instructions
matter to them only after they've decided you do.

The rule: **the first screen has to contain a result and a surprise.** Everything else moves down.

Replace the file with this shape.

---

```markdown
# rl-finetune

RLVR on a benchmark whose grader is itself a model — and what happened when I checked whether
the gains were real.

HealthBench grades responses with an LLM judging against physician-written rubrics. I used that
rubric score as a GRPO reward on Qwen3.5-2B, running entirely on an M-series Mac, and then held out
rubric *criteria* to test whether the model learned the task or learned the grader.

## Result

<!-- Plot 2: train vs held-out criteria, before/after -->

| | Baseline | After GRPO | Δ |
|---|---|---|---|
| Trained rubric criteria | 0.___ | 0.___ | +___ |
| **Held-out rubric criteria** | 0.___ | 0.___ | **+___** |
| Held-out, with official length penalty | 0.___ | 0.___ | +___ |
| Mean response length (chars) | ___ | ___ | ___ |

<!-- One paragraph: what the gap between rows 1 and 2 means. This is the whole point of the repo. -->

Absolute scores are low because this is a 2B model on a benchmark where frontier models scored in the
low 0.3s at release. The claim here is about the *shape* of the improvement, not its size.

## What surprised me

<!-- Three short paragraphs. This section is why anyone reads the repo. Be specific and concrete:
     a number, a behaviour you saw in rollouts, a thing you had to go back and fix.

     Candidates, depending on what you actually find:
     - where the local training judge disagreed with gpt-4.1, and whether the disagreement was
       systematic by tag rather than random
     - which rubric tags absorbed the gain, and whether completeness ate the rest
     - what the model started doing to rollouts that satisfied criteria without being better
       (hedging, enumerating, restating the question, safety-netting every answer with "see a doctor")
     - what response length did over training, and when you noticed
     - anything you had to throw away

     Do not write "further work is needed." Write what you saw. -->

## Judge calibration

<!-- Plot 1: local judge vs gpt-4.1 grader agreement -->

Training against `gpt-4.1` on every rollout isn't affordable, so the training reward comes from a local
judge and evaluation uses the official grader. Before training I measured how far apart those two are:
___ correlation over ___ responses, with the disagreement concentrated in ___.

<!-- Say what you did about it, or say that you accepted it and why. Both are fine answers.
     "I accepted a known bias and here's how it bounds the result" is a research answer. -->

## Method

- **Model** — Qwen3.5-2B (Q4_K_M), LoRA, rollouts through Metal-accelerated `llama-server`
- **Algorithm** — GRPO via TRL `GRPOTrainer`, group size 4
- **Reward** — per-rubric-item scoring, achieved / total possible positive points, clipped to [0,1]
- **Data** — [HealthBench](https://huggingface.co/datasets/openai/healthbench) (MIT), ___ conversations,
  rubric criteria split ___/___ into training reward and held-out evaluation
- **Evaluation** — official [simple-evals](https://github.com/openai/simple-evals) grader, `gpt-4.1`

<!-- Keep the architecture diagram here. It's good and it's already drawn. -->

## Limitations

<!-- Be direct. A short honest list here reads as confidence, not weakness:
     - single seed / n runs
     - subset size
     - the local judge's bias bounds what you can claim
     - 2B model, LoRA, quantized rollouts — what that does and doesn't let you conclude
     - what you'd run with a real cluster, in one sentence, not a wishlist -->

## Reproduce

<!-- Everything currently at the top of your README goes here, unchanged. Prerequisites, install,
     model download, `rl env start`, `rl serve`, `rl train`, `rl eval`. It's good material —
     it's just not the opening argument. -->

## Also in here

`envs/wordle.py` — the original multi-turn OpenEnv task, kept as the smoke test for the training loop.
```

---

## Before making it public

- [ ] `.env` is in `.gitignore` **and** confirm no key was ever committed — `git log -p -- .env` and
      check `keys/` in your other repos too. If anything leaked, rotate it and rewrite history before
      pushing.
- [ ] Add a LICENSE (MIT is the obvious choice and matches HealthBench's).
- [ ] Remove the stray `--local-dir` and `huggingface-cli` files sitting in the repo root — they're
      artifacts of a mistyped command and they're the first thing a careful reader notices.
- [ ] Check the `.gguf` model files and `checkpoints/` aren't tracked; they'll blow up the clone.
- [ ] Make sure the repo description and topics on GitHub say what it is. It's the only text a recruiter
      sees before deciding to click.
