# One-week plan: make this repo the artifact

## What the repo has to prove

Right now it proves you can wire up a GRPO loop. That's competent, and it's what a tutorial proves too.

What ElevenLabs, Cohere and Baseten are screening for is **research taste** — whether you know which
experiment is worth running and which measurement you shouldn't trust. So the repo needs to answer a
question, not demonstrate a pipeline. The Wordle environment can stay as the smoke test. It shouldn't be
the headline.

## The question

> HealthBench grades model responses with an LLM judging against physician-written rubrics. If you use
> that rubric score as an RLVR reward, does the model get better at health conversations — or better at
> the grader?

This is the exact shape of the thing you say you care about: post-training for domain adaptation on a
task whose correctness is hard to verify. It's in your domain. It's on a public MIT-licensed benchmark,
so a reviewer can check you. And the honest answer is almost certainly "some of both," which is a far more
interesting README than a learning curve going up.

One hint that you're pulling on a real thread: OpenAI's own grader has an optional **length penalty**
(`penalty_per_500_chars`). They built a verbosity-gaming correction into the scorer. That's a strong
signal that rubric gaming is a known, unsolved failure mode — and it means you're not inventing a problem.

## Dataset

- **HealthBench** — https://huggingface.co/datasets/openai/healthbench · **MIT license** · 246 MB
- Reference grader implementation — https://github.com/openai/simple-evals/blob/main/healthbench_eval.py

Each example is a multi-turn conversation plus a list of rubric items: `{criterion, points, tags}`.
Points can be negative (undesirable behaviors). Score = achieved points / total possible positive points,
clipped to [0,1]. The official grader is `gpt-4.1` returning `{explanation, criteria_met}` per item.

Multi-turn conversations wrap into your existing environment interface with very little new code — this is
why it fits your repo rather than requiring a rewrite.

## The experiment design (this is the part that matters)

Three measurements, and the second one is the whole point.

**1. Cheap judge for training, official grader for evaluation.**
Grading with `gpt-4.1` on every rollout is unaffordable. So use a local Qwen judge (or `gpt-4.1-mini`) as
the *training* reward, and the official `gpt-4.1` grader only for *evaluation*. Before you train, measure
agreement between the two on ~150 responses — report correlation and per-tag agreement. That number is
your judge-calibration result, and it costs you one evening and a few dollars.

**2. Hold out rubric criteria, not just conversations.**
For each conversation, split its rubric items: train the reward on half, evaluate on the other half. If
the score improves on *held-out criteria*, you learned something about health conversations. If it only
improves on the criteria you trained against, you learned the verifier. This split is the sharpest thing
in the whole plan — it's a clean, direct measurement of the thing everyone hand-waves about, and it's the
part a research interviewer will want to talk about.

**3. Decompose the gain.**
Report the delta three ways: with and without the official length penalty; broken down by rubric tag
(accuracy, completeness, context-awareness, emergency referral); and against response-length drift. My
guess is most of the gain lands in *completeness* and shrinks under the length penalty — because saying
more is the cheapest way to satisfy more criteria. If that's what you find, say it plainly; it's a better
result than a clean win.

## Day by day

Evenings are ~3h, weekend is the long block. Cut scope from the bottom, not the top.

**Day 1 — data and environment**
Pull HealthBench, filter to a working subset (200–300 conversations; consider a tag-filtered slice so the
distribution is coherent). Write `envs/healthbench.py` against your existing env interface. Get baseline
rollouts from the untuned 2B and eyeball twenty of them. Commit.

**Day 2 — reward and judge calibration**
Implement rubric-item scoring in `rewards.py` with aggressive caching (hash on
conversation + response + criterion). Score ~150 baseline responses with both the local judge and the
official `gpt-4.1` grader. Plot agreement, compute correlation and per-tag agreement. **Write this result
down even if you run out of time for everything else** — it stands alone.

**Day 3 — splits and baseline**
Implement the train/held-out rubric-criteria split. Run the official grader on the baseline model to get
your reference numbers on both halves. This is your "before."

**Day 4 — the run**
GRPO against the cheap-judge reward on the training criteria. Keep `num_generations` at 4 and `max_steps`
small — you are not trying to saturate, you are trying to move the number enough to measure. Watch mean
response length as a training curve; it's a leading indicator of gaming.

**Weekend — evaluation and writeup**
Official grader on the tuned model: train criteria, held-out criteria, with and without length penalty,
per-tag breakdown. Three plots (below). Rewrite the README. Make the repo public.

**Optional, only with a spare evening — the ElevenLabs-specific angle**
[PriMock57](https://github.com/babylonhealth/primock57): 57 mock primary-care consultations with audio,
utterance-level human transcripts, clinician notes, and human evaluation annotations. No training needed —
build a clinically-weighted transcription error metric (medication, dosage and negation errors weighted
above filler words) and show it ranks ASR outputs differently than WER does. Half a day, and it speaks
directly to ElevenLabs' audio domain. **Check the LICENSE file before redistributing anything derived from
it** — the README doesn't state one.

Note that you don't strictly need the audio piece for ElevenLabs. Their RE posting lists three
responsibilities — post-training, data curation, and benchmarks/evaluation frameworks — and the HealthBench
work hits all three. The methodology transfers even though the modality doesn't.

## Expectation setting

**A 2B model will score badly on HealthBench in absolute terms.** The benchmark is hard; frontier models
were in the low 0.3s at release. You might be at 0.10–0.15. That is completely fine and you should say so
in the README. Your result is a *relative* delta and a *decomposition* of that delta — not a leaderboard
position. Anyone evaluating you knows what a 2B model on a Mac can do, and claiming otherwise would be the
actual red flag.

## Three plots, no more

1. **Judge agreement** — local judge vs `gpt-4.1` grader, scatter with correlation, coloured by tag.
2. **Train vs held-out criteria** — before/after bars. The gap between the two bars is your headline.
3. **Gain decomposition** — per-tag delta, plus the same delta with the length penalty applied.

## What not to do

- Don't switch to a bigger model. The result is about method, not scale.
- Don't add a second environment. Wordle stays as the smoke test.
- Don't build a web UI or a dashboard beyond what exists.
- Don't chase a positive result. "It improved the score and most of it was verbosity" is a *better*
  portfolio outcome than a clean win, because it's the thing most people don't check.
- Don't run this past Sunday. A public repo with one honest result beats a private repo with three.

## Definition of done

- [ ] `envs/healthbench.py` and rubric reward in `rewards.py`, with cached grading
- [ ] `analysis/` script producing the three plots from saved run artifacts
- [ ] Judge-agreement number written down
- [ ] Train vs held-out criteria number written down
- [ ] README rewritten per `README.draft.md` — result first, setup last
- [ ] Repo public, LICENSE added, `.env` and any API keys confirmed out of git history

---

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
