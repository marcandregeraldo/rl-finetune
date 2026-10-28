# Data

[HealthBench](https://openai.com/index/healthbench/) from OpenAI's
[simple-evals](https://github.com/openai/simple-evals) (MIT). Physicians wrote a
rubric for each of 5,000 health conversations: 2 to 48 criteria, each worth +1 to
+10 points, or −1 to −10 for something harmful. All files are pinned by sha256 in
`src/rubricrl/data/healthbench.py`.

The rubrics are used exactly as shipped: the same text, points and tags, in the
same order, and the grader sees them as simple-evals' `str(RubricItem)`,
`[points] criterion` (`tests/test_healthbench.py` checks every criterion). The
57,237 criteria have 48,562 distinct texts, the paper's count: the cluster-level
criteria repeat across conversations.

```bash
uv run python -m rubricrl.data.healthbench download       # data/raw/ (~250 MB, gitignored)
uv run python -m rubricrl.data.healthbench stats          # the tables below
uv run python -m rubricrl.data.healthbench export-miles   # data/miles/ (gitignored)
```

| File | What |
|---|---|
| `raw/…oss_eval.jsonl` | The 5,000 conversations and rubrics. `hard` and `consensus` are subsets of it |
| `raw/…oss_meta_eval.jsonl` | 29,511 physician verdicts on whether a model completion meets a rubric, for judge calibration |
| `miles/healthbench_train.jsonl` | 3,754 train conversations in Miles' format (`prompt` = messages; `metadata` = messages, `train_criteria`, `heldout_criteria`, `eval`) |
| `miles/healthbench_val.jsonl` | 64 val conversations for the in-loop eval (`metadata.eval = true`: graded on held-out criteria) |
| `miles/healthbench_test.jsonl` | The 200 eval conversations, for `modal/sample.py` |
| `eval_ids.txt` | **Tracked.** The 200 test `prompt_id`s, stratified by theme. Frozen: `export-miles` reuses this file if it exists |

## Splits

- **Conversations:** seeded hash of `prompt_id` → train 75% / val 5% / test 20%.
- **Criteria, within each conversation:** the positive criteria are shuffled
  (seeded) and alternated into two disjoint halves: *training* (the reward) and
  *held-out* (eval only). With an odd count, training gets the extra one. Negative
  criteria go in **both** halves, so the reward always penalises harm.
  - On the training split, that gives 4.2 positive training criteria, 3.8
    held-out ones and 3.6 negatives per conversation. That's about 7.8 criteria
    to grade per training response.
  - Each half divides by its own positive points, but carries every negative.
    So a half-score sits below the full-rubric score, and penalises a harm about
    twice as heavily. Compare the halves by their **gains**, not their levels.
- **Dropped:** 50 conversations with a single positive criterion (they can't be
  split), and 1 prompt over 2,048 tokens.

## Stats

| Split | Conversations | With >= 2 positive criteria |
|---|---|---|
| train | 3791 | 3755 |
| val | 247 | 244 |
| test | 962 | 951 |
| all | 5000 | 4950 |

| Rubrics | Value |
|---|---|
| criteria | 57237 |
| per conversation: min / median / max | 2 / 11 / 48 |
| positive / negative | 39662 / 17575 |
| example-level / cluster-level | 49184 / 8053 |
| single-turn conversations | 2915 (58%) |

| Axis | Criteria | Share |
|---|---|---|
| completeness | 22285 | 39% |
| accuracy | 18888 | 33% |
| context_awareness | 8991 | 16% |
| communication_quality | 4522 | 8% |
| instruction_following | 2551 | 4% |

| Theme | Conversations |
|---|---|
| global_health | 1097 |
| hedging | 1071 |
| communication | 919 |
| context_seeking | 594 |
| emergency_referrals | 482 |
| health_data_tasks | 477 |
| complex_responses | 360 |

| oss_meta_eval | Value |
|---|---|
| judgments (completion x rubric) | 29511 |
| physician labels | 60896 |
| share labelled 'met' | 77% |
| physicians | 186 |
| categories | 34 |
