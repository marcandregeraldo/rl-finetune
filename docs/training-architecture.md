# Training architecture

How `nanorft` trains a policy, from one process on the Mac to Ray actors on Modal. Written as
it's built; each section says which milestone it lands in.

## Compute: Ray on Modal

![Ray on Modal: the Mac spawns a 2-node clustered function; rank 0 runs the Ray head and submits the experiment as a Ray Job, rank 1 joins as a worker](diagrams/ray-on-modal.svg)

Every experiment is a module run as `python -m <module> --config <yaml>` that calls
`ray.init()`. [`modal/ray_cluster.py`](../modal/ray_cluster.py) brings up a Ray cluster on
Modal and submits that module as a Ray Job. It runs on one node (1–8 H100s) or two 8-GPU nodes
with RDMA. The same module runs on the Mac against a local Ray. Before a multi-node run,
`--entry check` ([`cluster_check.py`](../src/nanorft/cluster_check.py)) checks NCCL bandwidth.

## Roles (M1 in one process, M7 as Ray actors)

| Role | File | Owns | Status |
|---|---|---|---|
| Driver | `src/nanorft/driver.py` | The loop; holds no weights | |
| Generator | `src/nanorft/generator.py` | Sampling: token ids + logprobs | |
| Learner | `src/nanorft/learner.py` | Weights, optimizer, loss (`forward_backward` / `optim_step`) | |
| Reference | `src/nanorft/reference.py` | Frozen logprobs for KL (optional) | |
| Reward | `src/nanorft/reward_worker.py` | Verifier, later the rubric judge | |

## The batch (`protocol.py`)

<!-- Keys, shapes, padding side, what response_mask covers (EOS?), meta. -->

## One training step

<!-- The step order, and what each role reads and writes. A diagram goes here once the loop runs. -->

## Algorithms (`algos.py`)

### Logprobs

<!-- log_softmax + gather, vocab chunking, temperature. -->

### Advantages

<!-- GRPO group normalisation; norm_by_std on/off; groups with zero variance. -->

### Policy loss

<!-- Clipped ratio, eps_low / eps_high, what the clip fraction means. -->

### KL

<!-- k3 estimator; why β = 0 at first. -->

### Loss aggregation

<!-- token-mean vs seq-mean-token-sum vs seq-mean-token-mean, and which one is used. -->

## Learner (`learner.py`)

<!-- Micro-batches, gradient accumulation, checkpointing, memory on the Mac. -->

## Generator (`generator.py`)

<!-- HF generate, left padding, chat template, why generate's logprobs differ from the learner's. -->

## Rewards

<!-- Rung 0 arithmetic verifier; rung 1 letter match; HealthBench rubric + judge later. -->

## Checks and metrics

<!-- Ratio == 1.0 on the first micro-batch; gen-vs-learner gap; reward, KL, clip fraction,
ratio min/max, entropy, length. What each one looked like when something was wrong. -->

## Decisions and surprises

<!-- Dated: what you expected, what happened, what you changed. -->
