from __future__ import annotations


def wordle_win_reward(environments: list, **kwargs) -> list[float]:
    """Binary reward: 1.0 if the model guessed the word, 0.0 otherwise.

    The Wordle environment sets `env.reward` to 1.0 on a successful guess
    within 6 turns and 0.0 on failure.
    """
    return [float(getattr(env, "reward", 0.0)) for env in environments]


def wordle_efficiency_reward(environments: list, **kwargs) -> list[float]:
    """Bonus reward for solving in fewer turns (up to +0.5).

    Only applied when the model wins. Encourages guessing efficiently:
    - Solve in 1 turn: +0.50
    - Solve in 2 turns: +0.45
    - ...
    - Solve in 6 turns: +0.25
    - Failed: 0.0
    """
    rewards = []
    for env in environments:
        won = getattr(env, "reward", 0.0) > 0
        turns = getattr(env, "turns_used", 6)
        rewards.append(max(0.0, 0.5 - turns * 0.05) if won else 0.0)
    return rewards
