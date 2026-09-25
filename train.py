from __future__ import annotations

import logging
from pathlib import Path

import click
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from dashboard import DashboardCallback
from envs.wordle import make_wordle_env
from export import WeightSyncCallback
from rewards import wordle_efficiency_reward, wordle_win_reward
from rollout import make_rollout_func

logger = logging.getLogger(__name__)


def build_wordle_dataset(num_words: int = 2000):
    """Build a simple Wordle training dataset.

    Returns a HF Dataset with a 'prompt' column of chat messages.
    The model must guess the target word through multi-turn interaction.
    """
    # Same prompt structure; the Wordle environment picks a different secret word each episode.
    prompts = [
        [
            {
                "role": "system",
                "content": (
                    "You are playing Wordle. You must guess a secret 5-letter word. "
                    "After each guess, you receive colored feedback:\n"
                    "🟩 = correct letter, correct position\n"
                    "🟨 = correct letter, wrong position\n"
                    "⬜ = letter not in word\n\n"
                    "Use the feedback to narrow down your guesses. "
                    "You have up to 6 attempts. Reply with only your 5-letter guess."
                ),
            },
            {"role": "user", "content": "Let's play Wordle. Make your first guess."},
        ]
    ] * num_words

    return Dataset.from_dict({"prompt": prompts})


def load_config(config_path: str = "config/default.yaml", overrides: dict | None = None) -> dict:
    """Load training config from YAML with optional overrides."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if overrides:
        config.update({k: v for k, v in overrides.items() if v is not None})
    return config


def main(config: dict):
    """Run GRPO training with the given config.

    Args:
        config: Dict loaded from config/default.yaml (possibly with overrides).
    """
    output_dir = config.get("output_dir", "checkpoints/grpo")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    dataset = build_wordle_dataset()

    peft_config = LoraConfig(
        r=config.get("lora_r", 8),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=config.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        bias="none",
        task_type="CAUSAL_LM",
    )

    grpo_args = GRPOConfig(
        output_dir=output_dir,
        max_steps=config.get("max_steps") or -1,
        num_train_epochs=config.get("num_train_epochs") or 1,
        learning_rate=config.get("learning_rate", 5e-6),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        max_grad_norm=config.get("max_grad_norm", 1.0),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        num_generations=config.get("num_generations", 4),
        max_completion_length=config.get("max_completion_length", 1024),
        bf16=config.get("bf16", True),
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        optim=config.get("optim", "adamw_torch"),
        logging_steps=config.get("logging_steps", 1),
        report_to="none",
        save_strategy="steps",
        save_steps=config.get("save_steps", 50),
        save_total_limit=config.get("save_total_limit", 3),
        push_to_hub=False,
    )

    sync_every = config.get("sync_every", 50)
    callbacks = [DashboardCallback()]
    if sync_every > 0:
        callbacks.append(WeightSyncCallback(config, sync_every=sync_every)())

    rollout_func = make_rollout_func(config)
    model_name = config.get("model", "Qwen/Qwen3.5-2B")
    logger.info("Loading model: %s", model_name)

    model_init_kwargs = {}
    if "device_map" in config:
        model_init_kwargs["device_map"] = config["device_map"]
    if "offload_folder" in config:
        model_init_kwargs["offload_folder"] = config["offload_folder"]

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_init_kwargs) if model_init_kwargs else model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[wordle_win_reward, wordle_efficiency_reward],
        args=grpo_args,
        train_dataset=dataset,
        peft_config=peft_config,
        rollout_func=rollout_func,
        environment_factory=make_wordle_env,
        callbacks=callbacks,
    )

    try:
        trainer.train()
    finally:
        for env in getattr(trainer, "environments", None) or []:
            if hasattr(env, "close"):
                env.close()

    trainer.save_model(str(Path(output_dir) / "final"))
    logger.info("Training complete. Final checkpoint saved to %s/final", output_dir)


if __name__ == "__main__":
    @click.command()
    @click.option("--config", "-c", default="config/default.yaml")
    @click.option("--max-steps", type=int, default=None)
    @click.option("--env", type=str, default=None)
    def _cli(config, max_steps, env):
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        cfg = load_config(config, {"max_steps": max_steps, "env": env})
        main(cfg)

    _cli()
