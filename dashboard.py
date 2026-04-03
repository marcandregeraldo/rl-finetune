from __future__ import annotations

import torch
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TimeRemainingColumn
from rich.table import Table
from rich.text import Text
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

_SPARK_CHARS = " ▁▂▃▄▅▆▇█"

console = Console()


def _sparkline(values: list[float], width: int = 20) -> str:
    """Render a list of floats as a Unicode sparkline string."""
    if not values:
        return " " * width
    min_v = min(values)
    max_v = max(values)
    span = max_v - min_v or 1.0
    chars = [
        _SPARK_CHARS[int((v - min_v) / span * (len(_SPARK_CHARS) - 1))]
        for v in values[-width:]
    ]
    return "".join(chars)


def _mps_memory_gb() -> float | None:
    """Return current MPS allocated memory in GB, or None if not available."""
    if torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / 1e9
    return None


class DashboardCallback(TrainerCallback):
    """Rich Live dashboard callback for TRL GRPOTrainer.

    Usage:
        trainer = GRPOTrainer(..., callbacks=[DashboardCallback()])
    """

    def __init__(self, reward_history_len: int = 40):
        self.reward_history: list[float] = []
        self.reward_history_len = reward_history_len
        self.latest_logs: dict = {}
        self.latest_generation: str = ""
        self.total_steps: int = 0
        self._live: Live | None = None
        self._progress = Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        )
        self._task_id = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.total_steps = state.max_steps or (args.num_train_epochs * state.num_train_epochs)
        self._task_id = self._progress.add_task("[cyan]Training", total=self.total_steps)
        self._live = Live(self._render(), console=console, refresh_per_second=4, transient=False)
        self._live.start()

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self._live:
            self._live.stop()
            self._live = None
        console.print("\n[bold green]Training complete.[/bold green]")

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ):
        if logs:
            self.latest_logs.update(logs)
            if "rewards/mean" in logs:
                self.reward_history.append(logs["rewards/mean"])
                if len(self.reward_history) > self.reward_history_len:
                    self.reward_history.pop(0)
        self._refresh()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self._task_id is not None:
            self._progress.advance(self._task_id)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        self._refresh()

    def on_prediction_step(self, *args, completions=None, **kwargs):
        """Capture a generation sample to display in the dashboard."""
        if completions:
            sample = completions[0] if isinstance(completions[0], str) else str(completions[0])
            self.latest_generation = sample[:300]
        self._refresh()

    def _refresh(self):
        if self._live:
            self._live.update(self._render())

    def _render(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="top", size=3),
            Layout(name="middle", size=10),
            Layout(name="bottom"),
        )
        layout["top"].update(self._progress)
        layout["middle"].update(self._metrics_panel())
        layout["bottom"].update(self._generation_panel())
        return layout

    def _metrics_panel(self) -> Panel:
        logs = self.latest_logs
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column()

        def _fmt(key: str, fmt: str = ".4f") -> str:
            v = logs.get(key)
            return f"{v:{fmt}}" if v is not None else "[dim]—[/dim]"

        table.add_row("loss", _fmt("loss"))
        table.add_row("reward (mean)", _fmt("rewards/mean"))
        table.add_row("KL divergence", _fmt("kl"))
        table.add_row("learning rate", _fmt("learning_rate", ".2e"))
        table.add_row("step", str(logs.get("step", "—")))
        table.add_row("reward trend", f"[yellow]{_sparkline(self.reward_history)}[/yellow]")
        mem = _mps_memory_gb()
        if mem is not None:
            table.add_row("MPS memory", f"{mem:.2f} GB")

        return Panel(table, title="[bold]Metrics[/bold]", border_style="blue")

    def _generation_panel(self) -> Panel:
        if self.latest_generation:
            text = Text(self.latest_generation, overflow="fold")
        else:
            text = Text("[dim]Waiting for first generation...[/dim]")
        return Panel(text, title="[bold]Latest Generation[/bold]", border_style="dim")
