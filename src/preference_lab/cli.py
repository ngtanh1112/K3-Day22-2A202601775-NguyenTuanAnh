from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich import print

from .config import load_config
from .data import load_jsonl
from .evaluate import pairwise_accuracy, write_metrics
from .trainers import PreferenceTrainer, TrainingConfig

app = typer.Typer(help="Preference alignment lab CLI")


def compute_response_score(prompt: str, response: str) -> float:
    """Deterministic score approximating preference alignment quality."""
    prompt_tokens = set(prompt.lower().split())
    resp_tokens = response.lower().split()
    overlap = sum(1 for tok in resp_tokens if tok in prompt_tokens)
    length_weight = min(len(resp_tokens) / 25.0, 1.2)
    return float((0.5 * overlap) + (0.8 * length_weight))


@app.command()
def validate(data: Path) -> None:
    examples = load_jsonl(data)
    print(f"[green]Loaded {len(examples)} preference examples[/green]")


@app.command()
def evaluate(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to configuration YAML file",
        ),
    ] = Path("configs/local.yaml"),
) -> None:
    cfg = load_config(config)
    examples = load_jsonl(cfg["paths"]["train_data"])

    chosen_scores = [compute_response_score(ex.prompt, ex.chosen) for ex in examples]
    rejected_scores = [compute_response_score(ex.prompt, ex.rejected) for ex in examples]

    acc = pairwise_accuracy(examples, chosen_scores, rejected_scores)

    training_cfg = TrainingConfig(
        method=cfg.get("training", {}).get("method", "dpo"),
        beta=cfg.get("training", {}).get("beta", 0.1),
        lambda_orpo=cfg.get("training", {}).get("lambda_orpo", 0.1),
        output_dir=cfg["paths"]["output_dir"],
    )
    trainer = PreferenceTrainer(training_cfg, examples)
    train_res = trainer.train()

    metrics = {
        "pairwise_accuracy": round(acc, 4),
        "final_loss": round(train_res["final_loss"], 4),
        "num_examples": float(len(examples)),
    }
    out = write_metrics(metrics, cfg["paths"]["output_dir"])
    print(f"[green]Wrote metrics to {out}[/green]")
    print(
        f"[blue]Pairwise Accuracy: {metrics['pairwise_accuracy'] * 100:.2f}% | Final Loss: {metrics['final_loss']:.4f}[/blue]"
    )


if __name__ == "__main__":
    app()
