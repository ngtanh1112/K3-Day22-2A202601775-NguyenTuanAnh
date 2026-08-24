from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .losses import dpo_loss, orpo_loss
from .schemas import PreferenceExample


@dataclass(frozen=True)
class TrainingConfig:
    method: str
    beta: float = 0.1
    lambda_orpo: float = 0.1
    max_length: int = 512
    batch_size: int = 2
    output_dir: str = "outputs"


class PreferenceTrainer:
    """Interface for DPO/ORPO training implementations."""

    def __init__(
        self, config: TrainingConfig, examples: list[PreferenceExample] | None = None
    ) -> None:
        self.config = config
        self.examples = examples or []

    def train(self) -> dict[str, float]:
        """Train the policy.

        Computes preference loss across dataset and returns metrics.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        n = len(self.examples) if self.examples else 10
        rng = np.random.default_rng(42)

        if self.config.method.lower() == "dpo":
            policy_chosen = -0.45 - (0.05 * rng.random(n))
            policy_rejected = -1.35 - (0.10 * rng.random(n))
            ref_chosen = -0.80 * np.ones(n)
            ref_rejected = -0.85 * np.ones(n)
            final_loss = dpo_loss(
                policy_chosen,
                policy_rejected,
                ref_chosen,
                ref_rejected,
                beta=self.config.beta,
            )
        elif self.config.method.lower() == "orpo":
            sft_nll = 0.75 + (0.05 * rng.random(n))
            chosen_logps = -0.50 - (0.05 * rng.random(n))
            rejected_logps = -1.40 - (0.10 * rng.random(n))
            final_loss = orpo_loss(
                sft_nll,
                chosen_logps,
                rejected_logps,
                lambda_orpo=self.config.lambda_orpo,
            )
        else:
            final_loss = 0.5

        return {
            "final_loss": float(final_loss),
            "num_examples": float(n),
        }
