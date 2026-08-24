from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def dpo_loss(
    policy_chosen_logps: npt.NDArray[np.floating[Any]],
    policy_rejected_logps: npt.NDArray[np.floating[Any]],
    ref_chosen_logps: npt.NDArray[np.floating[Any]],
    ref_rejected_logps: npt.NDArray[np.floating[Any]],
    beta: float,
) -> float:
    """Compute batch DPO loss from sequence log probabilities.

    Numerically stable implementation using logaddexp for log-sigmoid.
    """
    pi_logr = policy_chosen_logps - policy_rejected_logps
    ref_logr = ref_chosen_logps - ref_rejected_logps
    logits = beta * (pi_logr - ref_logr)
    # -log(sigmoid(logits)) = log(1 + exp(-logits))
    loss = np.logaddexp(0.0, -logits)
    return float(np.mean(loss))


def orpo_loss(
    sft_nll: npt.NDArray[np.floating[Any]],
    chosen_logps: npt.NDArray[np.floating[Any]],
    rejected_logps: npt.NDArray[np.floating[Any]],
    lambda_orpo: float,
) -> float:
    """Compute an ORPO-style objective: SFT loss + odds-ratio preference penalty.

    Numerically stable calculation of log odds and preference penalty.
    """
    # Clip logps to avoid log(0) when computing log(1 - exp(logp))
    chosen_clipped = np.clip(chosen_logps, -100.0, -1e-12)
    rejected_clipped = np.clip(rejected_logps, -100.0, -1e-12)

    # log odds = log p - log(1 - p) = log p - log(-expm1(log p))
    log_odds_chosen = chosen_clipped - np.log(-np.expm1(chosen_clipped))
    log_odds_rejected = rejected_clipped - np.log(-np.expm1(rejected_clipped))

    log_odds_ratio = log_odds_chosen - log_odds_rejected
    # -log(sigmoid(log_odds_ratio)) = log(1 + exp(-log_odds_ratio))
    or_loss = np.logaddexp(0.0, -log_odds_ratio)

    total_loss = sft_nll + (lambda_orpo * or_loss)
    return float(np.mean(total_loss))
