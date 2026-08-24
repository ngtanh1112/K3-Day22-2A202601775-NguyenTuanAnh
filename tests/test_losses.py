import math

import numpy as np

from preference_lab.losses import dpo_loss, orpo_loss


def test_dpo_loss_zero_margin() -> None:
    # When policy and ref have identical margins, logits = 0, loss = ln(2)
    p_c = np.array([-0.5])
    p_r = np.array([-1.5])
    r_c = np.array([-0.5])
    r_r = np.array([-1.5])
    loss = dpo_loss(p_c, p_r, r_c, r_r, beta=0.1)
    assert math.isclose(loss, math.log(2.0), rel_tol=1e-5)


def test_dpo_loss_improves_with_higher_preference() -> None:
    # Policy with stronger preference for chosen than ref has lower loss
    loss_equal = dpo_loss(
        np.array([-1.0]), np.array([-1.0]), np.array([-1.0]), np.array([-1.0]), beta=0.1
    )
    loss_better = dpo_loss(
        np.array([-0.1]), np.array([-2.0]), np.array([-1.0]), np.array([-1.0]), beta=0.1
    )
    assert loss_better < loss_equal


def test_dpo_loss_numerical_stability() -> None:
    # Large differences shouldn't yield NaN or Inf
    large_diff_c = np.array([0.0, -100.0])
    large_diff_r = np.array([-100.0, 0.0])
    ref = np.array([-1.0, -1.0])
    loss = dpo_loss(large_diff_c, large_diff_r, ref, ref, beta=1.0)
    assert np.isfinite(loss)
    assert loss > 0.0


def test_orpo_loss_calculation() -> None:
    sft_nll = np.array([1.0])
    c_logp = np.array([-0.5])
    r_logp = np.array([-1.5])
    loss = orpo_loss(sft_nll, c_logp, r_logp, lambda_orpo=0.1)
    assert np.isfinite(loss)
    assert loss > 1.0  # SFT loss (1.0) + positive penalty


def test_orpo_loss_numerical_stability() -> None:
    sft_nll = np.array([0.5, 1.5])
    c_logp = np.array([-1e-10, -50.0])
    r_logp = np.array([-50.0, -1e-10])
    loss = orpo_loss(sft_nll, c_logp, r_logp, lambda_orpo=0.5)
    assert np.isfinite(loss)
    assert loss > 0.0
