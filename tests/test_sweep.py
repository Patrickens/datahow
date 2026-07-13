"""Lightweight sweep tests that avoid expensive model training."""

from __future__ import annotations

import pytest

from titer_prediction import sweep


def test_sample_cde_configs_is_deterministic_and_seeded():
    configs = sweep.sample_cde_configs()
    assert configs == sweep.sample_cde_configs()  # deterministic under the fixed seed
    assert len(configs) == sweep.CDE_N_CONFIGS
    # A different seed samples a different set.
    assert sweep.sample_cde_configs(seed=sweep.SEED + 1) != configs
    # Configs are pure hyperparameters now — no per-config seed key.
    assert all("lr" in cfg and "hidden_size" in cfg and "batch_size" in cfg for cfg in configs)
    assert all("model_seed" not in cfg and "seed" not in cfg for cfg in configs)


def test_sample_cde_configs_rejects_too_many_configs():
    with pytest.raises(ValueError, match="Requested"):
        sweep.sample_cde_configs(n_configs=10_000)


def test_sample_xgb_configs_is_deterministic_and_seeded():
    configs = sweep.sample_xgb_configs()
    assert configs == sweep.sample_xgb_configs()
    assert len(configs) == sweep.XGB_N_CONFIGS
    assert sweep.sample_xgb_configs(seed=sweep.SEED + 1) != configs
    assert all("estimator_seed" not in cfg and "seed" not in cfg for cfg in configs)
    assert all(
        "max_depth" in cfg and "learning_rate" in cfg and "n_estimators" in cfg for cfg in configs
    )
