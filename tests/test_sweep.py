"""Lightweight sweep tests that avoid expensive model training."""

from __future__ import annotations

import pytest

from titer_prediction import sweep


def test_sample_cde_configs_is_deterministic_and_seeded():
    configs = sweep.sample_cde_configs()
    assert configs == sweep.sample_cde_configs()
    assert len(configs) == sweep.CDE_N_CONFIGS

    model_seeds = [cfg["model_seed"] for cfg in configs]
    assert model_seeds == list(
        range(
            sweep.CDE_MODEL_SEED_BASE + 1,
            sweep.CDE_MODEL_SEED_BASE + sweep.CDE_N_CONFIGS + 1,
        )
    )
    assert all("epochs" in cfg and "lr" in cfg and "hidden_size" in cfg for cfg in configs)


def test_sample_cde_configs_rejects_too_many_configs():
    with pytest.raises(ValueError, match="Requested"):
        sweep.sample_cde_configs(n_configs=10_000)


def test_sample_xgb_configs_is_deterministic_and_seeded():
    configs = sweep.sample_xgb_configs()
    assert configs == sweep.sample_xgb_configs()
    assert len(configs) == sweep.XGB_N_CONFIGS

    estimator_seeds = [cfg["estimator_seed"] for cfg in configs]
    assert estimator_seeds == list(
        range(
            sweep.XGB_ESTIMATOR_SEED_BASE + 1,
            sweep.XGB_ESTIMATOR_SEED_BASE + sweep.XGB_N_CONFIGS + 1,
        )
    )
    assert all(
        "max_depth" in cfg and "learning_rate" in cfg and "n_estimators" in cfg
        for cfg in configs
    )
