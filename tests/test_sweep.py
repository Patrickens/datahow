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
