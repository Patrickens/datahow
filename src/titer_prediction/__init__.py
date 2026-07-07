"""Titer prediction for a simulated mAb bioprocess (DataHow ML challenge).

The package is organised into three CLI-driven modules:

* :mod:`titer_prediction.data_preprocessing` — turn the raw long-format CSVs into
  per-experiment features (for the baseline) and padded sequences (for the CDE).
* :mod:`titer_prediction.regression` — a gradient-boosted (XGBoost) baseline.
* :mod:`titer_prediction.cde` — a neural controlled differential equation
  (diffrax).
"""

__version__ = "0.1.0"
