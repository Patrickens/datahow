"""Neural Controlled Differential Equation (diffrax) for titer prediction.

Placeholder: implemented after the preprocessing check-in. Will consume the
padded sequence tensors from :mod:`titer_prediction.data_preprocessing`, build a
neural CDE with diffrax/equinox, train it with optax, and expose train/predict
via a CLI.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - stub
    raise NotImplementedError("cde.py is not implemented yet; see the preprocessing check-in.")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
