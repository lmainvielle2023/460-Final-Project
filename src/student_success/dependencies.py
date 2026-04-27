"""Helpers for optional runtime dependencies."""

from __future__ import annotations


def require_scikit_learn() -> None:
    """Raise a clear error if scikit-learn is unavailable."""
    try:
        import sklearn  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "scikit-learn is required for model training. Install project "
            "dependencies with `pip install -e .`."
        ) from exc
