"""Causal joins used by all cross-dataset research features."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def causal_asof_join(
    decisions: pd.DataFrame,
    features: pd.DataFrame,
    *,
    decision_time_col: str = "decision_time",
    available_at_col: str = "available_at",
    by: str | Sequence[str] | None = None,
    suffix: str = "_feature",
) -> pd.DataFrame:
    """Attach the latest feature that was actually available at each decision.

    The right-hand timestamp is always `available_at`, not an observation's
    nominal period/open time. Exact matches are allowed because data available
    at decision time T is causal for T.
    """

    if decision_time_col not in decisions.columns:
        raise KeyError(f"Missing decision timestamp column: {decision_time_col}")
    if available_at_col not in features.columns:
        raise KeyError(f"Missing feature availability column: {available_at_col}")

    by_columns: list[str]
    if by is None:
        by_columns = []
    elif isinstance(by, str):
        by_columns = [by]
    else:
        by_columns = list(by)
    for column in by_columns:
        if column not in decisions.columns or column not in features.columns:
            raise KeyError(f"Causal join key must exist on both frames: {column}")

    left = decisions.copy()
    right = features.copy()
    left[decision_time_col] = pd.to_datetime(left[decision_time_col], utc=True, errors="raise")
    right[available_at_col] = pd.to_datetime(right[available_at_col], utc=True, errors="raise")

    # merge_asof requires the `on` keys to be globally monotonic. Group keys are
    # secondary so interleaved symbols still obey that requirement.
    left_sort = [decision_time_col, *by_columns]
    right_sort = [available_at_col, *by_columns]
    left = left.sort_values(left_sort, kind="stable")
    right = right.sort_values(right_sort, kind="stable")

    joined = pd.merge_asof(
        left,
        right,
        left_on=decision_time_col,
        right_on=available_at_col,
        by=by_columns or None,
        direction="backward",
        allow_exact_matches=True,
        suffixes=("", suffix),
    )
    available = pd.to_datetime(joined[available_at_col], utc=True, errors="coerce")
    decisions_at = pd.to_datetime(joined[decision_time_col], utc=True, errors="raise")
    leak = available.notna() & (available > decisions_at)
    if bool(leak.any()):
        raise AssertionError("Causal join produced a future feature value")
    return joined.sort_index(kind="stable")


def assert_causal_availability(
    frame: pd.DataFrame,
    *,
    decision_time_col: str = "decision_time",
    available_at_col: str = "available_at",
) -> None:
    """Fail if any attached value became available after its decision time."""

    decision = pd.to_datetime(frame[decision_time_col], utc=True, errors="raise")
    available = pd.to_datetime(frame[available_at_col], utc=True, errors="coerce")
    leak = available.notna() & (available > decision)
    if bool(leak.any()):
        examples = frame.loc[leak, [decision_time_col, available_at_col]].head(5)
        raise AssertionError(f"Future data detected:\n{examples.to_string(index=False)}")
