from __future__ import annotations

import cProfile

from tools.data_lake_profile import _profile_rows


def _busy_work() -> int:
    total = 0
    for value in range(200):
        total += value * value
    return total


def test_profile_rows_rank_and_shape_profile_functions() -> None:
    profile = cProfile.Profile()
    profile.enable()
    for _ in range(10):
        _busy_work()
    profile.disable()

    cumulative = _profile_rows(profile, "cumulative", 5)
    self_ranked = _profile_rows(profile, "self", 5)

    assert cumulative
    assert self_ranked
    assert len(cumulative) <= 5
    assert len(self_ranked) <= 5
    assert all("function" in row for row in cumulative)
    assert all("cumulative_seconds" in row for row in cumulative)
    assert all("self_seconds" in row for row in self_ranked)
    assert [row["cumulative_seconds"] for row in cumulative] == sorted(
        [row["cumulative_seconds"] for row in cumulative], reverse=True
    )
    assert [row["self_seconds"] for row in self_ranked] == sorted(
        [row["self_seconds"] for row in self_ranked], reverse=True
    )
    assert any(row["function"] == "_busy_work" for row in cumulative)
