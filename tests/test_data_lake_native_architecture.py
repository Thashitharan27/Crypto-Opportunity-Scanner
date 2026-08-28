"""Guard the canonical Data Lake path against retired CSV/bridge dependencies."""

from pathlib import Path


NATIVE_MODULES = (
    "crypto_strategy_lab/data/backtest_service.py",
    "crypto_strategy_lab/prepared_backtest.py",
    "crypto_strategy_lab/prepared_cache.py",
    "crypto_strategy_lab/data_lake_production_engine.py",
    "tools/data_lake_run.py",
    "tools/data_lake_benchmark.py",
    "tools/data_lake_profile.py",
    "tools/data_lake_preparation_profile.py",
    "crypto_strategy_lab/gui/data_lake_worker.py",
)
FORBIDDEN = (
    "crypto_strategy_lab.data.legacy_bridge",
    "legacy_bridge",
    "canonical_to_legacy_ohlcv",
    "_legacy_from_canonical",
    "legacy_conversion",
    "load_ohlcv_csv",
    "load_backtest_data",
    "input_csv",
    "intrabar_csv",
)


def test_native_modules_do_not_reference_retired_loading_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    violations = {
        relative: token
        for relative in NATIVE_MODULES
        for token in FORBIDDEN
        if token in (root / relative).read_text(encoding="utf-8")
    }
    assert not violations


def test_retired_migration_modules_are_deleted() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not (root / "crypto_strategy_lab/data/legacy_bridge.py").exists()
    assert not (root / "tools/data_lake_backtest_parity.py").exists()
    assert not (root / "tools/combine_binance_data.py").exists()
