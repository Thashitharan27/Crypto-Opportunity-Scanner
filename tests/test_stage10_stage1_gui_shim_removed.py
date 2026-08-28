from datetime import datetime, timezone
from pathlib import Path

from crypto_strategy_lab.data import MarketKind
from crypto_strategy_lab.gui.v2_controller import CatalogStatusService, GuiResearchRequest


def test_stage1_gui_shim_is_removed_and_app_uses_v2_main_window():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "crypto_strategy_lab" / "gui" / "stage1_window.py").exists()
    app_text = (root / "app.py").read_text(encoding="utf-8")
    assert "gui.v2_main_window import MainWindow" in app_text
    assert "gui.data_lake_main_window import MainWindow" not in app_text
    assert "stage1_window" not in app_text


def test_base_main_window_owns_single_di_direction_group():
    root = Path(__file__).resolve().parents[1]
    text = (root / "crypto_strategy_lab" / "gui" / "main_window.py").read_text(encoding="utf-8")
    assert 'QGroupBox("DI Direction Selection")' in text
    assert 'selection_box=QGroupBox("DI Direction Selection")' not in text
    assert 'QGroupBox("Direction Selection")' not in text


def test_safety_tests_no_longer_import_stage1_window():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "tests" / "test_di_tab_stage1.py").exists()
    text = (root / "tests" / "test_stage2_regime_permissions_removed.py").read_text(encoding="utf-8")
    assert "gui.main_window import MainWindow" in text
    assert "stage1_window" not in text


def test_gui_boundary_normalizes_qt_string_market_values():
    seen = []

    class Catalog:
        def inventory(self, _raw_root, *, market):
            seen.append(market)
            return []

    store = type("Store", (), {"raw_root": Path("raw"), "catalog": Catalog()})()
    service = CatalogStatusService(store)
    request = GuiResearchRequest(
        "binance",
        "futures_um",
        "BTCUSDT",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 1, 2, tzinfo=timezone.utc),
        "4h",
        "1m",
    )

    service.coverage(request)
    native = request.to_data_request()

    assert seen == [MarketKind.FUTURES_UM]
    assert native.market is MarketKind.FUTURES_UM
