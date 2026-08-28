from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from crypto_strategy_lab.data import MarketKind
from crypto_strategy_lab.data_lake_config import ResearchRunConfig
from crypto_strategy_lab.gui.run_progress import format_duration
from crypto_strategy_lab.gui.v2_controller import GuiApplicationService, GuiResearchRequest
from crypto_strategy_lab.progress import emit_progress


def test_emit_progress_is_best_effort_and_never_breaks_work():
    seen = []
    emit_progress(seen.append, kind="stage", label="Preparing")
    assert seen == [{"kind": "stage", "label": "Preparing"}]

    def broken(_event):
        raise RuntimeError("observer failure")

    # UI/observer failures are deliberately swallowed so progress cannot alter
    # research execution semantics.
    emit_progress(broken, kind="stage", label="Still safe")


def test_progress_eta_formatter_is_compact():
    assert format_duration(8) == "8s"
    assert format_duration(125) == "2m 05s"
    assert format_duration(3720) == "1h 02m"


def test_gui_service_attaches_progress_only_for_active_run(tmp_path):
    class Store:
        progress_callback = None

    store = Store()
    observed = []
    callback = observed.append

    class Runner:
        def run(self, request, config):
            assert store.progress_callback is callback
            emit_progress(
                store.progress_callback,
                kind="stage",
                phase="simulation",
                label="Running strategy simulation",
            )
            return "result"

    service = object.__new__(GuiApplicationService)
    service.store = store
    service.progress_callback = callback
    service._runner_factory = lambda _output: Runner()

    request = GuiResearchRequest(
        "binance",
        MarketKind.FUTURES_UM,
        "BTCUSDT",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 1, 2, tzinfo=timezone.utc),
        "4h",
        "1m",
    )
    base = ResearchRunConfig()
    config = replace(
        base,
        reporting=replace(base.reporting, output_dir=str(Path(tmp_path) / "runs")),
        data=replace(
            base.data,
            strategy_timeframe_minutes=240,
            intrabar_timeframe_minutes=1,
            use_intrabar_data=True,
        ),
    )

    assert service.run(request, config) == "result"
    assert observed == [
        {
            "kind": "stage",
            "phase": "simulation",
            "label": "Running strategy simulation",
        }
    ]
    assert store.progress_callback is None
