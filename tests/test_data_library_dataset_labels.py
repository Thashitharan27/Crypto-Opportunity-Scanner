from crypto_strategy_lab.data import DatasetKind
from crypto_strategy_lab.gui.dataset_labels import dataset_family_label


def test_candle_sources_keep_distinct_data_library_labels():
    assert dataset_family_label(DatasetKind.KLINES) == "Market Price Candles"
    assert dataset_family_label(DatasetKind.MARK_PRICE_KLINES) == "Mark Price Candles"
    assert dataset_family_label(DatasetKind.INDEX_PRICE_KLINES) == "Index Price Candles"
    assert dataset_family_label(DatasetKind.PREMIUM_INDEX_KLINES) == "Premium Index Candles"
    assert len({
        dataset_family_label(DatasetKind.KLINES),
        dataset_family_label(DatasetKind.MARK_PRICE_KLINES),
        dataset_family_label(DatasetKind.INDEX_PRICE_KLINES),
        dataset_family_label(DatasetKind.PREMIUM_INDEX_KLINES),
    }) == 4


def test_non_candle_dataset_family_labels_remain_friendly():
    assert dataset_family_label(DatasetKind.FUNDING_RATE) == "Funding"
    assert dataset_family_label(DatasetKind.FUTURES_METRICS) == "Futures Positioning"
    assert dataset_family_label(DatasetKind.AGG_TRADES) == "Trades"
    assert dataset_family_label(DatasetKind.BOOK_DEPTH) == "Order Book"


def test_active_app_window_uses_distinct_dataset_family_labels():
    from app import MainWindow

    assert MainWindow._dataset_family(DatasetKind.KLINES) == "Market Price Candles"
    assert MainWindow._dataset_family(DatasetKind.MARK_PRICE_KLINES) == "Mark Price Candles"
