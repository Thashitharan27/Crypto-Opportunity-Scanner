"""Stable application paths independent of the current working directory."""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
MARKET_DATA_ROOT = Path(r"C:\CryptoBots\Binance Market Data")
# Transitional alias used by the existing GUI composition.  It now denotes the
# Data Lake root rather than the retired combined-CSV directory.
DATA_DIR = MARKET_DATA_ROOT
CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR = PROJECT_ROOT / "cache"
