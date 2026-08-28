"""Rewrite a local Crypto Strategy Lab JSON config to the current strict schema.

This is a one-time migration utility, not a compatibility loader. It keeps only
settings that are part of the current GUI/BacktestConfig contract, drops retired
keys, normalizes current Strategy Profiles, and writes a clean version-2 JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crypto_strategy_lab.gui.config_logic import CONFIG_VERSION, DEFAULT_GUI_CONFIG
from crypto_strategy_lab.strategy_profiles import normalize_profiles, profiles_to_dict


def rewrite_config(values: dict) -> tuple[dict, tuple[str, ...]]:
    if not isinstance(values, dict):
        raise ValueError("Configuration JSON must contain an object")

    allowed = set(DEFAULT_GUI_CONFIG)
    retired = tuple(sorted(set(values) - allowed))
    cleaned = {key: value for key, value in values.items() if key in allowed}
    cleaned["config_version"] = CONFIG_VERSION

    if "strategy_profiles" in cleaned:
        cleaned["strategy_profiles"] = profiles_to_dict(normalize_profiles(cleaned["strategy_profiles"]))

    # Fill omitted current fields explicitly so the rewritten file is a complete,
    # self-contained snapshot of today's contract rather than another partial legacy file.
    current = dict(DEFAULT_GUI_CONFIG)
    current.update(cleaned)
    current["config_version"] = CONFIG_VERSION
    current["strategy_profiles"] = profiles_to_dict(normalize_profiles(current["strategy_profiles"]))
    return current, retired


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rewrite a Crypto Strategy Lab config to the current strict schema")
    parser.add_argument("config", type=Path, help="Existing JSON configuration")
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination JSON. Defaults to <name>.current.json next to the source.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Replace the source file instead of writing <name>.current.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.in_place and args.output is not None:
        raise ValueError("Use either --in-place or --output, not both")

    source = args.config.resolve()
    values = json.loads(source.read_text(encoding="utf-8-sig"))
    current, retired = rewrite_config(values)

    if args.in_place:
        destination = source
    elif args.output is not None:
        destination = args.output.resolve()
    else:
        destination = source.with_name(f"{source.stem}.current.json")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote current config: {destination}")
    if retired:
        print("Removed retired top-level fields: " + ", ".join(retired))
    else:
        print("No retired top-level fields were present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
