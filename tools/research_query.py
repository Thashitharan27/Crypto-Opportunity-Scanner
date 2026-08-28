"""Run an artifact-only Task-16 descriptive research query."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crypto_strategy_lab.feature_research import ResearchQueryService


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--spec", required=True, type=Path, help="JSON query specification"
    )
    parser.add_argument(
        "--output", type=Path, help="optional CSV output (JSON is printed otherwise)"
    )
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    with ResearchQueryService(args.run_dir) as service:
        result = service.query(spec)
        query_seconds = service.last_query_seconds

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
    else:
        print(result.to_json(orient="records", date_format="iso", indent=2))
    print(
        json.dumps(
            {
                "query_seconds": query_seconds,
                "rows": len(result),
                "source": "completed_run_artifacts_only",
            }
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
