from __future__ import annotations

import json
from pathlib import Path

import run_eml_uncertainty_benchmark as benchmark


def main() -> None:
    parser = benchmark.build_parser()
    parser.description = "Run EML uncertainty/resistance benchmark"
    parser.set_defaults(
        runs_root="reports/uncertainty_resistance_benchmark/runs",
        report="reports/EML_UNCERTAINTY_RESISTANCE_REPORT.md",
        dataset="synthetic_shape",
    )
    args = parser.parse_args()
    report_path = benchmark.run(args)
    print(json.dumps({"report": str(Path(report_path))}, sort_keys=True))


if __name__ == "__main__":
    main()
