from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eml_mnist.reporting import generate_validation_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the EML validation report from run artifacts")
    parser.add_argument("--runs-root", default="reports/runs")
    parser.add_argument("--output", default="reports/EML_VALIDATION_REPORT.md")
    args = parser.parse_args()
    output = generate_validation_report(args.runs_root, args.output)
    print(output)


if __name__ == "__main__":
    main()
