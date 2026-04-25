from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate MERC master report")
    parser.add_argument("--output", default="reports/MERC_MASTER_REPORT.md")
    parser.add_argument("--toy-report", default="reports/MERC_TOY_REPORT.md")
    parser.add_argument("--head-report", default="reports/MERC_HEAD_ABLATION_REPORT.md")
    parser.add_argument("--end-to-end-report", default="reports/MERC_END_TO_END_REPORT.md")
    parser.add_argument("--synthetic-report", default="reports/MERC_SYNTHETIC_EVIDENCE_REPORT.md")
    parser.add_argument("--server-report", default="reports/MERC_REAL_SERVER_VALIDATION_REPORT.md")
    return parser


def _section(title: str, path: Path) -> list[str]:
    lines = [f"## {title}", ""]
    if path.exists():
        lines.append(f"Source: `{path}`")
        lines.append("")
        lines.extend(path.read_text(encoding="utf-8").splitlines()[:80])
    else:
        lines.append("MISSING")
    lines.append("")
    return lines


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output)
    lines = [
        "# MERC Master Report",
        "",
        "This report only aggregates generated MERC artifacts. Missing files remain missing.",
        "",
        "## Claim Status",
        "",
        "- A. Does MERC beat linear? MISSING until the reports below show it.",
        "- B. Does MERC beat MLP? MISSING until the reports below show it.",
        "- C. Does MERC beat cosine prototype? MISSING until the reports below show it.",
        "- D. Does MERC beat old EML head? MISSING until the reports below show it.",
        "- E. Does MERC show support-factor alignment? MISSING until toy/synthetic evidence reports show it.",
        "- F. Does MERC show conflict/resistance alignment? MISSING until toy/synthetic evidence reports show it.",
        "- G. Is MERC worth using as a head? Inconclusive until the real-server report exists.",
        "- H. Is MERC worth exploring as a representation block? Inconclusive; this task only validates the head/hypothesis cell path.",
        "",
    ]
    lines.extend(_section("Toy Nonlinear Tasks", Path(args.toy_report)))
    lines.extend(_section("Frozen CNN Feature Head Isolation", Path(args.head_report)))
    lines.extend(_section("End-to-End CNN Plus Head", Path(args.end_to_end_report)))
    lines.extend(_section("Synthetic Evidence Diagnostics", Path(args.synthetic_report)))
    lines.extend(_section("CIFAR-10 Real Server Results", Path(args.server_report)))
    output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
