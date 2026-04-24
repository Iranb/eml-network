import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_history(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["epochs"]


def series_from_history(history: List[Dict], group: str, key: str) -> Tuple[List[int], List[float]]:
    epochs = [int(item["epoch"]) for item in history]
    values = [float(item[group][key]) for item in history]
    return epochs, values


def best_epoch(history: List[Dict], key: str = "acc") -> Dict:
    best = max(history, key=lambda item: float(item["eval"][key]))
    return {
        "epoch": int(best["epoch"]),
        "eval_acc": float(best["eval"]["acc"]),
        "eval_loss": float(best["eval"]["loss"]),
        "train_loss": float(best["train"]["loss"]),
    }


def ensure_dir(path: str) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def to_xy(points_x: List[int], points_y: List[float], x0: float, y0: float, width: float, height: float) -> str:
    if not points_x:
        return ""
    min_x, max_x = min(points_x), max(points_x)
    min_y, max_y = min(points_y), max(points_y)
    if max_x == min_x:
        max_x += 1
    if max_y == min_y:
        max_y += 1.0

    coords = []
    for x, y in zip(points_x, points_y):
        px = x0 + (x - min_x) / (max_x - min_x) * width
        py = y0 + height - (y - min_y) / (max_y - min_y) * height
        coords.append(f"{px:.2f},{py:.2f}")
    return " ".join(coords)


def draw_panel(
    title: str,
    lines: List[Dict[str, object]],
    x: int,
    y: int,
    width: int,
    height: int,
) -> str:
    all_x: List[int] = []
    all_y: List[float] = []
    for line in lines:
        all_x.extend(line["epochs"])  # type: ignore[arg-type]
        all_y.extend(line["values"])  # type: ignore[arg-type]

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    if max_x == min_x:
        max_x += 1
    if max_y == min_y:
        max_y += 1.0

    left_pad = 56
    right_pad = 24
    top_pad = 40
    bottom_pad = 36
    plot_x = x + left_pad
    plot_y = y + top_pad
    plot_w = width - left_pad - right_pad
    plot_h = height - top_pad - bottom_pad

    parts = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="12" fill="#ffffff" stroke="#d0d7de"/>',
        f'<text x="{x + 20}" y="{y + 24}" font-size="18" font-weight="700" fill="#0f172a">{title}</text>',
        f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="#334155"/>',
        f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="#334155"/>',
    ]

    for tick_idx in range(5):
        tick_value = min_y + (max_y - min_y) * tick_idx / 4
        tick_y = plot_y + plot_h - plot_h * tick_idx / 4
        parts.append(
            f'<line x1="{plot_x}" y1="{tick_y:.2f}" x2="{plot_x + plot_w}" y2="{tick_y:.2f}" stroke="#e2e8f0" />'
        )
        parts.append(
            f'<text x="{plot_x - 8}" y="{tick_y + 4:.2f}" text-anchor="end" font-size="11" fill="#475569">{tick_value:.4f}</text>'
        )

    for tick_idx in range(5):
        tick_value = int(round(min_x + (max_x - min_x) * tick_idx / 4))
        tick_x = plot_x + plot_w * tick_idx / 4
        parts.append(
            f'<line x1="{tick_x:.2f}" y1="{plot_y}" x2="{tick_x:.2f}" y2="{plot_y + plot_h}" stroke="#f1f5f9" />'
        )
        parts.append(
            f'<text x="{tick_x:.2f}" y="{plot_y + plot_h + 20}" text-anchor="middle" font-size="11" fill="#475569">{tick_value}</text>'
        )

    legend_x = x + 20
    legend_y = y + height - 12
    for idx, line in enumerate(lines):
        lx = legend_x + idx * 150
        ly = legend_y
        color = line["color"]  # type: ignore[index]
        label = line["label"]  # type: ignore[index]
        parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 22}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx + 28}" y="{ly + 4}" font-size="12" fill="#334155">{label}</text>')

    for line in lines:
        values = line["values"]  # type: ignore[index]
        epochs = line["epochs"]  # type: ignore[index]
        color = line["color"]  # type: ignore[index]
        points = []
        for x_val, y_val in zip(epochs, values):
            px = plot_x + (x_val - min_x) / (max_x - min_x) * plot_w
            py = plot_y + plot_h - (y_val - min_y) / (max_y - min_y) * plot_h
            points.append(f"{px:.2f},{py:.2f}")
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(points)}"/>'
        )

    return "\n".join(parts)


def build_svg(
    cnn_history: List[Dict],
    pure_history: List[Dict],
    output_path: Path,
    title: str,
    subtitle: str,
) -> None:
    cnn_epochs = [int(item["epoch"]) for item in cnn_history]
    pure_epochs = [int(item["epoch"]) for item in pure_history]
    _, cnn_train_loss = series_from_history(cnn_history, "train", "loss")
    _, pure_train_loss = series_from_history(pure_history, "train", "loss")
    _, cnn_eval_acc = series_from_history(cnn_history, "eval", "acc")
    _, pure_eval_acc = series_from_history(pure_history, "eval", "acc")

    width = 1200
    height = 720
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f8fafc"/>
<text x="40" y="46" font-size="28" font-weight="700" fill="#0f172a">{title}</text>
<text x="40" y="74" font-size="14" fill="#475569">{subtitle}</text>
{draw_panel("Train Loss", [
    {{"label": "cnn_eml", "color": "#2563eb", "epochs": cnn_epochs, "values": cnn_train_loss}},
    {{"label": "pure_eml", "color": "#dc2626", "epochs": pure_epochs, "values": pure_train_loss}},
], 36, 100, 1128, 260)}
{draw_panel("Test Accuracy", [
    {{"label": "cnn_eml", "color": "#2563eb", "epochs": cnn_epochs, "values": cnn_eval_acc}},
    {{"label": "pure_eml", "color": "#dc2626", "epochs": pure_epochs, "values": pure_eval_acc}},
], 36, 390, 1128, 260)}
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def write_csv(cnn_history: List[Dict], pure_history: List[Dict], output_path: Path) -> None:
    cnn_by_epoch = {int(item["epoch"]): item for item in cnn_history}
    pure_by_epoch = {int(item["epoch"]): item for item in pure_history}
    all_epochs = sorted(set(cnn_by_epoch) | set(pure_by_epoch))

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "cnn_train_loss",
                "cnn_eval_loss",
                "cnn_eval_acc",
                "pure_train_loss",
                "pure_eval_loss",
                "pure_eval_acc",
            ]
        )
        for epoch in all_epochs:
            cnn_item = cnn_by_epoch.get(epoch)
            pure_item = pure_by_epoch.get(epoch)
            writer.writerow(
                [
                    epoch,
                    cnn_item["train"]["loss"] if cnn_item is not None else "",
                    cnn_item["eval"]["loss"] if cnn_item is not None else "",
                    cnn_item["eval"]["acc"] if cnn_item is not None else "",
                    pure_item["train"]["loss"] if pure_item is not None else "",
                    pure_item["eval"]["loss"] if pure_item is not None else "",
                    pure_item["eval"]["acc"] if pure_item is not None else "",
                ]
            )


def write_summary(cnn_history: List[Dict], pure_history: List[Dict], output_path: Path) -> None:
    payload = {
        "cnn_eml": {
            "epochs": len(cnn_history),
            "best": best_epoch(cnn_history),
            "final": {
                "epoch": int(cnn_history[-1]["epoch"]),
                "train_loss": float(cnn_history[-1]["train"]["loss"]),
                "eval_loss": float(cnn_history[-1]["eval"]["loss"]),
                "eval_acc": float(cnn_history[-1]["eval"]["acc"]),
            },
        },
        "pure_eml": {
            "epochs": len(pure_history),
            "best": best_epoch(pure_history),
            "final": {
                "epoch": int(pure_history[-1]["epoch"]),
                "train_loss": float(pure_history[-1]["train"]["loss"]),
                "eval_loss": float(pure_history[-1]["eval"]["loss"]),
                "eval_acc": float(pure_history[-1]["eval"]["acc"]),
            },
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two EML experiment histories and render SVG plots")
    parser.add_argument("--cnn-history", required=True)
    parser.add_argument("--pure-history", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="EML Experiment Comparison")
    parser.add_argument("--subtitle", default="cnn_eml vs pure_eml")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    cnn_history = load_history(args.cnn_history)
    pure_history = load_history(args.pure_history)

    build_svg(
        cnn_history,
        pure_history,
        output_dir / "comparison.svg",
        title=args.title,
        subtitle=args.subtitle,
    )
    write_csv(cnn_history, pure_history, output_dir / "comparison_metrics.csv")
    write_summary(cnn_history, pure_history, output_dir / "comparison_summary.json")


if __name__ == "__main__":
    main()
