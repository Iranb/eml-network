from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot MERC comparison figure from pulled result CSVs")
    parser.add_argument(
        "--frozen-summary",
        default="reports/remote_merc_validation_20260425_105808/summaries/merc_head_ablation_summary.csv",
    )
    parser.add_argument(
        "--e2e-summary",
        default="reports/remote_merc_validation_20260425_105808/summaries/merc_end_to_end_summary.csv",
    )
    parser.add_argument(
        "--synthetic-summary",
        default="reports/remote_merc_validation_20260425_105808/summaries/merc_synthetic_evidence_summary.csv",
    )
    parser.add_argument("--output", default="reports/MERC_COMPARISON_FIGURE.png")
    return parser


def _load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            metrics_raw = row.get("metrics_json", "")
            try:
                metrics = json.loads(metrics_raw) if metrics_raw else {}
            except json.JSONDecodeError:
                metrics = {}
            row["metrics"] = metrics
            rows.append(row)
    return rows


def _to_float(value: object) -> float:
    try:
        if value in ("", None):
            return float("nan")
        number = float(value)  # type: ignore[arg-type]
        return number
    except Exception:
        return float("nan")


def _group_metric(
    rows: list[dict[str, object]],
    models: list[str],
    metric_key: str,
    *,
    status: str = "COMPLETED",
    loss_mode: str | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    means: list[float] = []
    stds: list[float] = []
    labels: list[str] = []
    for model in models:
        values: list[float] = []
        for row in rows:
            if row.get("status") != status or row.get("model_name") != model:
                continue
            if loss_mode is not None and row.get("loss_mode", "") != loss_mode:
                continue
            metrics = row.get("metrics")
            if not isinstance(metrics, dict):
                continue
            value = _to_float(metrics.get(metric_key))
            if not math.isnan(value):
                values.append(value)
        if values:
            labels.append(model)
            arr = np.asarray(values, dtype=float)
            means.append(float(arr.mean()))
            stds.append(float(arr.std(ddof=0)))
    return labels, np.asarray(means, dtype=float), np.asarray(stds, dtype=float)


def _group_dual_metric(
    rows: list[dict[str, object]],
    models: list[str],
    metric_a: str,
    metric_b: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    labels: list[str] = []
    a_vals: list[float] = []
    b_vals: list[float] = []
    for model in models:
        vals_a: list[float] = []
        vals_b: list[float] = []
        for row in rows:
            if row.get("status") != "COMPLETED" or row.get("model_name") != model:
                continue
            metrics = row.get("metrics")
            if not isinstance(metrics, dict):
                continue
            a = _to_float(metrics.get(metric_a))
            b = _to_float(metrics.get(metric_b))
            if not math.isnan(a):
                vals_a.append(a)
            if not math.isnan(b):
                vals_b.append(b)
        if vals_a or vals_b:
            labels.append(model)
            a_vals.append(float(np.mean(vals_a)) if vals_a else float("nan"))
            b_vals.append(float(np.mean(vals_b)) if vals_b else float("nan"))
    return labels, np.asarray(a_vals, dtype=float), np.asarray(b_vals, dtype=float)


def _pretty(name: str) -> str:
    mapping = {
        "cosine_prototype": "Cosine",
        "eml_centered_ambiguity": "Old EML centered",
        "eml_no_ambiguity": "Old EML no-amb",
        "merc_linear": "MERC linear",
        "merc_energy": "MERC energy",
        "merc_block_linear": "MERC block+linear",
        "merc_block_energy": "MERC block+energy",
        "merc_linear_small": "MERC linear small",
        "merc_energy_small": "MERC energy small",
        "mlp": "MLP",
        "linear": "Linear",
    }
    return mapping.get(name, name.replace("_", " "))


def _color(name: str) -> str:
    if name.startswith("merc"):
        return "#d95f02"
    if name.startswith("eml"):
        return "#7570b3"
    return "#1b9e77"


def _load_font(size: int) -> ImageFont.ImageFont:
    for candidate in [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _draw_rotated_text(
    image: Image.Image,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    *,
    angle: int = 0,
    fill: str = "black",
    anchor: str = "lt",
) -> None:
    dummy = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    dummy_draw = ImageDraw.Draw(dummy)
    width, height = _text_size(dummy_draw, text, font)
    text_image = Image.new("RGBA", (width + 4, height + 4), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_image)
    text_draw.text((2, 2), text, font=font, fill=fill)
    rotated = text_image.rotate(angle, expand=True)
    x, y = xy
    if anchor == "mm":
        x -= rotated.size[0] // 2
        y -= rotated.size[1] // 2
    elif anchor == "mt":
        x -= rotated.size[0] // 2
    elif anchor == "rm":
        x -= rotated.size[0]
        y -= rotated.size[1] // 2
    image.alpha_composite(rotated, (x, y))


def _draw_panel_frame(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, title_font: ImageFont.ImageFont) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=10, outline="#cfcfcf", width=2, fill="white")
    draw.text((x0 + 16, y0 + 12), title, fill="black", font=title_font)


def _draw_vertical_bar_chart(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    labels: list[str],
    values: np.ndarray,
    errors: np.ndarray,
    *,
    y_label: str,
    y_min: float,
    y_max: float,
    title: str,
) -> None:
    title_font = _load_font(24)
    label_font = _load_font(16)
    tick_font = _load_font(14)
    _draw_panel_frame(draw, box, title, title_font)
    x0, y0, x1, y1 = box
    plot_left = x0 + 70
    plot_right = x1 - 20
    plot_top = y0 + 60
    plot_bottom = y1 - 110
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="black", width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="black", width=2)
    for frac in np.linspace(0.0, 1.0, 5):
        value = y_min + (y_max - y_min) * (1.0 - frac)
        y = int(plot_top + frac * (plot_bottom - plot_top))
        draw.line((plot_left, y, plot_right, y), fill="#e8e8e8", width=1)
        text = f"{value:.2f}"
        tw, th = _text_size(draw, text, tick_font)
        draw.text((plot_left - tw - 8, y - th // 2), text, fill="#555555", font=tick_font)
    _draw_rotated_text(image, (x0 + 20, (plot_top + plot_bottom) // 2), y_label, label_font, angle=90, anchor="mm")
    n = max(len(labels), 1)
    span = plot_right - plot_left
    bar_width = max(18, int(span / (n * 1.8)))
    centers = [int(plot_left + span * (i + 0.5) / n) for i in range(n)]
    for center, label, value, err, raw_name in zip(centers, labels, values, errors, labels):
        usable = max(min(value, y_max), y_min)
        top = int(plot_bottom - (usable - y_min) / max(y_max - y_min, 1e-6) * (plot_bottom - plot_top))
        left = center - bar_width // 2
        right = center + bar_width // 2
        draw.rounded_rectangle((left, top, right, plot_bottom), radius=4, fill=_color(raw_name), outline=None)
        if not math.isnan(err) and err > 0:
            err_top_val = max(y_min, min(y_max, value + err))
            err_bottom_val = max(y_min, min(y_max, value - err))
            err_top = int(plot_bottom - (err_top_val - y_min) / max(y_max - y_min, 1e-6) * (plot_bottom - plot_top))
            err_bottom = int(plot_bottom - (err_bottom_val - y_min) / max(y_max - y_min, 1e-6) * (plot_bottom - plot_top))
            draw.line((center, err_top, center, err_bottom), fill="#333333", width=2)
            draw.line((center - 6, err_top, center + 6, err_top), fill="#333333", width=2)
            draw.line((center - 6, err_bottom, center + 6, err_bottom), fill="#333333", width=2)
        value_text = f"{value:.3f}"
        tw, th = _text_size(draw, value_text, tick_font)
        draw.text((center - tw // 2, top - th - 4), value_text, fill="#222222", font=tick_font)
        _draw_rotated_text(image, (center, plot_bottom + 20), _pretty(label), tick_font, angle=35, anchor="mt")


def _draw_horizontal_bar_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    labels: list[str],
    values: np.ndarray,
    errors: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    title: str,
) -> None:
    title_font = _load_font(24)
    label_font = _load_font(16)
    tick_font = _load_font(14)
    _draw_panel_frame(draw, box, title, title_font)
    x0, y0, x1, y1 = box
    plot_left = x0 + 165
    plot_right = x1 - 30
    plot_top = y0 + 70
    plot_bottom = y1 - 30
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="black", width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="black", width=2)
    for frac in np.linspace(0.0, 1.0, 5):
        value = x_min + frac * (x_max - x_min)
        x = int(plot_left + frac * (plot_right - plot_left))
        draw.line((x, plot_top, x, plot_bottom), fill="#efefef", width=1)
        text = f"{value:.2f}"
        tw, th = _text_size(draw, text, tick_font)
        draw.text((x - tw // 2, plot_bottom + 6), text, fill="#555555", font=tick_font)
    n = max(len(labels), 1)
    row_h = (plot_bottom - plot_top) / n
    bar_h = max(16, int(row_h * 0.55))
    ordered = sorted(zip(labels, values, errors), key=lambda item: item[1])
    for idx, (label, value, err) in enumerate(ordered):
        cy = int(plot_top + row_h * (idx + 0.5))
        width = int((max(min(value, x_max), x_min) - x_min) / max(x_max - x_min, 1e-6) * (plot_right - plot_left))
        draw.rounded_rectangle((plot_left, cy - bar_h // 2, plot_left + width, cy + bar_h // 2), radius=4, fill=_color(label))
        if not math.isnan(err) and err > 0:
            err_left_val = max(x_min, min(x_max, value - err))
            err_right_val = max(x_min, min(x_max, value + err))
            err_left = int(plot_left + (err_left_val - x_min) / max(x_max - x_min, 1e-6) * (plot_right - plot_left))
            err_right = int(plot_left + (err_right_val - x_min) / max(x_max - x_min, 1e-6) * (plot_right - plot_left))
            draw.line((err_left, cy, err_right, cy), fill="#333333", width=2)
            draw.line((err_left, cy - 6, err_left, cy + 6), fill="#333333", width=2)
            draw.line((err_right, cy - 6, err_right, cy + 6), fill="#333333", width=2)
        draw.text((x0 + 16, cy - 8), _pretty(label), fill="#222222", font=tick_font)
        value_text = f"{value:.3f}"
        draw.text((plot_left + width + 8, cy - 8), value_text, fill="#222222", font=tick_font)
    xlabel = "Test accuracy"
    tw, _ = _text_size(draw, xlabel, label_font)
    draw.text(((plot_left + plot_right - tw) // 2, y1 - 24), xlabel, fill="black", font=label_font)


def _draw_grouped_bar_chart(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    labels: list[str],
    support_values: np.ndarray,
    conflict_values: np.ndarray,
    *,
    title: str,
) -> None:
    title_font = _load_font(24)
    label_font = _load_font(16)
    tick_font = _load_font(14)
    _draw_panel_frame(draw, box, title, title_font)
    x0, y0, x1, y1 = box
    plot_left = x0 + 70
    plot_right = x1 - 20
    plot_top = y0 + 70
    plot_bottom = y1 - 110
    combined = [v for v in list(support_values) + list(conflict_values) if not math.isnan(v)]
    y_min = min(-0.25, min(combined) - 0.05 if combined else -0.25)
    y_max = max(0.75, max(combined) + 0.05 if combined else 0.75)
    zero_y = int(plot_bottom - (0 - y_min) / max(y_max - y_min, 1e-6) * (plot_bottom - plot_top))
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="black", width=2)
    draw.line((plot_left, zero_y, plot_right, zero_y), fill="black", width=2)
    for frac in np.linspace(0.0, 1.0, 5):
        value = y_min + (y_max - y_min) * (1.0 - frac)
        y = int(plot_top + frac * (plot_bottom - plot_top))
        draw.line((plot_left, y, plot_right, y), fill="#efefef", width=1)
        text = f"{value:.2f}"
        tw, th = _text_size(draw, text, tick_font)
        draw.text((plot_left - tw - 8, y - th // 2), text, fill="#555555", font=tick_font)
    n = max(len(labels), 1)
    span = plot_right - plot_left
    group_width = span / n
    bar_width = max(16, int(group_width * 0.28))
    for idx, (label, support, conflict) in enumerate(zip(labels, support_values, conflict_values)):
        center = int(plot_left + group_width * (idx + 0.5))
        for offset, value, color, text_dy in [
            (-bar_width // 2 - 4, support, "#1f78b4", 4),
            (bar_width // 2 + 4, conflict, "#e31a1c", -18 if conflict < 0 else 4),
        ]:
            left = center + offset - bar_width // 2
            right = center + offset + bar_width // 2
            y_value = int(plot_bottom - (value - y_min) / max(y_max - y_min, 1e-6) * (plot_bottom - plot_top))
            top = min(zero_y, y_value)
            bottom = max(zero_y, y_value)
            draw.rounded_rectangle((left, top, right, bottom), radius=4, fill=color)
            value_text = f"{value:.2f}"
            tw, th = _text_size(draw, value_text, tick_font)
            draw.text((center + offset - tw // 2, y_value - th - text_dy if value >= 0 else y_value + 4), value_text, fill="#222222", font=tick_font)
        _draw_rotated_text(image, (center, plot_bottom + 20), _pretty(label), tick_font, angle=30, anchor="mt")
    legend_y = y0 + 38
    draw.rectangle((x1 - 210, legend_y, x1 - 194, legend_y + 16), fill="#1f78b4")
    draw.text((x1 - 188, legend_y - 2), "Support-evidence corr", fill="#222222", font=tick_font)
    draw.rectangle((x1 - 210, legend_y + 24, x1 - 194, legend_y + 40), fill="#e31a1c")
    draw.text((x1 - 188, legend_y + 22), "Conflict-resistance corr", fill="#222222", font=tick_font)
    _draw_rotated_text(image, (x0 + 22, (plot_top + plot_bottom) // 2), "Correlation", label_font, angle=90, anchor="mm")


def main() -> None:
    args = build_parser().parse_args()
    frozen_rows = _load_rows(ROOT / args.frozen_summary)
    e2e_rows = _load_rows(ROOT / args.e2e_summary)
    synthetic_rows = _load_rows(ROOT / args.synthetic_summary)

    frozen_models = [
        "linear",
        "mlp",
        "cosine_prototype",
        "eml_centered_ambiguity",
        "merc_linear",
        "merc_energy",
    ]
    e2e_models = [
        "linear",
        "mlp",
        "cosine_prototype",
        "eml_centered_ambiguity",
        "eml_no_ambiguity",
        "merc_linear",
        "merc_energy",
        "merc_block_linear",
        "merc_block_energy",
    ]
    synthetic_models = [
        "merc_linear",
        "merc_energy",
        "merc_linear_small",
        "merc_energy_small",
    ]

    frozen_labels, frozen_means, frozen_stds = _group_metric(frozen_rows, frozen_models, "test_accuracy")
    e2e_labels, e2e_means, e2e_stds = _group_metric(e2e_rows, e2e_models, "test_accuracy", loss_mode="")
    syn_labels, syn_support, syn_conflict = _group_dual_metric(
        synthetic_rows,
        synthetic_models,
        "test_support_evidence_corr",
        "test_conflict_resistance_corr",
    )

    canvas = Image.new("RGBA", (1800, 680), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = _load_font(32)
    draw.text((32, 20), "MERC vs Baselines: Real-Server Comparison", fill="black", font=header_font)

    _draw_vertical_bar_chart(
        canvas,
        draw,
        (24, 78, 580, 648),
        frozen_labels,
        frozen_means,
        frozen_stds,
        y_label="Test accuracy",
        y_min=0.45,
        y_max=max(0.58, float(frozen_means.max()) + 0.03),
        title="Frozen CNN Features on CIFAR-10",
    )
    _draw_horizontal_bar_chart(
        draw,
        (612, 78, 1220, 648),
        e2e_labels,
        e2e_means,
        e2e_stds,
        x_min=0.10,
        x_max=max(0.60, float(e2e_means.max()) + 0.05),
        title="End-to-End CNN + Head on CIFAR-10",
    )
    _draw_grouped_bar_chart(
        canvas,
        draw,
        (1252, 78, 1776, 648),
        syn_labels,
        syn_support,
        syn_conflict,
        title="Synthetic Evidence Alignment",
    )

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output)
    print(output)


if __name__ == "__main__":
    main()
