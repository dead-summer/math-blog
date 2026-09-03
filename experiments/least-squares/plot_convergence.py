"""Draw the convergence figures the paper shows, and fit the observed orders.

Reads the ``summary.csv`` files ``study_runner.py`` writes under
``results/<model>/<study>/`` and produces:

* model-specific order (fixed-k N-convergence), power (fixed-N
  activation-power), and training-point figures under
  ``public/images/least-squares/campaign/``;
* the paired two-dimensional near-incompressibility convergence figure;
* ``results/observed-orders.csv`` — the log-log slopes quoted in the paper's
  order-comparison table, so no number in the text is without a CSV behind it.

Tables are not generated: the paper's tables are written by hand against these
CSVs.

Usage::

    python plot_convergence.py
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
RESULTS_ROOT = ROOT / "results"
IMAGE_DIR = PROJECT_ROOT / "public" / "images" / "least-squares" / "campaign"
ORDER_CSV = RESULTS_ROOT / "observed-orders.csv"

# CVD-safe two-series palette (blue / Okabe-Ito vermillion) plus neutral gray
# for reference-slope lines.
COLOR_PRIMARY = "#0077B6"
COLOR_SECONDARY = "#D55E00"
COLOR_TERTIARY = "#009E73"
COLOR_REFERENCE = "#8A8A8A"

# The paper reports the coefficient-ball algorithm; the other registered
# solvers only appear in the text's solver comparison, not in these figures.
PAPER_ALGORITHM = "ball"

MODELS = {
    "elasticity-2d": {
        "title": "2D linear elasticity",
        "metrics": ("sigma_hdiv_error", "u_h1_error"),
        "metric_labels": (
            r"$\|\sigma_N-\sigma_\star\|_{H(\mathrm{div})}$",
            r"$\|u_N-u_\star\|_{H^1}$",
        ),
        "dimension": 2,
        "sobolev_order": 1,
    },
    "elasticity-3d": {
        "title": "3D linear elasticity",
        "metrics": ("sigma_hdiv_error", "u_h1_error"),
        "metric_labels": (
            r"$\|\sigma_N-\sigma_\star\|_{H(\mathrm{div})}$",
            r"$\|u_N-u_\star\|_{H^1}$",
        ),
        "dimension": 3,
        "sobolev_order": 1,
    },
    "plane-stress": {
        "title": "Plane stress",
        "metrics": ("sigma_hdiv_error", "u_h1_error"),
        "metric_labels": (
            r"$\|\sigma_N-\sigma_\star\|_{H(\mathrm{div})}$",
            r"$\|u_N-u_\star\|_{H^1}$",
        ),
        "dimension": 2,
        "sobolev_order": 1,
    },
    "plate": {
        "title": "Kirchhoff–Love plate",
        "metrics": ("M_hdivdiv_error", "w_h2_error"),
        "metric_labels": (
            r"$\|M_N-M_\star\|_{H(\mathrm{div\,div})}$",
            r"$\|w_N-w_\star\|_{H^2}$",
        ),
        "dimension": 2,
        "sobolev_order": 2,
    },
}


def reference_beta(spec: dict, power: int) -> float:
    """Theoretical rate ``beta = (s_cap(d) - m)/d`` for one activation power.

    ``s_cap(d) = (d + 2k + 1)/2`` is the dictionary saturation index, so every
    extra derivative in the error norm costs one factor ``N^{1/d}`` and every
    extra unit of activation power buys ``N^{1/d}``.
    """

    dimension = spec["dimension"]
    saturation = 0.5 * (dimension + 2 * power + 1)
    return (saturation - spec["sobolev_order"]) / dimension


def observed_order(x: np.ndarray, y: np.ndarray) -> float:
    """Fitted decay rate: minus the least-squares slope of log(y) on log(x).

    Reported positive, matching the convention of the theoretical ``beta``.
    """

    mask = (x > 0) & (y > 0)
    if mask.sum() < 2:
        return float("nan")
    return -float(np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0])


def load_summary(model: str, study: str) -> list[dict]:
    """Rows of one study's ``summary.csv``, restricted to the paper's solver."""

    path = RESULTS_ROOT / model / study / "summary.csv"
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("algorithm", PAPER_ALGORITHM) != PAPER_ALGORITHM:
                continue
            rows.append({key: _maybe_float(value) for key, value in row.items()})
    return rows


def _maybe_float(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def series(rows: list[dict], field: str, metric: str) -> tuple[np.ndarray, ...]:
    """Sorted ``(x, mean, std)`` arrays for one metric over one varying field."""

    ordered = sorted(rows, key=lambda row: row[field])
    return (
        np.asarray([row[field] for row in ordered], dtype=float),
        np.asarray([row[f"{metric}_mean"] for row in ordered], dtype=float),
        np.asarray([row[f"{metric}_std"] for row in ordered], dtype=float),
    )


def draw(ax, x, mean, std, color, label, marker="o", linestyle="-") -> None:
    ax.errorbar(
        x,
        mean,
        yerr=std,
        color=color,
        marker=marker,
        linestyle=linestyle,
        markersize=5,
        linewidth=2,
        capsize=3,
        label=label,
    )


def finish(ax, xlabel: str, ylabel: str = "Graph-norm error") -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False, fontsize=8)


def save(fig, name: str) -> Path:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = IMAGE_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Model-specific figures
# ---------------------------------------------------------------------------

# The paper now includes the completed three-dimensional ladder as a primary
# panel.  Its incompressibility scan is deliberately kept out of this list:
# that diagnostic has moved to the two-dimensional experiment.
ORDER_PANELS = ("elasticity-2d", "plane-stress", "elasticity-3d", "plate")
# Only the two-dimensional elasticity and plate models sweep the activation
# power; the three-dimensional model is run at k=7 alone.
POWER_PANELS = ("elasticity-2d", "plate")


ORDER_STUDIES = ("order", "order-k3")


def collect_orders() -> list[dict]:
    """Fitted decay rates over the N ladder, for every model that has one.

    These are the "measured order" column of the paper's order-comparison
    table; the table's other columns come from the floors CSVs.
    """

    orders = []
    for model, spec in MODELS.items():
        for study in ORDER_STUDIES:
            rows = load_summary(model, study)
            if len(rows) < 2:
                continue
            power = int(rows[0]["activation_power"])
            for metric in spec["metrics"]:
                x, mean, _ = series(rows, "N", metric)
                orders.append({
                    "model": model,
                    "study": study,
                    "activation_power": power,
                    "lambda": "",
                    "widths": " ".join(f"{value:g}" for value in x),
                    "metric": metric,
                    "observed_order": f"{observed_order(x, mean):.4f}",
                })
    near_rows = load_summary("elasticity-2d", "near-incompressible")
    for lam in sorted({row["lambda"] for row in near_rows}):
        rows = [row for row in near_rows if row["lambda"] == lam]
        if len(rows) < 2:
            continue
        for metric in MODELS["elasticity-2d"]["metrics"]:
            x, mean, _ = series(rows, "N", metric)
            orders.append({
                "model": "elasticity-2d",
                "study": "near-incompressible",
                "activation_power": int(rows[0]["activation_power"]),
                "lambda": f"{lam:g}",
                "widths": " ".join(f"{value:g}" for value in x),
                "metric": metric,
                "observed_order": f"{observed_order(x, mean):.4f}",
            })
    return orders


def figure_order(model: str) -> Path | None:
    """Both graph-norm errors over the paper's fixed-k order N ladder."""

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    spec = MODELS[model]
    rows = load_summary(model, "order")
    if len(rows) < 2:
        plt.close(fig)
        return None

    power = int(rows[0]["activation_power"])
    for metric, color, marker, label in zip(
        spec["metrics"],
        (COLOR_PRIMARY, COLOR_SECONDARY),
        ("o", "s"),
        spec["metric_labels"],
    ):
        x, mean, std = series(rows, "N", metric)
        draw(ax, x, mean, std, color, label, marker)

    beta = reference_beta(spec, power)
    reference = x ** (-beta)
    _, first_mean, _ = series(rows, "N", spec["metrics"][0])
    ax.plot(
        x,
        first_mean[0] / reference[0] * reference,
        color=COLOR_REFERENCE,
        linestyle=":",
        linewidth=1.4,
        label=rf"$N^{{-{beta:.2f}}}$",
    )
    ax.set_xscale("log")
    ax.set_xticks(x, labels=[str(int(value)) for value in x])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_title(spec["title"], fontsize=10)
    finish(ax, "$N$")
    return save(fig, f"convergence-order-{model}.png")


def figure_power(model: str) -> Path | None:
    """One model's error against activation power at fixed N."""

    spec = MODELS[model]
    rows = load_summary(model, "power")
    if len(rows) < 2:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    powers, _, _ = series(rows, "activation_power", spec["metrics"][0])
    for metric, color, label in zip(
        spec["metrics"], (COLOR_PRIMARY, COLOR_SECONDARY), spec["metric_labels"]
    ):
        _, mean, std = series(rows, "activation_power", metric)
        draw(ax, powers, mean, std, color, label)
    # Anchor at the smallest power.  Constants in the approximation bound may
    # depend on k, so this is only an equal-constant heuristic, not a fixed-N
    # quantitative prediction.
    betas = np.array([reference_beta(spec, int(p)) for p in powers])
    width = float(sorted(rows, key=lambda row: row["activation_power"])[0]["N"])
    _, mean, _ = series(rows, "activation_power", spec["metrics"][0])
    ax.plot(
        powers,
        mean[0] * width ** (-(betas - betas[0])),
        color=COLOR_REFERENCE,
        linestyle=":",
        linewidth=1.4,
        label=rf"$N^{{-(\beta(k)-\beta({int(powers[0])}))}}$",
    )
    ax.set_xticks(powers)
    ax.set_title(spec["title"], fontsize=10)
    finish(ax, "$k$")
    return save(fig, f"convergence-power-{model}.png")


def figure_q(model: str) -> Path | None:
    """One model's training-point sweep against the Q^{-1/4} reference."""

    rows = load_summary(model, "q")
    if len(rows) < 2:
        return None
    spec = MODELS[model]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    x, mean, std = series(rows, "Q", spec["metrics"][0])
    draw(ax, x, mean, std, COLOR_PRIMARY, spec["metric_labels"][0])
    reference = x ** (-0.25)
    ax.plot(
        x,
        mean[0] / reference[0] * reference,
        color=COLOR_REFERENCE,
        linestyle=":",
        linewidth=1.4,
        label=r"$Q^{-1/4}$",
    )
    ax.set_xscale("log")
    ax.set_title(spec["title"], fontsize=10)
    finish(ax, "$Q$")
    return save(fig, f"convergence-q-{model}.png")


def figure_near_incompressible_2d() -> Path | None:
    """Complete N ladders at three Lamé parameters, following Cai's protocol."""

    rows = load_summary("elasticity-2d", "near-incompressible")
    lambdas = sorted({row["lambda"] for row in rows})
    if len(lambdas) < 2:
        return None

    spec = MODELS["elasticity-2d"]
    colors = (COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY)
    markers = ("o", "s", "^")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), sharex=True)
    for ax, metric, ylabel in zip(axes, spec["metrics"], spec["metric_labels"]):
        for lam, color, marker in zip(lambdas, colors, markers):
            material_rows = [row for row in rows if row["lambda"] == lam]
            x, mean, std = series(material_rows, "N", metric)
            exponent = round(math.log10(lam))
            if lam >= 1000 and math.isclose(lam, 10.0**exponent):
                parameter_label = rf"$\lambda=10^{{{exponent}}}$"
            else:
                parameter_label = rf"$\lambda={lam:g}$"
            draw(
                ax,
                x,
                mean,
                std,
                color,
                parameter_label,
                marker,
            )
        ax.set_xscale("log")
        ax.set_xticks(x, labels=[str(int(value)) for value in x])
        ax.set_xlim(x[0] / 1.08, x[-1] * 1.08)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(ylabel, fontsize=10)
        finish(ax, "$N$", "Error")
    return save(fig, "convergence-near-incompressible-elasticity-2d.png")


def write_orders(orders: list[dict]) -> Path | None:
    if not orders:
        return None
    ORDER_CSV.parent.mkdir(parents=True, exist_ok=True)
    with ORDER_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(orders[0]))
        writer.writeheader()
        writer.writerows(orders)
    return ORDER_CSV


def main() -> None:
    global RESULTS_ROOT, IMAGE_DIR, ORDER_CSV
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
        help="root containing <model>/<study>/summary.csv (default: results/)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=IMAGE_DIR,
        help="directory for generated figures (default: public/images/.../campaign)",
    )
    args = parser.parse_args()
    RESULTS_ROOT = args.results_root
    IMAGE_DIR = args.image_dir
    ORDER_CSV = RESULTS_ROOT / "observed-orders.csv"
    plt.rcParams["axes.unicode_minus"] = False

    produced = [
        *(figure_order(model) for model in ORDER_PANELS),
        *(figure_power(model) for model in POWER_PANELS),
        *(figure_q(model) for model in ORDER_PANELS),
        figure_near_incompressible_2d(),
        write_orders(collect_orders()),
    ]
    for path in produced:
        if path is not None:
            print(f"Wrote: {path}")
    if not any(path is not None for path in produced):
        print("No summary.csv found under results/.")


if __name__ == "__main__":
    main()
