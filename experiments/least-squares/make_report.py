"""Aggregate study results into Typst table fragments and convergence plots.

Reads ``results/<model>/<study>/results.json`` written by ``study_runner.py``
and produces:

* ``content/article/least-squares/<model>-<study>.typ`` — a
  ``three-line-table`` fragment the paper includes directly, so the tables in
  the paper are regenerated from data rather than typed by hand;
* ``public/images/least-squares/<model>/convergence-<study>.png`` — log-log
  convergence curves with sample-standard-deviation bars and the theoretical
  reference rates.

Usage::

    python make_report.py            # aggregate everything found in results/
    python make_report.py --model elasticity-2d --study main
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
TABLE_DIR = PROJECT_ROOT / "content" / "article" / "least-squares"
IMAGE_ROOT = PROJECT_ROOT / "public" / "images" / "least-squares"

# CVD-safe two-series palette (blue / Okabe-Ito vermillion) plus neutral gray
# for reference-slope lines.
COLOR_PRIMARY = "#0077B6"
COLOR_SECONDARY = "#D55E00"
COLOR_REFERENCE = "#8A8A8A"

MODELS = {
    "elasticity-2d": {
        "metrics": ("sigma_hdiv_error", "u_h1_error"),
        "metric_labels": (
            r"$\|\sigma_N-\sigma_\star\|_{H(\mathrm{div})}$",
            r"$\|u_N-u_\star\|_{H^1}$",
        ),
        "metric_headers": (
            "$norm(bold(sigma)_star - bold(sigma)_N)_(bold(H)(\"div\"))$",
            "$norm(bold(u)_star - bold(u)_N)_(H^1)$",
        ),
        "residuals": ("constitutive_residual", "equilibrium_residual"),
        "beta": (4.5 - 1.0) / 2.0,
        "image_dir": "linear-elasticity-2d",
    },
    "elasticity-3d": {
        "metrics": ("sigma_hdiv_error", "u_h1_error"),
        "metric_labels": (
            r"$\|\sigma_N-\sigma_\star\|_{H(\mathrm{div})}$",
            r"$\|u_N-u_\star\|_{H^1}$",
        ),
        "metric_headers": (
            "$norm(bold(sigma)_star - bold(sigma)_N)_(bold(H)(\"div\"))$",
            "$norm(bold(u)_star - bold(u)_N)_(H^1)$",
        ),
        "residuals": ("constitutive_residual", "equilibrium_residual"),
        "beta": (5.0 - 1.0) / 3.0,
        "image_dir": "linear-elasticity-3d",
    },
    "plane-stress": {
        "metrics": ("sigma_hdiv_error", "u_h1_error"),
        "metric_labels": (
            r"$\|\sigma_N-\sigma_\star\|_{H(\mathrm{div})}$",
            r"$\|u_N-u_\star\|_{H^1}$",
        ),
        "metric_headers": (
            "$norm(bold(sigma)_star - bold(sigma)_N)_(bold(H)(\"div\"))$",
            "$norm(bold(u)_star - bold(u)_N)_(H^1)$",
        ),
        "residuals": ("constitutive_residual", "equilibrium_residual"),
        "beta": (4.5 - 1.0) / 2.0,
        "image_dir": "plane-stress",
    },
    "plate": {
        "metrics": ("M_hdivdiv_error", "w_h2_error"),
        "metric_labels": (
            r"$\|M_N-M_\star\|_{H(\mathrm{div\,div})}$",
            r"$\|w_N-w_\star\|_{H^2}$",
        ),
        "metric_headers": (
            "$norm(bold(M)_star - bold(M)_N)_(bold(H)(\"div\" \"div\"))$",
            "$norm(w_star - w_N)_(H^2)$",
        ),
        "residuals": ("r_c", "r_e"),
        "beta": (4.5 - 2.0) / 2.0,
        "image_dir": "plate-bending",
    },
}

STUDIES = ("main", "q", "k", "nu")

# The paper reports the coefficient-ball algorithm; other registered solvers
# only appear in the auxiliary comparison artifacts.
PAPER_ALGORITHM = "ball"
ALGORITHM_PLOT_STYLE = {
    "ball": {"color": COLOR_PRIMARY, "marker": "o", "linestyle": "-"},
    "ridge": {"color": COLOR_SECONDARY, "marker": "s", "linestyle": "--"},
    "tsvd": {"color": "#009E73", "marker": "^", "linestyle": "-."},
}


def row_algorithm(row: dict) -> str:
    """Algorithm id of one summary row; legacy files predate the column."""

    return row.get("algorithm", PAPER_ALGORITHM)


STUDY_HEADER_CELLS = {
    "main": ["$N$", "$Q$", "$K$"],
    "q": ["$Q\\/(N+1)$", "$Q$"],
    "k": ["$K\\/(N+1)$", "$K$"],
    "nu": ["$nu$", "$lambda$"],
}


def study_key_cells(study: str, row: dict) -> list[str]:
    """Typst cells identifying one summary row within its study."""

    if study == "main":
        return [f"${row['N']}$", f"${row['Q']}$", f"${row['K']}$"]
    if study == "q":
        return [f"${round(row['Q'] / (row['N'] + 1))}$", f"${row['Q']}$"]
    if study == "k":
        return [f"${row['K'] / (row['N'] + 1):.2f}$", f"${row['K']}$"]
    lam = row["nu"] / ((1.0 + row["nu"]) * (1.0 - 2.0 * row["nu"])) * 4.0 / 3.0
    return [f"${row['nu']:g}$", f"${typst_scientific(lam)}$"]


def typst_scientific(value: float) -> str:
    """Format one positive number as Typst math scientific notation."""

    if not math.isfinite(value):
        return str(value)
    if value == 0.0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / 10.0**exponent
    return f"{mantissa:.2f} times 10^({exponent})"


def typst_mean_std(mean: float, std: float) -> str:
    """Format ``mean ± std`` as a Typst math cell."""

    if std > 0.0:
        return f"${typst_scientific(mean)} plus.minus {typst_scientific(std)}$"
    return f"${typst_scientific(mean)}$"


def load_summary(model: str, study: str) -> list[dict] | None:
    path = ROOT / "results" / model / study / "results.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["summary"]


def sort_key(study: str):
    field = {"main": "N", "q": "Q", "k": "K", "nu": "nu"}[study]
    return lambda row: row[field]


def observed_order(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of log(y) against log(x)."""

    mask = (x > 0) & (y > 0)
    if mask.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0])


def emit_table(model: str, study: str, rows: list[dict]) -> Path:
    spec = MODELS[model]
    err1, err2 = spec["metrics"]
    res1, res2 = spec["residuals"]
    header_cells = STUDY_HEADER_CELLS[study]
    headers = header_cells + [
        spec["metric_headers"][0],
        spec["metric_headers"][1],
        "本构残差",
        "平衡残差",
    ]

    lines = [
        "// Auto-generated by experiments/least-squares/make_report.py; do not edit.",
        '#import "/typ/templates/shared.typ": three-line-table',
        "",
        "#three-line-table(",
        f"  columns: {len(headers)},",
        f"  align: (right,) * {len(header_cells)} + (center,) * 4,",
        ")[",
        "  | " + " | ".join(headers) + " |",
        "  |" + "---|" * len(headers),
    ]
    for row in rows:
        cells = study_key_cells(study, row) + [
            typst_mean_std(row[f"{err1}_mean"], row[f"{err1}_std"]),
            typst_mean_std(row[f"{err2}_mean"], row[f"{err2}_std"]),
            typst_mean_std(row[f"{res1}_mean"], row[f"{res1}_std"]),
            typst_mean_std(row[f"{res2}_mean"], row[f"{res2}_std"]),
        ]
        lines.append("  | " + " | ".join(cells) + " |")
    lines.append("]")
    lines.append("")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / f"{model}-{study}.typ"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def emit_plot(model: str, study: str, rows: list[dict]) -> Path | None:
    spec = MODELS[model]
    err1, err2 = spec["metrics"]
    x_field = {"main": "N", "q": "Q", "k": "K", "nu": "nu"}[study]
    x = np.asarray([row[x_field] for row in rows], dtype=float)
    if study == "nu":
        # Plot against lambda to show flatness in the incompressible limit.
        x = x / ((1.0 + x) * (1.0 - 2.0 * x)) * 4.0 / 3.0
    means = [np.asarray([row[f"{e}_mean"] for row in rows]) for e in (err1, err2)]
    stds = [np.asarray([row[f"{e}_std"] for row in rows]) for e in (err1, err2)]
    if len(x) < 2:
        return None

    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for mean, std, color, label in zip(
        means,
        stds,
        (COLOR_PRIMARY, COLOR_SECONDARY),
        spec["metric_labels"],
    ):
        ax.errorbar(
            x,
            mean,
            yerr=std,
            color=color,
            marker="o",
            markersize=5,
            linewidth=2,
            capsize=3,
            label=label,
        )

    x_labels = {
        "main": "$N$",
        "q": "$Q$",
        "k": "$K$",
        "nu": r"$\lambda$",
    }
    ax.set_xlabel(x_labels[study])
    ax.set_ylabel("Graph-norm error")
    ax.set_xscale("log")
    ax.set_yscale("log")

    if study == "main":
        beta = spec["beta"]
        reference = (x / np.log(x)) ** (-beta)
        anchor = means[0][0] / reference[0]
        ax.plot(
            x,
            anchor * reference,
            color=COLOR_REFERENCE,
            linestyle="--",
            linewidth=1.5,
            label=rf"$(N/\log N)^{{-{beta:.2f}}}$",
        )
        slope = observed_order(x, means[0])
        ax.set_title(f"observed slope in N: {slope:.2f}", fontsize=10)
    elif study == "q":
        reference = x ** (-0.25)
        anchor = means[0][0] / reference[0]
        ax.plot(
            x,
            anchor * reference,
            color=COLOR_REFERENCE,
            linestyle="--",
            linewidth=1.5,
            label=r"$Q^{-1/4}$",
        )

    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    image_dir = IMAGE_ROOT / spec["image_dir"]
    image_dir.mkdir(parents=True, exist_ok=True)
    path = image_dir / f"convergence-{study}.png"
    fig.savefig(path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    return path


def emit_solver_table(model: str, study: str, rows: list[dict]) -> Path:
    """Emit the auxiliary solver-comparison table (all algorithms)."""

    spec = MODELS[model]
    err1, err2 = spec["metrics"]
    header_cells = STUDY_HEADER_CELLS[study]
    headers = ["算法"] + header_cells + [
        spec["metric_headers"][0],
        spec["metric_headers"][1],
    ]

    lines = [
        "// Auto-generated by experiments/least-squares/make_report.py; do not edit.",
        '#import "/typ/templates/shared.typ": three-line-table',
        "",
        "#three-line-table(",
        f"  columns: {len(headers)},",
        f"  align: (left,) + (right,) * {len(header_cells)} + (center,) * 2,",
        ")[",
        "  | " + " | ".join(headers) + " |",
        "  |" + "---|" * len(headers),
    ]
    for row in sorted(rows, key=lambda item: (row_algorithm(item), sort_key(study)(item))):
        cells = (
            [f"`{row_algorithm(row)}`"]
            + study_key_cells(study, row)
            + [
                typst_mean_std(row[f"{err1}_mean"], row[f"{err1}_std"]),
                typst_mean_std(row[f"{err2}_mean"], row[f"{err2}_std"]),
            ]
        )
        lines.append("  | " + " | ".join(cells) + " |")
    lines.append("]")
    lines.append("")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / f"{model}-{study}-solvers.typ"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def emit_solver_plot(model: str, study: str, rows: list[dict]) -> Path | None:
    """Plot the primary graph-norm error with one curve per algorithm."""

    spec = MODELS[model]
    err1 = spec["metrics"][0]
    x_field = {"main": "N", "q": "Q", "k": "K", "nu": "nu"}[study]

    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    plotted = False
    for algorithm in sorted({row_algorithm(row) for row in rows}):
        algorithm_rows = sorted(
            (row for row in rows if row_algorithm(row) == algorithm),
            key=sort_key(study),
        )
        x = np.asarray([row[x_field] for row in algorithm_rows], dtype=float)
        if study == "nu":
            x = x / ((1.0 + x) * (1.0 - 2.0 * x)) * 4.0 / 3.0
        if len(x) < 2:
            continue
        mean = np.asarray([row[f"{err1}_mean"] for row in algorithm_rows])
        std = np.asarray([row[f"{err1}_std"] for row in algorithm_rows])
        style = ALGORITHM_PLOT_STYLE.get(algorithm, {"color": COLOR_REFERENCE})
        ax.errorbar(
            x,
            mean,
            yerr=std,
            color=style.get("color"),
            marker=style.get("marker", "o"),
            linestyle=style.get("linestyle", "-"),
            markersize=5,
            linewidth=2,
            capsize=3,
            label=algorithm,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return None

    ax.set_xlabel({"main": "$N$", "q": "$Q$", "k": "$K$", "nu": r"$\lambda$"}[study])
    ax.set_ylabel(spec["metric_labels"][0])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False, fontsize=9, title="solver")
    fig.tight_layout()

    image_dir = IMAGE_ROOT / spec["image_dir"]
    image_dir.mkdir(parents=True, exist_ok=True)
    path = image_dir / f"convergence-{study}-solvers.png"
    fig.savefig(path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS))
    parser.add_argument("--study", choices=STUDIES)
    args = parser.parse_args()

    models = (args.model,) if args.model else tuple(MODELS)
    studies = (args.study,) if args.study else STUDIES
    produced = False
    for model in models:
        for study in studies:
            summary = load_summary(model, study)
            if summary is None:
                continue
            rows = sorted(summary, key=sort_key(study))
            paper_rows = [row for row in rows if row_algorithm(row) == PAPER_ALGORITHM]
            if paper_rows:
                table_path = emit_table(model, study, paper_rows)
                plot_path = emit_plot(model, study, paper_rows)
                produced = True
                print(f"{model}/{study}: table -> {table_path}")
                if plot_path is not None:
                    print(f"{model}/{study}: plot  -> {plot_path}")
            if len({row_algorithm(row) for row in rows}) > 1:
                solver_table = emit_solver_table(model, study, rows)
                solver_plot = emit_solver_plot(model, study, rows)
                produced = True
                print(f"{model}/{study}: solver table -> {solver_table}")
                if solver_plot is not None:
                    print(f"{model}/{study}: solver plot  -> {solver_plot}")
    if not produced:
        print("No results found under results/.")


if __name__ == "__main__":
    main()
