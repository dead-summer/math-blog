from __future__ import annotations

import gc
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from linear_elasticity_3d import (
    ALGO_STYLE,
    AlgorithmResult,
    LeastSquaresConfig,
    VALID_ALGORITHMS,
    build_shared_benchmark,
    build_shared_feature_space,
    clear_cuda_cache,
    configure_plotting,
    print_aligned_markdown_table,
    resolve_output_dir,
    run_experiment,
    validate_algorithm_selection,
)

DEFAULT_ABLATION_NU_LIST = [0.49, 0.499, 0.4999, 0.49999, 0.499999]
ALGORITHM_LABELS = {
    "direct": "LS",
}


@dataclass(frozen=True)
class AblationRecord:
    """One completed Poisson-ratio comparison record."""

    algorithm_id: str
    algorithm_name: str
    nu: float
    result: AlgorithmResult


def clear_experiment_memory() -> None:
    """Release CPU references and cached CUDA memory between runs."""

    gc.collect()
    clear_cuda_cache()


def format_nu(nu: float) -> str:
    """Format a Poisson ratio compactly for labels and tables."""

    return f"{nu:.6g}"


def validate_ablation_nu_list(ablation_nu_list: Sequence[float]) -> list[float]:
    """Validate and sort the ablation Poisson-ratio list."""

    if not ablation_nu_list:
        raise ValueError("ablation_nu_list must be non-empty.")

    seen: set[float] = set()
    validated_list: list[float] = []
    for nu in ablation_nu_list:
        nu_float = float(nu)
        if not math.isfinite(nu_float):
            raise ValueError("All nu values must be finite.")
        if not (-1.0 < nu_float < 0.5):
            raise ValueError("All nu values must lie in (-1, 0.5).")
        if nu_float in seen:
            raise ValueError(f"Duplicate nu value detected: {format_nu(nu_float)}")
        seen.add(nu_float)
        validated_list.append(nu_float)
    return sorted(validated_list)


def sort_ablation_records(
    records: list[AblationRecord],
    algorithm_ids: list[str],
) -> list[AblationRecord]:
    """Group rows by solver, then by ascending Poisson ratio."""

    algorithm_order = {
        algorithm_id: index for index, algorithm_id in enumerate(algorithm_ids)
    }
    return sorted(
        records,
        key=lambda record: (
            algorithm_order[record.algorithm_id],
            record.nu,
        ),
    )


def print_ablation_summary_table(
    records: list[AblationRecord],
    algorithm_ids: list[str],
) -> None:
    """Print the nu ablation summary grouped by method."""

    ordered_records = sort_ablation_records(records, algorithm_ids)
    if not ordered_records:
        return

    headers = (
        "Method",
        "nu",
        "‖Φ^u-u‖",
        "‖Φ^σ-σ‖",
        "Time(s)",
    )
    rows = [
        (
            record.algorithm_name,
            format_nu(record.nu),
            f"{record.result.u_l2_error:.2e}",
            f"{record.result.sigma_l2_error:.2e}",
            f"{record.result.wall_time:.2f}",
        )
        for record in ordered_records
    ]
    print_aligned_markdown_table(
        title="nu Ablation Summary",
        headers=headers,
        rows=rows,
        alignments=("left", "center", "center", "center", "center"),
    )


def plot_ablation_nu(
    results: dict[float, dict[str, AlgorithmResult]],
    algorithm_ids: list[str],
    save_path: str,
) -> None:
    """Plot absolute L2 error versus Poisson ratio for each configured solver."""

    if not results:
        print(f"  Skipped: {save_path} (no results to plot)")
        return

    configure_plotting()
    nu_list = sorted(results.keys())
    x_positions = np.arange(len(nu_list), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(max(10.5, 2.0 * len(nu_list)), 4.8))
    metric_specs = [
        (
            "u_l2_error",
            r"$\|\Phi^u - u_{ex}\|_0$",
            "$L^2$ error",
        ),
        (
            "sigma_l2_error",
            r"$\|\Phi^{\sigma} - \sigma_{ex}\|_0$",
            "$L^2$ error",
        ),
    ]

    for ax, (metric_name, title, ylabel) in zip(axes, metric_specs):
        for algorithm_id in algorithm_ids:
            algorithm_name = ALGORITHM_LABELS[algorithm_id]
            values = np.array(
                [
                    getattr(results[nu][algorithm_name], metric_name)
                    if algorithm_name in results[nu]
                    else float("nan")
                    for nu in nu_list
                ],
                dtype=float,
            )
            valid = np.isfinite(values) & (values > 0.0)
            if not valid.any():
                continue

            style = ALGO_STYLE.get(
                algorithm_name,
                {"color": "#4C78A8", "marker": "o", "linestyle": "-"},
            )
            ax.semilogy(
                x_positions[valid],
                values[valid],
                marker=style["marker"],
                color=style["color"],
                linestyle=style.get("linestyle", "-"),
                linewidth=1.5,
                markersize=6,
                label=algorithm_name,
            )

        ax.set_title(title)
        ax.set_xlabel(r"Poisson ratio $\nu$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([format_nu(nu) for nu in nu_list])
        ax.grid(alpha=0.3, linestyle="--")
        if ax.lines:
            ax.legend()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def run_ablation_nu(
    cfg: LeastSquaresConfig | None = None,
    ablation_nu_list: Sequence[float] | None = None,
) -> dict[float, dict[str, AlgorithmResult]]:
    """Run the synchronized Poisson-ratio ablation study and return all metrics."""

    cfg = LeastSquaresConfig() if cfg is None else cfg
    ablation_output_dir = resolve_output_dir(cfg.manufactured_solution) / "ablation" / "nu"
    ablation_output_dir.mkdir(parents=True, exist_ok=True)
    algorithm_ids = validate_algorithm_selection(cfg.algorithms_to_run, VALID_ALGORITHMS)
    nu_list = validate_ablation_nu_list(
        DEFAULT_ABLATION_NU_LIST if ablation_nu_list is None else list(ablation_nu_list)
    )

    print(f"Output: {ablation_output_dir}")
    print(f"Algorithms: {algorithm_ids}")
    print(f"Ablation nu list: {[format_nu(nu) for nu in nu_list]}")

    print("Building shared random feature space...")
    feature_space = build_shared_feature_space(
        N_s=cfg.N_s,
        N_u=cfg.N_u,
        gamma_s=cfg.gamma_s,
        gamma_u=cfg.gamma_u,
    )

    all_results: dict[float, dict[str, AlgorithmResult]] = {}
    records: list[AblationRecord] = []
    for nu in nu_list:
        print(f"\n=== Ablation nu: nu = {format_nu(nu)} ===")
        pair_cfg = replace(
            cfg,
            nu=nu,
            algorithms_to_run=list(algorithm_ids),
        )

        print("Building benchmark data for current nu...")
        benchmark = build_shared_benchmark(
            E=pair_cfg.E,
            nu=pair_cfg.nu,
            Q_train=pair_cfg.Q_train,
            Q_test=pair_cfg.Q_test,
            sampling_method=pair_cfg.sampling_method,
            body_force_batch_size=pair_cfg.body_force_batch_size,
            manufactured_solution=pair_cfg.manufactured_solution,
        )
        results = run_experiment(
            pair_cfg,
            print_table=False,
            plot_results=False,
            benchmark=benchmark,
            feature_space=feature_space,
        )

        method_results: dict[str, AlgorithmResult] = {}
        for algorithm_id, result in zip(algorithm_ids, results):
            algorithm_name = ALGORITHM_LABELS[algorithm_id]
            method_results[algorithm_name] = result
            records.append(
                AblationRecord(
                    algorithm_id=algorithm_id,
                    algorithm_name=algorithm_name,
                    nu=nu,
                    result=result,
                )
            )
        all_results[nu] = method_results

        del benchmark
        clear_experiment_memory()

    print("\nGenerating nu ablation plot...")
    plot_ablation_nu(
        all_results,
        algorithm_ids,
        str(ablation_output_dir / "ablation-nu.png"),
    )
    print_ablation_summary_table(records, algorithm_ids)
    return all_results


def main(
    cfg: LeastSquaresConfig | None = None,
    ablation_nu_list: Sequence[float] | None = None,
) -> None:
    """Script entrypoint."""

    run_ablation_nu(
        cfg=cfg,
        ablation_nu_list=ablation_nu_list,
    )


if __name__ == "__main__":
    main()
