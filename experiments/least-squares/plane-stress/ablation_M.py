from __future__ import annotations

import gc
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from plane_stress import (
    ALGO_STYLE,
    OUTPUT_DIR,
    AlgorithmResult,
    LeastSquaresConfig,
    SharedFeatureSpace,
    VALID_ALGORITHMS,
    build_shared_benchmark,
    build_shared_feature_space,
    clear_cuda_cache,
    configure_plotting,
    print_aligned_markdown_table,
    run_experiment,
    validate_algorithm_selection,
)


ABLATION_OUTPUT_DIR = OUTPUT_DIR / "ablation" / "M"
ABLATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_ABLATION_M_LIST = [200, 400, 600, 800, 1000]
ALGORITHM_LABELS = {
    "eigh": "LS (Eigh)",
    "lstsq": "LS (Lstsq)",
}


@dataclass(frozen=True)
class AblationRecord:
    """One completed solver-capacity comparison record."""

    algorithm_id: str
    algorithm_name: str
    M: int
    result: AlgorithmResult


def clear_experiment_memory() -> None:
    """Release CPU references and cached CUDA memory between runs."""

    gc.collect()
    clear_cuda_cache()


def validate_ablation_list(ablation_M_list: Sequence[int]) -> list[int]:
    """Validate and sort the ablation feature-count list."""

    if not ablation_M_list:
        raise ValueError("ablation_M_list must be non-empty.")

    seen: set[int] = set()
    validated_list: list[int] = []
    for M in ablation_M_list:
        M_int = int(M)
        if M_int <= 0:
            raise ValueError("All M values must be positive integers.")
        if M_int in seen:
            raise ValueError(f"Duplicate M value detected: {M_int}")
        seen.add(M_int)
        validated_list.append(M_int)
    return sorted(validated_list)


def build_pair_feature_space(
    full_feature_space: SharedFeatureSpace,
    M: int,
) -> SharedFeatureSpace:
    """Slice one synchronized feature-space pair from the shared maximum space."""

    return SharedFeatureSpace(
        a_s=full_feature_space.a_s[:M],
        r_s=full_feature_space.r_s[:M],
        a_u=full_feature_space.a_u[:M],
        r_u=full_feature_space.r_u[:M],
        gamma_s=full_feature_space.gamma_s,
        gamma_u=full_feature_space.gamma_u,
    )


def sort_ablation_records(
    records: list[AblationRecord],
    algorithm_ids: list[str],
) -> list[AblationRecord]:
    """Group rows by solver, then by ascending M."""

    algorithm_order = {
        algorithm_id: index for index, algorithm_id in enumerate(algorithm_ids)
    }
    return sorted(
        records,
        key=lambda record: (
            algorithm_order[record.algorithm_id],
            record.M,
        ),
    )


def print_ablation_summary_table(
    records: list[AblationRecord],
    algorithm_ids: list[str],
) -> None:
    """Print the M ablation summary grouped by method."""

    ordered_records = sort_ablation_records(records, algorithm_ids)
    if not ordered_records:
        return

    headers = (
        "Method",
        "M",
        "‖Φ^u-u‖",
        "‖Φ^σ-σ‖",
        "‖div(Φ^σ-σ)‖",
        "Time(s)",
    )
    rows = [
        (
            record.algorithm_name,
            str(record.M),
            f"{record.result.u_l2_error:.2e}",
            f"{record.result.sigma_l2_error:.2e}",
            f"{record.result.div_sigma_l2_error:.2e}",
            f"{record.result.wall_time:.2f}",
        )
        for record in ordered_records
    ]
    print_aligned_markdown_table(
        title="M Ablation Summary",
        headers=headers,
        rows=rows,
        alignments=("left", "center", "center", "center", "center", "center"),
    )


def plot_ablation_M(
    results: dict[int, dict[str, AlgorithmResult]],
    algorithm_ids: list[str],
    save_path: str,
) -> None:
    """Plot absolute L2 error versus M for each configured solver."""

    if not results:
        print(f"  Skipped: {save_path} (no results to plot)")
        return

    configure_plotting()
    M_list = sorted(results.keys())
    x_positions = np.arange(len(M_list), dtype=float)

    fig_width = max(10.0, 1.6 * len(M_list))
    fig, axes = plt.subplots(1, 3, figsize=(max(13.0, fig_width + 2.0), 4.8))
    metric_specs = [
        (
            "u_l2_error",
            r"$\|\Phi^u - u_{ex}\|_0$",
            "$L^2$ error",
        ),
        (
            "sigma_l2_error",
            r"$\|\Phi^\sigma - \sigma_{ex}\|_0$",
            "$L^2$ error",
        ),
        (
            "div_sigma_l2_error",
            r"$\|\operatorname{div}(\Phi^\sigma - \sigma_{ex})\|_0$",
            "$L^2$ error",
        ),
    ]

    for ax, (metric_name, title, ylabel) in zip(axes, metric_specs):
        for algorithm_id in algorithm_ids:
            algorithm_name = ALGORITHM_LABELS[algorithm_id]
            values = np.array(
                [
                    getattr(results[M][algorithm_name], metric_name)
                    if algorithm_name in results[M]
                    else float("nan")
                    for M in M_list
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
        ax.set_xlabel(r"Feature count $M$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(M) for M in M_list])
        ax.grid(alpha=0.3, linestyle="--")
        if ax.lines:
            ax.legend()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def run_ablation_M(
    cfg: LeastSquaresConfig | None = None,
    ablation_M_list: Sequence[int] | None = None,
) -> dict[int, dict[str, AlgorithmResult]]:
    """Run the synchronized M ablation study and return all metrics."""

    cfg = LeastSquaresConfig() if cfg is None else cfg
    algorithm_ids = validate_algorithm_selection(cfg.algorithms_to_run, VALID_ALGORITHMS)
    M_list = validate_ablation_list(
        DEFAULT_ABLATION_M_LIST if ablation_M_list is None else list(ablation_M_list)
    )

    print(f"Output: {ABLATION_OUTPUT_DIR}")
    print(f"Algorithms: {algorithm_ids}")
    print(f"Ablation M list: {M_list}")

    print("Building shared benchmark data...")
    benchmark = build_shared_benchmark(
        E=cfg.E,
        nu=cfg.nu,
        Q_train=cfg.Q_train,
        Q_test=cfg.Q_test,
        sampling_method=cfg.sampling_method,
        body_force_batch_size=cfg.body_force_batch_size,
    )

    print("Building full shared random feature space...")
    full_feature_space = build_shared_feature_space(
        M_s=max(M_list),
        M_u=max(M_list),
        gamma_s=cfg.gamma_s,
        gamma_u=cfg.gamma_u,
    )

    all_results: dict[int, dict[str, AlgorithmResult]] = {}
    records: list[AblationRecord] = []
    for M in M_list:
        print(f"\n=== Ablation M: M_s = M_u = {M} ===")
        pair_cfg = replace(
            cfg,
            M_s=M,
            M_u=M,
            algorithms_to_run=list(algorithm_ids),
        )
        pair_feature_space = build_pair_feature_space(full_feature_space, M)
        results = run_experiment(
            pair_cfg,
            print_table=False,
            plot_results=False,
            benchmark=benchmark,
            feature_space=pair_feature_space,
        )

        method_results: dict[str, AlgorithmResult] = {}
        for algorithm_id, result in zip(algorithm_ids, results):
            algorithm_name = ALGORITHM_LABELS[algorithm_id]
            method_results[algorithm_name] = result
            records.append(
                AblationRecord(
                    algorithm_id=algorithm_id,
                    algorithm_name=algorithm_name,
                    M=M,
                    result=result,
                )
            )
        all_results[M] = method_results

        del pair_feature_space
        clear_experiment_memory()

    print("\nGenerating M ablation plot...")
    plot_ablation_M(
        all_results,
        algorithm_ids,
        str(ABLATION_OUTPUT_DIR / "ablation-M.png"),
    )
    print_ablation_summary_table(records, algorithm_ids)
    return all_results


def main(
    cfg: LeastSquaresConfig | None = None,
    ablation_M_list: Sequence[int] | None = None,
) -> None:
    """Script entrypoint."""

    run_ablation_M(
        cfg=cfg,
        ablation_M_list=ablation_M_list,
    )


if __name__ == "__main__":
    main()
