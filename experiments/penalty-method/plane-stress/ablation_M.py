from __future__ import annotations

import gc
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from plane_stress import (
    ALGO_STYLE,
    AlgorithmResult,
    DEVICE,
    FeatureEvaluationData,
    MainConfig,
    SharedBenchmarkData,
    SharedComparisonConfig,
    SharedFeatureSpace,
    TOP_LEVEL_ALGORITHM_LABELS,
    VALID_TOP_LEVEL_ALGORITHMS,
    apply_shared_to_strong_config,
    apply_shared_to_weak_config,
    build_feature_evaluation_data,
    build_shared_benchmark,
    build_shared_feature_space,
    clear_cuda_cache,
    compute_plane_stress_parameters,
    compute_stress_voigt,
    configure_plotting,
    eval_exact_displacement,
    evaluate_feature_result,
    extract_scoped_algorithm_ids,
    make_default_main_config,
    print_result_summary,
    validate_algorithm_selection,
    validate_shared_comparison_config,
)
from projection import (
    apply_shared_to_projection_config,
    run_projection,
    validate_config as validate_projection_config,
)
from strong_form import (
    assemble_normal_equations,
    solve_eigh as solve_strong_eigh,
    solve_lstsq as solve_strong_lstsq,
    split_solution as split_strong_solution,
    validate_config as validate_strong_config,
)
from weak_form import (
    accumulate_boundary_gram as accumulate_weak_boundary_gram,
    accumulate_weak_form_moments,
    assemble_system as assemble_weak_system,
    solve_eigh as solve_weak_eigh,
    solve_lstsq as solve_weak_lstsq,
    validate_config as validate_weak_config,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "public" / "images" / "penalty-method" / "plane-stress" / "ablation" / "M"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_ABLATION_M_LIST = [100, 150, 200, 250, 300]


def clear_experiment_memory() -> None:
    """Release Python references and cached device memory between runs."""

    gc.collect()
    clear_cuda_cache()


def validate_ablation_list(ablation_M_list: Sequence[int]) -> list[int]:
    """Validate the synchronized feature-count ablation list."""

    if not ablation_M_list:
        raise ValueError("ablation_M_list must be non-empty.")

    validated_list: list[int] = []
    for M in ablation_M_list:
        if M <= 0:
            raise ValueError("All M values must be positive integers.")
        validated_list.append(int(M))
    return validated_list


def build_pair_feature_space(
    full_feature_space: SharedFeatureSpace,
    M: int,
) -> SharedFeatureSpace:
    """Slice the shared random feature spaces to one synchronized capacity point."""

    return SharedFeatureSpace(
        a_s=full_feature_space.a_s[:M],
        r_s=full_feature_space.r_s[:M],
        a_u=full_feature_space.a_u[:M],
        r_u=full_feature_space.r_u[:M],
        gamma_s=full_feature_space.gamma_s,
        gamma_u=full_feature_space.gamma_u,
    )


def resolve_feature_eval_batch_size(
    cfg: MainConfig,
    *,
    projection_enabled: bool,
    weak_enabled: bool,
    strong_enabled: bool,
) -> int:
    """Use the smallest enabled feature-method batch size for residual evaluation."""

    batch_sizes: list[int] = []
    if projection_enabled:
        if cfg.projection is None:
            raise RuntimeError("Projection config is required for feature evaluation.")
        batch_sizes.append(cfg.projection.assembly_batch_size)
    if weak_enabled:
        if cfg.weak is None:
            raise RuntimeError("Weak config is required for feature evaluation.")
        batch_sizes.append(cfg.weak.assembly_batch_size)
    if strong_enabled:
        if cfg.strong is None:
            raise RuntimeError("Strong config is required for feature evaluation.")
        batch_sizes.append(cfg.strong.assembly_batch_size)

    if not batch_sizes:
        raise RuntimeError("At least one feature-based algorithm is required.")
    return min(batch_sizes)


def run_projection_ablation(
    cfg: MainConfig,
    shared_cfg: SharedComparisonConfig,
    benchmark: SharedBenchmarkData,
    feature_space: SharedFeatureSpace,
    eval_data: FeatureEvaluationData,
    u_exact_train: torch.Tensor,
    sigma_exact_train: torch.Tensor,
) -> AlgorithmResult:
    """Run the projection baseline at one paired capacity point."""

    if cfg.projection is None:
        raise ValueError("MainConfig.projection is required when running projection.")

    projection_cfg = apply_shared_to_projection_config(cfg.projection, shared_cfg)
    validate_projection_config(projection_cfg)
    s, u, wall_time = run_projection(
        benchmark.x_int,
        feature_space.a_s,
        feature_space.r_s,
        projection_cfg.gamma_s,
        feature_space.a_u,
        feature_space.r_u,
        projection_cfg.gamma_u,
        u_exact_train,
        sigma_exact_train,
    )
    result = evaluate_feature_result("Projection", wall_time, s, u, eval_data)
    print_result_summary(result)
    return result


def run_weak_ablation(
    cfg: MainConfig,
    shared_cfg: SharedComparisonConfig,
    benchmark: SharedBenchmarkData,
    feature_space: SharedFeatureSpace,
    eval_data: FeatureEvaluationData,
    weak_algorithm_ids: Sequence[str],
) -> dict[str, AlgorithmResult]:
    """Run the selected weak-form solvers at one paired capacity point."""

    if cfg.weak is None:
        raise ValueError("MainConfig.weak is required when running weak-form algorithms.")

    weak_cfg = apply_shared_to_weak_config(cfg.weak, shared_cfg, weak_algorithm_ids)
    validate_weak_config(weak_cfg)

    gram_s, cross_u_grad_s, force_moment = accumulate_weak_form_moments(
        benchmark.x_int,
        benchmark.f_int,
        feature_space.a_s,
        feature_space.r_s,
        weak_cfg.gamma_s,
        feature_space.a_u,
        feature_space.r_u,
        weak_cfg.gamma_u,
        weak_cfg.assembly_batch_size,
    )
    gram_bc = accumulate_weak_boundary_gram(
        benchmark.x_bc,
        benchmark.w_bc,
        feature_space.a_u,
        feature_space.r_u,
        weak_cfg.gamma_u,
        weak_cfg.assembly_batch_size,
    )
    A, B, C, F = assemble_weak_system(
        gram_s,
        cross_u_grad_s,
        gram_bc,
        force_moment,
        benchmark.compliance_voigt,
        weak_cfg.lambda_bc,
    )

    del gram_s
    del cross_u_grad_s
    del force_moment
    del gram_bc
    clear_cuda_cache()

    results: dict[str, AlgorithmResult] = {}
    try:
        for algorithm_id in weak_algorithm_ids:
            if algorithm_id == "eigh":
                print("Running Weak (Eigh)...")
                s, u, wall_time = solve_weak_eigh(A, B, C, F, weak_cfg.eigh_rtol)
                label = "Weak (Eigh)"
            else:
                print("Running Weak (Lstsq)...")
                s, u, wall_time = solve_weak_lstsq(A, B, C, F)
                label = "Weak (Lstsq)"
            result = evaluate_feature_result(label, wall_time, s, u, eval_data)
            print_result_summary(result)
            results[label] = result
    finally:
        del A
        del B
        del C
        del F
        clear_cuda_cache()

    return results


def run_strong_ablation(
    cfg: MainConfig,
    shared_cfg: SharedComparisonConfig,
    benchmark: SharedBenchmarkData,
    feature_space: SharedFeatureSpace,
    eval_data: FeatureEvaluationData,
    strong_algorithm_ids: Sequence[str],
) -> dict[str, AlgorithmResult]:
    """Run the selected strong-form solvers at one paired capacity point."""

    if cfg.strong is None:
        raise ValueError("MainConfig.strong is required when running strong-form algorithms.")

    strong_cfg = apply_shared_to_strong_config(cfg.strong, shared_cfg, strong_algorithm_ids)
    validate_strong_config(strong_cfg)
    H, g = assemble_normal_equations(
        strong_cfg,
        benchmark.compliance_voigt,
        benchmark.x_int,
        benchmark.f_int,
        benchmark.x_bc,
        benchmark.w_bc,
        feature_space.a_s,
        feature_space.r_s,
        feature_space.a_u,
        feature_space.r_u,
    )
    clear_cuda_cache()

    results: dict[str, AlgorithmResult] = {}
    try:
        dim_s = 3 * (strong_cfg.M_s + 1)
        for algorithm_id in strong_algorithm_ids:
            if algorithm_id == "eigh":
                print("Running Strong (Eigh)...")
                z, wall_time = solve_strong_eigh(H, g, strong_cfg.eigh_rtol)
                label = "Strong (Eigh)"
            else:
                print("Running Strong (Lstsq)...")
                z, wall_time = solve_strong_lstsq(H, g)
                label = "Strong (Lstsq)"
            s, u = split_strong_solution(z, dim_s)
            result = evaluate_feature_result(label, wall_time, s, u, eval_data)
            print_result_summary(result)
            results[label] = result
    finally:
        del H
        del g
        clear_cuda_cache()

    return results


def plot_ablation_M(
    results: dict[int, dict[str, AlgorithmResult]],
    ordered_labels: Sequence[str],
    save_path: str,
) -> None:
    """Plot synchronized feature-count relative L2 errors across the selected algorithms."""

    if not results:
        return

    configure_plotting()
    M_list = list(results.keys())
    x_positions = np.arange(len(M_list), dtype=float)

    fig_width = max(10.0, 1.6 * len(M_list))
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 4.8))
    metric_specs = [
        (
            "rel_u",
            r"Displacement $\|\Phi^u - u_{ex}\|_{L^2} / \|u_{ex}\|_{L^2}$",
            "Relative $L^2$ error",
        ),
        (
            "rel_sigma",
            r"Stress $\|\Phi^\sigma - \sigma_{ex}\|_{L^2} / \|\sigma_{ex}\|_{L^2}$",
            "Relative $L^2$ error",
        ),
    ]

    for ax, (metric_name, title, ylabel) in zip(axes, metric_specs):
        for label in ordered_labels:
            values = np.array(
                [
                    getattr(results[M][label], metric_name) if label in results[M] else float("nan")
                    for M in M_list
                ],
                dtype=float,
            )
            valid = np.isfinite(values) & (values > 0.0)
            if not valid.any():
                continue

            style = ALGO_STYLE.get(label, {"color": "#4C78A8", "marker": "o", "linestyle": "-"})
            ax.semilogy(
                x_positions[valid],
                values[valid],
                marker=style["marker"],
                color=style["color"],
                linestyle=style.get("linestyle", "-"),
                linewidth=1.5,
                markersize=6,
                label=label,
            )

        ax.set_title(title)
        ax.set_xlabel(r"Feature count $M$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(M) for M in M_list])
        ax.grid(alpha=0.3, linestyle="--")
        if ax.lines:
            ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def print_summary_table(
    M_list: Sequence[int],
    results: dict[int, dict[str, AlgorithmResult]],
    ordered_labels: Sequence[str],
) -> None:
    """Print the final comparable summary table."""

    print("\n=== Ablation M Summary ===\n")
    print(
        f"| {'M':>6} | {'Algorithm':<16} | "
        f"{'rel_u':>12} | {'rel_sigma':>12} | {'Time(s)':>8} |"
    )
    print(
        f"|{'-' * 7}:|:{'-' * 17}|"
        f"{'-' * 13}:|{'-' * 13}:|{'-' * 9}:|"
    )

    for M in M_list:
        pair_results = results.get(M, {})
        for label in ordered_labels:
            metrics = pair_results.get(label)
            if metrics is None:
                continue
            print(
                f"| {M:>6} | {label:<16} | "
                f"{metrics.rel_u:>12.2e} | {metrics.rel_sigma:>12.2e} | "
                f"{metrics.wall_time:>8.2f} |"
            )


def run_ablation(
    cfg: MainConfig | None = None,
    ablation_M_list: Sequence[int] | None = None,
) -> dict[int, dict[str, AlgorithmResult]]:
    """Run the synchronized M ablation study and return all metrics."""

    cfg = make_default_main_config() if cfg is None else cfg
    shared_cfg = SharedComparisonConfig() if cfg.shared is None else cfg.shared
    validate_shared_comparison_config(shared_cfg)

    selected_algorithm_ids = validate_algorithm_selection(
        cfg.algorithms_to_run,
        VALID_TOP_LEVEL_ALGORITHMS,
    )
    ordered_labels = [
        TOP_LEVEL_ALGORITHM_LABELS[algorithm_id]
        for algorithm_id in selected_algorithm_ids
    ]

    ablation_M_list = DEFAULT_ABLATION_M_LIST if ablation_M_list is None else list(ablation_M_list)
    M_list = validate_ablation_list(ablation_M_list)

    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Algorithms: {selected_algorithm_ids}")
    print(f"Ablation M list: {M_list}")

    print("Building shared benchmark data...")
    benchmark = build_shared_benchmark(
        E=shared_cfg.E,
        nu=shared_cfg.nu,
        Q_int=shared_cfg.Q_int,
        Q_bc=shared_cfg.Q_bc,
        Q_test=shared_cfg.Q_test,
        sampling_method=shared_cfg.sampling_method,
        body_force_batch_size=shared_cfg.body_force_batch_size,
        interior_seed=shared_cfg.interior_seed,
        boundary_seed=shared_cfg.boundary_seed,
        test_seed=shared_cfg.test_seed,
    )

    feature_algorithm_ids = {
        "projection",
        "weak(eigh)",
        "weak(lstsq)",
        "strong(eigh)",
        "strong(lstsq)",
    }
    use_feature_algorithms = any(
        algorithm_id in feature_algorithm_ids
        for algorithm_id in selected_algorithm_ids
    )

    full_feature_space: SharedFeatureSpace | None = None
    if use_feature_algorithms:
        print("Generating full shared random feature spaces...")
        full_feature_space = build_shared_feature_space(
            M_s=max(M_list),
            M_u=max(M_list),
            gamma_s=shared_cfg.gamma_s,
            gamma_u=shared_cfg.gamma_u,
            stress_feature_seed=shared_cfg.stress_feature_seed,
            disp_feature_seed=shared_cfg.disp_feature_seed,
        )

    projection_train_fields: tuple[torch.Tensor, torch.Tensor] | None = None
    if "projection" in selected_algorithm_ids:
        mu, lambda_plane = compute_plane_stress_parameters(shared_cfg.E, shared_cfg.nu)
        print(
            "Computing exact projection targets with "
            f"mu={mu:.4f}, lambda_plane={lambda_plane:.4f}..."
        )
        projection_train_fields = (
            eval_exact_displacement(benchmark.x_int),
            compute_stress_voigt(benchmark.x_int, mu, lambda_plane),
        )

    weak_algorithm_ids = extract_scoped_algorithm_ids(selected_algorithm_ids, "weak")
    strong_algorithm_ids = extract_scoped_algorithm_ids(selected_algorithm_ids, "strong")
    feature_eval_batch_size = None
    if use_feature_algorithms:
        feature_eval_batch_size = resolve_feature_eval_batch_size(
            cfg,
            projection_enabled="projection" in selected_algorithm_ids,
            weak_enabled=bool(weak_algorithm_ids),
            strong_enabled=bool(strong_algorithm_ids),
        )

    all_results: dict[int, dict[str, AlgorithmResult]] = {}
    for M in M_list:
        print(f"\n{'=' * 72}")
        print(f"=== Ablation M: M_s = M_u = {M} ===")
        print(f"{'=' * 72}")

        pair_shared_cfg = replace(shared_cfg, M_s=M, M_u=M)
        pair_results: dict[str, AlgorithmResult] = {}

        pair_feature_space: SharedFeatureSpace | None = None
        pair_eval_data: FeatureEvaluationData | None = None
        if full_feature_space is not None:
            if feature_eval_batch_size is None:
                raise RuntimeError("Feature evaluation batch size must be resolved.")
            pair_feature_space = build_pair_feature_space(full_feature_space, M)
            pair_eval_data = build_feature_evaluation_data(
                benchmark,
                pair_feature_space,
                feature_eval_batch_size,
            )

        if "projection" in selected_algorithm_ids:
            if pair_feature_space is None or pair_eval_data is None or projection_train_fields is None:
                raise RuntimeError("Projection run requires shared feature space and exact targets.")
            print("Running Projection...")
            pair_results["Projection"] = run_projection_ablation(
                cfg,
                pair_shared_cfg,
                benchmark,
                pair_feature_space,
                pair_eval_data,
                projection_train_fields[0],
                projection_train_fields[1],
            )

        if weak_algorithm_ids:
            if pair_feature_space is None or pair_eval_data is None:
                raise RuntimeError("Weak-form ablation requires shared feature spaces.")
            pair_results.update(
                run_weak_ablation(
                    cfg,
                    pair_shared_cfg,
                    benchmark,
                    pair_feature_space,
                    pair_eval_data,
                    weak_algorithm_ids,
                )
            )

        if strong_algorithm_ids:
            if pair_feature_space is None or pair_eval_data is None:
                raise RuntimeError("Strong-form ablation requires shared feature spaces.")
            pair_results.update(
                run_strong_ablation(
                    cfg,
                    pair_shared_cfg,
                    benchmark,
                    pair_feature_space,
                    pair_eval_data,
                    strong_algorithm_ids,
                )
            )

        all_results[M] = pair_results

        del pair_feature_space
        del pair_eval_data
        clear_experiment_memory()

    if all_results:
        print("\nGenerating plots...")
        plot_ablation_M(
            all_results,
            ordered_labels,
            str(OUTPUT_DIR / "ablation-M.png"),
        )
        print_summary_table(M_list, all_results, ordered_labels)

    return all_results


def main(
    cfg: MainConfig | None = None,
    ablation_M_list: Sequence[int] | None = None,
) -> None:
    """Script entrypoint."""

    run_ablation(
        cfg=cfg,
        ablation_M_list=ablation_M_list,
    )


if __name__ == "__main__":
    main()
