"""
Ablation: feature count with synchronized split spaces.

This script varies M_s = M_u while keeping the stress and displacement feature
spaces independent.
"""

import gc
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from saddle_point import (
    ALGO_STYLE,
    BASE_SEED,
    DEVICE,
    DISP_SEED,
    DTYPE,
    STRESS_SEED,
    Config,
    activate_displacement_features,
    assemble_system_in_batches,
    build_compliance_matrix,
    compute_body_force,
    compute_lame_constants,
    compute_stress_voigt,
    eval_exact_displacement,
    eval_features,
    eval_zeta,
    generate_features,
    get_iterative_plot_data,
    get_l2_plot_data,
    get_summary_labels,
    plot_kkt_convergence,
    plot_l2_convergence,
    run_all_algorithms,
    sample_boundary_points,
    sample_points,
    validate_algorithms_to_run,
    validate_config,
)


# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "public" / "images" / "saddle-point" / "ablation" / "M"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clear_experiment_memory() -> None:
    """Release CPU references and cached CUDA memory between ablation runs."""

    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def collect_summary_metrics(
    results: dict[str, dict[str, object]],
) -> dict[str, dict[str, float]]:
    """Extract scalar summary metrics without retaining tensor results."""

    summary: dict[str, dict[str, float]] = {}
    for method in get_summary_labels(results):
        history = results[method]["history"]
        summary[method] = {
            "r_c": history["r_c"][-1],
            "r_e": history["r_e"][-1],
            "rel_u": history["rel_u"][-1],
            "rel_sigma": history["rel_sigma"][-1],
        }
    return summary


def plot_ablation_M(results: dict[int, dict[str, dict[str, float]]], save_path: str) -> None:
    """Line plot comparing final metrics across M values for all methods."""

    M_values = sorted(results.keys())
    method_names = list(results[M_values[0]].keys())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metric_specs = [
        ("r_c", r"$\|r_c\|_2$", r"$\|r_c\|_2$"),
        ("r_e", r"$\|r_e\|_2$", r"$\|r_e\|_2$"),
        ("rel_u", "Displacement error", "Relative $L^2$ error"),
        ("rel_sigma", "Stress error", "Relative $L^2$ error"),
    ]

    axes_flat = list(axes.flat)
    for ax, (metric_key, title, ylabel) in zip(axes_flat, metric_specs):
        for method in method_names:
            vals = np.array(
                [results[M].get(method, {}).get(metric_key, float("nan")) for M in M_values],
                dtype=float,
            )
            valid = np.isfinite(vals) & (vals > 0)
            if not valid.any():
                continue

            style = ALGO_STYLE.get(method, {"color": "gray", "marker": "x"})
            ax.semilogy(
                np.array(M_values)[valid],
                vals[valid],
                marker=style["marker"],
                color=style["color"],
                linestyle=style.get("linestyle", "-"),
                label=method,
                linewidth=1.5,
                markersize=6,
            )

        ax.set_xlabel(r"Feature count $M_s = M_u$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(M_values)
        ax.set_xticklabels([str(M) for M in M_values])
        ax.legend()
        ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def run_single_M_ablation(
    M_abl: int,
    cfg: Config,
    *,
    a_s_full: torch.Tensor,
    r_s_full: torch.Tensor,
    a_u_full: torch.Tensor,
    r_u_full: torch.Tensor,
    x_train: torch.Tensor,
    f_train: torch.Tensor,
    zeta_train: torch.Tensor,
    compliance_voigt: torch.Tensor,
    x_test: torch.Tensor,
    zeta_test: torch.Tensor,
    u_exact: torch.Tensor,
    sigma_exact: torch.Tensor,
    selected_algorithm_ids: list[str],
    projection_enabled: bool,
    u_exact_train: torch.Tensor | None = None,
    sigma_exact_train: torch.Tensor | None = None,
    x_bc: torch.Tensor | None = None,
    w_bc: torch.Tensor | None = None,
    zeta_bc: torch.Tensor | None = None,
) -> dict[str, dict[str, float]]:
    """Run one feature-count ablation while releasing large tensors eagerly."""

    print(f"\n{'=' * 60}")
    print(f"=== Ablation: M_s = M_u = {M_abl} ===")
    print(f"{'=' * 60}")

    a_s_abl, r_s_abl = a_s_full[:M_abl], r_s_full[:M_abl]
    a_u_abl, r_u_abl = a_u_full[:M_abl], r_u_full[:M_abl]

    xi_s_train_abl = None
    xi_u_train_abl = None
    xi_u_active_train_abl = None
    A_abl = None
    B_abl = None
    C_abl = None
    F_abl = None
    xi_s_test_abl = None
    xi_u_test_abl = None
    xi_u_active_test_abl = None
    abl_results = None
    iter_labels = None
    iter_histories = None
    l2_labels = None
    l2_histories = None

    try:
        if projection_enabled:
            xi_s_train_abl = eval_features(x_train, a_s_abl, r_s_abl, cfg.gamma_s)
            xi_u_train_abl = eval_features(x_train, a_u_abl, r_u_abl, cfg.gamma_u)
            xi_u_active_train_abl = activate_displacement_features(
                xi_u_train_abl,
                zeta_train,
            )
            del xi_u_train_abl
            xi_u_train_abl = None

        A_abl, B_abl, C_abl, F_abl = assemble_system_in_batches(
            x_train,
            f_train,
            a_s_abl,
            r_s_abl,
            cfg.gamma_s,
            a_u_abl,
            r_u_abl,
            cfg.gamma_u,
            compliance_voigt,
            zeta_train,
            cfg.assembly_batch_size,
            x_bc=x_bc,
            w_bc=w_bc,
            zeta_bc=zeta_bc,
            lambda_bc=cfg.lambda_bc if cfg.use_penalty else 0.0,
        )
        print(
            f"  A: {tuple(A_abl.shape)}, B: {tuple(B_abl.shape)}, "
            f"C: {tuple(C_abl.shape)}, F: {tuple(F_abl.shape)}"
        )

        xi_s_test_abl = eval_features(x_test, a_s_abl, r_s_abl, cfg.gamma_s)
        xi_u_test_abl = eval_features(x_test, a_u_abl, r_u_abl, cfg.gamma_u)
        xi_u_active_test_abl = activate_displacement_features(xi_u_test_abl, zeta_test)
        del xi_u_test_abl
        xi_u_test_abl = None

        abl_results = run_all_algorithms(
            A_abl,
            B_abl,
            C_abl,
            F_abl,
            xi_u_active_test_abl,
            xi_s_test_abl,
            u_exact,
            sigma_exact,
            algorithms_to_run=selected_algorithm_ids,
            K_max=cfg.K_max,
            eval_every=cfg.eval_every,
            rho=cfg.rho,
            eta_gda=cfg.eta_gda,
            beta_adam=cfg.beta_adam,
            eta_u_uzawa=cfg.eta_u_uzawa,
            eta_s_ah=cfg.eta_s_ah,
            eta_u_ah=cfg.eta_u_ah,
            xi_s_train=xi_s_train_abl if projection_enabled else None,
            xi_u_active_train=xi_u_active_train_abl if projection_enabled else None,
            u_exact_train=u_exact_train,
            sigma_exact_train=sigma_exact_train,
            eigh_rtol=cfg.eigh_rtol,
        )

        summary = collect_summary_metrics(abl_results)
        iter_labels, iter_histories = get_iterative_plot_data(abl_results)
        if iter_labels:
            l2_labels, l2_histories = get_l2_plot_data(abl_results)

        xi_s_train_abl = None
        xi_u_active_train_abl = None
        A_abl = None
        B_abl = None
        C_abl = None
        F_abl = None
        xi_s_test_abl = None
        xi_u_active_test_abl = None
        abl_results = None
        clear_experiment_memory()

        if iter_labels:
            print("\nGenerating plots...")
            plot_kkt_convergence(
                iter_histories,
                iter_labels,
                str(OUTPUT_DIR / f"kkt-convergence_M={M_abl}.png"),
            )
            plot_l2_convergence(
                l2_histories,
                l2_labels,
                str(OUTPUT_DIR / f"l2-error-convergence_M={M_abl}.png"),
            )
        else:
            print("Skipping convergence plots because no iterative algorithms were selected.")

        return summary
    finally:
        xi_s_train_abl = None
        xi_u_train_abl = None
        xi_u_active_train_abl = None
        A_abl = None
        B_abl = None
        C_abl = None
        F_abl = None
        xi_s_test_abl = None
        xi_u_test_abl = None
        xi_u_active_test_abl = None
        abl_results = None
        clear_experiment_memory()


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    cfg = Config()
    selected_algorithm_ids = validate_algorithms_to_run(cfg.algorithms_to_run)
    validate_config(cfg, selected_algorithm_ids)
    projection_enabled = "projection" in selected_algorithm_ids
    ablation_M_list = [200, 400, 600, 800, 1000]

    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Ablation M list: {ablation_M_list}")

    mu, lam = compute_lame_constants(cfg.E, cfg.nu)
    compliance_voigt = build_compliance_matrix(cfg.E, cfg.nu)

    max_M = max(ablation_M_list)
    a_s_full, r_s_full = generate_features(max_M, seed=STRESS_SEED)
    a_u_full, r_u_full = generate_features(max_M, seed=DISP_SEED)

    print(f"Sampling {cfg.Q_int} training points...")
    x_train = sample_points(cfg.Q_int, method=cfg.sampling_method, seed=BASE_SEED + 1)
    f_train = compute_body_force(x_train, mu, lam, batch_size=cfg.body_force_batch_size)
    zeta_train = (
        eval_zeta(x_train)
        if cfg.use_zeta
        else torch.ones(x_train.shape[0], dtype=DTYPE, device=DEVICE)
    )

    u_exact_train = None
    sigma_exact_train = None
    if projection_enabled:
        u_exact_train = eval_exact_displacement(x_train)
        sigma_exact_train = compute_stress_voigt(x_train, mu, lam)

    x_bc = None
    w_bc = None
    zeta_bc = None
    if cfg.use_penalty:
        print(f"Sampling {cfg.Q_bc} boundary points...")
        x_bc, w_bc = sample_boundary_points(
            cfg.Q_bc,
            method=cfg.sampling_method,
            seed=BASE_SEED + 2,
        )
        zeta_bc = (
            eval_zeta(x_bc)
            if cfg.use_zeta
            else torch.ones(x_bc.shape[0], dtype=DTYPE, device=DEVICE)
        )

    print(f"Sampling {cfg.Q_test} test points...")
    x_test = sample_points(cfg.Q_test, method=cfg.sampling_method, seed=BASE_SEED + 3)
    zeta_test = (
        eval_zeta(x_test)
        if cfg.use_zeta
        else torch.ones(x_test.shape[0], dtype=DTYPE, device=DEVICE)
    )
    u_exact = eval_exact_displacement(x_test)
    sigma_exact = compute_stress_voigt(x_test, mu, lam)

    ablation_M_results: dict[int, dict[str, dict[str, float]]] = {}

    for M_abl in ablation_M_list:
        ablation_M_results[M_abl] = run_single_M_ablation(
            M_abl,
            cfg,
            a_s_full=a_s_full,
            r_s_full=r_s_full,
            a_u_full=a_u_full,
            r_u_full=r_u_full,
            x_train=x_train,
            f_train=f_train,
            zeta_train=zeta_train,
            compliance_voigt=compliance_voigt,
            x_test=x_test,
            zeta_test=zeta_test,
            u_exact=u_exact,
            sigma_exact=sigma_exact,
            selected_algorithm_ids=selected_algorithm_ids,
            projection_enabled=projection_enabled,
            u_exact_train=u_exact_train,
            sigma_exact_train=sigma_exact_train,
            x_bc=x_bc,
            w_bc=w_bc,
            zeta_bc=zeta_bc,
        )

    if ablation_M_results:
        plot_ablation_M(ablation_M_results, str(OUTPUT_DIR / "ablation-M.png"))

    print("\n=== Ablation M: Summary ===\n")
    summary_methods = (
        list(next(iter(ablation_M_results.values())).keys())
        if ablation_M_results
        else []
    )
    print(
        f"| {'M':>6} | {'Algorithm':<14} | {'||r_c||':>10} | {'||r_e||':>10} | "
        f"{'rel_u':>12} | {'rel_sigma':>12} |"
    )
    print(
        f"|{'-' * 7}:|:{'-' * 15}|{'-' * 11}:|{'-' * 11}:|"
        f"{'-' * 13}:|{'-' * 13}:|"
    )
    for M_abl in ablation_M_list:
        for method in summary_methods:
            item = ablation_M_results[M_abl][method]
            print(
                f"| {M_abl:>6} | {method:<14} | {item['r_c']:>10.2e} | "
                f"{item['r_e']:>10.2e} | {item['rel_u']:>12.2e} | "
                f"{item['rel_sigma']:>12.2e} |"
            )
