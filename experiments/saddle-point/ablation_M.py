"""
Ablation: feature count M.

Generates per-M convergence plots and summary comparison.
"""

import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from saddle_point import (
    # Config
    Config,
    # Device and data types
    DEVICE, DTYPE, BASE_SEED,
    # Material
    compute_lame_constants, build_compliance_matrix,
    # Exact solution
    eval_exact_displacement, compute_stress_voigt, compute_body_force,
    # Features
    generate_features, eval_features, eval_feature_grads, eval_zeta,
    # System
    assemble_system,
    # Algorithms
    run_all_algorithms,
    # Plotting
    ALGO_STYLE, plot_kkt_convergence, plot_l2_convergence,
)

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "public" / "images" / "saddle-point" / "ablation" / "M"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_ablation_M(results, save_path):
    """Line plot comparing final metrics across M values for all methods.

    Args:
        results: { M: { method_name: {"r_s", "r_u", "rel_u", "rel_sigma"} } }
        save_path: output file path
    """
    M_values = sorted(results.keys())
    method_names = list(results[M_values[0]].keys())

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    metric_keys = ["r_s", "r_u", "rel_u", "rel_sigma"]
    titles = [
        r"$\|r_s\|_2$", r"$\|r_u\|_2$",
        "Displacement error", "Stress error",
    ]
    ylabels = [
        r"$\|r_s\|_2$", r"$\|r_u\|_2$",
        "Relative $L^2$ error", "Relative $L^2$ error",
    ]

    for ax, mkey, title, ylabel in zip(
        axes.flat, metric_keys, titles, ylabels
    ):
        for method in method_names:
            vals = []
            for M in M_values:
                v = results[M].get(method, {}).get(mkey, float("nan"))
                vals.append(v)
            vals = np.array(vals, dtype=float)
            valid = np.isfinite(vals) & (vals > 0)
            if valid.any():
                sty = ALGO_STYLE.get(method, {"color": "gray", "marker": "x"})
                ax.semilogy(
                    np.array(M_values)[valid], vals[valid],
                    marker=sty["marker"], color=sty["color"],
                    linestyle=sty.get("linestyle", "-"),
                    label=method, linewidth=1.5, markersize=6,
                )
        ax.set_xlabel("Feature count $M$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(M_values)
        ax.set_xticklabels([str(m) for m in M_values])
        ax.legend()
        ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # --- Configuration (override defaults as needed) ---
    # Override example: cfg = Config(M=512, K_max=10000)
    cfg = Config(K_max=100000)
    ablation_M_list = [64, 128, 256, 512, 1024]

    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Ablation M list: {ablation_M_list}")

    # --- Prepare common data ---
    mu, lam = compute_lame_constants(cfg.E, cfg.nu)
    S = build_compliance_matrix(cfg.E, cfg.nu)

    max_M = max(ablation_M_list)
    a_full, r_full = generate_features(max_M, seed=BASE_SEED)

    print(f"Sampling {cfg.Q_train} training points...")
    torch.manual_seed(BASE_SEED + 1)
    x_train = torch.rand(cfg.Q_train, 3, dtype=DTYPE, device=DEVICE)
    f_train = compute_body_force(x_train, mu, lam)
    zeta_train = eval_zeta(x_train)

    print(f"Sampling {cfg.Q_test} test points...")
    torch.manual_seed(BASE_SEED + 2)
    x_test = torch.rand(cfg.Q_test, 3, dtype=DTYPE, device=DEVICE)
    zeta_test = eval_zeta(x_test)
    u_exact = eval_exact_displacement(x_test)
    sigma_exact = compute_stress_voigt(x_test, mu, lam)

    # --- Ablation loop ---
    ablation_M_results = {}

    for M_abl in ablation_M_list:
        print(f"\n{'='*60}")
        print(f"=== Ablation: M={M_abl} ===")
        print(f"{'='*60}")

        # Feature subset
        a_abl, r_abl = a_full[:M_abl], r_full[:M_abl]
        xi_abl = eval_features(x_train, a_abl, r_abl, cfg.gamma)
        grad_xi_abl = eval_feature_grads(x_train, a_abl, r_abl, cfg.gamma)

        # Assemble system
        A_abl, B_abl, F_abl = assemble_system(xi_abl, grad_xi_abl, S, f_train, zeta_train)
        print(f"  A: {A_abl.shape}, B: {B_abl.shape}, F: {F_abl.shape}")

        # Test features
        xi_test_abl = eval_features(x_test, a_abl, r_abl, cfg.gamma)
        xi_hat_test_abl = zeta_test.unsqueeze(1) * xi_test_abl

        # Run all algorithms (record full history)
        abl_results = run_all_algorithms(
            A_abl, B_abl, F_abl, xi_hat_test_abl, xi_test_abl,
            u_exact, sigma_exact,
            K_max=cfg.K_max, eval_every=cfg.eval_every, rho=cfg.rho,
            eta_admm=cfg.eta_admm, beta_adam=cfg.beta_adam,
            eta_u_uzawa=cfg.eta_u_uzawa, eta_s_ah=cfg.eta_s_ah, eta_u_ah=cfg.eta_u_ah,
        )

        # Plot KKT convergence
        print("\nGenerating plots...")
        iter_labels = ["ADMM", "Uzawa", "Arrow-Hurwicz"]
        iter_histories = [abl_results[name]["history"] for name in iter_labels]
        plot_kkt_convergence(
            iter_histories, iter_labels,
            str(OUTPUT_DIR / f"kkt-convergence_M={M_abl}.png"),
        )

        # Plot L2 error convergence
        direct_l2 = {
            "rel_u": abl_results["Direct"]["history"]["rel_u"][-1],
            "rel_sigma": abl_results["Direct"]["history"]["rel_sigma"][-1],
        }
        plot_l2_convergence(
            iter_histories, iter_labels,
            str(OUTPUT_DIR / f"l2-error-convergence_M={M_abl}.png"),
            direct_l2=direct_l2,
        )

        # Collect summary data
        ablation_M_results[M_abl] = {}
        for method in ["Direct", "ADMM", "Uzawa", "Arrow-Hurwicz"]:
            h = abl_results[method]["history"]
            ablation_M_results[M_abl][method] = {
                "r_s": h["r_s"][-1],
                "r_u": h["r_u"][-1],
                "rel_u": h["rel_u"][-1],
                "rel_sigma": h["rel_sigma"][-1],
            }

    # --- Summary plot ---
    if ablation_M_results:
        plot_ablation_M(ablation_M_results, str(OUTPUT_DIR / "ablation-M.png"))

    # --- Print summary table (Markdown-compatible) ---
    print("\n=== Ablation M: Summary ===\n")
    print(f"| {'M':>6} | {'Algorithm':<14} | {'rel_u':>12} | {'rel_sigma':>12} |")
    print(f"|{'-'*7}:|:{'-'*14}|{'-'*13}:|{'-'*13}:|")
    for M_abl in ablation_M_list:
        for method in ["Direct", "ADMM", "Uzawa", "Arrow-Hurwicz"]:
            d = ablation_M_results[M_abl][method]
            print(f"| {M_abl:>6} | {method:<14} | {d['rel_u']:>12.2e} | {d['rel_sigma']:>12.2e} |")
