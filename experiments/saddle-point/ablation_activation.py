"""
Ablation: activation function type.

Generates per-activation convergence plots and summary comparison.
Independent implementation of feature evaluation for each activation function.
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
    # Features (only generate_features, not eval_features/eval_feature_grads)
    generate_features, eval_zeta,
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
OUTPUT_DIR = PROJECT_ROOT / "public" / "images" / "saddle-point" / "ablation" / "activation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Activation functions: features and gradients
# ---------------------------------------------------------------------------

def eval_tanh_features(x: torch.Tensor, a: torch.Tensor, r: torch.Tensor,
                       gamma: float) -> torch.Tensor:
    """Feature function: ξ_m(x) = tanh(γ(a_m^T x + r_m))"""
    Q = x.shape[0]
    z = gamma * (x @ a.T + r.unsqueeze(0))
    xi = torch.tanh(z)
    ones = torch.ones(Q, 1, dtype=DTYPE, device=DEVICE)
    return torch.cat([ones, xi], dim=1)


def eval_tanh_feature_grads(x: torch.Tensor, a: torch.Tensor, r: torch.Tensor,
                            gamma: float) -> torch.Tensor:
    """tanh feature gradient: ∇ξ_m = γ(1 - tanh²(z)) a_m"""
    Q = x.shape[0]
    z = gamma * (x @ a.T + r.unsqueeze(0))
    dtanh = 1.0 - torch.tanh(z) ** 2
    grad_xi = gamma * dtanh.unsqueeze(2) * a.unsqueeze(0)
    zeros = torch.zeros(Q, 1, 3, dtype=DTYPE, device=DEVICE)
    return torch.cat([zeros, grad_xi], dim=1)


def eval_sigmoid_features(x: torch.Tensor, a: torch.Tensor, r: torch.Tensor,
                          gamma: float) -> torch.Tensor:
    """Feature function: ξ_m(x) = σ(γ(a_m^T x + r_m)) where σ(z) = 1/(1+e^(-z))"""
    Q = x.shape[0]
    z = gamma * (x @ a.T + r.unsqueeze(0))
    xi = torch.sigmoid(z)
    ones = torch.ones(Q, 1, dtype=DTYPE, device=DEVICE)
    return torch.cat([ones, xi], dim=1)


def eval_sigmoid_feature_grads(x: torch.Tensor, a: torch.Tensor, r: torch.Tensor,
                               gamma: float) -> torch.Tensor:
    """sigmoid feature gradient: ∇ξ_m = γ σ(z)(1-σ(z)) a_m"""
    Q = x.shape[0]
    z = gamma * (x @ a.T + r.unsqueeze(0))
    sig = torch.sigmoid(z)
    dsig = sig * (1.0 - sig)
    grad_xi = gamma * dsig.unsqueeze(2) * a.unsqueeze(0)
    zeros = torch.zeros(Q, 1, 3, dtype=DTYPE, device=DEVICE)
    return torch.cat([zeros, grad_xi], dim=1)


def eval_relu_features(x: torch.Tensor, a: torch.Tensor, r: torch.Tensor,
                       gamma: float) -> torch.Tensor:
    """Feature function: ξ_m(x) = max(0, γ(a_m^T x + r_m))"""
    Q = x.shape[0]
    z = gamma * (x @ a.T + r.unsqueeze(0))
    xi = torch.relu(z)
    ones = torch.ones(Q, 1, dtype=DTYPE, device=DEVICE)
    return torch.cat([ones, xi], dim=1)


def eval_relu_feature_grads(x: torch.Tensor, a: torch.Tensor, r: torch.Tensor,
                            gamma: float) -> torch.Tensor:
    """ReLU feature gradient: ∇ξ_m = γ (z > 0) a_m"""
    Q = x.shape[0]
    z = gamma * (x @ a.T + r.unsqueeze(0))
    drelu = (z > 0).to(DTYPE)
    grad_xi = gamma * drelu.unsqueeze(2) * a.unsqueeze(0)
    zeros = torch.zeros(Q, 1, 3, dtype=DTYPE, device=DEVICE)
    return torch.cat([zeros, grad_xi], dim=1)


def eval_softplus_features(x: torch.Tensor, a: torch.Tensor, r: torch.Tensor,
                           gamma: float) -> torch.Tensor:
    """Feature function: ξ_m(x) = log(1 + e^(γ(a_m^T x + r_m)))"""
    Q = x.shape[0]
    z = gamma * (x @ a.T + r.unsqueeze(0))
    xi = torch.nn.functional.softplus(z)
    ones = torch.ones(Q, 1, dtype=DTYPE, device=DEVICE)
    return torch.cat([ones, xi], dim=1)


def eval_softplus_feature_grads(x: torch.Tensor, a: torch.Tensor, r: torch.Tensor,
                                gamma: float) -> torch.Tensor:
    """softplus feature gradient: ∇ξ_m = γ σ(z) a_m"""
    Q = x.shape[0]
    z = gamma * (x @ a.T + r.unsqueeze(0))
    dsoftplus = torch.sigmoid(z)
    grad_xi = gamma * dsoftplus.unsqueeze(2) * a.unsqueeze(0)
    zeros = torch.zeros(Q, 1, 3, dtype=DTYPE, device=DEVICE)
    return torch.cat([zeros, grad_xi], dim=1)


def eval_elu_features(x: torch.Tensor, a: torch.Tensor, r: torch.Tensor,
                      gamma: float) -> torch.Tensor:
    """Feature function: ξ_m(x) = z if z > 0 else e^z - 1, where z = γ(a_m^T x + r_m)"""
    Q = x.shape[0]
    z = gamma * (x @ a.T + r.unsqueeze(0))
    xi = torch.nn.functional.elu(z)
    ones = torch.ones(Q, 1, dtype=DTYPE, device=DEVICE)
    return torch.cat([ones, xi], dim=1)


def eval_elu_feature_grads(x: torch.Tensor, a: torch.Tensor, r: torch.Tensor,
                           gamma: float) -> torch.Tensor:
    """ELU feature gradient: ∇ξ_m = γ (1 if z > 0 else e^z) a_m"""
    Q = x.shape[0]
    z = gamma * (x @ a.T + r.unsqueeze(0))
    delu = torch.where(z > 0, torch.ones_like(z), torch.exp(z))
    grad_xi = gamma * delu.unsqueeze(2) * a.unsqueeze(0)
    zeros = torch.zeros(Q, 1, 3, dtype=DTYPE, device=DEVICE)
    return torch.cat([zeros, grad_xi], dim=1)


def eval_swish_features(x: torch.Tensor, a: torch.Tensor, r: torch.Tensor,
                        gamma: float) -> torch.Tensor:
    """Feature function: ξ_m(x) = z · σ(z), where z = γ(a_m^T x + r_m)"""
    Q = x.shape[0]
    z = gamma * (x @ a.T + r.unsqueeze(0))
    xi = z * torch.sigmoid(z)
    ones = torch.ones(Q, 1, dtype=DTYPE, device=DEVICE)
    return torch.cat([ones, xi], dim=1)


def eval_swish_feature_grads(x: torch.Tensor, a: torch.Tensor, r: torch.Tensor,
                             gamma: float) -> torch.Tensor:
    """Swish feature gradient: ∇ξ_m = γ (σ(z) + z·σ(z)·(1-σ(z))) a_m"""
    Q = x.shape[0]
    z = gamma * (x @ a.T + r.unsqueeze(0))
    sig = torch.sigmoid(z)
    dswish = sig + z * sig * (1.0 - sig)
    grad_xi = gamma * dswish.unsqueeze(2) * a.unsqueeze(0)
    zeros = torch.zeros(Q, 1, 3, dtype=DTYPE, device=DEVICE)
    return torch.cat([zeros, grad_xi], dim=1)


# ---------------------------------------------------------------------------
# Activation function registry
# ---------------------------------------------------------------------------
ACTIVATION_REGISTRY = {
    "tanh": (eval_tanh_features, eval_tanh_feature_grads),
    "sigmoid": (eval_sigmoid_features, eval_sigmoid_feature_grads),
    "relu": (eval_relu_features, eval_relu_feature_grads),
    "softplus": (eval_softplus_features, eval_softplus_feature_grads),
    "elu": (eval_elu_features, eval_elu_feature_grads),
    "swish": (eval_swish_features, eval_swish_feature_grads),
}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_ablation_activation(results, save_path):
    """Grouped bar chart comparing final metrics across activation functions for all methods.

    Args:
        results: { activation_name: { method_name: {"r_s", "r_u", "rel_u", "rel_sigma"} } }
        save_path: output file path
    """
    activation_names = list(results.keys())
    method_names = list(results[activation_names[0]].keys())
    n_methods = len(method_names)

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

    x_positions = np.arange(len(activation_names))
    width = 0.8 / n_methods

    for ax, mkey, title, ylabel in zip(
        axes.flat, metric_keys, titles, ylabels
    ):
        for i, method in enumerate(method_names):
            vals = []
            for act_name in activation_names:
                v = results[act_name].get(method, {}).get(mkey, float("nan"))
                vals.append(v)
            vals = np.array(vals, dtype=float)
            offset = (i - (n_methods - 1) / 2) * width
            sty = ALGO_STYLE.get(method, {"color": "gray"})
            ax.bar(
                x_positions + offset, vals, width,
                label=method, color=sty["color"],
            )
        ax.set_yscale("log")
        ax.set_xlabel("Activation function")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(activation_names, rotation=30, ha="right")
        ax.legend()
        ax.grid(alpha=0.3, linestyle="--", axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # --- Configuration (override defaults as needed) ---
    cfg = Config(K_max=100000)
    ablation_activation_list = ["tanh", "sigmoid", "relu", "softplus", "elu", "swish"]

    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Ablation activation list: {ablation_activation_list}")

    # --- Prepare common data ---
    mu, lam = compute_lame_constants(cfg.E, cfg.nu)
    S = build_compliance_matrix(cfg.E, cfg.nu)

    # Fixed M and gamma for activation ablation
    a, r = generate_features(cfg.M, seed=BASE_SEED)

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
    ablation_activation_results = {}

    for act_name in ablation_activation_list:
        print(f"\n{'='*60}")
        print(f"=== Ablation: activation={act_name} ===")
        print(f"{'='*60}")

        # Get feature functions for current activation
        features_fn, grads_fn = ACTIVATION_REGISTRY[act_name]

        # Evaluate features with current activation
        xi_abl = features_fn(x_train, a, r, cfg.gamma)
        grad_xi_abl = grads_fn(x_train, a, r, cfg.gamma)

        # Assemble system
        A_abl, B_abl, F_abl = assemble_system(xi_abl, grad_xi_abl, S, f_train, zeta_train)
        print(f"  A: {A_abl.shape}, B: {B_abl.shape}, F: {F_abl.shape}")

        # Test features with current activation
        xi_test_abl = features_fn(x_test, a, r, cfg.gamma)
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
            str(OUTPUT_DIR / f"kkt-convergence_activation={act_name}.png"),
        )

        # Plot L2 error convergence
        direct_l2 = {
            "rel_u": abl_results["Direct"]["history"]["rel_u"][-1],
            "rel_sigma": abl_results["Direct"]["history"]["rel_sigma"][-1],
        }
        plot_l2_convergence(
            iter_histories, iter_labels,
            str(OUTPUT_DIR / f"l2-error-convergence_activation={act_name}.png"),
            direct_l2=direct_l2,
        )

        # Collect summary data
        ablation_activation_results[act_name] = {}
        for method in ["Direct", "ADMM", "Uzawa", "Arrow-Hurwicz"]:
            h = abl_results[method]["history"]
            ablation_activation_results[act_name][method] = {
                "r_s": h["r_s"][-1],
                "r_u": h["r_u"][-1],
                "rel_u": h["rel_u"][-1],
                "rel_sigma": h["rel_sigma"][-1],
            }

    # --- Summary plot ---
    if ablation_activation_results:
        plot_ablation_activation(ablation_activation_results, str(OUTPUT_DIR / "ablation-activation.png"))

    # --- Print summary table (Markdown-compatible) ---
    print("\n=== Ablation activation: Summary ===\n")
    print(f"| {'Activation':<10} | {'Algorithm':<14} | {'rel_u':>12} | {'rel_sigma':>12} |")
    print(f"|{'-'*11}:|:{'-'*15}|{'-'*13}:|{'-'*13}:|")
    for act_name in ablation_activation_list:
        for method in ["Direct", "ADMM", "Uzawa", "Arrow-Hurwicz"]:
            d = ablation_activation_results[act_name][method]
            print(f"| {act_name:<10} | {method:<14} | {d['rel_u']:>12.2e} | {d['rel_sigma']:>12.2e} |")
