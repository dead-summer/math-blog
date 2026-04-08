"""
Ablation: activation function with synchronized split spaces.

The chosen activation is applied to both stress and displacement feature
spaces, while the random parameters of the two spaces remain independent.
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
OUTPUT_DIR = PROJECT_ROOT / "public" / "images" / "saddle-point" / "ablation" / "activation"
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


# ---------------------------------------------------------------------------
# Activation functions: features and gradients
# ---------------------------------------------------------------------------
def eval_tanh_features(
    x: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Feature function: xi_m(x) = tanh(gamma (a_m^T x + r_m))."""

    z = gamma * (x @ a.T + r.unsqueeze(0))
    xi = torch.tanh(z)
    ones = torch.ones(x.shape[0], 1, dtype=DTYPE, device=DEVICE)
    return torch.cat([ones, xi], dim=1)


def eval_tanh_feature_grads(
    x: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """tanh feature gradient."""

    z = gamma * (x @ a.T + r.unsqueeze(0))
    dtanh = 1.0 - torch.tanh(z) ** 2
    grad_xi = gamma * dtanh.unsqueeze(2) * a.unsqueeze(0)
    zeros = torch.zeros(x.shape[0], 1, 3, dtype=DTYPE, device=DEVICE)
    return torch.cat([zeros, grad_xi], dim=1)


def eval_sigmoid_features(
    x: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Feature function: xi_m(x) = sigmoid(gamma (a_m^T x + r_m))."""

    z = gamma * (x @ a.T + r.unsqueeze(0))
    xi = torch.sigmoid(z)
    ones = torch.ones(x.shape[0], 1, dtype=DTYPE, device=DEVICE)
    return torch.cat([ones, xi], dim=1)


def eval_sigmoid_feature_grads(
    x: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """sigmoid feature gradient."""

    z = gamma * (x @ a.T + r.unsqueeze(0))
    sig = torch.sigmoid(z)
    dsig = sig * (1.0 - sig)
    grad_xi = gamma * dsig.unsqueeze(2) * a.unsqueeze(0)
    zeros = torch.zeros(x.shape[0], 1, 3, dtype=DTYPE, device=DEVICE)
    return torch.cat([zeros, grad_xi], dim=1)


def eval_relu_features(
    x: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Feature function: xi_m(x) = relu(gamma (a_m^T x + r_m))."""

    z = gamma * (x @ a.T + r.unsqueeze(0))
    xi = torch.relu(z)
    ones = torch.ones(x.shape[0], 1, dtype=DTYPE, device=DEVICE)
    return torch.cat([ones, xi], dim=1)


def eval_relu_feature_grads(
    x: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """ReLU feature gradient."""

    z = gamma * (x @ a.T + r.unsqueeze(0))
    drelu = (z > 0).to(DTYPE)
    grad_xi = gamma * drelu.unsqueeze(2) * a.unsqueeze(0)
    zeros = torch.zeros(x.shape[0], 1, 3, dtype=DTYPE, device=DEVICE)
    return torch.cat([zeros, grad_xi], dim=1)


def eval_softplus_features(
    x: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Feature function: xi_m(x) = softplus(gamma (a_m^T x + r_m))."""

    z = gamma * (x @ a.T + r.unsqueeze(0))
    xi = torch.nn.functional.softplus(z)
    ones = torch.ones(x.shape[0], 1, dtype=DTYPE, device=DEVICE)
    return torch.cat([ones, xi], dim=1)


def eval_softplus_feature_grads(
    x: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """softplus feature gradient."""

    z = gamma * (x @ a.T + r.unsqueeze(0))
    dsoftplus = torch.sigmoid(z)
    grad_xi = gamma * dsoftplus.unsqueeze(2) * a.unsqueeze(0)
    zeros = torch.zeros(x.shape[0], 1, 3, dtype=DTYPE, device=DEVICE)
    return torch.cat([zeros, grad_xi], dim=1)


def eval_elu_features(
    x: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Feature function: xi_m(x) = elu(gamma (a_m^T x + r_m))."""

    z = gamma * (x @ a.T + r.unsqueeze(0))
    xi = torch.nn.functional.elu(z)
    ones = torch.ones(x.shape[0], 1, dtype=DTYPE, device=DEVICE)
    return torch.cat([ones, xi], dim=1)


def eval_elu_feature_grads(
    x: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """ELU feature gradient."""

    z = gamma * (x @ a.T + r.unsqueeze(0))
    delu = torch.where(z > 0, torch.ones_like(z), torch.exp(z))
    grad_xi = gamma * delu.unsqueeze(2) * a.unsqueeze(0)
    zeros = torch.zeros(x.shape[0], 1, 3, dtype=DTYPE, device=DEVICE)
    return torch.cat([zeros, grad_xi], dim=1)


def eval_swish_features(
    x: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Feature function: xi_m(x) = z sigmoid(z)."""

    z = gamma * (x @ a.T + r.unsqueeze(0))
    xi = z * torch.sigmoid(z)
    ones = torch.ones(x.shape[0], 1, dtype=DTYPE, device=DEVICE)
    return torch.cat([ones, xi], dim=1)


def eval_swish_feature_grads(
    x: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Swish feature gradient."""

    z = gamma * (x @ a.T + r.unsqueeze(0))
    sig = torch.sigmoid(z)
    dswish = sig + z * sig * (1.0 - sig)
    grad_xi = gamma * dswish.unsqueeze(2) * a.unsqueeze(0)
    zeros = torch.zeros(x.shape[0], 1, 3, dtype=DTYPE, device=DEVICE)
    return torch.cat([zeros, grad_xi], dim=1)


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
def plot_ablation_activation(
    results: dict[str, dict[str, dict[str, float]]],
    save_path: str,
) -> None:
    """Grouped bar chart comparing final metrics across activations."""

    activation_names = list(results.keys())
    method_names = list(results[activation_names[0]].keys())
    n_methods = len(method_names)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metric_specs = [
        ("r_c", r"$\|r_c\|_2$", r"$\|r_c\|_2$"),
        ("r_e", r"$\|r_e\|_2$", r"$\|r_e\|_2$"),
        ("rel_u", "Displacement error", "Relative $L^2$ error"),
        ("rel_sigma", "Stress error", "Relative $L^2$ error"),
    ]

    x_positions = np.arange(len(activation_names))
    width = 0.8 / max(n_methods, 1)

    axes_flat = list(axes.flat)
    for ax, (metric_key, title, ylabel) in zip(axes_flat, metric_specs):
        for idx, method in enumerate(method_names):
            vals = np.array(
                [
                    results[act_name].get(method, {}).get(metric_key, float("nan"))
                    for act_name in activation_names
                ],
                dtype=float,
            )
            offset = (idx - (n_methods - 1) / 2) * width
            style = ALGO_STYLE.get(method, {"color": "gray"})
            ax.bar(
                x_positions + offset,
                vals,
                width,
                label=method,
                color=style["color"],
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


def run_single_activation_ablation(
    act_name: str,
    cfg: Config,
    *,
    a_s: torch.Tensor,
    r_s: torch.Tensor,
    a_u: torch.Tensor,
    r_u: torch.Tensor,
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
    """Run one activation ablation while releasing large tensors eagerly."""

    print(f"\n{'=' * 60}")
    print(f"=== Ablation: activation = {act_name} ===")
    print(f"{'=' * 60}")

    features_fn, grads_fn = ACTIVATION_REGISTRY[act_name]

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
            xi_s_train_abl = features_fn(x_train, a_s, r_s, cfg.gamma_s)
            xi_u_train_abl = features_fn(x_train, a_u, r_u, cfg.gamma_u)
            xi_u_active_train_abl = activate_displacement_features(
                xi_u_train_abl,
                zeta_train,
            )
            del xi_u_train_abl
            xi_u_train_abl = None

        A_abl, B_abl, C_abl, F_abl = assemble_system_in_batches(
            x_train,
            f_train,
            a_s,
            r_s,
            cfg.gamma_s,
            a_u,
            r_u,
            cfg.gamma_u,
            compliance_voigt,
            zeta_train,
            cfg.assembly_batch_size,
            x_bc=x_bc,
            w_bc=w_bc,
            zeta_bc=zeta_bc,
            lambda_bc=cfg.lambda_bc if cfg.use_penalty else 0.0,
            features_fn=features_fn,
            grads_fn=grads_fn,
        )
        print(
            f"  A: {tuple(A_abl.shape)}, B: {tuple(B_abl.shape)}, "
            f"C: {tuple(C_abl.shape)}, F: {tuple(F_abl.shape)}"
        )

        xi_s_test_abl = features_fn(x_test, a_s, r_s, cfg.gamma_s)
        xi_u_test_abl = features_fn(x_test, a_u, r_u, cfg.gamma_u)
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
                str(OUTPUT_DIR / f"kkt-convergence_activation={act_name}.png"),
            )
            plot_l2_convergence(
                l2_histories,
                l2_labels,
                str(OUTPUT_DIR / f"l2-error-convergence_activation={act_name}.png"),
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
    ablation_activation_list = ["tanh", "sigmoid", "relu", "softplus", "elu", "swish"]

    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Ablation activation list: {ablation_activation_list}")

    mu, lam = compute_lame_constants(cfg.E, cfg.nu)
    compliance_voigt = build_compliance_matrix(cfg.E, cfg.nu)

    a_s, r_s = generate_features(cfg.M_s, seed=STRESS_SEED)
    a_u, r_u = generate_features(cfg.M_u, seed=DISP_SEED)

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

    ablation_activation_results: dict[str, dict[str, dict[str, float]]] = {}

    for act_name in ablation_activation_list:
        ablation_activation_results[act_name] = run_single_activation_ablation(
            act_name,
            cfg,
            a_s=a_s,
            r_s=r_s,
            a_u=a_u,
            r_u=r_u,
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

    if ablation_activation_results:
        plot_ablation_activation(
            ablation_activation_results,
            str(OUTPUT_DIR / "ablation-activation.png"),
        )

    print("\n=== Ablation activation: Summary ===\n")
    summary_methods = (
        list(next(iter(ablation_activation_results.values())).keys())
        if ablation_activation_results
        else []
    )
    print(
        f"| {'Activation':<10} | {'Algorithm':<14} | {'||r_c||':>10} | {'||r_e||':>10} | "
        f"{'rel_u':>12} | {'rel_sigma':>12} |"
    )
    print(
        f"|{'-' * 11}:|:{'-' * 15}|{'-' * 11}:|{'-' * 11}:|"
        f"{'-' * 13}:|{'-' * 13}:|"
    )
    for act_name in ablation_activation_list:
        for method in summary_methods:
            item = ablation_activation_results[act_name][method]
            print(
                f"| {act_name:<10} | {method:<14} | {item['r_c']:>10.2e} | "
                f"{item['r_e']:>10.2e} | {item['rel_u']:>12.2e} | "
                f"{item['rel_sigma']:>12.2e} |"
            )
