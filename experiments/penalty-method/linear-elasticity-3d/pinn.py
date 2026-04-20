from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from linear_elasticity_3d import (
    ALGO_STYLE,
    OUTPUT_DIR,
    BASE_SEED,
    DEVICE,
    DTYPE,
    AlgorithmResult,
    SharedBenchmarkData,
    build_shared_benchmark,
    compute_lame_constants,
    configure_plotting,
    print_result_summary,
    print_summary_table,
    synchronize_device,
    validate_algorithm_selection,
    validate_sampling_method,
)

VALID_PINN_ALGORITHMS = ("pinn",)


@dataclass
class PinnConfig:
    """Configuration for the mixed PINN experiment."""

    E: float = 1.0
    nu: float = 0.3
    M_s: int = 300
    M_u: int = 300
    Q_int: int = (2 ** 6) ** 3
    Q_bc: int = 6 * (2 ** 5) ** 2
    Q_test: int = (2 ** 5) ** 3
    sampling_method: str = "sobol"
    lambda_bc: float = 1.0e1
    lambda_c: float = 1.0
    lambda_e: float = 1.0
    train_batch_size: int = 5000
    epochs: int = 500
    lr: float = 1.0e-3
    report_every: int = 50
    seed: int = BASE_SEED
    body_force_batch_size: int = 5_000
    eval_batch_size: int = 2_000
    algorithms_to_run: list[str] = field(
        default_factory=lambda: [
            "pinn",
        ]
    )


@dataclass(frozen=True)
class PinnExperimentData:
    """All tensors needed to evaluate one trained PINN."""

    x_int: torch.Tensor
    f_int: torch.Tensor
    x_bc: torch.Tensor
    w_bc: torch.Tensor
    x_test: torch.Tensor
    u_exact_test: torch.Tensor
    sigma_exact_test: torch.Tensor
    compliance_voigt: torch.Tensor
    eval_batch_size: int


@dataclass
class PinnTrainingHistory:
    """Tracked training metrics recorded at console report points."""

    epochs: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    r_c: list[float] = field(default_factory=list)
    r_e: list[float] = field(default_factory=list)
    r_b: list[float] = field(default_factory=list)
    rel_u: list[float] = field(default_factory=list)
    rel_sigma: list[float] = field(default_factory=list)


class MixedElasticityNet(nn.Module):
    """Two-branch single-hidden-layer tanh MLP for stress and displacement."""

    def __init__(self, M_s: int, M_u: int) -> None:
        super().__init__()
        self.stress_net = nn.Sequential(
            nn.Linear(3, M_s),
            nn.Tanh(),
            nn.Linear(M_s, 6),
        )
        self.displacement_net = nn.Sequential(
            nn.Linear(3, M_u),
            nn.Tanh(),
            nn.Linear(M_u, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = self.stress_net(x)
        u = self.displacement_net(x)
        return torch.cat([sigma, u], dim=1)


def validate_config(cfg: PinnConfig) -> None:
    """Validate config before starting any expensive work."""

    if cfg.E <= 0.0:
        raise ValueError("Config.E must be positive.")
    if not (-1.0 < cfg.nu < 0.5):
        raise ValueError("Config.nu must lie in (-1, 0.5).")
    if cfg.Q_int <= 0:
        raise ValueError("Config.Q_int must be positive.")
    if cfg.Q_bc < 6:
        raise ValueError("Config.Q_bc must be at least 6.")
    if cfg.Q_test <= 0:
        raise ValueError("Config.Q_test must be positive.")
    if not math.isfinite(cfg.lambda_bc) or cfg.lambda_bc <= 0.0:
        raise ValueError("Config.lambda_bc must be finite and positive.")
    if not math.isfinite(cfg.lambda_c) or cfg.lambda_c <= 0.0:
        raise ValueError("Config.lambda_c must be finite and positive.")
    if not math.isfinite(cfg.lambda_e) or cfg.lambda_e <= 0.0:
        raise ValueError("Config.lambda_e must be finite and positive.")
    if cfg.M_s <= 0 or cfg.M_u <= 0:
        raise ValueError("Config.M_s and Config.M_u must be positive.")
    if cfg.train_batch_size <= 0:
        raise ValueError("Config.train_batch_size must be positive.")
    if cfg.epochs <= 0:
        raise ValueError("Config.epochs must be positive.")
    if not math.isfinite(cfg.lr) or cfg.lr <= 0.0:
        raise ValueError("Config.lr must be finite and positive.")
    if cfg.report_every <= 0:
        raise ValueError("Config.report_every must be positive.")
    if cfg.body_force_batch_size <= 0:
        raise ValueError("Config.body_force_batch_size must be positive.")
    if cfg.eval_batch_size <= 0:
        raise ValueError("Config.eval_batch_size must be positive.")
    validate_sampling_method(cfg.sampling_method)
    validate_algorithm_selection(cfg.algorithms_to_run, VALID_PINN_ALGORITHMS)


def split_outputs(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split network output into stress and displacement parts."""

    return raw[:, :6], raw[:, 6:]


def unpack_single_hidden_tanh_branch(
    branch: nn.Sequential,
    expected_output_dim: int,
    branch_name: str,
) -> tuple[nn.Linear, nn.Linear]:
    """Validate and unpack one Linear -> Tanh -> Linear branch."""

    if len(branch) != 3:
        raise TypeError(
            f"{branch_name} must contain exactly three modules, got {len(branch)}."
        )
    if not isinstance(branch[0], nn.Linear):
        raise TypeError(f"{branch_name}[0] must be nn.Linear.")
    if not isinstance(branch[1], nn.Tanh):
        raise TypeError(f"{branch_name}[1] must be nn.Tanh.")
    if not isinstance(branch[2], nn.Linear):
        raise TypeError(f"{branch_name}[2] must be nn.Linear.")

    input_layer = branch[0]
    output_layer = branch[2]
    if input_layer.in_features != 3:
        raise ValueError(
            f"{branch_name} must accept 3D coordinates, got {input_layer.in_features}."
        )
    if output_layer.out_features != expected_output_dim:
        raise ValueError(
            f"{branch_name} must output {expected_output_dim} values, "
            f"got {output_layer.out_features}."
        )
    if input_layer.out_features != output_layer.in_features:
        raise ValueError(
            f"{branch_name} hidden width mismatch: "
            f"{input_layer.out_features} != {output_layer.in_features}."
        )
    return input_layer, output_layer


def forward_branch_with_spatial_grads(
    branch: nn.Sequential,
    x: torch.Tensor,
    expected_output_dim: int,
    branch_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate one branch together with its first spatial derivatives."""

    input_layer, output_layer = unpack_single_hidden_tanh_branch(
        branch,
        expected_output_dim,
        branch_name,
    )
    pre = x @ input_layer.weight.T + input_layer.bias
    hidden = torch.tanh(pre)
    outputs = hidden @ output_layer.weight.T + output_layer.bias
    d_hidden = 1.0 - hidden.square()

    dx1 = (d_hidden * input_layer.weight[:, 0].unsqueeze(0)) @ output_layer.weight.T
    dx2 = (d_hidden * input_layer.weight[:, 1].unsqueeze(0)) @ output_layer.weight.T
    dx3 = (d_hidden * input_layer.weight[:, 2].unsqueeze(0)) @ output_layer.weight.T
    return outputs, dx1, dx2, dx3


def assemble_strain_and_div_sigma(
    du_dx1: torch.Tensor,
    du_dx2: torch.Tensor,
    du_dx3: torch.Tensor,
    ds_dx1: torch.Tensor,
    ds_dx2: torch.Tensor,
    ds_dx3: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble strain and stress divergence from directional derivatives."""

    eps = torch.stack(
        [
            du_dx1[:, 0],
            du_dx2[:, 1],
            du_dx3[:, 2],
            du_dx2[:, 0] + du_dx1[:, 1],
            du_dx3[:, 1] + du_dx2[:, 2],
            du_dx3[:, 0] + du_dx1[:, 2],
        ],
        dim=1,
    )

    div_sigma = torch.stack(
        [
            ds_dx1[:, 0] + ds_dx2[:, 3] + ds_dx3[:, 5],
            ds_dx1[:, 3] + ds_dx2[:, 1] + ds_dx3[:, 4],
            ds_dx1[:, 5] + ds_dx2[:, 4] + ds_dx3[:, 2],
        ],
        dim=1,
    )
    return eps, div_sigma


def evaluate_fields_with_spatial_terms(
    model: MixedElasticityNet,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate sigma, u, strain, and stress divergence on one batch."""

    sigma, ds_dx1, ds_dx2, ds_dx3 = forward_branch_with_spatial_grads(
        model.stress_net,
        x,
        expected_output_dim=6,
        branch_name="stress_net",
    )
    u, du_dx1, du_dx2, du_dx3 = forward_branch_with_spatial_grads(
        model.displacement_net,
        x,
        expected_output_dim=3,
        branch_name="displacement_net",
    )
    eps, div_sigma = assemble_strain_and_div_sigma(
        du_dx1,
        du_dx2,
        du_dx3,
        ds_dx1,
        ds_dx2,
        ds_dx3,
    )
    return sigma, u, eps, div_sigma


def compute_pinn_loss(
    model: MixedElasticityNet,
    x_int: torch.Tensor,
    f_int: torch.Tensor,
    x_bc_batch: torch.Tensor,
    compliance_voigt: torch.Tensor,
    lambda_c: float,
    lambda_e: float,
    lambda_bc: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Build the mixed PINN loss from constitutive, equilibrium, and boundary batches."""

    sigma_int, _, eps_int, div_sigma_int = evaluate_fields_with_spatial_terms(
        model,
        x_int,
    )

    r_c = sigma_int @ compliance_voigt.T - eps_int
    r_e = div_sigma_int + f_int
    constitutive_loss = r_c.square().sum(dim=1).mean()
    equilibrium_loss = r_e.square().sum(dim=1).mean()

    u_bc = model.displacement_net(x_bc_batch)
    boundary_loss = u_bc.square().sum(dim=1).mean()

    loss = (
        lambda_c * constitutive_loss
        + lambda_e * equilibrium_loss
        + lambda_bc * boundary_loss
    )
    metrics = {
        "loss": float(loss.detach().item()),
        "constitutive": float(constitutive_loss.detach().item()),
        "equilibrium": float(equilibrium_loss.detach().item()),
        "boundary": float(boundary_loss.detach().item()),
    }
    return loss, metrics


def take_random_cycled_batch_indices(
    permutation: torch.Tensor,
    position: int,
    n_points: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Take a random batch, reshuffling and cycling when one pass is exhausted."""

    index_parts: list[torch.Tensor] = []
    remaining = batch_size

    while remaining > 0:
        if position >= n_points:
            permutation = torch.randperm(n_points, device=device)
            position = 0

        take = min(remaining, n_points - position)
        index_parts.append(permutation[position : position + take])
        position += take
        remaining -= take

    if len(index_parts) == 1:
        return index_parts[0], permutation, position
    return torch.cat(index_parts), permutation, position


def record_pinn_history_entry(
    history: PinnTrainingHistory,
    epoch: int,
    loss: float,
    r_c: float,
    r_e: float,
    r_b: float,
    rel_u: float,
    rel_sigma: float,
) -> None:
    """Append one reported training snapshot into history."""

    history.epochs.append(epoch)
    history.loss.append(loss)
    history.r_c.append(r_c)
    history.r_e.append(r_e)
    history.r_b.append(r_b)
    history.rel_u.append(rel_u)
    history.rel_sigma.append(rel_sigma)


def train_pinn(
    cfg: PinnConfig,
    data: PinnExperimentData,
) -> tuple[MixedElasticityNet, float, PinnTrainingHistory]:
    """Train the mixed PINN with paired random interior and boundary mini-batches."""

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    model = MixedElasticityNet(cfg.M_s, cfg.M_u).to(
        device=DEVICE,
        dtype=DTYPE,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history = PinnTrainingHistory()

    synchronize_device()
    t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_metrics = {
            "loss": 0.0,
            "constitutive": 0.0,
            "equilibrium": 0.0,
            "boundary": 0.0,
        }
        int_permutation = torch.randperm(data.x_int.shape[0], device=data.x_int.device)
        bc_permutation = torch.randperm(data.x_bc.shape[0], device=data.x_bc.device)
        bc_position = 0

        for start in range(0, data.x_int.shape[0], cfg.train_batch_size):
            end = min(start + cfg.train_batch_size, data.x_int.shape[0])
            int_batch_idx = int_permutation[start:end]
            bc_batch_idx, bc_permutation, bc_position = take_random_cycled_batch_indices(
                bc_permutation,
                bc_position,
                data.x_bc.shape[0],
                end - start,
                data.x_bc.device,
            )
            x_int_batch = data.x_int[int_batch_idx]
            f_int_batch = data.f_int[int_batch_idx]
            x_bc_batch = data.x_bc[bc_batch_idx]

            optimizer.zero_grad(set_to_none=True)
            loss, metrics = compute_pinn_loss(
                model,
                x_int_batch,
                f_int_batch,
                x_bc_batch,
                data.compliance_voigt,
                cfg.lambda_c,
                cfg.lambda_e,
                cfg.lambda_bc,
            )
            loss.backward()
            optimizer.step()

            batch_weight = (end - start) / data.x_int.shape[0]
            for key in epoch_metrics:
                epoch_metrics[key] += batch_weight * metrics[key]

        should_report = (
            epoch == 1
            or epoch % cfg.report_every == 0
            or epoch == cfg.epochs
        )
        if should_report:
            r_c, r_e, r_b = compute_pinn_residual_norms(model, data)
            rel_u, rel_sigma = compute_pinn_l2_errors(model, data)
            record_pinn_history_entry(
                history,
                epoch,
                epoch_metrics["loss"],
                r_c,
                r_e,
                r_b,
                rel_u,
                rel_sigma,
            )
            print(
                f"    epoch {epoch:>5d}/{cfg.epochs}, "
                f"loss={epoch_metrics['loss']:.4e}, "
                f"r_c={r_c:.4e}, r_e={r_e:.4e}, r_b={r_b:.4e}, "
                f"rel_u={rel_u:.4e}, rel_sigma={rel_sigma:.4e}"
            )

    synchronize_device()
    return model, time.perf_counter() - t0, history


def compute_pinn_residual_norms(
    model: MixedElasticityNet,
    data: PinnExperimentData,
) -> tuple[float, float, float]:
    """Evaluate strong-form residual norms for a trained PINN."""

    constitutive_sq = 0.0
    equilibrium_sq = 0.0
    boundary_sq = 0.0
    w_int = 1.0 / data.x_int.shape[0]
    was_training = model.training

    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, data.x_int.shape[0], data.eval_batch_size):
                end = min(start + data.eval_batch_size, data.x_int.shape[0])
                xb = data.x_int[start:end]
                fb = data.f_int[start:end]
                sigma, _, eps, div_sigma = evaluate_fields_with_spatial_terms(model, xb)
                r_c = sigma @ data.compliance_voigt.T - eps
                r_e = div_sigma + fb
                constitutive_sq += w_int * r_c.square().sum(dim=1).sum().item()
                equilibrium_sq += w_int * r_e.square().sum(dim=1).sum().item()

            for start in range(0, data.x_bc.shape[0], data.eval_batch_size):
                end = min(start + data.eval_batch_size, data.x_bc.shape[0])
                xb = data.x_bc[start:end]
                wb = data.w_bc[start:end]
                u_bc = model.displacement_net(xb)
                boundary_sq += (wb * u_bc.square().sum(dim=1)).sum().item()
    finally:
        model.train(was_training)

    return constitutive_sq**0.5, equilibrium_sq**0.5, boundary_sq**0.5


def compute_pinn_l2_errors(
    model: MixedElasticityNet,
    data: PinnExperimentData,
) -> tuple[float, float]:
    """Compute relative test L2 errors for a trained PINN."""

    was_training = model.training
    model.eval()
    sigma_parts: list[torch.Tensor] = []
    u_parts: list[torch.Tensor] = []
    try:
        with torch.no_grad():
            for start in range(0, data.x_test.shape[0], data.eval_batch_size):
                end = min(start + data.eval_batch_size, data.x_test.shape[0])
                sigma_batch, u_batch = split_outputs(model(data.x_test[start:end]))
                sigma_parts.append(sigma_batch)
                u_parts.append(u_batch)
    finally:
        model.train(was_training)

    sigma_pred = torch.cat(sigma_parts, dim=0)
    u_pred = torch.cat(u_parts, dim=0)
    voigt_weight = torch.tensor(
        [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        dtype=DTYPE,
        device=DEVICE,
    )

    u_err = torch.sqrt(((u_pred - data.u_exact_test).square().sum(dim=1)).mean())
    u_ref = torch.sqrt((data.u_exact_test.square().sum(dim=1)).mean())
    rel_u = (u_err / u_ref).item() if u_ref > 0 else float("inf")

    sigma_err = torch.sqrt(
        (voigt_weight * (sigma_pred - data.sigma_exact_test).square())
        .sum(dim=1)
        .mean()
    )
    sigma_ref = torch.sqrt(
        (voigt_weight * data.sigma_exact_test.square()).sum(dim=1).mean()
    )
    rel_sigma = (sigma_err / sigma_ref).item() if sigma_ref > 0 else float("inf")
    return rel_u, rel_sigma


def plot_pinn_metric_history(
    ax: plt.Axes,
    epochs: list[int],
    title: str,
    ylabel: str,
    values: list[float],
    save_path: str,
) -> bool:
    """Plot one positive metric history on one semilog-y axis."""

    epoch_vals = np.array(epochs, dtype=float)
    metric_vals = np.array(values, dtype=float)
    valid = (
        np.isfinite(epoch_vals)
        & np.isfinite(metric_vals)
        & (metric_vals > 0.0)
    )
    style = ALGO_STYLE.get("PINN", {})

    if valid.any():
        n_markers = max(1, int(valid.sum()) // 10)
        ax.semilogy(
            epoch_vals[valid],
            metric_vals[valid],
            label="PINN",
            linewidth=1.4,
            color=style.get("color", "#E76F51"),
            linestyle=style.get("linestyle", "-"),
            marker=style.get("marker", "X"),
            markersize=5,
            markevery=n_markers,
        )
        ax.legend()
    else:
        print(f"  Skipped {title} in {save_path} (no positive finite values)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, linestyle="--")
    return bool(valid.any())


def plot_pinn_loss_history(history: PinnTrainingHistory, save_path: str) -> None:
    """Plot the reported training loss history shown in the console."""

    configure_plotting()
    if not history.epochs:
        print(f"  Skipped: {save_path} (no history to plot)")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    has_line = plot_pinn_metric_history(
        ax,
        history.epochs,
        title="PINN Training Loss",
        ylabel="Loss",
        values=history.loss,
        save_path=save_path,
    )
    if not has_line:
        plt.close(fig)
        print(f"  Skipped: {save_path} (all series invalid)")
        return

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_pinn_residual_history(history: PinnTrainingHistory, save_path: str) -> None:
    """Plot constitutive, equilibrium, and boundary residual histories."""

    configure_plotting()
    if not history.epochs:
        print(f"  Skipped: {save_path} (no history to plot)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plotted = [
        plot_pinn_metric_history(
            ax,
            history.epochs,
            title=title,
            ylabel="Residual norm",
            values=values,
            save_path=save_path,
        )
        for ax, title, values in zip(
            axes,
            [r"$\|r_c\|_2$", r"$\|r_e\|_2$", r"$\|r_b\|_2$"],
            [history.r_c, history.r_e, history.r_b],
        )
    ]
    if not any(plotted):
        plt.close(fig)
        print(f"  Skipped: {save_path} (all series invalid)")
        return

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_pinn_l2_history(history: PinnTrainingHistory, save_path: str) -> None:
    """Plot displacement and stress relative L2 error histories."""

    configure_plotting()
    if not history.epochs:
        print(f"  Skipped: {save_path} (no history to plot)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    plotted = [
        plot_pinn_metric_history(
            ax,
            history.epochs,
            title=title,
            ylabel="Relative $L^2$ error",
            values=values,
            save_path=save_path,
        )
        for ax, title, values in zip(
            axes,
            [
                r"Displacement $\|\Phi^u - u_{ex}\|_{L^2} / \|u_{ex}\|_{L^2}$",
                r"Stress $\|\Phi^\sigma - \sigma_{ex}\|_{L^2} / \|\sigma_{ex}\|_{L^2}$",
            ],
            [history.rel_u, history.rel_sigma],
        )
    ]
    if not any(plotted):
        plt.close(fig)
        print(f"  Skipped: {save_path} (all series invalid)")
        return

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def evaluate_pinn_result(
    name: str,
    wall_time: float,
    model: MixedElasticityNet,
    data: PinnExperimentData,
) -> AlgorithmResult:
    """Package the trained PINN metrics into a shared result struct."""

    r_c, r_e, r_b = compute_pinn_residual_norms(model, data)
    rel_u, rel_sigma = compute_pinn_l2_errors(model, data)
    return AlgorithmResult(
        name=name,
        r_c=r_c,
        r_e=r_e,
        r_b=r_b,
        rel_u=rel_u,
        rel_sigma=rel_sigma,
        wall_time=wall_time,
    )


def run_experiment(
    cfg: PinnConfig | None = None,
    print_table: bool = True,
    benchmark: SharedBenchmarkData | None = None,
) -> list[AlgorithmResult]:
    """Run the selected PINN-side algorithms and return their metrics."""

    cfg = PinnConfig() if cfg is None else cfg
    validate_config(cfg)
    selected_algorithm_ids = validate_algorithm_selection(
        cfg.algorithms_to_run,
        VALID_PINN_ALGORITHMS,
    )

    print(f"Device: {DEVICE}")
    print(
        f"Config: Q_int={cfg.Q_int}, Q_bc={cfg.Q_bc}, Q_test={cfg.Q_test}, "
        f"M_s={cfg.M_s}, M_u={cfg.M_u}, train_batch_size={cfg.train_batch_size}, "
        f"epochs={cfg.epochs}, "
        f"lr={cfg.lr:.2e}, lambda_c={cfg.lambda_c:.2e}, "
        f"lambda_e={cfg.lambda_e:.2e}, lambda_bc={cfg.lambda_bc:.2e}, "
        f"sampling={cfg.sampling_method}"
    )
    print(f"Algorithms: {selected_algorithm_ids}")

    mu, lam = compute_lame_constants(cfg.E, cfg.nu)
    print(f"Material: E={cfg.E}, nu={cfg.nu}, mu={mu:.4f}, lam={lam:.4f}")

    if benchmark is None:
        print("Building benchmark data...")
        benchmark = build_shared_benchmark(
            E=cfg.E,
            nu=cfg.nu,
            Q_int=cfg.Q_int,
            Q_bc=cfg.Q_bc,
            Q_test=cfg.Q_test,
            sampling_method=cfg.sampling_method,
            body_force_batch_size=cfg.body_force_batch_size,
            interior_seed=BASE_SEED + 11,
            boundary_seed=BASE_SEED + 12,
            test_seed=BASE_SEED + 13,
        )
    else:
        print("Using shared benchmark data...")

    experiment_data = PinnExperimentData(
        x_int=benchmark.x_int,
        f_int=benchmark.f_int,
        x_bc=benchmark.x_bc,
        w_bc=benchmark.w_bc,
        x_test=benchmark.x_test,
        u_exact_test=benchmark.u_exact_test,
        sigma_exact_test=benchmark.sigma_exact_test,
        compliance_voigt=benchmark.compliance_voigt,
        eval_batch_size=cfg.eval_batch_size,
    )

    print("Training PINN...")
    model, wall_time, history = train_pinn(
        cfg,
        experiment_data,
    )
    result = evaluate_pinn_result(
        "PINN",
        wall_time,
        model,
        experiment_data,
    )
    print_result_summary(result)
    plot_pinn_loss_history(
        history,
        str(OUTPUT_DIR / "pinn-loss-history.png"),
    )
    plot_pinn_residual_history(
        history,
        str(OUTPUT_DIR / "pinn-residual-history.png"),
    )
    plot_pinn_l2_history(
        history,
        str(OUTPUT_DIR / "pinn-l2-history.png"),
    )
    results = [result]

    if print_table:
        print_summary_table(results, title="PINN Summary")
    return results


def main(cfg: PinnConfig | None = None) -> None:
    """Script entrypoint."""

    run_experiment(cfg, print_table=True)


if __name__ == "__main__":
    main()
