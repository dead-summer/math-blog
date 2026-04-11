from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from linear_elasticity_3d import (
    BASE_SEED,
    DEVICE,
    DTYPE,
    AlgorithmResult,
    SharedBenchmarkData,
    build_shared_benchmark,
    compute_lame_constants,
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
    Q_int: int = (2 ** 5) ** 3
    Q_bc: int = 6 * (2 ** 4) ** 2
    Q_test: int = (2 ** 4) ** 3
    sampling_method: str = "sobol"
    lambda_bc: float = 1.0e1
    lambda_c: float = 1.0
    lambda_e: float = 1.0
    pinn_width: int = 64
    pinn_depth: int = 4
    pinn_epochs: int = 3000
    pinn_lr: float = 1.0e-3
    pinn_report_every: int = 100
    pinn_seed: int = BASE_SEED
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


class MixedElasticityNet(nn.Module):
    """Simple tanh MLP with 9 outputs: 6 stress + 3 displacement."""

    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 3
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.Tanh())
            in_dim = width
        layers.append(nn.Linear(in_dim, 9))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
    if cfg.pinn_width <= 0 or cfg.pinn_depth <= 0:
        raise ValueError("Config.pinn_width and Config.pinn_depth must be positive.")
    if cfg.pinn_epochs <= 0:
        raise ValueError("Config.pinn_epochs must be positive.")
    if not math.isfinite(cfg.pinn_lr) or cfg.pinn_lr <= 0.0:
        raise ValueError("Config.pinn_lr must be finite and positive.")
    if cfg.pinn_report_every <= 0:
        raise ValueError("Config.pinn_report_every must be positive.")
    if cfg.body_force_batch_size <= 0:
        raise ValueError("Config.body_force_batch_size must be positive.")
    if cfg.eval_batch_size <= 0:
        raise ValueError("Config.eval_batch_size must be positive.")
    validate_sampling_method(cfg.sampling_method)
    validate_algorithm_selection(cfg.algorithms_to_run, VALID_PINN_ALGORITHMS)


def split_outputs(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split network output into stress and displacement parts."""

    return raw[:, :6], raw[:, 6:]


def compute_output_gradients(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    create_graph: bool,
) -> list[torch.Tensor]:
    """Differentiate each output component with respect to the inputs."""

    grads: list[torch.Tensor] = []
    for comp in range(outputs.shape[1]):
        grads.append(
            torch.autograd.grad(
                outputs[:, comp].sum(),
                inputs,
                create_graph=create_graph,
                retain_graph=True,
            )[0]
        )
    return grads


def compute_eps_and_div_sigma(
    sigma: torch.Tensor,
    u: torch.Tensor,
    x: torch.Tensor,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute strain and stress divergence via autodiff."""

    grad_u = compute_output_gradients(u, x, create_graph=create_graph)
    grad_sigma = compute_output_gradients(sigma, x, create_graph=create_graph)

    du_dx1 = torch.stack([grad_u[comp][:, 0] for comp in range(3)], dim=1)
    du_dx2 = torch.stack([grad_u[comp][:, 1] for comp in range(3)], dim=1)
    du_dx3 = torch.stack([grad_u[comp][:, 2] for comp in range(3)], dim=1)
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
            grad_sigma[0][:, 0] + grad_sigma[3][:, 1] + grad_sigma[5][:, 2],
            grad_sigma[3][:, 0] + grad_sigma[1][:, 1] + grad_sigma[4][:, 2],
            grad_sigma[5][:, 0] + grad_sigma[4][:, 1] + grad_sigma[2][:, 2],
        ],
        dim=1,
    )
    return eps, div_sigma


def compute_pinn_loss(
    model: MixedElasticityNet,
    x_int: torch.Tensor,
    f_int: torch.Tensor,
    x_bc: torch.Tensor,
    compliance_voigt: torch.Tensor,
    lambda_c: float,
    lambda_e: float,
    lambda_bc: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Build the mixed PINN loss from constitutive, equilibrium, and boundary terms."""

    x_int_ad = x_int.detach().requires_grad_(True)
    sigma_int, u_int = split_outputs(model(x_int_ad))
    eps_int, div_sigma_int = compute_eps_and_div_sigma(
        sigma_int,
        u_int,
        x_int_ad,
        create_graph=True,
    )

    r_c = sigma_int @ compliance_voigt.T - eps_int
    r_e = div_sigma_int + f_int
    constitutive_loss = r_c.square().sum(dim=1).mean()
    equilibrium_loss = r_e.square().sum(dim=1).mean()

    _, u_bc = split_outputs(model(x_bc))
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


def train_pinn(
    cfg: PinnConfig,
    x_int: torch.Tensor,
    f_int: torch.Tensor,
    x_bc: torch.Tensor,
    compliance_voigt: torch.Tensor,
) -> tuple[MixedElasticityNet, float]:
    """Train the mixed PINN with full-batch Adam."""

    torch.manual_seed(cfg.pinn_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.pinn_seed)

    model = MixedElasticityNet(cfg.pinn_width, cfg.pinn_depth).to(
        device=DEVICE,
        dtype=DTYPE,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.pinn_lr)

    synchronize_device()
    t0 = time.perf_counter()
    for epoch in range(1, cfg.pinn_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = compute_pinn_loss(
            model,
            x_int,
            f_int,
            x_bc,
            compliance_voigt,
            cfg.lambda_c,
            cfg.lambda_e,
            cfg.lambda_bc,
        )
        loss.backward()
        optimizer.step()

        should_report = (
            epoch == 1
            or epoch % cfg.pinn_report_every == 0
            or epoch == cfg.pinn_epochs
        )
        if should_report:
            print(
                f"    epoch {epoch:>5d}/{cfg.pinn_epochs}, "
                f"loss={metrics['loss']:.4e}"
            )

    synchronize_device()
    return model, time.perf_counter() - t0


def compute_pinn_residual_norms(
    model: MixedElasticityNet,
    data: PinnExperimentData,
) -> tuple[float, float, float]:
    """Evaluate strong-form residual norms for a trained PINN."""

    constitutive_sq = 0.0
    equilibrium_sq = 0.0
    boundary_sq = 0.0
    w_int = 1.0 / data.x_int.shape[0]

    model.eval()
    for start in range(0, data.x_int.shape[0], data.eval_batch_size):
        end = min(start + data.eval_batch_size, data.x_int.shape[0])
        xb = data.x_int[start:end].detach().requires_grad_(True)
        fb = data.f_int[start:end]
        sigma, u = split_outputs(model(xb))
        eps, div_sigma = compute_eps_and_div_sigma(sigma, u, xb, create_graph=False)
        r_c = sigma @ data.compliance_voigt.T - eps
        r_e = div_sigma + fb
        constitutive_sq += w_int * r_c.square().sum(dim=1).sum().item()
        equilibrium_sq += w_int * r_e.square().sum(dim=1).sum().item()

    with torch.no_grad():
        for start in range(0, data.x_bc.shape[0], data.eval_batch_size):
            end = min(start + data.eval_batch_size, data.x_bc.shape[0])
            xb = data.x_bc[start:end]
            wb = data.w_bc[start:end]
            _, u_bc = split_outputs(model(xb))
            boundary_sq += (wb * u_bc.square().sum(dim=1)).sum().item()

    return constitutive_sq**0.5, equilibrium_sq**0.5, boundary_sq**0.5


def compute_pinn_l2_errors(
    model: MixedElasticityNet,
    data: PinnExperimentData,
) -> tuple[float, float]:
    """Compute relative test L2 errors for a trained PINN."""

    model.eval()
    sigma_parts: list[torch.Tensor] = []
    u_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, data.x_test.shape[0], data.eval_batch_size):
            end = min(start + data.eval_batch_size, data.x_test.shape[0])
            sigma_batch, u_batch = split_outputs(model(data.x_test[start:end]))
            sigma_parts.append(sigma_batch)
            u_parts.append(u_batch)

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
        f"width={cfg.pinn_width}, depth={cfg.pinn_depth}, epochs={cfg.pinn_epochs}, "
        f"lr={cfg.pinn_lr:.2e}, lambda_c={cfg.lambda_c:.2e}, "
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
    model, wall_time = train_pinn(
        cfg,
        benchmark.x_int,
        benchmark.f_int,
        benchmark.x_bc,
        benchmark.compliance_voigt,
    )
    result = evaluate_pinn_result(
        "PINN",
        wall_time,
        model,
        experiment_data,
    )
    print_result_summary(result)
    results = [result]

    if print_table:
        print_summary_table(results, title="PINN Summary")
    return results


def main(cfg: PinnConfig | None = None) -> None:
    """Script entrypoint."""

    run_experiment(cfg, print_table=True)


if __name__ == "__main__":
    main()
