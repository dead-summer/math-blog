from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch

from linear_elasticity_3d import (
    DEVICE,
    DTYPE,
    AlgorithmResult,
    FeatureEvaluationData,
    SharedBenchmarkData,
    SharedFeatureSpace,
    build_feature_evaluation_data,
    build_shared_benchmark,
    build_shared_feature_space,
    compute_lame_constants,
    eval_feature_grads,
    eval_features,
    evaluate_feature_result,
    print_result_summary,
    print_summary_table,
    synchronize_device,
    validate_algorithm_selection,
    validate_sampling_method,
    clear_cuda_cache,
)


VALID_STRONG_ALGORITHMS = ("eigh", "lstsq")


def build_strain_gradient_bases() -> torch.Tensor:
    """Return the three fixed gradient-to-strain coupling blocks."""

    base_1 = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )
    base_2 = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )
    base_3 = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )
    return torch.stack([base_1, base_2, base_3], dim=0)


STRAIN_GRAD_BASES = build_strain_gradient_bases()
IDENTITY_3 = torch.eye(3, dtype=DTYPE, device=DEVICE)


@dataclass
class StrongConfig:
    """Configuration for the strong-form least-squares experiment."""

    E: float = 1.0
    nu: float = 0.3
    gamma_s: float = 2.0
    gamma_u: float = 2.0
    M_s: int = 500
    M_u: int = 500
    Q_int: int = (2 ** 6) ** 3
    Q_bc: int = 6 * (2 ** 5) ** 2
    Q_test: int = (2 ** 5) ** 3
    sampling_method: str = "sobol"
    lambda_bc: float = 1.0e1
    eigh_rtol: float = 1.0e-15
    body_force_batch_size: int = 5_000
    assembly_batch_size: int = 5_000
    algorithms_to_run: list[str] = field(
        default_factory=lambda: [
            "eigh",
        ]
    )


@dataclass(frozen=True)
class StrongExperimentData:
    """All tensors needed to run and evaluate one strong-form solver."""

    H: torch.Tensor
    g: torch.Tensor
    dim_s: int
    eval_data: FeatureEvaluationData


def validate_config(cfg: StrongConfig) -> None:
    """Validate config before starting any expensive work."""

    if cfg.E <= 0.0:
        raise ValueError("Config.E must be positive.")
    if not (-1.0 < cfg.nu < 0.5):
        raise ValueError("Config.nu must lie in (-1, 0.5).")
    if cfg.gamma_s <= 0.0 or cfg.gamma_u <= 0.0:
        raise ValueError("Config.gamma_s and Config.gamma_u must be positive.")
    if cfg.M_s <= 0 or cfg.M_u <= 0:
        raise ValueError("Config.M_s and Config.M_u must be positive.")
    if cfg.Q_int <= 0:
        raise ValueError("Config.Q_int must be positive.")
    if cfg.Q_bc < 6:
        raise ValueError("Config.Q_bc must be at least 6.")
    if cfg.Q_test <= 0:
        raise ValueError("Config.Q_test must be positive.")
    if not math.isfinite(cfg.lambda_bc) or cfg.lambda_bc <= 0.0:
        raise ValueError("Config.lambda_bc must be finite and positive.")
    if not math.isfinite(cfg.eigh_rtol) or cfg.eigh_rtol < 0.0:
        raise ValueError("Config.eigh_rtol must be finite and non-negative.")
    if cfg.body_force_batch_size <= 0:
        raise ValueError("Config.body_force_batch_size must be positive.")
    if cfg.assembly_batch_size <= 0:
        raise ValueError("Config.assembly_batch_size must be positive.")
    validate_sampling_method(cfg.sampling_method)
    validate_algorithm_selection(cfg.algorithms_to_run, VALID_STRONG_ALGORITHMS)


def add_block_scaled(
    target: torch.Tensor,
    feature_matrix: torch.Tensor,
    block: torch.Tensor,
    row_stride: int,
    col_stride: int,
) -> None:
    """Add feature_matrix kron block into a strided matrix."""

    for row in range(block.shape[0]):
        for col in range(block.shape[1]):
            coeff = block[row, col].item()
            if coeff == 0.0:
                continue
            target[row::row_stride, col::col_stride] += coeff * feature_matrix


def add_rhs_feature_blocks(
    target: torch.Tensor,
    feature_by_vec: torch.Tensor,
    block: torch.Tensor,
    block_size: int,
    scale: float = 1.0,
) -> None:
    """Accumulate feature moments into a strided rhs vector."""

    contribution = scale * (feature_by_vec @ block.T)
    for row in range(block.shape[0]):
        target[row::block_size] += contribution[:, row]


def accumulate_interior_moments(
    x_int: torch.Tensor,
    f_int: torch.Tensor,
    a_s: torch.Tensor,
    r_s: torch.Tensor,
    gamma_s: float,
    a_u: torch.Tensor,
    r_u: torch.Tensor,
    gamma_u: float,
    batch_size: int,
) -> tuple[
    torch.Tensor,
    list[torch.Tensor],
    list[list[torch.Tensor]],
    list[list[torch.Tensor]],
    list[torch.Tensor],
]:
    """Accumulate moments for the strong-form normal equations."""

    mp1_s = a_s.shape[0] + 1
    mp1_u = a_u.shape[0] + 1
    weight = 1.0 / x_int.shape[0]

    gram_xi_s = torch.zeros(mp1_s, mp1_s, dtype=DTYPE, device=DEVICE)
    cross_xi_grad_u = [
        torch.zeros(mp1_s, mp1_u, dtype=DTYPE, device=DEVICE) for _ in range(3)
    ]
    grad_gram_u = [
        [
            torch.zeros(mp1_u, mp1_u, dtype=DTYPE, device=DEVICE)
            for _ in range(3)
        ]
        for _ in range(3)
    ]
    grad_gram_s = [
        [
            torch.zeros(mp1_s, mp1_s, dtype=DTYPE, device=DEVICE)
            for _ in range(3)
        ]
        for _ in range(3)
    ]
    grad_force_s = [
        torch.zeros(mp1_s, 3, dtype=DTYPE, device=DEVICE) for _ in range(3)
    ]

    with torch.no_grad():
        for start in range(0, x_int.shape[0], batch_size):
            end = min(start + batch_size, x_int.shape[0])
            xb = x_int[start:end]
            fb = f_int[start:end]

            xi_s_batch = eval_features(xb, a_s, r_s, gamma_s)
            grad_s_batch = eval_feature_grads(xb, a_s, r_s, gamma_s)
            grad_u_batch = eval_feature_grads(xb, a_u, r_u, gamma_u)

            gram_xi_s += weight * (xi_s_batch.T @ xi_s_batch)
            for dim_i in range(3):
                cross_xi_grad_u[dim_i] += weight * (
                    xi_s_batch.T @ grad_u_batch[:, :, dim_i]
                )
                grad_force_s[dim_i] += weight * (grad_s_batch[:, :, dim_i].T @ fb)
                for dim_j in range(3):
                    grad_gram_u[dim_i][dim_j] += weight * (
                        grad_u_batch[:, :, dim_i].T @ grad_u_batch[:, :, dim_j]
                    )
                    grad_gram_s[dim_i][dim_j] += weight * (
                        grad_s_batch[:, :, dim_i].T @ grad_s_batch[:, :, dim_j]
                    )

    return gram_xi_s, cross_xi_grad_u, grad_gram_u, grad_gram_s, grad_force_s


def accumulate_boundary_gram(
    x_bc: torch.Tensor,
    w_bc: torch.Tensor,
    a_u: torch.Tensor,
    r_u: torch.Tensor,
    gamma_u: float,
    batch_size: int,
) -> torch.Tensor:
    """Accumulate the weighted boundary Gram matrix for xi_u."""

    mp1_u = a_u.shape[0] + 1
    gram_bc = torch.zeros(mp1_u, mp1_u, dtype=DTYPE, device=DEVICE)

    with torch.no_grad():
        for start in range(0, x_bc.shape[0], batch_size):
            end = min(start + batch_size, x_bc.shape[0])
            xb = x_bc[start:end]
            wb = w_bc[start:end]
            xi_u_batch = eval_features(xb, a_u, r_u, gamma_u)
            gram_bc += xi_u_batch.T @ (wb.unsqueeze(1) * xi_u_batch)

    return gram_bc


def assemble_normal_equations(
    cfg: StrongConfig,
    compliance_voigt: torch.Tensor,
    x_int: torch.Tensor,
    f_int: torch.Tensor,
    x_bc: torch.Tensor,
    w_bc: torch.Tensor,
    a_s: torch.Tensor,
    r_s: torch.Tensor,
    a_u: torch.Tensor,
    r_u: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble the strong-form normal equations H z = g."""

    (
        gram_xi_s,
        cross_xi_grad_u,
        grad_gram_u,
        grad_gram_s,
        grad_force_s,
    ) = accumulate_interior_moments(
        x_int,
        f_int,
        a_s,
        r_s,
        cfg.gamma_s,
        a_u,
        r_u,
        cfg.gamma_u,
        cfg.assembly_batch_size,
    )
    gram_bc_u = accumulate_boundary_gram(
        x_bc,
        w_bc,
        a_u,
        r_u,
        cfg.gamma_u,
        cfg.assembly_batch_size,
    )

    mp1_s = a_s.shape[0] + 1
    mp1_u = a_u.shape[0] + 1
    dim_s = 6 * mp1_s
    dim_u = 3 * mp1_u

    H_ss = torch.zeros(dim_s, dim_s, dtype=DTYPE, device=DEVICE)
    H_su = torch.zeros(dim_s, dim_u, dtype=DTYPE, device=DEVICE)
    H_uu = torch.zeros(dim_u, dim_u, dtype=DTYPE, device=DEVICE)
    g_s = torch.zeros(dim_s, dtype=DTYPE, device=DEVICE)

    compliance_sq = compliance_voigt.T @ compliance_voigt
    add_block_scaled(H_ss, gram_xi_s, compliance_sq, row_stride=6, col_stride=6)

    for dim_i in range(3):
        constitutive_cross = compliance_voigt @ STRAIN_GRAD_BASES[dim_i]
        add_block_scaled(
            H_su,
            cross_xi_grad_u[dim_i],
            -constitutive_cross,
            row_stride=6,
            col_stride=3,
        )
        add_rhs_feature_blocks(
            g_s,
            grad_force_s[dim_i],
            STRAIN_GRAD_BASES[dim_i],
            block_size=6,
            scale=-1.0,
        )

        for dim_j in range(3):
            constitutive_uu = STRAIN_GRAD_BASES[dim_i].T @ STRAIN_GRAD_BASES[dim_j]
            equilibrium_ss = STRAIN_GRAD_BASES[dim_i] @ STRAIN_GRAD_BASES[dim_j].T
            add_block_scaled(
                H_uu,
                grad_gram_u[dim_i][dim_j],
                constitutive_uu,
                row_stride=3,
                col_stride=3,
            )
            add_block_scaled(
                H_ss,
                grad_gram_s[dim_i][dim_j],
                equilibrium_ss,
                row_stride=6,
                col_stride=6,
            )

    add_block_scaled(
        H_uu,
        gram_bc_u,
        cfg.lambda_bc * IDENTITY_3,
        row_stride=3,
        col_stride=3,
    )

    H = torch.zeros(dim_s + dim_u, dim_s + dim_u, dtype=DTYPE, device=DEVICE)
    H[:dim_s, :dim_s] = H_ss
    H[:dim_s, dim_s:] = H_su
    H[dim_s:, :dim_s] = H_su.T
    H[dim_s:, dim_s:] = H_uu
    H = 0.5 * (H + H.T)

    g = torch.zeros(dim_s + dim_u, dtype=DTYPE, device=DEVICE)
    g[:dim_s] = g_s
    return H, g


def solve_lstsq(H: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Solve the normal equations with torch.linalg.lstsq."""

    synchronize_device()
    t0 = time.perf_counter()
    try:
        sol = torch.linalg.lstsq(H, g.unsqueeze(1)).solution.squeeze(1)
        if not torch.isfinite(sol).all():
            raise RuntimeError("non-finite solution")
    except (RuntimeError, torch.linalg.LinAlgError) as exc:
        sol = torch.full((H.shape[0],), float("nan"), dtype=DTYPE, device=DEVICE)
        print(f"    Warning: torch.linalg.lstsq failed with {type(exc).__name__}")

    synchronize_device()
    return sol, time.perf_counter() - t0


def solve_eigh(H: torch.Tensor, g: torch.Tensor, rtol: float) -> tuple[torch.Tensor, float]:
    """Solve the normal equations with truncated eigen decomposition."""

    synchronize_device()
    t0 = time.perf_counter()
    try:
        eigvals, eigvecs = torch.linalg.eigh(H)
        threshold = rtol * eigvals.abs().max()
        keep = eigvals > threshold
        if not keep.any():
            raise RuntimeError("all eigenvalues were truncated")

        coeffs = eigvecs[:, keep].T @ g
        coeffs = coeffs / eigvals[keep]
        sol = eigvecs[:, keep] @ coeffs
        if not torch.isfinite(sol).all():
            raise RuntimeError("non-finite solution")
        print(
            f"    eigh truncation: kept {int(keep.sum().item())}/{eigvals.numel()} "
            f"eigenvalues, threshold={threshold.item():.2e}"
        )
    except (RuntimeError, torch.linalg.LinAlgError) as exc:
        sol = torch.full((H.shape[0],), float("nan"), dtype=DTYPE, device=DEVICE)
        print(f"    Warning: torch.linalg.eigh failed with {type(exc).__name__}")

    synchronize_device()
    return sol, time.perf_counter() - t0


def split_solution(z: torch.Tensor, dim_s: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the full coefficient vector into stress and displacement parts."""

    return z[:dim_s], z[dim_s:]


def run_algorithm(
    algorithm_id: str,
    data: StrongExperimentData,
    cfg: StrongConfig,
) -> AlgorithmResult:
    """Run one configured strong-form algorithm and evaluate it."""

    if algorithm_id == "eigh":
        print("Running Strong (Eigh)...")
        z, wall_time = solve_eigh(data.H, data.g, cfg.eigh_rtol)
        s, u = split_solution(z, data.dim_s)
        result = evaluate_feature_result("Strong (Eigh)", wall_time, s, u, data.eval_data)
    else:
        print("Running Strong (Lstsq)...")
        z, wall_time = solve_lstsq(data.H, data.g)
        s, u = split_solution(z, data.dim_s)
        result = evaluate_feature_result("Strong (Lstsq)", wall_time, s, u, data.eval_data)

    print_result_summary(result)
    return result


def run_experiment(
    cfg: StrongConfig | None = None,
    print_table: bool = True,
    benchmark: SharedBenchmarkData | None = None,
    feature_space: SharedFeatureSpace | None = None,
) -> list[AlgorithmResult]:
    """Run the selected strong-form methods and return their metrics."""

    cfg = StrongConfig() if cfg is None else cfg
    validate_config(cfg)
    selected_algorithm_ids = validate_algorithm_selection(
        cfg.algorithms_to_run,
        VALID_STRONG_ALGORITHMS,
    )

    print(f"Device: {DEVICE}")
    print(
        f"Config: M_s={cfg.M_s}, M_u={cfg.M_u}, "
        f"Q_int={cfg.Q_int}, Q_bc={cfg.Q_bc}, Q_test={cfg.Q_test}, "
        f"gamma_s={cfg.gamma_s}, gamma_u={cfg.gamma_u}, "
        f"lambda_bc={cfg.lambda_bc:.2e}, eigh_rtol={cfg.eigh_rtol:.2e}, "
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
        )
    else:
        print("Using shared benchmark data...")

    if feature_space is None:
        print("Generating random feature spaces...")
        feature_space = build_shared_feature_space(
            M_s=cfg.M_s,
            M_u=cfg.M_u,
            gamma_s=cfg.gamma_s,
            gamma_u=cfg.gamma_u,
        )
    else:
        print("Using shared random feature spaces...")

    if feature_space.gamma_s != cfg.gamma_s or feature_space.gamma_u != cfg.gamma_u:
        raise ValueError("SharedFeatureSpace gamma does not match StrongConfig.")

    compliance_voigt = benchmark.compliance_voigt

    print("Assembling strong-form normal equations...")
    H, g = assemble_normal_equations(
        cfg,
        compliance_voigt,
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
    print(f"System shapes: H={tuple(H.shape)}, g={tuple(g.shape)}")

    experiment_data = StrongExperimentData(
        H=H,
        g=g,
        dim_s=6 * (cfg.M_s + 1),
        eval_data=build_feature_evaluation_data(
            benchmark,
            feature_space,
            cfg.assembly_batch_size,
        ),
    )

    results = [
        run_algorithm(algorithm_id, experiment_data, cfg)
        for algorithm_id in selected_algorithm_ids
    ]
    if print_table:
        print_summary_table(results, title="Strong Form Summary")
    return results


def main(cfg: StrongConfig | None = None) -> None:
    """Script entrypoint."""

    run_experiment(cfg, print_table=True)


if __name__ == "__main__":
    main()
