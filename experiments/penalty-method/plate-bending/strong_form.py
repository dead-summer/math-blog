from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch

from plate_bending import (
    DEVICE,
    DTYPE,
    AlgorithmResult,
    FROBENIUS_WEIGHT_MATRIX,
    FeatureEvaluationData,
    SharedBenchmarkData,
    SharedFeatureSpace,
    build_feature_evaluation_data,
    build_shared_benchmark,
    build_shared_feature_space,
    clear_cuda_cache,
    compute_bending_stiffness,
    eval_feature_grads,
    eval_feature_hessians,
    eval_features,
    evaluate_feature_result,
    print_result_summary,
    print_summary_table,
    synchronize_device,
    validate_algorithm_selection,
    validate_sampling_method,
)


VALID_STRONG_ALGORITHMS = ("eigh", "lstsq")
EQUILIBRIUM_DIVDIV_WEIGHTS = torch.tensor([1.0, 1.0, 2.0], dtype=DTYPE, device=DEVICE)


@dataclass
class StrongConfig:
    """Configuration for the strong-form least-squares experiment."""

    E: float = 1.0
    nu: float = 0.3
    h: float = 1.0
    gamma_m: float = 2.0
    gamma_u: float = 2.0
    M_m: int = 500
    M_u: int = 500
    Q_int: int = (2 ** 7) ** 2
    Q_bc: int = 4 * (2 ** 6)
    Q_test: int = (2 ** 6) ** 2
    sampling_method: str = "sobol"
    lambda_0: float = 1.0e1
    lambda_1: float = 1.0e1
    eigh_rtol: float = 1.0e-15
    assembly_batch_size: int = 5_000
    algorithms_to_run: list[str] = field(
        default_factory=lambda: [
            "eigh",
            "lstsq",
        ]
    )


@dataclass(frozen=True)
class StrongExperimentData:
    """All tensors needed to run and evaluate one strong-form solver."""

    H: torch.Tensor
    g: torch.Tensor
    dim_m: int
    eval_data: FeatureEvaluationData


def validate_config(cfg: StrongConfig) -> None:
    """Validate config before starting any expensive work."""

    if cfg.E <= 0.0:
        raise ValueError("StrongConfig.E must be positive.")
    if not (-1.0 < cfg.nu < 0.5):
        raise ValueError("StrongConfig.nu must lie in (-1, 0.5).")
    if cfg.h <= 0.0:
        raise ValueError("StrongConfig.h must be positive.")
    if cfg.gamma_m <= 0.0 or cfg.gamma_u <= 0.0:
        raise ValueError("StrongConfig.gamma_m and StrongConfig.gamma_u must be positive.")
    if cfg.M_m <= 0 or cfg.M_u <= 0:
        raise ValueError("StrongConfig.M_m and StrongConfig.M_u must be positive.")
    if cfg.Q_int <= 0:
        raise ValueError("StrongConfig.Q_int must be positive.")
    if cfg.Q_bc < 4:
        raise ValueError("StrongConfig.Q_bc must be at least 4.")
    if cfg.Q_test <= 0:
        raise ValueError("StrongConfig.Q_test must be positive.")
    if not math.isfinite(cfg.lambda_0) or cfg.lambda_0 <= 0.0:
        raise ValueError("StrongConfig.lambda_0 must be finite and positive.")
    if not math.isfinite(cfg.lambda_1) or cfg.lambda_1 <= 0.0:
        raise ValueError("StrongConfig.lambda_1 must be finite and positive.")
    if not math.isfinite(cfg.eigh_rtol) or cfg.eigh_rtol < 0.0:
        raise ValueError("StrongConfig.eigh_rtol must be finite and non-negative.")
    if cfg.assembly_batch_size <= 0:
        raise ValueError("StrongConfig.assembly_batch_size must be positive.")
    validate_sampling_method(cfg.sampling_method)
    validate_algorithm_selection(cfg.algorithms_to_run, VALID_STRONG_ALGORITHMS)


def accumulate_interior_moments(
    x_int: torch.Tensor,
    f_int: torch.Tensor,
    a_m: torch.Tensor,
    r_m: torch.Tensor,
    gamma_m: float,
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

    mp1_m = a_m.shape[0] + 1
    mp1_u = a_u.shape[0] + 1
    weight = 1.0 / x_int.shape[0]

    gram_xi_m = torch.zeros(mp1_m, mp1_m, dtype=DTYPE, device=DEVICE)
    cross_xi_m_hess_u = [
        torch.zeros(mp1_m, mp1_u, dtype=DTYPE, device=DEVICE) for _ in range(3)
    ]
    hess_gram_u = [
        [
            torch.zeros(mp1_u, mp1_u, dtype=DTYPE, device=DEVICE)
            for _ in range(3)
        ]
        for _ in range(3)
    ]
    hess_gram_m = [
        [
            torch.zeros(mp1_m, mp1_m, dtype=DTYPE, device=DEVICE)
            for _ in range(3)
        ]
        for _ in range(3)
    ]
    hess_force_m = [
        torch.zeros(mp1_m, dtype=DTYPE, device=DEVICE) for _ in range(3)
    ]

    with torch.no_grad():
        for start in range(0, x_int.shape[0], batch_size):
            end = min(start + batch_size, x_int.shape[0])
            xb = x_int[start:end]
            fb = f_int[start:end]

            xi_m_batch = eval_features(xb, a_m, r_m, gamma_m)
            hess_m_batch = eval_feature_hessians(xb, a_m, r_m, gamma_m)
            hess_u_batch = eval_feature_hessians(xb, a_u, r_u, gamma_u)

            gram_xi_m += weight * (xi_m_batch.T @ xi_m_batch)
            for comp_i in range(3):
                cross_xi_m_hess_u[comp_i] += weight * (
                    xi_m_batch.T @ hess_u_batch[:, :, comp_i]
                )
                hess_force_m[comp_i] += weight * (hess_m_batch[:, :, comp_i].T @ fb)
                for comp_j in range(3):
                    hess_gram_u[comp_i][comp_j] += weight * (
                        hess_u_batch[:, :, comp_i].T @ hess_u_batch[:, :, comp_j]
                    )
                    hess_gram_m[comp_i][comp_j] += weight * (
                        hess_m_batch[:, :, comp_i].T @ hess_m_batch[:, :, comp_j]
                    )

    return (
        gram_xi_m,
        cross_xi_m_hess_u,
        hess_gram_u,
        hess_gram_m,
        hess_force_m,
    )


def accumulate_boundary_grams(
    x_bc: torch.Tensor,
    w_bc: torch.Tensor,
    n_bc: torch.Tensor,
    a_u: torch.Tensor,
    r_u: torch.Tensor,
    gamma_u: float,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate the weighted boundary Gram matrices for xi_u and d_n xi_u."""

    mp1_u = a_u.shape[0] + 1
    gram_bc = torch.zeros(mp1_u, mp1_u, dtype=DTYPE, device=DEVICE)
    gram_dn = torch.zeros(mp1_u, mp1_u, dtype=DTYPE, device=DEVICE)

    with torch.no_grad():
        for start in range(0, x_bc.shape[0], batch_size):
            end = min(start + batch_size, x_bc.shape[0])
            xb = x_bc[start:end]
            wb = w_bc[start:end]
            nb = n_bc[start:end]

            xi_u_batch = eval_features(xb, a_u, r_u, gamma_u)
            grad_u_batch = eval_feature_grads(xb, a_u, r_u, gamma_u)
            dn_xi_batch = (grad_u_batch * nb.unsqueeze(1)).sum(dim=2)

            gram_bc += xi_u_batch.T @ (wb.unsqueeze(1) * xi_u_batch)
            gram_dn += dn_xi_batch.T @ (wb.unsqueeze(1) * dn_xi_batch)

    return gram_bc, gram_dn


def assemble_normal_equations(
    cfg: StrongConfig,
    compliance_voigt: torch.Tensor,
    x_int: torch.Tensor,
    f_int: torch.Tensor,
    x_bc: torch.Tensor,
    w_bc: torch.Tensor,
    n_bc: torch.Tensor,
    a_m: torch.Tensor,
    r_m: torch.Tensor,
    a_u: torch.Tensor,
    r_u: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble the strong-form normal equations H z = g."""

    (
        gram_xi_m,
        cross_xi_m_hess_u,
        hess_gram_u,
        hess_gram_m,
        hess_force_m,
    ) = accumulate_interior_moments(
        x_int,
        f_int,
        a_m,
        r_m,
        cfg.gamma_m,
        a_u,
        r_u,
        cfg.gamma_u,
        cfg.assembly_batch_size,
    )
    gram_bc, gram_dn = accumulate_boundary_grams(
        x_bc,
        w_bc,
        n_bc,
        a_u,
        r_u,
        cfg.gamma_u,
        cfg.assembly_batch_size,
    )

    mp1_m = a_m.shape[0] + 1
    mp1_u = a_u.shape[0] + 1
    dim_m = 3 * mp1_m

    H_mm = torch.zeros(dim_m, dim_m, dtype=DTYPE, device=DEVICE)
    H_mu = torch.zeros(dim_m, mp1_u, dtype=DTYPE, device=DEVICE)
    H_uu = torch.zeros(mp1_u, mp1_u, dtype=DTYPE, device=DEVICE)
    g_m = torch.zeros(dim_m, dtype=DTYPE, device=DEVICE)

    constitutive_mm = compliance_voigt.T @ FROBENIUS_WEIGHT_MATRIX @ compliance_voigt
    constitutive_mu = compliance_voigt.T @ FROBENIUS_WEIGHT_MATRIX

    for comp_i in range(3):
        for comp_j in range(3):
            H_mm[comp_i::3, comp_j::3] += constitutive_mm[comp_i, comp_j] * gram_xi_m
            H_mm[comp_i::3, comp_j::3] += (
                EQUILIBRIUM_DIVDIV_WEIGHTS[comp_i]
                * EQUILIBRIUM_DIVDIV_WEIGHTS[comp_j]
                * hess_gram_m[comp_i][comp_j]
            )
            H_mu[comp_i::3, :] += (
                constitutive_mu[comp_i, comp_j] * cross_xi_m_hess_u[comp_j]
            )

        g_m[comp_i::3] = -EQUILIBRIUM_DIVDIV_WEIGHTS[comp_i] * hess_force_m[comp_i]

    H_uu += hess_gram_u[0][0]
    H_uu += hess_gram_u[1][1]
    H_uu += 2.0 * hess_gram_u[2][2]
    H_uu += cfg.lambda_0 * gram_bc
    H_uu += cfg.lambda_1 * gram_dn

    H = torch.zeros(dim_m + mp1_u, dim_m + mp1_u, dtype=DTYPE, device=DEVICE)
    H[:dim_m, :dim_m] = H_mm
    H[:dim_m, dim_m:] = H_mu
    H[dim_m:, :dim_m] = H_mu.T
    H[dim_m:, dim_m:] = H_uu
    H = 0.5 * (H + H.T)

    g = torch.zeros(dim_m + mp1_u, dtype=DTYPE, device=DEVICE)
    g[:dim_m] = g_m
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


def split_solution(z: torch.Tensor, dim_m: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the full coefficient vector into moment and deflection parts."""

    return z[:dim_m], z[dim_m:]


def run_algorithm(
    algorithm_id: str,
    data: StrongExperimentData,
    cfg: StrongConfig,
) -> AlgorithmResult:
    """Run one configured strong-form algorithm and evaluate it."""

    if algorithm_id == "eigh":
        print("Running Strong (Eigh)...")
        z, wall_time = solve_eigh(data.H, data.g, cfg.eigh_rtol)
        m, u = split_solution(z, data.dim_m)
        result = evaluate_feature_result("Strong (Eigh)", wall_time, m, u, data.eval_data)
    else:
        print("Running Strong (Lstsq)...")
        z, wall_time = solve_lstsq(data.H, data.g)
        m, u = split_solution(z, data.dim_m)
        result = evaluate_feature_result("Strong (Lstsq)", wall_time, m, u, data.eval_data)

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
        f"Config: h={cfg.h}, M_m={cfg.M_m}, M_u={cfg.M_u}, "
        f"Q_int={cfg.Q_int}, Q_bc={cfg.Q_bc}, Q_test={cfg.Q_test}, "
        f"gamma_m={cfg.gamma_m}, gamma_u={cfg.gamma_u}, "
        f"lambda_0={cfg.lambda_0:.2e}, lambda_1={cfg.lambda_1:.2e}, "
        f"eigh_rtol={cfg.eigh_rtol:.2e}, sampling={cfg.sampling_method}"
    )
    print(f"Algorithms: {selected_algorithm_ids}")

    D = compute_bending_stiffness(cfg.E, cfg.nu, cfg.h)
    print(f"Material: E={cfg.E}, nu={cfg.nu}, h={cfg.h}, D={D:.4f}")

    if benchmark is None:
        print("Building benchmark data...")
        benchmark = build_shared_benchmark(
            E=cfg.E,
            nu=cfg.nu,
            h=cfg.h,
            Q_int=cfg.Q_int,
            Q_bc=cfg.Q_bc,
            Q_test=cfg.Q_test,
            sampling_method=cfg.sampling_method,
        )
    else:
        print("Using shared benchmark data...")

    if feature_space is None:
        print("Generating random feature spaces...")
        feature_space = build_shared_feature_space(
            M_m=cfg.M_m,
            M_u=cfg.M_u,
            gamma_m=cfg.gamma_m,
            gamma_u=cfg.gamma_u,
        )
    else:
        print("Using shared random feature spaces...")

    if feature_space.gamma_m != cfg.gamma_m or feature_space.gamma_u != cfg.gamma_u:
        raise ValueError("SharedFeatureSpace gamma does not match StrongConfig.")

    print("Assembling strong-form normal equations...")
    H, g = assemble_normal_equations(
        cfg,
        benchmark.compliance_voigt,
        benchmark.x_int,
        benchmark.f_int,
        benchmark.x_bc,
        benchmark.w_bc,
        benchmark.n_bc,
        feature_space.a_m,
        feature_space.r_m,
        feature_space.a_u,
        feature_space.r_u,
    )
    clear_cuda_cache()
    print(f"System shapes: H={tuple(H.shape)}, g={tuple(g.shape)}")

    experiment_data = StrongExperimentData(
        H=H,
        g=g,
        dim_m=3 * (cfg.M_m + 1),
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
