from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch

from plate_bending import (
    DEVICE,
    DTYPE,
    AlgorithmResult,
    FeatureEvaluationData,
    SharedBenchmarkData,
    SharedFeatureSpace,
    build_compliance_bilinear_matrix,
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


VALID_WEAK_ALGORITHMS = ("eigh", "lstsq")


@dataclass
class WeakConfig:
    """Configuration for the boundary-penalized weak-form experiment."""

    E: float = 1.0
    nu: float = 0.3
    h: float = 1.0
    gamma_m: float = 2.0
    gamma_u: float = 2.0
    M_m: int = 300
    M_u: int = 300
    Q_int: int = (2 ** 8) ** 2
    Q_bc: int = 4 * (2 ** 7)
    Q_test: int = (2 ** 7) ** 2
    sampling_method: str = "sobol"
    lambda_0: float = 1.0
    lambda_1: float = 1.0
    eigh_rtol: float = 1.0e-15
    assembly_batch_size: int = 5_000
    algorithms_to_run: list[str] = field(
        default_factory=lambda: [
            "eigh",
            "lstsq",
        ]
    )


@dataclass(frozen=True)
class WeakExperimentData:
    """All tensors needed to run and evaluate one weak-form solver."""

    A: torch.Tensor
    B: torch.Tensor
    C0: torch.Tensor
    C1: torch.Tensor
    F: torch.Tensor
    eval_data: FeatureEvaluationData


def validate_config(cfg: WeakConfig) -> None:
    """Validate config before starting any expensive work."""

    if cfg.E <= 0.0:
        raise ValueError("WeakConfig.E must be positive.")
    if not (-1.0 < cfg.nu < 0.5):
        raise ValueError("WeakConfig.nu must lie in (-1, 0.5).")
    if cfg.h <= 0.0:
        raise ValueError("WeakConfig.h must be positive.")
    if cfg.gamma_m <= 0.0 or cfg.gamma_u <= 0.0:
        raise ValueError("WeakConfig.gamma_m and WeakConfig.gamma_u must be positive.")
    if cfg.M_m <= 0 or cfg.M_u <= 0:
        raise ValueError("WeakConfig.M_m and WeakConfig.M_u must be positive.")
    if cfg.Q_int <= 0:
        raise ValueError("WeakConfig.Q_int must be positive.")
    if cfg.Q_bc < 4:
        raise ValueError("WeakConfig.Q_bc must be at least 4.")
    if cfg.Q_test <= 0:
        raise ValueError("WeakConfig.Q_test must be positive.")
    if not math.isfinite(cfg.lambda_0) or cfg.lambda_0 <= 0.0:
        raise ValueError("WeakConfig.lambda_0 must be finite and positive.")
    if not math.isfinite(cfg.lambda_1) or cfg.lambda_1 <= 0.0:
        raise ValueError("WeakConfig.lambda_1 must be finite and positive.")
    if not math.isfinite(cfg.eigh_rtol) or cfg.eigh_rtol < 0.0:
        raise ValueError("WeakConfig.eigh_rtol must be finite and non-negative.")
    if cfg.assembly_batch_size <= 0:
        raise ValueError("WeakConfig.assembly_batch_size must be positive.")
    validate_sampling_method(cfg.sampling_method)
    validate_algorithm_selection(cfg.algorithms_to_run, VALID_WEAK_ALGORITHMS)


def accumulate_weak_form_moments(
    x_int: torch.Tensor,
    f_int: torch.Tensor,
    a_m: torch.Tensor,
    r_m: torch.Tensor,
    gamma_m: float,
    a_u: torch.Tensor,
    r_u: torch.Tensor,
    gamma_u: float,
    batch_size: int,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Accumulate interior moments for A, B, F assembly."""

    mp1_m = a_m.shape[0] + 1
    mp1_u = a_u.shape[0] + 1
    weight = 1.0 / x_int.shape[0]

    gram_m = torch.zeros(mp1_m, mp1_m, dtype=DTYPE, device=DEVICE)
    cross_u_hess_m = [
        torch.zeros(mp1_u, mp1_m, dtype=DTYPE, device=DEVICE)
        for _ in range(3)
    ]
    force_moment = torch.zeros(mp1_u, dtype=DTYPE, device=DEVICE)

    with torch.no_grad():
        for start in range(0, x_int.shape[0], batch_size):
            end = min(start + batch_size, x_int.shape[0])
            xb = x_int[start:end]
            fb = f_int[start:end]

            xi_m_batch = eval_features(xb, a_m, r_m, gamma_m)
            hess_m_batch = eval_feature_hessians(xb, a_m, r_m, gamma_m)
            xi_u_batch = eval_features(xb, a_u, r_u, gamma_u)

            gram_m += weight * (xi_m_batch.T @ xi_m_batch)
            force_moment += weight * (xi_u_batch.T @ fb)
            cross_u_hess_m[0] += weight * (xi_u_batch.T @ hess_m_batch[:, :, 0])
            cross_u_hess_m[1] += weight * (xi_u_batch.T @ hess_m_batch[:, :, 1])
            cross_u_hess_m[2] += weight * (xi_u_batch.T @ (2.0 * hess_m_batch[:, :, 2]))

    return gram_m, cross_u_hess_m, force_moment


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


def assemble_moment_matrix(
    gram_m: torch.Tensor,
    compliance_bilinear: torch.Tensor,
) -> torch.Tensor:
    """Assemble the moment block A."""

    dim_m = 3 * gram_m.shape[0]
    A = torch.zeros(dim_m, dim_m, dtype=DTYPE, device=DEVICE)
    for row in range(3):
        for col in range(3):
            A[row::3, col::3] = compliance_bilinear[row, col] * gram_m
    return A


def assemble_coupling_matrix(cross_u_hess_m: list[torch.Tensor]) -> torch.Tensor:
    """Assemble the weak-form coupling block B."""

    mp1_u = cross_u_hess_m[0].shape[0]
    mp1_m = cross_u_hess_m[0].shape[1]

    B = torch.zeros(3 * mp1_m, mp1_u, dtype=DTYPE, device=DEVICE)
    for comp in range(3):
        B[comp::3, :] = cross_u_hess_m[comp].T
    return B


def assemble_system(
    gram_m: torch.Tensor,
    cross_u_hess_m: list[torch.Tensor],
    gram_bc: torch.Tensor,
    gram_dn: torch.Tensor,
    force_moment: torch.Tensor,
    compliance_voigt: torch.Tensor,
    lambda_0: float,
    lambda_1: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble the weak-form linear system blocks."""

    A = assemble_moment_matrix(
        gram_m,
        build_compliance_bilinear_matrix(compliance_voigt),
    )
    B = assemble_coupling_matrix(cross_u_hess_m)
    C0 = lambda_0 * gram_bc
    C1 = lambda_1 * gram_dn
    F = force_moment
    return A, B, C0, C1, F


def build_kkt_system(
    A: torch.Tensor,
    B: torch.Tensor,
    C0: torch.Tensor,
    C1: torch.Tensor,
    F: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the KKT matrix K z = rhs."""

    dim_m = A.shape[0]
    dim_u = B.shape[1]
    K = torch.zeros(dim_m + dim_u, dim_m + dim_u, dtype=DTYPE, device=DEVICE)
    K[:dim_m, :dim_m] = A
    K[:dim_m, dim_m:] = B
    K[dim_m:, :dim_m] = B.T
    K[dim_m:, dim_m:] = C0 + C1

    rhs = torch.zeros(dim_m + dim_u, dtype=DTYPE, device=DEVICE)
    rhs[dim_m:] = -F
    return K, rhs


def solve_lstsq(
    A: torch.Tensor,
    B: torch.Tensor,
    C0: torch.Tensor,
    C1: torch.Tensor,
    F: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Solve the weak KKT system with torch.linalg.lstsq."""

    dim_m = A.shape[0]
    K, rhs = build_kkt_system(A, B, C0, C1, F)

    synchronize_device()
    t0 = time.perf_counter()
    try:
        sol = torch.linalg.lstsq(K, rhs.unsqueeze(1)).solution.squeeze(1)
        if not torch.isfinite(sol).all():
            raise RuntimeError("non-finite solution")
    except (RuntimeError, torch.linalg.LinAlgError) as exc:
        sol = torch.full((K.shape[0],), float("nan"), dtype=DTYPE, device=DEVICE)
        print(f"    Warning: torch.linalg.lstsq failed with {type(exc).__name__}")

    synchronize_device()
    wall_time = time.perf_counter() - t0
    return sol[:dim_m], sol[dim_m:], wall_time


def solve_eigh(
    A: torch.Tensor,
    B: torch.Tensor,
    C0: torch.Tensor,
    C1: torch.Tensor,
    F: torch.Tensor,
    rtol: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Solve the symmetric weak KKT system with truncated eigen decomposition."""

    dim_m = A.shape[0]
    K, rhs = build_kkt_system(A, B, C0, C1, F)

    synchronize_device()
    t0 = time.perf_counter()
    try:
        eigvals, eigvecs = torch.linalg.eigh(K)
        threshold = rtol * eigvals.abs().max()
        keep = eigvals.abs() > threshold
        if not keep.any():
            raise RuntimeError("all eigenvalues were truncated")

        coeffs = eigvecs[:, keep].T @ rhs
        coeffs = coeffs / eigvals[keep]
        sol = eigvecs[:, keep] @ coeffs
        if not torch.isfinite(sol).all():
            raise RuntimeError("non-finite solution")
        print(
            f"    eigh truncation: kept {int(keep.sum().item())}/{eigvals.numel()} "
            f"eigenvalues, threshold={threshold.item():.2e}"
        )
    except (RuntimeError, torch.linalg.LinAlgError) as exc:
        sol = torch.full((K.shape[0],), float("nan"), dtype=DTYPE, device=DEVICE)
        print(f"    Warning: torch.linalg.eigh failed with {type(exc).__name__}")

    synchronize_device()
    wall_time = time.perf_counter() - t0
    return sol[:dim_m], sol[dim_m:], wall_time


def run_algorithm(
    algorithm_id: str,
    data: WeakExperimentData,
    cfg: WeakConfig,
) -> AlgorithmResult:
    """Run one configured weak-form algorithm and evaluate it."""

    if algorithm_id == "eigh":
        print("Running Weak (Eigh)...")
        m, u, wall_time = solve_eigh(
            data.A,
            data.B,
            data.C0,
            data.C1,
            data.F,
            cfg.eigh_rtol,
        )
        result = evaluate_feature_result("Weak (Eigh)", wall_time, m, u, data.eval_data)
    else:
        print("Running Weak (Lstsq)...")
        m, u, wall_time = solve_lstsq(data.A, data.B, data.C0, data.C1, data.F)
        result = evaluate_feature_result("Weak (Lstsq)", wall_time, m, u, data.eval_data)

    print_result_summary(result)
    return result


def run_experiment(
    cfg: WeakConfig | None = None,
    print_table: bool = True,
    benchmark: SharedBenchmarkData | None = None,
    feature_space: SharedFeatureSpace | None = None,
) -> list[AlgorithmResult]:
    """Run the selected weak-form methods and return their metrics."""

    cfg = WeakConfig() if cfg is None else cfg
    validate_config(cfg)
    selected_algorithm_ids = validate_algorithm_selection(
        cfg.algorithms_to_run,
        VALID_WEAK_ALGORITHMS,
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
        raise ValueError("SharedFeatureSpace gamma does not match WeakConfig.")

    print("Assembling weak-form system...")
    gram_m, cross_u_hess_m, force_moment = accumulate_weak_form_moments(
        benchmark.x_int,
        benchmark.f_int,
        feature_space.a_m,
        feature_space.r_m,
        cfg.gamma_m,
        feature_space.a_u,
        feature_space.r_u,
        cfg.gamma_u,
        cfg.assembly_batch_size,
    )
    gram_bc, gram_dn = accumulate_boundary_grams(
        benchmark.x_bc,
        benchmark.w_bc,
        benchmark.n_bc,
        feature_space.a_u,
        feature_space.r_u,
        cfg.gamma_u,
        cfg.assembly_batch_size,
    )
    A, B, C0, C1, F = assemble_system(
        gram_m,
        cross_u_hess_m,
        gram_bc,
        gram_dn,
        force_moment,
        benchmark.compliance_voigt,
        cfg.lambda_0,
        cfg.lambda_1,
    )
    clear_cuda_cache()
    print(
        f"System shapes: A={tuple(A.shape)}, B={tuple(B.shape)}, "
        f"C0={tuple(C0.shape)}, C1={tuple(C1.shape)}, F={tuple(F.shape)}"
    )

    experiment_data = WeakExperimentData(
        A=A,
        B=B,
        C0=C0,
        C1=C1,
        F=F,
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
        print_summary_table(results, title="Weak Form Summary")
    return results


def main(cfg: WeakConfig | None = None) -> None:
    """Script entrypoint."""

    run_experiment(cfg, print_table=True)


if __name__ == "__main__":
    main()
