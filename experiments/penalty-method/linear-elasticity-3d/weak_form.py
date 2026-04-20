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

VALID_WEAK_ALGORITHMS = ("eigh", "lstsq")


@dataclass
class WeakConfig:
    """Configuration for the boundary-penalized weak-form experiment."""

    E: float = 1.0
    nu: float = 0.3
    gamma_s: float = 2.0
    gamma_u: float = 2.0
    M_s: int = 300
    M_u: int = 300
    Q_int: int = (2 ** 6) ** 3
    Q_bc: int = 6 * (2 ** 5) ** 2
    Q_test: int = (2 ** 5) ** 3
    sampling_method: str = "sobol"
    lambda_bc: float = 1.0
    eigh_rtol: float = 1.0e-15
    body_force_batch_size: int = 5_000
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
    C: torch.Tensor
    F: torch.Tensor
    eval_data: FeatureEvaluationData


def validate_config(cfg: WeakConfig) -> None:
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
    validate_algorithm_selection(cfg.algorithms_to_run, VALID_WEAK_ALGORITHMS)


def accumulate_weak_form_moments(
    x_int: torch.Tensor,
    f_int: torch.Tensor,
    a_s: torch.Tensor,
    r_s: torch.Tensor,
    gamma_s: float,
    a_u: torch.Tensor,
    r_u: torch.Tensor,
    gamma_u: float,
    batch_size: int,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Accumulate interior moments for A, B, F assembly."""

    mp1_s = a_s.shape[0] + 1
    mp1_u = a_u.shape[0] + 1
    weight = 1.0 / x_int.shape[0]

    gram_s = torch.zeros(mp1_s, mp1_s, dtype=DTYPE, device=DEVICE)
    cross_u_grad_s = [
        torch.zeros(mp1_u, mp1_s, dtype=DTYPE, device=DEVICE) for _ in range(3)
    ]
    force_moment = torch.zeros(mp1_u, 3, dtype=DTYPE, device=DEVICE)

    with torch.no_grad():
        for start in range(0, x_int.shape[0], batch_size):
            end = min(start + batch_size, x_int.shape[0])
            xb = x_int[start:end]
            fb = f_int[start:end]

            xi_s_batch = eval_features(xb, a_s, r_s, gamma_s)
            grad_s_batch = eval_feature_grads(xb, a_s, r_s, gamma_s)
            xi_u_batch = eval_features(xb, a_u, r_u, gamma_u)

            gram_s += weight * (xi_s_batch.T @ xi_s_batch)
            force_moment += weight * (xi_u_batch.T @ fb)
            for dim in range(3):
                cross_u_grad_s[dim] += weight * (
                    xi_u_batch.T @ grad_s_batch[:, :, dim]
                )

    return gram_s, cross_u_grad_s, force_moment


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


def assemble_stress_matrix(
    gram_s: torch.Tensor,
    compliance_voigt: torch.Tensor,
) -> torch.Tensor:
    """Assemble the stress block A."""

    dim_s = 6 * gram_s.shape[0]
    A = torch.zeros(dim_s, dim_s, dtype=DTYPE, device=DEVICE)
    for row in range(6):
        for col in range(6):
            A[row::6, col::6] = compliance_voigt[row, col] * gram_s
    return A


def assemble_coupling_matrix(cross_u_grad_s: list[torch.Tensor]) -> torch.Tensor:
    """Assemble the weak-form coupling block B."""

    mp1_u = cross_u_grad_s[0].shape[0]
    mp1_s = cross_u_grad_s[0].shape[1]
    D = [cross_u_grad_s[dim].T for dim in range(3)]

    dim_s = 6 * mp1_s
    dim_u = 3 * mp1_u
    B = torch.zeros(dim_s, dim_u, dtype=DTYPE, device=DEVICE)
    B[0::6, 0::3] = D[0]
    B[1::6, 1::3] = D[1]
    B[2::6, 2::3] = D[2]
    B[3::6, 0::3] = D[1]
    B[3::6, 1::3] = D[0]
    B[4::6, 1::3] = D[2]
    B[4::6, 2::3] = D[1]
    B[5::6, 0::3] = D[2]
    B[5::6, 2::3] = D[0]
    return B


def assemble_boundary_matrix(gram_bc: torch.Tensor) -> torch.Tensor:
    """Assemble the boundary penalty block C."""

    dim_u = 3 * gram_bc.shape[0]
    C = torch.zeros(dim_u, dim_u, dtype=DTYPE, device=DEVICE)
    for comp in range(3):
        C[comp::3, comp::3] = gram_bc
    return C


def assemble_rhs_vector(force_moment: torch.Tensor) -> torch.Tensor:
    """Assemble the displacement right-hand side F."""

    F = torch.zeros(3 * force_moment.shape[0], dtype=DTYPE, device=DEVICE)
    for comp in range(3):
        F[comp::3] = force_moment[:, comp]
    return F


def assemble_system(
    gram_s: torch.Tensor,
    cross_u_grad_s: list[torch.Tensor],
    gram_bc: torch.Tensor,
    force_moment: torch.Tensor,
    compliance_voigt: torch.Tensor,
    lambda_bc: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble the weak-form linear system blocks."""

    A = assemble_stress_matrix(gram_s, compliance_voigt)
    B = assemble_coupling_matrix(cross_u_grad_s)
    C = lambda_bc * assemble_boundary_matrix(gram_bc)
    F = assemble_rhs_vector(force_moment)
    return A, B, C, F


def build_kkt_system(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    F: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the KKT matrix K z = rhs."""

    dim_s = A.shape[0]
    dim_u = B.shape[1]
    K = torch.zeros(dim_s + dim_u, dim_s + dim_u, dtype=DTYPE, device=DEVICE)
    K[:dim_s, :dim_s] = A
    K[:dim_s, dim_s:] = B
    K[dim_s:, :dim_s] = B.T
    K[dim_s:, dim_s:] = C

    rhs = torch.zeros(dim_s + dim_u, dtype=DTYPE, device=DEVICE)
    rhs[dim_s:] = -F
    return K, rhs


def solve_lstsq(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    F: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Solve the weak KKT system with torch.linalg.lstsq."""

    dim_s = A.shape[0]
    K, rhs = build_kkt_system(A, B, C, F)

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
    return sol[:dim_s], sol[dim_s:], wall_time


def solve_eigh(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    F: torch.Tensor,
    rtol: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Solve the symmetric weak KKT system with truncated eigen decomposition."""

    dim_s = A.shape[0]
    K, rhs = build_kkt_system(A, B, C, F)

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
    return sol[:dim_s], sol[dim_s:], wall_time


def run_algorithm(
    algorithm_id: str,
    data: WeakExperimentData,
    cfg: WeakConfig,
) -> AlgorithmResult:
    """Run one configured weak-form algorithm and evaluate it."""

    if algorithm_id == "eigh":
        print("Running Weak (Eigh)...")
        s, u, wall_time = solve_eigh(data.A, data.B, data.C, data.F, cfg.eigh_rtol)
        result = evaluate_feature_result("Weak (Eigh)", wall_time, s, u, data.eval_data)
    else:
        print("Running Weak (Lstsq)...")
        s, u, wall_time = solve_lstsq(data.A, data.B, data.C, data.F)
        result = evaluate_feature_result("Weak (Lstsq)", wall_time, s, u, data.eval_data)

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
        raise ValueError("SharedFeatureSpace gamma does not match WeakConfig.")

    compliance_voigt = benchmark.compliance_voigt

    print("Assembling weak-form system...")
    gram_s, cross_u_grad_s, force_moment = accumulate_weak_form_moments(
        benchmark.x_int,
        benchmark.f_int,
        feature_space.a_s,
        feature_space.r_s,
        cfg.gamma_s,
        feature_space.a_u,
        feature_space.r_u,
        cfg.gamma_u,
        cfg.assembly_batch_size,
    )
    gram_bc = accumulate_boundary_gram(
        benchmark.x_bc,
        benchmark.w_bc,
        feature_space.a_u,
        feature_space.r_u,
        cfg.gamma_u,
        cfg.assembly_batch_size,
    )
    A, B, C, F = assemble_system(
        gram_s,
        cross_u_grad_s,
        gram_bc,
        force_moment,
        compliance_voigt,
        cfg.lambda_bc,
    )
    del gram_s
    del cross_u_grad_s
    del force_moment
    del gram_bc
    clear_cuda_cache()
    print(
        f"System shapes: A={tuple(A.shape)}, B={tuple(B.shape)}, "
        f"C={tuple(C.shape)}, F={tuple(F.shape)}"
    )

    experiment_data = WeakExperimentData(
        A=A,
        B=B,
        C=C,
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
