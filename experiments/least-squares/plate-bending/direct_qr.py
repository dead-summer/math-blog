from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass

import scipy.linalg
import torch

import plate_bending as plate


@dataclass(frozen=True)
class DirectDesign:
    matrix: torch.Tensor
    rhs: torch.Tensor
    dim_m: int


def assemble_direct_design(
    benchmark: plate.SharedBenchmarkData,
    feature_space: plate.SharedFeatureSpace,
) -> DirectDesign:
    """Assemble the weighted residual design matrix without forming normal equations."""

    if plate.DEVICE.type != "cpu":
        raise RuntimeError("The direct QR prototype currently requires the CPU device.")

    x = benchmark.x_int
    weights = benchmark.w_int
    force = benchmark.f_int
    sqrt_weights = torch.sqrt(weights)

    xi_m = plate.eval_features(
        x,
        feature_space.a_m,
        feature_space.r_m,
        feature_space.gamma_m,
    )
    hess_m = plate.eval_feature_hessians(
        x,
        feature_space.a_m,
        feature_space.r_m,
        feature_space.gamma_m,
    )
    _, hess_u = plate.eval_active_deflection_feature_data(
        x,
        feature_space.a_u,
        feature_space.r_u,
        feature_space.gamma_u,
    )

    q_count = x.shape[0]
    mp1_m = xi_m.shape[1]
    mp1_u = hess_u.shape[1]
    dim_m = 3 * mp1_m
    column_count = dim_m + mp1_u
    matrix = torch.zeros(
        4 * q_count,
        column_count,
        dtype=plate.DTYPE,
        device=plate.DEVICE,
    )
    rhs = torch.zeros(4 * q_count, dtype=plate.DTYPE, device=plate.DEVICE)

    compliance = benchmark.compliance_voigt
    sqrt_frobenius = torch.sqrt(plate.FROBENIUS_WEIGHT)
    weighted_xi = sqrt_weights.unsqueeze(1) * xi_m

    for residual_component in range(3):
        rows = slice(residual_component * q_count, (residual_component + 1) * q_count)
        residual_scale = sqrt_frobenius[residual_component]
        for moment_component in range(3):
            matrix[rows, moment_component:dim_m:3] = (
                residual_scale
                * compliance[residual_component, moment_component]
                * weighted_xi
            )
        matrix[rows, dim_m:] = (
            residual_scale
            * sqrt_weights.unsqueeze(1)
            * hess_u[:, :, residual_component]
        )

    equilibrium_rows = slice(3 * q_count, 4 * q_count)
    for moment_component in range(3):
        matrix[equilibrium_rows, moment_component:dim_m:3] = (
            plate.DIVDIV_WEIGHTS[moment_component]
            * sqrt_weights.unsqueeze(1)
            * hess_m[:, :, moment_component]
        )
    rhs[equilibrium_rows] = -sqrt_weights * force
    return DirectDesign(matrix=matrix, rhs=rhs, dim_m=dim_m)


def solve_scaled_direct(
    design: DirectDesign,
    driver: str,
    rcond: float | None,
) -> tuple[torch.Tensor, float, int, float]:
    """Column-scale the direct design and solve it with rank-revealing LAPACK."""

    matrix = design.matrix
    rhs = design.rhs
    column_norms = torch.linalg.vector_norm(matrix, dim=0)
    scale_floor = torch.finfo(matrix.dtype).eps * column_norms.max()
    safe_norms = column_norms.clamp_min(scale_floor)
    matrix /= safe_norms.unsqueeze(0)

    matrix_np = matrix.numpy()
    rhs_np = rhs.numpy()
    t0 = time.perf_counter()
    scaled_solution, _, rank, singular_values = scipy.linalg.lstsq(
        matrix_np,
        rhs_np,
        cond=rcond,
        overwrite_a=True,
        overwrite_b=False,
        check_finite=False,
        lapack_driver=driver,
    )
    wall_time = time.perf_counter() - t0

    solution = torch.from_numpy(scaled_solution).to(dtype=plate.DTYPE) / safe_norms
    if singular_values is None or len(singular_values) == 0:
        condition_estimate = float("nan")
    else:
        positive = singular_values[singular_values > 0.0]
        condition_estimate = (
            float(positive.max() / positive.min()) if len(positive) else float("inf")
        )
    return solution, wall_time, int(rank), condition_estimate


def check_normal_equation_consistency(
    cfg: plate.LeastSquaresConfig,
    benchmark: plate.SharedBenchmarkData,
    feature_space: plate.SharedFeatureSpace,
    design: DirectDesign,
) -> tuple[float, float]:
    """Check that the direct residual matrix reproduces the legacy G and F."""

    gram, load = plate.assemble_linear_system(
        cfg,
        benchmark.compliance_voigt,
        benchmark.x_int,
        benchmark.w_int,
        benchmark.f_int,
        feature_space.a_m,
        feature_space.r_m,
        feature_space.a_u,
        feature_space.r_u,
    )
    direct_gram = design.matrix.T @ design.matrix
    direct_load = design.matrix.T @ design.rhs
    gram_error = torch.linalg.vector_norm(direct_gram - gram) / torch.linalg.vector_norm(gram)
    load_error = torch.linalg.vector_norm(direct_load - load) / torch.linalg.vector_norm(load)
    return gram_error.item(), load_error.item()


def evaluate_direct_result(
    benchmark: plate.SharedBenchmarkData,
    feature_space: plate.SharedFeatureSpace,
    moment_coeffs: torch.Tensor,
    deflection_coeffs: torch.Tensor,
) -> tuple[float, float, float, float]:
    """Evaluate residuals and relative errors for the random-feature space."""

    xi_test = plate.eval_features(
        benchmark.x_test,
        feature_space.a_m,
        feature_space.r_m,
        feature_space.gamma_m,
    )
    psi_test = plate.eval_active_deflection_features(
        benchmark.x_test,
        feature_space.a_u,
        feature_space.r_u,
        feature_space.gamma_u,
    )
    moment_blocks = moment_coeffs.reshape(-1, 3)
    moment_test = xi_test @ moment_blocks
    deflection_test = psi_test @ deflection_coeffs

    u_error = torch.sqrt(
        (benchmark.w_test * (deflection_test - benchmark.u_exact_test).square()).sum()
    )
    u_reference = torch.sqrt(
        (benchmark.w_test * benchmark.u_exact_test.square()).sum()
    )
    moment_error = torch.sqrt(
        (
            benchmark.w_test
            * (
                plate.FROBENIUS_WEIGHT
                * (moment_test - benchmark.M_exact_test).square()
            ).sum(dim=1)
        ).sum()
    )
    moment_reference = torch.sqrt(
        (
            benchmark.w_test
            * (plate.FROBENIUS_WEIGHT * benchmark.M_exact_test.square()).sum(dim=1)
        ).sum()
    )

    xi_train = plate.eval_features(
        benchmark.x_int,
        feature_space.a_m,
        feature_space.r_m,
        feature_space.gamma_m,
    )
    hess_m_train = plate.eval_feature_hessians(
        benchmark.x_int,
        feature_space.a_m,
        feature_space.r_m,
        feature_space.gamma_m,
    )
    _, hess_u_train = plate.eval_active_deflection_feature_data(
        benchmark.x_int,
        feature_space.a_u,
        feature_space.r_u,
        feature_space.gamma_u,
    )
    moment_train = xi_train @ moment_blocks
    hessian_u = torch.einsum("qfj,f->qj", hess_u_train, deflection_coeffs)
    constitutive = moment_train @ benchmark.compliance_voigt.T + hessian_u
    equilibrium = (
        hess_m_train[:, :, 0] @ moment_blocks[:, 0]
        + hess_m_train[:, :, 1] @ moment_blocks[:, 1]
        + 2.0 * (hess_m_train[:, :, 2] @ moment_blocks[:, 2])
        + benchmark.f_int
    )
    constitutive_norm = torch.sqrt(
        (
            benchmark.w_int
            * (plate.FROBENIUS_WEIGHT * constitutive.square()).sum(dim=1)
        ).sum()
    )
    equilibrium_norm = torch.sqrt(
        (benchmark.w_int * equilibrium.square()).sum()
    )
    return (
        constitutive_norm.item(),
        equilibrium_norm.item(),
        (u_error / u_reference).item(),
        (moment_error / moment_reference).item(),
    )


def run(args: argparse.Namespace) -> dict[str, float | int | str]:
    cfg = plate.LeastSquaresConfig(
        N_m=args.N,
        N_u=args.N,
        gamma_m=args.gamma,
        gamma_u=args.gamma,
        Q_train=args.q_train,
        Q_test=args.q_test,
        sampling_method=args.sampling,
        algorithms_to_run=["ridge"],
    )
    benchmark = plate.build_shared_benchmark(
        E=cfg.E,
        nu=cfg.nu,
        h=cfg.h,
        Q_train=cfg.Q_train,
        Q_test=cfg.Q_test,
        sampling_method=cfg.sampling_method,
    )
    feature_space = plate.build_shared_feature_space(
        N_m=cfg.N_m,
        N_u=cfg.N_u,
        gamma_m=cfg.gamma_m,
        gamma_u=cfg.gamma_u,
    )

    assemble_start = time.perf_counter()
    design = assemble_direct_design(benchmark, feature_space)
    assemble_time = time.perf_counter() - assemble_start
    matrix_mebibytes = design.matrix.numel() * design.matrix.element_size() / (1024**2)

    gram_error = float("nan")
    load_error = float("nan")
    if args.check_gram:
        gram_error, load_error = check_normal_equation_consistency(
            cfg,
            benchmark,
            feature_space,
            design,
        )

    solution, solve_time, rank, condition_estimate = solve_scaled_direct(
        design,
        driver=args.driver,
        rcond=args.rcond,
    )
    moment_coeffs = solution[: design.dim_m]
    deflection_coeffs = solution[design.dim_m :]
    r_c, r_e, rel_u, rel_m = evaluate_direct_result(
        benchmark,
        feature_space,
        moment_coeffs,
        deflection_coeffs,
    )

    return {
        "method": f"Direct {args.driver.upper()}",
        "N": args.N,
        "Q_train": args.q_train,
        "Q_test": args.q_test,
        "sampling": args.sampling,
        "driver": args.driver,
        "rcond": args.rcond if args.rcond is not None else "lapack-default",
        "rows": design.matrix.shape[0],
        "columns": design.matrix.shape[1],
        "matrix_mib": matrix_mebibytes,
        "rank": rank,
        "feature_family": "constant+random-tanh",
        "condition_estimate": condition_estimate,
        "assemble_time": assemble_time,
        "solve_time": solve_time,
        "constitutive_residual": r_c,
        "equilibrium_residual": r_e,
        "relative_u_error": rel_u,
        "relative_moment_error": rel_m,
        "gram_relative_error": gram_error,
        "load_relative_error": load_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct rank-revealing least-squares prototype for plate bending."
    )
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--q-train", type=int, default=64**2)
    parser.add_argument("--q-test", type=int, default=128**2)
    parser.add_argument(
        "--sampling",
        choices=plate.VALID_SAMPLING_METHODS,
        default="gauss_legendre",
    )
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--driver", choices=("gelsy", "gelsd", "gelss"), default="gelsy")
    parser.add_argument("--rcond", type=float, default=None)
    parser.add_argument("--check-gram", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.N <= 0:
        raise ValueError("N must be positive.")
    if args.q_train <= 0 or args.q_test <= 0:
        raise ValueError("Quadrature sizes must be positive.")
    if args.sampling == "gauss_legendre":
        for name, value in (("q_train", args.q_train), ("q_test", args.q_test)):
            order = math.isqrt(value)
            if order * order != value:
                raise ValueError(f"{name} must be a perfect square for Gauss-Legendre.")
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
