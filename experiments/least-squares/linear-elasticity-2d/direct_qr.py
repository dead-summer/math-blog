from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass

import scipy.linalg
import torch

import linear_elasticity_2d as elasticity


@dataclass(frozen=True)
class DirectDesign:
    matrix: torch.Tensor
    rhs: torch.Tensor
    solved_dim_s: int
    stress_adapter: elasticity.StressBasisAdapter


def assemble_direct_design(
    cfg: elasticity.LeastSquaresConfig,
    benchmark: elasticity.SharedBenchmarkData,
    feature_space: elasticity.SharedFeatureSpace,
) -> DirectDesign:
    """Assemble the weighted constitutive/equilibrium residual matrix directly."""

    if elasticity.DEVICE.type != "cpu":
        raise RuntimeError("The direct QR prototype currently requires the CPU device.")

    x = benchmark.x_int
    sqrt_weights = torch.sqrt(benchmark.w_int)
    raw_sigma = elasticity.eval_raw_scalar_basis(
        x,
        feature_space.a_s,
        feature_space.r_s,
        feature_space.gamma_s,
    )
    grad_sigma = elasticity.eval_raw_scalar_basis_grads(
        x,
        feature_space.a_s,
        feature_space.r_s,
        feature_space.gamma_s,
    )
    _, grad_u = elasticity.eval_active_displacement_basis_data(
        x,
        feature_space.a_u,
        feature_space.r_u,
        feature_space.gamma_u,
    )

    mean_raw_sigma = (benchmark.w_int.unsqueeze(1) * raw_sigma).sum(dim=0)
    stress_adapter = elasticity.build_stress_basis_adapter(mean_raw_sigma)
    q_count = x.shape[0]
    np1_s = raw_sigma.shape[1]
    np1_u = grad_u.shape[1]
    raw_dim_s = 3 * np1_s
    active_dim_s = stress_adapter.active_dim
    dim_u = 2 * np1_u

    raw_stress_matrix = torch.zeros(
        5 * q_count,
        raw_dim_s,
        dtype=elasticity.DTYPE,
        device=elasticity.DEVICE,
    )
    displacement_matrix = torch.zeros(
        5 * q_count,
        dim_u,
        dtype=elasticity.DTYPE,
        device=elasticity.DEVICE,
    )
    rhs = torch.zeros(5 * q_count, dtype=elasticity.DTYPE, device=elasticity.DEVICE)

    mu, lam = elasticity.compute_lame_constants(cfg.E, cfg.nu)
    compliance = elasticity.build_compliance_matrix(mu, lam)
    weighted_sigma = sqrt_weights.unsqueeze(1) * raw_sigma
    for residual_component in range(3):
        rows = slice(residual_component * q_count, (residual_component + 1) * q_count)
        for stress_component in range(3):
            raw_stress_matrix[rows, stress_component:raw_dim_s:3] = (
                compliance[residual_component, stress_component] * weighted_sigma
            )
        for spatial_dimension in range(2):
            coupling = elasticity.STRAIN_GRAD_BASES[spatial_dimension]
            for displacement_component in range(2):
                displacement_matrix[rows, displacement_component:dim_u:2] -= (
                    coupling[residual_component, displacement_component]
                    * sqrt_weights.unsqueeze(1)
                    * grad_u[:, :, spatial_dimension]
                )

    for equilibrium_component in range(2):
        rows = slice(
            (3 + equilibrium_component) * q_count,
            (4 + equilibrium_component) * q_count,
        )
        for spatial_dimension in range(2):
            coupling = elasticity.STRAIN_GRAD_BASES[spatial_dimension]
            for stress_component in range(3):
                raw_stress_matrix[rows, stress_component:raw_dim_s:3] += (
                    coupling[stress_component, equilibrium_component]
                    * sqrt_weights.unsqueeze(1)
                    * grad_sigma[:, :, spatial_dimension]
                )
        rhs[rows] = -sqrt_weights * benchmark.f_int[:, equilibrium_component]

    matrix = torch.empty(
        5 * q_count,
        active_dim_s + dim_u,
        dtype=elasticity.DTYPE,
        device=elasticity.DEVICE,
    )
    matrix[:, :active_dim_s] = raw_stress_matrix @ stress_adapter.transform
    matrix[:, active_dim_s:] = displacement_matrix
    return DirectDesign(
        matrix=matrix,
        rhs=rhs,
        solved_dim_s=active_dim_s,
        stress_adapter=stress_adapter,
    )


def solve_scaled_direct(
    design: DirectDesign,
    driver: str,
    rcond: float | None,
) -> tuple[torch.Tensor, float, int, float]:
    matrix = design.matrix
    column_norms = torch.linalg.vector_norm(matrix, dim=0)
    floor = torch.finfo(matrix.dtype).eps * column_norms.max()
    safe_norms = column_norms.clamp_min(floor)
    matrix /= safe_norms.unsqueeze(0)

    t0 = time.perf_counter()
    scaled_solution, _, rank, singular_values = scipy.linalg.lstsq(
        matrix.numpy(),
        design.rhs.numpy(),
        cond=rcond,
        overwrite_a=True,
        overwrite_b=False,
        check_finite=False,
        lapack_driver=driver,
    )
    wall_time = time.perf_counter() - t0
    solution = torch.from_numpy(scaled_solution).to(dtype=elasticity.DTYPE) / safe_norms
    if singular_values is None or len(singular_values) == 0:
        condition_estimate = float("nan")
    else:
        positive = singular_values[singular_values > 0.0]
        condition_estimate = (
            float(positive.max() / positive.min()) if len(positive) else float("inf")
        )
    return solution, wall_time, int(rank), condition_estimate


def check_normal_equation_consistency(
    cfg: elasticity.LeastSquaresConfig,
    benchmark: elasticity.SharedBenchmarkData,
    feature_space: elasticity.SharedFeatureSpace,
    design: DirectDesign,
) -> tuple[float, float]:
    assembled = elasticity.assemble_linear_system(
        cfg,
        benchmark.x_int,
        benchmark.w_int,
        benchmark.f_int,
        feature_space.a_s,
        feature_space.r_s,
        feature_space.a_u,
        feature_space.r_u,
    )
    direct_gram = design.matrix.T @ design.matrix
    direct_load = design.matrix.T @ design.rhs
    gram_error = torch.linalg.vector_norm(direct_gram - assembled.G) / torch.linalg.vector_norm(
        assembled.G
    )
    load_error = torch.linalg.vector_norm(direct_load - assembled.F) / torch.linalg.vector_norm(
        assembled.F
    )
    return gram_error.item(), load_error.item()


def run(args: argparse.Namespace) -> dict[str, float | int | str]:
    cfg = elasticity.LeastSquaresConfig(
        N_s=args.N,
        N_u=args.N,
        gamma_s=args.gamma,
        gamma_u=args.gamma,
        Q_train=args.q_train,
        Q_test=args.q_test,
        sampling_method=args.sampling,
        manufactured_solution=args.manufactured_solution,
        algorithms_to_run=["ridge"],
    )
    benchmark = elasticity.build_shared_benchmark(
        E=cfg.E,
        nu=cfg.nu,
        Q_train=cfg.Q_train,
        Q_test=cfg.Q_test,
        sampling_method=cfg.sampling_method,
        body_force_batch_size=cfg.body_force_batch_size,
        manufactured_solution=cfg.manufactured_solution,
    )
    feature_space = elasticity.build_shared_feature_space(
        N_s=cfg.N_s,
        N_u=cfg.N_u,
        gamma_s=cfg.gamma_s,
        gamma_u=cfg.gamma_u,
    )

    t0 = time.perf_counter()
    design = assemble_direct_design(cfg, benchmark, feature_space)
    assemble_time = time.perf_counter() - t0
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
        args.driver,
        args.rcond,
    )
    active_sigma = solution[: design.solved_dim_s]
    displacement = solution[design.solved_dim_s :]
    raw_sigma = elasticity.lift_active_stress_coefficients(
        active_sigma,
        design.stress_adapter,
    )
    sigma_basis_test = elasticity.eval_raw_scalar_basis(
        benchmark.x_test,
        feature_space.a_s,
        feature_space.r_s,
        feature_space.gamma_s,
    )
    displacement_basis_test = elasticity.eval_active_displacement_basis(
        benchmark.x_test,
        feature_space.a_u,
        feature_space.r_u,
        feature_space.gamma_u,
    )
    u_l2_error, sigma_l2_error = elasticity.compute_absolute_errors(
        displacement_basis_test,
        sigma_basis_test,
        raw_sigma,
        displacement,
        benchmark.w_test,
        benchmark.u_exact_test,
        benchmark.sigma_exact_test,
    )
    return {
        "method": f"Direct {args.driver.upper()}",
        "N": args.N,
        "Q_train": args.q_train,
        "Q_test": args.q_test,
        "sampling": args.sampling,
        "manufactured_solution": args.manufactured_solution,
        "driver": args.driver,
        "rcond": args.rcond if args.rcond is not None else "lapack-default",
        "rows": design.matrix.shape[0],
        "columns": design.matrix.shape[1],
        "matrix_mib": design.matrix.numel() * design.matrix.element_size() / (1024**2),
        "rank": rank,
        "feature_family": "constant+random-tanh",
        "condition_estimate": condition_estimate,
        "assemble_time": assemble_time,
        "solve_time": solve_time,
        "u_l2_error": u_l2_error,
        "sigma_l2_error": sigma_l2_error,
        "gram_relative_error": gram_error,
        "load_relative_error": load_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct rank-revealing least squares for 2D linear elasticity."
    )
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--q-train", type=int, default=64**2)
    parser.add_argument("--q-test", type=int, default=128**2)
    parser.add_argument(
        "--sampling",
        choices=elasticity.VALID_SAMPLING_METHODS,
        default="gauss_legendre",
    )
    parser.add_argument(
        "--manufactured-solution",
        choices=elasticity.VALID_MANUFACTURED_SOLUTIONS,
        default="hu_zhang",
    )
    parser.add_argument("--gamma", type=float, default=3.0)
    parser.add_argument("--driver", choices=("gelsy", "gelsd", "gelss"), default="gelsd")
    parser.add_argument("--rcond", type=float, default=1.0e-14)
    parser.add_argument("--check-gram", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.N <= 0:
        raise ValueError("N must be positive.")
    if args.sampling == "gauss_legendre":
        for name, value in (("q_train", args.q_train), ("q_test", args.q_test)):
            order = math.isqrt(value)
            if order * order != value:
                raise ValueError(f"{name} must be a perfect square for Gauss-Legendre.")
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
