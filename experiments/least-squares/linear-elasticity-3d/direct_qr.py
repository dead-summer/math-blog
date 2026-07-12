from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass

import scipy.linalg
import torch

import linear_elasticity_3d as elasticity


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
    raw_dim_s = 6 * np1_s
    active_dim_s = stress_adapter.active_dim
    dim_u = 3 * np1_u
    raw_stress_matrix = torch.zeros(
        9 * q_count,
        raw_dim_s,
        dtype=elasticity.DTYPE,
        device=elasticity.DEVICE,
    )
    displacement_matrix = torch.zeros(
        9 * q_count,
        dim_u,
        dtype=elasticity.DTYPE,
        device=elasticity.DEVICE,
    )
    rhs = torch.zeros(9 * q_count, dtype=elasticity.DTYPE, device=elasticity.DEVICE)

    mu, lam = elasticity.compute_lame_constants(cfg.E, cfg.nu)
    compliance = elasticity.build_compliance_matrix(mu, lam)
    weighted_sigma = sqrt_weights.unsqueeze(1) * raw_sigma
    for residual_component in range(6):
        rows = slice(residual_component * q_count, (residual_component + 1) * q_count)
        for stress_component in range(6):
            raw_stress_matrix[rows, stress_component:raw_dim_s:6] = (
                compliance[residual_component, stress_component] * weighted_sigma
            )
        for spatial_dimension in range(3):
            coupling = elasticity.STRAIN_GRAD_BASES[spatial_dimension]
            for displacement_component in range(3):
                displacement_matrix[rows, displacement_component:dim_u:3] -= (
                    coupling[residual_component, displacement_component]
                    * sqrt_weights.unsqueeze(1)
                    * grad_u[:, :, spatial_dimension]
                )

    for equilibrium_component in range(3):
        rows = slice(
            (6 + equilibrium_component) * q_count,
            (7 + equilibrium_component) * q_count,
        )
        for spatial_dimension in range(3):
            coupling = elasticity.STRAIN_GRAD_BASES[spatial_dimension]
            for stress_component in range(6):
                raw_stress_matrix[rows, stress_component:raw_dim_s:6] += (
                    coupling[stress_component, equilibrium_component]
                    * sqrt_weights.unsqueeze(1)
                    * grad_sigma[:, :, spatial_dimension]
                )
        rhs[rows] = -sqrt_weights * benchmark.f_int[:, equilibrium_component]

    matrix = torch.empty(
        9 * q_count,
        active_dim_s + dim_u,
        dtype=elasticity.DTYPE,
        device=elasticity.DEVICE,
    )
    matrix[:, :active_dim_s] = raw_stress_matrix @ stress_adapter.transform
    matrix[:, active_dim_s:] = displacement_matrix
    del raw_stress_matrix, displacement_matrix
    gc.collect()
    return DirectDesign(matrix, rhs, active_dim_s, stress_adapter)


def solve_scaled_direct(
    design: DirectDesign,
    driver: str,
    rcond: float | None,
) -> tuple[torch.Tensor, float, int, float]:
    column_norms = torch.linalg.vector_norm(design.matrix, dim=0)
    floor = torch.finfo(design.matrix.dtype).eps * column_norms.max()
    safe_norms = column_norms.clamp_min(floor)
    design.matrix.div_(safe_norms.unsqueeze(0))
    t0 = time.perf_counter()
    scaled_solution, _, rank, singular_values = scipy.linalg.lstsq(
        design.matrix.numpy(),
        design.rhs.numpy(),
        cond=rcond,
        overwrite_a=True,
        overwrite_b=False,
        check_finite=False,
        lapack_driver=driver,
    )
    wall_time = time.perf_counter() - t0
    solution = torch.from_numpy(scaled_solution).to(dtype=elasticity.DTYPE) / safe_norms
    positive = (
        singular_values[singular_values > 0.0]
        if singular_values is not None
        else []
    )
    condition_estimate = (
        float(positive.max() / positive.min()) if len(positive) else float("nan")
    )
    return solution, wall_time, int(rank), condition_estimate


def run(args: argparse.Namespace) -> dict[str, float | int | str]:
    cfg = elasticity.LeastSquaresConfig(
        E=args.E,
        nu=args.nu,
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
    solution, solve_time, rank, condition_estimate = solve_scaled_direct(
        design,
        args.driver,
        args.rcond,
    )
    active_sigma = solution[: design.solved_dim_s]
    displacement = solution[design.solved_dim_s :]
    raw_sigma_coeffs = elasticity.lift_active_stress_coefficients(
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
    u_error, sigma_error = elasticity.compute_absolute_errors(
        displacement_basis_test,
        sigma_basis_test,
        raw_sigma_coeffs,
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
        "E": args.E,
        "nu": args.nu,
        "driver": args.driver,
        "rcond": args.rcond,
        "rows": design.matrix.shape[0],
        "columns": design.matrix.shape[1],
        "matrix_mib": design.matrix.numel() * design.matrix.element_size() / (1024**2),
        "rank": rank,
        "feature_family": "constant+random-tanh",
        "condition_estimate": condition_estimate,
        "assemble_time": assemble_time,
        "solve_time": solve_time,
        "u_l2_error": u_error,
        "sigma_l2_error": sigma_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct rank-revealing least squares for 3D linear elasticity."
    )
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--E", type=float, default=4.0 / 3.0)
    parser.add_argument("--nu", type=float, default=1.0 / 3.0)
    parser.add_argument("--q-train", type=int, default=16**3)
    parser.add_argument("--q-test", type=int, default=16**3)
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
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--driver", choices=("gelsy", "gelsd", "gelss"), default="gelsd")
    parser.add_argument("--rcond", type=float, default=1.0e-14)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.N <= 0:
        raise ValueError("N must be positive.")
    if args.sampling == "gauss_legendre":
        for name, value in (("q_train", args.q_train), ("q_test", args.q_test)):
            order = round(value ** (1.0 / 3.0))
            if order**3 != value:
                raise ValueError(f"{name} must be a perfect cube for Gauss-Legendre.")
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
