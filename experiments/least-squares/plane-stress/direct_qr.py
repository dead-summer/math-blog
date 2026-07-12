from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass

import scipy.linalg
import torch

import plane_stress as stress


@dataclass(frozen=True)
class DirectDesign:
    matrix: torch.Tensor
    rhs: torch.Tensor
    dim_s: int


def assemble_direct_design(
    benchmark: stress.SharedBenchmarkData,
    feature_space: stress.SharedFeatureSpace,
) -> DirectDesign:
    x = benchmark.x_int
    sqrt_weights = torch.sqrt(benchmark.w_int)
    sigma_values = stress.eval_features(
        x,
        feature_space.a_s,
        feature_space.r_s,
        feature_space.gamma_s,
    )
    sigma_gradients = stress.eval_feature_grads(
        x,
        feature_space.a_s,
        feature_space.r_s,
        feature_space.gamma_s,
    )
    _, displacement_gradients = stress.eval_active_displacement_feature_data(
        x,
        feature_space.a_u,
        feature_space.r_u,
        feature_space.gamma_u,
    )
    q_count = x.shape[0]
    dim_s = 3 * sigma_values.shape[1]
    dim_u = 2 * displacement_gradients.shape[1]
    matrix = torch.zeros(
        5 * q_count,
        dim_s + dim_u,
        dtype=stress.DTYPE,
        device=stress.DEVICE,
    )
    rhs = torch.zeros(5 * q_count, dtype=stress.DTYPE, device=stress.DEVICE)
    weighted_sigma = sqrt_weights.unsqueeze(1) * sigma_values

    for residual_component in range(3):
        rows = slice(residual_component * q_count, (residual_component + 1) * q_count)
        for sigma_component in range(3):
            matrix[rows, sigma_component:dim_s:3] = (
                benchmark.compliance_voigt[residual_component, sigma_component]
                * weighted_sigma
            )
        for spatial_dimension in range(2):
            coupling = stress.STRAIN_GRAD_BASES[spatial_dimension]
            for displacement_component in range(2):
                matrix[rows, dim_s + displacement_component :: 2] -= (
                    coupling[residual_component, displacement_component]
                    * sqrt_weights.unsqueeze(1)
                    * displacement_gradients[:, :, spatial_dimension]
                )

    for equilibrium_component in range(2):
        rows = slice(
            (3 + equilibrium_component) * q_count,
            (4 + equilibrium_component) * q_count,
        )
        for spatial_dimension in range(2):
            coupling = stress.STRAIN_GRAD_BASES[spatial_dimension]
            for sigma_component in range(3):
                matrix[rows, sigma_component:dim_s:3] += (
                    coupling[sigma_component, equilibrium_component]
                    * sqrt_weights.unsqueeze(1)
                    * sigma_gradients[:, :, spatial_dimension]
                )
        rhs[rows] = -sqrt_weights * benchmark.f_int[:, equilibrium_component]
    return DirectDesign(matrix, rhs, dim_s)


def solve(design: DirectDesign, rcond: float) -> tuple[torch.Tensor, float, int, float]:
    norms = torch.linalg.vector_norm(design.matrix, dim=0)
    safe_norms = norms.clamp_min(torch.finfo(stress.DTYPE).eps * norms.max())
    design.matrix.div_(safe_norms.unsqueeze(0))
    t0 = time.perf_counter()
    scaled_solution, _, rank, singular_values = scipy.linalg.lstsq(
        design.matrix.numpy(),
        design.rhs.numpy(),
        cond=rcond,
        overwrite_a=True,
        overwrite_b=False,
        check_finite=False,
        lapack_driver="gelsd",
    )
    wall_time = time.perf_counter() - t0
    solution = torch.from_numpy(scaled_solution).to(dtype=stress.DTYPE) / safe_norms
    positive = singular_values[singular_values > 0.0]
    condition = float(positive.max() / positive.min()) if len(positive) else float("nan")
    return solution, wall_time, int(rank), condition


def run(args: argparse.Namespace) -> dict[str, float | int | str]:
    cfg = stress.LeastSquaresConfig(
        N_s=args.N,
        N_u=args.N,
        gamma_s=args.gamma,
        gamma_u=args.gamma,
        Q_train=args.q_train,
        Q_test=args.q_test,
        sampling_method=args.sampling,
    )
    benchmark = stress.build_shared_benchmark(
        E=cfg.E,
        nu=cfg.nu,
        Q_train=cfg.Q_train,
        Q_test=cfg.Q_test,
        sampling_method=cfg.sampling_method,
        body_force_batch_size=cfg.body_force_batch_size,
    )
    feature_space = stress.build_shared_feature_space(
        N_s=cfg.N_s,
        N_u=cfg.N_u,
        gamma_s=cfg.gamma_s,
        gamma_u=cfg.gamma_u,
    )
    t0 = time.perf_counter()
    design = assemble_direct_design(benchmark, feature_space)
    assemble_time = time.perf_counter() - t0
    solution, solve_time, rank, condition = solve(design, args.rcond)
    sigma_coeffs = solution[: design.dim_s]
    displacement_coeffs = solution[design.dim_s :]
    sigma_test = stress.eval_features(
        benchmark.x_test,
        feature_space.a_s,
        feature_space.r_s,
        feature_space.gamma_s,
    )
    displacement_test = stress.eval_active_displacement_features(
        benchmark.x_test,
        feature_space.a_u,
        feature_space.r_u,
        feature_space.gamma_u,
    )
    u_error, sigma_error = stress.compute_absolute_errors(
        displacement_test,
        sigma_test,
        sigma_coeffs,
        displacement_coeffs,
        benchmark.w_test,
        benchmark.u_exact_test,
        benchmark.sigma_exact_test,
    )
    return {
        "method": "Direct GELSD",
        "N": args.N,
        "Q_train": args.q_train,
        "Q_test": args.q_test,
        "sampling": args.sampling,
        "rcond": args.rcond,
        "rows": design.matrix.shape[0],
        "columns": design.matrix.shape[1],
        "matrix_mib": design.matrix.numel() * design.matrix.element_size() / (1024**2),
        "rank": rank,
        "feature_family": "constant+random-tanh",
        "condition_estimate": condition,
        "assemble_time": assemble_time,
        "solve_time": solve_time,
        "u_l2_error": u_error,
        "sigma_l2_error": sigma_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct least squares for plane stress.")
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--q-train", type=int, default=64**2)
    parser.add_argument("--q-test", type=int, default=128**2)
    parser.add_argument(
        "--sampling", choices=stress.VALID_SAMPLING_METHODS, default="gauss_legendre"
    )
    parser.add_argument("--gamma", type=float, default=3.0)
    parser.add_argument("--rcond", type=float, default=1.0e-14)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sampling == "gauss_legendre":
        for name, value in (("q_train", args.q_train), ("q_test", args.q_test)):
            order = math.isqrt(value)
            if order * order != value:
                raise ValueError(f"{name} must be a perfect square for Gauss-Legendre.")
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
