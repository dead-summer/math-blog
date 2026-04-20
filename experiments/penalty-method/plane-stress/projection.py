from __future__ import annotations

import time
from dataclasses import dataclass, replace

import torch

from plane_stress import (
    DEVICE,
    DTYPE,
    AlgorithmResult,
    SharedBenchmarkData,
    SharedComparisonConfig,
    SharedFeatureSpace,
    build_feature_evaluation_data,
    build_shared_benchmark,
    build_shared_feature_space,
    clear_cuda_cache,
    compute_plane_stress_parameters,
    compute_stress_voigt,
    eval_exact_displacement,
    eval_features,
    evaluate_feature_result,
    print_result_summary,
    print_summary_table,
    synchronize_device,
    validate_sampling_method,
)


@dataclass
class ProjectionConfig:
    """Configuration for the random-feature projection baseline."""

    E: float = 1.0
    nu: float = 0.3
    gamma_s: float = 2.0
    gamma_u: float = 2.0
    M_s: int = 200
    M_u: int = 200
    Q_int: int = (2 ** 7) ** 2
    Q_bc: int = 4 * (2 ** 7)
    Q_test: int = (2 ** 7) ** 2
    sampling_method: str = "sobol"
    body_force_batch_size: int = 5_000
    assembly_batch_size: int = 5_000


def validate_config(cfg: ProjectionConfig) -> None:
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
    if cfg.Q_bc < 4:
        raise ValueError("Config.Q_bc must be at least 4.")
    if cfg.Q_test <= 0:
        raise ValueError("Config.Q_test must be positive.")
    if cfg.body_force_batch_size <= 0:
        raise ValueError("Config.body_force_batch_size must be positive.")
    if cfg.assembly_batch_size <= 0:
        raise ValueError("Config.assembly_batch_size must be positive.")
    validate_sampling_method(cfg.sampling_method)


def apply_shared_to_projection_config(
    cfg: ProjectionConfig,
    shared_cfg: SharedComparisonConfig,
) -> ProjectionConfig:
    """Override comparison-critical projection fields from the shared config."""

    return replace(
        cfg,
        E=shared_cfg.E,
        nu=shared_cfg.nu,
        gamma_s=shared_cfg.gamma_s,
        gamma_u=shared_cfg.gamma_u,
        M_s=shared_cfg.M_s,
        M_u=shared_cfg.M_u,
        Q_int=shared_cfg.Q_int,
        Q_bc=shared_cfg.Q_bc,
        Q_test=shared_cfg.Q_test,
        sampling_method=shared_cfg.sampling_method,
        body_force_batch_size=shared_cfg.body_force_batch_size,
    )


def run_projection(
    x_int: torch.Tensor,
    a_s: torch.Tensor,
    r_s: torch.Tensor,
    gamma_s: float,
    a_u: torch.Tensor,
    r_u: torch.Tensor,
    gamma_u: float,
    u_exact_train: torch.Tensor,
    sigma_exact_train: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Fit exact displacement and stress in the two random-feature spaces."""

    mp1_s = a_s.shape[0] + 1
    mp1_u = a_u.shape[0] + 1
    s = torch.zeros(3 * mp1_s, dtype=DTYPE, device=DEVICE)
    u = torch.zeros(2 * mp1_u, dtype=DTYPE, device=DEVICE)

    synchronize_device()
    t0 = time.perf_counter()
    with torch.no_grad():
        xi_u_train = eval_features(x_int, a_u, r_u, gamma_u)
        for comp in range(2):
            u[comp::2] = torch.linalg.lstsq(
                xi_u_train,
                u_exact_train[:, comp],
            ).solution
        del xi_u_train
        clear_cuda_cache()

        xi_s_train = eval_features(x_int, a_s, r_s, gamma_s)
        for comp in range(3):
            s[comp::3] = torch.linalg.lstsq(
                xi_s_train,
                sigma_exact_train[:, comp],
            ).solution
        del xi_s_train
        clear_cuda_cache()

    synchronize_device()
    return s, u, time.perf_counter() - t0


def run_experiment(
    cfg: ProjectionConfig | None = None,
    print_table: bool = True,
    benchmark: SharedBenchmarkData | None = None,
    feature_space: SharedFeatureSpace | None = None,
) -> list[AlgorithmResult]:
    """Run the projection baseline and return its metrics."""

    cfg = ProjectionConfig() if cfg is None else cfg
    validate_config(cfg)

    print(f"Device: {DEVICE}")
    print(
        f"Config: M_s={cfg.M_s}, M_u={cfg.M_u}, "
        f"Q_int={cfg.Q_int}, Q_bc={cfg.Q_bc}, Q_test={cfg.Q_test}, "
        f"gamma_s={cfg.gamma_s}, gamma_u={cfg.gamma_u}, "
        f"sampling={cfg.sampling_method}"
    )
    print("Algorithms: ['projection']")

    mu, lambda_plane = compute_plane_stress_parameters(cfg.E, cfg.nu)
    print(
        f"Material: E={cfg.E}, nu={cfg.nu}, "
        f"mu={mu:.4f}, lambda_plane={lambda_plane:.4f}"
    )

    if benchmark is None:
        print("Sampling shared benchmark data...")
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
        raise ValueError("SharedFeatureSpace gamma does not match ProjectionConfig.")

    print("Computing exact training fields...")
    u_exact_train = eval_exact_displacement(benchmark.x_int)
    sigma_exact_train = compute_stress_voigt(benchmark.x_int, mu, lambda_plane)
    eval_data = build_feature_evaluation_data(
        benchmark,
        feature_space,
        cfg.assembly_batch_size,
    )

    print("Running Projection...")
    s, u, wall_time = run_projection(
        benchmark.x_int,
        feature_space.a_s,
        feature_space.r_s,
        feature_space.gamma_s,
        feature_space.a_u,
        feature_space.r_u,
        feature_space.gamma_u,
        u_exact_train,
        sigma_exact_train,
    )
    result = evaluate_feature_result("Projection", wall_time, s, u, eval_data)
    print_result_summary(result)

    results = [result]
    if print_table:
        print_summary_table(results, title="Projection Summary")
    return results


def main(cfg: ProjectionConfig | None = None) -> None:
    """Script entrypoint."""

    run_experiment(cfg, print_table=True)


if __name__ == "__main__":
    main()
