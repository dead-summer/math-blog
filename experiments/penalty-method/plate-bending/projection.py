from __future__ import annotations

import time
from dataclasses import dataclass, replace

import torch

from plate_bending import (
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
    compute_bending_stiffness,
    eval_exact_deflection,
    eval_exact_moment,
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
    h: float = 1.0
    gamma_m: float = 2.0
    gamma_u: float = 2.0
    M_m: int = 200
    M_u: int = 200
    Q_int: int = (2 ** 7) ** 2
    Q_bc: int = 4 * (2 ** 7)
    Q_test: int = (2 ** 7) ** 2
    sampling_method: str = "sobol"
    assembly_batch_size: int = 5_000


def validate_config(cfg: ProjectionConfig) -> None:
    """Validate config before starting any expensive work."""

    if cfg.E <= 0.0:
        raise ValueError("ProjectionConfig.E must be positive.")
    if not (-1.0 < cfg.nu < 0.5):
        raise ValueError("ProjectionConfig.nu must lie in (-1, 0.5).")
    if cfg.h <= 0.0:
        raise ValueError("ProjectionConfig.h must be positive.")
    if cfg.gamma_m <= 0.0 or cfg.gamma_u <= 0.0:
        raise ValueError("ProjectionConfig.gamma_m and ProjectionConfig.gamma_u must be positive.")
    if cfg.M_m <= 0 or cfg.M_u <= 0:
        raise ValueError("ProjectionConfig.M_m and ProjectionConfig.M_u must be positive.")
    if cfg.Q_int <= 0:
        raise ValueError("ProjectionConfig.Q_int must be positive.")
    if cfg.Q_bc < 4:
        raise ValueError("ProjectionConfig.Q_bc must be at least 4.")
    if cfg.Q_test <= 0:
        raise ValueError("ProjectionConfig.Q_test must be positive.")
    if cfg.assembly_batch_size <= 0:
        raise ValueError("ProjectionConfig.assembly_batch_size must be positive.")
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
        h=shared_cfg.h,
        gamma_m=shared_cfg.gamma_m,
        gamma_u=shared_cfg.gamma_u,
        M_m=shared_cfg.M_m,
        M_u=shared_cfg.M_u,
        Q_int=shared_cfg.Q_int,
        Q_bc=shared_cfg.Q_bc,
        Q_test=shared_cfg.Q_test,
        sampling_method=shared_cfg.sampling_method,
    )


def run_projection(
    x_int: torch.Tensor,
    a_m: torch.Tensor,
    r_m: torch.Tensor,
    gamma_m: float,
    a_u: torch.Tensor,
    r_u: torch.Tensor,
    gamma_u: float,
    u_exact_train: torch.Tensor,
    M_exact_train: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Fit exact deflection and moments in the two random-feature spaces."""

    m = torch.zeros(3 * (a_m.shape[0] + 1), dtype=DTYPE, device=DEVICE)
    u = torch.zeros(a_u.shape[0] + 1, dtype=DTYPE, device=DEVICE)

    synchronize_device()
    t0 = time.perf_counter()
    with torch.no_grad():
        xi_u_train = eval_features(x_int, a_u, r_u, gamma_u)
        u.copy_(torch.linalg.lstsq(xi_u_train, u_exact_train).solution)
        del xi_u_train
        clear_cuda_cache()

        xi_m_train = eval_features(x_int, a_m, r_m, gamma_m)
        m_blocks = m.reshape(-1, 3)
        for comp in range(3):
            m_blocks[:, comp] = torch.linalg.lstsq(
                xi_m_train,
                M_exact_train[:, comp],
            ).solution
        del xi_m_train
        clear_cuda_cache()

    synchronize_device()
    return m, u, time.perf_counter() - t0


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
        f"Config: h={cfg.h}, M_m={cfg.M_m}, M_u={cfg.M_u}, "
        f"Q_int={cfg.Q_int}, Q_bc={cfg.Q_bc}, Q_test={cfg.Q_test}, "
        f"gamma_m={cfg.gamma_m}, gamma_u={cfg.gamma_u}, "
        f"sampling={cfg.sampling_method}"
    )
    print("Algorithms: ['projection']")

    D = compute_bending_stiffness(cfg.E, cfg.nu, cfg.h)
    print(f"Material: E={cfg.E}, nu={cfg.nu}, h={cfg.h}, D={D:.4f}")

    if benchmark is None:
        print("Sampling shared benchmark data...")
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
        raise ValueError("SharedFeatureSpace gamma does not match ProjectionConfig.")

    print("Computing exact training fields...")
    u_exact_train = eval_exact_deflection(benchmark.x_int)
    M_exact_train = eval_exact_moment(benchmark.x_int, D, cfg.nu)
    eval_data = build_feature_evaluation_data(
        benchmark,
        feature_space,
        cfg.assembly_batch_size,
    )

    print("Running Projection...")
    m, u, wall_time = run_projection(
        benchmark.x_int,
        feature_space.a_m,
        feature_space.r_m,
        feature_space.gamma_m,
        feature_space.a_u,
        feature_space.r_u,
        feature_space.gamma_u,
        u_exact_train,
        M_exact_train,
    )
    result = evaluate_feature_result("Projection", wall_time, m, u, eval_data)
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
