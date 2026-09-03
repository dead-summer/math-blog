"""Kirchhoff--Love plate-bending driver.

The moment--deflection physics is specific to this experiment; quadrature,
the quasi-uniform feature construction, streaming TSQR, and the
coefficient-ball solve are shared through ``ls_common``.  Manufactured
solutions are pluggable: a solution is a deflection callback, and the exact
Hessian, moment, and transverse load are derived from it with autodiff unless
closed forms are supplied.
"""

from __future__ import annotations

import math
import sys
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ls_common import (  # noqa: E402
    BASE_SEED,
    DEVICE,
    DTYPE,
    VALID_ALGORITHMS,
    VALID_DIRECT_SOLVERS,
    build_quadrature_rule,
    clear_cuda_cache,
    generate_features,
    iter_point_batches,
    load_config_defaults,
    plot_error_summary,
    print_aligned_markdown_table,
    streaming_tsqr_compress,
    validate_algorithm_selection,
    validate_sampling_method,
    warn_if_dense_infeasible,
)
from rfm_core import (  # noqa: E402
    RitzProjectedFeatures,
    build_ritz_projected_features,
    matched_spline_degree,
    relu_power_feature_values_and_hessians,
    saturation_index,
    validate_activation_power,
)
from solvers import get_solver_spec, run_solver  # noqa: E402
from system_backends import (  # noqa: E402
    GramResidualDesign,
    assemble_gram_residual_design,
    get_system_backend,
)


PROJECTION_SEED = BASE_SEED + 17_000
FEATURE_DIM = 2
HESSIAN_COMPONENTS = ((0, 0), (1, 1), (0, 1))
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "public" / "images" / "least-squares" / "plate-bending"

FROBENIUS_WEIGHT = torch.tensor([1.0, 1.0, 2.0], dtype=DTYPE, device=DEVICE)
DIVDIV_WEIGHTS = torch.tensor([1.0, 1.0, 2.0], dtype=DTYPE, device=DEVICE)


DeflectionFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class PlateSolution:
    """One manufactured clamped-plate solution.

    Only ``deflection`` is required; the Hessian and the transverse load
    ``f = D Delta^2 w`` fall back to autodiff when closed forms are omitted.
    """

    deflection: DeflectionFn
    hessian: Callable[[torch.Tensor], torch.Tensor] | None = None
    gradient: Callable[[torch.Tensor], torch.Tensor] | None = None
    body_force: Callable[[torch.Tensor, float], torch.Tensor] | None = None


def _autodiff_gradient(deflection: DeflectionFn, x: torch.Tensor) -> torch.Tensor:
    x_ad = x.detach().requires_grad_(True)
    return torch.autograd.grad(deflection(x_ad).sum(), x_ad)[0].detach()


def _autodiff_hessian(deflection: DeflectionFn, x: torch.Tensor) -> torch.Tensor:
    """Hessian components (11, 22, 12) of the deflection."""

    x_ad = x.detach().requires_grad_(True)
    grad = torch.autograd.grad(
        deflection(x_ad).sum(),
        x_ad,
        create_graph=True,
    )[0]
    h1 = torch.autograd.grad(grad[:, 0].sum(), x_ad, create_graph=False, retain_graph=True)[0]
    h2 = torch.autograd.grad(grad[:, 1].sum(), x_ad, create_graph=False)[0]
    return torch.stack([h1[:, 0], h2[:, 1], h1[:, 1]], dim=1).detach()


def _autodiff_body_force(
    deflection: DeflectionFn,
    x: torch.Tensor,
    D: float,
) -> torch.Tensor:
    """Evaluate ``f = D Delta^2 w`` with nested autodiff."""

    x_ad = x.detach().requires_grad_(True)
    grad = torch.autograd.grad(deflection(x_ad).sum(), x_ad, create_graph=True)[0]
    laplacian = torch.zeros(x.shape[0], dtype=DTYPE, device=DEVICE)
    for axis in range(2):
        second = torch.autograd.grad(
            grad[:, axis].sum(),
            x_ad,
            create_graph=True,
        )[0][:, axis]
        laplacian = laplacian + second
    grad_lap = torch.autograd.grad(laplacian.sum(), x_ad, create_graph=True)[0]
    bilaplacian = torch.zeros(x.shape[0], dtype=DTYPE, device=DEVICE)
    for axis in range(2):
        second = torch.autograd.grad(
            grad_lap[:, axis].sum(),
            x_ad,
            create_graph=axis < 1,
        )[0][:, axis]
        bilaplacian = bilaplacian + second
    return (D * bilaplacian).detach()


def solution_gradient(solution: PlateSolution, x: torch.Tensor) -> torch.Tensor:
    if solution.gradient is not None:
        return solution.gradient(x)
    return _autodiff_gradient(solution.deflection, x)


def solution_hessian(solution: PlateSolution, x: torch.Tensor) -> torch.Tensor:
    if solution.hessian is not None:
        return solution.hessian(x)
    return _autodiff_hessian(solution.deflection, x)


def solution_body_force(solution: PlateSolution, x: torch.Tensor, D: float) -> torch.Tensor:
    if solution.body_force is not None:
        return solution.body_force(x, D)
    return _autodiff_body_force(solution.deflection, x, D)


def solution_moment(
    solution: PlateSolution,
    x: torch.Tensor,
    D: float,
    nu: float,
) -> torch.Tensor:
    """Exact bending moments in Voigt order (11, 22, 12)."""

    hess = solution_hessian(solution, x)
    u_xx, u_yy, u_xy = hess[:, 0], hess[:, 1], hess[:, 2]
    M11 = -D * (u_xx + nu * u_yy)
    M22 = -D * (nu * u_xx + u_yy)
    M12 = -D * (1.0 - nu) * u_xy
    return torch.stack([M11, M22, M12], dim=1)


# --- default manufactured solution: w = p(x1) p(x2), p(t) = t^2 (1-t)^2 ----


def poly_p(t: torch.Tensor) -> torch.Tensor:
    return t.square() * (1.0 - t).square()


def poly_dp(t: torch.Tensor) -> torch.Tensor:
    return 2.0 * t - 6.0 * t.square() + 4.0 * t.pow(3)


def poly_d2p(t: torch.Tensor) -> torch.Tensor:
    return 2.0 - 12.0 * t + 12.0 * t.square()


def poly_d4p(t: torch.Tensor) -> torch.Tensor:
    return torch.full_like(t, 24.0)


DEFAULT_SOLUTION = PlateSolution(
    deflection=lambda x: poly_p(x[:, 0]) * poly_p(x[:, 1]),
    hessian=lambda x: torch.stack(
        [
            poly_d2p(x[:, 0]) * poly_p(x[:, 1]),
            poly_p(x[:, 0]) * poly_d2p(x[:, 1]),
            poly_dp(x[:, 0]) * poly_dp(x[:, 1]),
        ],
        dim=1,
    ),
    gradient=lambda x: torch.stack(
        [
            poly_dp(x[:, 0]) * poly_p(x[:, 1]),
            poly_p(x[:, 0]) * poly_dp(x[:, 1]),
        ],
        dim=1,
    ),
    body_force=lambda x, D: D
    * (
        poly_d4p(x[:, 0]) * poly_p(x[:, 1])
        + 2.0 * poly_d2p(x[:, 0]) * poly_d2p(x[:, 1])
        + poly_p(x[:, 0]) * poly_d4p(x[:, 1])
    ),
)

# --- non-polynomial clamped solution: w = sin^2(pi x1) sin^2(pi x2) -------
#
# The default deflection is a polynomial of total degree eight, so a
# rho_k dictionary whose polynomial supplement spans P_k reproduces it (and its
# moment, of degree six) exactly once k is large enough.  That is a genuine
# property of the space, but it makes the deflection floor measure polynomial
# reproduction rather than ridge approximation: at k = 7 the moment error drops
# to 3e-15 and the plate's primary metric stops measuring convergence at all.
# This solution is analytic and not a polynomial, so it keeps every study
# meaningful at every k, and it is what ``defaults.json`` selects.  Both traces
# vanish: sin^2(pi t) and its derivative pi sin(2 pi t) are zero at t = 0 and
# t = 1.


TRIG_SOLUTION = PlateSolution(
    deflection=lambda x: torch.sin(math.pi * x[:, 0]).square()
    * torch.sin(math.pi * x[:, 1]).square(),
)

SOLUTIONS: dict[str, PlateSolution] = {
    "default": DEFAULT_SOLUTION,
    "trig": TRIG_SOLUTION,
}


# ---------------------------------------------------------------------------
# Configuration and containers
# ---------------------------------------------------------------------------


@dataclass
class LeastSquaresConfig:
    """Configuration for the projected plate least-squares experiment."""

    E: float = 1.0
    nu: float = 0.3
    h: float = 1.0
    N_m: int = 1000
    N_u: int = 1000
    Q_train: int = 4 * 1001
    Q_test: int = 128**2
    sampling_method: str = "mc"
    activation_power: int = 7
    ritz_degree: int = 7
    ritz_ratio: float = 2.0
    projection_samples: int | None = None
    coefficient_budget: float = math.inf
    ridge_lambda: float = 1.0e-12
    manufactured_solution: str = "default"
    direct_rcond: float = 1.0e-12
    system_backend: str = "direct"
    direct_solver: str = "streaming_tsqr"
    direct_batch_size: int = 1_024
    direct_qr_block_size: int = 64
    evaluation_batch_size: int = 4_096
    algorithms_to_run: list[str] = field(
        default_factory=lambda: ["ball", "ridge", "tsvd"]
    )


def default_config() -> LeastSquaresConfig:
    """Load the folder's ``defaults.json`` on top of the dataclass defaults."""

    return load_config_defaults(LeastSquaresConfig, Path(__file__).resolve().parent)


def validate_config(cfg: LeastSquaresConfig) -> None:
    """Validate config before starting any expensive work."""

    if cfg.E <= 0.0:
        raise ValueError("Config.E must be positive.")
    if not (-1.0 < cfg.nu < 0.5):
        raise ValueError("Config.nu must lie in (-1, 0.5).")
    if cfg.h <= 0.0:
        raise ValueError("Config.h must be positive.")
    if cfg.N_m <= 0 or cfg.N_u <= 0:
        raise ValueError("Config.N_m and Config.N_u must be positive.")
    # The moment-deflection graph norm carries two derivatives, so the
    # training-generalization argument needs rho_k in W^{3,inf}.
    validate_activation_power(cfg.activation_power, sobolev_order=2)
    if cfg.ritz_degree < 3:
        raise ValueError("Config.ritz_degree must be at least 3.")
    if cfg.ritz_ratio < 1.0:
        raise ValueError("Config.ritz_ratio must be at least one.")
    if cfg.projection_samples is not None and cfg.projection_samples <= 0:
        raise ValueError("Config.projection_samples must be positive when set.")
    if cfg.coefficient_budget <= 0.0:
        raise ValueError("Config.coefficient_budget must be positive.")
    if not math.isfinite(cfg.ridge_lambda) or cfg.ridge_lambda <= 0.0:
        raise ValueError("Config.ridge_lambda must be finite and positive.")
    if cfg.Q_train <= 0 or cfg.Q_test <= 0:
        raise ValueError("Config.Q_train and Config.Q_test must be positive.")
    if not math.isfinite(cfg.direct_rcond) or cfg.direct_rcond <= 0.0:
        raise ValueError("Config.direct_rcond must be finite and positive.")
    get_system_backend(cfg.system_backend)
    if cfg.direct_solver not in VALID_DIRECT_SOLVERS:
        raise ValueError(
            f"Unknown direct_solver='{cfg.direct_solver}'. "
            f"Valid values: {list(VALID_DIRECT_SOLVERS)}"
        )
    if cfg.direct_batch_size <= 0 or cfg.direct_qr_block_size <= 0:
        raise ValueError("Config batch sizes must be positive.")
    if cfg.evaluation_batch_size <= 0:
        raise ValueError("Config.evaluation_batch_size must be positive.")
    if cfg.manufactured_solution not in SOLUTIONS:
        raise ValueError(
            f"Unknown manufactured_solution='{cfg.manufactured_solution}'. "
            f"Valid values: {sorted(SOLUTIONS)}"
        )
    validate_sampling_method(cfg.sampling_method)
    validate_algorithm_selection(cfg.algorithms_to_run, VALID_ALGORITHMS)


def compute_bending_stiffness(E: float, nu: float, h: float) -> float:
    """Return the Kirchhoff plate bending stiffness."""

    return E * h**3 / (12.0 * (1.0 - nu * nu))


def build_compliance_matrix(D: float, nu: float) -> torch.Tensor:
    """Build the 3x3 plate compliance matrix in Voigt-like form."""

    compliance_voigt = torch.zeros(3, 3, dtype=DTYPE, device=DEVICE)
    inv = 1.0 / (D * (1.0 - nu * nu))
    compliance_voigt[0, 0] = inv
    compliance_voigt[1, 1] = inv
    compliance_voigt[0, 1] = -nu * inv
    compliance_voigt[1, 0] = -nu * inv
    compliance_voigt[2, 2] = 1.0 / (D * (1.0 - nu))
    return compliance_voigt


@dataclass(frozen=True)
class AlgorithmResult:
    """Compact metrics for one completed algorithm."""

    name: str
    r_c: float
    r_e: float
    abs_u: float
    abs_M: float
    wall_time: float
    rank: int = 0
    columns: int = 0
    condition_estimate: float = float("nan")
    w_h2_error: float = float("nan")
    M_hdivdiv_error: float = float("nan")
    coefficient_ball_active: bool = False
    coefficient_norm: float = float("nan")
    algorithm: str = ""
    hyperparameter: float = float("nan")


@dataclass(frozen=True)
class SharedBenchmarkData:
    """Shared train/test samples reused across runs."""

    x_int: torch.Tensor
    w_int: torch.Tensor
    f_int: torch.Tensor
    x_test: torch.Tensor
    w_test: torch.Tensor
    u_exact_test: torch.Tensor
    M_exact_test: torch.Tensor
    compliance_voigt: torch.Tensor
    f_test: torch.Tensor
    u_grad_exact_test: torch.Tensor
    u_hess_exact_test: torch.Tensor


@dataclass(frozen=True)
class SharedFeatureSpace:
    """Shared deterministic feature spaces used by coefficient-based methods."""

    theta_m: torch.Tensor
    theta_w: torch.Tensor
    projected_w: RitzProjectedFeatures
    activation_power: int = 3


@dataclass(frozen=True)
class FeatureEvaluationData:
    """All tensors needed to evaluate coefficient-based methods.

    Only per-point exact data and the feature *descriptions* are held; the
    dictionary and Ritz bases are re-evaluated in blocks by
    :func:`evaluate_feature_result`, so nothing of size ``Q_test * N`` is
    stored.
    """

    x_test: torch.Tensor
    w_test: torch.Tensor
    f_test: torch.Tensor
    theta_m: torch.Tensor
    projected_w: RitzProjectedFeatures
    compliance_voigt: torch.Tensor
    activation_power: int
    evaluation_batch_size: int
    u_exact_test: torch.Tensor
    M_exact_test: torch.Tensor
    u_grad_exact_test: torch.Tensor
    u_hess_exact_test: torch.Tensor


@dataclass(frozen=True)
class DirectResidualDesign:
    """Weighted residual matrix and coefficient split metadata."""

    matrix: torch.Tensor
    rhs: torch.Tensor
    dim_m: int


@dataclass(frozen=True)
class LeastSquaresExperimentData:
    """All tensors needed to run and evaluate one least-squares solver."""

    residual_design: torch.Tensor | GramResidualDesign
    rhs: torch.Tensor | None
    column_count: int
    dim_m: int
    eval_data: FeatureEvaluationData
    source_rows: int
    preparation_time: float
    system_backend: str
    direct_solver: str


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def eval_features_and_hessians(
    x: torch.Tensor, theta: torch.Tensor, power: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate ``rho_power`` values and Hessian components (11, 22, 12)."""

    return relu_power_feature_values_and_hessians(
        x, theta, power, hessian_components=HESSIAN_COMPONENTS
    )


def build_shared_benchmark(
    E: float,
    nu: float,
    h: float,
    Q_train: int,
    Q_test: int,
    sampling_method: str,
    manufactured_solution: str = "default",
    interior_seed: int = BASE_SEED + 1,
    test_seed: int = BASE_SEED + 3,
) -> SharedBenchmarkData:
    """Build the shared train/test samples for one run."""

    D = compute_bending_stiffness(E, nu, h)
    solution = SOLUTIONS[manufactured_solution]
    compliance_voigt = build_compliance_matrix(D, nu)

    x_int, w_int = build_quadrature_rule(
        Q_train,
        method=sampling_method,
        dim=FEATURE_DIM,
        seed=interior_seed,
    )
    x_test, w_test = build_quadrature_rule(
        Q_test,
        method="gauss_legendre",
        dim=FEATURE_DIM,
        seed=test_seed,
    )
    return SharedBenchmarkData(
        x_int=x_int,
        w_int=w_int,
        f_int=solution_body_force(solution, x_int, D),
        x_test=x_test,
        w_test=w_test,
        u_exact_test=solution.deflection(x_test).detach(),
        M_exact_test=solution_moment(solution, x_test, D, nu),
        compliance_voigt=compliance_voigt,
        f_test=solution_body_force(solution, x_test, D),
        u_grad_exact_test=solution_gradient(solution, x_test),
        u_hess_exact_test=solution_hessian(solution, x_test),
    )


def build_shared_feature_space(
    N_m: int,
    N_u: int,
    activation_power: int = 3,
    ritz_degree: int = 3,
    ritz_ratio: float = 2.0,
    projection_samples: int | None = None,
    projection_seed: int = PROJECTION_SEED,
) -> SharedFeatureSpace:
    """Build the quasi-uniform moments and the H2 Ritz-projected deflection basis.

    The hidden parameters are deterministic; only the Monte Carlo rule that
    discretizes the projection inner product depends on ``projection_seed``.

    The clamped auxiliary space removes two boundary layers per face, so it has
    zero trace and zero normal derivative at any ``ritz_degree``.  Its
    saturation index ``ritz_degree + 1`` must stay at or above the dictionary's
    ``s_cap(2) = (2*activation_power + 3)/2``.
    """

    validate_activation_power(activation_power, sobolev_order=2)
    theta_m = generate_features(N_m, FEATURE_DIM, activation_power)
    theta_w = generate_features(N_u, FEATURE_DIM, activation_power)
    return SharedFeatureSpace(
        theta_m=theta_m,
        theta_w=theta_w,
        activation_power=activation_power,
        projected_w=build_ritz_projected_features(
            theta_w,
            sobolev_order=2,
            auxiliary_dimension=max(
                N_u + 1,
                int(math.ceil(ritz_ratio * (N_u + 1))),
            ),
            power=activation_power,
            degree=ritz_degree,
            quadrature_samples=projection_samples,
            quadrature_seed=projection_seed,
        ),
    )


def build_feature_evaluation_data(
    benchmark: SharedBenchmarkData,
    feature_space: SharedFeatureSpace,
    evaluation_batch_size: int,
) -> FeatureEvaluationData:
    """Build the shared evaluation tensors for coefficient-based methods."""

    return FeatureEvaluationData(
        x_test=benchmark.x_test,
        w_test=benchmark.w_test,
        f_test=benchmark.f_test,
        theta_m=feature_space.theta_m,
        projected_w=feature_space.projected_w,
        compliance_voigt=benchmark.compliance_voigt,
        activation_power=feature_space.activation_power,
        evaluation_batch_size=evaluation_batch_size,
        u_exact_test=benchmark.u_exact_test,
        M_exact_test=benchmark.M_exact_test,
        u_grad_exact_test=benchmark.u_grad_exact_test,
        u_hess_exact_test=benchmark.u_hess_exact_test,
    )


# ---------------------------------------------------------------------------
# Residual assembly
# ---------------------------------------------------------------------------


def _fill_residual_rows(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    batch_points: int,
    dim_m: int,
    compliance_voigt: torch.Tensor,
    xi_m: torch.Tensor,
    hess_m: torch.Tensor,
    hess_u: torch.Tensor,
    sqrt_weights: torch.Tensor,
    body_force: torch.Tensor,
) -> None:
    """Write the weighted plate residual rows for one batch of points.

    Feature tensors must already carry the sqrt-weight scaling.  The
    constitutive residual is ``A M_h - kappa(w_h) = A M_h + hess(w_h)``.
    """

    sqrt_frobenius = torch.sqrt(FROBENIUS_WEIGHT)
    for residual_component in range(3):
        rows = slice(
            residual_component * batch_points,
            (residual_component + 1) * batch_points,
        )
        residual_scale = sqrt_frobenius[residual_component]
        for moment_component in range(3):
            matrix[rows, moment_component:dim_m:3] = (
                residual_scale
                * compliance_voigt[residual_component, moment_component]
                * xi_m
            )
        matrix[rows, dim_m:] = residual_scale * hess_u[:, :, residual_component]

    equilibrium_rows = slice(3 * batch_points, 4 * batch_points)
    for moment_component in range(3):
        matrix[equilibrium_rows, moment_component:dim_m:3] = (
            DIVDIV_WEIGHTS[moment_component] * hess_m[:, :, moment_component]
        )
    rhs[equilibrium_rows] = -sqrt_weights * body_force


def _weighted_feature_data(
    benchmark: SharedBenchmarkData,
    feature_space: SharedFeatureSpace,
    start: int,
    stop: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = benchmark.x_int[start:stop]
    weights = benchmark.w_int[start:stop]
    body_force = benchmark.f_int[start:stop]
    xi_m, hess_m = eval_features_and_hessians(
        x, feature_space.theta_m, feature_space.activation_power
    )
    hess_u = feature_space.projected_w.evaluate(
        x,
        hessian_components=HESSIAN_COMPONENTS,
    )[2]
    sqrt_weights = torch.sqrt(weights)
    xi_m.mul_(sqrt_weights.unsqueeze(1))
    hess_m.mul_(sqrt_weights.view(-1, 1, 1))
    hess_u.mul_(sqrt_weights.view(-1, 1, 1))
    return xi_m, hess_m, hess_u, sqrt_weights, body_force


def assemble_direct_residual_batch(
    start: int,
    stop: int,
    benchmark: SharedBenchmarkData,
    feature_space: SharedFeatureSpace,
    dim_m: int,
    columns: int,
) -> np.ndarray:
    """Assemble one Fortran-contiguous augmented plate residual block."""

    point_count = benchmark.x_int.shape[0]
    if not (0 <= start < stop <= point_count):
        raise ValueError(
            f"Invalid residual batch [{start}, {stop}) for {point_count} points."
        )
    batch_points = stop - start
    augmented = np.zeros(
        (4 * batch_points, columns + 1),
        dtype=np.float64,
        order="F",
    )
    augmented_torch = torch.from_numpy(augmented)
    with torch.no_grad():
        xi_m, hess_m, hess_u, sqrt_weights, body_force = _weighted_feature_data(
            benchmark,
            feature_space,
            start,
            stop,
        )
        _fill_residual_rows(
            augmented_torch[:, :columns],
            augmented_torch[:, columns],
            batch_points,
            dim_m,
            benchmark.compliance_voigt,
            xi_m,
            hess_m,
            hess_u,
            sqrt_weights,
            body_force,
        )
    return augmented


def assemble_direct_residual_design(
    benchmark: SharedBenchmarkData,
    feature_space: SharedFeatureSpace,
) -> DirectResidualDesign:
    """Assemble the weighted constitutive and equilibrium residuals densely."""

    dim_m = 3 * (feature_space.theta_m.shape[0] + 1)
    dim_u = feature_space.theta_w.shape[0] + 1
    q_count = benchmark.x_int.shape[0]
    matrix = torch.zeros(dim_m + dim_u, 4 * q_count, dtype=DTYPE, device=DEVICE).T
    rhs = torch.zeros(4 * q_count, dtype=DTYPE, device=DEVICE)
    with torch.no_grad():
        xi_m, hess_m, hess_u, sqrt_weights, body_force = _weighted_feature_data(
            benchmark,
            feature_space,
            0,
            q_count,
        )
        _fill_residual_rows(
            matrix,
            rhs,
            q_count,
            dim_m,
            benchmark.compliance_voigt,
            xi_m,
            hess_m,
            hess_u,
            sqrt_weights,
            body_force,
        )
    return DirectResidualDesign(matrix, rhs, dim_m)


def assemble_streaming_tsqr_design(
    cfg: LeastSquaresConfig,
    benchmark: SharedBenchmarkData,
    feature_space: SharedFeatureSpace,
    show_progress: bool = True,
):
    """Compress ``[A, b]`` with streaming Householder TSQR."""

    dim_m = 3 * (feature_space.theta_m.shape[0] + 1)
    dim_u = feature_space.theta_w.shape[0] + 1
    columns = dim_m + dim_u
    matrix, rhs, stats = streaming_tsqr_compress(
        point_count=benchmark.x_int.shape[0],
        rows_per_point=4,
        columns=columns,
        batch_size=cfg.direct_batch_size,
        qr_block_size=cfg.direct_qr_block_size,
        assemble_augmented_batch=lambda start, stop: assemble_direct_residual_batch(
            start,
            stop,
            benchmark,
            feature_space,
            dim_m,
            columns,
        ),
        show_progress=show_progress,
    )
    return DirectResidualDesign(matrix, rhs, dim_m), stats


def assemble_streaming_gram_design(
    cfg: LeastSquaresConfig,
    benchmark: SharedBenchmarkData,
    feature_space: SharedFeatureSpace,
) -> tuple[GramResidualDesign, int]:
    """Stream the plate residual batches into ``A.T A`` and ``A.T b``.

    This backend deliberately reuses :func:`assemble_direct_residual_batch`.
    It therefore changes only how the linear system is stored, not the Monte
    Carlo functional, Frobenius metric, or physical coefficient norm.
    """

    dim_m = 3 * (feature_space.theta_m.shape[0] + 1)
    dim_u = feature_space.theta_w.shape[0] + 1
    columns = dim_m + dim_u
    design = assemble_gram_residual_design(
        point_count=benchmark.x_int.shape[0],
        batch_size=cfg.direct_batch_size,
        column_count=columns,
        assemble_augmented_batch=lambda start, stop: assemble_direct_residual_batch(
            start,
            stop,
            benchmark,
            feature_space,
            dim_m,
            columns,
        ),
        dtype=DTYPE,
        device=DEVICE,
    )
    return design, dim_m


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_feature_result(
    name: str,
    wall_time: float,
    moment_coeffs: torch.Tensor,
    deflection_coeffs: torch.Tensor,
    data: FeatureEvaluationData,
) -> AlgorithmResult:
    """Evaluate one coefficient-based method and package the metrics.

    Every metric is a weighted sum over the deterministic test rule, so the
    test points are consumed in blocks of ``data.evaluation_batch_size``: the
    moment dictionary, the Ritz deflection basis and their second derivatives
    are never materialized at the full ``Q_test x N`` size.  One pass now
    serves both the field errors and the continuous residual norms, which
    previously evaluated the same bases twice.
    """

    finite = bool(
        torch.isfinite(moment_coeffs).all() and torch.isfinite(deflection_coeffs).all()
    )
    moment_blocks = moment_coeffs.reshape(-1, 3)
    squared: dict[str, float] = dict.fromkeys(
        ("w_l2", "w_gradient", "w_hessian", "moment_l2", "constitutive", "equilibrium"),
        float("nan") if not finite else 0.0,
    )

    if finite:
        with torch.no_grad():
            for start, stop in iter_point_batches(
                data.x_test.shape[0],
                data.evaluation_batch_size,
            ):
                x = data.x_test[start:stop]
                w = data.w_test[start:stop]
                body_force = data.f_test[start:stop]

                xi_m, hess_m = eval_features_and_hessians(
                    x, data.theta_m, data.activation_power
                )
                psi, psi_gradients, psi_hessians = data.projected_w.evaluate(
                    x,
                    hessian_components=HESSIAN_COMPONENTS,
                )

                M_h = xi_m @ moment_blocks
                w_h = psi @ deflection_coeffs
                grad_w_h = torch.einsum("qfs,f->qs", psi_gradients, deflection_coeffs)
                hess_w_h = torch.einsum("qfj,f->qj", psi_hessians, deflection_coeffs)

                r_c = M_h @ data.compliance_voigt.T + hess_w_h
                r_e = (
                    hess_m[:, :, 0] @ moment_blocks[:, 0]
                    + hess_m[:, :, 1] @ moment_blocks[:, 1]
                    + 2.0 * (hess_m[:, :, 2] @ moment_blocks[:, 2])
                    + body_force
                )

                squared["w_l2"] += (
                    w * (w_h - data.u_exact_test[start:stop]).square()
                ).sum().item()
                squared["w_gradient"] += (
                    w
                    * (grad_w_h - data.u_grad_exact_test[start:stop]).square().sum(dim=1)
                ).sum().item()
                squared["w_hessian"] += (
                    w
                    * (
                        FROBENIUS_WEIGHT
                        * (hess_w_h - data.u_hess_exact_test[start:stop]).square()
                    ).sum(dim=1)
                ).sum().item()
                squared["moment_l2"] += (
                    w
                    * (
                        FROBENIUS_WEIGHT
                        * (M_h - data.M_exact_test[start:stop]).square()
                    ).sum(dim=1)
                ).sum().item()
                squared["constitutive"] += (
                    w * (FROBENIUS_WEIGHT * r_c.square()).sum(dim=1)
                ).sum().item()
                squared["equilibrium"] += (w * r_e.square()).sum().item()

    r_c_norm = math.sqrt(squared["constitutive"])
    r_e_norm = math.sqrt(squared["equilibrium"])
    abs_M = math.sqrt(squared["moment_l2"])
    return AlgorithmResult(
        name=name,
        r_c=r_c_norm,
        r_e=r_e_norm,
        abs_u=math.sqrt(squared["w_l2"]),
        abs_M=abs_M,
        wall_time=wall_time,
        w_h2_error=math.sqrt(
            squared["w_l2"] + squared["w_gradient"] + squared["w_hessian"]
        ),
        M_hdivdiv_error=math.hypot(abs_M, r_e_norm),
    )


def print_result_summary(result: AlgorithmResult) -> None:
    """Print one compact result line."""

    print(
        f"    [{result.name}] Done in {result.wall_time:.2f}s, "
        f"hyper={result.hyperparameter:g}, "
        f"‖w_N-w_*‖_H2={result.w_h2_error:.2e}, "
        f"‖M_N-M_*‖_Hdivdiv={result.M_hdivdiv_error:.2e}, "
        f"constitutive={result.r_c:.2e}, "
        f"equilibrium={result.r_e:.2e}, "
        f"rank={result.rank}/{result.columns}, "
        f"cond≈{result.condition_estimate:.2e}"
    )
    # ||c|| growing while the graph error stalls is the signature of a
    # saturated dictionary: the retained small singular directions carry
    # coefficient norm but no approximation power.
    print(
        f"    coefficients: ||c||_2={result.coefficient_norm:.3e}, "
        f"B=||c||_2*sqrt(m)={result.coefficient_norm * math.sqrt(max(result.columns, 1)):.3e}, "
        f"ball active={result.coefficient_ball_active}"
    )


def print_summary_table(results: Sequence[AlgorithmResult], title: str) -> None:
    """Print a compact markdown-style summary table."""

    if not results:
        return
    headers = (
        "Method",
        "‖w_N-w_*‖_L2",
        "‖w_N-w_*‖_H2",
        "‖M_N-M_*‖_L2",
        "‖M_N-M_*‖_Hdivdiv",
        "Time(s)",
    )
    rows = [
        (
            result.name,
            f"{result.abs_u:.2e}",
            f"{result.w_h2_error:.2e}",
            f"{result.abs_M:.2e}",
            f"{result.M_hdivdiv_error:.2e}",
            f"{result.wall_time:.2f}",
        )
        for result in results
    ]
    print_aligned_markdown_table(
        title=title,
        headers=headers,
        rows=rows,
        alignments=("left", "center", "center", "center", "center", "center"),
    )


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------


def run_algorithm(
    algorithm_id: str,
    data: LeastSquaresExperimentData,
    cfg: LeastSquaresConfig,
) -> AlgorithmResult:
    """Run one configured least-squares algorithm and evaluate it."""

    spec = get_solver_spec(algorithm_id)
    storage_label = (
        f"direct/{data.direct_solver} + SVD"
        if data.system_backend == "direct"
        else "streaming Gram + symmetric eigensolve"
    )
    print(
        f"Running {spec.label} ({storage_label}) on source system "
        f"({data.source_rows}, {data.column_count})..."
    )
    solve_started = time.perf_counter()
    output = run_solver(algorithm_id, data.residual_design, data.rhs, cfg)
    print(
        f"    {spec.hyperparameter_name}={output.hyperparameter:g}, "
        f"regularization active: {output.regularization_active}"
    )
    reduced_solve_time = time.perf_counter() - solve_started
    wall_time = data.preparation_time + reduced_solve_time
    print(
        f"    timings: preparation={data.preparation_time:.2f}s, "
        f"reduced solve={reduced_solve_time:.2f}s, "
        f"solver={output.solve_time:.2f}s, total={wall_time:.2f}s"
    )
    z = output.coefficients
    moment_coeffs = z[: data.dim_m]
    deflection_coeffs = z[data.dim_m :]
    evaluated = evaluate_feature_result(
        spec.label,
        wall_time,
        moment_coeffs,
        deflection_coeffs,
        data.eval_data,
    )
    result = AlgorithmResult(
        name=evaluated.name,
        r_c=evaluated.r_c,
        r_e=evaluated.r_e,
        abs_u=evaluated.abs_u,
        abs_M=evaluated.abs_M,
        wall_time=evaluated.wall_time,
        rank=output.rank,
        columns=data.column_count,
        condition_estimate=output.condition_estimate,
        w_h2_error=evaluated.w_h2_error,
        M_hdivdiv_error=evaluated.M_hdivdiv_error,
        coefficient_ball_active=output.regularization_active,
        coefficient_norm=float(torch.linalg.vector_norm(z)),
        algorithm=spec.id,
        hyperparameter=output.hyperparameter,
    )
    print_result_summary(result)
    return result


def prepare_experiment(
    cfg: LeastSquaresConfig,
    benchmark: SharedBenchmarkData,
    feature_space: SharedFeatureSpace,
) -> LeastSquaresExperimentData:
    """Assemble the residual system once; reusable across coefficient budgets."""

    validate_config(cfg)
    if feature_space.theta_m.shape[0] != cfg.N_m or feature_space.theta_w.shape[0] != cfg.N_u:
        raise ValueError("SharedFeatureSpace feature counts do not match LeastSquaresConfig.")
    source_rows = 4 * benchmark.x_int.shape[0]
    expected_columns = 3 * (cfg.N_m + 1) + (cfg.N_u + 1)
    backend = get_system_backend(cfg.system_backend)
    if not backend.stores_normal_equations:
        warn_if_dense_infeasible(cfg.direct_solver, source_rows, expected_columns)
        if cfg.direct_solver == "dense":
            print("Assembling dense direct weighted residual matrix...")
            t0 = time.perf_counter()
            direct_design = assemble_direct_residual_design(benchmark, feature_space)
            preparation_time = time.perf_counter() - t0
            print(
                f"Residual shapes: A={tuple(direct_design.matrix.shape)}, "
                f"b={tuple(direct_design.rhs.shape)}, "
                f"assembly={preparation_time:.2f}s"
            )
        else:
            print("Compressing direct residuals with streaming TSQR...")
            direct_design, streaming_stats = assemble_streaming_tsqr_design(
                cfg,
                benchmark,
                feature_space,
            )
            preparation_time = streaming_stats.total_time
            print(
                f"Residual shapes: source A=({streaming_stats.source_rows}, "
                f"{streaming_stats.columns}), "
                f"reduced A={tuple(direct_design.matrix.shape)}, "
                f"b={tuple(direct_design.rhs.shape)}"
            )
            print(
                f"TSQR timings: batch assembly={streaming_stats.assembly_time:.2f}s, "
                f"QR updates={streaming_stats.qr_time:.2f}s, "
                f"compression total={streaming_stats.total_time:.2f}s"
            )
        residual_design: torch.Tensor | GramResidualDesign = direct_design.matrix
        rhs: torch.Tensor | None = direct_design.rhs
        dim_m = direct_design.dim_m
        column_count = int(direct_design.matrix.shape[1])
    else:
        print("Accumulating weighted residuals with the streaming Gram backend...")
        t0 = time.perf_counter()
        gram_design, dim_m = assemble_streaming_gram_design(
            cfg,
            benchmark,
            feature_space,
        )
        preparation_time = time.perf_counter() - t0
        if gram_design.source_rows != source_rows:
            raise RuntimeError(
                "Gram residual row count mismatch: "
                f"assembled {gram_design.source_rows}, expected {source_rows}."
            )
        print(
            f"Residual shapes: source A=({gram_design.source_rows}, "
            f"{gram_design.column_count}), "
            f"G={tuple(gram_design.gram.shape)}, "
            f"assembly={preparation_time:.2f}s"
        )
        residual_design = gram_design
        rhs = None
        column_count = gram_design.column_count
    clear_cuda_cache()

    return LeastSquaresExperimentData(
        residual_design=residual_design,
        rhs=rhs,
        column_count=column_count,
        dim_m=dim_m,
        eval_data=build_feature_evaluation_data(
            benchmark,
            feature_space,
            cfg.evaluation_batch_size,
        ),
        source_rows=source_rows,
        preparation_time=preparation_time,
        system_backend=cfg.system_backend,
        direct_solver=cfg.direct_solver,
    )


def retarget_experiment_data(
    cfg: LeastSquaresConfig,
    data: LeastSquaresExperimentData,
    benchmark: SharedBenchmarkData,
    feature_space: SharedFeatureSpace,
) -> LeastSquaresExperimentData:
    """Point an assembled residual system at a different test rule.

    Only ``eval_data`` depends on the test quadrature; the residual system and
    its spectral factorization depend on the training points alone.  A run that
    selects its hyperparameter on validation points and then reports on test
    points therefore assembles and factorizes once, not twice.
    """

    return replace(
        data,
        eval_data=build_feature_evaluation_data(
            benchmark,
            feature_space,
            cfg.evaluation_batch_size,
        ),
    )


def run_experiment(
    cfg: LeastSquaresConfig | None = None,
    print_table: bool = True,
    plot_results: bool = True,
    benchmark: SharedBenchmarkData | None = None,
    feature_space: SharedFeatureSpace | None = None,
    experiment_data: LeastSquaresExperimentData | None = None,
) -> list[AlgorithmResult]:
    """Run direct residual least squares and return its metrics."""

    cfg = default_config() if cfg is None else cfg
    validate_config(cfg)
    selected_algorithm_ids = validate_algorithm_selection(cfg.algorithms_to_run)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}")
    print(
        f"Config: h={cfg.h}, N_m={cfg.N_m}, N_u={cfg.N_u}, "
        f"Q_train={cfg.Q_train}, Q_test={cfg.Q_test}, "
        f"activation=ReLU^{cfg.activation_power}, "
        f"ritz_degree={cfg.ritz_degree}, ritz_ratio={cfg.ritz_ratio}, "
        f"coefficient_budget={cfg.coefficient_budget}, "
        f"ridge_lambda={cfg.ridge_lambda:.2e}, "
        f"direct_rcond={cfg.direct_rcond:.2e}, "
        f"system_backend={cfg.system_backend}, "
        f"direct_solver={cfg.direct_solver}, "
        f"sampling={cfg.sampling_method}"
    )
    print(f"Algorithms: {selected_algorithm_ids}")
    required_degree = matched_spline_degree(FEATURE_DIM, cfg.activation_power)
    if cfg.ritz_degree < required_degree:
        warnings.warn(
            f"ritz_degree={cfg.ritz_degree} saturates at Sobolev index "
            f"{cfg.ritz_degree + 1}, below the dictionary's "
            f"s_cap({FEATURE_DIM})="
            f"{saturation_index(FEATURE_DIM, cfg.activation_power):g}. "
            f"The Ritz space, not the dictionary, will limit the deflection "
            f"error; use ritz_degree >= {required_degree}.",
            RuntimeWarning,
        )

    D = compute_bending_stiffness(cfg.E, cfg.nu, cfg.h)
    print(f"Material: E={cfg.E}, nu={cfg.nu}, h={cfg.h}, D={D:.4f}")

    if experiment_data is None:
        if benchmark is None:
            print("Building benchmark data...")
            benchmark = build_shared_benchmark(
                E=cfg.E,
                nu=cfg.nu,
                h=cfg.h,
                Q_train=cfg.Q_train,
                Q_test=cfg.Q_test,
                sampling_method=cfg.sampling_method,
                manufactured_solution=cfg.manufactured_solution,
            )
        else:
            print("Using shared benchmark data...")

        if feature_space is None:
            print("Generating quasi-uniform feature spaces...")
            feature_space = build_shared_feature_space(
                N_m=cfg.N_m,
                N_u=cfg.N_u,
                activation_power=cfg.activation_power,
                ritz_degree=cfg.ritz_degree,
                ritz_ratio=cfg.ritz_ratio,
                projection_samples=cfg.projection_samples,
            )
        else:
            print("Using shared feature spaces...")

        print(
            "Ritz projection: "
            f"K={feature_space.projected_w.space.dimension}, "
            f"degree={feature_space.projected_w.degree}, "
            f"samples={feature_space.projected_w.quadrature_samples}, "
            f"residual={feature_space.projected_w.gram_residual:.2e}, "
            f"boundary={feature_space.projected_w.boundary_residual():.2e}"
        )
        experiment_data = prepare_experiment(cfg, benchmark, feature_space)
    else:
        print("Using prepared residual system...")

    results = [
        run_algorithm(algorithm_id, experiment_data, cfg)
        for algorithm_id in selected_algorithm_ids
    ]
    if print_table:
        print_summary_table(results, title="LS Summary")

    if plot_results:
        print("\nGenerating plots...")
        plot_error_summary(
            [result.name for result in results],
            [
                r"Deflection $\|w_N-w_\star\|_{H^2}$",
                r"Moment $\|M_N-M_\star\|_{H(\mathrm{div\,div})}$",
            ],
            [
                [result.w_h2_error for result in results],
                [result.M_hdivdiv_error for result in results],
            ],
            str(OUTPUT_DIR / "graph-error-summary.png"),
        )

    return results


def main(cfg: LeastSquaresConfig | None = None) -> None:
    """Script entrypoint."""

    run_experiment(cfg, print_table=True, plot_results=True)


if __name__ == "__main__":
    main()
