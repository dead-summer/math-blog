"""Dimension-generic stress--displacement least-squares driver.

One implementation covers 2D linear elasticity, 3D linear elasticity, and 2D
plane stress: the three cases differ only in the spatial dimension ``d``, the
Voigt machinery generated from ``d``, the compliance matrix, and whether the
zero-mean-trace gauge is imposed on the stress space.

Manufactured solutions are pluggable: a problem provides one displacement
callback per solution name, and the exact stress, displacement gradient, and
body force ``f = -div sigma`` are all derived from it with autodiff.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from ls_common import (
    ALGO_STYLE,
    BASE_SEED,
    DEVICE,
    DTYPE,
    VALID_ALGORITHMS,
    VALID_DIRECT_SOLVERS,
    VALID_SAMPLING_METHODS,
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
from rfm_core import (
    RitzProjectedFeatures,
    build_ritz_projected_features,
    matched_spline_degree,
    relu_power_feature_box_means,
    relu_power_feature_values_and_gradients,
    saturation_index,
    validate_activation_power,
)
from solvers import get_solver_spec, run_solver
from system_backends import (
    GramResidualDesign,
    assemble_gram_residual_design,
    get_system_backend,
)


PROJECTION_SEED = BASE_SEED + 17_000

# Voigt component orders fixed to match the historical per-dimension drivers.
VOIGT_PAIRS = {
    2: ((0, 0), (1, 1), (0, 1)),
    3: ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)),
}

# Orthonormal bases of trace-free symmetric tensors in Voigt coordinates,
# fixed (not generated) so that column layouts match the historical drivers.
DEVIATORIC_BASES = {
    2: (
        (1.0 / math.sqrt(2.0), 0.0),
        (-1.0 / math.sqrt(2.0), 0.0),
        (0.0, 1.0 / math.sqrt(2.0)),
    ),
    3: (
        (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(6.0), 0.0, 0.0, 0.0),
        (-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(6.0), 0.0, 0.0, 0.0),
        (0.0, -2.0 / math.sqrt(6.0), 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0 / math.sqrt(2.0), 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0 / math.sqrt(2.0), 0.0),
        (0.0, 0.0, 0.0, 0.0, 1.0 / math.sqrt(2.0)),
    ),
}


@dataclass(frozen=True)
class VoigtSpec:
    """Voigt bookkeeping for symmetric ``d x d`` tensors."""

    dimension: int
    pairs: tuple[tuple[int, int], ...]
    voigt_weight: torch.Tensor
    trace_voigt: torch.Tensor
    strain_grad_bases: torch.Tensor
    deviatoric_bases: torch.Tensor
    hydrostatic_basis: torch.Tensor

    @property
    def components(self) -> int:
        return len(self.pairs)

    @property
    def deviatoric_dim(self) -> int:
        return self.deviatoric_bases.shape[1]

    @property
    def engineering_frobenius_weight(self) -> torch.Tensor:
        """Metric for engineering-strain Voigt vectors.

        Stress vectors store physical shear entries and therefore use
        ``voigt_weight``.  Engineering strain stores twice each shear entry,
        so its tensor Frobenius metric is the reciprocal weight.
        """

        return self.voigt_weight.reciprocal()


def make_voigt_spec(dimension: int) -> VoigtSpec:
    """Assemble the Voigt machinery for ``dimension`` in {2, 3}."""

    if dimension not in VOIGT_PAIRS:
        raise ValueError("Only spatial dimensions 2 and 3 are supported.")
    pairs = VOIGT_PAIRS[dimension]
    n_v = len(pairs)
    voigt_weight = torch.tensor(
        [1.0 if i == j else 2.0 for i, j in pairs],
        dtype=DTYPE,
        device=DEVICE,
    )
    trace_voigt = torch.tensor(
        [1.0 if i == j else 0.0 for i, j in pairs],
        dtype=DTYPE,
        device=DEVICE,
    )
    # strain_grad_bases[k, v, c]: contribution of d(u_c)/d(x_k) to the
    # engineering Voigt strain component v.
    strain_grad = torch.zeros(dimension, n_v, dimension, dtype=DTYPE, device=DEVICE)
    for v, (i, j) in enumerate(pairs):
        if i == j:
            strain_grad[i, v, i] = 1.0
        else:
            strain_grad[j, v, i] = 1.0
            strain_grad[i, v, j] = 1.0
    deviatoric = torch.tensor(DEVIATORIC_BASES[dimension], dtype=DTYPE, device=DEVICE)
    return VoigtSpec(
        dimension=dimension,
        pairs=pairs,
        voigt_weight=voigt_weight,
        trace_voigt=trace_voigt,
        strain_grad_bases=strain_grad,
        deviatoric_bases=deviatoric,
        hydrostatic_basis=trace_voigt / math.sqrt(float(dimension)),
    )


@dataclass(frozen=True)
class Material:
    """One constitutive law in engineering Voigt form."""

    compliance_voigt: torch.Tensor
    stiffness_voigt: torch.Tensor
    lam: float
    summary: str


def isotropic_material(spec: VoigtSpec, E: float, nu: float) -> Material:
    """Isotropic (mu, lambda) material in dimension ``spec.dimension``."""

    d = spec.dimension
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    n_v = spec.components
    compliance = torch.zeros(n_v, n_v, dtype=DTYPE, device=DEVICE)
    compliance[:d, :d] = torch.eye(d, dtype=DTYPE, device=DEVICE) / (2.0 * mu)
    compliance[:d, :d] -= (
        lam / (2.0 * mu * (2.0 * mu + d * lam))
    ) * torch.ones(d, d, dtype=DTYPE, device=DEVICE)
    for v in range(d, n_v):
        compliance[v, v] = 1.0 / mu
    return Material(
        compliance_voigt=compliance,
        stiffness_voigt=torch.linalg.inv(compliance),
        lam=lam,
        summary=f"E={E}, nu={nu}, mu={mu:.4f}, lam={lam:.4f}",
    )


def plane_stress_material(spec: VoigtSpec, E: float, nu: float) -> Material:
    """Plane-stress constitutive law (spec.dimension must be 2)."""

    if spec.dimension != 2:
        raise ValueError("Plane stress is a two-dimensional model.")
    mu = E / (2.0 * (1.0 + nu))
    lambda_plane = E * nu / (1.0 - nu * nu)
    compliance = torch.zeros(3, 3, dtype=DTYPE, device=DEVICE)
    compliance[0, 0] = 1.0 / E
    compliance[1, 1] = 1.0 / E
    compliance[0, 1] = -nu / E
    compliance[1, 0] = -nu / E
    compliance[2, 2] = 2.0 * (1.0 + nu) / E
    return Material(
        compliance_voigt=compliance,
        stiffness_voigt=torch.linalg.inv(compliance),
        lam=lambda_plane,
        summary=f"E={E}, nu={nu}, mu={mu:.4f}, lambda_plane={lambda_plane:.4f}",
    )


DisplacementFn = Callable[[torch.Tensor], torch.Tensor]
# A material-dependent manufactured solution returns the displacement callback.
SolutionFactory = Callable[[Material], DisplacementFn]


@dataclass(frozen=True)
class ElasticityProblem:
    """Everything that distinguishes one stress--displacement experiment."""

    name: str
    spec: VoigtSpec
    material_law: Callable[[VoigtSpec, float, float], Material]
    solutions: dict[str, SolutionFactory]
    default_solution: str
    use_trace_constraint: bool
    output_dir: Path
    nu_upper_inclusive: bool = False

    def make_material_from(self, E: float, nu: float) -> Material:
        return self.material_law(self.spec, E, nu)

    def make_material(self, cfg: "LeastSquaresConfig") -> Material:
        return self.make_material_from(cfg.E, cfg.nu)


@dataclass
class LeastSquaresConfig:
    """Configuration shared by all stress--displacement experiments."""

    E: float = 4.0 / 3.0
    nu: float = 1.0 / 3.0
    N_s: int = 1000
    N_u: int = 1000
    Q_train: int = 4 * 1001
    Q_test: int = 128**2
    sampling_method: str = "mc"
    activation_power: int = 7
    ritz_degree: int = 7
    ritz_ratio: float = 2.0
    projection_samples: int | None = None
    projection_batch_size: int = 2_048
    coefficient_budget: float = math.inf
    ridge_lambda: float = 1.0e-12
    manufactured_solution: str = "hu_zhang"
    direct_rcond: float = 1.0e-12
    system_backend: str = "direct"
    direct_solver: str = "streaming_tsqr"
    direct_batch_size: int = 1_024
    direct_qr_block_size: int = 64
    body_force_batch_size: int = 5_000
    evaluation_batch_size: int = 4_096
    algorithms_to_run: list[str] = field(
        default_factory=lambda: ["ball", "ridge", "tsvd"]
    )


def validate_config(problem: ElasticityProblem, cfg: LeastSquaresConfig) -> None:
    """Validate config before starting any expensive work."""

    if cfg.E <= 0.0:
        raise ValueError("Config.E must be positive.")
    if problem.nu_upper_inclusive:
        if not (-1.0 < cfg.nu <= 0.5):
            raise ValueError("Config.nu must lie in (-1, 0.5].")
    elif not (-1.0 < cfg.nu < 0.5):
        raise ValueError("Config.nu must lie in (-1, 0.5).")
    if cfg.N_s <= 0 or cfg.N_u <= 0:
        raise ValueError("Config.N_s and Config.N_u must be positive.")
    # The stress-displacement graph norm carries one derivative, so the
    # training-generalization argument needs rho_k in W^{2,inf}.
    validate_activation_power(cfg.activation_power, sobolev_order=1)
    if cfg.ritz_degree < 3:
        raise ValueError("Config.ritz_degree must be at least 3.")
    if cfg.ritz_ratio < 1.0:
        raise ValueError("Config.ritz_ratio must be at least one.")
    if cfg.projection_samples is not None and cfg.projection_samples <= 0:
        raise ValueError("Config.projection_samples must be positive when set.")
    if cfg.projection_batch_size <= 0:
        raise ValueError("Config.projection_batch_size must be positive.")
    if cfg.coefficient_budget <= 0.0:
        raise ValueError("Config.coefficient_budget must be positive.")
    if not math.isfinite(cfg.ridge_lambda) or cfg.ridge_lambda <= 0.0:
        raise ValueError("Config.ridge_lambda must be finite and positive.")
    if cfg.Q_train <= 0:
        raise ValueError("Config.Q_train must be positive.")
    if cfg.Q_test <= 0:
        raise ValueError("Config.Q_test must be positive.")
    if not math.isfinite(cfg.direct_rcond) or cfg.direct_rcond <= 0.0:
        raise ValueError("Config.direct_rcond must be finite and positive.")
    get_system_backend(cfg.system_backend)
    if cfg.direct_solver not in VALID_DIRECT_SOLVERS:
        raise ValueError(
            f"Unknown direct_solver='{cfg.direct_solver}'. "
            f"Valid values: {list(VALID_DIRECT_SOLVERS)}"
        )
    if cfg.direct_batch_size <= 0:
        raise ValueError("Config.direct_batch_size must be positive.")
    if cfg.direct_qr_block_size <= 0:
        raise ValueError("Config.direct_qr_block_size must be positive.")
    if cfg.body_force_batch_size <= 0:
        raise ValueError("Config.body_force_batch_size must be positive.")
    if cfg.evaluation_batch_size <= 0:
        raise ValueError("Config.evaluation_batch_size must be positive.")
    validate_sampling_method(cfg.sampling_method)
    resolve_solution_factory(problem, getattr(cfg, "manufactured_solution", None))
    validate_algorithm_selection(
        cfg.algorithms_to_run,
        VALID_ALGORITHMS,
    )


def resolve_solution_factory(
    problem: ElasticityProblem,
    manufactured_solution: str | None,
) -> SolutionFactory:
    """Look up one manufactured-solution factory by name."""

    name = manufactured_solution or problem.default_solution
    if name not in problem.solutions:
        raise ValueError(
            f"Unknown manufactured_solution='{name}'. "
            f"Valid values: {sorted(problem.solutions)}"
        )
    return problem.solutions[name]


# ---------------------------------------------------------------------------
# Exact fields from a displacement callback (autodiff pipeline)
# ---------------------------------------------------------------------------


def compute_displacement_gradient(
    displacement: DisplacementFn,
    x: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the exact displacement gradient ``grad[q, c, k]``."""

    d = x.shape[1]
    x_ad = x.detach().requires_grad_(True)
    u = displacement(x_ad)
    gradient = torch.stack(
        [
            torch.autograd.grad(
                u[:, component].sum(),
                x_ad,
                retain_graph=component < d - 1,
            )[0]
            for component in range(d)
        ],
        dim=1,
    )
    return gradient.detach()


def gradient_to_engineering_strain(
    spec: VoigtSpec,
    grad_u: torch.Tensor,
) -> torch.Tensor:
    """Convert vector gradients to engineering Voigt strain."""

    columns = []
    for i, j in spec.pairs:
        if i == j:
            columns.append(grad_u[:, i, i])
        else:
            columns.append(grad_u[:, i, j] + grad_u[:, j, i])
    return torch.stack(columns, dim=1)


def compute_stress_voigt(
    problem: ElasticityProblem,
    material: Material,
    displacement: DisplacementFn,
    x: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the exact stress in the problem's Voigt order."""

    grad_u = compute_displacement_gradient(displacement, x)
    strain = gradient_to_engineering_strain(problem.spec, grad_u)
    return strain @ material.stiffness_voigt.T


def compute_body_force(
    problem: ElasticityProblem,
    material: Material,
    displacement: DisplacementFn,
    x: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Compute ``f = -div(sigma(u_exact))`` with batched autodiff."""

    spec = problem.spec
    d = spec.dimension
    n_points = x.shape[0]
    f_all = torch.zeros(n_points, d, dtype=DTYPE, device=DEVICE)
    # sigma[i, j] sits in Voigt slot voigt_index[(min(i,j), max(i,j))].
    voigt_index = {pair: v for v, pair in enumerate(spec.pairs)}

    for start in range(0, n_points, batch_size):
        end = min(start + batch_size, n_points)
        xb = x[start:end].detach().requires_grad_(True)
        u = displacement(xb)
        grad_u = torch.stack(
            [
                torch.autograd.grad(
                    u[:, comp].sum(),
                    xb,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                for comp in range(d)
            ],
            dim=1,
        )
        strain = gradient_to_engineering_strain(spec, grad_u)
        sigma = strain @ material.stiffness_voigt.T

        for comp in range(d):
            div_sigma = torch.zeros(end - start, dtype=DTYPE, device=DEVICE)
            for dim_k in range(d):
                pair = (min(comp, dim_k), max(comp, dim_k))
                last = comp == d - 1 and dim_k == d - 1
                grad_sigma = torch.autograd.grad(
                    sigma[:, voigt_index[pair]].sum(),
                    xb,
                    create_graph=False,
                    retain_graph=not last,
                )[0]
                div_sigma += grad_sigma[:, dim_k]
            f_all[start:end, comp] = -div_sigma.detach()

    return f_all


# ---------------------------------------------------------------------------
# Shared data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlgorithmResult:
    """Compact metrics for one completed algorithm."""

    name: str
    u_l2_error: float
    sigma_l2_error: float
    wall_time: float
    rank: int = 0
    columns: int = 0
    condition_estimate: float = float("nan")
    u_h1_error: float = float("nan")
    sigma_hdiv_error: float = float("nan")
    constitutive_residual: float = float("nan")
    equilibrium_residual: float = float("nan")
    coefficient_ball_active: bool = False
    trace_residual: float = float("nan")
    continuous_trace_mean: float = float("nan")
    sigma_deviatoric_l2_error: float = float("nan")
    sigma_hydrostatic_l2_error: float = float("nan")
    u_h1_exact_norm: float = float("nan")
    sigma_hdiv_exact_norm: float = float("nan")
    relative_u_h1_error: float = float("nan")
    relative_sigma_hdiv_error: float = float("nan")
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
    sigma_exact_test: torch.Tensor
    u_grad_exact_test: torch.Tensor
    f_test: torch.Tensor
    compliance_voigt: torch.Tensor


@dataclass(frozen=True)
class SharedFeatureSpace:
    """Shared deterministic feature spaces used by coefficient-based methods."""

    theta_s: torch.Tensor
    theta_u: torch.Tensor
    projected_u: RitzProjectedFeatures
    activation_power: int = 3
    mean_raw_sigma: torch.Tensor | None = None


@dataclass(frozen=True)
class StressBasisAdapter:
    """Map between solved stress coefficients and the raw Voigt basis.

    With the trace gauge the solved coefficients consist of a deviatoric
    block (feature-major, stride ``deviatoric_dim``) followed by a zero-mean
    hydrostatic block; without it they are raw Voigt coefficients.
    """

    spec: VoigtSpec
    use_trace_constraint: bool
    mean_raw_sigma: torch.Tensor | None
    raw_dim: int
    active_dim: int

    @property
    def constraint(self) -> torch.Tensor:
        if self.mean_raw_sigma is None:
            raise ValueError("No trace constraint on this adapter.")
        return (
            self.mean_raw_sigma.unsqueeze(1) * self.spec.trace_voigt.unsqueeze(0)
        ).reshape(-1)


def build_stress_basis_adapter(
    spec: VoigtSpec,
    use_trace_constraint: bool,
    mean_raw_sigma: torch.Tensor | None,
    np1_s: int,
) -> StressBasisAdapter:
    raw_dim = spec.components * np1_s
    if not use_trace_constraint:
        return StressBasisAdapter(spec, False, None, raw_dim, raw_dim)
    if mean_raw_sigma is None or mean_raw_sigma.numel() != np1_s:
        raise ValueError("Stress-basis mean has the wrong dimension.")
    if torch.abs(mean_raw_sigma[0]) <= 1.0e-12:
        raise ValueError("Raw stress basis has degenerate mean; cannot build zero-mean basis.")
    return StressBasisAdapter(spec, True, mean_raw_sigma, raw_dim, raw_dim - 1)


def lift_stress_coefficients(
    adapter: StressBasisAdapter,
    solved: torch.Tensor,
) -> torch.Tensor:
    """Lift solved stress coefficients back to the raw Voigt basis."""

    if not adapter.use_trace_constraint:
        return solved
    spec = adapter.spec
    mean_raw_sigma = adapter.mean_raw_sigma
    np1_s = mean_raw_sigma.numel()
    n_dev = spec.deviatoric_dim
    deviatoric_dim_s = n_dev * np1_s
    if solved.numel() != adapter.active_dim:
        raise ValueError("Active stress coefficient dimension does not match its adapter.")

    deviatoric_coefficients = solved[:deviatoric_dim_s].reshape(np1_s, n_dev)
    raw_coefficients = deviatoric_coefficients @ spec.deviatoric_bases.T

    active_hydrostatic = solved[deviatoric_dim_s:]
    hydrostatic_coefficients = torch.empty(
        np1_s,
        dtype=solved.dtype,
        device=solved.device,
    )
    hydrostatic_coefficients[0] = -torch.dot(
        mean_raw_sigma[1:] / mean_raw_sigma[0],
        active_hydrostatic,
    )
    hydrostatic_coefficients[1:] = active_hydrostatic
    raw_coefficients = raw_coefficients + (
        hydrostatic_coefficients.unsqueeze(1) * spec.hydrostatic_basis.unsqueeze(0)
    )
    return raw_coefficients.reshape(-1)


@dataclass(frozen=True)
class FeatureEvaluationData:
    """All tensors needed to evaluate coefficient-based methods.

    Only per-point exact data and the feature *descriptions* are held.  The
    dictionary and Ritz bases are re-evaluated in blocks by
    :func:`evaluate_feature_result`, so nothing of size ``Q_test * N`` is ever
    stored; caching them would not save work either, because every metric that
    needs their values also needs their gradients from the same call.
    """

    w_test: torch.Tensor
    u_exact_test: torch.Tensor
    sigma_exact_test: torch.Tensor
    u_grad_exact_test: torch.Tensor
    f_test: torch.Tensor
    x_test: torch.Tensor
    theta_s: torch.Tensor
    projected_u: RitzProjectedFeatures
    compliance_voigt: torch.Tensor
    evaluation_batch_size: int
    activation_power: int = 3


@dataclass(frozen=True)
class DirectResidualDesign:
    """Weighted residual matrix plus coefficient-space metadata."""

    matrix: torch.Tensor
    rhs: torch.Tensor
    solved_dim_s: int
    stress_adapter: StressBasisAdapter


@dataclass(frozen=True)
class LeastSquaresExperimentData:
    """All tensors needed to run and evaluate one least-squares solver."""

    residual_design: torch.Tensor | GramResidualDesign
    rhs: torch.Tensor | None
    column_count: int
    solved_dim_s: int
    stress_adapter: StressBasisAdapter
    eval_data: FeatureEvaluationData
    source_rows: int
    preparation_time: float
    system_backend: str
    direct_solver: str


# ---------------------------------------------------------------------------
# Benchmark and feature-space builders
# ---------------------------------------------------------------------------


def build_shared_benchmark(
    problem: ElasticityProblem,
    E: float,
    nu: float,
    Q_train: int,
    Q_test: int,
    sampling_method: str,
    body_force_batch_size: int,
    manufactured_solution: str | None = None,
    interior_seed: int = BASE_SEED + 1,
    test_seed: int = BASE_SEED + 3,
) -> SharedBenchmarkData:
    """Build the shared train/test samples for one run."""

    d = problem.spec.dimension
    material = problem.make_material_from(E, nu)
    displacement = resolve_solution_factory(problem, manufactured_solution)(material)

    x_int, w_int = build_quadrature_rule(
        Q_train,
        method=sampling_method,
        dim=d,
        seed=interior_seed,
    )
    f_int = compute_body_force(
        problem,
        material,
        displacement,
        x_int,
        batch_size=body_force_batch_size,
    )

    x_test, w_test = build_quadrature_rule(
        Q_test,
        method="gauss_legendre",
        dim=d,
        seed=test_seed,
    )
    return SharedBenchmarkData(
        x_int=x_int,
        w_int=w_int,
        f_int=f_int,
        x_test=x_test,
        w_test=w_test,
        u_exact_test=displacement(x_test).detach(),
        sigma_exact_test=compute_stress_voigt(problem, material, displacement, x_test),
        u_grad_exact_test=compute_displacement_gradient(displacement, x_test),
        f_test=compute_body_force(
            problem,
            material,
            displacement,
            x_test,
            batch_size=body_force_batch_size,
        ),
        compliance_voigt=material.compliance_voigt,
    )


def build_shared_feature_space(
    problem: ElasticityProblem,
    N_s: int,
    N_u: int,
    activation_power: int = 3,
    ritz_degree: int = 3,
    ritz_ratio: float = 2.0,
    projection_samples: int | None = None,
    projection_batch_size: int | None = None,
    projection_seed: int = PROJECTION_SEED,
) -> SharedFeatureSpace:
    """Build the quasi-uniform features and the H1 Ritz-projected displacement basis.

    The hidden parameters are deterministic.  A Monte Carlo rule discretizes
    the displacement Ritz inner product, so the physical trial space does not
    depend on training data.  The stress trace projection needs no rule at
    all: on the unit box every raw stress feature has a closed-form mean, so
    the zero-mean gauge is exact and carries no quadrature bias.

    ``ritz_degree`` must keep the spline saturation ``ritz_degree + 1`` at or
    above the dictionary's ``s_cap(d)``; otherwise the displacement error is
    capped by this auxiliary space rather than by the dictionary.
    """

    d = problem.spec.dimension
    validate_activation_power(activation_power, sobolev_order=1)
    theta_s = generate_features(N_s, d, activation_power)
    theta_u = generate_features(N_u, d, activation_power)
    projected_u = build_ritz_projected_features(
        theta_u,
        sobolev_order=1,
        auxiliary_dimension=max(
            N_u + 1,
            int(math.ceil(ritz_ratio * (N_u + 1))),
        ),
        power=activation_power,
        degree=ritz_degree,
        quadrature_samples=projection_samples,
        quadrature_seed=projection_seed,
        batch_size=projection_batch_size,
    )
    mean_raw_sigma = (
        relu_power_feature_box_means(theta_s, activation_power)
        if problem.use_trace_constraint
        else None
    )
    return SharedFeatureSpace(
        theta_s=theta_s,
        theta_u=theta_u,
        projected_u=projected_u,
        activation_power=activation_power,
        mean_raw_sigma=mean_raw_sigma,
    )


def build_feature_evaluation_data(
    benchmark: SharedBenchmarkData,
    feature_space: SharedFeatureSpace,
    evaluation_batch_size: int,
) -> FeatureEvaluationData:
    """Build the shared evaluation tensors for coefficient-based methods."""

    return FeatureEvaluationData(
        w_test=benchmark.w_test,
        u_exact_test=benchmark.u_exact_test,
        sigma_exact_test=benchmark.sigma_exact_test,
        u_grad_exact_test=benchmark.u_grad_exact_test,
        f_test=benchmark.f_test,
        x_test=benchmark.x_test,
        theta_s=feature_space.theta_s,
        projected_u=feature_space.projected_u,
        compliance_voigt=benchmark.compliance_voigt,
        evaluation_batch_size=evaluation_batch_size,
        activation_power=feature_space.activation_power,
    )


def accumulate_raw_sigma_mean(
    benchmark: SharedBenchmarkData,
    feature_space: SharedFeatureSpace,
    batch_size: int,
) -> torch.Tensor:
    """Compatibility accessor for the closed-form stress-feature box means."""

    del benchmark, batch_size
    if feature_space.mean_raw_sigma is None:
        raise ValueError(
            "SharedFeatureSpace has no trace projection; rebuild it for a "
            "trace-constrained elasticity problem."
        )
    return feature_space.mean_raw_sigma


# ---------------------------------------------------------------------------
# Residual assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectResidualContext:
    """Dimensions and fixed coupling blocks shared by residual batches."""

    problem: ElasticityProblem
    stress_adapter: StressBasisAdapter
    compliance_voigt: torch.Tensor
    np1_s: int
    np1_u: int
    solved_dim_s: int
    dim_u: int
    columns: int


def build_direct_residual_context(
    problem: ElasticityProblem,
    benchmark: SharedBenchmarkData,
    feature_space: SharedFeatureSpace,
    mean_raw_sigma: torch.Tensor | None,
) -> DirectResidualContext:
    np1_s = feature_space.theta_s.shape[0] + 1
    np1_u = feature_space.theta_u.shape[0] + 1
    adapter = build_stress_basis_adapter(
        problem.spec,
        problem.use_trace_constraint,
        mean_raw_sigma,
        np1_s,
    )
    dim_u = problem.spec.dimension * np1_u
    return DirectResidualContext(
        problem=problem,
        stress_adapter=adapter,
        compliance_voigt=benchmark.compliance_voigt,
        np1_s=np1_s,
        np1_u=np1_u,
        solved_dim_s=adapter.active_dim,
        dim_u=dim_u,
        columns=adapter.active_dim + dim_u,
    )


def _fill_residual_rows(
    context: DirectResidualContext,
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    batch_points: int,
    raw_sigma: torch.Tensor,
    grad_sigma: torch.Tensor,
    grad_u: torch.Tensor,
    sqrt_weights: torch.Tensor,
    body_force: torch.Tensor,
) -> None:
    """Write the weighted residual rows for one batch of points.

    All feature tensors must already carry the sqrt-weight scaling.  The
    constitutive residual is ``A sigma_h - eps(u_h)``; the equilibrium
    residual is ``div sigma_h + f`` (``f`` moves to the right-hand side).
    """

    spec = context.problem.spec
    d = spec.dimension
    n_v = spec.components
    compliance = context.compliance_voigt
    adapter = context.stress_adapter
    constrained = adapter.use_trace_constraint
    if constrained:
        n_dev = spec.deviatoric_dim
        deviatoric_dim_s = n_dev * context.np1_s
        hydrostatic_basis = raw_sigma[:, 1:] - raw_sigma[:, :1] * (
            adapter.mean_raw_sigma[1:] / adapter.mean_raw_sigma[0]
        ).unsqueeze(0)
        compliance_deviatoric = compliance @ spec.deviatoric_bases
        compliance_hydrostatic = compliance @ spec.hydrostatic_basis

    for residual_component in range(n_v):
        rows = slice(
            residual_component * batch_points,
            (residual_component + 1) * batch_points,
        )
        if constrained:
            for deviatoric_component in range(n_dev):
                matrix[rows, deviatoric_component:deviatoric_dim_s:n_dev] = (
                    compliance_deviatoric[residual_component, deviatoric_component]
                    * raw_sigma
                )
            matrix[rows, deviatoric_dim_s : adapter.active_dim] = (
                compliance_hydrostatic[residual_component] * hydrostatic_basis
            )
        else:
            for sigma_component in range(n_v):
                matrix[rows, sigma_component : adapter.active_dim : n_v] = (
                    compliance[residual_component, sigma_component] * raw_sigma
                )
        for spatial_dimension in range(d):
            coupling = spec.strain_grad_bases[spatial_dimension]
            for displacement_component in range(d):
                matrix[
                    rows,
                    adapter.active_dim + displacement_component : context.columns : d,
                ] -= (
                    coupling[residual_component, displacement_component]
                    * grad_u[:, :, spatial_dimension]
                )
        # ``A sigma - epsilon(u)`` is stored in engineering-strain Voigt
        # coordinates.  Scale each component so its Euclidean row norm is
        # exactly the tensor Frobenius norm used by the paper.
        matrix[rows].mul_(
            torch.sqrt(spec.engineering_frobenius_weight[residual_component])
        )

    for equilibrium_component in range(d):
        rows = slice(
            (n_v + equilibrium_component) * batch_points,
            (n_v + 1 + equilibrium_component) * batch_points,
        )
        for spatial_dimension in range(d):
            coupling = spec.strain_grad_bases[spatial_dimension][
                :, equilibrium_component
            ]
            if constrained:
                deviatoric_coefficients = coupling @ spec.deviatoric_bases
                for deviatoric_component in range(n_dev):
                    matrix[rows, deviatoric_component:deviatoric_dim_s:n_dev] += (
                        deviatoric_coefficients[deviatoric_component]
                        * grad_sigma[:, :, spatial_dimension]
                    )
                # The zero-mean correction only changes the strictly constant
                # feature in column zero, whose gradient is exactly zero.
                matrix[rows, deviatoric_dim_s : adapter.active_dim] += (
                    torch.dot(coupling, spec.hydrostatic_basis)
                    * grad_sigma[:, 1:, spatial_dimension]
                )
            else:
                for sigma_component in range(n_v):
                    matrix[rows, sigma_component : adapter.active_dim : n_v] += (
                        coupling[sigma_component]
                        * grad_sigma[:, :, spatial_dimension]
                    )
        rhs[rows] = -sqrt_weights * body_force[:, equilibrium_component]


def _weighted_feature_data(
    benchmark: SharedBenchmarkData,
    feature_space: SharedFeatureSpace,
    start: int,
    stop: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate sqrt-weight-scaled features on one slice of training points."""

    x = benchmark.x_int[start:stop]
    weights = benchmark.w_int[start:stop]
    body_force = benchmark.f_int[start:stop]
    raw_sigma, grad_sigma = relu_power_feature_values_and_gradients(
        x, feature_space.theta_s, feature_space.activation_power
    )
    _, grad_u = feature_space.projected_u.evaluate_values_and_gradients(x)
    sqrt_weights = torch.sqrt(weights)
    raw_sigma.mul_(sqrt_weights.unsqueeze(1))
    grad_sigma.mul_(sqrt_weights.view(-1, 1, 1))
    grad_u.mul_(sqrt_weights.view(-1, 1, 1))
    return raw_sigma, grad_sigma, grad_u, sqrt_weights, body_force


def assemble_direct_residual_batch(
    start: int,
    stop: int,
    benchmark: SharedBenchmarkData,
    feature_space: SharedFeatureSpace,
    context: DirectResidualContext,
) -> np.ndarray:
    """Assemble one Fortran-contiguous augmented residual block ``[A_i, b_i]``."""

    point_count = benchmark.x_int.shape[0]
    if not (0 <= start < stop <= point_count):
        raise ValueError(
            f"Invalid residual batch [{start}, {stop}) for {point_count} points."
        )
    spec = context.problem.spec
    rows_per_point = spec.components + spec.dimension
    batch_points = stop - start
    augmented = np.zeros(
        (rows_per_point * batch_points, context.columns + 1),
        dtype=np.float64,
        order="F",
    )
    augmented_torch = torch.from_numpy(augmented)
    matrix = augmented_torch[:, : context.columns]
    rhs = augmented_torch[:, context.columns]

    with torch.no_grad():
        raw_sigma, grad_sigma, grad_u, sqrt_weights, body_force = _weighted_feature_data(
            benchmark,
            feature_space,
            start,
            stop,
        )
        _fill_residual_rows(
            context,
            matrix,
            rhs,
            batch_points,
            raw_sigma,
            grad_sigma,
            grad_u,
            sqrt_weights,
            body_force,
        )
    return augmented


def assemble_direct_residual_design(
    problem: ElasticityProblem,
    benchmark: SharedBenchmarkData,
    feature_space: SharedFeatureSpace,
) -> DirectResidualDesign:
    """Assemble the weighted constitutive and equilibrium residuals densely."""

    mean_raw_sigma = None
    if problem.use_trace_constraint:
        mean_raw_sigma = accumulate_raw_sigma_mean(
            benchmark,
            feature_space,
            benchmark.x_int.shape[0],
        )
    context = build_direct_residual_context(
        problem,
        benchmark,
        feature_space,
        mean_raw_sigma,
    )
    spec = problem.spec
    rows_per_point = spec.components + spec.dimension
    q_count = benchmark.x_int.shape[0]
    # Allocate transposed storage so the NumPy view is Fortran contiguous,
    # the layout LAPACK's SVD consumes directly.
    matrix = torch.zeros(
        context.columns,
        rows_per_point * q_count,
        dtype=DTYPE,
        device=DEVICE,
    ).T
    rhs = torch.zeros(rows_per_point * q_count, dtype=DTYPE, device=DEVICE)

    with torch.no_grad():
        raw_sigma, grad_sigma, grad_u, sqrt_weights, body_force = _weighted_feature_data(
            benchmark,
            feature_space,
            0,
            q_count,
        )
        _fill_residual_rows(
            context,
            matrix,
            rhs,
            q_count,
            raw_sigma,
            grad_sigma,
            grad_u,
            sqrt_weights,
            body_force,
        )
    return DirectResidualDesign(
        matrix,
        rhs,
        context.solved_dim_s,
        context.stress_adapter,
    )


def assemble_streaming_tsqr_design(
    problem: ElasticityProblem,
    cfg: LeastSquaresConfig,
    benchmark: SharedBenchmarkData,
    feature_space: SharedFeatureSpace,
    show_progress: bool = True,
):
    """Compress ``[A, b]`` with streaming Householder TSQR."""

    mean_raw_sigma = None
    if problem.use_trace_constraint:
        mean_raw_sigma = accumulate_raw_sigma_mean(
            benchmark,
            feature_space,
            cfg.direct_batch_size,
        )
    context = build_direct_residual_context(
        problem,
        benchmark,
        feature_space,
        mean_raw_sigma,
    )
    spec = problem.spec
    matrix, rhs, stats = streaming_tsqr_compress(
        point_count=benchmark.x_int.shape[0],
        rows_per_point=spec.components + spec.dimension,
        columns=context.columns,
        batch_size=cfg.direct_batch_size,
        qr_block_size=cfg.direct_qr_block_size,
        assemble_augmented_batch=lambda start, stop: assemble_direct_residual_batch(
            start,
            stop,
            benchmark,
            feature_space,
            context,
        ),
        show_progress=show_progress,
    )
    design = DirectResidualDesign(
        matrix,
        rhs,
        context.solved_dim_s,
        context.stress_adapter,
    )
    return design, stats


def assemble_streaming_gram_design(
    problem: ElasticityProblem,
    cfg: LeastSquaresConfig,
    benchmark: SharedBenchmarkData,
    feature_space: SharedFeatureSpace,
) -> tuple[GramResidualDesign, DirectResidualContext]:
    """Stream the current weighted residual batches into ``A.T A`` and ``A.T b``.

    This backend deliberately reuses :func:`assemble_direct_residual_batch`.
    It therefore changes only how the linear system is stored, not the Monte
    Carlo functional, Voigt/Frobenius metric, trace projection, or physical
    coefficient norm.
    """

    mean_raw_sigma = None
    if problem.use_trace_constraint:
        mean_raw_sigma = accumulate_raw_sigma_mean(
            benchmark,
            feature_space,
            cfg.direct_batch_size,
        )
    context = build_direct_residual_context(
        problem,
        benchmark,
        feature_space,
        mean_raw_sigma,
    )
    design = assemble_gram_residual_design(
        point_count=benchmark.x_int.shape[0],
        batch_size=cfg.direct_batch_size,
        column_count=context.columns,
        assemble_augmented_batch=lambda start, stop: assemble_direct_residual_batch(
            start,
            stop,
            benchmark,
            feature_space,
            context,
        ),
        dtype=DTYPE,
        device=DEVICE,
    )
    return design, context


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_feature_result(
    problem: ElasticityProblem,
    name: str,
    wall_time: float,
    sigma_coeffs: torch.Tensor,
    displacement_coeffs: torch.Tensor,
    data: FeatureEvaluationData,
) -> AlgorithmResult:
    """Evaluate one coefficient-based method and package the metrics.

    Every metric is a weighted sum over the deterministic test rule, so the
    test points are consumed in blocks of ``data.evaluation_batch_size``: the
    dictionary and Ritz bases and their gradients are never materialized at the
    full ``Q_test x N x d`` size, which otherwise dominates peak memory at the
    widths where the reduced system itself is still small.
    """

    spec = problem.spec
    d = spec.dimension
    n_v = spec.components
    stress_blocks = sigma_coeffs.reshape(-1, n_v)
    displacement_blocks = displacement_coeffs.reshape(-1, d)
    voigt_index = {pair: v for v, pair in enumerate(spec.pairs)}

    squared: dict[str, float] = dict.fromkeys(
        (
            "u_l2",
            "sigma_l2",
            "sigma_hydrostatic",
            "sigma_deviatoric",
            "displacement_gradient",
            "constitutive",
            "equilibrium",
            "u_exact_l2",
            "u_exact_grad",
            "sigma_exact_l2",
            "div_sigma_exact_l2",
        ),
        0.0,
    )
    continuous_trace_mean = 0.0

    with torch.no_grad():
        for start, stop in iter_point_batches(
            data.x_test.shape[0],
            data.evaluation_batch_size,
        ):
            w = data.w_test[start:stop]
            u_exact = data.u_exact_test[start:stop]
            sigma_exact = data.sigma_exact_test[start:stop]
            u_grad_exact = data.u_grad_exact_test[start:stop]
            body_force = data.f_test[start:stop]

            raw_sigma, raw_sigma_gradient = relu_power_feature_values_and_gradients(
                data.x_test[start:stop],
                data.theta_s,
                data.activation_power,
            )
            u_basis, displacement_gradient_basis = (
                data.projected_u.evaluate_values_and_gradients(data.x_test[start:stop])
            )

            u_h = u_basis @ displacement_blocks
            sigma_h = raw_sigma @ stress_blocks
            grad_u_h = torch.einsum(
                "qfs,fc->qcs",
                displacement_gradient_basis,
                displacement_blocks,
            )

            sigma_error = sigma_h - sigma_exact
            trace_error = sigma_error[:, :d].sum(dim=1)
            deviatoric_error = sigma_error.clone()
            deviatoric_error[:, :d] -= trace_error.unsqueeze(1) / float(d)

            strain_h = gradient_to_engineering_strain(spec, grad_u_h)
            constitutive = sigma_h @ data.compliance_voigt.T - strain_h

            div_sigma_columns = []
            for comp in range(d):
                accum = torch.zeros(stop - start, dtype=DTYPE, device=DEVICE)
                for dim_k in range(d):
                    pair = (min(comp, dim_k), max(comp, dim_k))
                    accum = accum + raw_sigma_gradient[:, :, dim_k] @ stress_blocks[
                        :, voigt_index[pair]
                    ]
                div_sigma_columns.append(accum)
            equilibrium = torch.stack(div_sigma_columns, dim=1) + body_force

            squared["u_l2"] += (w * (u_h - u_exact).square().sum(dim=1)).sum().item()
            squared["sigma_l2"] += (
                w * (spec.voigt_weight * sigma_error.square()).sum(dim=1)
            ).sum().item()
            squared["sigma_hydrostatic"] += (
                w * (trace_error.square() / float(d))
            ).sum().item()
            squared["sigma_deviatoric"] += (
                w * (spec.voigt_weight * deviatoric_error.square()).sum(dim=1)
            ).sum().item()
            squared["displacement_gradient"] += (
                w * (grad_u_h - u_grad_exact).square().sum((1, 2))
            ).sum().item()
            squared["constitutive"] += (
                w * (spec.engineering_frobenius_weight * constitutive.square()).sum(dim=1)
            ).sum().item()
            squared["equilibrium"] += (w * equilibrium.square().sum(dim=1)).sum().item()
            squared["u_exact_l2"] += (w * u_exact.square().sum(dim=1)).sum().item()
            squared["u_exact_grad"] += (w * u_grad_exact.square().sum((1, 2))).sum().item()
            squared["sigma_exact_l2"] += (
                w * (spec.voigt_weight * sigma_exact.square()).sum(dim=1)
            ).sum().item()
            squared["div_sigma_exact_l2"] += (
                w * body_force.square().sum(dim=1)
            ).sum().item()
            continuous_trace_mean += (w * sigma_h[:, :d].sum(dim=1)).sum().item()

    u_l2_error = math.sqrt(squared["u_l2"])
    sigma_l2_error = math.sqrt(squared["sigma_l2"])
    sigma_hydrostatic_l2_error = math.sqrt(squared["sigma_hydrostatic"])
    sigma_deviatoric_l2_error = math.sqrt(squared["sigma_deviatoric"])
    displacement_gradient_error = math.sqrt(squared["displacement_gradient"])
    constitutive_residual = math.sqrt(squared["constitutive"])
    equilibrium_residual = math.sqrt(squared["equilibrium"])
    u_exact_l2 = math.sqrt(squared["u_exact_l2"])
    u_exact_grad = math.sqrt(squared["u_exact_grad"])
    sigma_exact_l2 = math.sqrt(squared["sigma_exact_l2"])
    div_sigma_exact_l2 = math.sqrt(squared["div_sigma_exact_l2"])
    u_h1_error = math.hypot(u_l2_error, displacement_gradient_error)
    sigma_hdiv_error = math.hypot(sigma_l2_error, equilibrium_residual)
    u_h1_exact_norm = math.hypot(u_exact_l2, u_exact_grad)
    sigma_hdiv_exact_norm = math.hypot(sigma_exact_l2, div_sigma_exact_l2)
    return AlgorithmResult(
        name=name,
        u_l2_error=u_l2_error,
        sigma_l2_error=sigma_l2_error,
        wall_time=wall_time,
        u_h1_error=u_h1_error,
        sigma_hdiv_error=sigma_hdiv_error,
        constitutive_residual=constitutive_residual,
        equilibrium_residual=equilibrium_residual,
        continuous_trace_mean=continuous_trace_mean,
        sigma_deviatoric_l2_error=sigma_deviatoric_l2_error,
        sigma_hydrostatic_l2_error=sigma_hydrostatic_l2_error,
        u_h1_exact_norm=u_h1_exact_norm,
        sigma_hdiv_exact_norm=sigma_hdiv_exact_norm,
        relative_u_h1_error=u_h1_error / u_h1_exact_norm,
        relative_sigma_hdiv_error=sigma_hdiv_error / sigma_hdiv_exact_norm,
    )


def print_result_summary(result: AlgorithmResult) -> None:
    """Print one compact result line."""

    print(
        f"    [{result.name}] Done in {result.wall_time:.2f}s, "
        f"hyper={result.hyperparameter:g}, "
        f"‖u_N-u_*‖_H1={result.u_h1_error:.2e}, "
        f"‖sigma_N-sigma_*‖_Hdiv={result.sigma_hdiv_error:.2e}, "
        f"constitutive={result.constitutive_residual:.2e}, "
        f"equilibrium={result.equilibrium_residual:.2e}, "
        f"test-trace={result.continuous_trace_mean:.2e}, "
        f"rank={result.rank}/{result.columns}, "
        f"cond≈{result.condition_estimate:.2e}"
    )
    print(
        "    stress L2 decomposition: "
        f"total={result.sigma_l2_error:.2e}, "
        f"deviatoric={result.sigma_deviatoric_l2_error:.2e}, "
        f"hydrostatic={result.sigma_hydrostatic_l2_error:.2e}; "
        f"relative graph errors=(u {result.relative_u_h1_error:.2e}, "
        f"sigma {result.relative_sigma_hdiv_error:.2e})"
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
        "‖u_N-u_*‖_L2",
        "‖u_N-u_*‖_H1",
        "‖sigma_N-sigma_*‖_L2",
        "‖sigma_N-sigma_*‖_Hdiv",
        "Time(s)",
    )
    rows = [
        (
            result.name,
            f"{result.u_l2_error:.2e}",
            f"{result.u_h1_error:.2e}",
            f"{result.sigma_l2_error:.2e}",
            f"{result.sigma_hdiv_error:.2e}",
            f"{result.wall_time:.2f}",
        )
        for result in results
    ]
    print_aligned_markdown_table(
        title=title,
        headers=headers,
        rows=rows,
        alignments=(
            "left",
            "center",
            "center",
            "center",
            "center",
            "center",
        ),
    )


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------


def run_algorithm(
    problem: ElasticityProblem,
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
    solved_sigma = z[: data.solved_dim_s]
    displacement_coeffs = z[data.solved_dim_s :]
    sigma_coeffs = lift_stress_coefficients(data.stress_adapter, solved_sigma)
    trace_residual = float("nan")
    if data.stress_adapter.use_trace_constraint:
        trace_residual = torch.dot(
            data.stress_adapter.constraint,
            sigma_coeffs,
        ).abs().item()
        print(f"    trace constraint residual: {trace_residual:.2e}")

    evaluated = evaluate_feature_result(
        problem,
        spec.label,
        wall_time,
        sigma_coeffs,
        displacement_coeffs,
        data.eval_data,
    )
    result = AlgorithmResult(
        name=evaluated.name,
        u_l2_error=evaluated.u_l2_error,
        sigma_l2_error=evaluated.sigma_l2_error,
        wall_time=evaluated.wall_time,
        rank=output.rank,
        columns=data.column_count,
        condition_estimate=output.condition_estimate,
        u_h1_error=evaluated.u_h1_error,
        sigma_hdiv_error=evaluated.sigma_hdiv_error,
        constitutive_residual=evaluated.constitutive_residual,
        equilibrium_residual=evaluated.equilibrium_residual,
        coefficient_ball_active=output.regularization_active,
        trace_residual=trace_residual,
        continuous_trace_mean=evaluated.continuous_trace_mean,
        sigma_deviatoric_l2_error=evaluated.sigma_deviatoric_l2_error,
        sigma_hydrostatic_l2_error=evaluated.sigma_hydrostatic_l2_error,
        u_h1_exact_norm=evaluated.u_h1_exact_norm,
        sigma_hdiv_exact_norm=evaluated.sigma_hdiv_exact_norm,
        relative_u_h1_error=evaluated.relative_u_h1_error,
        relative_sigma_hdiv_error=evaluated.relative_sigma_hdiv_error,
        coefficient_norm=float(torch.linalg.vector_norm(z)),
        algorithm=spec.id,
        hyperparameter=output.hyperparameter,
    )
    print_result_summary(result)
    return result


def prepare_experiment(
    problem: ElasticityProblem,
    cfg: LeastSquaresConfig,
    benchmark: SharedBenchmarkData,
    feature_space: SharedFeatureSpace,
) -> LeastSquaresExperimentData:
    """Assemble the residual system once; reusable across coefficient budgets."""

    validate_config(problem, cfg)
    if feature_space.theta_s.shape[0] != cfg.N_s or feature_space.theta_u.shape[0] != cfg.N_u:
        raise ValueError("SharedFeatureSpace dimensions do not match LeastSquaresConfig.")
    spec = problem.spec
    source_rows = (spec.components + spec.dimension) * benchmark.x_int.shape[0]
    expected_columns = (
        spec.components * (cfg.N_s + 1)
        + spec.dimension * (cfg.N_u + 1)
        - (1 if problem.use_trace_constraint else 0)
    )
    backend = get_system_backend(cfg.system_backend)
    if not backend.stores_normal_equations:
        warn_if_dense_infeasible(cfg.direct_solver, source_rows, expected_columns)
        if cfg.direct_solver == "dense":
            print("Assembling dense direct weighted residual matrix...")
            t0 = time.perf_counter()
            direct_design = assemble_direct_residual_design(
                problem,
                benchmark,
                feature_space,
            )
            preparation_time = time.perf_counter() - t0
            print(
                f"Residual shapes: A={tuple(direct_design.matrix.shape)}, "
                f"b={tuple(direct_design.rhs.shape)}, "
                f"assembly={preparation_time:.2f}s"
            )
        else:
            print("Compressing direct residuals with streaming TSQR...")
            direct_design, streaming_stats = assemble_streaming_tsqr_design(
                problem,
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
        solved_dim_s = direct_design.solved_dim_s
        stress_adapter = direct_design.stress_adapter
        column_count = int(direct_design.matrix.shape[1])
    else:
        print("Accumulating weighted residuals with the streaming Gram backend...")
        t0 = time.perf_counter()
        gram_design, residual_context = assemble_streaming_gram_design(
            problem,
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
        solved_dim_s = residual_context.solved_dim_s
        stress_adapter = residual_context.stress_adapter
        column_count = gram_design.column_count
    clear_cuda_cache()
    if problem.use_trace_constraint:
        print(
            "trace constraint applied: "
            f"stress dof {stress_adapter.raw_dim} "
            f"-> {stress_adapter.active_dim}"
        )

    return LeastSquaresExperimentData(
        residual_design=residual_design,
        rhs=rhs,
        column_count=column_count,
        solved_dim_s=solved_dim_s,
        stress_adapter=stress_adapter,
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
    problem: ElasticityProblem,
    cfg: LeastSquaresConfig,
    print_table: bool = True,
    plot_results: bool = True,
    benchmark: SharedBenchmarkData | None = None,
    feature_space: SharedFeatureSpace | None = None,
    experiment_data: LeastSquaresExperimentData | None = None,
) -> list[AlgorithmResult]:
    """Run the selected residual-system backend and return graph-error metrics."""

    validate_config(problem, cfg)
    selected_algorithm_ids = validate_algorithm_selection(cfg.algorithms_to_run)
    problem.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Output: {problem.output_dir}")
    print(
        f"Config: N_s={cfg.N_s}, N_u={cfg.N_u}, "
        f"Q_train={cfg.Q_train}, Q_test={cfg.Q_test}, "
        f"activation=ReLU^{cfg.activation_power}, "
        f"ritz_degree={cfg.ritz_degree}, ritz_ratio={cfg.ritz_ratio}, "
        f"projection_samples={cfg.projection_samples}, "
        f"projection_batch_size={cfg.projection_batch_size}, "
        f"coefficient_budget={cfg.coefficient_budget}, "
        f"ridge_lambda={cfg.ridge_lambda:.2e}, "
        f"direct_rcond={cfg.direct_rcond:.2e}, "
        f"system_backend={cfg.system_backend}, "
        f"direct_solver={cfg.direct_solver}, "
        f"direct_batch_size={cfg.direct_batch_size}, "
        f"direct_qr_block_size={cfg.direct_qr_block_size}, "
        f"sampling={cfg.sampling_method}, "
        "manufactured_solution="
        f"{getattr(cfg, 'manufactured_solution', None) or problem.default_solution}"
    )
    required_degree = matched_spline_degree(problem.spec.dimension, cfg.activation_power)
    if cfg.ritz_degree < required_degree:
        warnings.warn(
            f"ritz_degree={cfg.ritz_degree} saturates at Sobolev index "
            f"{cfg.ritz_degree + 1}, below the dictionary's "
            f"s_cap({problem.spec.dimension})="
            f"{saturation_index(problem.spec.dimension, cfg.activation_power):g}. "
            f"The Ritz space, not the dictionary, will limit the displacement "
            f"error; use ritz_degree >= {required_degree}.",
            RuntimeWarning,
        )
    print(f"Algorithms: {selected_algorithm_ids}")
    print(f"Material: {problem.make_material(cfg).summary}")

    if experiment_data is None:
        if benchmark is None:
            print("Building benchmark data...")
            benchmark = build_shared_benchmark(
                problem,
                E=cfg.E,
                nu=cfg.nu,
                Q_train=cfg.Q_train,
                Q_test=cfg.Q_test,
                sampling_method=cfg.sampling_method,
                body_force_batch_size=cfg.body_force_batch_size,
                manufactured_solution=getattr(cfg, "manufactured_solution", None),
            )
        else:
            print("Using shared benchmark data...")

        if feature_space is None:
            print("Generating quasi-uniform feature spaces...")
            feature_space = build_shared_feature_space(
                problem,
                N_s=cfg.N_s,
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
            f"K={feature_space.projected_u.space.dimension}, "
            f"degree={feature_space.projected_u.degree}, "
            f"samples={feature_space.projected_u.quadrature_samples}, "
            f"residual={feature_space.projected_u.gram_residual:.2e}, "
            f"boundary={feature_space.projected_u.boundary_residual():.2e}"
        )
        if problem.use_trace_constraint:
            print("Trace gauge: exact box means")
        experiment_data = prepare_experiment(problem, cfg, benchmark, feature_space)
    else:
        print("Using prepared residual system...")

    results = [
        run_algorithm(problem, algorithm_id, experiment_data, cfg)
        for algorithm_id in selected_algorithm_ids
    ]
    if print_table:
        print_summary_table(results, title="LS Summary")

    if plot_results:
        print("\nGenerating plots...")
        plot_error_summary(
            [result.name for result in results],
            [
                r"$\|u_N-u_\star\|_{H^1}$",
                r"$\|\sigma_N-\sigma_\star\|_{H(\mathrm{div})}$",
            ],
            [
                [result.u_h1_error for result in results],
                [result.sigma_hdiv_error for result in results],
            ],
            str(problem.output_dir / "graph-error-summary.png"),
        )

    return results
