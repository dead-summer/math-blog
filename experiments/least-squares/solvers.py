"""Pluggable coefficient solvers for the least-squares experiments.

Every algorithm consumes one shared spectral factor of the weighted residual
system and returns physical coefficients plus compact diagnostics.  The
factor comes either from a direct SVD of ``A`` or an eigendecomposition of the
streamed normal equations ``G=A.T@A``.  Three algorithms are registered:

``ball``
    The paper's physical-coefficient ball solver: minimize ``||Ac-b||``
    subject to ``||c||_2 <= B / sqrt(m)`` where ``B`` is the coefficient
    budget and ``m`` the column count.
``ridge``
    Tikhonov regularization with ``lam = lambda_rel * sigma_max``; the
    hyperparameter is the relative level ``lambda_rel``.
``tsvd``
    Truncated-SVD minimum-norm solution; the hyperparameter is the relative
    cutoff ``rcond``.

All three share one cached factor, so hyperparameter ladders and algorithm
comparisons cost a single SVD/eigendecomposition.  Neither backend scales the
columns; coefficient-ball radii therefore retain their physical meaning.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import torch

from rfm_core import (
    BallLeastSquaresResult,
    factorize_l2_ball_least_squares,
)
from system_backends import (
    GramResidualDesign,
    LeastSquaresSpectralFactor,
    factorize_gram_least_squares,
)


@dataclass(frozen=True)
class SolverOutput:
    """Coefficients and diagnostics from one algorithm run."""

    coefficients: torch.Tensor
    rank: int
    condition_estimate: float
    regularization_active: bool
    hyperparameter: float
    solve_time: float


@dataclass(frozen=True)
class SolverSpec:
    """One registered coefficient solver.

    ``config_field`` names the configuration attribute holding the
    single-run hyperparameter, so callers can sweep a ladder with
    ``dataclasses.replace(cfg, **{spec.config_field: value})``.
    """

    id: str
    label: str
    hyperparameter_name: str
    config_field: str
    default_ladder: tuple[float, ...]
    solve: Callable[[LeastSquaresSpectralFactor, object, float], BallLeastSquaresResult]


def _solve_ball(
    factor: LeastSquaresSpectralFactor,
    cfg: object,
    budget: float,
) -> BallLeastSquaresResult:
    if budget <= 0.0:
        raise ValueError("coefficient budget must be positive")
    radius = budget / math.sqrt(max(factor.column_count, 1))
    return factor.solve(radius, rcond=cfg.direct_rcond)


def _solve_ridge(
    factor: LeastSquaresSpectralFactor,
    cfg: object,
    lambda_rel: float,
) -> BallLeastSquaresResult:
    if not (math.isfinite(lambda_rel) and lambda_rel > 0.0):
        raise ValueError("ridge lambda must be finite and positive")
    # solve_ridge itself handles an empty factor.  Avoid routing that case
    # through rcond=0: spectral cutoffs must stay finite and positive.
    spectral_scale = (
        float(factor.singular_values[0]) if factor.singular_values.size else 1.0
    )
    return factor.solve_ridge(lambda_rel * spectral_scale)


def _solve_tsvd(
    factor: LeastSquaresSpectralFactor,
    cfg: object,
    rcond: float,
) -> BallLeastSquaresResult:
    if not (math.isfinite(rcond) and rcond > 0.0):
        raise ValueError("tsvd rcond must be finite and positive")
    return factor.solve_min_norm(rcond)


SOLVERS: dict[str, SolverSpec] = {
    spec.id: spec
    for spec in (
        SolverSpec(
            id="ball",
            label="LS(ball)",
            hyperparameter_name="budget",
            config_field="coefficient_budget",
            default_ladder=(1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6, math.inf),
            solve=_solve_ball,
        ),
        SolverSpec(
            id="ridge",
            label="LS(ridge)",
            hyperparameter_name="lambda_rel",
            config_field="ridge_lambda",
            default_ladder=(1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5),
            solve=_solve_ridge,
        ),
        SolverSpec(
            id="tsvd",
            label="LS(tsvd)",
            hyperparameter_name="rcond",
            config_field="direct_rcond",
            default_ladder=(1.0e-10, 1.0e-8, 1.0e-6),
            solve=_solve_tsvd,
        ),
    )
}

# Historical name of the only algorithm before the registry existed.
ALGORITHM_ALIASES = {"direct": "ball"}
VALID_ALGORITHMS = tuple(SOLVERS)


def resolve_algorithm_id(algorithm_id: str) -> str:
    """Map legacy aliases onto registry ids; unknown ids pass through."""

    return ALGORITHM_ALIASES.get(algorithm_id, algorithm_id)


def get_solver_spec(algorithm_id: str) -> SolverSpec:
    resolved = resolve_algorithm_id(algorithm_id)
    if resolved not in SOLVERS:
        raise ValueError(
            f"Unknown algorithm '{algorithm_id}'. Valid ids: {list(SOLVERS)}"
        )
    return SOLVERS[resolved]


def get_factor(
    matrix: torch.Tensor | GramResidualDesign,
    rhs: torch.Tensor | None = None,
) -> LeastSquaresSpectralFactor:
    """Return a cached direct-SVD or Gram-eigh spectral factor."""

    if isinstance(matrix, GramResidualDesign):
        if rhs is not None:
            raise ValueError("rhs must be None for a GramResidualDesign")
        return factorize_gram_least_squares(matrix)
    if rhs is None:
        raise ValueError("rhs is required for a direct residual matrix")

    cached = getattr(matrix, "_ls_svd_factor", None)
    if cached is not None and cached[0] is rhs:
        return cached[1]
    factor = factorize_l2_ball_least_squares(matrix.numpy(), rhs.numpy())
    matrix._ls_svd_factor = (rhs, factor)
    return factor


def run_solver(
    algorithm_id: str,
    matrix: torch.Tensor | GramResidualDesign,
    rhs: torch.Tensor | None,
    cfg: object,
    hyperparameter: float | None = None,
) -> SolverOutput:
    """Solve the assembled system with one registered algorithm.

    ``hyperparameter`` defaults to the value stored on ``cfg`` under the
    solver's ``config_field``.
    """

    spec = get_solver_spec(algorithm_id)
    started = time.perf_counter()
    factor = get_factor(matrix, rhs)
    value = (
        float(getattr(cfg, spec.config_field))
        if hyperparameter is None
        else float(hyperparameter)
    )
    result = spec.solve(factor, cfg, value)
    return SolverOutput(
        coefficients=torch.from_numpy(result.coefficients).to(dtype=torch.float64),
        rank=result.rank,
        condition_estimate=result.condition_estimate,
        regularization_active=result.constraint_active,
        hyperparameter=value,
        solve_time=time.perf_counter() - started,
    )
