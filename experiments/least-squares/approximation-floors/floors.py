"""Best-approximation floors for the linearized-network trial spaces.

Every quantity here is a *best-approximation* problem: given a trial space, how
close can any element of it get to the exact solution, measured in the norm the
paper reports?  No least-squares solver, quadrature rule, or regularizer is
involved, so the numbers isolate the approximation power of the dictionary and
of the auxiliary Ritz space from everything the driver does afterwards.

Two design choices make the results trustworthy:

* every fit is solved by streaming Householder QR of the tall design matrix,
  never by normal equations, so a floor near ``1e-13`` is not an artefact of
  squaring the condition number;
* the exact fields come from the production drivers' own
  ``build_shared_benchmark``, so the manufactured solutions, the Voigt
  conventions, and the test quadrature match the campaign exactly.

The theory this checks is the saturation index of the ``rho_k`` dictionary,
``s_cap(d) = (d + 2k + 1) / 2``: a quasi-uniform ``N``-point parameter set
approximates ``H^{s_cap}`` functions in ``H^m`` at rate ``N^{-beta}`` with
``beta = (s_cap(d) - m) / d``.  Two consequences drive the whole analysis --
each derivative in the error norm costs one factor ``N^{1/d}``, and each unit of
activation power buys one.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import scipy.linalg
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rfm_core import (  # noqa: E402
    TensorSplineSpace,
    approximation_rate,
    build_ritz_projected_features,
    quasi_uniform_features,
    relu_power_feature_data,
    saturation_index,
)
from study_runner import MODEL_SPECS, load_model  # noqa: E402


POINT_BATCH = 4096
# Peak transient during accumulation is the block plus the vstack copy the QR
# makes of it, so budget for roughly twice one block.
BLOCK_BUDGET_BYTES = 192 << 20


def block_points(rows_per_point: int, column_count: int) -> int:
    """Points per accumulation block that keep one block inside the budget."""

    per_point = 8 * rows_per_point * (column_count + 1)
    return max(1, min(POINT_BATCH, BLOCK_BUDGET_BYTES // max(per_point, 1)))


# ---------------------------------------------------------------------------
# Streaming least squares by Householder QR
# ---------------------------------------------------------------------------


def batches(total: int, size: int) -> Iterator[tuple[int, int]]:
    for start in range(0, total, size):
        yield start, min(start + size, total)


def accumulate_qr(blocks: Iterator[np.ndarray], column_count: int) -> np.ndarray:
    """Reduce a stream of tall blocks to one ``R`` factor.

    Each block is stacked under the running factor and re-triangularized, so
    peak memory is one block plus ``R`` regardless of the total row count.
    """

    reduced = np.zeros((0, column_count), dtype=np.float64)
    for block in blocks:
        if block.shape[1] != column_count:
            raise ValueError(
                f"block has {block.shape[1]} columns, expected {column_count}"
            )
        reduced = scipy.linalg.qr(
            np.vstack([reduced, block]),
            mode="r",
            check_finite=False,
        )[0]
        reduced = reduced[: min(reduced.shape[0], column_count)]
    return reduced


def solve_augmented(reduced: np.ndarray, design_columns: int) -> np.ndarray:
    """Least-squares coefficients from the ``R`` factor of ``[A | B]``."""

    return scipy.linalg.solve_triangular(
        reduced[:design_columns, :design_columns],
        reduced[:design_columns, design_columns:],
        check_finite=False,
    )


BlockBuilder = Callable[[int, int], np.ndarray]


def best_fit(
    build_block: BlockBuilder,
    point_count: int,
    design_columns: int,
    rhs_columns: int,
    batch_size: int = POINT_BATCH,
) -> np.ndarray:
    """Solve one weighted least-squares fit and return its coefficients."""

    reduced = accumulate_qr(
        (build_block(start, stop) for start, stop in batches(point_count, batch_size)),
        design_columns + rhs_columns,
    )
    return solve_augmented(reduced, design_columns)


def weighted_block(parts: list[np.ndarray], right: list[np.ndarray]) -> np.ndarray:
    """Stack row groups into one Fortran-contiguous augmented block."""

    return np.asfortranarray(np.hstack([np.vstack(parts), np.vstack(right)]))


# ---------------------------------------------------------------------------
# Trial spaces
# ---------------------------------------------------------------------------


Evaluator = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


def raw_dictionary(parameters: torch.Tensor, power: int, dimension: int) -> Evaluator:
    """Values, gradients and Hessians of the unprojected ``rho_power`` dictionary."""

    components = tuple(
        (row, column) for row in range(dimension) for column in range(row, dimension)
    )

    def evaluate(points: torch.Tensor):
        return relu_power_feature_data(
            points, parameters, power, hessian_components=components
        )

    return evaluate


def spline_space(space: TensorSplineSpace) -> Evaluator:
    """Values, gradients and Hessians of a boundary-adapted spline space."""

    def evaluate(points: torch.Tensor):
        data = space.evaluate(points.numpy(), derivative_order=2)
        dense = lambda matrix: torch.from_numpy(np.asarray(matrix.todense()))
        return (
            dense(data.values),
            torch.stack([dense(matrix) for matrix in data.gradients], dim=2),
            torch.stack([dense(matrix) for matrix in data.hessians], dim=2),
        )

    return evaluate


def projected_dictionary(projected) -> Evaluator:
    """Values, gradients and Hessians of the Ritz-projected trial space."""

    def evaluate(points: torch.Tensor):
        return projected.evaluate(points)

    return evaluate


# ---------------------------------------------------------------------------
# Scalar-field floors (displacement components, plate deflection)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScalarFloor:
    l2: float
    h1: float
    h2: float


def scalar_floor(
    evaluate: Evaluator,
    column_count: int,
    points: torch.Tensor,
    weights: torch.Tensor,
    values: torch.Tensor,
    gradients: torch.Tensor,
    hessians: torch.Tensor | None,
    order: int,
) -> ScalarFloor:
    """Best fit of one vector field in ``H^order``; reports L2/H1/H2 errors.

    ``values`` has shape ``(Q, c)``, ``gradients`` ``(Q, c, d)`` and ``hessians``
    ``(Q, c, n_v)`` in the driver's Voigt order.  All ``c`` components share the
    dictionary, so they are fitted simultaneously as multiple right-hand sides.
    """

    dimension = points.shape[1]
    component_count = values.shape[1]
    sqrt_weights = torch.sqrt(weights)
    # The tensor Frobenius norm of a symmetric Hessian counts each off-diagonal
    # entry twice, so the fit and the error use the same sqrt(2) scaling.
    hessian_scale = np.array(
        [1.0 if row == column else math.sqrt(2.0)
         for row in range(dimension) for column in range(row, dimension)]
    )

    def build_block(start: int, stop: int) -> np.ndarray:
        scale = sqrt_weights[start:stop].numpy()[:, None]
        basis, basis_gradient, basis_hessian = evaluate(points[start:stop])
        parts = [basis.numpy() * scale]
        right = [values[start:stop].numpy() * scale]
        if order >= 1:
            for axis in range(dimension):
                parts.append(basis_gradient[:, :, axis].numpy() * scale)
                right.append(gradients[start:stop, :, axis].numpy() * scale)
        if order >= 2:
            for index, factor in enumerate(hessian_scale):
                parts.append(factor * basis_hessian[:, :, index].numpy() * scale)
                right.append(factor * hessians[start:stop, :, index].numpy() * scale)
        return weighted_block(parts, right)

    rows_per_point = 1 + (dimension if order >= 1 else 0)
    if order >= 2:
        rows_per_point += len(hessian_scale)
    coefficients = best_fit(
        build_block,
        points.shape[0],
        column_count,
        component_count,
        batch_size=block_points(rows_per_point, column_count + component_count),
    )

    squared = [0.0, 0.0, 0.0]
    for start, stop in batches(points.shape[0], POINT_BATCH):
        basis, basis_gradient, basis_hessian = evaluate(points[start:stop])
        block_weights = weights[start:stop].numpy()
        error = basis.numpy() @ coefficients - values[start:stop].numpy()
        squared[0] += float((block_weights * (error**2).sum(1)).sum())
        gradient_error = (
            np.einsum("qfk,fc->qck", basis_gradient.numpy(), coefficients)
            - gradients[start:stop].numpy()
        )
        squared[1] += float((block_weights * (gradient_error**2).sum((1, 2))).sum())
        if hessians is not None:
            hessian_error = (
                np.einsum("qfk,fc->qck", basis_hessian.numpy(), coefficients)
                - hessians[start:stop].numpy()
            )
            squared[2] += float(
                (
                    block_weights
                    * (hessian_scale**2 * hessian_error**2).sum((1, 2))
                ).sum()
            )
    return ScalarFloor(
        l2=math.sqrt(squared[0]),
        h1=math.sqrt(squared[0] + squared[1]),
        h2=math.sqrt(sum(squared)),
    )


# ---------------------------------------------------------------------------
# Tensor-field floors (stress in H(div), moment in H(div div))
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorFloor:
    l2: float
    divergence: float
    graph: float


def tensor_l2_floor(
    evaluate: Evaluator,
    feature_count: int,
    voigt_weight: np.ndarray,
    points: torch.Tensor,
    weights: torch.Tensor,
    tensor: torch.Tensor,
) -> float:
    """Componentwise ``L^2`` best approximation of a symmetric tensor field.

    In ``L^2`` the Voigt components decouple, so this is the same dictionary
    fitted to each component independently and the result is a genuine floor.
    It is the honest partner of the two graph-norm routines below: the ``l2``
    they report is a *component* of the graph-optimal fit, and once the
    divergence term dominates the graph norm that component is nearly
    unconstrained -- it can rise with ``N`` while the graph error falls.
    """

    component_count = tensor.shape[1]
    sqrt_weights = torch.sqrt(weights)

    def build_block(start: int, stop: int) -> np.ndarray:
        scale = sqrt_weights[start:stop].numpy()[:, None]
        basis, _, _ = evaluate(points[start:stop])
        return weighted_block(
            [basis.numpy() * scale], [tensor[start:stop].numpy() * scale]
        )

    coefficients = best_fit(
        build_block,
        points.shape[0],
        feature_count,
        component_count,
        batch_size=block_points(1, feature_count + component_count),
    )

    squared = 0.0
    for start, stop in batches(points.shape[0], POINT_BATCH):
        basis, _, _ = evaluate(points[start:stop])
        error = basis.numpy() @ coefficients - tensor[start:stop].numpy()
        squared += float(
            (weights[start:stop].numpy() * (voigt_weight * error**2).sum(1)).sum()
        )
    return math.sqrt(squared)


def stress_hdiv_floor(
    evaluate: Evaluator,
    feature_count: int,
    pairs: tuple[tuple[int, int], ...],
    voigt_weight: np.ndarray,
    points: torch.Tensor,
    weights: torch.Tensor,
    stress: torch.Tensor,
    body_force: torch.Tensor,
) -> TensorFloor:
    """Joint ``H(div)`` best approximation of the exact stress.

    The exact stress satisfies ``div sigma_* = -f``, so the divergence rows
    target ``-f`` exactly as the driver's equilibrium residual does.  Fitting
    the components jointly is essential: the ``H(div)`` optimum trades a worse
    ``L^2`` error for a better divergence, which is precisely what the
    least-squares functional does.
    """

    dimension = points.shape[1]
    component_count = len(pairs)
    column_count = component_count * feature_count
    voigt_index = {pair: index for index, pair in enumerate(pairs)}
    sqrt_weights = torch.sqrt(weights)

    def build_block(start: int, stop: int) -> np.ndarray:
        count = stop - start
        scale = sqrt_weights[start:stop].numpy()
        basis, basis_gradient, _ = evaluate(points[start:stop])
        basis_values = basis.numpy() * scale[:, None]
        gradient = basis_gradient.numpy() * scale[:, None, None]
        rows = component_count + dimension
        design = np.zeros((rows * count, column_count))
        right = np.zeros((rows * count, 1))
        for component in range(component_count):
            block = slice(component * count, (component + 1) * count)
            factor = math.sqrt(voigt_weight[component])
            design[block, component * feature_count : (component + 1) * feature_count] = (
                factor * basis_values
            )
            right[block, 0] = factor * stress[start:stop, component].numpy() * scale
        for component in range(dimension):
            block = slice(
                (component_count + component) * count,
                (component_count + component + 1) * count,
            )
            for axis in range(dimension):
                index = voigt_index[(min(component, axis), max(component, axis))]
                design[
                    block, index * feature_count : (index + 1) * feature_count
                ] += gradient[:, :, axis]
            right[block, 0] = -body_force[start:stop, component].numpy() * scale
        return np.asfortranarray(np.hstack([design, right]))

    coefficients = best_fit(
        build_block,
        points.shape[0],
        column_count,
        1,
        batch_size=block_points(component_count + dimension, column_count),
    )[:, 0]

    squared_l2 = 0.0
    squared_div = 0.0
    for start, stop in batches(points.shape[0], POINT_BATCH):
        basis, basis_gradient, _ = evaluate(points[start:stop])
        basis_values = basis.numpy()
        gradient = basis_gradient.numpy()
        block_weights = weights[start:stop].numpy()
        blocks = coefficients.reshape(component_count, feature_count)
        fitted = np.stack([basis_values @ blocks[i] for i in range(component_count)], 1)
        squared_l2 += float(
            (
                block_weights
                * (voigt_weight * (fitted - stress[start:stop].numpy()) ** 2).sum(1)
            ).sum()
        )
        divergence = np.zeros((stop - start, dimension))
        for component in range(dimension):
            for axis in range(dimension):
                index = voigt_index[(min(component, axis), max(component, axis))]
                divergence[:, component] += gradient[:, :, axis] @ blocks[index]
        residual = divergence + body_force[start:stop].numpy()
        squared_div += float((block_weights * (residual**2).sum(1)).sum())
    return TensorFloor(
        l2=math.sqrt(squared_l2),
        divergence=math.sqrt(squared_div),
        graph=math.sqrt(squared_l2 + squared_div),
    )


def moment_hdivdiv_floor(
    evaluate: Evaluator,
    feature_count: int,
    points: torch.Tensor,
    weights: torch.Tensor,
    moment: torch.Tensor,
    load: torch.Tensor,
) -> TensorFloor:
    """Joint ``H(div div)`` best approximation of the exact bending moment.

    ``div div M = d11 M11 + d22 M22 + 2 d12 M12`` in the driver's Voigt order,
    and the exact moment satisfies ``div div M_* + f = 0``.
    """

    voigt_weight = np.array([1.0, 1.0, 2.0])
    # Hessian components come back as (0,0), (0,1), (1,1); the moment Voigt
    # order is (11, 22, 12), so reorder once here.
    hessian_order = (0, 2, 1)
    divdiv_factor = np.array([1.0, 1.0, 2.0])
    column_count = 3 * feature_count
    sqrt_weights = torch.sqrt(weights)

    def build_block(start: int, stop: int) -> np.ndarray:
        count = stop - start
        scale = sqrt_weights[start:stop].numpy()
        basis, _, basis_hessian = evaluate(points[start:stop])
        basis_values = basis.numpy() * scale[:, None]
        hessian = basis_hessian.numpy()[:, :, hessian_order] * scale[:, None, None]
        design = np.zeros((4 * count, column_count))
        right = np.zeros((4 * count, 1))
        for component in range(3):
            block = slice(component * count, (component + 1) * count)
            factor = math.sqrt(voigt_weight[component])
            design[block, component * feature_count : (component + 1) * feature_count] = (
                factor * basis_values
            )
            right[block, 0] = factor * moment[start:stop, component].numpy() * scale
        block = slice(3 * count, 4 * count)
        for component in range(3):
            design[
                block, component * feature_count : (component + 1) * feature_count
            ] += divdiv_factor[component] * hessian[:, :, component]
        right[block, 0] = -load[start:stop].numpy() * scale
        return np.asfortranarray(np.hstack([design, right]))

    coefficients = best_fit(
        build_block,
        points.shape[0],
        column_count,
        1,
        batch_size=block_points(4, column_count),
    )[:, 0]

    squared_l2 = 0.0
    squared_div = 0.0
    blocks = coefficients.reshape(3, feature_count)
    for start, stop in batches(points.shape[0], POINT_BATCH):
        basis, _, basis_hessian = evaluate(points[start:stop])
        basis_values = basis.numpy()
        hessian = basis_hessian.numpy()[:, :, hessian_order]
        block_weights = weights[start:stop].numpy()
        fitted = np.stack([basis_values @ blocks[i] for i in range(3)], 1)
        squared_l2 += float(
            (
                block_weights
                * (voigt_weight * (fitted - moment[start:stop].numpy()) ** 2).sum(1)
            ).sum()
        )
        divdiv = sum(
            divdiv_factor[i] * (hessian[:, :, i] @ blocks[i]) for i in range(3)
        )
        squared_div += float(
            (block_weights * (divdiv + load[start:stop].numpy()) ** 2).sum()
        )
    return TensorFloor(
        l2=math.sqrt(squared_l2),
        divergence=math.sqrt(squared_div),
        graph=math.sqrt(squared_l2 + squared_div),
    )


# ---------------------------------------------------------------------------
# Model adapters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Benchmark:
    """Exact fields on the deterministic test rule, pulled from a driver."""

    dimension: int
    sobolev_order: int
    points: torch.Tensor
    weights: torch.Tensor
    scalar_values: torch.Tensor
    scalar_gradients: torch.Tensor
    scalar_hessians: torch.Tensor | None
    tensor_values: torch.Tensor
    load: torch.Tensor
    pairs: tuple[tuple[int, int], ...]
    voigt_weight: np.ndarray
    # Resolved manufactured-solution name, so a floor row records the field it
    # was measured against rather than whichever override the caller passed.
    solution: str


def load_benchmark(
    model_key: str,
    width: int,
    q_test: int | None = None,
    solution: str | None = None,
) -> Benchmark:
    """Build one driver's exact test fields without running any solver."""

    spec = MODEL_SPECS[model_key]
    module = load_model(spec)
    cfg = module.default_config()
    setattr(cfg, spec.width_fields[0], width)
    setattr(cfg, spec.width_fields[1], width)
    if q_test is not None:
        cfg.Q_test = q_test
    if solution is not None:
        cfg.manufactured_solution = solution

    if model_key == "plate":
        benchmark = module.build_shared_benchmark(
            E=cfg.E,
            nu=cfg.nu,
            h=cfg.h,
            Q_train=64,
            Q_test=cfg.Q_test,
            sampling_method=cfg.sampling_method,
            manufactured_solution=cfg.manufactured_solution,
        )
        # The plate driver stores symmetric 2-tensors as (11, 22, 12); the
        # generic evaluators here emit Hessians as (0,0), (0,1), (1,1).
        # Reorder the exact Hessian once so the scalar floor compares like
        # with like.  ``M_exact_test`` keeps the driver order because
        # ``moment_hdivdiv_floor`` permutes the basis instead.
        return Benchmark(
            dimension=2,
            sobolev_order=2,
            points=benchmark.x_test,
            weights=benchmark.w_test,
            scalar_values=benchmark.u_exact_test.unsqueeze(1),
            scalar_gradients=benchmark.u_grad_exact_test.unsqueeze(1),
            scalar_hessians=benchmark.u_hess_exact_test[:, [0, 2, 1]].unsqueeze(1),
            tensor_values=benchmark.M_exact_test,
            load=benchmark.f_test,
            pairs=((0, 0), (1, 1), (0, 1)),
            voigt_weight=np.array([1.0, 1.0, 2.0]),
            solution=str(cfg.manufactured_solution),
        )

    problem = module.PROBLEM
    benchmark = module.build_shared_benchmark(
        E=cfg.E,
        nu=cfg.nu,
        Q_train=64,
        Q_test=cfg.Q_test,
        sampling_method=cfg.sampling_method,
        body_force_batch_size=cfg.body_force_batch_size,
        manufactured_solution=cfg.manufactured_solution,
    )
    return Benchmark(
        dimension=problem.spec.dimension,
        sobolev_order=1,
        points=benchmark.x_test,
        weights=benchmark.w_test,
        scalar_values=benchmark.u_exact_test,
        # The driver stores grad[q, component, axis], which is the layout the
        # scalar floor expects.
        scalar_gradients=benchmark.u_grad_exact_test,
        scalar_hessians=None,
        tensor_values=benchmark.sigma_exact_test,
        load=benchmark.f_test,
        pairs=problem.spec.pairs,
        voigt_weight=problem.spec.voigt_weight.numpy(),
        solution=str(cfg.manufactured_solution or problem.default_solution),
    )


def ritz_projection(parameters: torch.Tensor, sobolev_order: int, power: int,
                    degree: int, ritz_ratio: float, width: int):
    """Build the driver's Ritz-projected trial space for one configuration."""

    return build_ritz_projected_features(
        parameters,
        sobolev_order=sobolev_order,
        auxiliary_dimension=max(width + 1, int(math.ceil(ritz_ratio * (width + 1)))),
        power=power,
        degree=degree,
    )


def fitted_order(
    widths: list[float],
    errors: list[float],
) -> float:
    """Least-squares slope of ``log(error)`` against ``log(N)``.

    Every finite positive error enters the fit.
    """

    x = np.asarray(widths, dtype=float)
    y = np.asarray(errors, dtype=float)
    mask = (x > 0) & (y > 0) & np.isfinite(y)
    return float(-np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0])


def field_norms(benchmark: Benchmark) -> tuple[float, float]:
    """Graph-norm sizes of the exact scalar and tensor fields.

    These are recorded alongside the absolute approximation errors as reference
    scales for interpreting their magnitude.
    """

    weights = benchmark.weights
    squared = float((weights * benchmark.scalar_values.square().sum(1)).sum())
    squared += float((weights * benchmark.scalar_gradients.square().sum((1, 2))).sum())
    if benchmark.scalar_hessians is not None:
        squared += float(
            (weights * benchmark.scalar_hessians.square().sum((1, 2))).sum()
        )
    tensor_squared = float(
        (
            weights
            * (
                torch.from_numpy(benchmark.voigt_weight)
                * benchmark.tensor_values.square()
            ).sum(1)
        ).sum()
    )
    tensor_squared += float((weights * benchmark.load.square().sum(-1)).sum())
    return math.sqrt(squared), math.sqrt(tensor_squared)


__all__ = [
    "Benchmark",
    "ScalarFloor",
    "TensorFloor",
    "accumulate_qr",
    "approximation_rate",
    "batches",
    "best_fit",
    "block_points",
    "field_norms",
    "fitted_order",
    "load_benchmark",
    "moment_hdivdiv_floor",
    "projected_dictionary",
    "quasi_uniform_features",
    "raw_dictionary",
    "ritz_projection",
    "saturation_index",
    "scalar_floor",
    "spline_space",
    "stress_hdiv_floor",
    "TensorSplineSpace",
]
