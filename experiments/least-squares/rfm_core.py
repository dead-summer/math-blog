"""Shared numerical primitives for the least-squares linearized-network experiments.

The module deliberately keeps the four model drivers free of activation-,
projection-, and coefficient-solver details.  Features are ReLU powers
``rho_k(t) = max(t, 0)^k`` whose hidden parameters form a deterministic
quasi-uniform tensor point set on the direction--bias domain
``S^{d-1} x [-c, c]``; no feature randomness remains.  Essential boundary
conditions are imposed by an empirical Sobolev--Ritz projection into
tensor-product B-spline spaces.  The output coefficients are constrained in
their physical, unscaled Euclidean norm.

The activation power ``k`` is the dictionary's single most important knob: the
spherical Legendre coefficients of ``rho_k`` decay at rate
``-(d + 2k + 1)/2``, so the dictionary saturates at Sobolev index
``s_cap(d) = (d + 2k + 1)/2`` and approximates ``H^m`` at rate
``N^{-(s_cap(d) - m)/d}``.  Every extra derivative in the error norm costs one
factor ``N^{1/d}``, which is why graph-norm errors sit orders of magnitude above
``L^2`` errors at fixed ``N``.
"""

from __future__ import annotations

import functools
import itertools
import math
import os
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from scipy.interpolate import BSpline
import torch


ArrayTuple = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
ValueGradientTuple = tuple[torch.Tensor, torch.Tensor]

GOLDEN_RATIO_CONJUGATE = (math.sqrt(5.0) - 1.0) / 2.0
DEFAULT_BIAS_RANGE = 2.0
DEFAULT_ACTIVATION_POWER = 3


def validate_activation_power(power: int, sobolev_order: int = 1) -> int:
    """Reject activation powers outside the range the analysis covers.

    ``rho_k`` must be smooth enough for the graph norm in use: the training
    generalization argument needs ``rho_k in W^(m+1,oo)``, and the paper states
    its results for ``k >= 3``.
    """

    if not isinstance(power, (int, np.integer)) or isinstance(power, bool):
        raise ValueError("activation power must be an integer")
    minimum = max(3, sobolev_order + 1)
    if power < minimum:
        raise ValueError(
            f"activation power must be at least {minimum} for Sobolev order "
            f"{sobolev_order}; got {power}"
        )
    return int(power)


def saturation_index(spatial_dimension: int, power: int) -> float:
    """Dictionary saturation index ``s_cap(d) = (d + 2k + 1)/2``."""

    return 0.5 * (spatial_dimension + 2 * power + 1)


def approximation_rate(spatial_dimension: int, power: int, sobolev_order: int) -> float:
    """Best-approximation rate exponent ``beta = (s_cap(d) - m)/d``.

    The error of the quasi-uniform ``rho_power`` dictionary in ``H^m`` behaves
    like ``N^{-beta}`` once the target is at least ``s_cap(d)``-smooth.
    """

    return (
        saturation_index(spatial_dimension, power) - sobolev_order
    ) / spatial_dimension


def matched_spline_degree(spatial_dimension: int, power: int) -> int:
    """Smallest spline degree whose saturation ``p+1`` covers ``s_cap(d)``.

    Using a lower degree makes the auxiliary Ritz term dominate the total
    error asymptotically, so the observed convergence order falls below
    :func:`approximation_rate`.
    """

    return max(3, math.ceil(saturation_index(spatial_dimension, power)) - 1)


def _circle_directions(count: int, rotation: float) -> np.ndarray:
    """Equispaced directions on ``S^1`` rotated by ``rotation`` turns."""

    angles = 2.0 * math.pi * ((np.arange(count) + 0.5) / count + rotation)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _fibonacci_sphere_directions(count: int, rotation: float) -> np.ndarray:
    """Fibonacci-lattice directions on ``S^2`` rotated by ``rotation`` turns."""

    indices = np.arange(count)
    heights = 1.0 - (2.0 * indices + 1.0) / count
    radii = np.sqrt(np.clip(1.0 - heights * heights, 0.0, None))
    azimuths = 2.0 * math.pi * (indices * GOLDEN_RATIO_CONJUGATE + rotation)
    return np.stack(
        [radii * np.cos(azimuths), radii * np.sin(azimuths), heights],
        axis=1,
    )


def _total_degree_indices(spatial_dimension: int, degree: int) -> list[tuple[int, ...]]:
    """Return monomial multi-indices ordered by total degree."""

    indices: list[tuple[int, ...]] = []

    def append_compositions(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            indices.append(tuple([*prefix, remaining]))
            return
        for value in range(remaining + 1):
            append_compositions([*prefix, value], remaining - value, slots - 1)

    for total_degree in range(degree + 1):
        append_compositions([], total_degree, spatial_dimension)
    return indices


def _affine_power_coefficients(
    direction: np.ndarray,
    bias: float,
    power: int,
    multi_indices: list[tuple[int, ...]],
) -> np.ndarray:
    """Coefficients of ``(direction @ x + bias)^power`` in a monomial basis."""

    coefficients = []
    for alpha in multi_indices:
        alpha_degree = sum(alpha)
        multinomial = math.factorial(power) / (
            math.factorial(power - alpha_degree)
            * math.prod(math.factorial(component) for component in alpha)
        )
        direction_factor = math.prod(
            direction[axis] ** exponent for axis, exponent in enumerate(alpha)
        )
        coefficients.append(
            multinomial * bias ** (power - alpha_degree) * direction_factor
        )
    return np.asarray(coefficients)


def _monomial_box_gram_factor(multi_indices: list[tuple[int, ...]]) -> np.ndarray:
    """Cholesky factor of the monomial Gram matrix on the unit box.

    ``int_{[0,1]^d} x^alpha x^beta dx = prod_i 1 / (alpha_i + beta_i + 1)``, so
    the factor turns monomial coefficient vectors into an isometric coordinate
    system for ``L^2(Omega)``.  Selecting the supplement in that metric keeps
    the retained ridge powers well separated as functions, which the raw
    monomial coordinates fail to do once ``power`` is large.
    """

    size = len(multi_indices)
    gram = np.empty((size, size), dtype=np.float64)
    for row, alpha in enumerate(multi_indices):
        for column, beta in enumerate(multi_indices):
            gram[row, column] = math.prod(
                1.0 / (a + b + 1.0) for a, b in zip(alpha, beta)
            )
    return scipy.linalg.cholesky(gram, lower=False)


def _quadrature_box_isometry(
    spatial_dimension: int,
    power: int,
    parameters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``L^2(Omega)``-isometric coordinates of ridge powers, without monomials.

    A tensor Gauss--Legendre rule with ``power + 1`` nodes per axis integrates
    every product of two degree-``power`` polynomials on the box exactly, so
    the weighted evaluation matrix ``sqrt(w_q) (omega_j . x_q + b_j)^power``
    has exactly the Gram matrix that :func:`_monomial_box_gram_factor` builds
    an isometry for.  Its columns therefore rank the candidates identically
    under pivoted QR, which is invariant under the orthogonal change of
    coordinates relating the two representations.

    This route exists because the monomial one stops working in double
    precision: the multinomial expansion of ``(omega . x + b)^power``
    alternates in sign, and the tensor-product Hilbert Gram matrix loses
    numerical positive definiteness, both once the total degree passes about
    twelve.  Evaluating the affine form directly avoids both.

    Returns the weighted evaluation matrix and the isometric coordinates of
    the constant function, which are the square-rooted quadrature weights.
    """

    nodes, weights = np.polynomial.legendre.leggauss(power + 1)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    grids = np.meshgrid(*([nodes] * spatial_dimension), indexing="ij")
    weight_grids = np.meshgrid(*([weights] * spatial_dimension), indexing="ij")
    points = np.stack([grid.reshape(-1) for grid in grids], axis=1)
    root_weights = np.sqrt(
        np.prod(
            np.stack([grid.reshape(-1) for grid in weight_grids], axis=1),
            axis=1,
        )
    )
    affine = (
        points @ parameters[:, :spatial_dimension].T
        + parameters[:, spatial_dimension]
    )
    return root_weights[:, None] * affine**power, root_weights


@functools.lru_cache(maxsize=16)
def _polynomial_supplement_parameters_cached(
    spatial_dimension: int,
    power: int,
    bias_range: float,
) -> np.ndarray:
    """Select globally positive ridge powers spanning ``P_power / constants``.

    ``power``-th powers of affine forms span all polynomials of total degree at
    most ``power``.  We choose a small, deterministic, well-conditioned subset
    by pivoted QR in the ``L^2(Omega)`` metric.  The separate constant feature
    supplies the one omitted polynomial degree of freedom.
    """

    polynomial_dimension = math.comb(spatial_dimension + power, power)
    supplement_count = polynomial_dimension - 1
    # The candidate pool only needs to be a small multiple of the number of
    # retained polynomial directions.  The old ``8 * dimension`` pool with
    # ``power + 2`` bias layers made the pivoted QR needlessly expensive in
    # three dimensions (for ``d=3,k=9`` it factored a 220 x 19,360 matrix for
    # every fresh Python process).  Four direction copies and five bias layers
    # still leave an order-of-magnitude oversampling margin while keeping the
    # deterministic pivot selection well conditioned.
    direction_count = max(
        64 if spatial_dimension == 2 else 128,
        4 * polynomial_dimension,
    )
    direction_builder = (
        _circle_directions
        if spatial_dimension == 2
        else _fibonacci_sphere_directions
    )
    directions = direction_builder(direction_count, 0.37)

    # Every candidate is strictly positive on [0,1]^d, hence its ReLU power
    # is an ordinary affine power there.  Varying both direction and bias is
    # essential: a single fixed-bias sphere does not span all of P_power.
    #
    # The lower end of each direction's admissible bias interval is the exact
    # per-direction threshold b_+(omega) = -sum_i min(omega_i, 0) rather than
    # the direction-independent sqrt(d).  Biases just above b_+(omega) put the
    # hyperplane against a corner of the box and so maximize the variation of
    # (omega . x + b)^power over it; the resulting supplement is far better
    # conditioned as a function basis (500x at d=2, 6600x at d=3, for
    # power = 7).  A small offset keeps every candidate strictly degenerate.
    corner_thresholds = -np.minimum(directions, 0.0).sum(axis=1)
    bias_fractions = np.linspace(0.02, 0.9, 5)
    multi_indices = _total_degree_indices(spatial_dimension, degree=power)
    candidate_parameters: list[np.ndarray] = []
    candidate_coefficients: list[np.ndarray] = []
    for direction, threshold in zip(directions, corner_thresholds):
        for fraction in bias_fractions:
            bias = threshold + fraction * (bias_range - threshold)
            candidate_parameters.append(np.concatenate([direction, [bias]]))
            candidate_coefficients.append(
                _affine_power_coefficients(
                    direction,
                    float(bias),
                    power,
                    multi_indices,
                )
            )

    coefficient_matrix = np.stack(candidate_coefficients, axis=1)
    candidate_matrix = np.stack(candidate_parameters, axis=0)
    # Move to L^2-isometric coordinates, drop the component along the constant
    # feature, and normalize: pivoted QR then ranks candidates by how much new
    # L^2 direction each one adds, independently of the monomial scaling.
    #
    # The monomial route is the reference one and stays in force wherever it
    # is numerically sound.  Its Gram matrix is a tensor-product Hilbert
    # matrix, which stops being positive definite in double precision at
    # ``power`` around twelve; there the quadrature route supplies the same
    # isometry without ever expanding the ridge power in monomials.
    try:
        gram_factor = _monomial_box_gram_factor(multi_indices)
    except np.linalg.LinAlgError:
        gram_factor = None
    if gram_factor is not None:
        isometric = gram_factor @ coefficient_matrix
        # The constant function has monomial coefficients e_0, so its
        # isometric coordinate is the first column of the factor.
        constant_column = gram_factor[:, 0]
        rank_columns = coefficient_matrix
        constant_rank_column = np.eye(polynomial_dimension)[:, 0]
    else:
        isometric, constant_column = _quadrature_box_isometry(
            spatial_dimension,
            power,
            candidate_matrix,
        )
        rank_columns = isometric
        constant_rank_column = constant_column
    constant_direction = constant_column / np.linalg.norm(constant_column)
    deflated = isometric - np.outer(constant_direction, constant_direction @ isometric)
    deflated /= np.maximum(np.linalg.norm(deflated, axis=0), np.finfo(float).tiny)
    _, _, pivots = scipy.linalg.qr(
        deflated,
        mode="economic",
        pivoting=True,
        check_finite=False,
    )
    selected = candidate_matrix[pivots[:supplement_count]]
    augmented = np.column_stack(
        [
            constant_rank_column,
            rank_columns[:, pivots[:supplement_count]],
        ]
    )
    if np.linalg.matrix_rank(augmented) != polynomial_dimension:
        raise RuntimeError(
            f"Failed to construct a complete degree-{power} polynomial supplement."
        )
    selected.setflags(write=False)
    return selected


def polynomial_supplement_parameters(
    spatial_dimension: int,
    *,
    power: int = 3,
    bias_range: float = DEFAULT_BIAS_RANGE,
) -> np.ndarray:
    """Return ridge parameters whose restrictions supplement ``P_power``."""

    if spatial_dimension not in (2, 3):
        raise ValueError("polynomial supplements are implemented for d in {2, 3}")
    validate_activation_power(power)
    if bias_range <= math.sqrt(float(spatial_dimension)):
        raise ValueError("bias_range must exceed sqrt(d) for the polynomial supplement")
    return _polynomial_supplement_parameters_cached(
        spatial_dimension,
        int(power),
        float(bias_range),
    ).copy()


def quasi_uniform_features(
    width: int,
    spatial_dimension: int,
    *,
    power: int = DEFAULT_ACTIVATION_POWER,
    bias_range: float = DEFAULT_BIAS_RANGE,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build ``width`` useful quasi-uniform parameters ``(omega, b)``.

    Non-polynomial rows tile the active parameter domain: for each direction,
    the hyperplane ``omega @ x + b = 0`` intersects the unit box.  Biases are
    midpoint layers in the relative bias, and each layer has a
    quasi-uniform direction set (equispaced angles in 2D, a Fibonacci lattice
    in 3D) with deterministic rotations.  Parameters outside this domain give
    either the zero function or an ordinary degree-``power`` polynomial on the
    box.  Instead of wasting a positive fraction of ``width`` on those
    degenerate rows, a fixed QR-selected set of globally positive ridge powers
    supplies ``P_power`` exactly.  Together with the evaluator's constant
    column, this is the finite polynomial supplement used in the approximation
    theorem.  Its size ``binom(d + power, power)`` grows like ``power^d`` but
    not with ``width``; at small ``width`` and large ``power`` it can consume a
    noticeable share of the budget, which is why the drivers report both counts.
    """

    if width < 0:
        raise ValueError("width must be nonnegative")
    if spatial_dimension not in (2, 3):
        raise ValueError("quasi-uniform parameters are implemented for d in {2, 3}")
    validate_activation_power(power)
    if not (
        math.isfinite(bias_range)
        and bias_range > math.sqrt(float(spatial_dimension))
    ):
        raise ValueError("bias_range must be finite and exceed sqrt(d)")
    if width == 0:
        return torch.empty(0, spatial_dimension + 1, dtype=dtype, device=device)

    supplement = polynomial_supplement_parameters(
        spatial_dimension,
        power=power,
        bias_range=bias_range,
    )
    supplement_count = min(width, supplement.shape[0])
    active_count = width - supplement_count
    if active_count == 0:
        return torch.from_numpy(supplement[:supplement_count].copy()).to(
            dtype=dtype,
            device=device,
        )

    # Retain the original asymptotic layer balance.  Replacing the physical
    # bias by its position relative to that interval only changes fixed
    # constants.
    if spatial_dimension == 2:
        layer_estimate = math.sqrt(bias_range * active_count / math.pi)
    else:
        layer_estimate = (
            bias_range * bias_range * active_count / math.pi
        ) ** (1.0 / 3.0)
    layer_count = min(active_count, max(1, round(layer_estimate)))

    base_count, remainder = divmod(active_count, layer_count)
    direction_builder = (
        _circle_directions if spatial_dimension == 2 else _fibonacci_sphere_directions
    )
    blocks: list[np.ndarray] = []
    for layer in range(layer_count):
        count = base_count + (1 if layer < remainder else 0)
        directions = direction_builder(count, math.modf(layer * GOLDEN_RATIO_CONJUGATE)[0])
        minimum = np.minimum(directions, 0.0).sum(axis=1)
        maximum = np.maximum(directions, 0.0).sum(axis=1)
        relative_bias = (layer + 0.5) / layer_count
        # omega @ x ranges over [minimum, maximum] on the unit box, so these
        # midpoint biases make every retained hyperplane cross its interior.
        biases = -maximum + relative_bias * (maximum - minimum)
        blocks.append(
            np.concatenate([directions, biases[:, None]], axis=1)
        )
    parameters = np.concatenate(
        [supplement[:supplement_count], *blocks],
        axis=0,
    )
    return torch.from_numpy(parameters).to(dtype=dtype, device=device)


@dataclass(frozen=True)
class ParameterSetDiagnostics:
    """Covering/separation of active parameters and degeneracy counts."""

    covering_radius: float
    separation: float
    active_count: int
    polynomial_count: int
    zero_count: int

    @property
    def mesh_ratio(self) -> float:
        if self.separation <= 0.0:
            return float("inf")
        return self.covering_radius / self.separation


def parameter_set_diagnostics(
    parameters: torch.Tensor,
    *,
    bias_range: float = DEFAULT_BIAS_RANGE,
    probe_count: int = 8192,
    probe_seed: int = 0,
) -> ParameterSetDiagnostics:
    """Estimate covering and separation on the active parameter manifold.

    Rows ``(omega, b)`` are mapped to ``S^d`` by the normalization
    ``(omega, b) / sqrt(1 + b^2)``.  The covering radius is measured against
    fixed-seed probes whose hyperplanes intersect the unit box.  The separation
    is the exact minimum pairwise distance among the corresponding active
    dictionary rows; the fixed polynomial supplement is counted separately.
    """

    if parameters.ndim != 2 or parameters.shape[0] < 1:
        raise ValueError("parameters must be a nonempty matrix")
    if probe_count < 1:
        raise ValueError("probe_count must be positive")
    spatial_dimension = parameters.shape[1] - 1
    directions = parameters[:, :spatial_dimension]
    biases = parameters[:, spatial_dimension]
    lower = biases + torch.minimum(directions, torch.zeros_like(directions)).sum(dim=1)
    upper = biases + torch.maximum(directions, torch.zeros_like(directions)).sum(dim=1)
    tolerance = 64.0 * torch.finfo(parameters.dtype).eps
    active_mask = (lower < -tolerance) & (upper > tolerance)
    polynomial_mask = lower >= -tolerance
    zero_mask = upper <= tolerance
    active_parameters = parameters[active_mask]
    normalized = active_parameters / active_parameters.norm(
        dim=1,
        keepdim=True,
    ).clamp_min(1.0e-15)

    if active_parameters.shape[0] < 2:
        separation = float("inf")
    else:
        gram = (normalized @ normalized.T).clamp(-1.0, 1.0)
        gram.fill_diagonal_(-1.0)
        separation = float(torch.arccos(gram.max()))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(probe_seed)
    raw_directions = torch.randn(
        probe_count,
        spatial_dimension,
        generator=generator,
        dtype=normalized.dtype,
    )
    raw_directions /= raw_directions.norm(dim=1, keepdim=True).clamp_min(1.0e-15)
    minimum = torch.minimum(raw_directions, torch.zeros_like(raw_directions)).sum(
        dim=1,
        keepdim=True,
    )
    maximum = torch.maximum(raw_directions, torch.zeros_like(raw_directions)).sum(
        dim=1,
        keepdim=True,
    )
    relative_bias = torch.rand(
        probe_count,
        1,
        generator=generator,
        dtype=normalized.dtype,
    )
    biases = -maximum + relative_bias * (maximum - minimum)
    probes = torch.cat([raw_directions, biases], dim=1)
    probes /= probes.norm(dim=1, keepdim=True)
    if normalized.shape[0] == 0:
        covering = float("inf")
        return ParameterSetDiagnostics(
            covering_radius=covering,
            separation=separation,
            active_count=0,
            polynomial_count=int(polynomial_mask.sum()),
            zero_count=int(zero_mask.sum()),
        )
    cosines = (probes @ normalized.T.to(device="cpu")).clamp(-1.0, 1.0)
    covering = float(torch.arccos(cosines.max(dim=1).values).max())
    return ParameterSetDiagnostics(
        covering_radius=covering,
        separation=separation,
        active_count=int(active_mask.sum()),
        polynomial_count=int(polynomial_mask.sum()),
        zero_count=int(zero_mask.sum()),
    )


def _relu_positive_parts(
    points: torch.Tensor,
    parameters: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate the feature inputs and return directions and positive parts."""

    if points.ndim != 2 or parameters.ndim != 2:
        raise ValueError("points and parameters must be matrices")
    spatial_dimension = points.shape[1]
    if parameters.shape[1] != spatial_dimension + 1:
        raise ValueError("parameter dimension does not match the points")

    directions = parameters[:, :spatial_dimension]
    biases = parameters[:, spatial_dimension]
    preactivation = points @ directions.T + biases.unsqueeze(0)
    return directions, torch.relu(preactivation)


def _augmented_feature_tensor(
    points: torch.Tensor,
    feature_count: int,
    trailing_shape: tuple[int, ...],
    leading_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate a feature tensor and return it beside its random-feature view.

    Column zero holds the deterministic constant feature, whose value is one
    and whose every derivative vanishes; ``leading_value`` selects which.  The
    random columns are returned as a strided view so callers can write them
    with ``out=``/``copy_`` instead of concatenating a separate block, which
    would double the peak footprint of the largest tensor in an evaluation.
    """

    tensor = torch.empty(
        points.shape[0],
        feature_count + 1,
        *trailing_shape,
        dtype=points.dtype,
        device=points.device,
    )
    tensor[:, 0] = leading_value
    return tensor, tensor[:, 1:]


def relu_power_feature_values(
    points: torch.Tensor,
    parameters: torch.Tensor,
    power: int = DEFAULT_ACTIVATION_POWER,
) -> torch.Tensor:
    """Evaluate only ``rho_power`` feature values.

    This is the memory-light path for projections and diagnostics that do not
    use derivatives.  As in :func:`relu_power_feature_data`, the leading column
    is the deterministic constant feature ``1``.
    """

    _, positive = _relu_positive_parts(points, parameters)
    values, random_values = _augmented_feature_tensor(
        points, parameters.shape[0], (), 1.0
    )
    random_values.copy_(positive.pow_(power))
    return values


def relu_power_feature_box_means(
    parameters: torch.Tensor,
    power: int = DEFAULT_ACTIVATION_POWER,
) -> torch.Tensor:
    """Integrate all ``rho_power`` features exactly over the unit box.

    For a crossing hyperplane, reflect coordinates with negative direction
    components and apply the inclusion--exclusion antiderivative formula: each
    of the ``active_dimension`` integrations raises the exponent by one and
    divides by the new exponent, giving the vertex sum of
    ``relu(.)^(power + active_dimension)`` over
    ``prod(magnitudes) * (power+1)...(power+active_dimension)``.  A component at
    floating-point zero is removed before division.  Globally positive affine
    powers are degree-``power`` polynomials, so a tensor Gauss--Legendre rule
    with ``ceil((power+1)/2)`` nodes per axis integrates them exactly and
    without the cancellation the inclusion--exclusion form would suffer there;
    globally negative features integrate to zero.  The leading returned entry is
    the exact mean of the constant feature.
    """

    if parameters.ndim != 2:
        raise ValueError("parameters must be a matrix")
    if not parameters.dtype.is_floating_point:
        raise ValueError("parameters must use a floating-point dtype")
    validate_activation_power(power)
    spatial_dimension = parameters.shape[1] - 1
    if spatial_dimension not in (2, 3):
        raise ValueError("exact ReLU-power box means are implemented for d in {2, 3}")

    means = torch.zeros(
        parameters.shape[0] + 1,
        dtype=parameters.dtype,
        device=parameters.device,
    )
    means[0] = 1.0
    if parameters.shape[0] == 0:
        return means

    tolerance = 64.0 * torch.finfo(parameters.dtype).eps
    cleaned_directions = parameters[:, :spatial_dimension].clone()
    cleaned_directions[
        cleaned_directions.abs() <= tolerance
    ] = 0.0
    biases = parameters[:, spatial_dimension]
    lower = biases + torch.minimum(
        cleaned_directions,
        torch.zeros_like(cleaned_directions),
    ).sum(dim=1)
    upper = biases + torch.maximum(
        cleaned_directions,
        torch.zeros_like(cleaned_directions),
    ).sum(dim=1)

    positive_mask = lower >= 0.0
    if bool(positive_mask.any()):
        nodes, node_weights = np.polynomial.legendre.leggauss((power + 2) // 2)
        axis_nodes = torch.from_numpy(0.5 * (nodes + 1.0)).to(
            dtype=parameters.dtype,
            device=parameters.device,
        )
        axis_weights = torch.from_numpy(0.5 * node_weights).to(
            dtype=parameters.dtype,
            device=parameters.device,
        )
        grids = torch.meshgrid(*([axis_nodes] * spatial_dimension), indexing="ij")
        quadrature_points = torch.stack([grid.reshape(-1) for grid in grids], dim=1)
        weight_grids = torch.meshgrid(
            *([axis_weights] * spatial_dimension),
            indexing="ij",
        )
        quadrature_weights = torch.stack(
            [grid.reshape(-1) for grid in weight_grids],
            dim=1,
        ).prod(dim=1)
        positive_parameters = parameters[positive_mask]
        preactivation = (
            quadrature_points @ positive_parameters[:, :spatial_dimension].T
            + positive_parameters[:, spatial_dimension].unsqueeze(0)
        )
        means[1:][positive_mask] = (
            quadrature_weights.unsqueeze(1) * preactivation.pow(power)
        ).sum(dim=0)

    crossing_indices = torch.nonzero(
        (lower < 0.0) & (upper > 0.0),
        as_tuple=False,
    ).flatten()
    for feature_index in crossing_indices.tolist():
        direction = cleaned_directions[feature_index]
        nonzero = direction != 0.0
        magnitudes = direction[nonzero].abs()
        oriented_bias = biases[feature_index] + direction[
            direction < 0.0
        ].sum()
        active_dimension = int(magnitudes.numel())
        vertex_power = power + active_dimension
        numerator = torch.zeros((), dtype=parameters.dtype, device=parameters.device)
        for vertex in itertools.product((0, 1), repeat=active_dimension):
            vertex_tensor = torch.tensor(
                vertex,
                dtype=parameters.dtype,
                device=parameters.device,
            )
            value = torch.relu(oriented_bias + torch.dot(magnitudes, vertex_tensor))
            sign = -1.0 if (active_dimension - sum(vertex)) % 2 else 1.0
            numerator = numerator + sign * value.pow(vertex_power)
        denominator = magnitudes.prod() * float(
            math.prod(range(power + 1, power + 1 + active_dimension))
        )
        integral = numerator / denominator
        # Roundoff can only create a tiny negative value in this nonnegative
        # integral.  Clamping preserves the exact formula in normal cases.
        means[feature_index + 1] = integral.clamp_min(0.0)
    return means


def relu_power_feature_values_and_gradients(
    points: torch.Tensor,
    parameters: torch.Tensor,
    power: int = DEFAULT_ACTIVATION_POWER,
) -> ValueGradientTuple:
    """Evaluate ``rho_power`` values and gradients without forming Hessians."""

    directions, positive = _relu_positive_parts(points, parameters)
    feature_count = parameters.shape[0]
    values, random_values = _augmented_feature_tensor(points, feature_count, (), 1.0)
    gradients, random_gradients = _augmented_feature_tensor(
        points, feature_count, (points.shape[1],), 0.0
    )
    first_derivative = positive.pow(power - 1).mul_(float(power))
    torch.mul(
        first_derivative.unsqueeze(2),
        directions.unsqueeze(0),
        out=random_gradients,
    )
    del first_derivative
    random_values.copy_(positive.pow_(power))
    return values, gradients


def relu_power_feature_values_and_hessians(
    points: torch.Tensor,
    parameters: torch.Tensor,
    power: int = DEFAULT_ACTIVATION_POWER,
    *,
    hessian_components: Iterable[tuple[int, int]] | None = None,
) -> ValueGradientTuple:
    """Evaluate ``rho_power`` values and Hessian entries, skipping gradients.

    The fourth-order graph norm of the plate uses values and second
    derivatives but never the gradients of the dictionary, which would be
    another ``Q x (N+1) x d`` block.
    """

    directions, positive = _relu_positive_parts(points, parameters)
    spatial_dimension = points.shape[1]
    feature_count = parameters.shape[0]
    if hessian_components is None:
        components = tuple(
            (row, column)
            for row in range(spatial_dimension)
            for column in range(row, spatial_dimension)
        )
    else:
        components = tuple(hessian_components)

    values, random_values = _augmented_feature_tensor(points, feature_count, (), 1.0)
    hessians, random_hessians = _augmented_feature_tensor(
        points, feature_count, (len(components),), 0.0
    )
    if components:
        second_derivative = positive.pow(power - 2).mul_(float(power * (power - 1)))
        for index, (row, column) in enumerate(components):
            entry = random_hessians[..., index]
            torch.mul(second_derivative, directions[:, row].unsqueeze(0), out=entry)
            entry.mul_(directions[:, column].unsqueeze(0))
        del second_derivative
    random_values.copy_(positive.pow_(power))
    return values, hessians


def relu_power_feature_data(
    points: torch.Tensor,
    parameters: torch.Tensor,
    power: int = DEFAULT_ACTIVATION_POWER,
    *,
    hessian_components: Iterable[tuple[int, int]] | None = None,
) -> ArrayTuple:
    """Evaluate ``rho_power`` features, gradients, and selected Hessian entries.

    The first column is the deterministic feature ``1``.  Parameters have
    shape ``(N, d + 1)`` and are interpreted as ``(omega, bias)``.  Call
    :func:`relu_power_feature_values` or
    :func:`relu_power_feature_values_and_gradients` when higher derivatives are
    not needed, so the corresponding tensors are never allocated.
    """

    directions, positive = _relu_positive_parts(points, parameters)
    spatial_dimension = points.shape[1]
    feature_count = parameters.shape[0]
    if hessian_components is None:
        components = tuple(
            (row, column)
            for row in range(spatial_dimension)
            for column in range(row, spatial_dimension)
        )
    else:
        components = tuple(hessian_components)

    values, random_values = _augmented_feature_tensor(points, feature_count, (), 1.0)
    gradients, random_gradients = _augmented_feature_tensor(
        points, feature_count, (spatial_dimension,), 0.0
    )
    hessians, random_hessians = _augmented_feature_tensor(
        points, feature_count, (len(components),), 0.0
    )

    # Descending derivative order: each power of the positive part is consumed
    # before the next one overwrites it, so only one scratch block is live at a
    # time and the final power can be taken in place.
    if components:
        second_derivative = positive.pow(power - 2).mul_(float(power * (power - 1)))
        for index, (row, column) in enumerate(components):
            entry = random_hessians[..., index]
            torch.mul(second_derivative, directions[:, row].unsqueeze(0), out=entry)
            entry.mul_(directions[:, column].unsqueeze(0))
        del second_derivative
    first_derivative = positive.pow(power - 1).mul_(float(power))
    torch.mul(
        first_derivative.unsqueeze(2),
        directions.unsqueeze(0),
        out=random_gradients,
    )
    del first_derivative
    random_values.copy_(positive.pow_(power))
    return values, gradients, hessians


def uniform_monte_carlo_rule(
    sample_count: int,
    spatial_dimension: int,
    seed: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return iid uniform samples and normalized weights on the unit box."""

    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    points = torch.rand(
        sample_count,
        spatial_dimension,
        generator=generator,
        dtype=dtype,
    ).to(device=device)
    weights = torch.full(
        (sample_count,),
        1.0 / sample_count,
        dtype=dtype,
        device=device,
    )
    return points, weights


def _open_uniform_knots(basis_count: int, degree: int) -> np.ndarray:
    if basis_count <= degree:
        raise ValueError("basis_count must exceed the spline degree")
    interior_count = basis_count - degree - 1
    if interior_count:
        interior = np.linspace(0.0, 1.0, interior_count + 2)[1:-1]
    else:
        interior = np.empty(0, dtype=np.float64)
    return np.concatenate(
        [
            np.zeros(degree + 1, dtype=np.float64),
            interior,
            np.ones(degree + 1, dtype=np.float64),
        ]
    )


def _row_tensor_product(matrices: list[np.ndarray]) -> scipy.sparse.csr_matrix:
    """Form row-wise tensor products while retaining local spline support.

    Each factor holds at most ``degree + 1`` nonzeros per row, so the product
    holds at most ``(degree + 1)^d``.  Gathering those supports into
    fixed-width arrays keeps the construction vectorized: the row-by-row
    ``itertools.product`` this replaces cost ``(degree + 1)^d`` Python-level
    iterations per row, which dominated evaluation once the degree grew past
    three.
    """

    row_count = matrices[0].shape[0]
    widths = [matrix.shape[1] for matrix in matrices]
    column_count = math.prod(widths)
    if row_count == 0:
        return scipy.sparse.csr_matrix((0, column_count))

    strides = np.cumprod([1, *widths[:0:-1]])[::-1]
    accumulated_values: np.ndarray | None = None
    accumulated_columns: np.ndarray | None = None
    for matrix, stride in zip(matrices, strides):
        mask = np.abs(matrix) > 1.0e-14
        support = max(int(mask.sum(axis=1).max()), 1)
        # A stable argsort of the negated mask lists the nonzero columns of
        # each row first, in their original order, and pads with arbitrary
        # zero columns that the value mask then neutralizes.
        order = np.argsort(~mask, axis=1, kind="stable")[:, :support]
        values = np.where(
            np.take_along_axis(mask, order, axis=1),
            np.take_along_axis(matrix, order, axis=1),
            0.0,
        )
        columns = order * stride
        if accumulated_values is None:
            accumulated_values, accumulated_columns = values, columns
            continue
        accumulated_values = (
            accumulated_values[:, :, None] * values[:, None, :]
        ).reshape(row_count, -1)
        accumulated_columns = (
            accumulated_columns[:, :, None] + columns[:, None, :]
        ).reshape(row_count, -1)

    flat_values = accumulated_values.reshape(-1)
    nonzero = flat_values != 0.0
    rows = np.repeat(np.arange(row_count), accumulated_values.shape[1])
    return scipy.sparse.csr_matrix(
        (
            flat_values[nonzero],
            (rows[nonzero], accumulated_columns.reshape(-1)[nonzero]),
        ),
        shape=(row_count, column_count),
    )


@dataclass(frozen=True)
class TensorSplineEvaluation:
    """Values and derivatives of one tensor-product spline space."""

    values: scipy.sparse.csr_matrix
    gradients: tuple[scipy.sparse.csr_matrix, ...]
    hessians: tuple[scipy.sparse.csr_matrix, ...]


@dataclass(frozen=True)
class TensorSplineSpace:
    """Boundary-adapted tensor-product B-spline space on ``[0,1]^d``.

    ``sobolev_order`` boundary layers are removed per face, which for an open
    knot vector of any degree gives exactly zero trace (``m=1``) or zero trace
    and normal derivative (``m=2``).  The space saturates at Sobolev index
    ``degree + 1``.
    """

    spatial_dimension: int
    sobolev_order: int
    degree: int
    knots: np.ndarray
    active_indices: np.ndarray

    @property
    def axis_dimension(self) -> int:
        return int(self.active_indices.size)

    @property
    def dimension(self) -> int:
        return self.axis_dimension**self.spatial_dimension

    @classmethod
    def with_minimum_dimension(
        cls,
        spatial_dimension: int,
        sobolev_order: int,
        minimum_dimension: int,
        *,
        degree: int = 3,
    ) -> "TensorSplineSpace":
        if sobolev_order not in (1, 2):
            raise ValueError("only H1 and H2 Ritz spaces are supported")
        if degree < max(3, sobolev_order + 1):
            raise ValueError(
                f"spline degree must be at least {max(3, sobolev_order + 1)} for "
                f"Sobolev order {sobolev_order}; got {degree}"
            )
        axis_active = max(
            degree + 1,
            int(math.ceil(minimum_dimension ** (1.0 / spatial_dimension))),
        )
        boundary_layers = sobolev_order
        total_basis = axis_active + 2 * boundary_layers
        knots = _open_uniform_knots(total_basis, degree)
        active = np.arange(boundary_layers, total_basis - boundary_layers)
        return cls(
            spatial_dimension=spatial_dimension,
            sobolev_order=sobolev_order,
            degree=degree,
            knots=knots,
            active_indices=active,
        )

    def _axis_data(
        self,
        coordinates: np.ndarray,
        derivative_order: int,
    ) -> tuple[np.ndarray, ...]:
        total_basis = len(self.knots) - self.degree - 1
        coefficient_identity = np.eye(total_basis)
        spline = BSpline(
            self.knots,
            coefficient_identity,
            self.degree,
            axis=0,
            extrapolate=False,
        )
        return tuple(
            np.nan_to_num(spline(coordinates, nu=order))[:, self.active_indices]
            for order in range(derivative_order + 1)
        )

    def evaluate(
        self,
        points: np.ndarray,
        *,
        derivative_order: int = 2,
    ) -> TensorSplineEvaluation:
        if points.ndim != 2 or points.shape[1] != self.spatial_dimension:
            raise ValueError("point dimension does not match the spline space")
        if derivative_order not in (0, 1, 2):
            raise ValueError("derivative_order must be 0, 1, or 2")
        axis = [
            self._axis_data(points[:, coordinate], derivative_order)
            for coordinate in range(self.spatial_dimension)
        ]
        values = _row_tensor_product([entry[0] for entry in axis])

        gradients = []
        if derivative_order >= 1:
            for derivative_axis in range(self.spatial_dimension):
                factors = [
                    axis[coordinate][1 if coordinate == derivative_axis else 0]
                    for coordinate in range(self.spatial_dimension)
                ]
                gradients.append(_row_tensor_product(factors))

        hessians = []
        if derivative_order >= 2:
            for row in range(self.spatial_dimension):
                for column in range(row, self.spatial_dimension):
                    factors = []
                    for coordinate in range(self.spatial_dimension):
                        if coordinate == row == column:
                            derivative = 2
                        elif coordinate == row or coordinate == column:
                            derivative = 1
                        else:
                            derivative = 0
                        factors.append(axis[coordinate][derivative])
                    hessians.append(_row_tensor_product(factors))
        return TensorSplineEvaluation(values, tuple(gradients), tuple(hessians))


@dataclass(frozen=True)
class RitzProjectedFeatures:
    """Coefficient representation of projected scalar random features."""

    space: TensorSplineSpace
    coefficients: np.ndarray
    gram_residual: float
    quadrature_samples: int
    activation_power: int = DEFAULT_ACTIVATION_POWER

    @property
    def feature_count(self) -> int:
        return self.coefficients.shape[1]

    @property
    def degree(self) -> int:
        return self.space.degree

    def _batched(
        self,
        points: torch.Tensor,
        derivative_order: int,
        selected_hessians: tuple[int, ...] | None,
        batch_size: int,
        outputs: list[np.ndarray],
    ) -> None:
        """Project ``points`` in row blocks, writing into ``outputs`` in place.

        A tensor-product spline row carries ``(degree + 1)^d`` nonzeros, so the
        intermediate sparse blocks grow steeply with the degree.  Blocking keeps
        peak memory proportional to ``batch_size`` rather than to the whole
        evaluation set, without changing any value.  Callers pass the
        destinations -- possibly strided views into one derivative-major array --
        so no block is ever copied a second time to be concatenated or stacked.
        """

        point_array = points.detach().cpu().numpy()
        for start in range(0, point_array.shape[0], batch_size):
            stop = min(start + batch_size, point_array.shape[0])
            data = self.space.evaluate(
                point_array[start:stop],
                derivative_order=derivative_order,
            )
            matrices = [data.values, *data.gradients]
            if selected_hessians is not None:
                matrices.extend(data.hessians[index] for index in selected_hessians)
            if len(matrices) != len(outputs):
                raise ValueError(
                    f"expected {len(outputs)} spline blocks, got {len(matrices)}"
                )
            for destination, matrix in zip(outputs, matrices):
                destination[start:stop] = np.asarray(matrix @ self.coefficients)

    def _allocate_output(self, point_count: int, *trailing: int) -> np.ndarray:
        """Allocate one derivative-major destination for :meth:`_batched`."""

        return np.empty(
            (point_count, self.feature_count, *trailing),
            dtype=self.coefficients.dtype,
        )

    def evaluate_values(
        self,
        points: torch.Tensor,
        *,
        batch_size: int = 8_192,
    ) -> torch.Tensor:
        """Evaluate projected values without forming derivative matrices."""

        values = self._allocate_output(points.shape[0])
        self._batched(points, 0, None, batch_size, [values])
        return torch.from_numpy(values).to(dtype=points.dtype, device=points.device)

    def evaluate_values_and_gradients(
        self,
        points: torch.Tensor,
        *,
        batch_size: int = 8_192,
    ) -> ValueGradientTuple:
        """Evaluate projected values and gradients without forming Hessians."""

        dimension = self.space.spatial_dimension
        values = self._allocate_output(points.shape[0])
        gradients = self._allocate_output(points.shape[0], dimension)
        self._batched(
            points,
            1,
            None,
            batch_size,
            [values, *(gradients[:, :, axis] for axis in range(dimension))],
        )
        to_tensor = lambda array: torch.from_numpy(array).to(
            dtype=points.dtype,
            device=points.device,
        )
        return to_tensor(values), to_tensor(gradients)

    def evaluate(
        self,
        points: torch.Tensor,
        *,
        hessian_components: tuple[tuple[int, int], ...] | None = None,
        batch_size: int = 8_192,
    ) -> ArrayTuple:
        dimension = self.space.spatial_dimension
        all_components = tuple(
            (row, column)
            for row in range(dimension)
            for column in range(row, dimension)
        )
        selected = all_components if hessian_components is None else hessian_components
        component_map = {component: index for index, component in enumerate(all_components)}
        order = tuple(component_map[component] for component in selected)

        values = self._allocate_output(points.shape[0])
        gradients = self._allocate_output(points.shape[0], dimension)
        hessians = self._allocate_output(points.shape[0], len(order))
        self._batched(
            points,
            2,
            order,
            batch_size,
            [
                values,
                *(gradients[:, :, axis] for axis in range(dimension)),
                *(hessians[:, :, index] for index in range(len(order))),
            ],
        )
        to_tensor = lambda array: torch.from_numpy(array).to(
            dtype=points.dtype,
            device=points.device,
        )
        return to_tensor(values), to_tensor(gradients), to_tensor(hessians)

    def boundary_residual(self, *, samples_per_face: int = 33) -> float:
        grid = np.linspace(0.0, 1.0, samples_per_face)
        residual = 0.0
        for axis in range(self.space.spatial_dimension):
            other_axes = [index for index in range(self.space.spatial_dimension) if index != axis]
            meshes = np.meshgrid(*([grid] * len(other_axes)), indexing="ij")
            tangential = np.stack([mesh.reshape(-1) for mesh in meshes], axis=1)
            for side in (0.0, 1.0):
                points = np.empty((tangential.shape[0], self.space.spatial_dimension))
                points[:, axis] = side
                for local, coordinate in enumerate(other_axes):
                    points[:, coordinate] = tangential[:, local]
                data = self.space.evaluate(points)
                residual = max(residual, float(np.max(np.abs(data.values @ self.coefficients))))
                if self.space.sobolev_order == 2:
                    residual = max(
                        residual,
                        float(np.max(np.abs(data.gradients[axis] @ self.coefficients))),
                    )
        return residual


def build_ritz_projected_features(
    parameters: torch.Tensor,
    *,
    sobolev_order: int,
    auxiliary_dimension: int,
    power: int = DEFAULT_ACTIVATION_POWER,
    degree: int = 3,
    quadrature_samples: int | None = None,
    quadrature_seed: int = 91_003,
    regularization: float = 1.0e-12,
    batch_size: int | None = None,
) -> RitzProjectedFeatures:
    """Project ``rho_power`` features into a boundary-adapted spline space.

    The Sobolev inner product is discretized with an independent Monte Carlo
    rule.  Boundary conditions remain exact because every auxiliary basis
    function has the required zero traces.  Gram and cross moments are
    accumulated by batches so memory scales with ``batch_size`` instead of the
    total quadrature count.  The Monte Carlo points are drawn once before
    batching, hence changing ``batch_size`` does not change the quadrature
    rule.

    ``degree`` sets the spline saturation exponent ``degree + 1``.  Keep it at
    or above the dictionary's saturation index ``s_cap(d) = (d + 2*power+1)/2``,
    otherwise this auxiliary space -- not the dictionary -- limits the achievable
    graph error.
    """

    validate_activation_power(power, sobolev_order)
    spatial_dimension = parameters.shape[1] - 1
    space = TensorSplineSpace.with_minimum_dimension(
        spatial_dimension,
        sobolev_order,
        auxiliary_dimension,
        degree=degree,
    )
    sample_count = quadrature_samples or max(4 * space.dimension, 2_048)
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive")
    effective_batch_size = min(sample_count, batch_size or 2_048)
    points, weights = uniform_monte_carlo_rule(
        sample_count,
        spatial_dimension,
        quadrature_seed,
        dtype=parameters.dtype,
        device="cpu",
    )
    cpu_parameters = parameters.cpu()
    gram = scipy.sparse.csr_matrix(
        (space.dimension, space.dimension),
        dtype=np.float64,
    )
    cross = np.zeros(
        (space.dimension, parameters.shape[0] + 1),
        dtype=np.float64,
    )

    def accumulate_moment(
        spline_matrix: scipy.sparse.csr_matrix,
        raw_matrix: np.ndarray,
        sqrt_weights: np.ndarray,
        multiplier: float = 1.0,
    ) -> None:
        nonlocal gram, cross
        scaled_weights = multiplier * sqrt_weights
        weighted_spline = spline_matrix.multiply(scaled_weights[:, None])
        weighted_raw = scaled_weights[:, None] * raw_matrix
        gram = gram + weighted_spline.T @ weighted_spline
        cross += np.asarray(weighted_spline.T @ weighted_raw)

    for start in range(0, sample_count, effective_batch_size):
        stop = min(start + effective_batch_size, sample_count)
        batch_points = points[start:stop]
        sqrt_weights = np.sqrt(weights[start:stop].numpy())
        spline = space.evaluate(
            batch_points.numpy(),
            derivative_order=sobolev_order,
        )
        if sobolev_order == 1:
            raw_values, raw_gradients = relu_power_feature_values_and_gradients(
                batch_points,
                cpu_parameters,
                power,
            )
            raw_hessians = None
        else:
            raw_values, raw_gradients, raw_hessians = relu_power_feature_data(
                batch_points,
                cpu_parameters,
                power,
            )
        accumulate_moment(spline.values, raw_values.numpy(), sqrt_weights)
        for axis, matrix in enumerate(spline.gradients):
            accumulate_moment(
                matrix,
                raw_gradients[:, :, axis].numpy(),
                sqrt_weights,
            )
        if sobolev_order == 2:
            assert raw_hessians is not None
            component = 0
            for row in range(spatial_dimension):
                for column in range(row, spatial_dimension):
                    multiplicity = math.sqrt(2.0) if row != column else 1.0
                    accumulate_moment(
                        spline.hessians[component],
                        raw_hessians[:, :, component].numpy(),
                        sqrt_weights,
                        multiplicity,
                    )
                    component += 1

    gram = gram.tocsc()
    scale = max(float(np.max(np.abs(gram.diagonal()))), 1.0)
    gram = gram + regularization * scale * scipy.sparse.eye(space.dimension, format="csc")
    factor = scipy.sparse.linalg.splu(gram)
    coefficients = factor.solve(cross)
    residual = gram @ coefficients - cross
    relative_residual = float(
        np.linalg.norm(residual) / max(np.linalg.norm(cross), np.finfo(float).tiny)
    )
    return RitzProjectedFeatures(
        space,
        coefficients,
        relative_residual,
        sample_count,
        activation_power=int(power),
    )


@dataclass(frozen=True)
class BallLeastSquaresResult:
    coefficients: np.ndarray
    rank: int
    singular_values: np.ndarray
    lagrange_multiplier: float
    constraint_active: bool

    @property
    def condition_estimate(self) -> float:
        retained = self.singular_values[: self.rank]
        if not retained.size:
            return float("inf")
        return float(retained[0] / retained[-1])


def solve_l2_ball_least_squares(
    matrix: np.ndarray,
    rhs: np.ndarray,
    *,
    radius: float,
    rcond: float = 1.0e-12,
    secular_tolerance: float = 1.0e-12,
) -> BallLeastSquaresResult:
    """Solve ``min ||Ac-b||`` subject to ``||c||_2 <= radius``.

    No column scaling is performed: ``radius`` therefore applies to the
    physical coefficients used to represent the numerical fields.
    """

    factor = factorize_l2_ball_least_squares(matrix, rhs)
    return factor.solve(radius, rcond=rcond, secular_tolerance=secular_tolerance)


@dataclass(frozen=True)
class L2BallLeastSquaresFactor:
    """Full SVD of one least-squares system, reusable across solvers.

    The expensive SVD depends only on ``(matrix, rhs)``.  Every downstream
    solve -- truncated minimum norm, ridge, or coefficient ball at any radius
    -- reduces to spectral filtering of this factorization, so hyperparameter
    sweeps and solver comparisons pay for a single decomposition.
    """

    right_transpose: np.ndarray
    singular_values: np.ndarray
    transformed_rhs: np.ndarray
    column_count: int

    def _active_mask(self, rcond: float) -> np.ndarray:
        if not self.singular_values.size:
            return np.zeros(0, dtype=bool)
        return self.singular_values > rcond * self.singular_values[0]

    def solve_min_norm(self, rcond: float) -> BallLeastSquaresResult:
        """Minimum-norm solution with singular values below ``rcond`` cut."""

        active = self._active_mask(rcond)
        coefficients = np.zeros(self.column_count, dtype=self.right_transpose.dtype)
        if np.any(active):
            coefficients = self.right_transpose[active].T @ (
                self.transformed_rhs[active] / self.singular_values[active]
            )
        return BallLeastSquaresResult(
            coefficients,
            int(np.count_nonzero(active)),
            self.singular_values,
            0.0,
            False,
        )

    def solve_ridge(self, regularization: float) -> BallLeastSquaresResult:
        """Tikhonov solution ``c = V diag(s/(s^2+lam^2)) U^T b``.

        The reported rank counts singular values at or above ``lam``, the
        scale below which spectral components are essentially suppressed.
        """

        if not (math.isfinite(regularization) and regularization > 0.0):
            raise ValueError("regularization must be finite and positive")
        if not self.singular_values.size:
            return self.solve_min_norm(0.0)
        filtered = (
            self.singular_values
            / (np.square(self.singular_values) + regularization * regularization)
        ) * self.transformed_rhs
        return BallLeastSquaresResult(
            self.right_transpose.T @ filtered,
            int(np.count_nonzero(self.singular_values >= regularization)),
            self.singular_values,
            regularization * regularization,
            True,
        )

    def solve(
        self,
        radius: float,
        *,
        rcond: float = 1.0e-12,
        secular_tolerance: float = 1.0e-12,
    ) -> BallLeastSquaresResult:
        if radius <= 0.0 and not math.isinf(radius):
            raise ValueError("radius must be positive")
        minimum = self.solve_min_norm(rcond)
        if math.isinf(radius) or np.linalg.norm(minimum.coefficients) <= radius:
            return minimum

        active = self._active_mask(rcond)
        squared_singular = np.square(self.singular_values[active])
        projected = self.singular_values[active] * self.transformed_rhs[active]

        def squared_norm(multiplier: float) -> float:
            return float(np.sum((projected / (squared_singular + multiplier)) ** 2))

        lower = 0.0
        upper = max(float(squared_singular.max()), 1.0)
        target = radius * radius
        while squared_norm(upper) > target:
            upper *= 2.0
        for _ in range(100):
            midpoint = 0.5 * (lower + upper)
            value = squared_norm(midpoint)
            if abs(value - target) <= secular_tolerance * max(target, 1.0):
                lower = upper = midpoint
                break
            if value > target:
                lower = midpoint
            else:
                upper = midpoint
        multiplier = 0.5 * (lower + upper)
        constrained = self.right_transpose[active].T @ (
            projected / (squared_singular + multiplier)
        )
        return BallLeastSquaresResult(
            constrained,
            minimum.rank,
            self.singular_values,
            multiplier,
            True,
        )


def factorize_l2_ball_least_squares(
    matrix: np.ndarray,
    rhs: np.ndarray,
) -> L2BallLeastSquaresFactor:
    """Compute the SVD factorization shared by every downstream solver.

    ``matrix`` is overwritten: callers hand over a reduced system they no
    longer need, and the LAPACK copy it would otherwise take is another
    ``m x m`` block at the widths where memory binds.
    """

    # SciPy's OpenBLAS SVD is the historical default.  The rented CPU image
    # also ships a PyTorch build linked against Intel MKL; on the large
    # three-dimensional reduced systems MKL is substantially faster.  Keep
    # the backend opt-in so the published/reference numbers remain unchanged
    # unless a run explicitly requests it (``LS_SVD_BACKEND=torch``).
    svd_backend = os.environ.get("LS_SVD_BACKEND", "scipy").strip().lower()
    if svd_backend in {"torch", "pytorch", "mkl"}:
        thread_token = os.environ.get("LS_SVD_THREADS")
        if thread_token:
            try:
                thread_count = int(thread_token)
            except ValueError as exc:
                raise ValueError("LS_SVD_THREADS must be a positive integer") from exc
            if thread_count < 1:
                raise ValueError("LS_SVD_THREADS must be a positive integer")
            torch.set_num_threads(thread_count)
        torch_matrix = torch.from_numpy(np.asarray(matrix))
        left_t, singular_t, right_t = torch.linalg.svd(
            torch_matrix,
            full_matrices=False,
        )
        left = left_t.numpy()
        singular_values = singular_t.numpy()
        right_transpose = right_t.numpy()
        del left_t, singular_t, right_t, torch_matrix
    elif svd_backend in {"scipy", "openblas"}:
        left, singular_values, right_transpose = scipy.linalg.svd(
            matrix,
            full_matrices=False,
            overwrite_a=True,
            check_finite=False,
            lapack_driver="gesdd",
        )
    else:
        raise ValueError(
            "LS_SVD_BACKEND must be one of scipy/openblas or torch/pytorch/mkl"
        )
    return L2BallLeastSquaresFactor(
        right_transpose=right_transpose,
        singular_values=singular_values,
        transformed_rhs=left.T @ rhs,
        column_count=matrix.shape[1],
    )
