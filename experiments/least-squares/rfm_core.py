"""Shared numerical primitives for the least-squares linearized-network experiments.

The module deliberately keeps the four model drivers free of activation-,
projection-, and coefficient-solver details.  Features are ReLU cubics whose
hidden parameters form a deterministic quasi-uniform tensor point set on the
direction--bias domain ``S^{d-1} x [-c, c]``; no feature randomness remains.
Essential boundary conditions are imposed by an empirical Sobolev--Ritz
projection into tensor-product cubic B-spline spaces.  The output coefficients
are constrained in their physical, unscaled Euclidean norm.
"""

from __future__ import annotations

import functools
import itertools
import math
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


def _affine_cube_coefficients(
    direction: np.ndarray,
    bias: float,
    multi_indices: list[tuple[int, ...]],
) -> np.ndarray:
    """Coefficients of ``(direction @ x + bias)^3`` in a monomial basis."""

    coefficients = []
    for alpha in multi_indices:
        alpha_degree = sum(alpha)
        multinomial = math.factorial(3) / (
            math.factorial(3 - alpha_degree)
            * math.prod(math.factorial(component) for component in alpha)
        )
        direction_factor = math.prod(
            direction[axis] ** exponent for axis, exponent in enumerate(alpha)
        )
        coefficients.append(
            multinomial * bias ** (3 - alpha_degree) * direction_factor
        )
    return np.asarray(coefficients)


@functools.lru_cache(maxsize=8)
def _polynomial_supplement_parameters_cached(
    spatial_dimension: int,
    bias_range: float,
) -> np.ndarray:
    """Select globally positive ridge cubics spanning ``P_3 / constants``.

    Cubes of affine forms span all polynomials of total degree at most three.
    We choose a small, deterministic, well-conditioned subset by pivoted QR.
    The separate constant feature supplies the one omitted polynomial degree
    of freedom.
    """

    polynomial_dimension = math.comb(spatial_dimension + 3, 3)
    supplement_count = polynomial_dimension - 1
    direction_count = 64 if spatial_dimension == 2 else 128
    direction_builder = (
        _circle_directions
        if spatial_dimension == 2
        else _fibonacci_sphere_directions
    )
    directions = direction_builder(direction_count, 0.37)

    # Every candidate is strictly positive on [0,1]^d, hence its ReLU cubic
    # is an ordinary affine cubic there.  Varying both direction and bias is
    # essential: a single fixed-bias sphere does not span all of P_3.
    root_dimension = math.sqrt(float(spatial_dimension))
    margin = bias_range - root_dimension
    candidate_biases = np.linspace(
        root_dimension + 0.1 * margin,
        root_dimension + 0.9 * margin,
        5,
    )
    multi_indices = _total_degree_indices(spatial_dimension, degree=3)
    candidate_parameters: list[np.ndarray] = []
    candidate_coefficients: list[np.ndarray] = []
    for direction in directions:
        for bias in candidate_biases:
            candidate_parameters.append(np.concatenate([direction, [bias]]))
            candidate_coefficients.append(
                _affine_cube_coefficients(direction, float(bias), multi_indices)
            )

    coefficient_matrix = np.stack(candidate_coefficients, axis=1)
    # The first row is the constant coefficient.  Projecting it out fixes the
    # explicit constant feature and lets QR choose the remaining P_3 basis.
    _, _, pivots = scipy.linalg.qr(
        coefficient_matrix[1:, :],
        mode="economic",
        pivoting=True,
        check_finite=False,
    )
    selected = np.stack(candidate_parameters, axis=0)[pivots[:supplement_count]]
    augmented = np.column_stack(
        [
            np.eye(polynomial_dimension)[:, 0],
            coefficient_matrix[:, pivots[:supplement_count]],
        ]
    )
    if np.linalg.matrix_rank(augmented) != polynomial_dimension:
        raise RuntimeError("Failed to construct a complete cubic polynomial supplement.")
    selected.setflags(write=False)
    return selected


def polynomial_supplement_parameters(
    spatial_dimension: int,
    *,
    bias_range: float = DEFAULT_BIAS_RANGE,
) -> np.ndarray:
    """Return ridge parameters whose restrictions supplement ``P_3``."""

    if spatial_dimension not in (2, 3):
        raise ValueError("polynomial supplements are implemented for d in {2, 3}")
    if bias_range <= math.sqrt(float(spatial_dimension)):
        raise ValueError("bias_range must exceed sqrt(d) for the P_3 supplement")
    return _polynomial_supplement_parameters_cached(
        spatial_dimension,
        float(bias_range),
    ).copy()


def quasi_uniform_features(
    width: int,
    spatial_dimension: int,
    *,
    bias_range: float = DEFAULT_BIAS_RANGE,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build ``width`` useful quasi-uniform parameters ``(omega, b)``.

    Non-polynomial rows tile the active parameter domain: for each direction,
    the hyperplane ``omega @ x + b = 0`` intersects the unit box.  Biases are
    midpoint layers in the normalized fibre coordinate, and each layer has a
    quasi-uniform direction set (equispaced angles in 2D, a Fibonacci lattice
    in 3D) with deterministic rotations.  Parameters outside this domain give
    either the zero function or an ordinary cubic on the box.  Instead of
    wasting a positive fraction of ``width`` on those degenerate rows, a fixed
    QR-selected set of globally positive ridge cubics supplies ``P_3`` exactly.
    Together with the evaluator's constant column, this is the finite
    polynomial supplement used in the approximation theorem.
    """

    if width < 0:
        raise ValueError("width must be nonnegative")
    if spatial_dimension not in (2, 3):
        raise ValueError("quasi-uniform parameters are implemented for d in {2, 3}")
    if not (
        math.isfinite(bias_range)
        and bias_range > math.sqrt(float(spatial_dimension))
    ):
        raise ValueError("bias_range must be finite and exceed sqrt(d)")
    if width == 0:
        return torch.empty(0, spatial_dimension + 1, dtype=dtype, device=device)

    supplement = polynomial_supplement_parameters(
        spatial_dimension,
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
    # bias by the normalized fibre coordinate only changes fixed constants.
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
        fibre_coordinate = (layer + 0.5) / layer_count
        # omega @ x ranges over [minimum, maximum] on the unit box, so these
        # midpoint biases make every retained hyperplane cross its interior.
        biases = -maximum + fibre_coordinate * (maximum - minimum)
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
    fibre_coordinate = torch.rand(
        probe_count,
        1,
        generator=generator,
        dtype=normalized.dtype,
    )
    biases = -maximum + fibre_coordinate * (maximum - minimum)
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


def _relu3_positive_parts(
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


def _prepend_constant_values(
    points: torch.Tensor,
    random_values: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        [
            torch.ones(
                points.shape[0],
                1,
                dtype=points.dtype,
                device=points.device,
            ),
            random_values,
        ],
        dim=1,
    )


def relu3_feature_values(
    points: torch.Tensor,
    parameters: torch.Tensor,
) -> torch.Tensor:
    """Evaluate only ReLU-cubic feature values.

    This is the memory-light path for projections and diagnostics that do not
    use derivatives.  As in :func:`relu3_feature_data`, the leading column is
    the deterministic constant feature ``1``.
    """

    _, positive = _relu3_positive_parts(points, parameters)
    return _prepend_constant_values(points, positive.pow(3))


def relu3_feature_box_means(parameters: torch.Tensor) -> torch.Tensor:
    """Integrate all ReLU-cubic features exactly over the unit box.

    For a crossing hyperplane, reflect coordinates with negative direction
    components and apply the inclusion--exclusion antiderivative formula.  A
    component at floating-point zero is removed before division.  Globally
    positive affine cubics use their stable closed-form third moment instead;
    globally negative features integrate to zero.  The leading returned entry
    is the exact mean of the constant feature.
    """

    if parameters.ndim != 2:
        raise ValueError("parameters must be a matrix")
    if not parameters.dtype.is_floating_point:
        raise ValueError("parameters must use a floating-point dtype")
    spatial_dimension = parameters.shape[1] - 1
    if spatial_dimension not in (2, 3):
        raise ValueError("exact ReLU-cubic box means are implemented for d in {2, 3}")

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
        positive_directions = cleaned_directions[positive_mask]
        centred_mean = biases[positive_mask] + 0.5 * positive_directions.sum(dim=1)
        # If X_i ~ U(0,1), the centred third moment of omega @ X is zero and
        # its variance is sum(omega_i^2)/12.
        means[1:][positive_mask] = (
            centred_mean.pow(3)
            + 0.25 * centred_mean * positive_directions.square().sum(dim=1)
        )

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
        power = 3 + active_dimension
        numerator = torch.zeros((), dtype=parameters.dtype, device=parameters.device)
        for vertex in itertools.product((0, 1), repeat=active_dimension):
            vertex_tensor = torch.tensor(
                vertex,
                dtype=parameters.dtype,
                device=parameters.device,
            )
            value = torch.relu(oriented_bias + torch.dot(magnitudes, vertex_tensor))
            sign = -1.0 if (active_dimension - sum(vertex)) % 2 else 1.0
            numerator = numerator + sign * value.pow(power)
        denominator = magnitudes.prod() * float(
            math.prod(range(4, 4 + active_dimension))
        )
        integral = numerator / denominator
        # Roundoff can only create a tiny negative value in this nonnegative
        # integral.  Clamping preserves the exact formula in normal cases.
        means[feature_index + 1] = integral.clamp_min(0.0)
    return means


def relu3_feature_values_and_gradients(
    points: torch.Tensor,
    parameters: torch.Tensor,
) -> ValueGradientTuple:
    """Evaluate ReLU-cubic values and gradients without forming Hessians."""

    directions, positive = _relu3_positive_parts(points, parameters)
    values = _prepend_constant_values(points, positive.pow(3))
    random_gradients = (
        3.0 * positive.pow(2).unsqueeze(2) * directions.unsqueeze(0)
    )
    gradients = torch.cat(
        [
            torch.zeros(
                points.shape[0],
                1,
                points.shape[1],
                dtype=points.dtype,
                device=points.device,
            ),
            random_gradients,
        ],
        dim=1,
    )
    return values, gradients


def relu3_feature_data(
    points: torch.Tensor,
    parameters: torch.Tensor,
    *,
    hessian_components: Iterable[tuple[int, int]] | None = None,
) -> ArrayTuple:
    """Evaluate ReLU-cubic features, gradients, and selected Hessian entries.

    The first column is the deterministic feature ``1``.  Parameters have
    shape ``(N, d + 1)`` and are interpreted as ``(omega, bias)``.  Call
    :func:`relu3_feature_values` or
    :func:`relu3_feature_values_and_gradients` when higher derivatives are not
    needed, so the corresponding tensors are never allocated.
    """

    directions, positive = _relu3_positive_parts(points, parameters)
    spatial_dimension = points.shape[1]
    values = _prepend_constant_values(points, positive.pow(3))
    random_gradients = (
        3.0 * positive.pow(2).unsqueeze(2) * directions.unsqueeze(0)
    )
    gradients = torch.cat(
        [
            torch.zeros(
                points.shape[0],
                1,
                spatial_dimension,
                dtype=points.dtype,
                device=points.device,
            ),
            random_gradients,
        ],
        dim=1,
    )

    if hessian_components is None:
        components = tuple(
            (row, column)
            for row in range(spatial_dimension)
            for column in range(row, spatial_dimension)
        )
    else:
        components = tuple(hessian_components)
    if components:
        random_hessians = torch.stack(
            [
                6.0
                * positive
                * directions[:, row].unsqueeze(0)
                * directions[:, column].unsqueeze(0)
                for row, column in components
            ],
            dim=2,
        )
    else:
        random_hessians = torch.empty(
            points.shape[0],
            parameters.shape[0],
            0,
            dtype=points.dtype,
            device=points.device,
        )
    hessians = torch.cat(
        [
            torch.zeros(
                points.shape[0],
                1,
                len(components),
                dtype=points.dtype,
                device=points.device,
            ),
            random_hessians,
        ],
        dim=1,
    )
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
    """Form row-wise tensor products while retaining local spline support."""

    row_count = matrices[0].shape[0]
    widths = [matrix.shape[1] for matrix in matrices]
    column_count = math.prod(widths)
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []

    for row in range(row_count):
        supports: list[list[tuple[int, float]]] = []
        for matrix in matrices:
            nonzero = np.flatnonzero(np.abs(matrix[row]) > 1.0e-14)
            supports.append([(int(index), float(matrix[row, index])) for index in nonzero])
        for factors in itertools.product(*supports):
            multi_index = tuple(index for index, _ in factors)
            value = math.prod(entry for _, entry in factors)
            if value == 0.0:
                continue
            row_indices.append(row)
            column_indices.append(int(np.ravel_multi_index(multi_index, widths)))
            values.append(value)
    return scipy.sparse.csr_matrix(
        (values, (row_indices, column_indices)),
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
    """Boundary-adapted tensor-product cubic B-spline space on ``[0,1]^d``."""

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

    @property
    def feature_count(self) -> int:
        return self.coefficients.shape[1]

    def evaluate_values(self, points: torch.Tensor) -> torch.Tensor:
        """Evaluate projected values without forming derivative matrices."""

        point_array = points.detach().cpu().numpy()
        spline_values = self.space.evaluate(
            point_array,
            derivative_order=0,
        ).values
        values = spline_values @ self.coefficients
        return torch.from_numpy(np.asarray(values)).to(
            dtype=points.dtype,
            device=points.device,
        )

    def evaluate_values_and_gradients(
        self,
        points: torch.Tensor,
    ) -> ValueGradientTuple:
        """Evaluate projected values and gradients without forming Hessians."""

        point_array = points.detach().cpu().numpy()
        spline_data = self.space.evaluate(point_array, derivative_order=1)
        values = spline_data.values @ self.coefficients
        gradients = np.stack(
            [matrix @ self.coefficients for matrix in spline_data.gradients],
            axis=2,
        )
        return (
            torch.from_numpy(np.asarray(values)).to(
                dtype=points.dtype,
                device=points.device,
            ),
            torch.from_numpy(np.asarray(gradients)).to(
                dtype=points.dtype,
                device=points.device,
            ),
        )

    def evaluate(
        self,
        points: torch.Tensor,
        *,
        hessian_components: tuple[tuple[int, int], ...] | None = None,
    ) -> ArrayTuple:
        point_array = points.detach().cpu().numpy()
        spline_data = self.space.evaluate(point_array)
        values = spline_data.values @ self.coefficients
        gradients = np.stack(
            [matrix @ self.coefficients for matrix in spline_data.gradients],
            axis=2,
        )
        all_components = tuple(
            (row, column)
            for row in range(self.space.spatial_dimension)
            for column in range(row, self.space.spatial_dimension)
        )
        selected = all_components if hessian_components is None else hessian_components
        component_map = {component: index for index, component in enumerate(all_components)}
        hessians = np.stack(
            [
                spline_data.hessians[component_map[component]] @ self.coefficients
                for component in selected
            ],
            axis=2,
        )
        return (
            torch.from_numpy(np.asarray(values)).to(dtype=points.dtype, device=points.device),
            torch.from_numpy(np.asarray(gradients)).to(dtype=points.dtype, device=points.device),
            torch.from_numpy(np.asarray(hessians)).to(dtype=points.dtype, device=points.device),
        )

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
    quadrature_samples: int | None = None,
    quadrature_seed: int = 91_003,
    regularization: float = 1.0e-12,
    batch_size: int | None = None,
) -> RitzProjectedFeatures:
    """Project ReLU-cubic features into a boundary-adapted spline space.

    The Sobolev inner product is discretized with an independent Monte Carlo
    rule.  Boundary conditions remain exact because every auxiliary basis
    function has the required zero traces.  Gram and cross moments are
    accumulated by batches so memory scales with ``batch_size`` instead of the
    total quadrature count.  The Monte Carlo points are drawn once before
    batching, hence changing ``batch_size`` does not change the quadrature
    rule.
    """

    spatial_dimension = parameters.shape[1] - 1
    space = TensorSplineSpace.with_minimum_dimension(
        spatial_dimension,
        sobolev_order,
        auxiliary_dimension,
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
            raw_values, raw_gradients = relu3_feature_values_and_gradients(
                batch_points,
                cpu_parameters,
            )
            raw_hessians = None
        else:
            raw_values, raw_gradients, raw_hessians = relu3_feature_data(
                batch_points,
                cpu_parameters,
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
    return RitzProjectedFeatures(space, coefficients, relative_residual, sample_count)


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
    """Compute the SVD factorization shared by every downstream solver."""

    left, singular_values, right_transpose = scipy.linalg.svd(
        matrix,
        full_matrices=False,
        overwrite_a=False,
        check_finite=False,
        lapack_driver="gesdd",
    )
    return L2BallLeastSquaresFactor(
        right_transpose=right_transpose,
        singular_values=singular_values,
        transformed_rhs=left.T @ rhs,
        column_count=matrix.shape[1],
    )
