"""System-design backends for the least-squares experiments.

The direct backend keeps a residual matrix ``A`` and right-hand side ``b``.
The Gram backend instead assembles the discrete Galerkin system of the
least-squares variational problem ``a(u, v) = l(v)``: under the shared
quadrature rule the stiffness matrix and load vector are exactly

``G = A.T @ A``, ``g = A.T @ b``, and ``bTb = b.T @ b``,

so the backend streams the *same* residual batches and stores only these
moments.  Consequently it changes neither the Monte Carlo functional nor the
physical coefficient norm.  In particular, this module deliberately performs
no column normalization.  The generic batch accumulator is the integration
point for the elasticity/plate residual assemblers: pass their existing
``[A_i, b_i]`` callback rather than duplicating PDE-specific moment formulae.

Normal equations square the condition number, so ``sqrt(machine epsilon)``
is the theoretical resolution limit for relative singular-value cutoffs.
In the campaign hyperparameter regimes this limit is not binding: measured
graph-norm errors at ``rcond = 1e-8`` agree with the direct SVD backend to
within one percent, hence smaller cutoffs are accepted and the direct
backend remains available as the high-precision verification path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol, runtime_checkable

import numpy as np
import scipy.linalg
import torch

from rfm_core import BallLeastSquaresResult, L2BallLeastSquaresFactor


AugmentedResidualBatch = np.ndarray | torch.Tensor
AugmentedBatchAssembler = Callable[[int, int], AugmentedResidualBatch]


@dataclass(frozen=True)
class SystemBackendSpec:
    """Metadata for one residual-system storage backend."""

    id: str
    label: str
    stores_normal_equations: bool


SYSTEM_BACKENDS: dict[str, SystemBackendSpec] = {
    spec.id: spec
    for spec in (
        SystemBackendSpec("direct", "direct residual matrix", False),
        SystemBackendSpec("gram", "streaming Gram matrix", True),
    )
}
VALID_SYSTEM_BACKENDS = tuple(SYSTEM_BACKENDS)


def get_system_backend(backend_id: str) -> SystemBackendSpec:
    """Return registered backend metadata, rejecting unknown identifiers."""

    try:
        return SYSTEM_BACKENDS[backend_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown system_backend='{backend_id}'. "
            f"Valid values: {list(VALID_SYSTEM_BACKENDS)}"
        ) from exc


@runtime_checkable
class LeastSquaresSpectralFactor(Protocol):
    """Structural interface shared by direct-SVD and Gram-eigh factors."""

    singular_values: np.ndarray
    column_count: int

    def solve_min_norm(self, rcond: float) -> BallLeastSquaresResult: ...

    def solve_ridge(self, regularization: float) -> BallLeastSquaresResult: ...

    def solve(
        self,
        radius: float,
        *,
        rcond: float = 1.0e-12,
        secular_tolerance: float = 1.0e-12,
    ) -> BallLeastSquaresResult: ...


@dataclass
class GramResidualDesign:
    """Galerkin (normal-equation) representation ``G c = g`` of a weighted
    residual design.

    ``gram`` and ``rhs_moment`` are kept as CPU float64 tensors in the normal
    experiment path.  The class also accepts other floating dtypes/devices so
    small standalone tests and future assemblers can reuse the interface.
    Treat the tensors as immutable after construction: the cached spectral
    factor corresponds to their values at the first factorization.
    """

    gram: torch.Tensor
    rhs_moment: torch.Tensor
    rhs_norm_squared: float
    source_rows: int
    _spectral_factor: "GramSpectralFactor | None" = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.gram.ndim != 2 or self.gram.shape[0] != self.gram.shape[1]:
            raise ValueError("gram must be a square matrix")
        if self.rhs_moment.ndim != 1 or self.rhs_moment.numel() != self.gram.shape[0]:
            raise ValueError("rhs_moment dimension must match gram")
        if not self.gram.is_floating_point() or not self.rhs_moment.is_floating_point():
            raise TypeError("Gram data must use a floating-point dtype")
        if self.gram.dtype != self.rhs_moment.dtype:
            raise TypeError("gram and rhs_moment must use the same dtype")
        if self.gram.device != self.rhs_moment.device:
            raise ValueError("gram and rhs_moment must be on the same device")
        if self.source_rows < 0:
            raise ValueError("source_rows must be non-negative")
        if not math.isfinite(float(self.rhs_norm_squared)) or self.rhs_norm_squared < 0.0:
            raise ValueError("rhs_norm_squared must be finite and non-negative")

    @property
    def column_count(self) -> int:
        return int(self.gram.shape[0])

    # Short mathematical aliases are convenient for diagnostics and mirror
    # the notation in the paper/implementation plan.
    @property
    def G(self) -> torch.Tensor:
        return self.gram

    @property
    def g(self) -> torch.Tensor:
        return self.rhs_moment

    @property
    def bTb(self) -> float:
        return self.rhs_norm_squared

    def objective_squared(self, coefficients: np.ndarray | torch.Tensor) -> float:
        """Evaluate ``||A c-b||^2`` without retaining ``A`` or ``b``."""

        coefficient_tensor = torch.as_tensor(
            coefficients,
            dtype=self.gram.dtype,
            device=self.gram.device,
        )
        if coefficient_tensor.ndim != 1 or coefficient_tensor.numel() != self.column_count:
            raise ValueError("coefficient dimension must match gram")
        value = (
            coefficient_tensor @ (self.gram @ coefficient_tensor)
            - 2.0 * (self.rhs_moment @ coefficient_tensor)
            + self.rhs_norm_squared
        )
        # A tiny negative value is possible after cancellation near an exact
        # fit.  Do not conceal a materially invalid normal system.
        result = float(value)
        tolerance = 128.0 * torch.finfo(self.gram.dtype).eps * max(
            abs(float(self.rhs_norm_squared)),
            1.0,
        )
        if result < -tolerance:
            raise ValueError(f"normal-equation objective is negative ({result:.3e})")
        return max(result, 0.0)

    def factorize(self) -> "GramSpectralFactor":
        return factorize_gram_least_squares(self)


def _as_floating_tensor(
    batch: AugmentedResidualBatch,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    tensor = batch if isinstance(batch, torch.Tensor) else torch.from_numpy(np.asarray(batch))
    if tensor.ndim != 2:
        raise ValueError("each augmented residual batch must be a matrix")
    if not tensor.is_floating_point():
        raise TypeError("augmented residual batches must use a floating-point dtype")
    return tensor.to(dtype=dtype, device=device)


def accumulate_gram_from_batches(
    batches: Iterable[AugmentedResidualBatch],
    *,
    column_count: int | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> GramResidualDesign:
    """Stream augmented blocks ``[A_i,b_i]`` into ``G``, ``g``, and ``bTb``.

    ``Tensor.addmm_``/``addmv_`` accumulate directly into the final buffers;
    no additional column-square product is materialized.  Peak storage is
    therefore one Gram matrix plus one residual batch.
    """

    if not dtype.is_floating_point:
        raise TypeError("Gram accumulation dtype must be floating point")
    if column_count is not None and column_count < 0:
        raise ValueError("column_count must be non-negative")

    gram: torch.Tensor | None = None
    rhs_moment: torch.Tensor | None = None
    rhs_norm = torch.zeros((), dtype=dtype, device=device)
    source_rows = 0

    for raw_batch in batches:
        augmented = _as_floating_tensor(raw_batch, dtype=dtype, device=device)
        inferred_columns = augmented.shape[1] - 1
        if inferred_columns < 0:
            raise ValueError("an augmented residual batch needs at least one column")
        if column_count is None:
            column_count = inferred_columns
        if inferred_columns != column_count:
            raise ValueError(
                "augmented residual batch width does not match column_count: "
                f"got {inferred_columns}, expected {column_count}"
            )
        if gram is None:
            gram = torch.zeros(
                column_count,
                column_count,
                dtype=dtype,
                device=device,
            )
            rhs_moment = torch.zeros(column_count, dtype=dtype, device=device)

        matrix = augmented[:, :column_count]
        rhs = augmented[:, column_count]
        gram.addmm_(matrix.T, matrix, beta=1.0, alpha=1.0)
        rhs_moment.addmv_(matrix.T, rhs, beta=1.0, alpha=1.0)
        rhs_norm.add_(torch.dot(rhs, rhs))
        source_rows += int(augmented.shape[0])

    if gram is None or rhs_moment is None:
        if column_count is None:
            raise ValueError("cannot infer column_count from an empty batch stream")
        gram = torch.zeros(column_count, column_count, dtype=dtype, device=device)
        rhs_moment = torch.zeros(column_count, dtype=dtype, device=device)

    return GramResidualDesign(
        gram=gram,
        rhs_moment=rhs_moment,
        rhs_norm_squared=float(rhs_norm),
        source_rows=source_rows,
    )


def assemble_gram_residual_design(
    *,
    point_count: int,
    batch_size: int,
    assemble_augmented_batch: AugmentedBatchAssembler,
    column_count: int | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> GramResidualDesign:
    """Accumulate a Gram design from the existing point-range assembler."""

    if point_count < 0:
        raise ValueError("point_count must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    ranges = (
        assemble_augmented_batch(start, min(start + batch_size, point_count))
        for start in range(0, point_count, batch_size)
    )
    return accumulate_gram_from_batches(
        ranges,
        column_count=column_count,
        dtype=dtype,
        device=device,
    )


# Verb-oriented alias retained for callers that describe the operation as an
# accumulation rather than an assembly.
accumulate_gram_residual_design = assemble_gram_residual_design


def _symmetrize_in_place(matrix: np.ndarray) -> None:
    """Average both triangles using only O(n) temporary storage."""

    for row in range(matrix.shape[0] - 1):
        average = 0.5 * (matrix[row, row + 1 :] + matrix[row + 1 :, row])
        matrix[row, row + 1 :] = average
        matrix[row + 1 :, row] = average


@dataclass(frozen=True)
class GramSpectralFactor:
    """Eigendecomposition of the Galerkin system ``G c = g``.

    If ``G = V diag(eta) V.T`` and ``g=A.T b``, then the equivalent direct
    factor has singular values ``sqrt(eta)`` and transformed right-hand side
    ``(V.T g) / sqrt(eta)``; a singular-value cutoff ``rcond`` therefore acts
    as the eigenvalue cutoff ``rcond**2``.  Delegating the spectral filters
    to :class:`L2BallLeastSquaresFactor` then gives exactly these mappings:

    * TSVD: ``eta > rcond**2 * eta_max``;
    * ridge: normal shift ``lambda_rel**2 * eta_max`` when the caller passes
      ``lambda_rel * sigma_max``;
    * ball: ``(G + mu I)c=g`` with the unchanged physical radius.
    """

    direct_compatible_factor: L2BallLeastSquaresFactor
    eigenvalues: np.ndarray
    projected_rhs_moment: np.ndarray
    rhs_norm_squared: float
    machine_epsilon: float

    @property
    def right_transpose(self) -> np.ndarray:
        return self.direct_compatible_factor.right_transpose

    @property
    def singular_values(self) -> np.ndarray:
        return self.direct_compatible_factor.singular_values

    @property
    def transformed_rhs(self) -> np.ndarray:
        return self.direct_compatible_factor.transformed_rhs

    @property
    def column_count(self) -> int:
        return self.direct_compatible_factor.column_count

    def _validate_rcond(self, rcond: float) -> None:
        if not (math.isfinite(rcond) and rcond > 0.0):
            raise ValueError("Gram rcond must be finite and positive")

    def solve_min_norm(self, rcond: float) -> BallLeastSquaresResult:
        self._validate_rcond(rcond)
        return self.direct_compatible_factor.solve_min_norm(rcond)

    def solve_ridge(self, regularization: float) -> BallLeastSquaresResult:
        return self.direct_compatible_factor.solve_ridge(regularization)

    def solve(
        self,
        radius: float,
        *,
        rcond: float = 1.0e-12,
        secular_tolerance: float = 1.0e-12,
    ) -> BallLeastSquaresResult:
        self._validate_rcond(rcond)
        return self.direct_compatible_factor.solve(
            radius,
            rcond=rcond,
            secular_tolerance=secular_tolerance,
        )


def factorize_gram_least_squares(design: GramResidualDesign) -> GramSpectralFactor:
    """Eigendecompose a Gram design and cache a direct-compatible factor."""

    if design._spectral_factor is not None:
        return design._spectral_factor
    if design.gram.device.type != "cpu" or design.rhs_moment.device.type != "cpu":
        raise ValueError("Gram eigendecomposition currently requires CPU tensors")

    gram = design.gram.detach().numpy()
    rhs_moment = design.rhs_moment.detach().numpy()
    if not np.isfinite(gram).all() or not np.isfinite(rhs_moment).all():
        raise ValueError("Gram normal equations contain non-finite values")

    # Accumulated A.T@A blocks are symmetric to roundoff.  Explicitly average
    # the triangles before eigh, but do so in-place to avoid another O(n^2)
    # allocation in the memory-saving backend.
    _symmetrize_in_place(gram)
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        gram,
        lower=True,
        overwrite_a=False,
        check_finite=False,
        driver="evd",
    )
    order = np.arange(eigenvalues.size - 1, -1, -1)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    dtype = gram.dtype
    epsilon = float(np.finfo(dtype).eps)
    spectral_scale = max(
        float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0,
        float(np.finfo(dtype).tiny),
    )
    negative_tolerance = 64.0 * max(design.column_count, 1) * epsilon * spectral_scale
    if eigenvalues.size and float(eigenvalues[-1]) < -negative_tolerance:
        raise ValueError(
            "Gram matrix is materially indefinite: "
            f"lambda_min={eigenvalues[-1]:.3e}, tolerance={negative_tolerance:.3e}"
        )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    singular_values = np.sqrt(eigenvalues)
    projected_rhs = eigenvectors.T @ rhs_moment
    transformed_rhs = np.zeros_like(projected_rhs)
    positive = singular_values > 0.0
    transformed_rhs[positive] = projected_rhs[positive] / singular_values[positive]

    compatible = L2BallLeastSquaresFactor(
        right_transpose=eigenvectors.T,
        singular_values=singular_values,
        transformed_rhs=transformed_rhs,
        column_count=design.column_count,
    )
    factor = GramSpectralFactor(
        direct_compatible_factor=compatible,
        eigenvalues=eigenvalues,
        projected_rhs_moment=projected_rhs,
        rhs_norm_squared=design.rhs_norm_squared,
        machine_epsilon=epsilon,
    )
    design._spectral_factor = factor
    return factor


__all__ = [
    "GramResidualDesign",
    "GramSpectralFactor",
    "LeastSquaresSpectralFactor",
    "SYSTEM_BACKENDS",
    "VALID_SYSTEM_BACKENDS",
    "accumulate_gram_from_batches",
    "accumulate_gram_residual_design",
    "assemble_gram_residual_design",
    "factorize_gram_least_squares",
    "get_system_backend",
]
