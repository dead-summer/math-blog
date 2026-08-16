"""Problem-independent utilities shared by the least-squares drivers.

This module owns everything that does not depend on the PDE being solved:
quadrature rules on the unit box, the deterministic quasi-uniform ReLU-cubic
feature construction, the streaming Householder TSQR compression loop,
per-model defaults loading, and console table / plot helpers.  The pluggable
coefficient solvers live in ``solvers.py``; the physics lives in
``elasticity_common.py`` and the plate driver.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import get_lapack_funcs
import torch

from rfm_core import quasi_uniform_features
from solvers import (
    SOLVERS,
    VALID_ALGORITHMS,
    get_solver_spec,
    resolve_algorithm_id,
    run_solver,
)


DTYPE = torch.float64
DEVICE = torch.device("cpu")
BASE_SEED = 42
VALID_SAMPLING_METHODS = ("mc", "sobol", "gauss_legendre")
VALID_DIRECT_SOLVERS = ("dense", "streaming_tsqr")
ALGO_STYLE = {
    "LS(ball)": {"color": "#0077B6", "marker": "o", "linestyle": "-"},
    "LS(ridge)": {"color": "#D55E00", "marker": "s", "linestyle": "--"},
    "LS(tsvd)": {"color": "#009E73", "marker": "^", "linestyle": "-."},
}

torch.manual_seed(BASE_SEED)


def load_config_defaults(
    config_cls: type,
    directory: Path | str,
    filename: str = "defaults.json",
):
    """Build ``config_cls`` from a per-model defaults file.

    Unknown keys raise so typos surface immediately; a missing file falls
    back to the dataclass defaults.  The strings ``"inf"``/``"Infinity"``
    are accepted for unbounded values because JSON has no infinity literal.
    """

    path = Path(directory) / filename
    if not path.exists():
        return config_cls()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    field_names = {field.name for field in dataclasses.fields(config_cls)}
    unknown = sorted(set(payload) - field_names)
    if unknown:
        raise ValueError(f"Unknown keys {unknown} in {path}. Valid keys: {sorted(field_names)}")
    for key, value in payload.items():
        if isinstance(value, str) and value.lower() in {"inf", "infinity"}:
            payload[key] = math.inf
    return config_cls(**payload)


def clear_cuda_cache() -> None:
    """Release cached CUDA buffers after large tensors are freed."""

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def validate_sampling_method(method: str) -> None:
    """Reject unsupported sampling modes early."""

    if method not in VALID_SAMPLING_METHODS:
        raise ValueError(
            f"Unknown sampling_method='{method}'. "
            f"Valid values: {list(VALID_SAMPLING_METHODS)}"
        )


def validate_algorithm_selection(
    algorithm_ids: Sequence[str],
    valid_algorithm_ids: Sequence[str] = VALID_ALGORITHMS,
) -> list[str]:
    """Resolve aliases, validate algorithm ids, and preserve user order."""

    if not algorithm_ids:
        raise ValueError("algorithms_to_run must contain at least one algorithm id.")

    resolved_ids = [resolve_algorithm_id(algorithm_id) for algorithm_id in algorithm_ids]
    unknown_ids = [
        algorithm_id
        for algorithm_id, resolved in zip(algorithm_ids, resolved_ids)
        if resolved not in valid_algorithm_ids
    ]
    if unknown_ids:
        raise ValueError(
            f"Unknown algorithm ids: {unknown_ids}. Valid ids: {list(valid_algorithm_ids)}"
        )

    seen: set[str] = set()
    duplicates: list[str] = []
    for algorithm_id in resolved_ids:
        if algorithm_id in seen and algorithm_id not in duplicates:
            duplicates.append(algorithm_id)
        seen.add(algorithm_id)
    if duplicates:
        raise ValueError(f"Duplicate algorithm ids: {duplicates}")

    return resolved_ids


def infer_tensor_product_order(n_points: int, dim: int) -> int:
    """Infer the tensor-product order and reject non-perfect powers."""

    order = int(round(n_points ** (1.0 / dim)))
    if order <= 0 or order**dim != n_points:
        raise ValueError(
            f"gauss_legendre requires n_points = n^{dim}, got {n_points}."
        )
    return order


def build_quadrature_rule(
    n_points: int,
    method: str,
    dim: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build quadrature points and weights on the unit box."""

    validate_sampling_method(method)
    if method == "gauss_legendre":
        order = infer_tensor_product_order(n_points, dim)
        nodes_1d, weights_1d = np.polynomial.legendre.leggauss(order)
        nodes_1d = 0.5 * (nodes_1d + 1.0)
        weights_1d = 0.5 * weights_1d

        grids = np.meshgrid(*([nodes_1d] * dim), indexing="ij")
        weight_grids = np.meshgrid(*([weights_1d] * dim), indexing="ij")
        points = np.stack([grid.reshape(-1) for grid in grids], axis=1)
        weights = np.prod(
            np.stack([grid.reshape(-1) for grid in weight_grids], axis=1),
            axis=1,
        )
        return (
            torch.from_numpy(points).to(dtype=DTYPE, device=DEVICE),
            torch.from_numpy(weights).to(dtype=DTYPE, device=DEVICE),
        )

    if method == "sobol":
        engine = torch.quasirandom.SobolEngine(
            dimension=dim,
            scramble=True,
            seed=seed,
        )
        points = engine.draw(n_points).to(dtype=DTYPE, device=DEVICE)
        weights = torch.full(
            (n_points,),
            1.0 / n_points,
            dtype=DTYPE,
            device=DEVICE,
        )
        return points, weights

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    points = torch.rand(n_points, dim, generator=generator, dtype=DTYPE).to(DEVICE)
    weights = torch.full(
        (n_points,),
        1.0 / n_points,
        dtype=DTYPE,
        device=DEVICE,
    )
    return points, weights


def generate_features(N: int, dim: int) -> torch.Tensor:
    """Build the deterministic quasi-uniform ReLU-cubic parameters (omega, bias)."""

    return quasi_uniform_features(N, dim, dtype=DTYPE, device=DEVICE)


def iter_point_batches(point_count: int, batch_size: int) -> Iterator[tuple[int, int]]:
    """Yield half-open point ranges with a possibly shorter final batch."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, point_count, batch_size):
        yield start, min(start + batch_size, point_count)


@dataclass(frozen=True)
class StreamingTSQRStats:
    """Timing and shape diagnostics for streaming TSQR compression."""

    source_rows: int
    columns: int
    batch_count: int
    assembly_time: float
    qr_time: float
    total_time: float


def streaming_tsqr_compress(
    point_count: int,
    rows_per_point: int,
    columns: int,
    batch_size: int,
    qr_block_size: int,
    assemble_augmented_batch: Callable[[int, int], np.ndarray],
    show_progress: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, StreamingTSQRStats]:
    """Compress ``[A, b]`` with streaming Householder TSQR.

    ``assemble_augmented_batch(start, stop)`` must return a Fortran-contiguous
    ``(rows_per_point * (stop - start), columns + 1)`` block whose last column
    is the right-hand side.
    """

    total_started = time.perf_counter()
    augmented_columns = columns + 1
    reduced = np.zeros(
        (augmented_columns, augmented_columns),
        dtype=np.float64,
        order="F",
    )
    batch_count = math.ceil(point_count / batch_size)
    progress_stride = max(1, batch_count // 8)
    assembly_time = 0.0
    qr_time = 0.0
    tpqrt = None

    for batch_index, (start, stop) in enumerate(
        iter_point_batches(point_count, batch_size),
        start=1,
    ):
        assembly_started = time.perf_counter()
        augmented = assemble_augmented_batch(start, stop)
        assembly_time += time.perf_counter() - assembly_started
        if not augmented.flags.f_contiguous:
            raise RuntimeError("Augmented residual batch must be Fortran contiguous.")
        if tpqrt is None:
            tpqrt = get_lapack_funcs("tpqrt", (reduced, augmented))

        qr_started = time.perf_counter()
        updated_reduced, overwritten_batch, block_reflectors, info = tpqrt(
            0,
            min(qr_block_size, augmented_columns),
            reduced,
            augmented,
            overwrite_a=True,
            overwrite_b=True,
        )
        qr_time += time.perf_counter() - qr_started
        if info != 0:
            raise RuntimeError(f"DTPQRT failed on batch {batch_index} with info={info}.")
        if not np.shares_memory(updated_reduced, reduced):
            raise RuntimeError("DTPQRT copied the reduced factor instead of updating in-place.")
        if not np.shares_memory(overwritten_batch, augmented):
            raise RuntimeError("DTPQRT copied a residual batch instead of updating in-place.")
        reduced = updated_reduced
        del augmented, overwritten_batch, block_reflectors

        if show_progress and (
            batch_index == 1
            or batch_index == batch_count
            or batch_index % progress_stride == 0
        ):
            elapsed = time.perf_counter() - total_started
            eta = elapsed * (batch_count - batch_index) / batch_index
            print(
                f"  TSQR batch {batch_index}/{batch_count}: "
                f"points [{start}, {stop}), elapsed={elapsed:.1f}s, eta={eta:.1f}s"
            )

    reduced_matrix_numpy = reduced[:, :columns]
    if not reduced_matrix_numpy.flags.f_contiguous:
        raise RuntimeError("Reduced direct residual matrix must be Fortran contiguous.")
    reduced_rhs_numpy = np.array(reduced[:, columns], copy=True)
    stats = StreamingTSQRStats(
        source_rows=rows_per_point * point_count,
        columns=columns,
        batch_count=batch_count,
        assembly_time=assembly_time,
        qr_time=qr_time,
        total_time=time.perf_counter() - total_started,
    )
    return (
        torch.from_numpy(reduced_matrix_numpy),
        torch.from_numpy(reduced_rhs_numpy),
        stats,
    )


def print_aligned_markdown_table(
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    alignments: Sequence[str],
) -> None:
    """Print a compact markdown-style table with content-aware widths."""

    if not rows:
        return
    if len(headers) != len(alignments):
        raise ValueError("headers and alignments must have the same length.")
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("Each row must have the same number of columns as headers.")

    widths = [
        max(len(header), max(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]

    def format_row(row: Sequence[str]) -> str:
        cells: list[str] = []
        for index, cell in enumerate(row):
            if alignments[index] == "left":
                cells.append(f"{cell:<{widths[index]}}")
            elif alignments[index] == "center":
                cells.append(f"{cell:^{widths[index]}}")
            elif alignments[index] == "right":
                cells.append(f"{cell:>{widths[index]}}")
            else:
                raise ValueError(f"Unsupported alignment: {alignments[index]}")
        return f"| {' | '.join(cells)} |"

    def format_separator() -> str:
        cells: list[str] = []
        for index, alignment in enumerate(alignments):
            if alignment == "left":
                cells.append(f":{'-' * (widths[index] + 1)}")
            elif alignment == "center":
                cells.append(f":{'-' * widths[index]}:")
            elif alignment == "right":
                cells.append(f"{'-' * (widths[index] + 1)}:")
            else:
                raise ValueError(f"Unsupported alignment: {alignment}")
        return f"|{'|'.join(cells)}|"

    print(f"\n=== {title} ===\n")
    print(format_row(headers))
    print(format_separator())
    for row in rows:
        print(format_row(row))


def configure_plotting() -> None:
    """Apply the shared matplotlib settings used by experiment plots."""

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_error_summary(
    labels: Sequence[str],
    metric_titles: Sequence[str],
    metric_values: Sequence[Sequence[float]],
    save_path: str,
) -> None:
    """Plot final graph-norm errors as log-scale bar charts."""

    if not labels:
        print(f"  Skipped: {save_path} (no results to plot)")
        return

    configure_plotting()
    fig, axes = plt.subplots(1, len(metric_titles), figsize=(10.0, 4.5))
    if len(metric_titles) == 1:
        axes = [axes]
    x_positions = np.arange(len(labels), dtype=float)
    colors = [
        ALGO_STYLE.get(label, {}).get("color", "#4C78A8")
        for label in labels
    ]

    for ax, title, values_row in zip(axes, metric_titles, metric_values):
        values = np.asarray(values_row, dtype=float)
        valid = np.isfinite(values) & (values > 0.0)
        if valid.any():
            valid_indices = np.flatnonzero(valid)
            ax.bar(
                x_positions[valid],
                values[valid],
                width=0.65,
                color=[colors[index] for index in valid_indices],
            )
        for index in np.flatnonzero(~valid):
            print(f"  Skipped {labels[index]} {title}={values[index]!r} in {save_path}")

        ax.set_yscale("log")
        ax.set_ylabel("Graph-norm error")
        ax.set_title(title)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(alpha=0.3, linestyle="--", axis="y")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def warn_if_dense_infeasible(
    direct_solver: str,
    source_rows: int,
    columns: int,
) -> float:
    """Print the dense-system estimate and warn when it may exhaust memory."""

    dense_matrix_gib = (
        source_rows * columns * torch.tensor([], dtype=DTYPE).element_size() / 2**30
    )
    print(
        f"Direct system estimate: ({source_rows}, {columns}), "
        f"rows/columns={source_rows / columns:.2f}, "
        f"dense A={dense_matrix_gib:.2f} GiB"
    )
    if direct_solver == "dense" and dense_matrix_gib > 4.0:
        warnings.warn(
            "The selected dense direct backend may exceed workstation memory; "
            "use direct_solver='streaming_tsqr' for this configuration.",
            RuntimeWarning,
        )
    return dense_matrix_gib
