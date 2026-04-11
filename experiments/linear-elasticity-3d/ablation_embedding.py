from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass, replace
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

from linear_elasticity_3d import (
    ALGO_STYLE,
    BASE_SEED,
    DEVICE,
    DTYPE,
    OUTPUT_DIR as BASE_OUTPUT_DIR,
    AlgorithmResult,
    MainConfig,
    SharedBenchmarkData,
    SharedComparisonConfig,
    TOP_LEVEL_ALGORITHM_LABELS,
    VALID_TOP_LEVEL_ALGORITHMS,
    apply_shared_to_strong_config,
    apply_shared_to_weak_config,
    build_shared_benchmark,
    clear_cuda_cache,
    compute_lame_constants,
    compute_stress_voigt,
    configure_plotting,
    eval_exact_displacement,
    extract_scoped_algorithm_ids,
    make_default_main_config,
    print_result_summary,
    synchronize_device,
    validate_algorithm_selection,
    validate_shared_comparison_config,
)
from projection import (
    apply_shared_to_projection_config,
    validate_config as validate_projection_config,
)
from strong_form import (
    IDENTITY_3,
    STRAIN_GRAD_BASES,
    add_block_scaled,
    add_rhs_feature_blocks,
    solve_eigh as solve_strong_eigh,
    solve_lstsq as solve_strong_lstsq,
    split_solution as split_strong_solution,
    validate_config as validate_strong_config,
)
from weak_form import (
    assemble_system as assemble_weak_system,
    solve_eigh as solve_weak_eigh,
    solve_lstsq as solve_weak_lstsq,
    validate_config as validate_weak_config,
)


FEATURE_ALGORITHM_IDS = (
    "projection",
    "weak(eigh)",
    "weak(lstsq)",
    "strong(eigh)",
    "strong(lstsq)",
)
OUTPUT_DIR = BASE_OUTPUT_DIR / "ablation" / "embedding"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class CoordinateEmbeddingSpec:
    """Fixed Fourier coordinate embedding configuration."""

    dim: int = 16
    sigma: float = 0.1
    gamma_s: float = 1.5
    gamma_u: float = 1.5
    weak_lambda_bc_scale: float = 1.0
    seed: int = BASE_SEED + 10


@dataclass(frozen=True)
class EmbeddingCase:
    """One embedding toggle used by the ablation."""

    label: str
    enabled: bool


DEFAULT_EMBEDDING_SPEC = CoordinateEmbeddingSpec()
DEFAULT_EMBEDDING_CASES = (
    EmbeddingCase(label="No Embedding", enabled=False),
    EmbeddingCase(label="Embedding", enabled=True),
)


@dataclass(frozen=True)
class LocalFeatureSpace:
    """Local random-feature spaces used only by this ablation."""

    a_s: torch.Tensor
    r_s: torch.Tensor
    a_u: torch.Tensor
    r_u: torch.Tensor
    gamma_s: float
    gamma_u: float
    embedding_B: torch.Tensor | None = None

    @property
    def stress_dim(self) -> int:
        return self.a_s.shape[0] + 1

    @property
    def displacement_dim(self) -> int:
        return self.a_u.shape[0] + 1


@dataclass(frozen=True)
class LocalEvaluationData:
    """All tensors needed to evaluate one local feature run."""

    x_int: torch.Tensor
    f_int: torch.Tensor
    x_bc: torch.Tensor
    w_bc: torch.Tensor
    feature_space: LocalFeatureSpace
    compliance_voigt: torch.Tensor
    assembly_batch_size: int
    xi_s_test: torch.Tensor
    xi_u_test: torch.Tensor
    u_exact_test: torch.Tensor
    sigma_exact_test: torch.Tensor


def clear_experiment_memory() -> None:
    """Release Python references and cached device memory between runs."""

    gc.collect()
    clear_cuda_cache()


def validate_embedding_spec(spec: CoordinateEmbeddingSpec) -> CoordinateEmbeddingSpec:
    """Validate the fixed embedding parameters."""

    if spec.dim <= 0:
        raise ValueError("CoordinateEmbeddingSpec.dim must be positive.")
    if spec.sigma <= 0.0:
        raise ValueError("CoordinateEmbeddingSpec.sigma must be positive.")
    if spec.gamma_s <= 0.0 or spec.gamma_u <= 0.0:
        raise ValueError("CoordinateEmbeddingSpec.gamma_s and gamma_u must be positive.")
    if spec.weak_lambda_bc_scale <= 0.0:
        raise ValueError("CoordinateEmbeddingSpec.weak_lambda_bc_scale must be positive.")
    return CoordinateEmbeddingSpec(
        dim=int(spec.dim),
        sigma=float(spec.sigma),
        gamma_s=float(spec.gamma_s),
        gamma_u=float(spec.gamma_u),
        weak_lambda_bc_scale=float(spec.weak_lambda_bc_scale),
        seed=int(spec.seed),
    )


def validate_feature_only_algorithms(selected_algorithm_ids: Sequence[str]) -> None:
    """Reject algorithms not supported by the local embedding ablation."""

    unsupported_algorithm_ids = [
        algorithm_id
        for algorithm_id in selected_algorithm_ids
        if algorithm_id not in FEATURE_ALGORITHM_IDS
    ]
    if unsupported_algorithm_ids:
        raise ValueError(
            "ablation_embedding only supports feature-based algorithms. "
            f"Unsupported ids: {unsupported_algorithm_ids}"
        )


def resolve_feature_eval_batch_size(
    cfg: MainConfig,
    *,
    projection_enabled: bool,
    weak_enabled: bool,
    strong_enabled: bool,
) -> int:
    """Use the smallest enabled feature-method batch size for residual evaluation."""

    batch_sizes: list[int] = []
    if projection_enabled:
        if cfg.projection is None:
            raise RuntimeError("Projection config is required for feature evaluation.")
        batch_sizes.append(cfg.projection.assembly_batch_size)
    if weak_enabled:
        if cfg.weak is None:
            raise RuntimeError("Weak config is required for feature evaluation.")
        batch_sizes.append(cfg.weak.assembly_batch_size)
    if strong_enabled:
        if cfg.strong is None:
            raise RuntimeError("Strong config is required for feature evaluation.")
        batch_sizes.append(cfg.strong.assembly_batch_size)

    if not batch_sizes:
        raise RuntimeError("At least one feature-based algorithm is required.")
    return min(batch_sizes)


def build_coordinate_embedding(spec: CoordinateEmbeddingSpec) -> torch.Tensor:
    """Sample the fixed Fourier coordinate embedding matrix."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(spec.seed)
    B = spec.sigma * torch.randn(spec.dim, 3, generator=generator, dtype=DTYPE)
    return B.to(device=DEVICE)


def generate_random_features(
    M: int,
    input_dim: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random ridge directions and offsets in the active input space."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    raw = torch.randn(M, input_dim, generator=generator, dtype=DTYPE)
    norms = raw.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
    a = (raw / norms).to(device=DEVICE)
    r = torch.rand(M, generator=generator, dtype=DTYPE).to(device=DEVICE)
    return a, r


def apply_coordinate_embedding(x: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Map x to iota(x) = [cos(2 pi Bx), sin(2 pi Bx)]."""

    phase = 2.0 * math.pi * (x @ B.T)
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=1)


def coordinate_embedding_jacobian(x: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Return J_iota(x) with shape (n_points, 2 * dim, 3)."""

    phase = 2.0 * math.pi * (x @ B.T)
    wave = 2.0 * math.pi * B
    jac_cos = -torch.sin(phase).unsqueeze(2) * wave.unsqueeze(0)
    jac_sin = torch.cos(phase).unsqueeze(2) * wave.unsqueeze(0)
    return torch.cat([jac_cos, jac_sin], dim=1)


def map_feature_inputs(x: torch.Tensor, embedding_B: torch.Tensor | None) -> torch.Tensor:
    """Return the active coordinates used by the ridge basis."""

    if embedding_B is None:
        return x
    return apply_coordinate_embedding(x, embedding_B)


def build_local_feature_space(
    shared_cfg: SharedComparisonConfig,
    case: EmbeddingCase,
    embedding_spec: CoordinateEmbeddingSpec,
) -> LocalFeatureSpace:
    """Create the stress/displacement feature spaces for one embedding case."""

    embedding_B = build_coordinate_embedding(embedding_spec) if case.enabled else None
    input_dim = 3 if embedding_B is None else 2 * embedding_spec.dim

    a_s, r_s = generate_random_features(
        shared_cfg.M_s,
        input_dim,
        shared_cfg.stress_feature_seed,
    )
    a_u, r_u = generate_random_features(
        shared_cfg.M_u,
        input_dim,
        shared_cfg.disp_feature_seed,
    )
    gamma_s = embedding_spec.gamma_s if case.enabled else shared_cfg.gamma_s
    gamma_u = embedding_spec.gamma_u if case.enabled else shared_cfg.gamma_u
    return LocalFeatureSpace(
        a_s=a_s,
        r_s=r_s,
        a_u=a_u,
        r_u=r_u,
        gamma_s=gamma_s,
        gamma_u=gamma_u,
        embedding_B=embedding_B,
    )


def eval_local_features(
    x: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    gamma: float,
    embedding_B: torch.Tensor | None,
) -> torch.Tensor:
    """Evaluate [1, tanh(gamma (a^T z + r))] with z = x or iota(x)."""

    z = map_feature_inputs(x, embedding_B)
    pre = z @ a.T + r.unsqueeze(0)
    xi = torch.tanh(gamma * pre)
    ones = torch.ones(x.shape[0], 1, dtype=DTYPE, device=DEVICE)
    return torch.cat([ones, xi], dim=1)


def eval_local_feature_grads(
    x: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    gamma: float,
    embedding_B: torch.Tensor | None,
) -> torch.Tensor:
    """Evaluate gradients of all local basis functions with respect to x."""

    z = map_feature_inputs(x, embedding_B)
    pre = z @ a.T + r.unsqueeze(0)
    dtanh = 1.0 - torch.tanh(gamma * pre).square()

    if embedding_B is None:
        grad_pre = a.unsqueeze(0).expand(x.shape[0], -1, -1)
    else:
        J_iota = coordinate_embedding_jacobian(x, embedding_B)
        grad_pre = torch.einsum("nkd,mk->nmd", J_iota, a)

    grad_xi = gamma * dtanh.unsqueeze(2) * grad_pre
    zeros = torch.zeros(x.shape[0], 1, 3, dtype=DTYPE, device=DEVICE)
    return torch.cat([zeros, grad_xi], dim=1)


def build_local_evaluation_data(
    benchmark: SharedBenchmarkData,
    feature_space: LocalFeatureSpace,
    assembly_batch_size: int,
) -> LocalEvaluationData:
    """Create the shared evaluation tensors for one embedding case."""

    return LocalEvaluationData(
        x_int=benchmark.x_int,
        f_int=benchmark.f_int,
        x_bc=benchmark.x_bc,
        w_bc=benchmark.w_bc,
        feature_space=feature_space,
        compliance_voigt=benchmark.compliance_voigt,
        assembly_batch_size=assembly_batch_size,
        xi_s_test=eval_local_features(
            benchmark.x_test,
            feature_space.a_s,
            feature_space.r_s,
            feature_space.gamma_s,
            feature_space.embedding_B,
        ),
        xi_u_test=eval_local_features(
            benchmark.x_test,
            feature_space.a_u,
            feature_space.r_u,
            feature_space.gamma_u,
            feature_space.embedding_B,
        ),
        u_exact_test=benchmark.u_exact_test,
        sigma_exact_test=benchmark.sigma_exact_test,
    )


def compute_l2_errors_local(
    xi_u_test: torch.Tensor,
    xi_s_test: torch.Tensor,
    s: torch.Tensor,
    u: torch.Tensor,
    u_exact: torch.Tensor,
    sigma_exact: torch.Tensor,
) -> tuple[float, float]:
    """Compute relative L2 errors for displacement and stress."""

    n_points = xi_u_test.shape[0]
    u_h = torch.zeros(n_points, 3, dtype=DTYPE, device=DEVICE)
    for comp in range(3):
        u_h[:, comp] = xi_u_test @ u[comp::3]

    sigma_h = torch.zeros(n_points, 6, dtype=DTYPE, device=DEVICE)
    for comp in range(6):
        sigma_h[:, comp] = xi_s_test @ s[comp::6]

    voigt_weight = torch.tensor(
        [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        dtype=DTYPE,
        device=DEVICE,
    )

    u_err = torch.sqrt(((u_h - u_exact).square().sum(dim=1)).mean())
    u_ref = torch.sqrt((u_exact.square().sum(dim=1)).mean())
    rel_u = (u_err / u_ref).item() if u_ref > 0 else float("inf")

    sigma_err = torch.sqrt(
        (voigt_weight * (sigma_h - sigma_exact).square()).sum(dim=1).mean()
    )
    sigma_ref = torch.sqrt((voigt_weight * sigma_exact.square()).sum(dim=1).mean())
    rel_sigma = (sigma_err / sigma_ref).item() if sigma_ref > 0 else float("inf")
    return rel_u, rel_sigma


def compute_residual_norms_local(
    data: LocalEvaluationData,
    s: torch.Tensor,
    u: torch.Tensor,
) -> tuple[float, float, float]:
    """Evaluate constitutive, equilibrium, and boundary residual norms."""

    if not torch.isfinite(s).all() or not torch.isfinite(u).all():
        return float("nan"), float("nan"), float("nan")

    fs = data.feature_space
    s_blocks = s.reshape(-1, 6)
    u_blocks = u.reshape(-1, 3)
    constitutive_sq = 0.0
    equilibrium_sq = 0.0
    boundary_sq = 0.0
    w_int = 1.0 / data.x_int.shape[0]

    with torch.no_grad():
        for start in range(0, data.x_int.shape[0], data.assembly_batch_size):
            end = min(start + data.assembly_batch_size, data.x_int.shape[0])
            xb = data.x_int[start:end]
            fb = data.f_int[start:end]

            xi_s_batch = eval_local_features(
                xb,
                fs.a_s,
                fs.r_s,
                fs.gamma_s,
                fs.embedding_B,
            )
            grad_s_batch = eval_local_feature_grads(
                xb,
                fs.a_s,
                fs.r_s,
                fs.gamma_s,
                fs.embedding_B,
            )
            grad_u_batch = eval_local_feature_grads(
                xb,
                fs.a_u,
                fs.r_u,
                fs.gamma_u,
                fs.embedding_B,
            )

            sigma_h = xi_s_batch @ s_blocks

            du_dx1 = grad_u_batch[:, :, 0] @ u_blocks
            du_dx2 = grad_u_batch[:, :, 1] @ u_blocks
            du_dx3 = grad_u_batch[:, :, 2] @ u_blocks
            eps_h = torch.stack(
                [
                    du_dx1[:, 0],
                    du_dx2[:, 1],
                    du_dx3[:, 2],
                    du_dx2[:, 0] + du_dx1[:, 1],
                    du_dx3[:, 1] + du_dx2[:, 2],
                    du_dx3[:, 0] + du_dx1[:, 2],
                ],
                dim=1,
            )

            ds_dx1 = grad_s_batch[:, :, 0] @ s_blocks
            ds_dx2 = grad_s_batch[:, :, 1] @ s_blocks
            ds_dx3 = grad_s_batch[:, :, 2] @ s_blocks
            div_sigma_h = torch.stack(
                [
                    ds_dx1[:, 0] + ds_dx2[:, 3] + ds_dx3[:, 5],
                    ds_dx1[:, 3] + ds_dx2[:, 1] + ds_dx3[:, 4],
                    ds_dx1[:, 5] + ds_dx2[:, 4] + ds_dx3[:, 2],
                ],
                dim=1,
            )

            r_c = sigma_h @ data.compliance_voigt.T - eps_h
            r_e = div_sigma_h + fb
            constitutive_sq += w_int * r_c.square().sum(dim=1).sum().item()
            equilibrium_sq += w_int * r_e.square().sum(dim=1).sum().item()

        for start in range(0, data.x_bc.shape[0], data.assembly_batch_size):
            end = min(start + data.assembly_batch_size, data.x_bc.shape[0])
            xb = data.x_bc[start:end]
            wb = data.w_bc[start:end]
            xi_u_batch = eval_local_features(
                xb,
                fs.a_u,
                fs.r_u,
                fs.gamma_u,
                fs.embedding_B,
            )
            u_bc = xi_u_batch @ u_blocks
            boundary_sq += (wb * u_bc.square().sum(dim=1)).sum().item()

    return constitutive_sq**0.5, equilibrium_sq**0.5, boundary_sq**0.5


def evaluate_local_result(
    name: str,
    wall_time: float,
    s: torch.Tensor,
    u: torch.Tensor,
    data: LocalEvaluationData,
) -> AlgorithmResult:
    """Evaluate one local coefficient-based result."""

    r_c, r_e, r_b = compute_residual_norms_local(data, s, u)
    rel_u, rel_sigma = compute_l2_errors_local(
        data.xi_u_test,
        data.xi_s_test,
        s,
        u,
        data.u_exact_test,
        data.sigma_exact_test,
    )
    return AlgorithmResult(
        name=name,
        r_c=r_c,
        r_e=r_e,
        r_b=r_b,
        rel_u=rel_u,
        rel_sigma=rel_sigma,
        wall_time=wall_time,
    )


def run_projection_case(
    feature_space: LocalFeatureSpace,
    eval_data: LocalEvaluationData,
    x_int: torch.Tensor,
    u_exact_train: torch.Tensor,
    sigma_exact_train: torch.Tensor,
) -> AlgorithmResult:
    """Project exact fields into the local feature spaces."""

    dim_s = feature_space.stress_dim
    dim_u = feature_space.displacement_dim
    s = torch.full((6 * dim_s,), float("nan"), dtype=DTYPE, device=DEVICE)
    u = torch.full((3 * dim_u,), float("nan"), dtype=DTYPE, device=DEVICE)

    synchronize_device()
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            xi_u_train = eval_local_features(
                x_int,
                feature_space.a_u,
                feature_space.r_u,
                feature_space.gamma_u,
                feature_space.embedding_B,
            )
            for comp in range(3):
                u[comp::3] = torch.linalg.lstsq(
                    xi_u_train,
                    u_exact_train[:, comp],
                ).solution

            xi_s_train = eval_local_features(
                x_int,
                feature_space.a_s,
                feature_space.r_s,
                feature_space.gamma_s,
                feature_space.embedding_B,
            )
            for comp in range(6):
                s[comp::6] = torch.linalg.lstsq(
                    xi_s_train,
                    sigma_exact_train[:, comp],
                ).solution
    except (RuntimeError, torch.linalg.LinAlgError) as exc:
        print(f"    Warning: projection solve failed with {type(exc).__name__}")

    synchronize_device()
    wall_time = time.perf_counter() - t0
    result = evaluate_local_result("Projection", wall_time, s, u, eval_data)
    print_result_summary(result)
    return result


def accumulate_boundary_gram_local(
    x_bc: torch.Tensor,
    w_bc: torch.Tensor,
    a_u: torch.Tensor,
    r_u: torch.Tensor,
    gamma_u: float,
    embedding_B: torch.Tensor | None,
    batch_size: int,
) -> torch.Tensor:
    """Accumulate the weighted boundary Gram matrix for the local xi_u."""

    mp1_u = a_u.shape[0] + 1
    gram_bc = torch.zeros(mp1_u, mp1_u, dtype=DTYPE, device=DEVICE)

    with torch.no_grad():
        for start in range(0, x_bc.shape[0], batch_size):
            end = min(start + batch_size, x_bc.shape[0])
            xb = x_bc[start:end]
            wb = w_bc[start:end]
            xi_u_batch = eval_local_features(xb, a_u, r_u, gamma_u, embedding_B)
            gram_bc += xi_u_batch.T @ (wb.unsqueeze(1) * xi_u_batch)

    return gram_bc


def accumulate_weak_form_moments_local(
    x_int: torch.Tensor,
    f_int: torch.Tensor,
    feature_space: LocalFeatureSpace,
    batch_size: int,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Accumulate local interior moments for the weak-form system."""

    mp1_s = feature_space.stress_dim
    mp1_u = feature_space.displacement_dim
    weight = 1.0 / x_int.shape[0]

    gram_s = torch.zeros(mp1_s, mp1_s, dtype=DTYPE, device=DEVICE)
    cross_u_grad_s = [
        torch.zeros(mp1_u, mp1_s, dtype=DTYPE, device=DEVICE) for _ in range(3)
    ]
    force_moment = torch.zeros(mp1_u, 3, dtype=DTYPE, device=DEVICE)

    with torch.no_grad():
        for start in range(0, x_int.shape[0], batch_size):
            end = min(start + batch_size, x_int.shape[0])
            xb = x_int[start:end]
            fb = f_int[start:end]

            xi_s_batch = eval_local_features(
                xb,
                feature_space.a_s,
                feature_space.r_s,
                feature_space.gamma_s,
                feature_space.embedding_B,
            )
            grad_s_batch = eval_local_feature_grads(
                xb,
                feature_space.a_s,
                feature_space.r_s,
                feature_space.gamma_s,
                feature_space.embedding_B,
            )
            xi_u_batch = eval_local_features(
                xb,
                feature_space.a_u,
                feature_space.r_u,
                feature_space.gamma_u,
                feature_space.embedding_B,
            )

            gram_s += weight * (xi_s_batch.T @ xi_s_batch)
            force_moment += weight * (xi_u_batch.T @ fb)
            for dim in range(3):
                cross_u_grad_s[dim] += weight * (
                    xi_u_batch.T @ grad_s_batch[:, :, dim]
                )

    return gram_s, cross_u_grad_s, force_moment


def run_weak_case(
    weak_cfg,
    benchmark: SharedBenchmarkData,
    feature_space: LocalFeatureSpace,
    eval_data: LocalEvaluationData,
    weak_algorithm_ids: Sequence[str],
) -> dict[str, AlgorithmResult]:
    """Run the selected weak-form solvers with local basis evaluation."""

    gram_s, cross_u_grad_s, force_moment = accumulate_weak_form_moments_local(
        benchmark.x_int,
        benchmark.f_int,
        feature_space,
        weak_cfg.assembly_batch_size,
    )
    gram_bc = accumulate_boundary_gram_local(
        benchmark.x_bc,
        benchmark.w_bc,
        feature_space.a_u,
        feature_space.r_u,
        feature_space.gamma_u,
        feature_space.embedding_B,
        weak_cfg.assembly_batch_size,
    )
    A, B, C, F = assemble_weak_system(
        gram_s,
        cross_u_grad_s,
        gram_bc,
        force_moment,
        benchmark.compliance_voigt,
        weak_cfg.lambda_bc,
    )

    del gram_s
    del cross_u_grad_s
    del force_moment
    del gram_bc
    clear_cuda_cache()

    results: dict[str, AlgorithmResult] = {}
    try:
        for algorithm_id in weak_algorithm_ids:
            if algorithm_id == "eigh":
                print("Running Weak (Eigh)...")
                s, u, wall_time = solve_weak_eigh(A, B, C, F, weak_cfg.eigh_rtol)
                label = "Weak (Eigh)"
            else:
                print("Running Weak (Lstsq)...")
                s, u, wall_time = solve_weak_lstsq(A, B, C, F)
                label = "Weak (Lstsq)"
            result = evaluate_local_result(label, wall_time, s, u, eval_data)
            print_result_summary(result)
            results[label] = result
    finally:
        del A
        del B
        del C
        del F
        clear_cuda_cache()

    return results


def accumulate_interior_moments_local(
    x_int: torch.Tensor,
    f_int: torch.Tensor,
    feature_space: LocalFeatureSpace,
    batch_size: int,
) -> tuple[
    torch.Tensor,
    list[torch.Tensor],
    list[list[torch.Tensor]],
    list[list[torch.Tensor]],
    list[torch.Tensor],
]:
    """Accumulate local moments for the strong-form normal equations."""

    mp1_s = feature_space.stress_dim
    mp1_u = feature_space.displacement_dim
    weight = 1.0 / x_int.shape[0]

    gram_xi_s = torch.zeros(mp1_s, mp1_s, dtype=DTYPE, device=DEVICE)
    cross_xi_grad_u = [
        torch.zeros(mp1_s, mp1_u, dtype=DTYPE, device=DEVICE) for _ in range(3)
    ]
    grad_gram_u = [
        [
            torch.zeros(mp1_u, mp1_u, dtype=DTYPE, device=DEVICE)
            for _ in range(3)
        ]
        for _ in range(3)
    ]
    grad_gram_s = [
        [
            torch.zeros(mp1_s, mp1_s, dtype=DTYPE, device=DEVICE)
            for _ in range(3)
        ]
        for _ in range(3)
    ]
    grad_force_s = [
        torch.zeros(mp1_s, 3, dtype=DTYPE, device=DEVICE) for _ in range(3)
    ]

    with torch.no_grad():
        for start in range(0, x_int.shape[0], batch_size):
            end = min(start + batch_size, x_int.shape[0])
            xb = x_int[start:end]
            fb = f_int[start:end]

            xi_s_batch = eval_local_features(
                xb,
                feature_space.a_s,
                feature_space.r_s,
                feature_space.gamma_s,
                feature_space.embedding_B,
            )
            grad_s_batch = eval_local_feature_grads(
                xb,
                feature_space.a_s,
                feature_space.r_s,
                feature_space.gamma_s,
                feature_space.embedding_B,
            )
            grad_u_batch = eval_local_feature_grads(
                xb,
                feature_space.a_u,
                feature_space.r_u,
                feature_space.gamma_u,
                feature_space.embedding_B,
            )

            gram_xi_s += weight * (xi_s_batch.T @ xi_s_batch)
            for dim_i in range(3):
                cross_xi_grad_u[dim_i] += weight * (
                    xi_s_batch.T @ grad_u_batch[:, :, dim_i]
                )
                grad_force_s[dim_i] += weight * (
                    grad_s_batch[:, :, dim_i].T @ fb
                )
                for dim_j in range(3):
                    grad_gram_u[dim_i][dim_j] += weight * (
                        grad_u_batch[:, :, dim_i].T @ grad_u_batch[:, :, dim_j]
                    )
                    grad_gram_s[dim_i][dim_j] += weight * (
                        grad_s_batch[:, :, dim_i].T @ grad_s_batch[:, :, dim_j]
                    )

    return gram_xi_s, cross_xi_grad_u, grad_gram_u, grad_gram_s, grad_force_s


def assemble_normal_equations_local(
    strong_cfg,
    compliance_voigt: torch.Tensor,
    benchmark: SharedBenchmarkData,
    feature_space: LocalFeatureSpace,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Assemble the strong-form normal equations for one local feature space."""

    (
        gram_xi_s,
        cross_xi_grad_u,
        grad_gram_u,
        grad_gram_s,
        grad_force_s,
    ) = accumulate_interior_moments_local(
        benchmark.x_int,
        benchmark.f_int,
        feature_space,
        strong_cfg.assembly_batch_size,
    )
    gram_bc_u = accumulate_boundary_gram_local(
        benchmark.x_bc,
        benchmark.w_bc,
        feature_space.a_u,
        feature_space.r_u,
        feature_space.gamma_u,
        feature_space.embedding_B,
        strong_cfg.assembly_batch_size,
    )

    mp1_s = feature_space.stress_dim
    mp1_u = feature_space.displacement_dim
    dim_s = 6 * mp1_s
    dim_u = 3 * mp1_u

    H_ss = torch.zeros(dim_s, dim_s, dtype=DTYPE, device=DEVICE)
    H_su = torch.zeros(dim_s, dim_u, dtype=DTYPE, device=DEVICE)
    H_uu = torch.zeros(dim_u, dim_u, dtype=DTYPE, device=DEVICE)
    g_s = torch.zeros(dim_s, dtype=DTYPE, device=DEVICE)

    compliance_sq = compliance_voigt.T @ compliance_voigt
    add_block_scaled(H_ss, gram_xi_s, compliance_sq, row_stride=6, col_stride=6)

    for dim_i in range(3):
        constitutive_cross = compliance_voigt @ STRAIN_GRAD_BASES[dim_i]
        add_block_scaled(
            H_su,
            cross_xi_grad_u[dim_i],
            -constitutive_cross,
            row_stride=6,
            col_stride=3,
        )
        add_rhs_feature_blocks(
            g_s,
            grad_force_s[dim_i],
            STRAIN_GRAD_BASES[dim_i],
            block_size=6,
            scale=-1.0,
        )

        for dim_j in range(3):
            constitutive_uu = STRAIN_GRAD_BASES[dim_i].T @ STRAIN_GRAD_BASES[dim_j]
            equilibrium_ss = STRAIN_GRAD_BASES[dim_i] @ STRAIN_GRAD_BASES[dim_j].T
            add_block_scaled(
                H_uu,
                grad_gram_u[dim_i][dim_j],
                constitutive_uu,
                row_stride=3,
                col_stride=3,
            )
            add_block_scaled(
                H_ss,
                grad_gram_s[dim_i][dim_j],
                equilibrium_ss,
                row_stride=6,
                col_stride=6,
            )

    add_block_scaled(
        H_uu,
        gram_bc_u,
        strong_cfg.lambda_bc * IDENTITY_3,
        row_stride=3,
        col_stride=3,
    )

    H = torch.zeros(dim_s + dim_u, dim_s + dim_u, dtype=DTYPE, device=DEVICE)
    H[:dim_s, :dim_s] = H_ss
    H[:dim_s, dim_s:] = H_su
    H[dim_s:, :dim_s] = H_su.T
    H[dim_s:, dim_s:] = H_uu
    H = 0.5 * (H + H.T)

    g = torch.zeros(dim_s + dim_u, dtype=DTYPE, device=DEVICE)
    g[:dim_s] = g_s
    return H, g, dim_s


def run_strong_case(
    strong_cfg,
    benchmark: SharedBenchmarkData,
    feature_space: LocalFeatureSpace,
    eval_data: LocalEvaluationData,
    strong_algorithm_ids: Sequence[str],
) -> dict[str, AlgorithmResult]:
    """Run the selected strong-form solvers with local basis evaluation."""

    H, g, dim_s = assemble_normal_equations_local(
        strong_cfg,
        benchmark.compliance_voigt,
        benchmark,
        feature_space,
    )
    clear_cuda_cache()

    results: dict[str, AlgorithmResult] = {}
    try:
        for algorithm_id in strong_algorithm_ids:
            if algorithm_id == "eigh":
                print("Running Strong (Eigh)...")
                z, wall_time = solve_strong_eigh(H, g, strong_cfg.eigh_rtol)
                label = "Strong (Eigh)"
            else:
                print("Running Strong (Lstsq)...")
                z, wall_time = solve_strong_lstsq(H, g)
                label = "Strong (Lstsq)"
            s, u = split_strong_solution(z, dim_s)
            result = evaluate_local_result(label, wall_time, s, u, eval_data)
            print_result_summary(result)
            results[label] = result
    finally:
        del H
        del g
        clear_cuda_cache()

    return results


def plot_ablation_embedding(
    results: dict[str, dict[str, AlgorithmResult]],
    ordered_labels: Sequence[str],
    save_path: str,
) -> None:
    """Plot grouped bar charts for no-embedding versus embedding."""

    if not results:
        return

    configure_plotting()
    case_labels = list(results.keys())
    x_positions = np.arange(len(ordered_labels), dtype=float)
    bar_width = 0.34
    offsets = (
        np.arange(len(case_labels), dtype=float) - 0.5 * (len(case_labels) - 1)
    ) * bar_width

    fig_width = max(10.0, 1.4 * len(ordered_labels) + 2.4)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 4.8))
    metric_specs = [
        (
            "rel_u",
            r"Displacement $\|u_M - u_{ex}\|_{L^2} / \|u_{ex}\|_{L^2}$",
            "Relative $L^2$ error",
        ),
        (
            "rel_sigma",
            r"Stress $\|\sigma_M - \sigma_{ex}\|_{L^2} / \|\sigma_{ex}\|_{L^2}$",
            "Relative $L^2$ error",
        ),
    ]

    for ax, (metric_name, title, ylabel) in zip(axes, metric_specs):
        for case_index, case_label in enumerate(case_labels):
            values = np.array(
                [
                    getattr(results[case_label][label], metric_name)
                    if label in results[case_label]
                    else float("nan")
                    for label in ordered_labels
                ],
                dtype=float,
            )
            valid = np.isfinite(values) & (values > 0.0)
            if valid.any():
                valid_indices = np.flatnonzero(valid)
                colors = [
                    ALGO_STYLE.get(ordered_labels[index], {}).get("color", "#4C78A8")
                    for index in valid_indices
                ]
                if case_label == "Embedding":
                    ax.bar(
                        x_positions[valid] + offsets[case_index],
                        values[valid],
                        width=bar_width,
                        color=colors,
                        edgecolor="#1F1F1F",
                        linewidth=1.0,
                        hatch="//",
                    )
                else:
                    ax.bar(
                        x_positions[valid] + offsets[case_index],
                        values[valid],
                        width=bar_width,
                        color=colors,
                    )

            invalid_indices = np.flatnonzero(~valid)
            for invalid_index in invalid_indices:
                print(
                    f"  Skipped {case_label} {ordered_labels[invalid_index]} "
                    f"{metric_name}={values[invalid_index]!r} in {save_path}"
                )

        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Algorithm")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(ordered_labels, rotation=15, ha="right")
        ax.grid(alpha=0.3, linestyle="--", axis="y")
        ax.legend(
            handles=[
                Patch(
                    facecolor="#BDBDBD",
                    edgecolor="#BDBDBD",
                    label="No Embedding",
                ),
                Patch(
                    facecolor="white",
                    edgecolor="#1F1F1F",
                    hatch="//",
                    label="Embedding",
                ),
            ]
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def print_ablation_embedding_summary_table(
    case_labels: Sequence[str],
    results: dict[str, dict[str, AlgorithmResult]],
    ordered_labels: Sequence[str],
) -> None:
    """Print the final comparable summary table."""

    print("\n=== Ablation Embedding Summary ===\n")
    print(
        f"| {'Case':<14} | {'Method':<16} | "
        f"{'rel_u':>12} | {'rel_sigma':>12} | {'Time(s)':>8} |"
    )
    print(
        f"|:{'-' * 15}|:{'-' * 17}|"
        f"{'-' * 13}:|{'-' * 13}:|{'-' * 9}:|"
    )

    for case_label in case_labels:
        case_results = results[case_label]
        for algorithm_label in ordered_labels:
            if algorithm_label not in case_results:
                continue
            result = case_results[algorithm_label]
            print(
                f"| {case_label:<14} | {algorithm_label:<16} | "
                f"{result.rel_u:12.2e} | {result.rel_sigma:12.2e} | {result.wall_time:8.2f} |"
            )


def run_ablation(
    cfg: MainConfig | None = None,
    embedding_spec: CoordinateEmbeddingSpec | None = None,
) -> dict[str, dict[str, AlgorithmResult]]:
    """Run the local embedding ablation and return all metrics."""

    cfg = make_default_main_config() if cfg is None else cfg
    shared_cfg = SharedComparisonConfig() if cfg.shared is None else cfg.shared
    validate_shared_comparison_config(shared_cfg)

    selected_algorithm_ids = validate_algorithm_selection(
        cfg.algorithms_to_run,
        VALID_TOP_LEVEL_ALGORITHMS,
    )
    validate_feature_only_algorithms(selected_algorithm_ids)
    ordered_labels = [
        TOP_LEVEL_ALGORITHM_LABELS[algorithm_id]
        for algorithm_id in selected_algorithm_ids
    ]

    embedding_spec = (
        DEFAULT_EMBEDDING_SPEC if embedding_spec is None else embedding_spec
    )
    embedding_spec = validate_embedding_spec(embedding_spec)

    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Algorithms: {selected_algorithm_ids}")
    print(
        "Embedding spec: "
        f"dim={embedding_spec.dim}, sigma={embedding_spec.sigma}, "
        f"gamma_s={embedding_spec.gamma_s}, gamma_u={embedding_spec.gamma_u}, "
        f"weak_lambda_bc_scale={embedding_spec.weak_lambda_bc_scale}, "
        f"seed={embedding_spec.seed}"
    )

    print("Building shared benchmark data...")
    benchmark = build_shared_benchmark(
        E=shared_cfg.E,
        nu=shared_cfg.nu,
        Q_int=shared_cfg.Q_int,
        Q_bc=shared_cfg.Q_bc,
        Q_test=shared_cfg.Q_test,
        sampling_method=shared_cfg.sampling_method,
        body_force_batch_size=shared_cfg.body_force_batch_size,
        interior_seed=shared_cfg.interior_seed,
        boundary_seed=shared_cfg.boundary_seed,
        test_seed=shared_cfg.test_seed,
    )

    projection_enabled = "projection" in selected_algorithm_ids
    weak_algorithm_ids = extract_scoped_algorithm_ids(selected_algorithm_ids, "weak")
    strong_algorithm_ids = extract_scoped_algorithm_ids(selected_algorithm_ids, "strong")
    feature_eval_batch_size = resolve_feature_eval_batch_size(
        cfg,
        projection_enabled=projection_enabled,
        weak_enabled=bool(weak_algorithm_ids),
        strong_enabled=bool(strong_algorithm_ids),
    )

    if projection_enabled:
        if cfg.projection is None:
            raise ValueError("MainConfig.projection is required when running projection.")
        projection_cfg = apply_shared_to_projection_config(cfg.projection, shared_cfg)
        validate_projection_config(projection_cfg)
        mu, lam = compute_lame_constants(shared_cfg.E, shared_cfg.nu)
        print(f"Computing exact projection targets with mu={mu:.4f}, lam={lam:.4f}...")
        u_exact_train = eval_exact_displacement(benchmark.x_int)
        sigma_exact_train = compute_stress_voigt(benchmark.x_int, mu, lam)
    else:
        projection_cfg = None
        u_exact_train = None
        sigma_exact_train = None

    if weak_algorithm_ids:
        if cfg.weak is None:
            raise ValueError("MainConfig.weak is required when running weak-form algorithms.")
        weak_cfg = apply_shared_to_weak_config(cfg.weak, shared_cfg, weak_algorithm_ids)
        validate_weak_config(weak_cfg)
    else:
        weak_cfg = None

    if strong_algorithm_ids:
        if cfg.strong is None:
            raise ValueError(
                "MainConfig.strong is required when running strong-form algorithms."
            )
        strong_cfg = apply_shared_to_strong_config(cfg.strong, shared_cfg, strong_algorithm_ids)
        validate_strong_config(strong_cfg)
    else:
        strong_cfg = None

    all_results: dict[str, dict[str, AlgorithmResult]] = {}
    for case in DEFAULT_EMBEDDING_CASES:
        print(f"\n{'=' * 72}")
        print(f"=== Ablation Embedding: {case.label} ===")
        print(f"{'=' * 72}")

        feature_space = build_local_feature_space(shared_cfg, case, embedding_spec)
        eval_data = build_local_evaluation_data(
            benchmark,
            feature_space,
            feature_eval_batch_size,
        )
        case_results: dict[str, AlgorithmResult] = {}

        if projection_enabled:
            if projection_cfg is None or u_exact_train is None or sigma_exact_train is None:
                raise RuntimeError("Projection targets must be precomputed.")
            print("Running Projection...")
            case_results["Projection"] = run_projection_case(
                feature_space,
                eval_data,
                benchmark.x_int,
                u_exact_train,
                sigma_exact_train,
            )

        if weak_algorithm_ids:
            if weak_cfg is None:
                raise RuntimeError("Weak config must be prepared before running the ablation.")
            case_weak_cfg = (
                replace(
                    weak_cfg,
                    lambda_bc=weak_cfg.lambda_bc * embedding_spec.weak_lambda_bc_scale,
                )
                if case.enabled
                else weak_cfg
            )
            case_results.update(
                run_weak_case(
                    case_weak_cfg,
                    benchmark,
                    feature_space,
                    eval_data,
                    weak_algorithm_ids,
                )
            )

        if strong_algorithm_ids:
            if strong_cfg is None:
                raise RuntimeError("Strong config must be prepared before running the ablation.")
            case_results.update(
                run_strong_case(
                    strong_cfg,
                    benchmark,
                    feature_space,
                    eval_data,
                    strong_algorithm_ids,
                )
            )

        all_results[case.label] = case_results
        del feature_space
        del eval_data
        clear_experiment_memory()

    if all_results:
        print("\nGenerating plots...")
        plot_ablation_embedding(
            all_results,
            ordered_labels,
            str(OUTPUT_DIR / "ablation-embedding.png"),
        )
        print_ablation_embedding_summary_table(
            list(all_results.keys()),
            all_results,
            ordered_labels,
        )

    return all_results


def main(
    cfg: MainConfig | None = None,
    embedding_spec: CoordinateEmbeddingSpec | None = None,
) -> None:
    """Script entrypoint."""

    run_ablation(cfg=cfg, embedding_spec=embedding_spec)


if __name__ == "__main__":
    main()
