from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass
from typing import Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from linear_elasticity_3d import (
    ALGO_STYLE,
    DEVICE,
    DISP_SEED,
    DTYPE,
    OUTPUT_DIR as BASE_OUTPUT_DIR,
    STRESS_SEED,
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
from strong_form import validate_config as validate_strong_config
from weak_form import validate_config as validate_weak_config

FeatureKind = Literal["tanh_ridge", "fourier", "multiscale_fourier"]
FEATURE_ALGORITHM_IDS = (
    "projection",
    "weak(eigh)",
    "weak(lstsq)",
    "strong(eigh)",
    "strong(lstsq)",
)
DEFAULT_CONDITION_PROBE_SIZE = 2_048
OUTPUT_DIR = BASE_OUTPUT_DIR / "ablation" / "feature"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class FourierBand:
    """One fixed Fourier band."""

    sigma: float
    B: torch.Tensor


@dataclass(frozen=True)
class FeatureSet:
    """A fixed feature family used by one field."""

    kind: FeatureKind
    scalar_feature_count: int
    gamma: float | None = None
    a: torch.Tensor | None = None
    r: torch.Tensor | None = None
    bands: tuple[FourierBand, ...] = ()

    @property
    def feature_dim(self) -> int:
        return self.scalar_feature_count + 1

    @property
    def sigmas(self) -> tuple[float, ...]:
        return tuple(band.sigma for band in self.bands)


@dataclass(frozen=True)
class SharedFeatureSets:
    """Stress and displacement feature spaces."""

    stress: FeatureSet
    displacement: FeatureSet


@dataclass(frozen=True)
class FeatureMatrixStats:
    """Compact conditioning diagnostics."""

    sigma_max: float
    sigma_min: float
    cond: float


@dataclass(frozen=True)
class FeatureAblationCase:
    """One comparable feature-generation setup."""

    label: str
    kind: FeatureKind
    sigma_bands_s: tuple[float, ...] = ()
    sigma_bands_u: tuple[float, ...] = ()
    band_weights_s: tuple[float, ...] = ()
    band_weights_u: tuple[float, ...] = ()


DEFAULT_ABLATION_FEATURE_LIST = [
    FeatureAblationCase(
        label="Tanh Ridge",
        kind="tanh_ridge",
    ),
    FeatureAblationCase(
        label="Fourier",
        kind="fourier",
        sigma_bands_s=(0.5,),
        sigma_bands_u=(0.5,),
    ),
    FeatureAblationCase(
        label="Multiscale Fourier",
        kind="multiscale_fourier",
        sigma_bands_s=(0.5, 1.0),
        sigma_bands_u=(0.5, 1.0),
        band_weights_s=(0.5, 1.0),
        band_weights_u=(0.5, 1.0),
    ),
]


@dataclass(frozen=True)
class FeatureFamilyConfig:
    """Configuration for the fixed feature families."""

    kind: FeatureKind
    M_s: int
    M_u: int
    gamma_s: float
    gamma_u: float
    sigma_bands_s: tuple[float, ...] = ()
    sigma_bands_u: tuple[float, ...] = ()
    band_weights_s: tuple[float, ...] = ()
    band_weights_u: tuple[float, ...] = ()
    stress_feature_seed: int = STRESS_SEED
    disp_feature_seed: int = DISP_SEED


@dataclass(frozen=True)
class FamilyEvaluationData:
    """All tensors needed to evaluate coefficient-based methods."""

    x_int: torch.Tensor
    f_int: torch.Tensor
    x_bc: torch.Tensor
    w_bc: torch.Tensor
    stress_features: FeatureSet
    displacement_features: FeatureSet
    compliance_voigt: torch.Tensor
    assembly_batch_size: int
    xi_s_test: torch.Tensor
    xi_u_test: torch.Tensor
    u_exact_test: torch.Tensor
    sigma_exact_test: torch.Tensor


@dataclass(frozen=True)
class WeakExperimentData:
    """Weak-form linear system plus evaluation data."""

    A: torch.Tensor
    B: torch.Tensor
    C: torch.Tensor
    F: torch.Tensor
    eval_data: FamilyEvaluationData


@dataclass(frozen=True)
class StrongExperimentData:
    """Strong-form normal equations plus evaluation data."""

    H: torch.Tensor
    g: torch.Tensor
    dim_s: int
    eval_data: FamilyEvaluationData


def build_strain_gradient_bases() -> torch.Tensor:
    """Return the three fixed gradient-to-strain coupling blocks."""

    base_1 = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )
    base_2 = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )
    base_3 = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )
    return torch.stack([base_1, base_2, base_3], dim=0)


STRAIN_GRAD_BASES = build_strain_gradient_bases()
IDENTITY_3 = torch.eye(3, dtype=DTYPE, device=DEVICE)


def clear_experiment_memory() -> None:
    """Release Python references and cached device memory between runs."""

    gc.collect()
    clear_cuda_cache()


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


def build_feature_family_config(
    case: FeatureAblationCase,
    shared_cfg: SharedComparisonConfig,
) -> FeatureFamilyConfig:
    """Lift one feature case into a fully specified runtime config."""

    return FeatureFamilyConfig(
        kind=case.kind,
        M_s=shared_cfg.M_s,
        M_u=shared_cfg.M_u,
        gamma_s=shared_cfg.gamma_s,
        gamma_u=shared_cfg.gamma_u,
        sigma_bands_s=case.sigma_bands_s,
        sigma_bands_u=case.sigma_bands_u,
        band_weights_s=case.band_weights_s,
        band_weights_u=case.band_weights_u,
        stress_feature_seed=shared_cfg.stress_feature_seed,
        disp_feature_seed=shared_cfg.disp_feature_seed,
    )


def validate_feature_family_config(cfg: FeatureFamilyConfig) -> None:
    """Validate feature-family parameters."""

    if cfg.kind not in ("tanh_ridge", "fourier", "multiscale_fourier"):
        raise ValueError(f"Unknown feature family kind={cfg.kind!r}.")
    if cfg.M_s <= 0 or cfg.M_u <= 0:
        raise ValueError("FeatureFamilyConfig.M_s and M_u must be positive.")
    if cfg.gamma_s <= 0.0 or cfg.gamma_u <= 0.0:
        raise ValueError("FeatureFamilyConfig.gamma_s and gamma_u must be positive.")
    if cfg.kind == "tanh_ridge":
        return

    for name, value in (("M_s", cfg.M_s), ("M_u", cfg.M_u)):
        if value % 2 != 0:
            raise ValueError(
                f"FeatureFamilyConfig.{name} must be even for Fourier-based features."
            )

    for name, sigmas, weights in (
        ("stress", cfg.sigma_bands_s, cfg.band_weights_s),
        ("displacement", cfg.sigma_bands_u, cfg.band_weights_u),
    ):
        if not sigmas:
            raise ValueError(f"{name} sigma bands must be non-empty.")
        if any(sigma <= 0.0 for sigma in sigmas):
            raise ValueError(f"{name} sigma bands must be strictly positive.")
        if cfg.kind == "fourier":
            if len(sigmas) != 1:
                raise ValueError(f"{name} single-scale Fourier expects exactly one sigma.")
            continue

        if len(sigmas) < 2:
            raise ValueError(
                f"{name} multiscale Fourier expects at least two sigma bands."
            )
        if len(sigmas) != len(weights):
            raise ValueError(
                f"{name} sigma bands and band weights must have identical length."
            )
        if any(weight <= 0.0 for weight in weights):
            raise ValueError(f"{name} band weights must be strictly positive.")


def validate_feature_case_list(
    feature_case_list: Sequence[FeatureAblationCase],
    shared_cfg: SharedComparisonConfig,
) -> list[FeatureAblationCase]:
    """Validate the feature-ablation cases against the shared config."""

    if not feature_case_list:
        raise ValueError("feature_case_list must be non-empty.")

    validated_cases: list[FeatureAblationCase] = []
    seen_labels: set[str] = set()
    for case in feature_case_list:
        if not case.label:
            raise ValueError("Each feature case must define a non-empty label.")
        if case.label in seen_labels:
            raise ValueError(f"Duplicate feature case label: {case.label!r}")
        seen_labels.add(case.label)
        validate_feature_family_config(build_feature_family_config(case, shared_cfg))
        validated_cases.append(case)

    return validated_cases


def resolve_multiscale_band_counts(
    scalar_feature_count: int,
    band_weights: Sequence[float],
) -> list[int]:
    """Convert scalar-feature budget to wave-vector counts per band."""

    if scalar_feature_count % 2 != 0:
        raise ValueError("Fourier-based scalar_feature_count must be even.")

    total_vectors = scalar_feature_count // 2
    n_bands = len(band_weights)
    if total_vectors < n_bands:
        raise ValueError(
            "Need at least one Fourier wave vector per band. "
            f"Got total_vectors={total_vectors}, n_bands={n_bands}."
        )

    weight_tensor = torch.tensor(band_weights, dtype=DTYPE)
    normalized = weight_tensor / weight_tensor.sum()
    base_counts = [1] * n_bands
    remaining = total_vectors - n_bands
    if remaining == 0:
        return base_counts

    raw_extra = normalized * remaining
    extra_floor = torch.floor(raw_extra).to(dtype=torch.int64)
    counts = [base_counts[i] + int(extra_floor[i].item()) for i in range(n_bands)]
    leftovers = remaining - int(extra_floor.sum().item())
    if leftovers > 0:
        fractional = raw_extra - extra_floor.to(dtype=DTYPE)
        order = torch.argsort(fractional, descending=True).tolist()
        for index in order[:leftovers]:
            counts[index] += 1

    return counts


def generate_random_hyperplanes(
    scalar_feature_count: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random unit normals and offsets for tanh ridge features."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    raw = torch.randn(scalar_feature_count, 3, generator=generator, dtype=DTYPE)
    norms = raw.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
    a = (raw / norms).to(device=DEVICE)
    r = torch.rand(scalar_feature_count, generator=generator, dtype=DTYPE).to(DEVICE)
    return a, r


def generate_fourier_vectors(
    vector_count: int,
    sigma: float,
    seed: int,
) -> torch.Tensor:
    """Sample fixed Fourier wave vectors."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    B = sigma * torch.randn(vector_count, 3, generator=generator, dtype=DTYPE)
    return B.to(device=DEVICE)


def generate_feature_set(
    kind: FeatureKind,
    scalar_feature_count: int,
    gamma: float,
    sigma_bands: Sequence[float],
    band_weights: Sequence[float],
    seed: int,
) -> FeatureSet:
    """Create one fixed feature family."""

    if kind == "tanh_ridge":
        a, r = generate_random_hyperplanes(scalar_feature_count, seed=seed)
        return FeatureSet(
            kind=kind,
            scalar_feature_count=scalar_feature_count,
            gamma=gamma,
            a=a,
            r=r,
        )

    if scalar_feature_count % 2 != 0:
        raise ValueError("Fourier-based scalar_feature_count must be even.")

    if kind == "fourier":
        vector_count = scalar_feature_count // 2
        sigma = float(sigma_bands[0])
        return FeatureSet(
            kind=kind,
            scalar_feature_count=scalar_feature_count,
            bands=(FourierBand(sigma=sigma, B=generate_fourier_vectors(vector_count, sigma, seed)),),
        )

    counts = resolve_multiscale_band_counts(scalar_feature_count, band_weights)
    bands: list[FourierBand] = []
    for band_index, (sigma, vector_count) in enumerate(zip(sigma_bands, counts)):
        bands.append(
            FourierBand(
                sigma=float(sigma),
                B=generate_fourier_vectors(
                    vector_count=vector_count,
                    sigma=float(sigma),
                    seed=seed + 1_000 * band_index,
                ),
            )
        )
    return FeatureSet(
        kind=kind,
        scalar_feature_count=scalar_feature_count,
        bands=tuple(bands),
    )


def build_shared_feature_sets(cfg: FeatureFamilyConfig) -> SharedFeatureSets:
    """Create the stress/displacement feature spaces."""

    return SharedFeatureSets(
        stress=generate_feature_set(
            kind=cfg.kind,
            scalar_feature_count=cfg.M_s,
            gamma=cfg.gamma_s,
            sigma_bands=cfg.sigma_bands_s,
            band_weights=cfg.band_weights_s,
            seed=cfg.stress_feature_seed,
        ),
        displacement=generate_feature_set(
            kind=cfg.kind,
            scalar_feature_count=cfg.M_u,
            gamma=cfg.gamma_u,
            sigma_bands=cfg.sigma_bands_u,
            band_weights=cfg.band_weights_u,
            seed=cfg.disp_feature_seed,
        ),
    )


def eval_features(x: torch.Tensor, feature_set: FeatureSet) -> torch.Tensor:
    """Evaluate all basis functions, including the constant mode."""

    ones = torch.ones(x.shape[0], 1, dtype=DTYPE, device=DEVICE)
    if feature_set.kind == "tanh_ridge":
        assert feature_set.a is not None and feature_set.r is not None
        assert feature_set.gamma is not None
        pre = x @ feature_set.a.T + feature_set.r.unsqueeze(0)
        xi = torch.tanh(feature_set.gamma * pre)
        return torch.cat([ones, xi], dim=1)

    parts: list[torch.Tensor] = [ones]
    scale = 2.0 * math.pi
    for band in feature_set.bands:
        phase = scale * (x @ band.B.T)
        parts.append(torch.cos(phase))
        parts.append(torch.sin(phase))
    features = torch.cat(parts, dim=1)
    if features.shape[1] != feature_set.feature_dim:
        raise RuntimeError(
            f"Feature dimension mismatch: expected {feature_set.feature_dim}, "
            f"got {features.shape[1]}."
        )
    return features


def eval_feature_grads(x: torch.Tensor, feature_set: FeatureSet) -> torch.Tensor:
    """Evaluate gradients of all basis functions."""

    zeros = torch.zeros(x.shape[0], 1, 3, dtype=DTYPE, device=DEVICE)
    if feature_set.kind == "tanh_ridge":
        assert feature_set.a is not None and feature_set.r is not None
        assert feature_set.gamma is not None
        pre = x @ feature_set.a.T + feature_set.r.unsqueeze(0)
        dtanh = 1.0 - torch.tanh(feature_set.gamma * pre).square()
        grad_xi = feature_set.gamma * dtanh.unsqueeze(2) * feature_set.a.unsqueeze(0)
        return torch.cat([zeros, grad_xi], dim=1)

    scale = 2.0 * math.pi
    parts: list[torch.Tensor] = [zeros]
    for band in feature_set.bands:
        phase = scale * (x @ band.B.T)
        wave = scale * band.B
        grad_cos = -torch.sin(phase).unsqueeze(2) * wave.unsqueeze(0)
        grad_sin = torch.cos(phase).unsqueeze(2) * wave.unsqueeze(0)
        parts.append(grad_cos)
        parts.append(grad_sin)
    grads = torch.cat(parts, dim=1)
    if grads.shape[1] != feature_set.feature_dim:
        raise RuntimeError(
            f"Gradient dimension mismatch: expected {feature_set.feature_dim}, "
            f"got {grads.shape[1]}."
        )
    return grads


def compute_matrix_stats(matrix: torch.Tensor) -> FeatureMatrixStats:
    """Compute compact singular-value diagnostics."""

    singular_values = torch.linalg.svdvals(matrix)
    sigma_max = singular_values[0].item()
    sigma_min = singular_values[-1].item()
    if sigma_min <= 0.0:
        cond = float("inf")
    else:
        cond = sigma_max / sigma_min
    return FeatureMatrixStats(
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        cond=cond,
    )


def print_feature_set_summary(
    label: str,
    feature_set: FeatureSet,
    x_probe: torch.Tensor,
) -> None:
    """Print one compact feature-family summary."""

    xi = eval_features(x_probe, feature_set)
    stats = compute_matrix_stats(xi)
    if feature_set.kind == "tanh_ridge":
        detail = f"gamma={feature_set.gamma:.2f}"
    else:
        detail = f"sigmas={tuple(round(sigma, 4) for sigma in feature_set.sigmas)}"

    print(
        f"{label}: kind={feature_set.kind}, M={feature_set.scalar_feature_count}, "
        f"dim={feature_set.feature_dim}, {detail}, "
        f"cond≈{stats.cond:.2e}, s_min≈{stats.sigma_min:.2e}"
    )


def build_feature_evaluation_data(
    benchmark: SharedBenchmarkData,
    feature_sets: SharedFeatureSets,
    assembly_batch_size: int,
) -> FamilyEvaluationData:
    """Create evaluation tensors for coefficient-based methods."""

    return FamilyEvaluationData(
        x_int=benchmark.x_int,
        f_int=benchmark.f_int,
        x_bc=benchmark.x_bc,
        w_bc=benchmark.w_bc,
        stress_features=feature_sets.stress,
        displacement_features=feature_sets.displacement,
        compliance_voigt=benchmark.compliance_voigt,
        assembly_batch_size=assembly_batch_size,
        xi_s_test=eval_features(benchmark.x_test, feature_sets.stress),
        xi_u_test=eval_features(benchmark.x_test, feature_sets.displacement),
        u_exact_test=benchmark.u_exact_test,
        sigma_exact_test=benchmark.sigma_exact_test,
    )


def run_projection(
    x_int: torch.Tensor,
    stress_features: FeatureSet,
    displacement_features: FeatureSet,
    u_exact_train: torch.Tensor,
    sigma_exact_train: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Project exact fields into the two feature spaces."""

    dim_s = stress_features.feature_dim
    dim_u = displacement_features.feature_dim
    s = torch.zeros(6 * dim_s, dtype=DTYPE, device=DEVICE)
    u = torch.zeros(3 * dim_u, dtype=DTYPE, device=DEVICE)

    synchronize_device()
    t0 = time.perf_counter()
    with torch.no_grad():
        xi_u_train = eval_features(x_int, displacement_features)
        for comp in range(3):
            u[comp::3] = torch.linalg.lstsq(
                xi_u_train,
                u_exact_train[:, comp],
            ).solution

        xi_s_train = eval_features(x_int, stress_features)
        for comp in range(6):
            s[comp::6] = torch.linalg.lstsq(
                xi_s_train,
                sigma_exact_train[:, comp],
            ).solution

        del xi_u_train
        del xi_s_train
        clear_cuda_cache()

    synchronize_device()
    return s, u, time.perf_counter() - t0


def compute_l2_errors(
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


def compute_coefficient_residual_norms(
    data: FamilyEvaluationData,
    s: torch.Tensor,
    u: torch.Tensor,
) -> tuple[float, float, float]:
    """Evaluate constitutive, equilibrium, and boundary residual norms."""

    if not torch.isfinite(s).all() or not torch.isfinite(u).all():
        return float("nan"), float("nan"), float("nan")

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

            xi_s_batch = eval_features(xb, data.stress_features)
            grad_s_batch = eval_feature_grads(xb, data.stress_features)
            grad_u_batch = eval_feature_grads(xb, data.displacement_features)

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
            xi_u_batch = eval_features(xb, data.displacement_features)
            u_bc = xi_u_batch @ u_blocks
            boundary_sq += (wb * u_bc.square().sum(dim=1)).sum().item()

    return constitutive_sq**0.5, equilibrium_sq**0.5, boundary_sq**0.5


def evaluate_feature_result(
    name: str,
    wall_time: float,
    s: torch.Tensor,
    u: torch.Tensor,
    data: FamilyEvaluationData,
) -> AlgorithmResult:
    """Package one completed algorithm result."""

    r_c, r_e, r_b = compute_coefficient_residual_norms(data, s, u)
    rel_u, rel_sigma = compute_l2_errors(
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


def accumulate_weak_form_moments(
    x_int: torch.Tensor,
    f_int: torch.Tensor,
    stress_features: FeatureSet,
    displacement_features: FeatureSet,
    batch_size: int,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Accumulate weak-form interior moments."""

    dim_s = stress_features.feature_dim
    dim_u = displacement_features.feature_dim
    weight = 1.0 / x_int.shape[0]

    gram_s = torch.zeros(dim_s, dim_s, dtype=DTYPE, device=DEVICE)
    cross_u_grad_s = [
        torch.zeros(dim_u, dim_s, dtype=DTYPE, device=DEVICE) for _ in range(3)
    ]
    force_moment = torch.zeros(dim_u, 3, dtype=DTYPE, device=DEVICE)

    with torch.no_grad():
        for start in range(0, x_int.shape[0], batch_size):
            end = min(start + batch_size, x_int.shape[0])
            xb = x_int[start:end]
            fb = f_int[start:end]

            xi_s_batch = eval_features(xb, stress_features)
            grad_s_batch = eval_feature_grads(xb, stress_features)
            xi_u_batch = eval_features(xb, displacement_features)

            gram_s += weight * (xi_s_batch.T @ xi_s_batch)
            force_moment += weight * (xi_u_batch.T @ fb)
            for dim in range(3):
                cross_u_grad_s[dim] += weight * (
                    xi_u_batch.T @ grad_s_batch[:, :, dim]
                )

    return gram_s, cross_u_grad_s, force_moment


def accumulate_boundary_gram(
    x_bc: torch.Tensor,
    w_bc: torch.Tensor,
    displacement_features: FeatureSet,
    batch_size: int,
) -> torch.Tensor:
    """Accumulate weighted boundary Gram matrix."""

    dim_u = displacement_features.feature_dim
    gram_bc = torch.zeros(dim_u, dim_u, dtype=DTYPE, device=DEVICE)

    with torch.no_grad():
        for start in range(0, x_bc.shape[0], batch_size):
            end = min(start + batch_size, x_bc.shape[0])
            xb = x_bc[start:end]
            wb = w_bc[start:end]
            xi_u_batch = eval_features(xb, displacement_features)
            gram_bc += xi_u_batch.T @ (wb.unsqueeze(1) * xi_u_batch)

    return gram_bc


def assemble_stress_matrix(
    gram_s: torch.Tensor,
    compliance_voigt: torch.Tensor,
) -> torch.Tensor:
    """Assemble the stress block A."""

    dim_s = 6 * gram_s.shape[0]
    A = torch.zeros(dim_s, dim_s, dtype=DTYPE, device=DEVICE)
    for row in range(6):
        for col in range(6):
            A[row::6, col::6] = compliance_voigt[row, col] * gram_s
    return A


def assemble_coupling_matrix(cross_u_grad_s: list[torch.Tensor]) -> torch.Tensor:
    """Assemble the weak-form coupling block B."""

    dim_u = 3 * cross_u_grad_s[0].shape[0]
    dim_s = 6 * cross_u_grad_s[0].shape[1]
    D = [cross_u_grad_s[dim].T for dim in range(3)]

    B = torch.zeros(dim_s, dim_u, dtype=DTYPE, device=DEVICE)
    B[0::6, 0::3] = D[0]
    B[1::6, 1::3] = D[1]
    B[2::6, 2::3] = D[2]
    B[3::6, 0::3] = D[1]
    B[3::6, 1::3] = D[0]
    B[4::6, 1::3] = D[2]
    B[4::6, 2::3] = D[1]
    B[5::6, 0::3] = D[2]
    B[5::6, 2::3] = D[0]
    return B


def assemble_boundary_matrix(gram_bc: torch.Tensor) -> torch.Tensor:
    """Assemble the boundary penalty block C."""

    dim_u = 3 * gram_bc.shape[0]
    C = torch.zeros(dim_u, dim_u, dtype=DTYPE, device=DEVICE)
    for comp in range(3):
        C[comp::3, comp::3] = gram_bc
    return C


def assemble_rhs_vector(force_moment: torch.Tensor) -> torch.Tensor:
    """Assemble the displacement rhs vector F."""

    F = torch.zeros(3 * force_moment.shape[0], dtype=DTYPE, device=DEVICE)
    for comp in range(3):
        F[comp::3] = force_moment[:, comp]
    return F


def build_kkt_system(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    F: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build K z = rhs for the weak KKT system."""

    dim_s = A.shape[0]
    dim_u = B.shape[1]
    K = torch.zeros(dim_s + dim_u, dim_s + dim_u, dtype=DTYPE, device=DEVICE)
    K[:dim_s, :dim_s] = A
    K[:dim_s, dim_s:] = B
    K[dim_s:, :dim_s] = B.T
    K[dim_s:, dim_s:] = C

    rhs = torch.zeros(dim_s + dim_u, dtype=DTYPE, device=DEVICE)
    rhs[dim_s:] = -F
    return K, rhs


def solve_weak_lstsq(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    F: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Solve the weak KKT system with torch.linalg.lstsq."""

    dim_s = A.shape[0]
    K, rhs = build_kkt_system(A, B, C, F)

    synchronize_device()
    t0 = time.perf_counter()
    try:
        sol = torch.linalg.lstsq(K, rhs.unsqueeze(1)).solution.squeeze(1)
        if not torch.isfinite(sol).all():
            raise RuntimeError("non-finite solution")
    except (RuntimeError, torch.linalg.LinAlgError) as exc:
        sol = torch.full((K.shape[0],), float("nan"), dtype=DTYPE, device=DEVICE)
        print(f"    Warning: weak lstsq failed with {type(exc).__name__}")

    synchronize_device()
    wall_time = time.perf_counter() - t0
    return sol[:dim_s], sol[dim_s:], wall_time


def solve_weak_eigh(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    F: torch.Tensor,
    rtol: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Solve the weak KKT system with truncated eigendecomposition."""

    dim_s = A.shape[0]
    K, rhs = build_kkt_system(A, B, C, F)

    synchronize_device()
    t0 = time.perf_counter()
    try:
        eigvals, eigvecs = torch.linalg.eigh(K)
        threshold = rtol * eigvals.abs().max()
        keep = eigvals.abs() > threshold
        if not keep.any():
            raise RuntimeError("all eigenvalues were truncated")

        coeffs = eigvecs[:, keep].T @ rhs
        coeffs = coeffs / eigvals[keep]
        sol = eigvecs[:, keep] @ coeffs
        if not torch.isfinite(sol).all():
            raise RuntimeError("non-finite solution")
        print(
            f"    weak eigh truncation: kept {int(keep.sum().item())}/{eigvals.numel()} "
            f"eigenvalues, threshold={threshold.item():.2e}"
        )
    except (RuntimeError, torch.linalg.LinAlgError) as exc:
        sol = torch.full((K.shape[0],), float("nan"), dtype=DTYPE, device=DEVICE)
        print(f"    Warning: weak eigh failed with {type(exc).__name__}")

    synchronize_device()
    wall_time = time.perf_counter() - t0
    return sol[:dim_s], sol[dim_s:], wall_time


def add_block_scaled(
    target: torch.Tensor,
    feature_matrix: torch.Tensor,
    block: torch.Tensor,
    row_stride: int,
    col_stride: int,
) -> None:
    """Add feature_matrix kron block into a strided matrix."""

    for row in range(block.shape[0]):
        for col in range(block.shape[1]):
            coeff = block[row, col].item()
            if coeff == 0.0:
                continue
            target[row::row_stride, col::col_stride] += coeff * feature_matrix


def add_rhs_feature_blocks(
    target: torch.Tensor,
    feature_by_vec: torch.Tensor,
    block: torch.Tensor,
    block_size: int,
    scale: float = 1.0,
) -> None:
    """Accumulate feature moments into a strided rhs vector."""

    contribution = scale * (feature_by_vec @ block.T)
    for row in range(block.shape[0]):
        target[row::block_size] += contribution[:, row]


def accumulate_strong_interior_moments(
    x_int: torch.Tensor,
    f_int: torch.Tensor,
    stress_features: FeatureSet,
    displacement_features: FeatureSet,
    batch_size: int,
) -> tuple[
    torch.Tensor,
    list[torch.Tensor],
    list[list[torch.Tensor]],
    list[list[torch.Tensor]],
    list[torch.Tensor],
]:
    """Accumulate moments for the strong-form normal equations."""

    dim_s = stress_features.feature_dim
    dim_u = displacement_features.feature_dim
    weight = 1.0 / x_int.shape[0]

    gram_xi_s = torch.zeros(dim_s, dim_s, dtype=DTYPE, device=DEVICE)
    cross_xi_grad_u = [
        torch.zeros(dim_s, dim_u, dtype=DTYPE, device=DEVICE) for _ in range(3)
    ]
    grad_gram_u = [
        [
            torch.zeros(dim_u, dim_u, dtype=DTYPE, device=DEVICE)
            for _ in range(3)
        ]
        for _ in range(3)
    ]
    grad_gram_s = [
        [
            torch.zeros(dim_s, dim_s, dtype=DTYPE, device=DEVICE)
            for _ in range(3)
        ]
        for _ in range(3)
    ]
    grad_force_s = [
        torch.zeros(dim_s, 3, dtype=DTYPE, device=DEVICE) for _ in range(3)
    ]

    with torch.no_grad():
        for start in range(0, x_int.shape[0], batch_size):
            end = min(start + batch_size, x_int.shape[0])
            xb = x_int[start:end]
            fb = f_int[start:end]

            xi_s_batch = eval_features(xb, stress_features)
            grad_s_batch = eval_feature_grads(xb, stress_features)
            grad_u_batch = eval_feature_grads(xb, displacement_features)

            gram_xi_s += weight * (xi_s_batch.T @ xi_s_batch)
            for dim_i in range(3):
                cross_xi_grad_u[dim_i] += weight * (
                    xi_s_batch.T @ grad_u_batch[:, :, dim_i]
                )
                grad_force_s[dim_i] += weight * (grad_s_batch[:, :, dim_i].T @ fb)
                for dim_j in range(3):
                    grad_gram_u[dim_i][dim_j] += weight * (
                        grad_u_batch[:, :, dim_i].T @ grad_u_batch[:, :, dim_j]
                    )
                    grad_gram_s[dim_i][dim_j] += weight * (
                        grad_s_batch[:, :, dim_i].T @ grad_s_batch[:, :, dim_j]
                    )

    return gram_xi_s, cross_xi_grad_u, grad_gram_u, grad_gram_s, grad_force_s


def assemble_strong_normal_equations(
    strong_cfg,
    compliance_voigt: torch.Tensor,
    x_int: torch.Tensor,
    f_int: torch.Tensor,
    x_bc: torch.Tensor,
    w_bc: torch.Tensor,
    feature_sets: SharedFeatureSets,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble H z = g for the strong-form least-squares system."""

    (
        gram_xi_s,
        cross_xi_grad_u,
        grad_gram_u,
        grad_gram_s,
        grad_force_s,
    ) = accumulate_strong_interior_moments(
        x_int,
        f_int,
        feature_sets.stress,
        feature_sets.displacement,
        strong_cfg.assembly_batch_size,
    )
    gram_bc_u = accumulate_boundary_gram(
        x_bc,
        w_bc,
        feature_sets.displacement,
        strong_cfg.assembly_batch_size,
    )

    dim_s = 6 * feature_sets.stress.feature_dim
    dim_u = 3 * feature_sets.displacement.feature_dim

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
    return H, g


def solve_strong_lstsq(H: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Solve the strong-form normal equations with lstsq."""

    synchronize_device()
    t0 = time.perf_counter()
    try:
        sol = torch.linalg.lstsq(H, g.unsqueeze(1)).solution.squeeze(1)
        if not torch.isfinite(sol).all():
            raise RuntimeError("non-finite solution")
    except (RuntimeError, torch.linalg.LinAlgError) as exc:
        sol = torch.full((H.shape[0],), float("nan"), dtype=DTYPE, device=DEVICE)
        print(f"    Warning: strong lstsq failed with {type(exc).__name__}")

    synchronize_device()
    return sol, time.perf_counter() - t0


def solve_strong_eigh(
    H: torch.Tensor,
    g: torch.Tensor,
    rtol: float,
) -> tuple[torch.Tensor, float]:
    """Solve the strong-form normal equations with truncated eigendecomposition."""

    synchronize_device()
    t0 = time.perf_counter()
    try:
        eigvals, eigvecs = torch.linalg.eigh(H)
        threshold = rtol * eigvals.abs().max()
        keep = eigvals > threshold
        if not keep.any():
            raise RuntimeError("all eigenvalues were truncated")

        coeffs = eigvecs[:, keep].T @ g
        coeffs = coeffs / eigvals[keep]
        sol = eigvecs[:, keep] @ coeffs
        if not torch.isfinite(sol).all():
            raise RuntimeError("non-finite solution")
        print(
            f"    strong eigh truncation: kept {int(keep.sum().item())}/{eigvals.numel()} "
            f"eigenvalues, threshold={threshold.item():.2e}"
        )
    except (RuntimeError, torch.linalg.LinAlgError) as exc:
        sol = torch.full((H.shape[0],), float("nan"), dtype=DTYPE, device=DEVICE)
        print(f"    Warning: strong eigh failed with {type(exc).__name__}")

    synchronize_device()
    return sol, time.perf_counter() - t0


def split_solution(z: torch.Tensor, dim_s: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the full coefficient vector into stress and displacement parts."""

    return z[:dim_s], z[dim_s:]


def run_weak_algorithm(
    algorithm_id: str,
    data: WeakExperimentData,
    weak_cfg,
) -> AlgorithmResult:
    """Run one weak-form solver."""

    if algorithm_id == "eigh":
        print("Running Weak (Eigh)...")
        s, u, wall_time = solve_weak_eigh(
            data.A,
            data.B,
            data.C,
            data.F,
            weak_cfg.eigh_rtol,
        )
        result = evaluate_feature_result("Weak (Eigh)", wall_time, s, u, data.eval_data)
    else:
        print("Running Weak (Lstsq)...")
        s, u, wall_time = solve_weak_lstsq(data.A, data.B, data.C, data.F)
        result = evaluate_feature_result("Weak (Lstsq)", wall_time, s, u, data.eval_data)

    print_result_summary(result)
    return result


def run_strong_algorithm(
    algorithm_id: str,
    data: StrongExperimentData,
    strong_cfg,
) -> AlgorithmResult:
    """Run one strong-form solver."""

    if algorithm_id == "eigh":
        print("Running Strong (Eigh)...")
        z, wall_time = solve_strong_eigh(data.H, data.g, strong_cfg.eigh_rtol)
        s, u = split_solution(z, data.dim_s)
        result = evaluate_feature_result("Strong (Eigh)", wall_time, s, u, data.eval_data)
    else:
        print("Running Strong (Lstsq)...")
        z, wall_time = solve_strong_lstsq(data.H, data.g)
        s, u = split_solution(z, data.dim_s)
        result = evaluate_feature_result("Strong (Lstsq)", wall_time, s, u, data.eval_data)

    print_result_summary(result)
    return result


def run_projection_case(
    benchmark: SharedBenchmarkData,
    feature_sets: SharedFeatureSets,
    eval_data: FamilyEvaluationData,
    u_exact_train: torch.Tensor,
    sigma_exact_train: torch.Tensor,
) -> AlgorithmResult:
    """Run the projection baseline for one feature case."""

    print("Running Projection...")
    s, u, wall_time = run_projection(
        benchmark.x_int,
        feature_sets.stress,
        feature_sets.displacement,
        u_exact_train,
        sigma_exact_train,
    )
    result = evaluate_feature_result("Projection", wall_time, s, u, eval_data)
    print_result_summary(result)
    return result


def run_weak_case(
    benchmark: SharedBenchmarkData,
    feature_sets: SharedFeatureSets,
    eval_data: FamilyEvaluationData,
    weak_cfg,
    weak_algorithm_ids: Sequence[str],
) -> dict[str, AlgorithmResult]:
    """Run the selected weak-form solvers for one feature case."""

    gram_s, cross_u_grad_s, force_moment = accumulate_weak_form_moments(
        benchmark.x_int,
        benchmark.f_int,
        feature_sets.stress,
        feature_sets.displacement,
        weak_cfg.assembly_batch_size,
    )
    gram_bc = accumulate_boundary_gram(
        benchmark.x_bc,
        benchmark.w_bc,
        feature_sets.displacement,
        weak_cfg.assembly_batch_size,
    )
    A = assemble_stress_matrix(gram_s, benchmark.compliance_voigt)
    B = assemble_coupling_matrix(cross_u_grad_s)
    C = weak_cfg.lambda_bc * assemble_boundary_matrix(gram_bc)
    F = assemble_rhs_vector(force_moment)

    del gram_s
    del cross_u_grad_s
    del force_moment
    del gram_bc
    clear_cuda_cache()

    print(
        f"System shapes: A={tuple(A.shape)}, B={tuple(B.shape)}, "
        f"C={tuple(C.shape)}, F={tuple(F.shape)}"
    )
    weak_data = WeakExperimentData(A=A, B=B, C=C, F=F, eval_data=eval_data)
    try:
        return {
            TOP_LEVEL_ALGORITHM_LABELS[f"weak({algorithm_id})"]: run_weak_algorithm(
                algorithm_id,
                weak_data,
                weak_cfg,
            )
            for algorithm_id in weak_algorithm_ids
        }
    finally:
        del A
        del B
        del C
        del F
        clear_cuda_cache()


def run_strong_case(
    benchmark: SharedBenchmarkData,
    feature_sets: SharedFeatureSets,
    eval_data: FamilyEvaluationData,
    strong_cfg,
    strong_algorithm_ids: Sequence[str],
) -> dict[str, AlgorithmResult]:
    """Run the selected strong-form solvers for one feature case."""

    H, g = assemble_strong_normal_equations(
        strong_cfg,
        benchmark.compliance_voigt,
        benchmark.x_int,
        benchmark.f_int,
        benchmark.x_bc,
        benchmark.w_bc,
        feature_sets,
    )
    clear_cuda_cache()
    print(f"System shapes: H={tuple(H.shape)}, g={tuple(g.shape)}")

    strong_data = StrongExperimentData(
        H=H,
        g=g,
        dim_s=6 * feature_sets.stress.feature_dim,
        eval_data=eval_data,
    )
    try:
        return {
            TOP_LEVEL_ALGORITHM_LABELS[f"strong({algorithm_id})"]: run_strong_algorithm(
                algorithm_id,
                strong_data,
                strong_cfg,
            )
            for algorithm_id in strong_algorithm_ids
        }
    finally:
        del H
        del g
        clear_cuda_cache()


def plot_ablation_feature(
    results: dict[str, dict[str, AlgorithmResult]],
    ordered_labels: Sequence[str],
    save_path: str,
) -> None:
    """Plot relative L2 errors across feature cases as grouped bar charts."""

    if not results:
        return

    configure_plotting()
    feature_labels = list(results.keys())
    x_positions = np.arange(len(feature_labels), dtype=float)
    bar_count = max(len(ordered_labels), 1)
    bar_width = min(0.72 / bar_count, 0.3)
    offsets = (
        np.arange(bar_count, dtype=float) - 0.5 * (bar_count - 1)
    ) * bar_width

    fig_width = max(10.0, 1.8 * len(feature_labels) + 0.9 * bar_count)
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
        for index, label in enumerate(ordered_labels):
            values = np.array(
                [
                    getattr(results[feature_label][label], metric_name)
                    if label in results[feature_label]
                    else float("nan")
                    for feature_label in feature_labels
                ],
                dtype=float,
            )
            valid = np.isfinite(values) & (values > 0.0)
            if not valid.any():
                continue

            style = ALGO_STYLE.get(label, {"color": "#4C78A8"})
            ax.bar(
                x_positions[valid] + offsets[index],
                values[valid],
                color=style["color"],
                width=bar_width,
                label=label,
            )

            invalid_indices = np.flatnonzero(~valid)
            for invalid_index in invalid_indices:
                print(
                    f"  Skipped {feature_labels[invalid_index]} {label} "
                    f"{metric_name}={values[invalid_index]!r} in {save_path}"
                )

        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Feature family")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(feature_labels, rotation=15, ha="right")
        ax.grid(alpha=0.3, linestyle="--", axis="y")
        if ax.containers:
            ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def print_ablation_feature_summary_table(
    feature_labels: Sequence[str],
    results: dict[str, dict[str, AlgorithmResult]],
    ordered_labels: Sequence[str],
) -> None:
    """Print the final comparable summary table."""

    print("\n=== Ablation Feature Summary ===\n")
    print(
        f"| {'Feature':<20} | {'Method':<16} | "
        f"{'rel_u':>12} | {'rel_sigma':>12} | {'Time(s)':>8} |"
    )
    print(
        f"|:{'-' * 21}|:{'-' * 17}|"
        f"{'-' * 13}:|{'-' * 13}:|{'-' * 9}:|"
    )

    for feature_label in feature_labels:
        case_results = results[feature_label]
        for algorithm_label in ordered_labels:
            if algorithm_label not in case_results:
                continue
            result = case_results[algorithm_label]
            print(
                f"| {feature_label:<20} | {algorithm_label:<16} | "
                f"{result.rel_u:12.2e} | {result.rel_sigma:12.2e} | {result.wall_time:8.2f} |"
            )


def run_ablation(
    cfg: MainConfig | None = None,
    feature_case_list: Sequence[FeatureAblationCase] | None = None,
) -> dict[str, dict[str, AlgorithmResult]]:
    """Run the feature-generation ablation study and return all metrics."""

    cfg = make_default_main_config() if cfg is None else cfg
    shared_cfg = SharedComparisonConfig() if cfg.shared is None else cfg.shared
    validate_shared_comparison_config(shared_cfg)

    selected_algorithm_ids = validate_algorithm_selection(
        cfg.algorithms_to_run,
        VALID_TOP_LEVEL_ALGORITHMS,
    )
    unsupported_algorithm_ids = [
        algorithm_id
        for algorithm_id in selected_algorithm_ids
        if algorithm_id not in FEATURE_ALGORITHM_IDS
    ]
    if unsupported_algorithm_ids:
        raise ValueError(
            "ablation_feature only supports feature-based algorithms. "
            f"Unsupported ids: {unsupported_algorithm_ids}"
        )

    ordered_labels = [
        TOP_LEVEL_ALGORITHM_LABELS[algorithm_id]
        for algorithm_id in selected_algorithm_ids
    ]
    feature_case_list = (
        DEFAULT_ABLATION_FEATURE_LIST
        if feature_case_list is None
        else list(feature_case_list)
    )
    validated_cases = validate_feature_case_list(feature_case_list, shared_cfg)

    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Algorithms: {selected_algorithm_ids}")
    print(f"Feature cases: {[case.label for case in validated_cases]}")

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
    x_probe = benchmark.x_int[: min(DEFAULT_CONDITION_PROBE_SIZE, benchmark.x_int.shape[0])]

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
            raise ValueError("MainConfig.strong is required when running strong-form algorithms.")
        strong_cfg = apply_shared_to_strong_config(cfg.strong, shared_cfg, strong_algorithm_ids)
        validate_strong_config(strong_cfg)
    else:
        strong_cfg = None

    all_results: dict[str, dict[str, AlgorithmResult]] = {}
    for case in validated_cases:
        print(f"\n{'=' * 72}")
        print(f"=== Ablation Feature: {case.label} ({case.kind}) ===")
        print(f"{'=' * 72}")

        feature_cfg = build_feature_family_config(case, shared_cfg)
        feature_sets = build_shared_feature_sets(feature_cfg)
        eval_data = build_feature_evaluation_data(
            benchmark,
            feature_sets,
            feature_eval_batch_size,
        )
        print_feature_set_summary("Stress features", feature_sets.stress, x_probe)
        print_feature_set_summary("Displacement features", feature_sets.displacement, x_probe)

        case_results: dict[str, AlgorithmResult] = {}
        if projection_enabled:
            if u_exact_train is None or sigma_exact_train is None:
                raise RuntimeError("Projection targets must be precomputed.")
            case_results["Projection"] = run_projection_case(
                benchmark,
                feature_sets,
                eval_data,
                u_exact_train,
                sigma_exact_train,
            )

        if weak_algorithm_ids:
            if weak_cfg is None:
                raise RuntimeError("Weak config must be prepared before running the ablation.")
            case_results.update(
                run_weak_case(
                    benchmark,
                    feature_sets,
                    eval_data,
                    weak_cfg,
                    weak_algorithm_ids,
                )
            )

        if strong_algorithm_ids:
            if strong_cfg is None:
                raise RuntimeError("Strong config must be prepared before running the ablation.")
            case_results.update(
                run_strong_case(
                    benchmark,
                    feature_sets,
                    eval_data,
                    strong_cfg,
                    strong_algorithm_ids,
                )
            )

        all_results[case.label] = case_results
        del feature_sets
        del eval_data
        clear_experiment_memory()

    if all_results:
        print("\nGenerating plots...")
        plot_ablation_feature(
            all_results,
            ordered_labels,
            str(OUTPUT_DIR / "ablation-feature.png"),
        )
        print_ablation_feature_summary_table(
            list(all_results.keys()),
            all_results,
            ordered_labels,
        )

    return all_results


def main(
    cfg: MainConfig | None = None,
    feature_case_list: Sequence[FeatureAblationCase] | None = None,
) -> None:
    """Script entrypoint."""

    run_ablation(cfg=cfg, feature_case_list=feature_case_list)


if __name__ == "__main__":
    main()
