from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.modules.setdefault("plate_bending", sys.modules[__name__])

if TYPE_CHECKING:
    from projection import ProjectionConfig
    from strong_form import StrongConfig
    from weak_form import WeakConfig


BASE_SEED = 42
MOMENT_SEED = BASE_SEED
DISP_SEED = BASE_SEED + 1_000
DTYPE = torch.float64
VALID_SAMPLING_METHODS = ("mc", "sobol")
VALID_TOP_LEVEL_ALGORITHMS = (
    "projection",
    "weak(eigh)",
    "weak(lstsq)",
    "strong(eigh)",
    "strong(lstsq)",
)
TOP_LEVEL_ALGORITHM_LABELS = {
    "projection": "Projection",
    "weak(eigh)": "Weak (Eigh)",
    "weak(lstsq)": "Weak (Lstsq)",
    "strong(eigh)": "Strong (Eigh)",
    "strong(lstsq)": "Strong (Lstsq)",
}
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "public" / "images" / "penalty-method" / "plate-bending"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ALGO_STYLE = {
    "Projection": {"color": "#9B2226", "marker": "P", "linestyle": ":"},
    "Weak (Eigh)": {"color": "#0077B6", "marker": "o", "linestyle": "-"},
    "Weak (Lstsq)": {"color": "#264653", "marker": "s", "linestyle": "--"},
    "Strong (Eigh)": {"color": "#2A9D8F", "marker": "D", "linestyle": "-."},
    "Strong (Lstsq)": {"color": "#6D597A", "marker": "^", "linestyle": "--"},
}


def detect_device() -> torch.device:
    """Prefer CUDA when available and stay quiet otherwise."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


DEVICE = detect_device()

torch.manual_seed(BASE_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(BASE_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

FROBENIUS_WEIGHT = torch.tensor([1.0, 1.0, 2.0], dtype=DTYPE, device=DEVICE)
FROBENIUS_WEIGHT_MATRIX = torch.diag(FROBENIUS_WEIGHT)


@dataclass
class SharedComparisonConfig:
    """Shared fairness budget used by the top-level comparison runner."""

    E: float = 1.0
    nu: float = 0.3
    h: float = 1.0
    gamma_m: float = 2.0
    gamma_u: float = 2.0
    M_m: int = 300
    M_u: int = 300
    Q_int: int = (2 ** 8) ** 2
    Q_bc: int = 4 * (2 ** 7)
    Q_test: int = (2 ** 7) ** 2
    sampling_method: str = "sobol"
    interior_seed: int = BASE_SEED + 1
    boundary_seed: int = BASE_SEED + 2
    test_seed: int = BASE_SEED + 3
    moment_feature_seed: int = MOMENT_SEED
    disp_feature_seed: int = DISP_SEED


@dataclass
class MainConfig:
    """Top-level coordinator configuration."""

    algorithms_to_run: list[str] = field(
        default_factory=lambda: [
            # "projection",
            "weak(eigh)",
            "weak(lstsq)",
            "strong(eigh)",
            "strong(lstsq)",
        ]
    )
    shared: "SharedComparisonConfig | None" = None
    projection: "ProjectionConfig | None" = None
    weak: "WeakConfig | None" = None
    strong: "StrongConfig | None" = None


@dataclass(frozen=True)
class AlgorithmResult:
    """Compact metrics for one completed algorithm."""

    name: str
    r_c: float
    r_e: float
    r_b0: float
    r_b1: float
    rel_u: float
    rel_M: float
    wall_time: float


@dataclass(frozen=True)
class SharedBenchmarkData:
    """Shared train/test samples reused across selected algorithms."""

    x_int: torch.Tensor
    f_int: torch.Tensor
    x_bc: torch.Tensor
    w_bc: torch.Tensor
    n_bc: torch.Tensor
    x_test: torch.Tensor
    u_exact_test: torch.Tensor
    M_exact_test: torch.Tensor
    compliance_voigt: torch.Tensor


@dataclass(frozen=True)
class SharedFeatureSpace:
    """Shared random feature spaces used by feature-based methods."""

    a_m: torch.Tensor
    r_m: torch.Tensor
    a_u: torch.Tensor
    r_u: torch.Tensor
    gamma_m: float
    gamma_u: float


@dataclass(frozen=True)
class FeatureEvaluationData:
    """All tensors needed to evaluate coefficient-based methods."""

    x_int: torch.Tensor
    f_int: torch.Tensor
    x_bc: torch.Tensor
    w_bc: torch.Tensor
    n_bc: torch.Tensor
    a_m: torch.Tensor
    r_m: torch.Tensor
    a_u: torch.Tensor
    r_u: torch.Tensor
    gamma_m: float
    gamma_u: float
    compliance_voigt: torch.Tensor
    assembly_batch_size: int
    xi_m_test: torch.Tensor
    xi_u_test: torch.Tensor
    u_exact_test: torch.Tensor
    M_exact_test: torch.Tensor


def clear_cuda_cache() -> None:
    """Release cached CUDA buffers after large tensors are freed."""

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def synchronize_device() -> None:
    """Synchronize queued device work before reading wall-clock timings."""

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def validate_sampling_method(method: str) -> None:
    """Reject unsupported sampling modes early."""

    if method not in VALID_SAMPLING_METHODS:
        raise ValueError(
            f"Unknown sampling_method='{method}'. "
            f"Valid values: {list(VALID_SAMPLING_METHODS)}"
        )


def validate_shared_comparison_config(cfg: SharedComparisonConfig) -> None:
    """Validate the shared fairness budget used by the top-level runner."""

    if cfg.E <= 0.0:
        raise ValueError("SharedComparisonConfig.E must be positive.")
    if not (-1.0 < cfg.nu < 0.5):
        raise ValueError("SharedComparisonConfig.nu must lie in (-1, 0.5).")
    if cfg.h <= 0.0:
        raise ValueError("SharedComparisonConfig.h must be positive.")
    if cfg.gamma_m <= 0.0 or cfg.gamma_u <= 0.0:
        raise ValueError(
            "SharedComparisonConfig.gamma_m and SharedComparisonConfig.gamma_u "
            "must be positive."
        )
    if cfg.M_m <= 0 or cfg.M_u <= 0:
        raise ValueError("SharedComparisonConfig.M_m and SharedComparisonConfig.M_u must be positive.")
    if cfg.Q_int <= 0:
        raise ValueError("SharedComparisonConfig.Q_int must be positive.")
    if cfg.Q_bc < 4:
        raise ValueError("SharedComparisonConfig.Q_bc must be at least 4.")
    if cfg.Q_test <= 0:
        raise ValueError("SharedComparisonConfig.Q_test must be positive.")
    validate_sampling_method(cfg.sampling_method)


def validate_algorithm_selection(
    algorithm_ids: Sequence[str],
    valid_algorithm_ids: Sequence[str],
) -> list[str]:
    """Validate algorithm ids and preserve user order."""

    if not algorithm_ids:
        raise ValueError("algorithms_to_run must contain at least one algorithm id.")

    unknown_ids = [
        algorithm_id
        for algorithm_id in algorithm_ids
        if algorithm_id not in valid_algorithm_ids
    ]
    if unknown_ids:
        raise ValueError(
            f"Unknown algorithm ids: {unknown_ids}. Valid ids: {list(valid_algorithm_ids)}"
        )

    seen: set[str] = set()
    duplicates: list[str] = []
    for algorithm_id in algorithm_ids:
        if algorithm_id in seen and algorithm_id not in duplicates:
            duplicates.append(algorithm_id)
        seen.add(algorithm_id)
    if duplicates:
        raise ValueError(f"Duplicate algorithm ids: {duplicates}")

    return list(algorithm_ids)


def extract_scoped_algorithm_ids(
    algorithm_ids: Sequence[str],
    scope: str,
) -> list[str]:
    """Return inner algorithm ids from top-level scoped ids like weak(eigh)."""

    prefix = f"{scope}("
    return [
        algorithm_id[len(prefix) : -1]
        for algorithm_id in algorithm_ids
        if algorithm_id.startswith(prefix) and algorithm_id.endswith(")")
    ]


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


def build_compliance_bilinear_matrix(compliance_voigt: torch.Tensor) -> torch.Tensor:
    """Return the weighted compliance block for bilinear forms."""

    return FROBENIUS_WEIGHT_MATRIX @ compliance_voigt


def eval_exact_deflection(x: torch.Tensor) -> torch.Tensor:
    """Evaluate the manufactured clamped deflection field."""

    x1 = x[:, 0]
    x2 = x[:, 1]
    return poly_p(x1) * poly_p(x2)


def eval_exact_hessian(x: torch.Tensor) -> torch.Tensor:
    """Evaluate the Hessian components (11, 22, 12) of the exact deflection."""

    x1 = x[:, 0]
    x2 = x[:, 1]
    return torch.stack(
        [
            poly_d2p(x1) * poly_p(x2),
            poly_p(x1) * poly_d2p(x2),
            poly_dp(x1) * poly_dp(x2),
        ],
        dim=1,
    )


def eval_exact_moment(x: torch.Tensor, D: float, nu: float) -> torch.Tensor:
    """Evaluate the exact bending moment field in Voigt order (11, 22, 12)."""

    hess_u = eval_exact_hessian(x)
    u_xx = hess_u[:, 0]
    u_yy = hess_u[:, 1]
    u_xy = hess_u[:, 2]
    M11 = -D * (u_xx + nu * u_yy)
    M22 = -D * (nu * u_xx + u_yy)
    M12 = -D * (1.0 - nu) * u_xy
    return torch.stack([M11, M22, M12], dim=1)


def compute_body_force(x: torch.Tensor, D: float) -> torch.Tensor:
    """Evaluate the manufactured transverse load f = D * Delta^2 u."""

    x1 = x[:, 0]
    x2 = x[:, 1]
    return D * (
        poly_d4p(x1) * poly_p(x2)
        + 2.0 * poly_d2p(x1) * poly_d2p(x2)
        + poly_p(x1) * poly_d4p(x2)
    )


def poly_p(t: torch.Tensor) -> torch.Tensor:
    """Evaluate p(t) = t^2 (1 - t)^2."""

    return t.square() * (1.0 - t).square()


def poly_dp(t: torch.Tensor) -> torch.Tensor:
    """Evaluate p'(t) for p(t) = t^2 (1 - t)^2."""

    return 2.0 * t - 6.0 * t.square() + 4.0 * t.pow(3)


def poly_d2p(t: torch.Tensor) -> torch.Tensor:
    """Evaluate p''(t) for p(t) = t^2 (1 - t)^2."""

    return 2.0 - 12.0 * t + 12.0 * t.square()


def poly_d4p(t: torch.Tensor) -> torch.Tensor:
    """Evaluate p''''(t) for p(t) = t^2 (1 - t)^2."""

    return torch.full_like(t, 24.0)


def sample_points(
    n_points: int,
    method: str,
    dim: int = 2,
    seed: int = 0,
) -> torch.Tensor:
    """Sample points from the unit box."""

    validate_sampling_method(method)
    if method == "sobol":
        engine = torch.quasirandom.SobolEngine(
            dimension=dim,
            scramble=True,
            seed=seed,
        )
        return engine.draw(n_points).to(dtype=DTYPE, device=DEVICE)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.rand(n_points, dim, generator=generator, dtype=DTYPE).to(DEVICE)


def sample_boundary_points(
    n_points: int,
    method: str,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample points and outward normals on the boundary of the unit square."""

    counts = [n_points // 4] * 4
    for edge_id in range(n_points % 4):
        counts[edge_id] += 1

    point_parts: list[torch.Tensor] = []
    weight_parts: list[torch.Tensor] = []
    normal_parts: list[torch.Tensor] = []
    normals = torch.tensor(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [0.0, 1.0],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )

    for edge_id, count in enumerate(counts):
        t = sample_points(count, method=method, dim=1, seed=seed + edge_id).squeeze(1)
        edge_points = torch.zeros(count, 2, dtype=DTYPE, device=DEVICE)
        if edge_id == 0:
            edge_points[:, 0] = 0.0
            edge_points[:, 1] = t
        elif edge_id == 1:
            edge_points[:, 0] = 1.0
            edge_points[:, 1] = t
        elif edge_id == 2:
            edge_points[:, 0] = t
            edge_points[:, 1] = 0.0
        else:
            edge_points[:, 0] = t
            edge_points[:, 1] = 1.0

        point_parts.append(edge_points)
        weight_parts.append(
            torch.full((count,), 1.0 / count, dtype=DTYPE, device=DEVICE)
        )
        normal_parts.append(normals[edge_id].expand(count, -1))

    return (
        torch.cat(point_parts, dim=0),
        torch.cat(weight_parts, dim=0),
        torch.cat(normal_parts, dim=0),
    )


def generate_features(M: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random feature normals and offsets."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    raw = torch.randn(M, 2, generator=generator, dtype=DTYPE)
    norms = raw.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
    a = (raw / norms).to(DEVICE)
    r = torch.rand(M, generator=generator, dtype=DTYPE).to(DEVICE)
    return a, r


def eval_features(
    x: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Evaluate xi_0 = 1 and xi_m = tanh(gamma (a_m^T x + r_m))."""

    pre = x @ a.T + r.unsqueeze(0)
    xi = torch.tanh(gamma * pre)
    ones = torch.ones(x.shape[0], 1, dtype=DTYPE, device=DEVICE)
    return torch.cat([ones, xi], dim=1)


def eval_feature_grads(
    x: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Evaluate gradients of all random features."""

    pre = x @ a.T + r.unsqueeze(0)
    xi = torch.tanh(gamma * pre)
    dphi = gamma * (1.0 - xi.square())
    grad_xi = dphi.unsqueeze(2) * a.unsqueeze(0)
    zeros = torch.zeros(x.shape[0], 1, 2, dtype=DTYPE, device=DEVICE)
    return torch.cat([zeros, grad_xi], dim=1)


def eval_feature_hessians(
    x: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Evaluate Hessian components (11, 22, 12) of all random features."""

    pre = x @ a.T + r.unsqueeze(0)
    xi = torch.tanh(gamma * pre)
    d2phi = -2.0 * gamma * gamma * xi * (1.0 - xi.square())

    h11 = d2phi.unsqueeze(2) * a[:, 0].square().unsqueeze(0).unsqueeze(2)
    h22 = d2phi.unsqueeze(2) * a[:, 1].square().unsqueeze(0).unsqueeze(2)
    h12 = d2phi.unsqueeze(2) * (a[:, 0] * a[:, 1]).unsqueeze(0).unsqueeze(2)
    hess = torch.cat([h11, h22, h12], dim=2)
    zeros = torch.zeros(x.shape[0], 1, 3, dtype=DTYPE, device=DEVICE)
    return torch.cat([zeros, hess], dim=1)


def build_shared_benchmark(
    E: float,
    nu: float,
    h: float,
    Q_int: int,
    Q_bc: int,
    Q_test: int,
    sampling_method: str,
    interior_seed: int = BASE_SEED + 1,
    boundary_seed: int = BASE_SEED + 2,
    test_seed: int = BASE_SEED + 3,
) -> SharedBenchmarkData:
    """Build the shared train/test samples for a fair comparison run."""

    D = compute_bending_stiffness(E, nu, h)
    compliance_voigt = build_compliance_matrix(D, nu)

    x_int = sample_points(Q_int, method=sampling_method, seed=interior_seed)
    f_int = compute_body_force(x_int, D)

    x_bc, w_bc, n_bc = sample_boundary_points(
        Q_bc,
        method=sampling_method,
        seed=boundary_seed,
    )

    x_test = sample_points(Q_test, method=sampling_method, seed=test_seed)
    u_exact_test = eval_exact_deflection(x_test)
    M_exact_test = eval_exact_moment(x_test, D, nu)

    return SharedBenchmarkData(
        x_int=x_int,
        f_int=f_int,
        x_bc=x_bc,
        w_bc=w_bc,
        n_bc=n_bc,
        x_test=x_test,
        u_exact_test=u_exact_test,
        M_exact_test=M_exact_test,
        compliance_voigt=compliance_voigt,
    )


def build_shared_feature_space(
    M_m: int,
    M_u: int,
    gamma_m: float,
    gamma_u: float,
    moment_feature_seed: int = MOMENT_SEED,
    disp_feature_seed: int = DISP_SEED,
) -> SharedFeatureSpace:
    """Build the shared random feature spaces used by feature methods."""

    a_m, r_m = generate_features(M_m, seed=moment_feature_seed)
    a_u, r_u = generate_features(M_u, seed=disp_feature_seed)
    return SharedFeatureSpace(
        a_m=a_m,
        r_m=r_m,
        a_u=a_u,
        r_u=r_u,
        gamma_m=gamma_m,
        gamma_u=gamma_u,
    )


def build_feature_evaluation_data(
    benchmark: SharedBenchmarkData,
    feature_space: SharedFeatureSpace,
    assembly_batch_size: int,
) -> FeatureEvaluationData:
    """Build the shared evaluation tensors for coefficient-based methods."""

    return FeatureEvaluationData(
        x_int=benchmark.x_int,
        f_int=benchmark.f_int,
        x_bc=benchmark.x_bc,
        w_bc=benchmark.w_bc,
        n_bc=benchmark.n_bc,
        a_m=feature_space.a_m,
        r_m=feature_space.r_m,
        a_u=feature_space.a_u,
        r_u=feature_space.r_u,
        gamma_m=feature_space.gamma_m,
        gamma_u=feature_space.gamma_u,
        compliance_voigt=benchmark.compliance_voigt,
        assembly_batch_size=assembly_batch_size,
        xi_m_test=eval_features(
            benchmark.x_test,
            feature_space.a_m,
            feature_space.r_m,
            feature_space.gamma_m,
        ),
        xi_u_test=eval_features(
            benchmark.x_test,
            feature_space.a_u,
            feature_space.r_u,
            feature_space.gamma_u,
        ),
        u_exact_test=benchmark.u_exact_test,
        M_exact_test=benchmark.M_exact_test,
    )


def compute_l2_errors(
    xi_u_test: torch.Tensor,
    xi_m_test: torch.Tensor,
    m: torch.Tensor,
    u: torch.Tensor,
    u_exact: torch.Tensor,
    M_exact: torch.Tensor,
) -> tuple[float, float]:
    """Compute relative L2 errors for deflection and bending moments."""

    M_h = xi_m_test @ m.reshape(-1, 3)
    u_h = xi_u_test @ u

    u_err = torch.sqrt((u_h - u_exact).square().mean())
    u_ref = torch.sqrt(u_exact.square().mean())
    rel_u = (u_err / u_ref).item() if u_ref > 0 else float("inf")

    M_err = torch.sqrt((FROBENIUS_WEIGHT * (M_h - M_exact).square()).sum(dim=1).mean())
    M_ref = torch.sqrt((FROBENIUS_WEIGHT * M_exact.square()).sum(dim=1).mean())
    rel_M = (M_err / M_ref).item() if M_ref > 0 else float("inf")
    return rel_u, rel_M


def compute_coefficient_residual_norms(
    data: FeatureEvaluationData,
    m: torch.Tensor,
    u: torch.Tensor,
) -> tuple[float, float, float, float]:
    """Evaluate strong-form residual norms on sampled interior and boundary points."""

    if not torch.isfinite(m).all() or not torch.isfinite(u).all():
        return float("nan"), float("nan"), float("nan"), float("nan")

    m_blocks = m.reshape(-1, 3)
    constitutive_sq = 0.0
    equilibrium_sq = 0.0
    boundary_value_sq = 0.0
    boundary_normal_sq = 0.0
    w_int = 1.0 / data.x_int.shape[0]

    with torch.no_grad():
        for start in range(0, data.x_int.shape[0], data.assembly_batch_size):
            end = min(start + data.assembly_batch_size, data.x_int.shape[0])
            xb = data.x_int[start:end]
            fb = data.f_int[start:end]

            xi_m_batch = eval_features(xb, data.a_m, data.r_m, data.gamma_m)
            hess_m_batch = eval_feature_hessians(xb, data.a_m, data.r_m, data.gamma_m)
            hess_u_batch = eval_feature_hessians(xb, data.a_u, data.r_u, data.gamma_u)

            M_h = xi_m_batch @ m_blocks
            hess_u = torch.einsum("qfj,f->qj", hess_u_batch, u)

            r_c = M_h @ data.compliance_voigt.T + hess_u
            r_e = (
                hess_m_batch[:, :, 0] @ m_blocks[:, 0]
                + hess_m_batch[:, :, 1] @ m_blocks[:, 1]
                + 2.0 * (hess_m_batch[:, :, 2] @ m_blocks[:, 2])
                + fb
            )

            constitutive_sq += (
                w_int * (FROBENIUS_WEIGHT * r_c.square()).sum(dim=1).sum().item()
            )
            equilibrium_sq += w_int * r_e.square().sum().item()

        for start in range(0, data.x_bc.shape[0], data.assembly_batch_size):
            end = min(start + data.assembly_batch_size, data.x_bc.shape[0])
            xb = data.x_bc[start:end]
            wb = data.w_bc[start:end]
            nb = data.n_bc[start:end]

            xi_u_batch = eval_features(xb, data.a_u, data.r_u, data.gamma_u)
            grad_u_batch = eval_feature_grads(xb, data.a_u, data.r_u, data.gamma_u)
            dn_xi_batch = (grad_u_batch * nb.unsqueeze(1)).sum(dim=2)

            u_bc = xi_u_batch @ u
            dn_u_bc = dn_xi_batch @ u
            boundary_value_sq += (wb * u_bc.square()).sum().item()
            boundary_normal_sq += (wb * dn_u_bc.square()).sum().item()

    return (
        constitutive_sq**0.5,
        equilibrium_sq**0.5,
        boundary_value_sq**0.5,
        boundary_normal_sq**0.5,
    )


def evaluate_feature_result(
    name: str,
    wall_time: float,
    m: torch.Tensor,
    u: torch.Tensor,
    data: FeatureEvaluationData,
) -> AlgorithmResult:
    """Evaluate one coefficient-based method and package the metrics."""

    r_c, r_e, r_b0, r_b1 = compute_coefficient_residual_norms(data, m, u)
    rel_u, rel_M = compute_l2_errors(
        data.xi_u_test,
        data.xi_m_test,
        m,
        u,
        data.u_exact_test,
        data.M_exact_test,
    )
    return AlgorithmResult(
        name=name,
        r_c=r_c,
        r_e=r_e,
        r_b0=r_b0,
        r_b1=r_b1,
        rel_u=rel_u,
        rel_M=rel_M,
        wall_time=wall_time,
    )


def print_result_summary(result: AlgorithmResult) -> None:
    """Print one compact result line."""

    print(
        f"    Done in {result.wall_time:.2f}s, "
        f"||r_c||={result.r_c:.2e}, "
        f"||r_e||={result.r_e:.2e}, "
        f"||r_b0||={result.r_b0:.2e}, "
        f"||r_b1||={result.r_b1:.2e}, "
        f"rel_u={result.rel_u:.2e}, "
        f"rel_M={result.rel_M:.2e}"
    )


def print_summary_table(
    results: Sequence[AlgorithmResult],
    title: str,
    include_residuals: bool = True,
) -> None:
    """Print a compact markdown-style summary table."""

    if not results:
        return

    print(f"\n=== {title} ===\n")
    if include_residuals:
        print(
            f"| {'Method':<18} | {'||r_c||':>10} | {'||r_e||':>10} | "
            f"{'||r_b0||':>10} | {'||r_b1||':>10} | {'rel_u':>10} | "
            f"{'rel_M':>10} | {'Time(s)':>8} |"
        )
        print(
            f"|:{'-'*19}|{'-'*11}:|{'-'*11}:|{'-'*11}:|{'-'*11}:|"
            f"{'-'*11}:|{'-'*11}:|{'-'*9}:|"
        )
        for result in results:
            print(
                f"| {result.name:<18} | {result.r_c:>10.2e} | {result.r_e:>10.2e} | "
                f"{result.r_b0:>10.2e} | {result.r_b1:>10.2e} | {result.rel_u:>10.2e} | "
                f"{result.rel_M:>10.2e} | {result.wall_time:>8.2f} |"
            )
        return

    print(
        f"| {'Method':<18} | {'rel_u':>10} | {'rel_M':>10} | {'Time(s)':>8} |"
    )
    print(f"|:{'-'*19}|{'-'*11}:|{'-'*11}:|{'-'*9}:|")
    for result in results:
        print(
            f"| {result.name:<18} | {result.rel_u:>10.2e} | "
            f"{result.rel_M:>10.2e} | {result.wall_time:>8.2f} |"
        )


def configure_plotting() -> None:
    """Apply the shared matplotlib settings used by experiment plots."""

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False


def order_results_by_label(
    results: Sequence[AlgorithmResult],
    ordered_labels: Sequence[str],
) -> list[AlgorithmResult]:
    """Reorder results to match a user-selected algorithm display order."""

    result_by_name = {result.name: result for result in results}
    ordered_results = [
        result_by_name[label]
        for label in ordered_labels
        if label in result_by_name
    ]
    ordered_name_set = set(ordered_labels)
    ordered_results.extend(
        result
        for result in results
        if result.name not in ordered_name_set
    )
    return ordered_results


def plot_l2_summary(
    results: Sequence[AlgorithmResult],
    ordered_labels: Sequence[str],
    save_path: str,
) -> None:
    """Plot final relative L2 errors as bar charts."""

    configure_plotting()
    ordered_results = order_results_by_label(results, ordered_labels)
    if not ordered_results:
        print(f"  Skipped: {save_path} (no results to plot)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    titles = [
        r"Deflection $\|\Phi^u - u_{ex}\|_{L^2} / \|u_{ex}\|_{L^2}$",
        r"Moment $\|\Phi^M - M_{ex}\|_{L^2} / \|M_{ex}\|_{L^2}$",
    ]
    keys = ["rel_u", "rel_M"]
    labels = [result.name for result in ordered_results]
    x_positions = np.arange(len(ordered_results), dtype=float)
    colors = [
        ALGO_STYLE.get(label, {}).get("color", "#4C78A8")
        for label in labels
    ]

    for ax, title, key in zip(axes, titles, keys):
        values = np.array(
            [getattr(result, key) for result in ordered_results],
            dtype=float,
        )
        valid = np.isfinite(values) & (values > 0.0)
        if valid.any():
            valid_indices = np.flatnonzero(valid)
            ax.bar(
                x_positions[valid],
                values[valid],
                width=0.65,
                color=[colors[index] for index in valid_indices],
            )

        invalid_indices = np.flatnonzero(~valid)
        for index in invalid_indices:
            print(
                f"  Skipped {labels[index]} {key}={values[index]!r} in {save_path}"
            )

        ax.set_yscale("log")
        ax.set_ylabel("Relative $L^2$ error")
        ax.set_title(title)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(alpha=0.3, linestyle="--", axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def apply_shared_to_weak_config(
    cfg: "WeakConfig",
    shared_cfg: SharedComparisonConfig,
    algorithm_ids: Sequence[str],
) -> "WeakConfig":
    """Override comparison-critical weak-form fields from the shared config."""

    return replace(
        cfg,
        E=shared_cfg.E,
        nu=shared_cfg.nu,
        h=shared_cfg.h,
        gamma_m=shared_cfg.gamma_m,
        gamma_u=shared_cfg.gamma_u,
        M_m=shared_cfg.M_m,
        M_u=shared_cfg.M_u,
        Q_int=shared_cfg.Q_int,
        Q_bc=shared_cfg.Q_bc,
        Q_test=shared_cfg.Q_test,
        sampling_method=shared_cfg.sampling_method,
        algorithms_to_run=list(algorithm_ids),
    )


def apply_shared_to_strong_config(
    cfg: "StrongConfig",
    shared_cfg: SharedComparisonConfig,
    algorithm_ids: Sequence[str],
) -> "StrongConfig":
    """Override comparison-critical strong-form fields from the shared config."""

    return replace(
        cfg,
        E=shared_cfg.E,
        nu=shared_cfg.nu,
        h=shared_cfg.h,
        gamma_m=shared_cfg.gamma_m,
        gamma_u=shared_cfg.gamma_u,
        M_m=shared_cfg.M_m,
        M_u=shared_cfg.M_u,
        Q_int=shared_cfg.Q_int,
        Q_bc=shared_cfg.Q_bc,
        Q_test=shared_cfg.Q_test,
        sampling_method=shared_cfg.sampling_method,
        algorithms_to_run=list(algorithm_ids),
    )


def make_default_main_config() -> MainConfig:
    """Construct a default top-level config without circular imports."""

    from projection import ProjectionConfig
    from strong_form import StrongConfig
    from weak_form import WeakConfig

    return MainConfig(
        shared=SharedComparisonConfig(),
        projection=ProjectionConfig(),
        weak=WeakConfig(),
        strong=StrongConfig(),
    )


def main(cfg: MainConfig | None = None) -> None:
    """Run the selected algorithms across the implemented experiment families."""

    cfg = make_default_main_config() if cfg is None else cfg
    shared_cfg = SharedComparisonConfig() if cfg.shared is None else cfg.shared
    validate_shared_comparison_config(shared_cfg)

    selected_algorithm_ids = validate_algorithm_selection(
        cfg.algorithms_to_run,
        VALID_TOP_LEVEL_ALGORITHMS,
    )
    ordered_labels = [
        TOP_LEVEL_ALGORITHM_LABELS[algorithm_id]
        for algorithm_id in selected_algorithm_ids
    ]

    print(f"Output: {OUTPUT_DIR}")
    print(
        "Shared comparison config: "
        f"h={shared_cfg.h}, M_m={shared_cfg.M_m}, M_u={shared_cfg.M_u}, "
        f"Q_int={shared_cfg.Q_int}, Q_bc={shared_cfg.Q_bc}, Q_test={shared_cfg.Q_test}, "
        f"sampling={shared_cfg.sampling_method}"
    )

    print("Building shared benchmark data...")
    benchmark = build_shared_benchmark(
        E=shared_cfg.E,
        nu=shared_cfg.nu,
        h=shared_cfg.h,
        Q_int=shared_cfg.Q_int,
        Q_bc=shared_cfg.Q_bc,
        Q_test=shared_cfg.Q_test,
        sampling_method=shared_cfg.sampling_method,
        interior_seed=shared_cfg.interior_seed,
        boundary_seed=shared_cfg.boundary_seed,
        test_seed=shared_cfg.test_seed,
    )

    feature_space: SharedFeatureSpace | None = None
    if selected_algorithm_ids:
        print("Generating shared random feature spaces...")
        feature_space = build_shared_feature_space(
            M_m=shared_cfg.M_m,
            M_u=shared_cfg.M_u,
            gamma_m=shared_cfg.gamma_m,
            gamma_u=shared_cfg.gamma_u,
            moment_feature_seed=shared_cfg.moment_feature_seed,
            disp_feature_seed=shared_cfg.disp_feature_seed,
        )

    results: list[AlgorithmResult] = []

    if "projection" in selected_algorithm_ids:
        if cfg.projection is None:
            raise ValueError("MainConfig.projection is required when running projection.")
        from projection import (
            apply_shared_to_projection_config,
            run_experiment as run_projection_experiment,
        )

        projection_cfg = apply_shared_to_projection_config(cfg.projection, shared_cfg)
        results.extend(
            run_projection_experiment(
                projection_cfg,
                print_table=False,
                benchmark=benchmark,
                feature_space=feature_space,
            )
        )

    weak_algorithm_ids = extract_scoped_algorithm_ids(selected_algorithm_ids, "weak")
    if weak_algorithm_ids:
        if cfg.weak is None:
            raise ValueError("MainConfig.weak is required when running weak-form algorithms.")
        from weak_form import run_experiment as run_weak_experiment

        weak_cfg = apply_shared_to_weak_config(cfg.weak, shared_cfg, weak_algorithm_ids)
        results.extend(
            run_weak_experiment(
                weak_cfg,
                print_table=False,
                benchmark=benchmark,
                feature_space=feature_space,
            )
        )

    strong_algorithm_ids = extract_scoped_algorithm_ids(selected_algorithm_ids, "strong")
    if strong_algorithm_ids:
        if cfg.strong is None:
            raise ValueError("MainConfig.strong is required when running strong-form algorithms.")
        from strong_form import run_experiment as run_strong_experiment

        strong_cfg = apply_shared_to_strong_config(
            cfg.strong,
            shared_cfg,
            strong_algorithm_ids,
        )
        results.extend(
            run_strong_experiment(
                strong_cfg,
                print_table=False,
                benchmark=benchmark,
                feature_space=feature_space,
            )
        )

    ordered_results = order_results_by_label(results, ordered_labels)
    print_summary_table(
        ordered_results,
        title="Overall Summary",
        include_residuals=False,
    )

    print("\nGenerating plots...")
    plot_l2_summary(
        ordered_results,
        ordered_labels,
        str(OUTPUT_DIR / "l2-error-summary.png"),
    )


if __name__ == "__main__":
    main()
