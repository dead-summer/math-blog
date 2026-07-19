from __future__ import annotations

import math
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import torch


BASE_SEED = 42
MOMENT_SEED = BASE_SEED
DEFLECTION_SEED = BASE_SEED + 1_000
DTYPE = torch.float64
VALID_SAMPLING_METHODS = ("mc", "sobol", "gauss_legendre")
VALID_ALGORITHMS = ("direct",)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "public" / "images" / "least-squares" / "plate-bending"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ALGO_STYLE = {
    "Direct LS (GELSD)": {"color": "#0077B6", "marker": "o", "linestyle": "-"},
}


def detect_device() -> torch.device:
    """Prefer CUDA when available and stay quiet otherwise."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


DEVICE = torch.device("cpu")

torch.manual_seed(BASE_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(BASE_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

FROBENIUS_WEIGHT = torch.tensor([1.0, 1.0, 2.0], dtype=DTYPE, device=DEVICE)
FROBENIUS_WEIGHT_MATRIX = torch.diag(FROBENIUS_WEIGHT)
DIVDIV_WEIGHTS = torch.tensor([1.0, 1.0, 2.0], dtype=DTYPE, device=DEVICE)


@dataclass(frozen=True)
class AlgorithmResult:
    """Compact metrics for one completed algorithm."""

    name: str
    r_c: float
    r_e: float
    rel_u: float
    rel_M: float
    wall_time: float
    rank: int = 0
    columns: int = 0
    condition_estimate: float = float("nan")


@dataclass(frozen=True)
class SharedBenchmarkData:
    """Shared train/test samples reused across selected algorithms."""

    x_int: torch.Tensor
    w_int: torch.Tensor
    f_int: torch.Tensor
    x_test: torch.Tensor
    w_test: torch.Tensor
    u_exact_test: torch.Tensor
    M_exact_test: torch.Tensor
    compliance_voigt: torch.Tensor


@dataclass(frozen=True)
class SharedFeatureSpace:
    """Shared random feature spaces used by coefficient-based methods."""

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
    w_int: torch.Tensor
    f_int: torch.Tensor
    a_m: torch.Tensor
    r_m: torch.Tensor
    a_u: torch.Tensor
    r_u: torch.Tensor
    gamma_m: float
    gamma_u: float
    compliance_voigt: torch.Tensor
    assembly_batch_size: int
    xi_m_test: torch.Tensor
    psi_u_test: torch.Tensor
    w_test: torch.Tensor
    u_exact_test: torch.Tensor
    M_exact_test: torch.Tensor


@dataclass
class LeastSquaresConfig:
    """Configuration for the conforming least-squares experiment."""

    E: float = 1.0
    nu: float = 0.3
    h: float = 1.0
    gamma_m: float = 2.0
    gamma_u: float = 2.0
    N_m: int = 1000
    N_u: int = 1000
    Q_train: int = 64**2
    Q_test: int = (2 ** 7) ** 2
    sampling_method: str = "gauss_legendre"
    direct_rcond: float = 1.0e-14
    assembly_batch_size: int = 5_000
    algorithms_to_run: list[str] = field(
        default_factory=lambda: ["direct"]
    )


@dataclass(frozen=True)
class LeastSquaresExperimentData:
    """All tensors needed to run and evaluate one least-squares solver."""

    matrix: torch.Tensor
    rhs: torch.Tensor
    dim_m: int
    eval_data: FeatureEvaluationData


@dataclass(frozen=True)
class DirectResidualDesign:
    """Weighted residual matrix and coefficient split metadata."""

    matrix: torch.Tensor
    rhs: torch.Tensor
    dim_m: int


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


def validate_config(cfg: LeastSquaresConfig) -> None:
    """Validate config before starting any expensive work."""

    if cfg.E <= 0.0:
        raise ValueError("Config.E must be positive.")
    if not (-1.0 < cfg.nu < 0.5):
        raise ValueError("Config.nu must lie in (-1, 0.5).")
    if cfg.h <= 0.0:
        raise ValueError("Config.h must be positive.")
    if cfg.gamma_m <= 0.0 or cfg.gamma_u <= 0.0:
        raise ValueError("Config.gamma_m and Config.gamma_u must be positive.")
    if cfg.N_m <= 0 or cfg.N_u <= 0:
        raise ValueError("Config.N_m and Config.N_u must be positive.")
    if cfg.Q_train <= 0:
        raise ValueError("Config.Q_train must be positive.")
    if cfg.Q_test <= 0:
        raise ValueError("Config.Q_test must be positive.")
    if not math.isfinite(cfg.direct_rcond) or cfg.direct_rcond <= 0.0:
        raise ValueError("Config.direct_rcond must be finite and positive.")
    if cfg.assembly_batch_size <= 0:
        raise ValueError("Config.assembly_batch_size must be positive.")
    validate_sampling_method(cfg.sampling_method)
    validate_algorithm_selection(cfg.algorithms_to_run, VALID_ALGORITHMS)


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


def eval_exact_deflection(x: torch.Tensor) -> torch.Tensor:
    """Evaluate the manufactured clamped deflection field."""

    return poly_p(x[:, 0]) * poly_p(x[:, 1])


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


def eval_zeta(x: torch.Tensor) -> torch.Tensor:
    """Evaluate the H_0^2 envelope zeta(x)."""

    a = x[:, 0] * (1.0 - x[:, 0])
    b = x[:, 1] * (1.0 - x[:, 1])
    return a.square() * b.square()


def eval_zeta_grad(x: torch.Tensor) -> torch.Tensor:
    """Evaluate the gradient of the H_0^2 envelope."""

    x1 = x[:, 0]
    x2 = x[:, 1]
    a = x1 * (1.0 - x1)
    b = x2 * (1.0 - x2)
    da = 1.0 - 2.0 * x1
    db = 1.0 - 2.0 * x2
    grad_x = 2.0 * a * da * b.square()
    grad_y = 2.0 * a.square() * b * db
    return torch.stack([grad_x, grad_y], dim=1)


def eval_zeta_hessian(x: torch.Tensor) -> torch.Tensor:
    """Evaluate Hessian components (11, 22, 12) of the H_0^2 envelope."""

    x1 = x[:, 0]
    x2 = x[:, 1]
    a = x1 * (1.0 - x1)
    b = x2 * (1.0 - x2)
    da = 1.0 - 2.0 * x1
    db = 1.0 - 2.0 * x2
    d2a = torch.full_like(x1, -2.0)
    d2b = torch.full_like(x2, -2.0)
    zeta_xx = 2.0 * (da.square() + a * d2a) * b.square()
    zeta_yy = 2.0 * a.square() * (db.square() + b * d2b)
    zeta_xy = 4.0 * a * da * b * db
    return torch.stack([zeta_xx, zeta_yy, zeta_xy], dim=1)


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
    dim: int = 2,
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


def generate_features(N: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random feature normals and offsets."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    raw = torch.randn(N, 2, generator=generator, dtype=DTYPE)
    norms = raw.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
    a = (raw / norms).to(DEVICE)
    r = torch.rand(N, generator=generator, dtype=DTYPE).to(DEVICE)
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


def eval_active_deflection_features(
    x: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Evaluate the conforming deflection basis psi = zeta * xi."""

    xi = eval_features(x, a, r, gamma)
    return eval_zeta(x).unsqueeze(1) * xi


def eval_active_deflection_feature_data(
    x: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate conforming deflection features and their Hessians."""

    xi = eval_features(x, a, r, gamma)
    grad_xi = eval_feature_grads(x, a, r, gamma)
    hess_xi = eval_feature_hessians(x, a, r, gamma)
    zeta = eval_zeta(x)
    grad_zeta = eval_zeta_grad(x)
    hess_zeta = eval_zeta_hessian(x)

    psi = zeta.unsqueeze(1) * xi
    hess_psi_xx = (
        xi * hess_zeta[:, 0].unsqueeze(1)
        + 2.0 * grad_xi[:, :, 0] * grad_zeta[:, 0].unsqueeze(1)
        + zeta.unsqueeze(1) * hess_xi[:, :, 0]
    )
    hess_psi_yy = (
        xi * hess_zeta[:, 1].unsqueeze(1)
        + 2.0 * grad_xi[:, :, 1] * grad_zeta[:, 1].unsqueeze(1)
        + zeta.unsqueeze(1) * hess_xi[:, :, 1]
    )
    hess_psi_xy = (
        xi * hess_zeta[:, 2].unsqueeze(1)
        + grad_xi[:, :, 0] * grad_zeta[:, 1].unsqueeze(1)
        + grad_xi[:, :, 1] * grad_zeta[:, 0].unsqueeze(1)
        + zeta.unsqueeze(1) * hess_xi[:, :, 2]
    )
    hess_psi = torch.stack([hess_psi_xx, hess_psi_yy, hess_psi_xy], dim=2)
    return psi, hess_psi


def build_shared_benchmark(
    E: float,
    nu: float,
    h: float,
    Q_train: int,
    Q_test: int,
    sampling_method: str,
    interior_seed: int = BASE_SEED + 1,
    test_seed: int = BASE_SEED + 3,
) -> SharedBenchmarkData:
    """Build the shared train/test samples for a fair comparison run."""

    D = compute_bending_stiffness(E, nu, h)
    compliance_voigt = build_compliance_matrix(D, nu)

    x_int, w_int = build_quadrature_rule(
        Q_train,
        method=sampling_method,
        seed=interior_seed,
    )
    f_int = compute_body_force(x_int, D)

    x_test, w_test = build_quadrature_rule(
        Q_test,
        method=sampling_method,
        seed=test_seed,
    )
    u_exact_test = eval_exact_deflection(x_test)
    M_exact_test = eval_exact_moment(x_test, D, nu)
    return SharedBenchmarkData(
        x_int=x_int,
        w_int=w_int,
        f_int=f_int,
        x_test=x_test,
        w_test=w_test,
        u_exact_test=u_exact_test,
        M_exact_test=M_exact_test,
        compliance_voigt=compliance_voigt,
    )


def build_shared_feature_space(
    N_m: int,
    N_u: int,
    gamma_m: float,
    gamma_u: float,
    moment_feature_seed: int = MOMENT_SEED,
    deflection_feature_seed: int = DEFLECTION_SEED,
) -> SharedFeatureSpace:
    """Build the shared random feature spaces used by coefficient-based methods."""

    a_m, r_m = generate_features(N_m, seed=moment_feature_seed)
    a_u, r_u = generate_features(N_u, seed=deflection_feature_seed)
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
        w_int=benchmark.w_int,
        f_int=benchmark.f_int,
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
        psi_u_test=eval_active_deflection_features(
            benchmark.x_test,
            feature_space.a_u,
            feature_space.r_u,
            feature_space.gamma_u,
        ),
        w_test=benchmark.w_test,
        u_exact_test=benchmark.u_exact_test,
        M_exact_test=benchmark.M_exact_test,
    )


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


def accumulate_interior_moments(
    x_int: torch.Tensor,
    w_int: torch.Tensor,
    f_int: torch.Tensor,
    a_m: torch.Tensor,
    r_m: torch.Tensor,
    gamma_m: float,
    a_u: torch.Tensor,
    r_u: torch.Tensor,
    gamma_u: float,
    batch_size: int,
) -> tuple[
    torch.Tensor,
    list[torch.Tensor],
    list[list[torch.Tensor]],
    list[list[torch.Tensor]],
    list[torch.Tensor],
]:
    """Accumulate moments for the least-squares linear system."""

    mp1_m = a_m.shape[0] + 1
    mp1_u = a_u.shape[0] + 1

    gram_xi_m = torch.zeros(mp1_m, mp1_m, dtype=DTYPE, device=DEVICE)
    cross_xi_m_hess_psi = [
        torch.zeros(mp1_m, mp1_u, dtype=DTYPE, device=DEVICE) for _ in range(3)
    ]
    hess_gram_psi = [
        [
            torch.zeros(mp1_u, mp1_u, dtype=DTYPE, device=DEVICE)
            for _ in range(3)
        ]
        for _ in range(3)
    ]
    hess_gram_m = [
        [
            torch.zeros(mp1_m, mp1_m, dtype=DTYPE, device=DEVICE)
            for _ in range(3)
        ]
        for _ in range(3)
    ]
    hess_force_m = [
        torch.zeros(mp1_m, dtype=DTYPE, device=DEVICE) for _ in range(3)
    ]

    with torch.no_grad():
        for start in range(0, x_int.shape[0], batch_size):
            end = min(start + batch_size, x_int.shape[0])
            xb = x_int[start:end]
            wb = w_int[start:end]
            fb = f_int[start:end]

            xi_m_batch = eval_features(xb, a_m, r_m, gamma_m)
            hess_m_batch = eval_feature_hessians(xb, a_m, r_m, gamma_m)
            _, hess_psi_batch = eval_active_deflection_feature_data(
                xb,
                a_u,
                r_u,
                gamma_u,
            )

            weighted_xi_m = wb.unsqueeze(1) * xi_m_batch
            weighted_hess_psi = [
                wb.unsqueeze(1) * hess_psi_batch[:, :, comp_i] for comp_i in range(3)
            ]
            weighted_hess_m = [
                wb.unsqueeze(1) * hess_m_batch[:, :, comp_i] for comp_i in range(3)
            ]

            gram_xi_m += xi_m_batch.T @ weighted_xi_m
            for comp_i in range(3):
                cross_xi_m_hess_psi[comp_i] += xi_m_batch.T @ weighted_hess_psi[comp_i]
                hess_force_m[comp_i] += weighted_hess_m[comp_i].T @ fb
                for comp_j in range(3):
                    hess_gram_psi[comp_i][comp_j] += (
                        hess_psi_batch[:, :, comp_i].T @ weighted_hess_psi[comp_j]
                    )
                    hess_gram_m[comp_i][comp_j] += (
                        hess_m_batch[:, :, comp_i].T @ weighted_hess_m[comp_j]
                    )

    return (
        gram_xi_m,
        cross_xi_m_hess_psi,
        hess_gram_psi,
        hess_gram_m,
        hess_force_m,
    )


def assemble_linear_system(
    cfg: LeastSquaresConfig,
    compliance_voigt: torch.Tensor,
    x_int: torch.Tensor,
    w_int: torch.Tensor,
    f_int: torch.Tensor,
    a_m: torch.Tensor,
    r_m: torch.Tensor,
    a_u: torch.Tensor,
    r_u: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble the conforming least-squares system G z = F."""

    (
        gram_xi_m,
        cross_xi_m_hess_psi,
        hess_gram_psi,
        hess_gram_m,
        hess_force_m,
    ) = accumulate_interior_moments(
        x_int,
        w_int,
        f_int,
        a_m,
        r_m,
        cfg.gamma_m,
        a_u,
        r_u,
        cfg.gamma_u,
        cfg.assembly_batch_size,
    )

    mp1_m = a_m.shape[0] + 1
    mp1_u = a_u.shape[0] + 1
    dim_m = 3 * mp1_m

    G_mm = torch.zeros(dim_m, dim_m, dtype=DTYPE, device=DEVICE)
    G_mu = torch.zeros(dim_m, mp1_u, dtype=DTYPE, device=DEVICE)
    G_uu = torch.zeros(mp1_u, mp1_u, dtype=DTYPE, device=DEVICE)
    F_m = torch.zeros(dim_m, dtype=DTYPE, device=DEVICE)

    constitutive_mm = compliance_voigt.T @ FROBENIUS_WEIGHT_MATRIX @ compliance_voigt
    constitutive_mu = compliance_voigt.T @ FROBENIUS_WEIGHT_MATRIX

    add_block_scaled(
        G_mm,
        gram_xi_m,
        constitutive_mm,
        row_stride=3,
        col_stride=3,
    )

    for comp_i in range(3):
        for comp_j in range(3):
            G_mm[comp_i::3, comp_j::3] += (
                DIVDIV_WEIGHTS[comp_i]
                * DIVDIV_WEIGHTS[comp_j]
                * hess_gram_m[comp_i][comp_j]
            )
            G_mu[comp_i::3, :] += (
                constitutive_mu[comp_i, comp_j] * cross_xi_m_hess_psi[comp_j]
            )
            G_uu += (
                FROBENIUS_WEIGHT_MATRIX[comp_i, comp_j]
                * hess_gram_psi[comp_i][comp_j]
            )

        F_m[comp_i::3] = -DIVDIV_WEIGHTS[comp_i] * hess_force_m[comp_i]

    G = torch.zeros(dim_m + mp1_u, dim_m + mp1_u, dtype=DTYPE, device=DEVICE)
    G[:dim_m, :dim_m] = G_mm
    G[:dim_m, dim_m:] = G_mu
    G[dim_m:, :dim_m] = G_mu.T
    G[dim_m:, dim_m:] = G_uu
    G = 0.5 * (G + G.T)

    F = torch.zeros(dim_m + mp1_u, dtype=DTYPE, device=DEVICE)
    F[:dim_m] = F_m
    return G, F


def assemble_direct_residual_design(
    benchmark: SharedBenchmarkData,
    feature_space: SharedFeatureSpace,
) -> DirectResidualDesign:
    """Assemble the weighted constitutive and equilibrium residuals directly."""

    x = benchmark.x_int
    sqrt_weights = torch.sqrt(benchmark.w_int)
    xi_m = eval_features(
        x,
        feature_space.a_m,
        feature_space.r_m,
        feature_space.gamma_m,
    )
    hess_m = eval_feature_hessians(
        x,
        feature_space.a_m,
        feature_space.r_m,
        feature_space.gamma_m,
    )
    _, hess_u = eval_active_deflection_feature_data(
        x,
        feature_space.a_u,
        feature_space.r_u,
        feature_space.gamma_u,
    )

    q_count = x.shape[0]
    dim_m = 3 * xi_m.shape[1]
    matrix = torch.zeros(
        4 * q_count,
        dim_m + hess_u.shape[1],
        dtype=DTYPE,
        device=DEVICE,
    )
    rhs = torch.zeros(4 * q_count, dtype=DTYPE, device=DEVICE)
    sqrt_frobenius = torch.sqrt(FROBENIUS_WEIGHT)
    weighted_xi = sqrt_weights.unsqueeze(1) * xi_m

    for residual_component in range(3):
        rows = slice(residual_component * q_count, (residual_component + 1) * q_count)
        residual_scale = sqrt_frobenius[residual_component]
        for moment_component in range(3):
            matrix[rows, moment_component:dim_m:3] = (
                residual_scale
                * benchmark.compliance_voigt[residual_component, moment_component]
                * weighted_xi
            )
        matrix[rows, dim_m:] = (
            residual_scale
            * sqrt_weights.unsqueeze(1)
            * hess_u[:, :, residual_component]
        )

    equilibrium_rows = slice(3 * q_count, 4 * q_count)
    for moment_component in range(3):
        matrix[equilibrium_rows, moment_component:dim_m:3] = (
            DIVDIV_WEIGHTS[moment_component]
            * sqrt_weights.unsqueeze(1)
            * hess_m[:, :, moment_component]
        )
    rhs[equilibrium_rows] = -sqrt_weights * benchmark.f_int
    return DirectResidualDesign(matrix, rhs, dim_m)


def solve_direct_residual(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    rcond: float,
) -> tuple[torch.Tensor, float, int, float]:
    """Column-scale and solve the residual least-squares problem with GELSD."""

    column_norms = torch.linalg.vector_norm(matrix, dim=0)
    floor = torch.finfo(matrix.dtype).eps * column_norms.max()
    safe_norms = column_norms.clamp_min(floor)
    matrix.div_(safe_norms.unsqueeze(0))
    t0 = time.perf_counter()
    scaled_solution, _, rank, singular_values = scipy.linalg.lstsq(
        matrix.numpy(),
        rhs.numpy(),
        cond=rcond,
        overwrite_a=True,
        overwrite_b=False,
        check_finite=False,
        lapack_driver="gelsd",
    )
    wall_time = time.perf_counter() - t0
    solution = torch.from_numpy(scaled_solution).to(dtype=DTYPE) / safe_norms
    positive = singular_values[singular_values > 0.0]
    condition_estimate = (
        float(positive.max() / positive.min()) if len(positive) else float("inf")
    )
    return solution, wall_time, int(rank), condition_estimate


def split_solution(z: torch.Tensor, dim_m: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the full coefficient vector into moment and deflection parts."""

    return z[:dim_m], z[dim_m:]


def compute_l2_errors(
    psi_u_test: torch.Tensor,
    xi_m_test: torch.Tensor,
    moment_coeffs: torch.Tensor,
    deflection_coeffs: torch.Tensor,
    w_test: torch.Tensor,
    u_exact: torch.Tensor,
    M_exact: torch.Tensor,
) -> tuple[float, float]:
    """Compute relative L2 errors for deflection and moments."""

    M_h = xi_m_test @ moment_coeffs.reshape(-1, 3)
    u_h = psi_u_test @ deflection_coeffs

    u_err = torch.sqrt((w_test * (u_h - u_exact).square()).sum())
    u_ref = torch.sqrt((w_test * u_exact.square()).sum())
    rel_u = (u_err / u_ref).item() if u_ref > 0 else float("inf")

    M_err = torch.sqrt(
        (w_test * (FROBENIUS_WEIGHT * (M_h - M_exact).square()).sum(dim=1)).sum()
    )
    M_ref = torch.sqrt(
        (w_test * (FROBENIUS_WEIGHT * M_exact.square()).sum(dim=1)).sum()
    )
    rel_M = (M_err / M_ref).item() if M_ref > 0 else float("inf")
    return rel_u, rel_M


def compute_coefficient_residual_norms(
    data: FeatureEvaluationData,
    moment_coeffs: torch.Tensor,
    deflection_coeffs: torch.Tensor,
) -> tuple[float, float]:
    """Evaluate least-squares residual norms on sampled interior points."""

    if not torch.isfinite(moment_coeffs).all() or not torch.isfinite(deflection_coeffs).all():
        return float("nan"), float("nan")

    moment_blocks = moment_coeffs.reshape(-1, 3)
    constitutive_sq = 0.0
    equilibrium_sq = 0.0

    with torch.no_grad():
        for start in range(0, data.x_int.shape[0], data.assembly_batch_size):
            end = min(start + data.assembly_batch_size, data.x_int.shape[0])
            xb = data.x_int[start:end]
            wb = data.w_int[start:end]
            fb = data.f_int[start:end]

            xi_m_batch = eval_features(xb, data.a_m, data.r_m, data.gamma_m)
            hess_m_batch = eval_feature_hessians(xb, data.a_m, data.r_m, data.gamma_m)
            _, hess_psi_batch = eval_active_deflection_feature_data(
                xb,
                data.a_u,
                data.r_u,
                data.gamma_u,
            )

            M_h = xi_m_batch @ moment_blocks
            hess_u = torch.einsum("qfj,f->qj", hess_psi_batch, deflection_coeffs)
            r_c = M_h @ data.compliance_voigt.T + hess_u
            r_e = (
                hess_m_batch[:, :, 0] @ moment_blocks[:, 0]
                + hess_m_batch[:, :, 1] @ moment_blocks[:, 1]
                + 2.0 * (hess_m_batch[:, :, 2] @ moment_blocks[:, 2])
                + fb
            )

            constitutive_sq += (
                wb * (FROBENIUS_WEIGHT * r_c.square()).sum(dim=1)
            ).sum().item()
            equilibrium_sq += (wb * r_e.square()).sum().item()

    return constitutive_sq**0.5, equilibrium_sq**0.5


def evaluate_feature_result(
    name: str,
    wall_time: float,
    moment_coeffs: torch.Tensor,
    deflection_coeffs: torch.Tensor,
    data: FeatureEvaluationData,
) -> AlgorithmResult:
    """Evaluate one coefficient-based method and package the metrics."""

    r_c, r_e = compute_coefficient_residual_norms(
        data,
        moment_coeffs,
        deflection_coeffs,
    )
    rel_u, rel_M = compute_l2_errors(
        data.psi_u_test,
        data.xi_m_test,
        moment_coeffs,
        deflection_coeffs,
        data.w_test,
        data.u_exact_test,
        data.M_exact_test,
    )
    return AlgorithmResult(
        name=name,
        r_c=r_c,
        r_e=r_e,
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
        f"rel_u={result.rel_u:.2e}, "
        f"rel_M={result.rel_M:.2e}, "
        f"rank={result.rank}/{result.columns}, "
        f"cond≈{result.condition_estimate:.2e}"
    )


def print_summary_table(
    results: Sequence[AlgorithmResult],
    title: str,
) -> None:
    """Print a compact markdown-style summary table."""

    if not results:
        return

    method_width = max(18, max(len(result.name) for result in results))
    print(f"\n=== {title} ===\n")
    print(
        f"| {'Method':<{method_width}} | {'rel_u':>10} | "
        f"{'rel_M':>10} | {'Time(s)':>8} |"
    )
    print(
        f"|:{'-' * (method_width + 1)}|{'-' * 11}:|"
        f"{'-' * 11}:|{'-' * 9}:|"
    )
    for result in results:
        print(
            f"| {result.name:<{method_width}} | {result.rel_u:>10.2e} | "
            f"{result.rel_M:>10.2e} | {result.wall_time:>8.2f} |"
        )


def configure_plotting() -> None:
    """Apply the shared matplotlib settings used by experiment plots."""

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_l2_summary(
    results: Sequence[AlgorithmResult],
    save_path: str,
) -> None:
    """Plot final relative L2 errors as bar charts."""

    if not results:
        print(f"  Skipped: {save_path} (no results to plot)")
        return

    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    titles = [
        r"Deflection $\|\Phi^u - u_{ex}\|_{L^2} / \|u_{ex}\|_{L^2}$",
        r"Moment $\|\Phi^M - M_{ex}\|_{L^2} / \|M_{ex}\|_{L^2}$",
    ]
    keys = ["rel_u", "rel_M"]
    labels = [result.name for result in results]
    x_positions = np.arange(len(results), dtype=float)
    colors = [
        ALGO_STYLE.get(label, {}).get("color", "#4C78A8")
        for label in labels
    ]

    for ax, title, key in zip(axes, titles, keys):
        values = np.array([getattr(result, key) for result in results], dtype=float)
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
            print(f"  Skipped {labels[index]} {key}={values[index]!r} in {save_path}")

        ax.set_yscale("log")
        ax.set_ylabel("Relative $L^2$ error")
        ax.set_title(title)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(alpha=0.3, linestyle="--", axis="y")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def run_algorithm(
    algorithm_id: str,
    data: LeastSquaresExperimentData,
    cfg: LeastSquaresConfig,
) -> AlgorithmResult:
    """Run one configured least-squares algorithm and evaluate it."""

    if algorithm_id != "direct":
        raise ValueError(f"Unsupported algorithm: {algorithm_id}")
    print("Running Direct LS (GELSD)...")
    z, wall_time, rank, condition_estimate = solve_direct_residual(
        data.matrix,
        data.rhs,
        cfg.direct_rcond,
    )
    moment_coeffs, deflection_coeffs = split_solution(z, data.dim_m)
    evaluated = evaluate_feature_result(
        "Direct LS (GELSD)",
        wall_time,
        moment_coeffs,
        deflection_coeffs,
        data.eval_data,
    )
    result = AlgorithmResult(
        name=evaluated.name,
        r_c=evaluated.r_c,
        r_e=evaluated.r_e,
        rel_u=evaluated.rel_u,
        rel_M=evaluated.rel_M,
        wall_time=evaluated.wall_time,
        rank=rank,
        columns=data.matrix.shape[1],
        condition_estimate=condition_estimate,
    )

    print_result_summary(result)
    return result


def run_experiment(
    cfg: LeastSquaresConfig | None = None,
    print_table: bool = True,
    plot_results: bool = True,
    benchmark: SharedBenchmarkData | None = None,
    feature_space: SharedFeatureSpace | None = None,
) -> list[AlgorithmResult]:
    """Run direct residual least squares and return its metrics."""

    cfg = LeastSquaresConfig() if cfg is None else cfg
    validate_config(cfg)
    selected_algorithm_ids = validate_algorithm_selection(
        cfg.algorithms_to_run,
        VALID_ALGORITHMS,
    )

    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}")
    print(
        f"Config: h={cfg.h}, N_m={cfg.N_m}, N_u={cfg.N_u}, "
        f"Q_train={cfg.Q_train}, Q_test={cfg.Q_test}, "
        f"gamma_m={cfg.gamma_m}, gamma_u={cfg.gamma_u}, "
        f"direct_rcond={cfg.direct_rcond:.2e}, "
        f"sampling={cfg.sampling_method}"
    )
    print(f"Algorithms: {selected_algorithm_ids}")

    D = compute_bending_stiffness(cfg.E, cfg.nu, cfg.h)
    print(f"Material: E={cfg.E}, nu={cfg.nu}, h={cfg.h}, D={D:.4f}")

    if benchmark is None:
        print("Building benchmark data...")
        benchmark = build_shared_benchmark(
            E=cfg.E,
            nu=cfg.nu,
            h=cfg.h,
            Q_train=cfg.Q_train,
            Q_test=cfg.Q_test,
            sampling_method=cfg.sampling_method,
        )
    else:
        print("Using shared benchmark data...")

    if feature_space is None:
        print("Generating random feature spaces...")
        feature_space = build_shared_feature_space(
            N_m=cfg.N_m,
            N_u=cfg.N_u,
            gamma_m=cfg.gamma_m,
            gamma_u=cfg.gamma_u,
        )
    else:
        print("Using shared random feature spaces...")

    if feature_space.gamma_m != cfg.gamma_m or feature_space.gamma_u != cfg.gamma_u:
        raise ValueError("SharedFeatureSpace gamma does not match LeastSquaresConfig.")
    if feature_space.a_m.shape[0] != cfg.N_m or feature_space.a_u.shape[0] != cfg.N_u:
        raise ValueError("SharedFeatureSpace feature counts do not match LeastSquaresConfig.")

    print("Assembling direct weighted residual matrix...")
    t0 = time.perf_counter()
    direct_design = assemble_direct_residual_design(benchmark, feature_space)
    assembly_time = time.perf_counter() - t0
    clear_cuda_cache()

    print(
        f"Residual shapes: A={tuple(direct_design.matrix.shape)}, "
        f"b={tuple(direct_design.rhs.shape)}, assembly={assembly_time:.2f}s"
    )

    experiment_data = LeastSquaresExperimentData(
        matrix=direct_design.matrix,
        rhs=direct_design.rhs,
        dim_m=direct_design.dim_m,
        eval_data=build_feature_evaluation_data(
            benchmark,
            feature_space,
            cfg.assembly_batch_size,
        ),
    )

    results = [
        run_algorithm(algorithm_id, experiment_data, cfg)
        for algorithm_id in selected_algorithm_ids
    ]
    if print_table:
        print_summary_table(results, title="LS Summary")

    if plot_results:
        print("\nGenerating plots...")
        plot_l2_summary(results, str(OUTPUT_DIR / "l2-error-summary.png"))

    return results


def main(cfg: LeastSquaresConfig | None = None) -> None:
    """Script entrypoint."""

    run_experiment(cfg, print_table=True, plot_results=True)


if __name__ == "__main__":
    main()
