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
STRESS_SEED = BASE_SEED
DISP_SEED = BASE_SEED + 1_000
DTYPE = torch.float64
VALID_SAMPLING_METHODS = ("mc", "sobol", "gauss_legendre")
VALID_ALGORITHMS = ("direct",)
FEATURE_DIM = 2
FEATURE_CENTER = 0.5
FEATURE_INV_RADIUS = 2.0 / math.sqrt(FEATURE_DIM)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "public" / "images" / "least-squares" / "plane-stress"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ALGO_STYLE = {
    "Direct LS (GELSD)": {"color": "#0077B6", "marker": "o", "linestyle": "-"},
}
VOIGT_WEIGHT = torch.tensor([1.0, 1.0, 2.0], dtype=DTYPE)


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


def build_strain_gradient_bases() -> torch.Tensor:
    """Return the two fixed gradient-to-strain coupling blocks."""

    base_1 = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )
    base_2 = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )
    return torch.stack([base_1, base_2], dim=0)


STRAIN_GRAD_BASES = build_strain_gradient_bases()


@dataclass(frozen=True)
class AlgorithmResult:
    """Compact metrics for one completed algorithm."""

    name: str
    u_l2_error: float
    sigma_l2_error: float
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
    sigma_exact_test: torch.Tensor
    compliance_voigt: torch.Tensor


@dataclass(frozen=True)
class SharedFeatureSpace:
    """Shared random feature spaces used by coefficient-based methods."""

    a_s: torch.Tensor
    r_s: torch.Tensor
    a_u: torch.Tensor
    r_u: torch.Tensor
    gamma_s: float
    gamma_u: float


@dataclass(frozen=True)
class FeatureEvaluationData:
    """All tensors needed to evaluate coefficient-based methods."""

    xi_s_test: torch.Tensor
    psi_u_test: torch.Tensor
    w_test: torch.Tensor
    u_exact_test: torch.Tensor
    sigma_exact_test: torch.Tensor


@dataclass
class LeastSquaresConfig:
    """Configuration for the conforming least-squares experiment."""

    E: float = 1.5
    nu: float = 0.5
    gamma_s: float = 3.0
    gamma_u: float = 3.0
    N_s: int = 1000
    N_u: int = 1000
    Q_train: int = 64**2
    Q_test: int = (2 ** 7) ** 2
    sampling_method: str = "gauss_legendre"
    direct_rcond: float = 1.0e-14
    body_force_batch_size: int = 5_000
    assembly_batch_size: int = 5_000
    algorithms_to_run: list[str] = field(
        default_factory=lambda: ["direct"]
    )


@dataclass(frozen=True)
class LeastSquaresExperimentData:
    """All tensors needed to run and evaluate one least-squares solver."""

    matrix: torch.Tensor
    rhs: torch.Tensor
    dim_s: int
    eval_data: FeatureEvaluationData


@dataclass(frozen=True)
class DirectResidualDesign:
    """Weighted residual matrix and coefficient split metadata."""

    matrix: torch.Tensor
    rhs: torch.Tensor
    dim_s: int


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
    if not (-1.0 < cfg.nu <= 0.5):
        raise ValueError("Config.nu must lie in (-1, 0.5].")
    if cfg.gamma_s <= 0.0 or cfg.gamma_u <= 0.0:
        raise ValueError("Config.gamma_s and Config.gamma_u must be positive.")
    if cfg.N_s <= 0 or cfg.N_u <= 0:
        raise ValueError("Config.N_s and Config.N_u must be positive.")
    if cfg.Q_train <= 0:
        raise ValueError("Config.Q_train must be positive.")
    if cfg.Q_test <= 0:
        raise ValueError("Config.Q_test must be positive.")
    if not math.isfinite(cfg.direct_rcond) or cfg.direct_rcond <= 0.0:
        raise ValueError("Config.direct_rcond must be finite and positive.")
    if cfg.body_force_batch_size <= 0:
        raise ValueError("Config.body_force_batch_size must be positive.")
    if cfg.assembly_batch_size <= 0:
        raise ValueError("Config.assembly_batch_size must be positive.")
    validate_sampling_method(cfg.sampling_method)
    validate_algorithm_selection(cfg.algorithms_to_run, VALID_ALGORITHMS)


def compute_plane_stress_parameters(E: float, nu: float) -> tuple[float, float]:
    """Return (mu, lambda_plane) for the plane-stress constitutive law."""

    mu = E / (2.0 * (1.0 + nu))
    lambda_plane = E * nu / (1.0 - nu * nu)
    return mu, lambda_plane


def build_compliance_matrix(E: float, nu: float) -> torch.Tensor:
    """Build the 3x3 plane-stress compliance matrix in Voigt form."""

    compliance_voigt = torch.zeros(3, 3, dtype=DTYPE, device=DEVICE)
    compliance_voigt[0, 0] = 1.0 / E
    compliance_voigt[1, 1] = 1.0 / E
    compliance_voigt[0, 1] = -nu / E
    compliance_voigt[1, 0] = -nu / E
    compliance_voigt[2, 2] = 2.0 * (1.0 + nu) / E
    return compliance_voigt


def eval_exact_displacement(x: torch.Tensor) -> torch.Tensor:
    """Evaluate the manufactured in-plane displacement field."""

    x1, x2 = x[:, 0], x[:, 1]
    pi = math.pi
    boundary_factor = x1 * (1.0 - x1) * x2 * (1.0 - x2)
    u1 = torch.exp(x1 - x2) * boundary_factor
    u2 = torch.sin(pi * x1) * torch.sin(pi * x2)
    return torch.stack([u1, u2], dim=1)


def compute_engineering_strain(grad_u: torch.Tensor) -> torch.Tensor:
    """Convert displacement gradients to plane-stress Voigt strain."""

    return torch.stack(
        [
            grad_u[:, 0, 0],
            grad_u[:, 1, 1],
            grad_u[:, 0, 1] + grad_u[:, 1, 0],
        ],
        dim=1,
    )


def apply_plane_stress_stiffness(
    strain_voigt: torch.Tensor,
    mu: float,
    lambda_plane: float,
) -> torch.Tensor:
    """Apply the plane-stress stiffness operator to engineering strain."""

    sigma11 = (2.0 * mu + lambda_plane) * strain_voigt[:, 0]
    sigma11 = sigma11 + lambda_plane * strain_voigt[:, 1]
    sigma22 = lambda_plane * strain_voigt[:, 0]
    sigma22 = sigma22 + (2.0 * mu + lambda_plane) * strain_voigt[:, 1]
    sigma12 = mu * strain_voigt[:, 2]
    return torch.stack([sigma11, sigma22, sigma12], dim=1)


def compute_stress_voigt(
    x: torch.Tensor,
    mu: float,
    lambda_plane: float,
) -> torch.Tensor:
    """Evaluate the exact plane-stress field in Voigt order (11, 22, 12)."""

    x_ad = x.detach().requires_grad_(True)
    u = eval_exact_displacement(x_ad)

    n_points = x.shape[0]
    grad_u = torch.zeros(n_points, 2, 2, dtype=DTYPE, device=DEVICE)
    for comp in range(2):
        grad_u[:, comp, :] = torch.autograd.grad(
            u[:, comp].sum(),
            x_ad,
            create_graph=False,
            retain_graph=(comp < 1),
        )[0]

    strain_voigt = compute_engineering_strain(grad_u)
    return apply_plane_stress_stiffness(strain_voigt, mu, lambda_plane).detach()


def compute_body_force(
    x: torch.Tensor,
    mu: float,
    lambda_plane: float,
    batch_size: int,
) -> torch.Tensor:
    """Compute f = -div(sigma(u_exact)) with batched autodiff."""

    n_points = x.shape[0]
    f_all = torch.zeros(n_points, 2, dtype=DTYPE, device=DEVICE)

    for start in range(0, n_points, batch_size):
        end = min(start + batch_size, n_points)
        xb = x[start:end].detach().requires_grad_(True)
        u = eval_exact_displacement(xb)

        grad_u = torch.stack(
            [
                torch.autograd.grad(
                    u[:, comp].sum(),
                    xb,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                for comp in range(2)
            ],
            dim=1,
        )
        strain_voigt = compute_engineering_strain(grad_u)
        sigma = apply_plane_stress_stiffness(strain_voigt, mu, lambda_plane)

        grad_sigma11 = torch.autograd.grad(
            sigma[:, 0].sum(),
            xb,
            create_graph=False,
            retain_graph=True,
        )[0]
        grad_sigma22 = torch.autograd.grad(
            sigma[:, 1].sum(),
            xb,
            create_graph=False,
            retain_graph=True,
        )[0]
        grad_sigma12 = torch.autograd.grad(
            sigma[:, 2].sum(),
            xb,
            create_graph=False,
            retain_graph=False,
        )[0]

        f_all[start:end, 0] = -(grad_sigma11[:, 0] + grad_sigma12[:, 1]).detach()
        f_all[start:end, 1] = -(grad_sigma12[:, 0] + grad_sigma22[:, 1]).detach()

    return f_all


def eval_zeta(x: torch.Tensor) -> torch.Tensor:
    """Evaluate the Dirichlet envelope zeta(x)."""

    return x[:, 0] * (1.0 - x[:, 0]) * x[:, 1] * (1.0 - x[:, 1])


def eval_zeta_grad(x: torch.Tensor) -> torch.Tensor:
    """Evaluate the gradient of the Dirichlet envelope."""

    x1, x2 = x[:, 0], x[:, 1]
    grad_x = (1.0 - 2.0 * x1) * x2 * (1.0 - x2)
    grad_y = x1 * (1.0 - x1) * (1.0 - 2.0 * x2)
    return torch.stack([grad_x, grad_y], dim=1)


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


def normalize_feature_coordinates(x: torch.Tensor) -> torch.Tensor:
    """Map unit-box coordinates into the centered unit ball for features."""

    return (x - FEATURE_CENTER) * FEATURE_INV_RADIUS


def eval_features(
    x: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Evaluate xi_0 = 1 and xi_m = tanh(gamma (a_m^T x_hat + r_m))."""

    x_hat = normalize_feature_coordinates(x)
    pre = x_hat @ a.T + r.unsqueeze(0)
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

    x_hat = normalize_feature_coordinates(x)
    pre = x_hat @ a.T + r.unsqueeze(0)
    dtanh = 1.0 - torch.tanh(gamma * pre).square()
    grad_xi = gamma * FEATURE_INV_RADIUS * dtanh.unsqueeze(2) * a.unsqueeze(0)
    zeros = torch.zeros(x.shape[0], 1, 2, dtype=DTYPE, device=DEVICE)
    return torch.cat([zeros, grad_xi], dim=1)


def eval_active_displacement_features(
    x: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Evaluate the conforming displacement basis psi = zeta * xi."""

    xi = eval_features(x, a, r, gamma)
    return eval_zeta(x).unsqueeze(1) * xi


def eval_active_displacement_feature_data(
    x: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate conforming displacement features and their gradients."""

    xi = eval_features(x, a, r, gamma)
    grad_xi = eval_feature_grads(x, a, r, gamma)
    zeta = eval_zeta(x)
    grad_zeta = eval_zeta_grad(x)
    psi = zeta.unsqueeze(1) * xi
    grad_psi = (
        xi.unsqueeze(2) * grad_zeta.unsqueeze(1)
        + zeta.view(-1, 1, 1) * grad_xi
    )
    return psi, grad_psi


def build_shared_benchmark(
    E: float,
    nu: float,
    Q_train: int,
    Q_test: int,
    sampling_method: str,
    body_force_batch_size: int,
    interior_seed: int = BASE_SEED + 1,
    test_seed: int = BASE_SEED + 3,
) -> SharedBenchmarkData:
    """Build the shared train/test samples for a fair comparison run."""

    mu, lambda_plane = compute_plane_stress_parameters(E, nu)
    compliance_voigt = build_compliance_matrix(E, nu)

    x_int, w_int = build_quadrature_rule(
        Q_train,
        method=sampling_method,
        seed=interior_seed,
    )
    f_int = compute_body_force(x_int, mu, lambda_plane, batch_size=body_force_batch_size)

    x_test, w_test = build_quadrature_rule(
        Q_test,
        method=sampling_method,
        seed=test_seed,
    )
    u_exact_test = eval_exact_displacement(x_test)
    sigma_exact_test = compute_stress_voigt(x_test, mu, lambda_plane)
    return SharedBenchmarkData(
        x_int=x_int,
        w_int=w_int,
        f_int=f_int,
        x_test=x_test,
        w_test=w_test,
        u_exact_test=u_exact_test,
        sigma_exact_test=sigma_exact_test,
        compliance_voigt=compliance_voigt,
    )


def build_shared_feature_space(
    N_s: int,
    N_u: int,
    gamma_s: float,
    gamma_u: float,
    stress_feature_seed: int = STRESS_SEED,
    disp_feature_seed: int = DISP_SEED,
) -> SharedFeatureSpace:
    """Build the shared random feature spaces used by coefficient-based methods."""

    a_s, r_s = generate_features(N_s, seed=stress_feature_seed)
    a_u, r_u = generate_features(N_u, seed=disp_feature_seed)
    return SharedFeatureSpace(
        a_s=a_s,
        r_s=r_s,
        a_u=a_u,
        r_u=r_u,
        gamma_s=gamma_s,
        gamma_u=gamma_u,
    )


def build_feature_evaluation_data(
    benchmark: SharedBenchmarkData,
    feature_space: SharedFeatureSpace,
) -> FeatureEvaluationData:
    """Build the shared evaluation tensors for coefficient-based methods."""

    return FeatureEvaluationData(
        xi_s_test=eval_features(
            benchmark.x_test,
            feature_space.a_s,
            feature_space.r_s,
            feature_space.gamma_s,
        ),
        psi_u_test=eval_active_displacement_features(
            benchmark.x_test,
            feature_space.a_u,
            feature_space.r_u,
            feature_space.gamma_u,
        ),
        w_test=benchmark.w_test,
        u_exact_test=benchmark.u_exact_test,
        sigma_exact_test=benchmark.sigma_exact_test,
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


def accumulate_interior_moments(
    x_int: torch.Tensor,
    w_int: torch.Tensor,
    f_int: torch.Tensor,
    a_s: torch.Tensor,
    r_s: torch.Tensor,
    gamma_s: float,
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

    mp1_s = a_s.shape[0] + 1
    mp1_u = a_u.shape[0] + 1

    gram_xi_s = torch.zeros(mp1_s, mp1_s, dtype=DTYPE, device=DEVICE)
    cross_xi_grad_psi = [
        torch.zeros(mp1_s, mp1_u, dtype=DTYPE, device=DEVICE) for _ in range(2)
    ]
    grad_gram_psi = [
        [
            torch.zeros(mp1_u, mp1_u, dtype=DTYPE, device=DEVICE)
            for _ in range(2)
        ]
        for _ in range(2)
    ]
    grad_gram_s = [
        [
            torch.zeros(mp1_s, mp1_s, dtype=DTYPE, device=DEVICE)
            for _ in range(2)
        ]
        for _ in range(2)
    ]
    grad_force_s = [
        torch.zeros(mp1_s, 2, dtype=DTYPE, device=DEVICE) for _ in range(2)
    ]

    with torch.no_grad():
        for start in range(0, x_int.shape[0], batch_size):
            end = min(start + batch_size, x_int.shape[0])
            xb = x_int[start:end]
            wb = w_int[start:end]
            fb = f_int[start:end]

            xi_s_batch = eval_features(xb, a_s, r_s, gamma_s)
            grad_s_batch = eval_feature_grads(xb, a_s, r_s, gamma_s)
            _, grad_psi_batch = eval_active_displacement_feature_data(
                xb,
                a_u,
                r_u,
                gamma_u,
            )

            weighted_xi_s = wb.unsqueeze(1) * xi_s_batch
            weighted_grad_psi = [
                wb.unsqueeze(1) * grad_psi_batch[:, :, dim_i] for dim_i in range(2)
            ]
            weighted_grad_s = [
                wb.unsqueeze(1) * grad_s_batch[:, :, dim_i] for dim_i in range(2)
            ]
            gram_xi_s += xi_s_batch.T @ weighted_xi_s
            for dim_i in range(2):
                cross_xi_grad_psi[dim_i] += xi_s_batch.T @ weighted_grad_psi[dim_i]
                grad_force_s[dim_i] += weighted_grad_s[dim_i].T @ fb
                for dim_j in range(2):
                    grad_gram_psi[dim_i][dim_j] += (
                        grad_psi_batch[:, :, dim_i].T @ weighted_grad_psi[dim_j]
                    )
                    grad_gram_s[dim_i][dim_j] += (
                        grad_s_batch[:, :, dim_i].T @ weighted_grad_s[dim_j]
                    )

    return gram_xi_s, cross_xi_grad_psi, grad_gram_psi, grad_gram_s, grad_force_s


def assemble_linear_system(
    cfg: LeastSquaresConfig,
    compliance_voigt: torch.Tensor,
    x_int: torch.Tensor,
    w_int: torch.Tensor,
    f_int: torch.Tensor,
    a_s: torch.Tensor,
    r_s: torch.Tensor,
    a_u: torch.Tensor,
    r_u: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble the conforming least-squares system G z = F."""

    (
        gram_xi_s,
        cross_xi_grad_psi,
        grad_gram_psi,
        grad_gram_s,
        grad_force_s,
    ) = accumulate_interior_moments(
        x_int,
        w_int,
        f_int,
        a_s,
        r_s,
        cfg.gamma_s,
        a_u,
        r_u,
        cfg.gamma_u,
        cfg.assembly_batch_size,
    )

    mp1_s = a_s.shape[0] + 1
    mp1_u = a_u.shape[0] + 1
    dim_s = 3 * mp1_s
    dim_u = 2 * mp1_u

    G_ss = torch.zeros(dim_s, dim_s, dtype=DTYPE, device=DEVICE)
    G_su = torch.zeros(dim_s, dim_u, dtype=DTYPE, device=DEVICE)
    G_uu = torch.zeros(dim_u, dim_u, dtype=DTYPE, device=DEVICE)
    F_s = torch.zeros(dim_s, dtype=DTYPE, device=DEVICE)

    compliance_sq = compliance_voigt.T @ compliance_voigt
    add_block_scaled(G_ss, gram_xi_s, compliance_sq, row_stride=3, col_stride=3)

    for dim_i in range(2):
        constitutive_cross = compliance_voigt @ STRAIN_GRAD_BASES[dim_i]
        add_block_scaled(
            G_su,
            cross_xi_grad_psi[dim_i],
            -constitutive_cross,
            row_stride=3,
            col_stride=2,
        )
        add_rhs_feature_blocks(
            F_s,
            grad_force_s[dim_i],
            STRAIN_GRAD_BASES[dim_i],
            block_size=3,
            scale=-1.0,
        )

        for dim_j in range(2):
            constitutive_uu = STRAIN_GRAD_BASES[dim_i].T @ STRAIN_GRAD_BASES[dim_j]
            equilibrium_ss = STRAIN_GRAD_BASES[dim_i] @ STRAIN_GRAD_BASES[dim_j].T
            add_block_scaled(
                G_uu,
                grad_gram_psi[dim_i][dim_j],
                constitutive_uu,
                row_stride=2,
                col_stride=2,
            )
            add_block_scaled(
                G_ss,
                grad_gram_s[dim_i][dim_j],
                equilibrium_ss,
                row_stride=3,
                col_stride=3,
            )

    G = torch.zeros(dim_s + dim_u, dim_s + dim_u, dtype=DTYPE, device=DEVICE)
    G[:dim_s, :dim_s] = G_ss
    G[:dim_s, dim_s:] = G_su
    G[dim_s:, :dim_s] = G_su.T
    G[dim_s:, dim_s:] = G_uu
    G = 0.5 * (G + G.T)

    F = torch.zeros(dim_s + dim_u, dtype=DTYPE, device=DEVICE)
    F[:dim_s] = F_s
    return G, F


def assemble_direct_residual_design(
    benchmark: SharedBenchmarkData,
    feature_space: SharedFeatureSpace,
) -> DirectResidualDesign:
    """Assemble the weighted constitutive and equilibrium residuals directly."""

    x = benchmark.x_int
    sqrt_weights = torch.sqrt(benchmark.w_int)
    sigma_values = eval_features(
        x,
        feature_space.a_s,
        feature_space.r_s,
        feature_space.gamma_s,
    )
    sigma_gradients = eval_feature_grads(
        x,
        feature_space.a_s,
        feature_space.r_s,
        feature_space.gamma_s,
    )
    _, displacement_gradients = eval_active_displacement_feature_data(
        x,
        feature_space.a_u,
        feature_space.r_u,
        feature_space.gamma_u,
    )
    q_count = x.shape[0]
    dim_s = 3 * sigma_values.shape[1]
    dim_u = 2 * displacement_gradients.shape[1]
    matrix = torch.zeros(5 * q_count, dim_s + dim_u, dtype=DTYPE, device=DEVICE)
    rhs = torch.zeros(5 * q_count, dtype=DTYPE, device=DEVICE)
    weighted_sigma = sqrt_weights.unsqueeze(1) * sigma_values

    for residual_component in range(3):
        rows = slice(residual_component * q_count, (residual_component + 1) * q_count)
        for sigma_component in range(3):
            matrix[rows, sigma_component:dim_s:3] = (
                benchmark.compliance_voigt[residual_component, sigma_component]
                * weighted_sigma
            )
        for spatial_dimension in range(2):
            coupling = STRAIN_GRAD_BASES[spatial_dimension]
            for displacement_component in range(2):
                matrix[rows, dim_s + displacement_component :: 2] -= (
                    coupling[residual_component, displacement_component]
                    * sqrt_weights.unsqueeze(1)
                    * displacement_gradients[:, :, spatial_dimension]
                )

    for equilibrium_component in range(2):
        rows = slice(
            (3 + equilibrium_component) * q_count,
            (4 + equilibrium_component) * q_count,
        )
        for spatial_dimension in range(2):
            coupling = STRAIN_GRAD_BASES[spatial_dimension]
            for sigma_component in range(3):
                matrix[rows, sigma_component:dim_s:3] += (
                    coupling[sigma_component, equilibrium_component]
                    * sqrt_weights.unsqueeze(1)
                    * sigma_gradients[:, :, spatial_dimension]
                )
        rhs[rows] = -sqrt_weights * benchmark.f_int[:, equilibrium_component]
    return DirectResidualDesign(matrix, rhs, dim_s)


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


def split_solution(z: torch.Tensor, dim_s: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the full coefficient vector into stress and displacement parts."""

    return z[:dim_s], z[dim_s:]


def compute_absolute_errors(
    psi_u_test: torch.Tensor,
    xi_s_test: torch.Tensor,
    sigma_coeffs: torch.Tensor,
    displacement_coeffs: torch.Tensor,
    w_test: torch.Tensor,
    u_exact: torch.Tensor,
    sigma_exact: torch.Tensor,
) -> tuple[float, float]:
    """Compute absolute L2 errors for displacement and stress."""

    displacement_blocks = displacement_coeffs.reshape(-1, 2)
    stress_blocks = sigma_coeffs.reshape(-1, 3)

    u_h = psi_u_test @ displacement_blocks
    sigma_h = xi_s_test @ stress_blocks

    voigt_weight = VOIGT_WEIGHT.to(device=DEVICE)

    u_l2_error = torch.sqrt((w_test * (u_h - u_exact).square().sum(dim=1)).sum())
    sigma_l2_error = torch.sqrt(
        (w_test * (voigt_weight * (sigma_h - sigma_exact).square()).sum(dim=1)).sum()
    )

    return (
        u_l2_error.item(),
        sigma_l2_error.item(),
    )


def evaluate_feature_result(
    name: str,
    wall_time: float,
    sigma_coeffs: torch.Tensor,
    displacement_coeffs: torch.Tensor,
    data: FeatureEvaluationData,
) -> AlgorithmResult:
    """Evaluate one coefficient-based method and package the metrics."""

    u_l2_error, sigma_l2_error = compute_absolute_errors(
        data.psi_u_test,
        data.xi_s_test,
        sigma_coeffs,
        displacement_coeffs,
        data.w_test,
        data.u_exact_test,
        data.sigma_exact_test,
    )
    return AlgorithmResult(
        name=name,
        u_l2_error=u_l2_error,
        sigma_l2_error=sigma_l2_error,
        wall_time=wall_time,
    )


def print_result_summary(result: AlgorithmResult) -> None:
    """Print one compact result line."""

    print(
        f"    Done in {result.wall_time:.2f}s, "
        f"‖Φ^u-u‖={result.u_l2_error:.2e}, "
        f"‖Φ^σ-σ‖={result.sigma_l2_error:.2e}, "
        f"rank={result.rank}/{result.columns}, "
        f"cond≈{result.condition_estimate:.2e}"
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


def print_summary_table(
    results: Sequence[AlgorithmResult],
    title: str,
) -> None:
    """Print a compact markdown-style summary table."""

    if not results:
        return

    headers = (
        "Method",
        "‖Φ^u-u‖",
        "‖Φ^σ-σ‖",
        "Time(s)",
    )
    rows = [
        (
            result.name,
            f"{result.u_l2_error:.2e}",
            f"{result.sigma_l2_error:.2e}",
            f"{result.wall_time:.2f}",
        )
        for result in results
    ]
    print_aligned_markdown_table(
        title=title,
        headers=headers,
        rows=rows,
        alignments=("left", "center", "center", "center"),
    )


def configure_plotting() -> None:
    """Apply the shared matplotlib settings used by experiment plots."""

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_l2_summary(
    results: Sequence[AlgorithmResult],
    save_path: str,
) -> None:
    """Plot final absolute L2 errors as bar charts."""

    if not results:
        print(f"  Skipped: {save_path} (no results to plot)")
        return

    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    titles = [
        r"$\|\Phi^u - u_{ex}\|_0$",
        r"$\|\Phi^\sigma - \sigma_{ex}\|_0$",
    ]
    keys = ["u_l2_error", "sigma_l2_error"]
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
        ax.set_ylabel("$L^2$ error")
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
    sigma_coeffs, displacement_coeffs = split_solution(z, data.dim_s)
    evaluated = evaluate_feature_result(
        "Direct LS (GELSD)",
        wall_time,
        sigma_coeffs,
        displacement_coeffs,
        data.eval_data,
    )
    result = AlgorithmResult(
        name=evaluated.name,
        u_l2_error=evaluated.u_l2_error,
        sigma_l2_error=evaluated.sigma_l2_error,
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
        f"Config: N_s={cfg.N_s}, N_u={cfg.N_u}, "
        f"Q_train={cfg.Q_train}, Q_test={cfg.Q_test}, "
        f"gamma_s={cfg.gamma_s}, gamma_u={cfg.gamma_u}, "
        f"direct_rcond={cfg.direct_rcond:.2e}, "
        f"sampling={cfg.sampling_method}"
    )
    print(f"Algorithms: {selected_algorithm_ids}")

    mu, lambda_plane = compute_plane_stress_parameters(cfg.E, cfg.nu)
    print(
        f"Material: E={cfg.E}, nu={cfg.nu}, "
        f"mu={mu:.4f}, lambda_plane={lambda_plane:.4f}"
    )

    if benchmark is None:
        print("Building benchmark data...")
        benchmark = build_shared_benchmark(
            E=cfg.E,
            nu=cfg.nu,
            Q_train=cfg.Q_train,
            Q_test=cfg.Q_test,
            sampling_method=cfg.sampling_method,
            body_force_batch_size=cfg.body_force_batch_size,
        )
    else:
        print("Using shared benchmark data...")

    if feature_space is None:
        print("Generating random feature spaces...")
        feature_space = build_shared_feature_space(
            N_s=cfg.N_s,
            N_u=cfg.N_u,
            gamma_s=cfg.gamma_s,
            gamma_u=cfg.gamma_u,
        )
    else:
        print("Using shared random feature spaces...")

    if feature_space.gamma_s != cfg.gamma_s or feature_space.gamma_u != cfg.gamma_u:
        raise ValueError("SharedFeatureSpace gamma does not match LeastSquaresConfig.")
    if feature_space.a_s.shape[0] != cfg.N_s or feature_space.a_u.shape[0] != cfg.N_u:
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
        dim_s=direct_design.dim_s,
        eval_data=build_feature_evaluation_data(
            benchmark,
            feature_space,
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
