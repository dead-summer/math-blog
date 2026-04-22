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
import torch


BASE_SEED = 42
STRESS_SEED = BASE_SEED
DISP_SEED = BASE_SEED + 1_000
DTYPE = torch.float64
VALID_SAMPLING_METHODS = ("mc", "sobol", "gauss_legendre")
VALID_ALGORITHMS = ("eigh", "lstsq")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "public" / "images" / "least-squares" / "linear-elasticity-3d"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ALGO_STYLE = {
    "LS (Eigh)": {"color": "#0077B6", "marker": "o", "linestyle": "-"},
    "LS (Lstsq)": {"color": "#264653", "marker": "s", "linestyle": "--"},
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


@dataclass(frozen=True)
class AlgorithmResult:
    """Compact metrics for one completed algorithm."""

    name: str
    r_c: float
    r_e: float
    rel_u: float
    rel_sigma: float
    wall_time: float


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

    x_int: torch.Tensor
    w_int: torch.Tensor
    f_int: torch.Tensor
    a_s: torch.Tensor
    r_s: torch.Tensor
    a_u: torch.Tensor
    r_u: torch.Tensor
    gamma_s: float
    gamma_u: float
    compliance_voigt: torch.Tensor
    assembly_batch_size: int
    xi_s_test: torch.Tensor
    psi_u_test: torch.Tensor
    w_test: torch.Tensor
    u_exact_test: torch.Tensor
    sigma_exact_test: torch.Tensor


@dataclass
class LeastSquaresConfig:
    """Configuration for the conforming least-squares experiment."""

    E: float = 1.0
    nu: float = 0.3
    gamma_s: float = 2.0
    gamma_u: float = 2.0
    M_s: int = 300
    M_u: int = 300
    Q_train: int = (2 ** 5) ** 3
    Q_test: int = (2 ** 4) ** 3
    sampling_method: str = "sobol"
    eigh_rtol: float = 1.0e-15
    body_force_batch_size: int = 5_000
    assembly_batch_size: int = 5_000
    algorithms_to_run: list[str] = field(
        default_factory=lambda: [
            "eigh",
            "lstsq",
        ]
    )


@dataclass(frozen=True)
class LeastSquaresExperimentData:
    """All tensors needed to run and evaluate one least-squares solver."""

    G: torch.Tensor
    F: torch.Tensor
    dim_s: int
    eval_data: FeatureEvaluationData


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
    if cfg.gamma_s <= 0.0 or cfg.gamma_u <= 0.0:
        raise ValueError("Config.gamma_s and Config.gamma_u must be positive.")
    if cfg.M_s <= 0 or cfg.M_u <= 0:
        raise ValueError("Config.M_s and Config.M_u must be positive.")
    if cfg.Q_train <= 0:
        raise ValueError("Config.Q_train must be positive.")
    if cfg.Q_test <= 0:
        raise ValueError("Config.Q_test must be positive.")
    if not math.isfinite(cfg.eigh_rtol) or cfg.eigh_rtol < 0.0:
        raise ValueError("Config.eigh_rtol must be finite and non-negative.")
    if cfg.body_force_batch_size <= 0:
        raise ValueError("Config.body_force_batch_size must be positive.")
    if cfg.assembly_batch_size <= 0:
        raise ValueError("Config.assembly_batch_size must be positive.")
    validate_sampling_method(cfg.sampling_method)
    validate_algorithm_selection(cfg.algorithms_to_run, VALID_ALGORITHMS)


def compute_lame_constants(E: float, nu: float) -> tuple[float, float]:
    """Return Lamé constants (mu, lambda)."""

    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def build_compliance_matrix(E: float, nu: float) -> torch.Tensor:
    """Build the 6x6 isotropic compliance matrix in engineering Voigt form."""

    compliance_voigt = torch.zeros(6, 6, dtype=DTYPE, device=DEVICE)
    compliance_voigt[0, 0] = 1.0 / E
    compliance_voigt[1, 1] = 1.0 / E
    compliance_voigt[2, 2] = 1.0 / E
    compliance_voigt[0, 1] = -nu / E
    compliance_voigt[0, 2] = -nu / E
    compliance_voigt[1, 0] = -nu / E
    compliance_voigt[1, 2] = -nu / E
    compliance_voigt[2, 0] = -nu / E
    compliance_voigt[2, 1] = -nu / E
    shear = 2.0 * (1.0 + nu) / E
    compliance_voigt[3, 3] = shear
    compliance_voigt[4, 4] = shear
    compliance_voigt[5, 5] = shear
    return compliance_voigt


def eval_exact_displacement(x: torch.Tensor) -> torch.Tensor:
    """Evaluate the manufactured displacement field."""

    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    pi = math.pi
    u1 = torch.sin(pi * x1) * torch.sin(pi * x2) * torch.sin(pi * x3)
    u2 = torch.sin(2.0 * pi * x1) * torch.sin(pi * x2) * torch.sin(pi * x3)
    u3 = torch.sin(pi * x1) * torch.sin(2.0 * pi * x2) * torch.sin(pi * x3)
    return torch.stack([u1, u2, u3], dim=1)


def compute_stress_voigt(x: torch.Tensor, mu: float, lam: float) -> torch.Tensor:
    """Evaluate the exact stress in Voigt order (11, 22, 33, 12, 23, 13)."""

    x_ad = x.detach().requires_grad_(True)
    u = eval_exact_displacement(x_ad)

    n_points = x.shape[0]
    grad_u = torch.zeros(n_points, 3, 3, dtype=DTYPE, device=DEVICE)
    for comp in range(3):
        grad_u[:, comp, :] = torch.autograd.grad(
            u[:, comp].sum(),
            x_ad,
            create_graph=False,
            retain_graph=(comp < 2),
        )[0]

    eps = 0.5 * (grad_u + grad_u.transpose(1, 2))
    tr_eps = eps[:, 0, 0] + eps[:, 1, 1] + eps[:, 2, 2]

    sigma = 2.0 * mu * eps
    for comp in range(3):
        sigma[:, comp, comp] += lam * tr_eps

    return torch.stack(
        [
            sigma[:, 0, 0],
            sigma[:, 1, 1],
            sigma[:, 2, 2],
            sigma[:, 0, 1],
            sigma[:, 1, 2],
            sigma[:, 0, 2],
        ],
        dim=1,
    ).detach()


def compute_body_force(
    x: torch.Tensor,
    mu: float,
    lam: float,
    batch_size: int,
) -> torch.Tensor:
    """Compute f = -div(sigma(u_exact)) with batched autodiff."""

    n_points = x.shape[0]
    f_all = torch.zeros(n_points, 3, dtype=DTYPE, device=DEVICE)

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
                for comp in range(3)
            ],
            dim=1,
        )

        eps = 0.5 * (grad_u + grad_u.transpose(1, 2))
        tr_eps = eps[:, 0, 0] + eps[:, 1, 1] + eps[:, 2, 2]

        sigma = 2.0 * mu * eps
        for comp in range(3):
            sigma[:, comp, comp] += lam * tr_eps

        for comp in range(3):
            div_sigma = torch.zeros(end - start, dtype=DTYPE, device=DEVICE)
            for dim in range(3):
                grad_sigma = torch.autograd.grad(
                    sigma[:, comp, dim].sum(),
                    xb,
                    create_graph=False,
                    retain_graph=not (comp == 2 and dim == 2),
                )[0]
                div_sigma += grad_sigma[:, dim]
            f_all[start:end, comp] = -div_sigma.detach()

    return f_all


def eval_zeta(x: torch.Tensor) -> torch.Tensor:
    """Evaluate the Dirichlet envelope zeta(x)."""

    return (
        x[:, 0]
        * (1.0 - x[:, 0])
        * x[:, 1]
        * (1.0 - x[:, 1])
        * x[:, 2]
        * (1.0 - x[:, 2])
    )


def eval_zeta_grad(x: torch.Tensor) -> torch.Tensor:
    """Evaluate the gradient of the Dirichlet envelope."""

    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    grad_x = (1.0 - 2.0 * x1) * x2 * (1.0 - x2) * x3 * (1.0 - x3)
    grad_y = x1 * (1.0 - x1) * (1.0 - 2.0 * x2) * x3 * (1.0 - x3)
    grad_z = x1 * (1.0 - x1) * x2 * (1.0 - x2) * (1.0 - 2.0 * x3)
    return torch.stack([grad_x, grad_y, grad_z], dim=1)


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
    dim: int = 3,
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


def generate_features(M: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random feature normals and offsets."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    raw = torch.randn(M, 3, generator=generator, dtype=DTYPE)
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
    dtanh = 1.0 - torch.tanh(gamma * pre).square()
    grad_xi = gamma * dtanh.unsqueeze(2) * a.unsqueeze(0)
    zeros = torch.zeros(x.shape[0], 1, 3, dtype=DTYPE, device=DEVICE)
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

    mu, lam = compute_lame_constants(E, nu)
    compliance_voigt = build_compliance_matrix(E, nu)

    x_int, w_int = build_quadrature_rule(
        Q_train,
        method=sampling_method,
        seed=interior_seed,
    )
    f_int = compute_body_force(x_int, mu, lam, batch_size=body_force_batch_size)

    x_test, w_test = build_quadrature_rule(
        Q_test,
        method=sampling_method,
        seed=test_seed,
    )
    u_exact_test = eval_exact_displacement(x_test)
    sigma_exact_test = compute_stress_voigt(x_test, mu, lam)
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
    M_s: int,
    M_u: int,
    gamma_s: float,
    gamma_u: float,
    stress_feature_seed: int = STRESS_SEED,
    disp_feature_seed: int = DISP_SEED,
) -> SharedFeatureSpace:
    """Build the shared random feature spaces used by coefficient-based methods."""

    a_s, r_s = generate_features(M_s, seed=stress_feature_seed)
    a_u, r_u = generate_features(M_u, seed=disp_feature_seed)
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
    assembly_batch_size: int,
) -> FeatureEvaluationData:
    """Build the shared evaluation tensors for coefficient-based methods."""

    return FeatureEvaluationData(
        x_int=benchmark.x_int,
        w_int=benchmark.w_int,
        f_int=benchmark.f_int,
        a_s=feature_space.a_s,
        r_s=feature_space.r_s,
        a_u=feature_space.a_u,
        r_u=feature_space.r_u,
        gamma_s=feature_space.gamma_s,
        gamma_u=feature_space.gamma_u,
        compliance_voigt=benchmark.compliance_voigt,
        assembly_batch_size=assembly_batch_size,
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
        torch.zeros(mp1_s, mp1_u, dtype=DTYPE, device=DEVICE) for _ in range(3)
    ]
    grad_gram_psi = [
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
                wb.unsqueeze(1) * grad_psi_batch[:, :, dim_i] for dim_i in range(3)
            ]
            weighted_grad_s = [
                wb.unsqueeze(1) * grad_s_batch[:, :, dim_i] for dim_i in range(3)
            ]
            gram_xi_s += xi_s_batch.T @ weighted_xi_s
            for dim_i in range(3):
                cross_xi_grad_psi[dim_i] += xi_s_batch.T @ weighted_grad_psi[dim_i]
                grad_force_s[dim_i] += weighted_grad_s[dim_i].T @ fb
                for dim_j in range(3):
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
    dim_s = 6 * mp1_s
    dim_u = 3 * mp1_u

    G_ss = torch.zeros(dim_s, dim_s, dtype=DTYPE, device=DEVICE)
    G_su = torch.zeros(dim_s, dim_u, dtype=DTYPE, device=DEVICE)
    G_uu = torch.zeros(dim_u, dim_u, dtype=DTYPE, device=DEVICE)
    F_s = torch.zeros(dim_s, dtype=DTYPE, device=DEVICE)

    compliance_sq = compliance_voigt.T @ compliance_voigt
    add_block_scaled(G_ss, gram_xi_s, compliance_sq, row_stride=6, col_stride=6)

    for dim_i in range(3):
        constitutive_cross = compliance_voigt @ STRAIN_GRAD_BASES[dim_i]
        add_block_scaled(
            G_su,
            cross_xi_grad_psi[dim_i],
            -constitutive_cross,
            row_stride=6,
            col_stride=3,
        )
        add_rhs_feature_blocks(
            F_s,
            grad_force_s[dim_i],
            STRAIN_GRAD_BASES[dim_i],
            block_size=6,
            scale=-1.0,
        )

        for dim_j in range(3):
            constitutive_uu = STRAIN_GRAD_BASES[dim_i].T @ STRAIN_GRAD_BASES[dim_j]
            equilibrium_ss = STRAIN_GRAD_BASES[dim_i] @ STRAIN_GRAD_BASES[dim_j].T
            add_block_scaled(
                G_uu,
                grad_gram_psi[dim_i][dim_j],
                constitutive_uu,
                row_stride=3,
                col_stride=3,
            )
            add_block_scaled(
                G_ss,
                grad_gram_s[dim_i][dim_j],
                equilibrium_ss,
                row_stride=6,
                col_stride=6,
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


def solve_lstsq(G: torch.Tensor, F: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Solve the linear system with torch.linalg.lstsq."""

    synchronize_device()
    t0 = time.perf_counter()
    try:
        sol = torch.linalg.lstsq(G, F.unsqueeze(1)).solution.squeeze(1)
        if not torch.isfinite(sol).all():
            raise RuntimeError("non-finite solution")
    except (RuntimeError, torch.linalg.LinAlgError) as exc:
        sol = torch.full((G.shape[0],), float("nan"), dtype=DTYPE, device=DEVICE)
        print(f"    Warning: torch.linalg.lstsq failed with {type(exc).__name__}")

    synchronize_device()
    return sol, time.perf_counter() - t0


def solve_eigh(
    G: torch.Tensor,
    F: torch.Tensor,
    rtol: float,
) -> tuple[torch.Tensor, float]:
    """Solve the linear system with truncated eigen decomposition."""

    synchronize_device()
    t0 = time.perf_counter()
    try:
        eigvals, eigvecs = torch.linalg.eigh(G)
        threshold = rtol * eigvals.abs().max()
        keep = eigvals > threshold
        if not keep.any():
            raise RuntimeError("all eigenvalues were truncated")

        coeffs = eigvecs[:, keep].T @ F
        coeffs = coeffs / eigvals[keep]
        sol = eigvecs[:, keep] @ coeffs
        if not torch.isfinite(sol).all():
            raise RuntimeError("non-finite solution")
        print(
            f"    eigh truncation: kept {int(keep.sum().item())}/{eigvals.numel()} "
            f"eigenvalues, threshold={threshold.item():.2e}"
        )
    except (RuntimeError, torch.linalg.LinAlgError) as exc:
        sol = torch.full((G.shape[0],), float("nan"), dtype=DTYPE, device=DEVICE)
        print(f"    Warning: torch.linalg.eigh failed with {type(exc).__name__}")

    synchronize_device()
    return sol, time.perf_counter() - t0


def split_solution(z: torch.Tensor, dim_s: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the full coefficient vector into stress and displacement parts."""

    return z[:dim_s], z[dim_s:]


def compute_l2_errors(
    psi_u_test: torch.Tensor,
    xi_s_test: torch.Tensor,
    sigma_coeffs: torch.Tensor,
    displacement_coeffs: torch.Tensor,
    w_test: torch.Tensor,
    u_exact: torch.Tensor,
    sigma_exact: torch.Tensor,
) -> tuple[float, float]:
    """Compute relative L2 errors for displacement and stress."""

    displacement_blocks = displacement_coeffs.reshape(-1, 3)
    stress_blocks = sigma_coeffs.reshape(-1, 6)

    u_h = psi_u_test @ displacement_blocks
    sigma_h = xi_s_test @ stress_blocks

    voigt_weight = torch.tensor(
        [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        dtype=DTYPE,
        device=DEVICE,
    )

    u_err = torch.sqrt((w_test * (u_h - u_exact).square().sum(dim=1)).sum())
    u_ref = torch.sqrt((w_test * u_exact.square().sum(dim=1)).sum())
    rel_u = (u_err / u_ref).item() if u_ref > 0 else float("inf")

    sigma_err = torch.sqrt(
        (w_test * (voigt_weight * (sigma_h - sigma_exact).square()).sum(dim=1)).sum()
    )
    sigma_ref = torch.sqrt(
        (w_test * (voigt_weight * sigma_exact.square()).sum(dim=1)).sum()
    )
    rel_sigma = (sigma_err / sigma_ref).item() if sigma_ref > 0 else float("inf")
    return rel_u, rel_sigma


def compute_coefficient_residual_norms(
    data: FeatureEvaluationData,
    sigma_coeffs: torch.Tensor,
    displacement_coeffs: torch.Tensor,
) -> tuple[float, float]:
    """Evaluate least-squares residual norms on sampled interior points."""

    if not torch.isfinite(sigma_coeffs).all() or not torch.isfinite(displacement_coeffs).all():
        return float("nan"), float("nan")

    stress_blocks = sigma_coeffs.reshape(-1, 6)
    displacement_blocks = displacement_coeffs.reshape(-1, 3)
    constitutive_sq = 0.0
    equilibrium_sq = 0.0
    with torch.no_grad():
        for start in range(0, data.x_int.shape[0], data.assembly_batch_size):
            end = min(start + data.assembly_batch_size, data.x_int.shape[0])
            xb = data.x_int[start:end]
            wb = data.w_int[start:end]
            fb = data.f_int[start:end]

            xi_s_batch = eval_features(xb, data.a_s, data.r_s, data.gamma_s)
            grad_s_batch = eval_feature_grads(xb, data.a_s, data.r_s, data.gamma_s)
            _, grad_psi_batch = eval_active_displacement_feature_data(
                xb,
                data.a_u,
                data.r_u,
                data.gamma_u,
            )

            sigma_h = xi_s_batch @ stress_blocks

            du_dx1 = grad_psi_batch[:, :, 0] @ displacement_blocks
            du_dx2 = grad_psi_batch[:, :, 1] @ displacement_blocks
            du_dx3 = grad_psi_batch[:, :, 2] @ displacement_blocks
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

            ds_dx1 = grad_s_batch[:, :, 0] @ stress_blocks
            ds_dx2 = grad_s_batch[:, :, 1] @ stress_blocks
            ds_dx3 = grad_s_batch[:, :, 2] @ stress_blocks
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
            constitutive_sq += (wb * r_c.square().sum(dim=1)).sum().item()
            equilibrium_sq += (wb * r_e.square().sum(dim=1)).sum().item()

    return constitutive_sq**0.5, equilibrium_sq**0.5


def evaluate_feature_result(
    name: str,
    wall_time: float,
    sigma_coeffs: torch.Tensor,
    displacement_coeffs: torch.Tensor,
    data: FeatureEvaluationData,
) -> AlgorithmResult:
    """Evaluate one coefficient-based method and package the metrics."""

    r_c, r_e = compute_coefficient_residual_norms(
        data,
        sigma_coeffs,
        displacement_coeffs,
    )
    rel_u, rel_sigma = compute_l2_errors(
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
        r_c=r_c,
        r_e=r_e,
        rel_u=rel_u,
        rel_sigma=rel_sigma,
        wall_time=wall_time,
    )


def print_result_summary(result: AlgorithmResult) -> None:
    """Print one compact result line."""

    print(
        f"    Done in {result.wall_time:.2f}s, "
        f"||r_c||={result.r_c:.2e}, "
        f"||r_e||={result.r_e:.2e}, "
        f"rel_u={result.rel_u:.2e}, "
        f"rel_sigma={result.rel_sigma:.2e}"
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
        f"{'rel_sigma':>10} | {'Time(s)':>8} |"
    )
    print(
        f"|:{'-' * (method_width + 1)}|{'-' * 11}:|"
        f"{'-' * 11}:|{'-' * 9}:|"
    )
    for result in results:
        print(
            f"| {result.name:<{method_width}} | {result.rel_u:>10.2e} | "
            f"{result.rel_sigma:>10.2e} | {result.wall_time:>8.2f} |"
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
        r"Displacement $\|\Phi^u - u_{ex}\|_{L^2} / \|u_{ex}\|_{L^2}$",
        r"Stress $\|\Phi^\sigma - \sigma_{ex}\|_{L^2} / \|\sigma_{ex}\|_{L^2}$",
    ]
    keys = ["rel_u", "rel_sigma"]
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

    if algorithm_id == "eigh":
        print("Running LS (Eigh)...")
        z, wall_time = solve_eigh(data.G, data.F, cfg.eigh_rtol)
        sigma_coeffs, displacement_coeffs = split_solution(z, data.dim_s)
        result = evaluate_feature_result(
            "LS (Eigh)",
            wall_time,
            sigma_coeffs,
            displacement_coeffs,
            data.eval_data,
        )
    else:
        print("Running LS (Lstsq)...")
        z, wall_time = solve_lstsq(data.G, data.F)
        sigma_coeffs, displacement_coeffs = split_solution(z, data.dim_s)
        result = evaluate_feature_result(
            "LS (Lstsq)",
            wall_time,
            sigma_coeffs,
            displacement_coeffs,
            data.eval_data,
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
    """Run the selected least-squares methods and return their metrics."""

    cfg = LeastSquaresConfig() if cfg is None else cfg
    validate_config(cfg)
    selected_algorithm_ids = validate_algorithm_selection(
        cfg.algorithms_to_run,
        VALID_ALGORITHMS,
    )

    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}")
    print(
        f"Config: M_s={cfg.M_s}, M_u={cfg.M_u}, "
        f"Q_train={cfg.Q_train}, Q_test={cfg.Q_test}, "
        f"gamma_s={cfg.gamma_s}, gamma_u={cfg.gamma_u}, "
        f"eigh_rtol={cfg.eigh_rtol:.2e}, sampling={cfg.sampling_method}"
    )
    print(f"Algorithms: {selected_algorithm_ids}")

    mu, lam = compute_lame_constants(cfg.E, cfg.nu)
    print(f"Material: E={cfg.E}, nu={cfg.nu}, mu={mu:.4f}, lam={lam:.4f}")

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
            M_s=cfg.M_s,
            M_u=cfg.M_u,
            gamma_s=cfg.gamma_s,
            gamma_u=cfg.gamma_u,
        )
    else:
        print("Using shared random feature spaces...")

    if feature_space.gamma_s != cfg.gamma_s or feature_space.gamma_u != cfg.gamma_u:
        raise ValueError("SharedFeatureSpace gamma does not match LeastSquaresConfig.")

    print("Assembling conforming least-squares system...")
    G, F = assemble_linear_system(
        cfg,
        benchmark.compliance_voigt,
        benchmark.x_int,
        benchmark.w_int,
        benchmark.f_int,
        feature_space.a_s,
        feature_space.r_s,
        feature_space.a_u,
        feature_space.r_u,
    )
    clear_cuda_cache()

    print(f"System shapes: G={tuple(G.shape)}, F={tuple(F.shape)}")

    experiment_data = LeastSquaresExperimentData(
        G=G,
        F=F,
        dim_s=6 * (cfg.M_s + 1),
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
