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
VALID_ALGORITHMS = ("lstsq", "tsvd", "ridge")
VALID_MANUFACTURED_SOLUTIONS = ("hu_zhang", "div_free", "near_incompressible")
FEATURE_DIM = 2
FEATURE_CENTER = 0.5
FEATURE_INV_RADIUS = 2.0 / math.sqrt(FEATURE_DIM)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "public" / "images" / "least-squares" / "linear-elasticity-2d"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ALGO_STYLE = {
    "LS (Lstsq)": {"color": "#264653", "marker": "s", "linestyle": "--"},
    "LS (TSVD)": {"color": "#0077B6", "marker": "o", "linestyle": "-"},
    "LS (Ridge)": {"color": "#E76F51", "marker": "D", "linestyle": "-."},
}


def detect_device() -> torch.device:
    """Prefer CUDA when available and stay quiet otherwise."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


DEVICE = detect_device()

VOIGT_WEIGHT = torch.tensor([1.0, 1.0, 2.0], dtype=DTYPE, device=DEVICE)
TRACE_VOIGT = torch.tensor([1.0, 1.0, 0.0], dtype=DTYPE, device=DEVICE)

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


def build_deviatoric_stress_bases() -> torch.Tensor:
    """Return a fixed 3x2 basis spanning trace-free stresses in Voigt form."""

    return torch.tensor(
        [
            [1.0 / math.sqrt(2.0), 0.0],
            [-1.0 / math.sqrt(2.0), 0.0],
            [0.0, 1.0],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )


STRAIN_GRAD_BASES = build_strain_gradient_bases()
DEVIATORIC_STRESS_BASES = build_deviatoric_stress_bases()
HYDROSTATIC_STRESS_BASIS = TRACE_VOIGT / math.sqrt(2.0)


@dataclass(frozen=True)
class AlgorithmResult:
    """Compact metrics for one completed algorithm."""

    name: str
    u_l2_error: float
    sigma_l2_error: float
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

    raw_sigma_basis_test: torch.Tensor
    active_u_basis_test: torch.Tensor
    w_test: torch.Tensor
    u_exact_test: torch.Tensor
    sigma_exact_test: torch.Tensor


@dataclass
class LeastSquaresConfig:
    """Configuration for the raw/active-basis least-squares experiment."""

    E: float = 4.0 / 3.0
    nu: float = 1.0 / 3.0
    gamma_s: float = 2.0
    gamma_u: float = 2.0
    N_s: int = 1000
    N_u: int = 1000
    Q_train: int = (2**8) ** 2
    Q_test: int = (2**7) ** 2
    sampling_method: str = "sobol"
    manufactured_solution: str = "hu_zhang"
    tsvd_tau_rel: float = 1.0e-15
    ridge_alpha_rel: float = 1.0e-15
    body_force_batch_size: int = 5_000
    assembly_batch_size: int = 5_000
    algorithms_to_run: list[str] = field(
        default_factory=lambda: [
            "lstsq",
            "tsvd",
            "ridge",
        ]
    )


@dataclass(frozen=True)
class LeastSquaresExperimentData:
    """All tensors needed to run and evaluate one least-squares solver."""

    G: torch.Tensor
    F: torch.Tensor
    solved_dim_s: int
    stress_adapter: "StressBasisAdapter"
    eval_data: FeatureEvaluationData


@dataclass(frozen=True)
class StressBasisAdapter:
    """Linear map between active stress coefficients and the raw stress basis."""

    transform: torch.Tensor
    constraint: torch.Tensor
    raw_dim: int
    active_dim: int


@dataclass(frozen=True)
class RawStressLinearBlocks:
    """Raw stress-space blocks before the active-stress projection is applied."""

    G_ss_raw: torch.Tensor
    G_su_raw: torch.Tensor
    G_uu: torch.Tensor
    F_s_raw: torch.Tensor


@dataclass(frozen=True)
class AssembledLinearSystem:
    """Linear system plus metadata needed to interpret its stress block."""

    G: torch.Tensor
    F: torch.Tensor
    solved_dim_s: int
    stress_adapter: StressBasisAdapter


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


def validate_manufactured_solution(solution_id: str) -> None:
    """Reject unsupported manufactured-solution identifiers early."""

    if solution_id not in VALID_MANUFACTURED_SOLUTIONS:
        raise ValueError(
            f"Unknown manufactured_solution='{solution_id}'. "
            f"Valid values: {list(VALID_MANUFACTURED_SOLUTIONS)}"
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
    if cfg.N_s <= 0 or cfg.N_u <= 0:
        raise ValueError("Config.N_s and Config.N_u must be positive.")
    if cfg.Q_train <= 0:
        raise ValueError("Config.Q_train must be positive.")
    if cfg.Q_test <= 0:
        raise ValueError("Config.Q_test must be positive.")
    if not math.isfinite(cfg.tsvd_tau_rel) or cfg.tsvd_tau_rel < 0.0:
        raise ValueError("Config.tsvd_tau_rel must be finite and non-negative.")
    if not math.isfinite(cfg.ridge_alpha_rel) or cfg.ridge_alpha_rel <= 0.0:
        raise ValueError("Config.ridge_alpha_rel must be finite and positive.")
    if cfg.body_force_batch_size <= 0:
        raise ValueError("Config.body_force_batch_size must be positive.")
    if cfg.assembly_batch_size <= 0:
        raise ValueError("Config.assembly_batch_size must be positive.")
    validate_sampling_method(cfg.sampling_method)
    validate_manufactured_solution(cfg.manufactured_solution)
    validate_algorithm_selection(cfg.algorithms_to_run, VALID_ALGORITHMS)


def resolve_output_dir(manufactured_solution: str) -> Path:
    """Return the fixed output directory used by this experiment."""

    validate_manufactured_solution(manufactured_solution)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def compute_lame_constants(E: float, nu: float) -> tuple[float, float]:
    """Return Lamé constants (mu, lambda)."""

    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def build_compliance_matrix(mu: float, lam: float) -> torch.Tensor:
    """Build the 2D isotropic compliance matrix in engineering Voigt form."""

    stiffness = torch.tensor(
        [
            [2.0 * mu + lam, lam, 0.0],
            [lam, 2.0 * mu + lam, 0.0],
            [0.0, 0.0, mu],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )
    return torch.linalg.inv(stiffness)


def eval_hu_zhang_displacement(x: torch.Tensor) -> torch.Tensor:
    """Evaluate the Hu-Zhang 2D manufactured displacement field."""

    x1, x2 = x[:, 0], x[:, 1]
    u1 = torch.exp(x1 - x2) * x1 * (1.0 - x1) * x2 * (1.0 - x2)
    u2 = torch.sin(math.pi * x1) * torch.sin(math.pi * x2)
    return torch.stack([u1, u2], dim=1)


def eval_div_free_displacement(x: torch.Tensor) -> torch.Tensor:
    """Evaluate a divergence-free manufactured displacement with zero boundary trace."""

    x1, x2 = x[:, 0], x[:, 1]
    pi = math.pi
    h1 = torch.sin(2.0 * pi * x1)
    h2 = torch.sin(2.0 * pi * x2)
    g1 = (1.0 - torch.cos(2.0 * pi * x1)) / (2.0 * pi)
    g2 = (1.0 - torch.cos(2.0 * pi * x2)) / (2.0 * pi)

    u1 = -g1 * h2
    u2 = h1 * g2
    return torch.stack([u1, u2], dim=1)


def eval_near_incompressible_displacement(x: torch.Tensor, lam: float) -> torch.Tensor:
    """Evaluate a lambda-scaled nearly incompressible zero-boundary displacement."""

    if not math.isfinite(lam) or lam == 0.0:
        raise ValueError("lam must be finite and nonzero for near_incompressible.")

    x1, x2 = x[:, 0], x[:, 1]
    pi = math.pi
    div_free = eval_div_free_displacement(x)
    perturbation = torch.stack(
        [
            torch.sin(2.0 * pi * x1) * x2 * (1.0 - x2),
            torch.sin(2.0 * pi * x2) * x1 * (1.0 - x1),
        ],
        dim=1,
    )
    return div_free + perturbation / lam


def eval_exact_displacement(
    x: torch.Tensor,
    manufactured_solution: str,
    lam: float | None = None,
) -> torch.Tensor:
    """Evaluate the selected manufactured displacement field."""

    validate_manufactured_solution(manufactured_solution)
    if manufactured_solution == "hu_zhang":
        return eval_hu_zhang_displacement(x)
    if manufactured_solution == "div_free":
        return eval_div_free_displacement(x)
    if manufactured_solution == "near_incompressible":
        if lam is None:
            raise ValueError("lam is required for manufactured_solution='near_incompressible'.")
        return eval_near_incompressible_displacement(x, lam)
    raise AssertionError(f"Unhandled manufactured_solution='{manufactured_solution}'.")


def compute_engineering_strain(grad_u: torch.Tensor) -> torch.Tensor:
    """Convert displacement gradients to 2D engineering Voigt strain."""

    return torch.stack(
        [
            grad_u[:, 0, 0],
            grad_u[:, 1, 1],
            grad_u[:, 0, 1] + grad_u[:, 1, 0],
        ],
        dim=1,
    )


def apply_isotropic_stiffness(
    strain_voigt: torch.Tensor,
    mu: float,
    lam: float,
) -> torch.Tensor:
    """Apply the 2D isotropic stiffness operator to engineering strain."""

    sigma11 = (2.0 * mu + lam) * strain_voigt[:, 0] + lam * strain_voigt[:, 1]
    sigma22 = lam * strain_voigt[:, 0] + (2.0 * mu + lam) * strain_voigt[:, 1]
    sigma12 = mu * strain_voigt[:, 2]
    return torch.stack([sigma11, sigma22, sigma12], dim=1)


def compute_stress_voigt(
    x: torch.Tensor,
    mu: float,
    lam: float,
    manufactured_solution: str,
) -> torch.Tensor:
    """Evaluate the exact stress in Voigt order (11, 22, 12)."""

    x_ad = x.detach().requires_grad_(True)
    u = eval_exact_displacement(
        x_ad,
        manufactured_solution=manufactured_solution,
        lam=lam,
    )

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
    return apply_isotropic_stiffness(strain_voigt, mu, lam).detach()


def compute_body_force(
    x: torch.Tensor,
    mu: float,
    lam: float,
    batch_size: int,
    manufactured_solution: str,
) -> torch.Tensor:
    """Compute f = -div(sigma(u_exact)) with batched autodiff."""

    n_points = x.shape[0]
    f_all = torch.zeros(n_points, 2, dtype=DTYPE, device=DEVICE)

    for start in range(0, n_points, batch_size):
        end = min(start + batch_size, n_points)
        xb = x[start:end].detach().requires_grad_(True)
        u = eval_exact_displacement(
            xb,
            manufactured_solution=manufactured_solution,
            lam=lam,
        )

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
        sigma = apply_isotropic_stiffness(strain_voigt, mu, lam)

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


def eval_displacement_adapter_weight(x: torch.Tensor) -> torch.Tensor:
    """Evaluate the scalar weight used by the displacement basis adapter."""

    return x[:, 0] * (1.0 - x[:, 0]) * x[:, 1] * (1.0 - x[:, 1])


def eval_displacement_adapter_weight_grad(x: torch.Tensor) -> torch.Tensor:
    """Evaluate the gradient of the displacement basis-adapter weight."""

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

    raw = torch.randn(N, FEATURE_DIM, generator=generator, dtype=DTYPE)
    norms = raw.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
    a = (raw / norms).to(DEVICE)
    r = torch.rand(N, generator=generator, dtype=DTYPE).to(DEVICE)
    return a, r


def normalize_feature_coordinates(x: torch.Tensor) -> torch.Tensor:
    """Map unit-box coordinates into the centered unit ball for features."""

    return (x - FEATURE_CENTER) * FEATURE_INV_RADIUS


def eval_raw_scalar_basis(
    x: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Evaluate the raw scalar basis with a leading constant feature."""

    x_hat = normalize_feature_coordinates(x)
    pre = x_hat @ a.T + r.unsqueeze(0)
    raw_basis = torch.tanh(gamma * pre)
    ones = torch.ones(x.shape[0], 1, dtype=DTYPE, device=DEVICE)
    return torch.cat([ones, raw_basis], dim=1)


def eval_raw_scalar_basis_grads(
    x: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Evaluate gradients of the raw scalar basis."""

    x_hat = normalize_feature_coordinates(x)
    pre = x_hat @ a.T + r.unsqueeze(0)
    dtanh = 1.0 - torch.tanh(gamma * pre).square()
    grad_xi = gamma * FEATURE_INV_RADIUS * dtanh.unsqueeze(2) * a.unsqueeze(0)
    zeros = torch.zeros(x.shape[0], 1, FEATURE_DIM, dtype=DTYPE, device=DEVICE)
    return torch.cat([zeros, grad_xi], dim=1)


def eval_active_displacement_basis(
    x: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Evaluate the active displacement basis built from the raw scalar basis."""

    raw_basis = eval_raw_scalar_basis(x, a, r, gamma)
    adapter_weight = eval_displacement_adapter_weight(x)
    return adapter_weight.unsqueeze(1) * raw_basis


def eval_active_displacement_basis_data(
    x: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the active displacement basis and its gradients."""

    raw_basis = eval_raw_scalar_basis(x, a, r, gamma)
    raw_basis_grad = eval_raw_scalar_basis_grads(x, a, r, gamma)
    adapter_weight = eval_displacement_adapter_weight(x)
    adapter_weight_grad = eval_displacement_adapter_weight_grad(x)
    active_basis = adapter_weight.unsqueeze(1) * raw_basis
    active_basis_grad = (
        raw_basis.unsqueeze(2) * adapter_weight_grad.unsqueeze(1)
        + adapter_weight.view(-1, 1, 1) * raw_basis_grad
    )
    return active_basis, active_basis_grad


def build_shared_benchmark(
    E: float,
    nu: float,
    Q_train: int,
    Q_test: int,
    sampling_method: str,
    body_force_batch_size: int,
    manufactured_solution: str,
    interior_seed: int = BASE_SEED + 1,
    test_seed: int = BASE_SEED + 3,
) -> SharedBenchmarkData:
    """Build the shared train/test samples for a fair comparison run."""

    validate_manufactured_solution(manufactured_solution)
    mu, lam = compute_lame_constants(E, nu)

    x_int, w_int = build_quadrature_rule(
        Q_train,
        method=sampling_method,
        seed=interior_seed,
    )
    f_int = compute_body_force(
        x_int,
        mu,
        lam,
        batch_size=body_force_batch_size,
        manufactured_solution=manufactured_solution,
    )

    x_test, w_test = build_quadrature_rule(
        Q_test,
        method=sampling_method,
        seed=test_seed,
    )
    u_exact_test = eval_exact_displacement(
        x_test,
        manufactured_solution=manufactured_solution,
        lam=lam,
    )
    sigma_exact_test = compute_stress_voigt(
        x_test,
        mu,
        lam,
        manufactured_solution=manufactured_solution,
    )
    return SharedBenchmarkData(
        x_int=x_int,
        w_int=w_int,
        f_int=f_int,
        x_test=x_test,
        w_test=w_test,
        u_exact_test=u_exact_test,
        sigma_exact_test=sigma_exact_test,
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
        raw_sigma_basis_test=eval_raw_scalar_basis(
            benchmark.x_test,
            feature_space.a_s,
            feature_space.r_s,
            feature_space.gamma_s,
        ),
        active_u_basis_test=eval_active_displacement_basis(
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
    torch.Tensor,
    list[torch.Tensor],
    list[list[torch.Tensor]],
    list[list[torch.Tensor]],
    list[torch.Tensor],
]:
    """Accumulate raw/active basis moments for the least-squares linear system."""

    np1_s = a_s.shape[0] + 1
    np1_u = a_u.shape[0] + 1

    gram_raw_sigma = torch.zeros(np1_s, np1_s, dtype=DTYPE, device=DEVICE)
    mean_raw_sigma = torch.zeros(np1_s, dtype=DTYPE, device=DEVICE)
    cross_raw_sigma_grad_active_u = [
        torch.zeros(np1_s, np1_u, dtype=DTYPE, device=DEVICE) for _ in range(2)
    ]
    grad_gram_active_u = [
        [
            torch.zeros(np1_u, np1_u, dtype=DTYPE, device=DEVICE)
            for _ in range(2)
        ]
        for _ in range(2)
    ]
    grad_gram_raw_sigma = [
        [
            torch.zeros(np1_s, np1_s, dtype=DTYPE, device=DEVICE)
            for _ in range(2)
        ]
        for _ in range(2)
    ]
    grad_force_raw_sigma = [
        torch.zeros(np1_s, 2, dtype=DTYPE, device=DEVICE) for _ in range(2)
    ]

    with torch.no_grad():
        for start in range(0, x_int.shape[0], batch_size):
            end = min(start + batch_size, x_int.shape[0])
            xb = x_int[start:end]
            wb = w_int[start:end]
            fb = f_int[start:end]

            raw_sigma_basis_batch = eval_raw_scalar_basis(xb, a_s, r_s, gamma_s)
            raw_sigma_basis_grad_batch = eval_raw_scalar_basis_grads(
                xb,
                a_s,
                r_s,
                gamma_s,
            )
            _, active_u_basis_grad_batch = eval_active_displacement_basis_data(
                xb,
                a_u,
                r_u,
                gamma_u,
            )

            weighted_raw_sigma = wb.unsqueeze(1) * raw_sigma_basis_batch
            weighted_grad_active_u = [
                wb.unsqueeze(1) * active_u_basis_grad_batch[:, :, dim_i]
                for dim_i in range(2)
            ]
            weighted_grad_raw_sigma = [
                wb.unsqueeze(1) * raw_sigma_basis_grad_batch[:, :, dim_i]
                for dim_i in range(2)
            ]
            gram_raw_sigma += raw_sigma_basis_batch.T @ weighted_raw_sigma
            mean_raw_sigma += weighted_raw_sigma.sum(dim=0)
            for dim_i in range(2):
                cross_raw_sigma_grad_active_u[dim_i] += (
                    raw_sigma_basis_batch.T @ weighted_grad_active_u[dim_i]
                )
                grad_force_raw_sigma[dim_i] += weighted_grad_raw_sigma[dim_i].T @ fb
                for dim_j in range(2):
                    grad_gram_active_u[dim_i][dim_j] += (
                        active_u_basis_grad_batch[:, :, dim_i].T
                        @ weighted_grad_active_u[dim_j]
                    )
                    grad_gram_raw_sigma[dim_i][dim_j] += (
                        raw_sigma_basis_grad_batch[:, :, dim_i].T
                        @ weighted_grad_raw_sigma[dim_j]
                    )

    return (
        gram_raw_sigma,
        mean_raw_sigma,
        cross_raw_sigma_grad_active_u,
        grad_gram_active_u,
        grad_gram_raw_sigma,
        grad_force_raw_sigma,
    )


def build_stress_trace_constraint(mean_raw_sigma: torch.Tensor) -> torch.Tensor:
    """Return c such that c^T sigma_coeffs = ∫ tr(sigma_h) dx."""

    return (mean_raw_sigma.unsqueeze(1) * TRACE_VOIGT.unsqueeze(0)).reshape(-1)


def build_zero_mean_hydrostatic_transform(
    mean_raw_sigma: torch.Tensor,
    tol: float = 1.0e-12,
) -> torch.Tensor:
    """Map hydrostatic active coefficients to raw zero-mean scalar coefficients."""

    np1_s = mean_raw_sigma.numel()
    mean_0 = mean_raw_sigma[0]
    if torch.abs(mean_0) <= tol:
        raise ValueError("Raw stress basis has degenerate mean; cannot build zero-mean basis.")

    transform = torch.zeros(
        np1_s,
        max(np1_s - 1, 0),
        dtype=mean_raw_sigma.dtype,
        device=mean_raw_sigma.device,
    )
    if np1_s <= 1:
        return transform

    transform[0, :] = -mean_raw_sigma[1:] / mean_0
    transform[1:, :] = torch.eye(
        np1_s - 1,
        dtype=mean_raw_sigma.dtype,
        device=mean_raw_sigma.device,
    )
    return transform


def build_stress_basis_adapter(
    mean_raw_sigma: torch.Tensor,
    tol: float = 1.0e-10,
) -> StressBasisAdapter:
    """Build the active stress basis using deviatoric and zero-mean hydrostatic modes."""

    np1_s = mean_raw_sigma.numel()
    hydro_transform = build_zero_mean_hydrostatic_transform(mean_raw_sigma)
    identity_features = torch.eye(np1_s, dtype=DTYPE, device=DEVICE)
    raw_dim = 3 * np1_s

    transform_deviatoric = torch.kron(identity_features, DEVIATORIC_STRESS_BASES)
    transform_hydrostatic = torch.kron(
        hydro_transform,
        HYDROSTATIC_STRESS_BASIS.unsqueeze(1),
    )
    transform = torch.cat([transform_deviatoric, transform_hydrostatic], dim=1)
    constraint = build_stress_trace_constraint(mean_raw_sigma)
    constraint_residual = torch.abs(constraint @ transform).max()
    if constraint_residual > tol:
        raise ValueError(
            "Stress active basis violates the trace-mean constraint: "
            f"{constraint_residual.item():.2e}"
        )

    return StressBasisAdapter(
        transform=transform,
        constraint=constraint,
        raw_dim=raw_dim,
        active_dim=transform.shape[1],
    )


def assemble_weighted_raw_stress_blocks(
    mu: float,
    lam: float,
    gram_raw_sigma: torch.Tensor,
    cross_raw_sigma_grad_active_u: list[torch.Tensor],
    grad_gram_active_u: list[list[torch.Tensor]],
    grad_gram_raw_sigma: list[list[torch.Tensor]],
    grad_force_raw_sigma: list[torch.Tensor],
) -> RawStressLinearBlocks:
    """Assemble the two-term least-squares system in the raw stress basis."""

    np1_s = gram_raw_sigma.shape[0]
    np1_u = grad_gram_active_u[0][0].shape[0]
    dim_s = 3 * np1_s
    dim_u = 2 * np1_u

    G_ss = torch.zeros(dim_s, dim_s, dtype=DTYPE, device=DEVICE)
    G_su = torch.zeros(dim_s, dim_u, dtype=DTYPE, device=DEVICE)
    G_uu = torch.zeros(dim_u, dim_u, dtype=DTYPE, device=DEVICE)
    F_s = torch.zeros(dim_s, dtype=DTYPE, device=DEVICE)

    compliance_voigt = build_compliance_matrix(mu, lam)
    compliance_sq = compliance_voigt.T @ compliance_voigt
    add_block_scaled(
        G_ss,
        gram_raw_sigma,
        compliance_sq,
        row_stride=3,
        col_stride=3,
    )

    for dim_i in range(2):
        constitutive_cross = compliance_voigt @ STRAIN_GRAD_BASES[dim_i]
        add_block_scaled(
            G_su,
            cross_raw_sigma_grad_active_u[dim_i],
            -constitutive_cross,
            row_stride=3,
            col_stride=2,
        )
        add_rhs_feature_blocks(
            F_s,
            grad_force_raw_sigma[dim_i],
            STRAIN_GRAD_BASES[dim_i],
            block_size=3,
            scale=-1.0,
        )

        for dim_j in range(2):
            constitutive_uu = STRAIN_GRAD_BASES[dim_i].T @ STRAIN_GRAD_BASES[dim_j]
            equilibrium_ss = STRAIN_GRAD_BASES[dim_i] @ STRAIN_GRAD_BASES[dim_j].T
            add_block_scaled(
                G_uu,
                grad_gram_active_u[dim_i][dim_j],
                constitutive_uu,
                row_stride=2,
                col_stride=2,
            )
            add_block_scaled(
                G_ss,
                grad_gram_raw_sigma[dim_i][dim_j],
                equilibrium_ss,
                row_stride=3,
                col_stride=3,
            )

    return RawStressLinearBlocks(
        G_ss_raw=G_ss,
        G_su_raw=G_su,
        G_uu=G_uu,
        F_s_raw=F_s,
    )


def apply_stress_basis_adapter(
    raw_blocks: RawStressLinearBlocks,
    stress_adapter: StressBasisAdapter,
) -> AssembledLinearSystem:
    """Project raw stress blocks onto the active stress basis."""

    transform = stress_adapter.transform
    active_dim_s = stress_adapter.active_dim
    dim_u = raw_blocks.G_uu.shape[0]
    G_ss = transform.T @ raw_blocks.G_ss_raw @ transform
    G_su = transform.T @ raw_blocks.G_su_raw
    F_s = transform.T @ raw_blocks.F_s_raw

    G = torch.zeros(active_dim_s + dim_u, active_dim_s + dim_u, dtype=DTYPE, device=DEVICE)
    G[:active_dim_s, :active_dim_s] = G_ss
    G[:active_dim_s, active_dim_s:] = G_su
    G[active_dim_s:, :active_dim_s] = G_su.T
    G[active_dim_s:, active_dim_s:] = raw_blocks.G_uu
    G = 0.5 * (G + G.T)

    F = torch.zeros(active_dim_s + dim_u, dtype=DTYPE, device=DEVICE)
    F[:active_dim_s] = F_s
    return AssembledLinearSystem(
        G=G,
        F=F,
        solved_dim_s=active_dim_s,
        stress_adapter=stress_adapter,
    )


def assemble_linear_system(
    cfg: LeastSquaresConfig,
    x_int: torch.Tensor,
    w_int: torch.Tensor,
    f_int: torch.Tensor,
    a_s: torch.Tensor,
    r_s: torch.Tensor,
    a_u: torch.Tensor,
    r_u: torch.Tensor,
) -> AssembledLinearSystem:
    """Assemble the least-squares system in the active basis."""

    (
        gram_raw_sigma,
        mean_raw_sigma,
        cross_raw_sigma_grad_active_u,
        grad_gram_active_u,
        grad_gram_raw_sigma,
        grad_force_raw_sigma,
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

    stress_adapter = build_stress_basis_adapter(mean_raw_sigma)
    mu, lam = compute_lame_constants(cfg.E, cfg.nu)
    raw_blocks = assemble_weighted_raw_stress_blocks(
        mu,
        lam,
        gram_raw_sigma,
        cross_raw_sigma_grad_active_u,
        grad_gram_active_u,
        grad_gram_raw_sigma,
        grad_force_raw_sigma,
    )
    return apply_stress_basis_adapter(raw_blocks, stress_adapter)


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


def solve_tsvd(
    G: torch.Tensor,
    F: torch.Tensor,
    tau_rel: float,
) -> tuple[torch.Tensor, float]:
    """Solve the linear system with relative TSVD threshold tau_TSVD."""

    synchronize_device()
    t0 = time.perf_counter()
    try:
        eigvals, eigvecs = torch.linalg.eigh(G)
        threshold = tau_rel * eigvals.abs().max()
        keep = eigvals > threshold
        if not keep.any():
            raise RuntimeError("all eigenvalues were truncated")

        coeffs = eigvecs[:, keep].T @ F
        coeffs = coeffs / eigvals[keep]
        sol = eigvecs[:, keep] @ coeffs
        if not torch.isfinite(sol).all():
            raise RuntimeError("non-finite solution")
        print(
            f"    tsvd truncation: kept {int(keep.sum().item())}/{eigvals.numel()} "
            f"eigenvalues, tau_TSVD={tau_rel:.2e}, threshold={threshold.item():.2e}"
        )
    except (RuntimeError, torch.linalg.LinAlgError) as exc:
        sol = torch.full((G.shape[0],), float("nan"), dtype=DTYPE, device=DEVICE)
        print(f"    Warning: torch.linalg.eigh failed with {type(exc).__name__}")

    synchronize_device()
    return sol, time.perf_counter() - t0


def solve_ridge(G: torch.Tensor, F: torch.Tensor, alpha_Ridge: float) -> tuple[torch.Tensor, float]:
    """Solve the standard Tikhonov/ridge system with alpha_Ridge."""

    synchronize_device()
    t0 = time.perf_counter()
    try:
        spectral_scale = torch.linalg.eigvalsh(G).abs().max()
        alpha = alpha_Ridge * spectral_scale
        if not torch.isfinite(spectral_scale) or spectral_scale <= 0.0:
            raise RuntimeError("invalid spectral scale")
        if not torch.isfinite(alpha) or alpha <= 0.0:
            raise RuntimeError("invalid ridge strength")

        ridge_matrix = G + alpha * torch.eye(G.shape[0], dtype=DTYPE, device=DEVICE)
        sol = torch.linalg.lstsq(ridge_matrix, F.unsqueeze(1)).solution.squeeze(1)
        if not torch.isfinite(sol).all():
            raise RuntimeError("non-finite solution")
        print(
            f"    ridge shift: alpha_Ridge={alpha_Ridge:.2e}, "
            f"alpha={alpha.item():.2e}, scale={spectral_scale.item():.2e}"
        )
    except (RuntimeError, torch.linalg.LinAlgError) as exc:
        sol = torch.full((G.shape[0],), float("nan"), dtype=DTYPE, device=DEVICE)
        print(f"    Warning: torch.linalg.solve failed with {type(exc).__name__}")

    synchronize_device()
    return sol, time.perf_counter() - t0


def split_solution(z: torch.Tensor, solved_dim_s: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the solver coefficient vector into stress and displacement parts."""

    return z[:solved_dim_s], z[solved_dim_s:]


def lift_active_stress_coefficients(
    active_sigma_coeffs: torch.Tensor,
    stress_adapter: StressBasisAdapter,
) -> torch.Tensor:
    """Lift active stress coefficients back to the raw stress basis."""

    return stress_adapter.transform @ active_sigma_coeffs


def compute_absolute_errors(
    active_u_basis_test: torch.Tensor,
    raw_sigma_basis_test: torch.Tensor,
    sigma_coeffs: torch.Tensor,
    displacement_coeffs: torch.Tensor,
    w_test: torch.Tensor,
    u_exact: torch.Tensor,
    sigma_exact: torch.Tensor,
) -> tuple[float, float]:
    """Compute the absolute L2 errors for displacement and stress."""

    displacement_blocks = displacement_coeffs.reshape(-1, 2)
    stress_blocks = sigma_coeffs.reshape(-1, 3)

    u_h = active_u_basis_test @ displacement_blocks
    sigma_h = raw_sigma_basis_test @ stress_blocks

    u_l2_error = torch.sqrt((w_test * (u_h - u_exact).square().sum(dim=1)).sum())
    sigma_l2_error = torch.sqrt(
        (w_test * (VOIGT_WEIGHT * (sigma_h - sigma_exact).square()).sum(dim=1)).sum()
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
        data.active_u_basis_test,
        data.raw_sigma_basis_test,
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
        f"‖Φ^σ-σ‖={result.sigma_l2_error:.2e}"
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
    """Plot the final absolute error metrics as bar charts."""

    if not results:
        print(f"  Skipped: {save_path} (no results to plot)")
        return

    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))
    titles = [
        r"$\|\Phi^u - u_{ex}\|_0$",
        r"$\|\Phi^{\sigma} - \sigma_{ex}\|_0$",
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

    def build_result(name: str, z: torch.Tensor, wall_time: float) -> AlgorithmResult:
        active_sigma_coeffs, displacement_coeffs = split_solution(z, data.solved_dim_s)
        sigma_coeffs = lift_active_stress_coefficients(
            active_sigma_coeffs,
            data.stress_adapter,
        )
        constraint_residual = torch.dot(
            data.stress_adapter.constraint,
            sigma_coeffs,
        ).abs()
        print(
            "    trace constraint residual: "
            f"{constraint_residual.item():.2e}"
        )
        return evaluate_feature_result(
            name,
            wall_time,
            sigma_coeffs,
            displacement_coeffs,
            data.eval_data,
        )

    if algorithm_id == "tsvd":
        print("Running LS (TSVD)...")
        z, wall_time = solve_tsvd(data.G, data.F, cfg.tsvd_tau_rel)
        result = build_result("LS (TSVD)", z, wall_time)
    elif algorithm_id == "ridge":
        print("Running LS (Ridge)...")
        z, wall_time = solve_ridge(data.G, data.F, cfg.ridge_alpha_rel)
        result = build_result("LS (Ridge)", z, wall_time)
    else:
        print("Running LS (Lstsq)...")
        z, wall_time = solve_lstsq(data.G, data.F)
        result = build_result("LS (Lstsq)", z, wall_time)

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
    output_dir = resolve_output_dir(cfg.manufactured_solution)

    print(f"Device: {DEVICE}")
    print(f"Output: {output_dir}")
    print(
        f"Config: N_s={cfg.N_s}, N_u={cfg.N_u}, "
        f"Q_train={cfg.Q_train}, Q_test={cfg.Q_test}, "
        f"gamma_s={cfg.gamma_s}, gamma_u={cfg.gamma_u}, "
        f"tsvd_tau_rel={cfg.tsvd_tau_rel:.2e}, "
        f"ridge_alpha_rel={cfg.ridge_alpha_rel:.2e}, "
        f"sampling={cfg.sampling_method}, "
        f"manufactured_solution={cfg.manufactured_solution}"
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
            manufactured_solution=cfg.manufactured_solution,
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

    print("Assembling active-basis least-squares system...")
    assembled_system = assemble_linear_system(
        cfg,
        benchmark.x_int,
        benchmark.w_int,
        benchmark.f_int,
        feature_space.a_s,
        feature_space.r_s,
        feature_space.a_u,
        feature_space.r_u,
    )
    clear_cuda_cache()

    print(
        f"System shapes: G={tuple(assembled_system.G.shape)}, "
        f"F={tuple(assembled_system.F.shape)}"
    )
    print(
        "trace constraint applied: "
        f"stress dof {assembled_system.stress_adapter.raw_dim} "
        f"-> {assembled_system.stress_adapter.active_dim}"
    )

    experiment_data = LeastSquaresExperimentData(
        G=assembled_system.G,
        F=assembled_system.F,
        solved_dim_s=assembled_system.solved_dim_s,
        stress_adapter=assembled_system.stress_adapter,
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
        plot_l2_summary(results, str(output_dir / "l2-error-summary.png"))

    return results


def main(cfg: LeastSquaresConfig | None = None) -> None:
    """Script entrypoint."""

    run_experiment(cfg, print_table=True, plot_results=True)


if __name__ == "__main__":
    main()
