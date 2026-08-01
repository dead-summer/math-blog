from __future__ import annotations

import math
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import get_lapack_funcs
import torch


BASE_SEED = 42
STRESS_SEED = BASE_SEED
DISP_SEED = BASE_SEED + 1_000
DTYPE = torch.float64
VALID_SAMPLING_METHODS = ("mc", "sobol", "gauss_legendre")
VALID_ALGORITHMS = ("direct",)
VALID_DIRECT_SOLVERS = ("dense", "streaming_tsqr")
VALID_MANUFACTURED_SOLUTIONS = ("hu_zhang", "div_free", "near_incompressible")
FEATURE_DIM = 3
FEATURE_CENTER = 0.5
FEATURE_INV_RADIUS = 2.0 / math.sqrt(FEATURE_DIM)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "public" / "images" / "least-squares" / "linear-elasticity-3d"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ALGO_STYLE = {
    "LS": {"color": "#0077B6", "marker": "o", "linestyle": "-"},
}

def detect_device() -> torch.device:
    """Prefer CUDA when available and stay quiet otherwise."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


DEVICE = torch.device("cpu")

VOIGT_WEIGHT = torch.tensor(
    [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
    dtype=DTYPE,
    device=DEVICE,
)
TRACE_VOIGT = torch.tensor(
    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    dtype=DTYPE,
    device=DEVICE,
)

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


def build_deviatoric_stress_bases() -> torch.Tensor:
    """Return a fixed 6x5 basis spanning trace-free stresses in Voigt form."""

    return torch.tensor(
        [
            [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(6.0), 0.0, 0.0, 0.0],
            [-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(6.0), 0.0, 0.0, 0.0],
            [0.0, -2.0 / math.sqrt(6.0), 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )


STRAIN_GRAD_BASES = build_strain_gradient_bases()
DEVIATORIC_STRESS_BASES = build_deviatoric_stress_bases()
HYDROSTATIC_STRESS_BASIS = TRACE_VOIGT / math.sqrt(3.0)


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
    Q_train: int = 16**3
    Q_test: int = 16**3
    sampling_method: str = "gauss_legendre"
    manufactured_solution: str = "div_free"
    direct_rcond: float = 1.0e-14
    direct_solver: str = "streaming_tsqr"
    direct_batch_size: int = 1_024
    direct_qr_block_size: int = 64
    body_force_batch_size: int = 5_000
    algorithms_to_run: list[str] = field(
        default_factory=lambda: ["direct"]
    )


@dataclass(frozen=True)
class LeastSquaresExperimentData:
    """All tensors needed to run and evaluate one least-squares solver."""

    matrix: torch.Tensor
    rhs: torch.Tensor
    solved_dim_s: int
    stress_adapter: "StressBasisAdapter"
    eval_data: FeatureEvaluationData
    source_rows: int
    preparation_time: float
    direct_solver: str


@dataclass(frozen=True)
class StressBasisAdapter:
    """Linear map between active stress coefficients and the raw stress basis."""

    transform: torch.Tensor | None
    constraint: torch.Tensor
    mean_raw_sigma: torch.Tensor
    raw_dim: int
    active_dim: int


@dataclass(frozen=True)
class DirectResidualDesign:
    """Weighted residual matrix plus coefficient-space metadata."""

    matrix: torch.Tensor
    rhs: torch.Tensor
    solved_dim_s: int
    stress_adapter: StressBasisAdapter


@dataclass(frozen=True)
class DirectResidualContext:
    """Dimensions and fixed coupling blocks shared by all residual batches."""

    stress_adapter: StressBasisAdapter
    np1_s: int
    np1_u: int
    deviatoric_dim_s: int
    active_dim_s: int
    dim_u: int
    columns: int
    compliance_deviatoric: torch.Tensor
    compliance_hydrostatic: torch.Tensor
    equilibrium_deviatoric: torch.Tensor
    equilibrium_hydrostatic: torch.Tensor


@dataclass(frozen=True)
class DirectResidualBatch:
    """One Fortran-contiguous augmented residual block ``[A_i, b_i]``."""

    start: int
    stop: int
    augmented: np.ndarray

    @property
    def matrix(self) -> np.ndarray:
        return self.augmented[:, :-1]

    @property
    def rhs(self) -> np.ndarray:
        return self.augmented[:, -1]


@dataclass(frozen=True)
class StreamingTSQRStats:
    """Timing and shape diagnostics for the streaming compression pass."""

    source_rows: int
    columns: int
    batch_count: int
    mean_pass_time: float
    assembly_time: float
    qr_time: float
    total_time: float


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
    if not math.isfinite(cfg.direct_rcond) or cfg.direct_rcond <= 0.0:
        raise ValueError("Config.direct_rcond must be finite and positive.")
    if cfg.direct_solver not in VALID_DIRECT_SOLVERS:
        raise ValueError(
            f"Unknown direct_solver='{cfg.direct_solver}'. "
            f"Valid values: {list(VALID_DIRECT_SOLVERS)}"
        )
    if cfg.direct_batch_size <= 0:
        raise ValueError("Config.direct_batch_size must be positive.")
    if cfg.direct_qr_block_size <= 0:
        raise ValueError("Config.direct_qr_block_size must be positive.")
    if cfg.body_force_batch_size <= 0:
        raise ValueError("Config.body_force_batch_size must be positive.")
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
    """Build the 3D isotropic compliance matrix in engineering Voigt form."""

    compliance = torch.zeros(6, 6, dtype=DTYPE, device=DEVICE)
    compliance[:3, :3] = torch.eye(3, dtype=DTYPE, device=DEVICE) / (2.0 * mu)
    compliance[:3, :3] -= (
        lam / (2.0 * mu * (2.0 * mu + 3.0 * lam))
    ) * torch.ones(3, 3, dtype=DTYPE, device=DEVICE)
    compliance[3, 3] = 1.0 / mu
    compliance[4, 4] = 1.0 / mu
    compliance[5, 5] = 1.0 / mu
    return compliance


def eval_hu_zhang_displacement(x: torch.Tensor) -> torch.Tensor:
    """Evaluate the Hu-Zhang 3D manufactured displacement field."""

    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    vanishing_factor = x1 * (1.0 - x1) * x2 * (1.0 - x2) * x3 * (1.0 - x3)
    u1 = 16.0 * vanishing_factor
    u2 = 32.0 * vanishing_factor
    u3 = 64.0 * vanishing_factor
    return torch.stack([u1, u2, u3], dim=1)


def eval_div_free_displacement(x: torch.Tensor) -> torch.Tensor:
    """Evaluate a divergence-free manufactured displacement with zero boundary trace."""

    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    pi = math.pi
    h1 = torch.sin(2.0 * pi * x1)
    h2 = torch.sin(2.0 * pi * x2)
    h3 = torch.sin(2.0 * pi * x3)
    g1 = (1.0 - torch.cos(2.0 * pi * x1)) / (2.0 * pi)
    g2 = (1.0 - torch.cos(2.0 * pi * x2)) / (2.0 * pi)
    g3 = (1.0 - torch.cos(2.0 * pi * x3)) / (2.0 * pi)

    u1 = -2.0 * g1 * h2 * h3
    u2 = g2 * h1 * h3
    u3 = g3 * h1 * h2
    return torch.stack([u1, u2, u3], dim=1)


def eval_near_incompressible_displacement(x: torch.Tensor, lam: float) -> torch.Tensor:
    """Evaluate a lambda-scaled nearly incompressible zero-boundary displacement."""

    if not math.isfinite(lam) or lam == 0.0:
        raise ValueError("lam must be finite and nonzero for near_incompressible.")

    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    pi = math.pi
    div_free = eval_div_free_displacement(x)
    perturbation = torch.stack(
        [
            torch.sin(2.0 * pi * x1) * x2 * (1.0 - x2) * x3 * (1.0 - x3),
            torch.sin(2.0 * pi * x2) * x1 * (1.0 - x1) * x3 * (1.0 - x3),
            torch.sin(2.0 * pi * x3) * x1 * (1.0 - x1) * x2 * (1.0 - x2),
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


def compute_stress_voigt(
    x: torch.Tensor,
    mu: float,
    lam: float,
    manufactured_solution: str,
) -> torch.Tensor:
    """Evaluate the exact stress in Voigt order (11, 22, 33, 12, 23, 13)."""

    x_ad = x.detach().requires_grad_(True)
    u = eval_exact_displacement(
        x_ad,
        manufactured_solution=manufactured_solution,
        lam=lam,
    )

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
    manufactured_solution: str,
) -> torch.Tensor:
    """Compute f = -div(sigma(u_exact)) with batched autodiff."""

    n_points = x.shape[0]
    f_all = torch.zeros(n_points, 3, dtype=DTYPE, device=DEVICE)

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


def eval_displacement_adapter_weight(x: torch.Tensor) -> torch.Tensor:
    """Evaluate the scalar weight used by the displacement basis adapter."""

    return (
        x[:, 0]
        * (1.0 - x[:, 0])
        * x[:, 1]
        * (1.0 - x[:, 1])
        * x[:, 2]
        * (1.0 - x[:, 2])
    )


def eval_displacement_adapter_weight_grad(x: torch.Tensor) -> torch.Tensor:
    """Evaluate the gradient of the displacement basis-adapter weight."""

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


def generate_features(N: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random feature normals and offsets."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    raw = torch.randn(N, 3, generator=generator, dtype=DTYPE)
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
    zeros = torch.zeros(x.shape[0], 1, 3, dtype=DTYPE, device=DEVICE)
    return torch.cat([zeros, grad_xi], dim=1)


def eval_raw_scalar_basis_data(
    x: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate scalar basis values and gradients with one feature projection."""

    x_hat = normalize_feature_coordinates(x)
    pre = x_hat @ a.T + r.unsqueeze(0)
    raw_features = torch.tanh(gamma * pre)
    ones = torch.ones(x.shape[0], 1, dtype=DTYPE, device=DEVICE)
    raw_basis = torch.cat([ones, raw_features], dim=1)
    dtanh = 1.0 - raw_features.square()
    grad_xi = gamma * FEATURE_INV_RADIUS * dtanh.unsqueeze(2) * a.unsqueeze(0)
    zeros = torch.zeros(x.shape[0], 1, 3, dtype=DTYPE, device=DEVICE)
    raw_basis_grad = torch.cat([zeros, grad_xi], dim=1)
    return raw_basis, raw_basis_grad


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

    raw_basis, raw_basis_grad = eval_raw_scalar_basis_data(x, a, r, gamma)
    adapter_weight = eval_displacement_adapter_weight(x)
    adapter_weight_grad = eval_displacement_adapter_weight_grad(x)
    active_basis = adapter_weight.unsqueeze(1) * raw_basis
    raw_basis_grad.mul_(adapter_weight.view(-1, 1, 1))
    raw_basis_grad.add_(raw_basis.unsqueeze(2) * adapter_weight_grad.unsqueeze(1))
    active_basis_grad = raw_basis_grad
    return active_basis, active_basis_grad


def eval_active_displacement_basis_grads(
    x: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Evaluate only active displacement gradients for residual assembly."""

    raw_basis, raw_basis_grad = eval_raw_scalar_basis_data(x, a, r, gamma)
    adapter_weight = eval_displacement_adapter_weight(x)
    adapter_weight_grad = eval_displacement_adapter_weight_grad(x)
    raw_basis_grad.mul_(adapter_weight.view(-1, 1, 1))
    raw_basis_grad.add_(raw_basis.unsqueeze(2) * adapter_weight_grad.unsqueeze(1))
    return raw_basis_grad


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
    materialize_transform: bool = True,
) -> StressBasisAdapter:
    """Build the active stress basis using deviatoric and zero-mean hydrostatic modes."""

    np1_s = mean_raw_sigma.numel()
    raw_dim = 6 * np1_s
    active_dim = raw_dim - 1
    constraint = build_stress_trace_constraint(mean_raw_sigma)
    transform = None

    if torch.abs(mean_raw_sigma[0]) <= 1.0e-12:
        raise ValueError("Raw stress basis has degenerate mean; cannot build zero-mean basis.")

    if materialize_transform:
        hydro_transform = build_zero_mean_hydrostatic_transform(mean_raw_sigma)
        identity_features = torch.eye(np1_s, dtype=DTYPE, device=DEVICE)
        transform_deviatoric = torch.kron(identity_features, DEVIATORIC_STRESS_BASES)
        transform_hydrostatic = torch.kron(
            hydro_transform,
            HYDROSTATIC_STRESS_BASIS.unsqueeze(1),
        )
        transform = torch.cat([transform_deviatoric, transform_hydrostatic], dim=1)
        constraint_residual = torch.abs(constraint @ transform).max()
        if constraint_residual > tol:
            raise ValueError(
                "Stress active basis violates the trace-mean constraint: "
                f"{constraint_residual.item():.2e}"
            )

    return StressBasisAdapter(
        transform=transform,
        constraint=constraint,
        mean_raw_sigma=mean_raw_sigma,
        raw_dim=raw_dim,
        active_dim=active_dim,
    )


def iter_point_batches(point_count: int, batch_size: int) -> Iterator[tuple[int, int]]:
    """Yield exact half-open point batches, including a possibly short tail."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, point_count, batch_size):
        yield start, min(start + batch_size, point_count)


def accumulate_raw_sigma_mean(
    benchmark: SharedBenchmarkData,
    feature_space: SharedFeatureSpace,
    batch_size: int,
) -> torch.Tensor:
    """Accumulate the global stress-basis mean without materializing all values."""

    mean_raw_sigma = torch.zeros(
        feature_space.a_s.shape[0] + 1,
        dtype=DTYPE,
        device=DEVICE,
    )
    with torch.no_grad():
        for start, stop in iter_point_batches(
            benchmark.x_int.shape[0],
            batch_size,
        ):
            raw_sigma = eval_raw_scalar_basis(
                benchmark.x_int[start:stop],
                feature_space.a_s,
                feature_space.r_s,
                feature_space.gamma_s,
            )
            mean_raw_sigma.add_(
                (
                    benchmark.w_int[start:stop].unsqueeze(1)
                    * raw_sigma
                ).sum(dim=0)
            )
    return mean_raw_sigma


def build_direct_residual_context(
    cfg: LeastSquaresConfig,
    feature_space: SharedFeatureSpace,
    mean_raw_sigma: torch.Tensor,
) -> DirectResidualContext:
    """Prepare dimensions and small material couplings for batch assembly."""

    np1_s = feature_space.a_s.shape[0] + 1
    np1_u = feature_space.a_u.shape[0] + 1
    if np1_s != cfg.N_s + 1 or np1_u != cfg.N_u + 1:
        raise ValueError("SharedFeatureSpace dimensions do not match LeastSquaresConfig.")
    if mean_raw_sigma.numel() != np1_s:
        raise ValueError("Stress-basis mean has the wrong dimension.")

    stress_adapter = build_stress_basis_adapter(
        mean_raw_sigma,
        materialize_transform=False,
    )
    deviatoric_dim_s = 5 * np1_s
    active_dim_s = stress_adapter.active_dim
    dim_u = 3 * np1_u

    mu, lam = compute_lame_constants(cfg.E, cfg.nu)
    compliance = build_compliance_matrix(mu, lam)
    compliance_deviatoric = compliance @ DEVIATORIC_STRESS_BASES
    compliance_hydrostatic = compliance @ HYDROSTATIC_STRESS_BASIS
    equilibrium_deviatoric = torch.empty(
        3,
        3,
        5,
        dtype=DTYPE,
        device=DEVICE,
    )
    equilibrium_hydrostatic = torch.empty(3, 3, dtype=DTYPE, device=DEVICE)
    for equilibrium_component in range(3):
        for spatial_dimension in range(3):
            coupling = STRAIN_GRAD_BASES[spatial_dimension][
                :, equilibrium_component
            ]
            equilibrium_deviatoric[equilibrium_component, spatial_dimension] = (
                coupling @ DEVIATORIC_STRESS_BASES
            )
            equilibrium_hydrostatic[equilibrium_component, spatial_dimension] = (
                torch.dot(coupling, HYDROSTATIC_STRESS_BASIS)
            )

    return DirectResidualContext(
        stress_adapter=stress_adapter,
        np1_s=np1_s,
        np1_u=np1_u,
        deviatoric_dim_s=deviatoric_dim_s,
        active_dim_s=active_dim_s,
        dim_u=dim_u,
        columns=active_dim_s + dim_u,
        compliance_deviatoric=compliance_deviatoric,
        compliance_hydrostatic=compliance_hydrostatic,
        equilibrium_deviatoric=equilibrium_deviatoric,
        equilibrium_hydrostatic=equilibrium_hydrostatic,
    )


def assemble_direct_residual_batch(
    start: int,
    stop: int,
    benchmark: SharedBenchmarkData,
    feature_space: SharedFeatureSpace,
    context: DirectResidualContext,
) -> DirectResidualBatch:
    """Assemble one augmented residual block in a single Fortran allocation."""

    point_count = benchmark.x_int.shape[0]
    if not (0 <= start < stop <= point_count):
        raise ValueError(
            f"Invalid residual batch [{start}, {stop}) for {point_count} points."
        )
    batch_points = stop - start
    augmented = np.zeros(
        (9 * batch_points, context.columns + 1),
        dtype=np.float64,
        order="F",
    )
    augmented_torch = torch.from_numpy(augmented)
    matrix = augmented_torch[:, : context.columns]
    rhs = augmented_torch[:, context.columns]

    x = benchmark.x_int[start:stop]
    weights = benchmark.w_int[start:stop]
    body_force = benchmark.f_int[start:stop]
    with torch.no_grad():
        raw_sigma, grad_sigma = eval_raw_scalar_basis_data(
            x,
            feature_space.a_s,
            feature_space.r_s,
            feature_space.gamma_s,
        )
        grad_u = eval_active_displacement_basis_grads(
            x,
            feature_space.a_u,
            feature_space.r_u,
            feature_space.gamma_u,
        )
        sqrt_weights = torch.sqrt(weights)
        mean_raw_sigma = context.stress_adapter.mean_raw_sigma
        hydrostatic_basis = raw_sigma[:, 1:] - raw_sigma[:, :1] * (
            mean_raw_sigma[1:] / mean_raw_sigma[0]
        ).unsqueeze(0)

        raw_sigma.mul_(sqrt_weights.unsqueeze(1))
        hydrostatic_basis.mul_(sqrt_weights.unsqueeze(1))
        grad_sigma.mul_(sqrt_weights.view(-1, 1, 1))
        grad_u.mul_(sqrt_weights.view(-1, 1, 1))

        for residual_component in range(6):
            rows = slice(
                residual_component * batch_points,
                (residual_component + 1) * batch_points,
            )
            for deviatoric_component in range(5):
                matrix[
                    rows,
                    deviatoric_component : context.deviatoric_dim_s : 5,
                ] = (
                    context.compliance_deviatoric[
                        residual_component,
                        deviatoric_component,
                    ]
                    * raw_sigma
                )
            matrix[
                rows,
                context.deviatoric_dim_s : context.active_dim_s,
            ] = (
                context.compliance_hydrostatic[residual_component]
                * hydrostatic_basis
            )
            for spatial_dimension in range(3):
                coupling = STRAIN_GRAD_BASES[spatial_dimension]
                for displacement_component in range(3):
                    matrix[
                        rows,
                        context.active_dim_s
                        + displacement_component : context.columns : 3,
                    ] -= (
                        coupling[residual_component, displacement_component]
                        * grad_u[:, :, spatial_dimension]
                    )

        for equilibrium_component in range(3):
            rows = slice(
                (6 + equilibrium_component) * batch_points,
                (7 + equilibrium_component) * batch_points,
            )
            for spatial_dimension in range(3):
                for deviatoric_component in range(5):
                    matrix[
                        rows,
                        deviatoric_component : context.deviatoric_dim_s : 5,
                    ] += (
                        context.equilibrium_deviatoric[
                            equilibrium_component,
                            spatial_dimension,
                            deviatoric_component,
                        ]
                        * grad_sigma[:, :, spatial_dimension]
                    )
                matrix[
                    rows,
                    context.deviatoric_dim_s : context.active_dim_s,
                ] += (
                    context.equilibrium_hydrostatic[
                        equilibrium_component,
                        spatial_dimension,
                    ]
                    * grad_sigma[:, 1:, spatial_dimension]
                )
            rhs[rows] = -sqrt_weights * body_force[:, equilibrium_component]

    if not augmented.flags.f_contiguous:
        raise RuntimeError("Augmented residual batch must be Fortran contiguous.")
    return DirectResidualBatch(start=start, stop=stop, augmented=augmented)


def iter_direct_residual_batches(
    benchmark: SharedBenchmarkData,
    feature_space: SharedFeatureSpace,
    context: DirectResidualContext,
    batch_size: int,
) -> Iterator[DirectResidualBatch]:
    """Stream augmented residual blocks without retaining earlier batches."""

    for start, stop in iter_point_batches(benchmark.x_int.shape[0], batch_size):
        yield assemble_direct_residual_batch(
            start,
            stop,
            benchmark,
            feature_space,
            context,
        )


def assemble_streaming_tsqr_design(
    cfg: LeastSquaresConfig,
    benchmark: SharedBenchmarkData,
    feature_space: SharedFeatureSpace,
    show_progress: bool = True,
) -> tuple[DirectResidualDesign, StreamingTSQRStats]:
    """Compress ``[A, b]`` with streaming Householder TSQR, never storing ``A``."""

    total_started = time.perf_counter()
    mean_started = time.perf_counter()
    mean_raw_sigma = accumulate_raw_sigma_mean(
        benchmark,
        feature_space,
        cfg.direct_batch_size,
    )
    mean_pass_time = time.perf_counter() - mean_started
    context = build_direct_residual_context(cfg, feature_space, mean_raw_sigma)

    augmented_columns = context.columns + 1
    reduced = np.zeros(
        (augmented_columns, augmented_columns),
        dtype=np.float64,
        order="F",
    )
    batch_count = math.ceil(benchmark.x_int.shape[0] / cfg.direct_batch_size)
    progress_stride = max(1, batch_count // 8)
    assembly_time = 0.0
    qr_time = 0.0
    tpqrt = None

    for batch_index, (start, stop) in enumerate(
        iter_point_batches(benchmark.x_int.shape[0], cfg.direct_batch_size),
        start=1,
    ):
        assembly_started = time.perf_counter()
        batch = assemble_direct_residual_batch(
            start,
            stop,
            benchmark,
            feature_space,
            context,
        )
        assembly_time += time.perf_counter() - assembly_started
        augmented = batch.augmented
        if tpqrt is None:
            tpqrt = get_lapack_funcs("tpqrt", (reduced, augmented))

        qr_started = time.perf_counter()
        updated_reduced, overwritten_batch, block_reflectors, info = tpqrt(
            0,
            min(cfg.direct_qr_block_size, augmented_columns),
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
        del batch, augmented, overwritten_batch, block_reflectors

        if show_progress and (
            batch_index == 1
            or batch_index == batch_count
            or batch_index % progress_stride == 0
        ):
            elapsed = time.perf_counter() - total_started
            eta = elapsed * (batch_count - batch_index) / batch_index
            print(
                f"  TSQR batch {batch_index}/{batch_count}: "
                f"points [{start}, {stop}), "
                f"elapsed={elapsed:.1f}s, eta={eta:.1f}s"
            )

    reduced_matrix_numpy = reduced[:, : context.columns]
    if not reduced_matrix_numpy.flags.f_contiguous:
        raise RuntimeError("Reduced direct residual matrix must be Fortran contiguous.")
    reduced_rhs_numpy = np.array(reduced[:, context.columns], copy=True)
    design = DirectResidualDesign(
        matrix=torch.from_numpy(reduced_matrix_numpy),
        rhs=torch.from_numpy(reduced_rhs_numpy),
        solved_dim_s=context.active_dim_s,
        stress_adapter=context.stress_adapter,
    )
    stats = StreamingTSQRStats(
        source_rows=9 * benchmark.x_int.shape[0],
        columns=context.columns,
        batch_count=batch_count,
        mean_pass_time=mean_pass_time,
        assembly_time=assembly_time,
        qr_time=qr_time,
        total_time=time.perf_counter() - total_started,
    )
    return design, stats


def assemble_direct_residual_design(
    cfg: LeastSquaresConfig,
    benchmark: SharedBenchmarkData,
    feature_space: SharedFeatureSpace,
) -> DirectResidualDesign:
    """Assemble the active residual matrix without dense raw-space intermediates."""

    x = benchmark.x_int
    sqrt_weights = torch.sqrt(benchmark.w_int)
    raw_sigma = eval_raw_scalar_basis(
        x,
        feature_space.a_s,
        feature_space.r_s,
        feature_space.gamma_s,
    )
    grad_sigma = eval_raw_scalar_basis_grads(
        x,
        feature_space.a_s,
        feature_space.r_s,
        feature_space.gamma_s,
    )
    _, grad_u = eval_active_displacement_basis_data(
        x,
        feature_space.a_u,
        feature_space.r_u,
        feature_space.gamma_u,
    )
    mean_raw_sigma = (benchmark.w_int.unsqueeze(1) * raw_sigma).sum(dim=0)
    stress_adapter = build_stress_basis_adapter(
        mean_raw_sigma,
        materialize_transform=False,
    )

    q_count = x.shape[0]
    np1_s = raw_sigma.shape[1]
    np1_u = grad_u.shape[1]
    deviatoric_dim_s = 5 * np1_s
    active_dim_s = stress_adapter.active_dim
    dim_u = 3 * np1_u
    # Allocate transposed storage so the NumPy view is Fortran contiguous.
    # DGELSD can then overwrite this single copy in-place.
    matrix = torch.zeros(
        active_dim_s + dim_u,
        9 * q_count,
        dtype=DTYPE,
        device=DEVICE,
    ).T
    rhs = torch.zeros(9 * q_count, dtype=DTYPE, device=DEVICE)

    mu, lam = compute_lame_constants(cfg.E, cfg.nu)
    compliance = build_compliance_matrix(mu, lam)
    weighted_sigma = sqrt_weights.unsqueeze(1) * raw_sigma
    hydrostatic_basis = raw_sigma[:, 1:] - raw_sigma[:, :1] * (
        mean_raw_sigma[1:] / mean_raw_sigma[0]
    ).unsqueeze(0)
    weighted_hydrostatic = sqrt_weights.unsqueeze(1) * hydrostatic_basis
    for residual_component in range(6):
        rows = slice(residual_component * q_count, (residual_component + 1) * q_count)
        deviatoric_coefficients = (
            compliance[residual_component] @ DEVIATORIC_STRESS_BASES
        )
        for deviatoric_component in range(5):
            matrix[rows, deviatoric_component:deviatoric_dim_s:5] = (
                deviatoric_coefficients[deviatoric_component] * weighted_sigma
            )
        matrix[rows, deviatoric_dim_s:active_dim_s] = (
            torch.dot(compliance[residual_component], HYDROSTATIC_STRESS_BASIS)
            * weighted_hydrostatic
        )
        for spatial_dimension in range(3):
            coupling = STRAIN_GRAD_BASES[spatial_dimension]
            for displacement_component in range(3):
                matrix[
                    rows,
                    active_dim_s + displacement_component : active_dim_s + dim_u : 3,
                ] -= (
                    coupling[residual_component, displacement_component]
                    * sqrt_weights.unsqueeze(1)
                    * grad_u[:, :, spatial_dimension]
                )

    for equilibrium_component in range(3):
        rows = slice(
            (6 + equilibrium_component) * q_count,
            (7 + equilibrium_component) * q_count,
        )
        for spatial_dimension in range(3):
            coupling = STRAIN_GRAD_BASES[spatial_dimension][
                :, equilibrium_component
            ]
            deviatoric_coefficients = coupling @ DEVIATORIC_STRESS_BASES
            weighted_grad_sigma = (
                sqrt_weights.unsqueeze(1) * grad_sigma[:, :, spatial_dimension]
            )
            for deviatoric_component in range(5):
                matrix[rows, deviatoric_component:deviatoric_dim_s:5] += (
                    deviatoric_coefficients[deviatoric_component]
                    * weighted_grad_sigma
                )
            # The zero-mean correction only changes the strictly constant
            # feature in column zero, whose gradient is exactly zero.
            matrix[rows, deviatoric_dim_s:active_dim_s] += (
                torch.dot(coupling, HYDROSTATIC_STRESS_BASIS)
                * sqrt_weights.unsqueeze(1)
                * grad_sigma[:, 1:, spatial_dimension]
            )
        rhs[rows] = -sqrt_weights * benchmark.f_int[:, equilibrium_component]

    return DirectResidualDesign(matrix, rhs, active_dim_s, stress_adapter)


def solve_direct_residual(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    rcond: float,
) -> tuple[torch.Tensor, float, int, float]:
    """Column-scale and solve in-place with DGELSD, destroying ``matrix``."""

    column_norms = torch.linalg.vector_norm(matrix, dim=0)
    floor = torch.finfo(matrix.dtype).eps * column_norms.max()
    safe_norms = column_norms.clamp_min(floor)
    matrix.div_(safe_norms.unsqueeze(0))

    matrix_numpy = matrix.numpy()
    if not matrix_numpy.flags.f_contiguous:
        raise ValueError("Direct residual matrix must be Fortran contiguous.")
    m, n = matrix_numpy.shape
    rhs_numpy = np.zeros((max(m, n), 1), dtype=matrix_numpy.dtype, order="F")
    rhs_numpy[:m, 0] = rhs.numpy()
    gelsd, gelsd_lwork = get_lapack_funcs(
        ("gelsd", "gelsd_lwork"),
        (matrix_numpy, rhs_numpy),
    )
    work_size, iwork_size, info = gelsd_lwork(m, n, 1, rcond)
    if info != 0:
        raise RuntimeError(f"DGELSD workspace query failed with info={info}.")

    t0 = time.perf_counter()
    solution_buffer, singular_values, rank, info = gelsd(
        matrix_numpy,
        rhs_numpy,
        int(work_size),
        int(iwork_size),
        rcond,
        overwrite_a=True,
        overwrite_b=True,
    )
    wall_time = time.perf_counter() - t0
    if info > 0:
        raise RuntimeError("DGELSD did not converge.")
    if info < 0:
        raise ValueError(f"DGELSD received an invalid argument at position {-info}.")

    scaled_solution = np.array(solution_buffer[:n, 0], copy=True)
    solution = torch.from_numpy(scaled_solution).to(dtype=DTYPE) / safe_norms
    condition_estimate = (
        float(singular_values[0] / singular_values[rank - 1])
        if rank > 0
        else float("inf")
    )
    return solution, wall_time, int(rank), condition_estimate


def split_solution(z: torch.Tensor, solved_dim_s: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the solver coefficient vector into stress and displacement parts."""

    return z[:solved_dim_s], z[solved_dim_s:]


def lift_active_stress_coefficients(
    active_sigma_coeffs: torch.Tensor,
    stress_adapter: StressBasisAdapter,
) -> torch.Tensor:
    """Lift active stress coefficients back to the raw stress basis."""

    if stress_adapter.transform is not None:
        return stress_adapter.transform @ active_sigma_coeffs

    np1_s = stress_adapter.mean_raw_sigma.numel()
    deviatoric_dim_s = 5 * np1_s
    if active_sigma_coeffs.numel() != stress_adapter.active_dim:
        raise ValueError("Active stress coefficient dimension does not match its adapter.")

    deviatoric_coefficients = active_sigma_coeffs[:deviatoric_dim_s].reshape(
        np1_s,
        5,
    )
    raw_coefficients = deviatoric_coefficients @ DEVIATORIC_STRESS_BASES.T

    active_hydrostatic = active_sigma_coeffs[deviatoric_dim_s:]
    hydrostatic_coefficients = torch.empty(
        np1_s,
        dtype=active_sigma_coeffs.dtype,
        device=active_sigma_coeffs.device,
    )
    hydrostatic_coefficients[0] = -torch.dot(
        stress_adapter.mean_raw_sigma[1:] / stress_adapter.mean_raw_sigma[0],
        active_hydrostatic,
    )
    hydrostatic_coefficients[1:] = active_hydrostatic
    raw_coefficients += (
        hydrostatic_coefficients.unsqueeze(1)
        * HYDROSTATIC_STRESS_BASIS.unsqueeze(0)
    )
    return raw_coefficients.reshape(-1)


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

    displacement_blocks = displacement_coeffs.reshape(-1, 3)
    stress_blocks = sigma_coeffs.reshape(-1, 6)

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

    def build_result(
        name: str,
        z: torch.Tensor,
        wall_time: float,
        rank: int,
        condition_estimate: float,
    ) -> AlgorithmResult:
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
        evaluated = evaluate_feature_result(
            name,
            wall_time,
            sigma_coeffs,
            displacement_coeffs,
            data.eval_data,
        )
        return AlgorithmResult(
            name=evaluated.name,
            u_l2_error=evaluated.u_l2_error,
            sigma_l2_error=evaluated.sigma_l2_error,
            wall_time=evaluated.wall_time,
            rank=rank,
            columns=data.matrix.shape[1],
            condition_estimate=condition_estimate,
        )

    if algorithm_id != "direct":
        raise ValueError(f"Unsupported algorithm: {algorithm_id}")
    print(
        f"Running LS ({data.direct_solver} + GELSD) on source system "
        f"({data.source_rows}, {data.matrix.shape[1]})..."
    )
    solve_started = time.perf_counter()
    z, gelsd_time, rank, condition_estimate = solve_direct_residual(
        data.matrix,
        data.rhs,
        cfg.direct_rcond,
    )
    reduced_solve_time = time.perf_counter() - solve_started
    wall_time = data.preparation_time + reduced_solve_time
    print(
        f"    timings: preparation={data.preparation_time:.2f}s, "
        f"reduced solve={reduced_solve_time:.2f}s, "
        f"DGELSD={gelsd_time:.2f}s, total={wall_time:.2f}s"
    )
    result = build_result(
        "LS",
        z,
        wall_time,
        rank,
        condition_estimate,
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
    output_dir = resolve_output_dir(cfg.manufactured_solution)

    print(f"Device: {DEVICE}")
    print(f"Output: {output_dir}")
    print(
        f"Config: N_s={cfg.N_s}, N_u={cfg.N_u}, "
        f"Q_train={cfg.Q_train}, Q_test={cfg.Q_test}, "
        f"gamma_s={cfg.gamma_s}, gamma_u={cfg.gamma_u}, "
        f"direct_rcond={cfg.direct_rcond:.2e}, "
        f"direct_solver={cfg.direct_solver}, "
        f"direct_batch_size={cfg.direct_batch_size}, "
        f"direct_qr_block_size={cfg.direct_qr_block_size}, "
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
    if feature_space.a_s.shape[0] != cfg.N_s or feature_space.a_u.shape[0] != cfg.N_u:
        raise ValueError("SharedFeatureSpace dimensions do not match LeastSquaresConfig.")

    source_rows = 9 * benchmark.x_int.shape[0]
    expected_columns = 6 * cfg.N_s + 3 * cfg.N_u + 8
    dense_matrix_gib = (
        source_rows
        * expected_columns
        * torch.tensor([], dtype=DTYPE).element_size()
        / 2**30
    )
    print(
        f"Direct system estimate: ({source_rows}, {expected_columns}), "
        f"rows/columns={source_rows / expected_columns:.2f}, "
        f"dense A={dense_matrix_gib:.2f} GiB"
    )
    if cfg.direct_solver == "dense" and dense_matrix_gib > 4.0:
        warnings.warn(
            "The selected dense direct backend may exceed workstation memory; "
            "use direct_solver='streaming_tsqr' for this configuration.",
            RuntimeWarning,
        )
    if cfg.direct_solver == "dense":
        print("Assembling dense direct weighted residual matrix...")
        t0 = time.perf_counter()
        direct_design = assemble_direct_residual_design(
            cfg,
            benchmark,
            feature_space,
        )
        preparation_time = time.perf_counter() - t0
        print(
            f"Residual shapes: A={tuple(direct_design.matrix.shape)}, "
            f"b={tuple(direct_design.rhs.shape)}, "
            f"assembly={preparation_time:.2f}s"
        )
    else:
        print("Compressing direct residuals with streaming TSQR...")
        direct_design, streaming_stats = assemble_streaming_tsqr_design(
            cfg,
            benchmark,
            feature_space,
        )
        preparation_time = streaming_stats.total_time
        print(
            f"Residual shapes: source A=({streaming_stats.source_rows}, "
            f"{streaming_stats.columns}), "
            f"reduced A={tuple(direct_design.matrix.shape)}, "
            f"b={tuple(direct_design.rhs.shape)}"
        )
        print(
            f"TSQR timings: mean pass={streaming_stats.mean_pass_time:.2f}s, "
            f"batch assembly={streaming_stats.assembly_time:.2f}s, "
            f"QR updates={streaming_stats.qr_time:.2f}s, "
            f"compression total={streaming_stats.total_time:.2f}s"
        )
    clear_cuda_cache()
    print(
        "trace constraint applied: "
        f"stress dof {direct_design.stress_adapter.raw_dim} "
        f"-> {direct_design.stress_adapter.active_dim}"
    )

    experiment_data = LeastSquaresExperimentData(
        matrix=direct_design.matrix,
        rhs=direct_design.rhs,
        solved_dim_s=direct_design.solved_dim_s,
        stress_adapter=direct_design.stress_adapter,
        eval_data=build_feature_evaluation_data(
            benchmark,
            feature_space,
        ),
        source_rows=source_rows,
        preparation_time=preparation_time,
        direct_solver=cfg.direct_solver,
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
