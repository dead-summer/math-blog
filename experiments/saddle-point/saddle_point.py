"""
3D Hellinger-Reissner saddle-point numerical experiments.

This script supports:
1. Split stress/displacement random-feature spaces
2. Optional displacement envelope zeta
3. Optional boundary penalty block in the KKT system
4. Projection, eigh, lstsq, GDA, Uzawa, and Arrow-Hurwicz solvers
"""

import math
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "public" / "images" / "saddle-point"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Device, seeds, and dtype
# ---------------------------------------------------------------------------
BASE_SEED = 42
STRESS_SEED = BASE_SEED
DISP_SEED = BASE_SEED + 1000
DTYPE = torch.float64

FeatureEvaluator = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, float],
    torch.Tensor,
]
FeatureGradientEvaluator = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, float],
    torch.Tensor,
]


def detect_device() -> torch.device:
    """Prefer CUDA when available, otherwise fall back to CPU quietly."""

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


# ---------------------------------------------------------------------------
# Algorithm metadata and styles
# ---------------------------------------------------------------------------
DEFAULT_ALGORITHM_IDS = (
    "projection",
    "eigh",
    "lstsq",
    "gda",
    "uzawa",
    "arrow_hurwicz",
)
SUMMARY_DISPLAY_ORDER = (
    "Projection",
    "Eigh",
    "Lstsq",
    "GDA",
    "Uzawa",
    "Arrow-Hurwicz",
)
ITERATIVE_DISPLAY_ORDER = (
    "GDA",
    "Uzawa",
    "Arrow-Hurwicz",
)
KKT_RESIDUAL_ALGORITHM_IDS = frozenset({"eigh", "lstsq"})

ALGO_STYLE = {
    "Projection": {"color": "#9B2226", "marker": "P", "linestyle": ":"},
    "Eigh": {"color": "#264653", "marker": "s", "linestyle": "--"},
    "Lstsq": {"color": "#6D597A", "marker": "X", "linestyle": "-."},
    "GDA": {"color": "#0077B6", "marker": "o", "linestyle": "-"},
    "Uzawa": {"color": "#E76F51", "marker": "^", "linestyle": "--"},
    "Arrow-Hurwicz": {"color": "#2A9D8F", "marker": "D", "linestyle": "-."},
}


@dataclass(frozen=True)
class AlgorithmSpec:
    """Metadata for one selectable algorithm."""

    display_name: str
    backend: str | None = None
    is_iterative: bool = False


ALGORITHM_SPECS: dict[str, AlgorithmSpec] = {
    "projection": AlgorithmSpec(display_name="Projection"),
    "eigh": AlgorithmSpec(
        display_name="Eigh",
        backend="torch.linalg.eigh",
    ),
    "lstsq": AlgorithmSpec(
        display_name="Lstsq",
        backend="torch.linalg.lstsq",
    ),
    "gda": AlgorithmSpec(
        display_name="GDA",
        backend="Adam",
        is_iterative=True,
    ),
    "uzawa": AlgorithmSpec(
        display_name="Uzawa",
        backend="Cholesky",
        is_iterative=True,
    ),
    "arrow_hurwicz": AlgorithmSpec(
        display_name="Arrow-Hurwicz",
        backend="Jacobi",
        is_iterative=True,
    ),
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Experiment configuration."""

    E: float = 1.0
    nu: float = 0.3

    gamma_s: float = 2.0
    gamma_u: float = 2.0
    M_s: int = 300
    M_u: int = 300

    Q_int: int = (2 ** 6) ** 3
    Q_bc: int = 6 * (2 ** 5) ** 2
    Q_test: int = (2 ** 5) ** 3
    sampling_method: str = "sobol"

    K_max: int = 20000
    rho: float = 1.0e-6
    eta_gda: float = 2.0e-2
    beta_adam: tuple[float, float] = (0.9, 0.98)
    eta_u_uzawa: float | None = None
    eta_s_ah: float | None = None
    eta_u_ah: float | None = None
    eval_every: int = 50

    use_zeta: bool = True
    use_penalty: bool = False
    lambda_bc: float = 1.0
    eigh_rtol: float = 1.0e-12
    body_force_batch_size: int = 5000
    assembly_batch_size: int = 5000
    algorithms_to_run: list[str] = field(
        default_factory=lambda: [
            "projection",
            "eigh",
            "lstsq",
            "gda",
            "uzawa",
            "arrow_hurwicz",
        ]
    )


# ---------------------------------------------------------------------------
# Material parameters
# ---------------------------------------------------------------------------
def compute_lame_constants(E: float, nu: float) -> tuple[float, float]:
    """Return (mu, lambda) from Young's modulus and Poisson's ratio."""

    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def build_compliance_matrix(E: float, nu: float) -> torch.Tensor:
    """Build the 6x6 compliance matrix in Voigt form."""

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


# ---------------------------------------------------------------------------
# Manufactured solution and body force
# ---------------------------------------------------------------------------
def eval_exact_displacement(x: torch.Tensor) -> torch.Tensor:
    """Evaluate the manufactured displacement field."""

    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    pi = math.pi
    u1 = torch.sin(pi * x1) * torch.sin(pi * x2) * torch.sin(pi * x3)
    u2 = torch.sin(2.0 * pi * x1) * torch.sin(pi * x2) * torch.sin(pi * x3)
    u3 = torch.sin(pi * x1) * torch.sin(2.0 * pi * x2) * torch.sin(pi * x3)
    return torch.stack([u1, u2, u3], dim=1)


def compute_stress_voigt(x: torch.Tensor, mu: float, lam: float) -> torch.Tensor:
    """Evaluate the exact stress in Voigt order (11,22,33,12,23,13)."""

    x_ad = x.detach().requires_grad_(True)
    u = eval_exact_displacement(x_ad)

    n_points = x.shape[0]
    grad_u = torch.zeros(n_points, 3, 3, dtype=DTYPE, device=DEVICE)
    for comp in range(3):
        grad_comp = torch.autograd.grad(
            u[:, comp].sum(),
            x_ad,
            create_graph=False,
            retain_graph=(comp < 2),
        )[0]
        grad_u[:, comp, :] = grad_comp

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
    batch_size: int = 5000,
) -> torch.Tensor:
    """Compute body force f = -div(sigma(u_ex)) by autograd."""

    n_points = x.shape[0]
    f_all = torch.zeros(n_points, 3, dtype=DTYPE, device=DEVICE)

    for start in range(0, n_points, batch_size):
        end = min(start + batch_size, n_points)
        xb = x[start:end].detach().requires_grad_(True)
        u = eval_exact_displacement(xb)

        grad_u_list = []
        for comp in range(3):
            grad_u_list.append(
                torch.autograd.grad(u[:, comp].sum(), xb, create_graph=True)[0]
            )
        grad_u = torch.stack(grad_u_list, dim=1)

        eps = 0.5 * (grad_u + grad_u.transpose(1, 2))
        tr_eps = eps[:, 0, 0] + eps[:, 1, 1] + eps[:, 2, 2]

        sigma = 2.0 * mu * eps
        for comp in range(3):
            sigma[:, comp, comp] += lam * tr_eps

        for comp_i in range(3):
            div_sigma_i = torch.zeros(end - start, dtype=DTYPE, device=DEVICE)
            for comp_j in range(3):
                grad_sigma = torch.autograd.grad(
                    sigma[:, comp_i, comp_j].sum(),
                    xb,
                    create_graph=False,
                    retain_graph=not (comp_i == 2 and comp_j == 2),
                )[0]
                div_sigma_i += grad_sigma[:, comp_j]
            f_all[start:end, comp_i] = -div_sigma_i.detach()

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


# ---------------------------------------------------------------------------
# Sampling and feature spaces
# ---------------------------------------------------------------------------
def sample_points(
    n_points: int,
    method: str = "mc",
    dim: int = 3,
    seed: int = 0,
) -> torch.Tensor:
    """Sample points from [0, 1]^dim."""

    if method == "sobol":
        engine = torch.quasirandom.SobolEngine(
            dimension=dim,
            scramble=True,
            seed=seed,
        )
        return engine.draw(n_points).to(dtype=DTYPE, device=DEVICE)

    torch.manual_seed(seed)
    return torch.rand(n_points, dim, dtype=DTYPE, device=DEVICE)


def sample_boundary_points(
    n_points: int,
    method: str = "mc",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample points on the six faces of the unit cube."""

    counts = [n_points // 6] * 6
    for face_id in range(n_points % 6):
        counts[face_id] += 1

    point_parts: list[torch.Tensor] = []
    weight_parts: list[torch.Tensor] = []

    for face_id, count in enumerate(counts):
        uv = sample_points(count, method=method, dim=2, seed=seed + face_id)
        face_points = torch.zeros(count, 3, dtype=DTYPE, device=DEVICE)

        if face_id == 0:
            face_points[:, 0] = 0.0
            face_points[:, 1:] = uv
        elif face_id == 1:
            face_points[:, 0] = 1.0
            face_points[:, 1:] = uv
        elif face_id == 2:
            face_points[:, 1] = 0.0
            face_points[:, 0] = uv[:, 0]
            face_points[:, 2] = uv[:, 1]
        elif face_id == 3:
            face_points[:, 1] = 1.0
            face_points[:, 0] = uv[:, 0]
            face_points[:, 2] = uv[:, 1]
        elif face_id == 4:
            face_points[:, 2] = 0.0
            face_points[:, :2] = uv
        else:
            face_points[:, 2] = 1.0
            face_points[:, :2] = uv

        point_parts.append(face_points)
        weight_parts.append(
            torch.full((count,), 1.0 / count, dtype=DTYPE, device=DEVICE)
        )

    return torch.cat(point_parts, dim=0), torch.cat(weight_parts, dim=0)


def generate_features(M: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random feature parameters."""

    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    raw = torch.randn(M, 3, generator=rng, dtype=DTYPE)
    norms = raw.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
    a = (raw / norms).to(DEVICE)
    r = torch.rand(M, generator=rng, dtype=DTYPE).to(DEVICE)
    return a, r


def eval_features(
    x: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Evaluate xi_0 = 1 and xi_m = tanh(gamma * (a_m^T x + r_m))."""

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
    """Evaluate gradients of all features."""

    pre = x @ a.T + r.unsqueeze(0)
    dtanh = 1.0 - torch.tanh(gamma * pre) ** 2
    grad_xi = gamma * dtanh.unsqueeze(2) * a.unsqueeze(0)
    zeros = torch.zeros(x.shape[0], 1, 3, dtype=DTYPE, device=DEVICE)
    return torch.cat([zeros, grad_xi], dim=1)


def activate_displacement_features(
    xi_u: torch.Tensor,
    zeta: torch.Tensor,
) -> torch.Tensor:
    """Apply the displacement envelope to one feature matrix."""

    return zeta.unsqueeze(1) * xi_u


# ---------------------------------------------------------------------------
# System assembly
# ---------------------------------------------------------------------------
def assemble_stress_matrix(
    xi_s: torch.Tensor,
    compliance_voigt: torch.Tensor,
) -> torch.Tensor:
    """Assemble the stress block A using the stress feature space only."""

    n_points = xi_s.shape[0]
    gram_s = (1.0 / n_points) * (xi_s.T @ xi_s)
    return assemble_stress_matrix_from_gram(gram_s, compliance_voigt)


def assemble_stress_matrix_from_gram(
    gram_s: torch.Tensor,
    compliance_voigt: torch.Tensor,
) -> torch.Tensor:
    """Assemble the stress block A from the stress Gram matrix."""

    mp1_s = gram_s.shape[0]
    dim_s = 6 * mp1_s
    A = torch.zeros(dim_s, dim_s, dtype=DTYPE, device=DEVICE)
    for alpha in range(6):
        for beta in range(6):
            A[alpha::6, beta::6] = compliance_voigt[alpha, beta] * gram_s
    return A


def assemble_coupling_matrix(
    xi_u_active: torch.Tensor,
    grad_xi_s: torch.Tensor,
) -> torch.Tensor:
    """Assemble the split-space coupling matrix B."""

    n_points = xi_u_active.shape[0]
    cross_u_grad_s = []
    for dim_k in range(3):
        cross_u_grad_s.append((1.0 / n_points) * (xi_u_active.T @ grad_xi_s[:, :, dim_k]))
    return assemble_coupling_matrix_from_cross_moments(cross_u_grad_s)

def assemble_coupling_matrix_from_cross_moments(
    cross_u_grad_s: list[torch.Tensor],
) -> torch.Tensor:
    """Assemble the split-space coupling matrix B from interior cross moments."""

    mp1_u = cross_u_grad_s[0].shape[0]
    mp1_s = cross_u_grad_s[0].shape[1]
    D = [cross_moment.T for cross_moment in cross_u_grad_s]

    dim_s = 6 * mp1_s
    dim_u = 3 * mp1_u
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


def assemble_rhs_vector(
    xi_u_active: torch.Tensor,
    f_vals: torch.Tensor,
) -> torch.Tensor:
    """Assemble the displacement right-hand side F."""

    n_points = xi_u_active.shape[0]
    force_moment = (1.0 / n_points) * (xi_u_active.T @ f_vals)
    return assemble_rhs_vector_from_moment(force_moment)


def assemble_rhs_vector_from_moment(
    force_moment: torch.Tensor,
) -> torch.Tensor:
    """Assemble the displacement right-hand side F from interior moments."""

    mp1_u = force_moment.shape[0]
    F = torch.zeros(3 * mp1_u, dtype=DTYPE, device=DEVICE)
    for comp in range(3):
        F[comp::3] = force_moment[:, comp]
    return F


def assemble_boundary_matrix(
    xi_u_active_bc: torch.Tensor,
    w_bc: torch.Tensor,
) -> torch.Tensor:
    """Assemble the boundary penalty block C."""

    gram_bc = xi_u_active_bc.T @ (w_bc.unsqueeze(1) * xi_u_active_bc)
    return assemble_boundary_matrix_from_gram(gram_bc)


def assemble_boundary_matrix_from_gram(
    gram_bc: torch.Tensor,
) -> torch.Tensor:
    """Assemble the boundary penalty block C from a boundary Gram matrix."""

    mp1_u = gram_bc.shape[0]
    dim_u = 3 * mp1_u
    C = torch.zeros(dim_u, dim_u, dtype=DTYPE, device=DEVICE)
    for comp in range(3):
        C[comp::3, comp::3] = gram_bc
    return C


def accumulate_interior_moments(
    x_train: torch.Tensor,
    f_vals: torch.Tensor,
    a_s: torch.Tensor,
    r_s: torch.Tensor,
    gamma_s: float,
    a_u: torch.Tensor,
    r_u: torch.Tensor,
    gamma_u: float,
    zeta: torch.Tensor,
    batch_size: int,
    *,
    features_fn: FeatureEvaluator = eval_features,
    grads_fn: FeatureGradientEvaluator = eval_feature_grads,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Accumulate the interior moments needed by the saddle-point system."""

    mp1_s = a_s.shape[0] + 1
    mp1_u = a_u.shape[0] + 1
    weight = 1.0 / x_train.shape[0]

    gram_s = torch.zeros(mp1_s, mp1_s, dtype=DTYPE, device=DEVICE)
    cross_u_grad_s = [
        torch.zeros(mp1_u, mp1_s, dtype=DTYPE, device=DEVICE) for _ in range(3)
    ]
    force_moment = torch.zeros(mp1_u, 3, dtype=DTYPE, device=DEVICE)

    with torch.no_grad():
        for start in range(0, x_train.shape[0], batch_size):
            end = min(start + batch_size, x_train.shape[0])
            xb = x_train[start:end]
            fb = f_vals[start:end]
            zeta_batch = zeta[start:end]

            xi_s_batch = features_fn(xb, a_s, r_s, gamma_s)
            grad_xi_s_batch = grads_fn(xb, a_s, r_s, gamma_s)
            xi_u_batch = features_fn(xb, a_u, r_u, gamma_u)
            xi_u_active_batch = activate_displacement_features(xi_u_batch, zeta_batch)

            gram_s += weight * (xi_s_batch.T @ xi_s_batch)
            force_moment += weight * (xi_u_active_batch.T @ fb)
            for dim_k in range(3):
                cross_u_grad_s[dim_k] += weight * (
                    xi_u_active_batch.T @ grad_xi_s_batch[:, :, dim_k]
                )

    return gram_s, cross_u_grad_s, force_moment


def accumulate_boundary_gram(
    x_bc: torch.Tensor,
    w_bc: torch.Tensor,
    a_u: torch.Tensor,
    r_u: torch.Tensor,
    gamma_u: float,
    zeta_bc: torch.Tensor,
    batch_size: int,
    *,
    features_fn: FeatureEvaluator = eval_features,
) -> torch.Tensor:
    """Accumulate the weighted boundary Gram matrix for displacement features."""

    mp1_u = a_u.shape[0] + 1
    gram_bc = torch.zeros(mp1_u, mp1_u, dtype=DTYPE, device=DEVICE)

    with torch.no_grad():
        for start in range(0, x_bc.shape[0], batch_size):
            end = min(start + batch_size, x_bc.shape[0])
            xb = x_bc[start:end]
            wb = w_bc[start:end]
            zeta_batch = zeta_bc[start:end]

            xi_u_batch = features_fn(xb, a_u, r_u, gamma_u)
            xi_u_active_batch = activate_displacement_features(xi_u_batch, zeta_batch)
            gram_bc += xi_u_active_batch.T @ (wb.unsqueeze(1) * xi_u_active_batch)

    return gram_bc


def assemble_system_from_moments(
    gram_s: torch.Tensor,
    cross_u_grad_s: list[torch.Tensor],
    force_moment: torch.Tensor,
    compliance_voigt: torch.Tensor,
    *,
    gram_bc: torch.Tensor | None = None,
    lambda_bc: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble A, B, C, F from accumulated interior and boundary moments."""

    A = assemble_stress_matrix_from_gram(gram_s, compliance_voigt)
    B = assemble_coupling_matrix_from_cross_moments(cross_u_grad_s)
    F = assemble_rhs_vector_from_moment(force_moment)
    C = torch.zeros(B.shape[1], B.shape[1], dtype=DTYPE, device=DEVICE)
    if gram_bc is not None and lambda_bc > 0.0:
        C = lambda_bc * assemble_boundary_matrix_from_gram(gram_bc)
    return A, B, C, F


def assemble_system(
    xi_s: torch.Tensor,
    grad_xi_s: torch.Tensor,
    xi_u: torch.Tensor,
    compliance_voigt: torch.Tensor,
    f_vals: torch.Tensor,
    zeta: torch.Tensor,
    xi_u_bc: torch.Tensor | None = None,
    w_bc: torch.Tensor | None = None,
    zeta_bc: torch.Tensor | None = None,
    lambda_bc: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble A, B, C, F for split spaces with an optional penalty block."""

    xi_u_active = activate_displacement_features(xi_u, zeta)
    n_points = xi_s.shape[0]
    gram_s = (1.0 / n_points) * (xi_s.T @ xi_s)
    cross_u_grad_s = []
    for dim_k in range(3):
        cross_u_grad_s.append((1.0 / n_points) * (xi_u_active.T @ grad_xi_s[:, :, dim_k]))
    force_moment = (1.0 / n_points) * (xi_u_active.T @ f_vals)

    gram_bc = None
    if xi_u_bc is not None and w_bc is not None and lambda_bc > 0.0:
        xi_u_bc_active = xi_u_bc
        if zeta_bc is not None:
            xi_u_bc_active = activate_displacement_features(xi_u_bc_active, zeta_bc)
        gram_bc = xi_u_bc_active.T @ (w_bc.unsqueeze(1) * xi_u_bc_active)

    return assemble_system_from_moments(
        gram_s,
        cross_u_grad_s,
        force_moment,
        compliance_voigt,
        gram_bc=gram_bc,
        lambda_bc=lambda_bc,
    )


def assemble_system_in_batches(
    x_train: torch.Tensor,
    f_vals: torch.Tensor,
    a_s: torch.Tensor,
    r_s: torch.Tensor,
    gamma_s: float,
    a_u: torch.Tensor,
    r_u: torch.Tensor,
    gamma_u: float,
    compliance_voigt: torch.Tensor,
    zeta: torch.Tensor,
    batch_size: int,
    *,
    x_bc: torch.Tensor | None = None,
    w_bc: torch.Tensor | None = None,
    zeta_bc: torch.Tensor | None = None,
    lambda_bc: float = 0.0,
    features_fn: FeatureEvaluator = eval_features,
    grads_fn: FeatureGradientEvaluator = eval_feature_grads,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble A, B, C, F by accumulating moments in batches."""

    gram_s, cross_u_grad_s, force_moment = accumulate_interior_moments(
        x_train,
        f_vals,
        a_s,
        r_s,
        gamma_s,
        a_u,
        r_u,
        gamma_u,
        zeta,
        batch_size,
        features_fn=features_fn,
        grads_fn=grads_fn,
    )

    gram_bc = None
    if lambda_bc > 0.0:
        if x_bc is None or w_bc is None or zeta_bc is None:
            raise ValueError("Boundary data is required when lambda_bc > 0.")
        gram_bc = accumulate_boundary_gram(
            x_bc,
            w_bc,
            a_u,
            r_u,
            gamma_u,
            zeta_bc,
            batch_size,
            features_fn=features_fn,
        )

    return assemble_system_from_moments(
        gram_s,
        cross_u_grad_s,
        force_moment,
        compliance_voigt,
        gram_bc=gram_bc,
        lambda_bc=lambda_bc,
    )


def build_kkt_system(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    F: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the KKT matrix and right-hand side."""

    dim_s = A.shape[0]
    dim_u = B.shape[1]
    dim_total = dim_s + dim_u

    K = torch.zeros(dim_total, dim_total, dtype=DTYPE, device=DEVICE)
    K[:dim_s, :dim_s] = A
    K[:dim_s, dim_s:] = B
    K[dim_s:, :dim_s] = B.T
    K[dim_s:, dim_s:] = C

    rhs = torch.zeros(dim_total, dtype=DTYPE, device=DEVICE)
    rhs[dim_s:] = -F
    return K, rhs


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def estimate_schur_spectral_radius(
    A: torch.Tensor,
    B: torch.Tensor,
    rho: float = 1.0e-6,
) -> tuple[float, torch.Tensor]:
    """Estimate the spectral radius of S = B^T (A + rho I)^(-1) B."""

    dim_s = A.shape[0]
    dim_u = B.shape[1]
    A_reg = A + rho * torch.eye(dim_s, dtype=DTYPE, device=DEVICE)
    L = torch.linalg.cholesky(A_reg)

    v = torch.randn(dim_u, dtype=DTYPE, device=DEVICE)
    v = v / v.norm()

    lam = torch.tensor(0.0, dtype=DTYPE, device=DEVICE)
    for _ in range(100):
        w = B @ v
        z = torch.cholesky_solve(w.unsqueeze(1), L).squeeze(1)
        u_new = B.T @ z
        lam = v.dot(u_new)
        nrm = u_new.norm()
        if nrm <= 1.0e-15:
            break
        v = u_new / nrm

    return float(lam.item()), L


def estimate_jacobi_spectral_radius(
    A: torch.Tensor,
    rho: float = 1.0e-6,
) -> float:
    """Estimate the spectral radius of the Jacobi iteration matrix."""

    dim_s = A.shape[0]
    A_reg = A + rho * torch.eye(dim_s, dtype=DTYPE, device=DEVICE)
    d_inv = 1.0 / A_reg.diag()

    v = torch.randn(dim_s, dtype=DTYPE, device=DEVICE)
    v = v / v.norm()

    lam = torch.tensor(0.0, dtype=DTYPE, device=DEVICE)
    for _ in range(200):
        Rv = v - d_inv * (A_reg @ v)
        lam = v.dot(Rv)
        nrm = Rv.norm()
        if nrm <= 1.0e-15:
            break
        v = Rv / nrm

    return abs(float(lam.item()))


# ---------------------------------------------------------------------------
# Residuals, errors, and histories
# ---------------------------------------------------------------------------
def compute_residual_components(
    A: torch.Tensor,
    B: torch.Tensor,
    F: torch.Tensor,
    s: torch.Tensor,
    u: torch.Tensor,
    penalty_matrix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute constitutive residual and the tracked equilibrium residual."""

    r_c = A @ s + B @ u
    r_e = B.T @ s + F
    if penalty_matrix is not None:
        r_e = r_e + penalty_matrix @ u
    return r_c, r_e


def compute_residual_norms(
    A: torch.Tensor,
    B: torch.Tensor,
    F: torch.Tensor,
    s: torch.Tensor,
    u: torch.Tensor,
    penalty_matrix: torch.Tensor | None = None,
) -> tuple[float, float]:
    """Compute constitutive and tracked equilibrium residual norms."""

    r_c, r_e = compute_residual_components(
        A,
        B,
        F,
        s,
        u,
        penalty_matrix=penalty_matrix,
    )
    return r_c.norm().item(), r_e.norm().item()


def compute_l2_errors(
    xi_u_active_test: torch.Tensor,
    s: torch.Tensor,
    u: torch.Tensor,
    u_exact: torch.Tensor,
    sigma_exact: torch.Tensor,
    xi_s_test: torch.Tensor,
) -> tuple[float, float]:
    """Compute relative L2 errors for split displacement and stress spaces."""

    n_points = xi_u_active_test.shape[0]

    u_h = torch.zeros(n_points, 3, dtype=DTYPE, device=DEVICE)
    for comp in range(3):
        u_h[:, comp] = xi_u_active_test @ u[comp::3]

    sigma_h = torch.zeros(n_points, 6, dtype=DTYPE, device=DEVICE)
    for alpha in range(6):
        sigma_h[:, alpha] = xi_s_test @ s[alpha::6]

    w_frob = torch.tensor(
        [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        dtype=DTYPE,
        device=DEVICE,
    )

    u_err = torch.sqrt(((u_h - u_exact) ** 2).sum(1).mean())
    u_ref = torch.sqrt((u_exact**2).sum(1).mean())
    rel_u = (u_err / u_ref).item() if u_ref > 0 else float("inf")

    sigma_err = torch.sqrt((w_frob * (sigma_h - sigma_exact) ** 2).sum(1).mean())
    sigma_ref = torch.sqrt((w_frob * sigma_exact**2).sum(1).mean())
    rel_sigma = (sigma_err / sigma_ref).item() if sigma_ref > 0 else float("inf")
    return rel_u, rel_sigma


def evaluate_result(
    A: torch.Tensor,
    B: torch.Tensor,
    F: torch.Tensor,
    xi_u_active_test: torch.Tensor,
    xi_s_test: torch.Tensor,
    u_exact: torch.Tensor,
    sigma_exact: torch.Tensor,
    s: torch.Tensor,
    u: torch.Tensor,
    penalty_matrix: torch.Tensor | None = None,
) -> tuple[float, float, float, float]:
    """Evaluate residuals and relative L2 errors for one result."""

    r_c, r_e = compute_residual_norms(
        A,
        B,
        F,
        s,
        u,
        penalty_matrix=penalty_matrix,
    )
    rel_u, rel_sigma = compute_l2_errors(
        xi_u_active_test,
        s,
        u,
        u_exact,
        sigma_exact,
        xi_s_test,
    )
    return r_c, r_e, rel_u, rel_sigma


def record_history_entry(
    history: dict[str, list[float | int]],
    step: int,
    A: torch.Tensor,
    B: torch.Tensor,
    F: torch.Tensor,
    xi_u_active_test: torch.Tensor,
    xi_s_test: torch.Tensor,
    u_exact: torch.Tensor,
    sigma_exact: torch.Tensor,
    s: torch.Tensor,
    u: torch.Tensor,
    penalty_matrix: torch.Tensor | None = None,
) -> None:
    """Append one evaluation point into a history dictionary."""

    r_c, r_e, rel_u, rel_sigma = evaluate_result(
        A,
        B,
        F,
        xi_u_active_test,
        xi_s_test,
        u_exact,
        sigma_exact,
        s,
        u,
        penalty_matrix=penalty_matrix,
    )
    history["r_c"].append(r_c)
    history["r_e"].append(r_e)
    history["rel_u"].append(rel_u)
    history["rel_sigma"].append(rel_sigma)
    history["steps"].append(step)


def make_eval_callback(
    A: torch.Tensor,
    B: torch.Tensor,
    F: torch.Tensor,
    xi_u_active_test: torch.Tensor,
    xi_s_test: torch.Tensor,
    u_exact: torch.Tensor,
    sigma_exact: torch.Tensor,
    eval_every: int = 50,
) -> tuple[Callable[[int, torch.Tensor, torch.Tensor], None], dict[str, list[float | int]]]:
    """Create a callback that records tracked residuals and L2 errors."""

    history: dict[str, list[float | int]] = {
        "r_c": [],
        "r_e": [],
        "rel_u": [],
        "rel_sigma": [],
        "steps": [],
    }

    def callback(step: int, s: torch.Tensor, u: torch.Tensor) -> None:
        if step % eval_every == 0 or step <= 1:
            record_history_entry(
                history,
                step,
                A,
                B,
                F,
                xi_u_active_test,
                xi_s_test,
                u_exact,
                sigma_exact,
                s,
                u,
            )

    return callback, history


def build_single_step_history(
    r_c: float,
    r_e: float,
    rel_u: float,
    rel_sigma: float,
) -> dict[str, list[float | int]]:
    """Build a single-step history for direct methods and projection."""

    return {
        "r_c": [r_c],
        "r_e": [r_e],
        "rel_u": [rel_u],
        "rel_sigma": [rel_sigma],
        "steps": [0],
    }


def finalize_iterative_history(
    history: dict[str, list[float | int]],
    final_step: int,
    A: torch.Tensor,
    B: torch.Tensor,
    F: torch.Tensor,
    xi_u_active_test: torch.Tensor,
    xi_s_test: torch.Tensor,
    u_exact: torch.Tensor,
    sigma_exact: torch.Tensor,
    s: torch.Tensor,
    u: torch.Tensor,
) -> None:
    """Ensure the final iterate is present in history."""

    if history["steps"] and history["steps"][-1] == final_step:
        return

    record_history_entry(
        history,
        final_step,
        A,
        B,
        F,
        xi_u_active_test,
        xi_s_test,
        u_exact,
        sigma_exact,
        s,
        u,
    )


# ---------------------------------------------------------------------------
# Projection and direct solvers
# ---------------------------------------------------------------------------
def run_projection(
    xi_s_train: torch.Tensor,
    xi_u_active_train: torch.Tensor,
    u_exact_train: torch.Tensor,
    sigma_exact_train: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Fit exact displacement and stress independently."""

    mp1_s = xi_s_train.shape[1]
    mp1_u = xi_u_active_train.shape[1]
    s = torch.zeros(6 * mp1_s, dtype=DTYPE, device=DEVICE)
    u = torch.zeros(3 * mp1_u, dtype=DTYPE, device=DEVICE)

    t0 = time.perf_counter()
    for comp in range(3):
        u[comp::3] = torch.linalg.lstsq(
            xi_u_active_train,
            u_exact_train[:, comp],
        ).solution
    for alpha in range(6):
        s[alpha::6] = torch.linalg.lstsq(
            xi_s_train,
            sigma_exact_train[:, alpha],
        ).solution
    wall_time = time.perf_counter() - t0

    return s, u, wall_time


def run_lstsq(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    F: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Solve the KKT system in the least-squares sense."""

    dim_s = A.shape[0]
    K, rhs = build_kkt_system(A, B, C, F)

    t0 = time.perf_counter()
    try:
        sol = torch.linalg.lstsq(K, rhs.unsqueeze(1)).solution.squeeze(1)
        if not torch.isfinite(sol).all():
            raise RuntimeError("non-finite solution")
    except (RuntimeError, torch.linalg.LinAlgError) as exc:
        sol = torch.full((K.shape[0],), float("nan"), dtype=DTYPE, device=DEVICE)
        print(f"    Warning: torch.linalg.lstsq failed with {type(exc).__name__}")

    wall_time = time.perf_counter() - t0
    return sol[:dim_s], sol[dim_s:], wall_time


def run_eigh(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    F: torch.Tensor,
    rtol: float = 1.0e-12,
    atol: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Solve the symmetric KKT system with a truncated eigen decomposition."""

    dim_s = A.shape[0]
    K, rhs = build_kkt_system(A, B, C, F)

    t0 = time.perf_counter()
    try:
        eigvals, eigvecs = torch.linalg.eigh(K)
        eig_abs_max = eigvals.abs().max()
        thresh = rtol * eig_abs_max
        if atol is not None:
            thresh = torch.maximum(
                thresh,
                torch.tensor(atol, dtype=DTYPE, device=DEVICE),
            )

        keep = eigvals.abs() > thresh
        if not keep.any():
            raise RuntimeError("all eigenvalues were truncated")

        coeffs = eigvecs[:, keep].T @ rhs
        coeffs = coeffs / eigvals[keep]
        sol = eigvecs[:, keep] @ coeffs

        if not torch.isfinite(sol).all():
            raise RuntimeError("non-finite solution")

        print(
            f"    eigh truncation: kept {int(keep.sum().item())}/{eigvals.numel()} "
            f"eigenvalues, threshold={thresh.item():.2e}"
        )
    except (RuntimeError, torch.linalg.LinAlgError) as exc:
        sol = torch.full((K.shape[0],), float("nan"), dtype=DTYPE, device=DEVICE)
        print(f"    Warning: torch.linalg.eigh failed with {type(exc).__name__}")

    wall_time = time.perf_counter() - t0
    return sol[:dim_s], sol[dim_s:], wall_time


# ---------------------------------------------------------------------------
# Iterative solvers
# ---------------------------------------------------------------------------
def run_gda(
    A: torch.Tensor,
    B: torch.Tensor,
    F: torch.Tensor,
    K_max: int = 2000,
    eta_gda: float = 0.02,
    beta_adam: tuple[float, float] = (0.9, 0.98),
    eval_callback: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """GDA with manual Adam: alternate u-ascent and s-descent."""

    dim_s = A.shape[0]
    dim_u = B.shape[1]
    eps_adam = 1.0e-8

    s = torch.zeros(dim_s, dtype=DTYPE, device=DEVICE)
    u = torch.zeros(dim_u, dtype=DTYPE, device=DEVICE)

    m_s = torch.zeros_like(s)
    v_s = torch.zeros_like(s)
    m_u = torch.zeros_like(u)
    v_u = torch.zeros_like(u)

    b1, b2 = beta_adam
    t0 = time.perf_counter()

    for step in range(1, K_max + 1):
        g_u = B.T @ s + F
        m_u = b1 * m_u + (1.0 - b1) * g_u
        v_u = b2 * v_u + (1.0 - b2) * g_u**2
        m_hat_u = m_u / (1.0 - b1**step)
        v_hat_u = v_u / (1.0 - b2**step)
        u = u + eta_gda * m_hat_u / (v_hat_u.sqrt() + eps_adam)

        g_s = A @ s + B @ u
        m_s = b1 * m_s + (1.0 - b1) * g_s
        v_s = b2 * v_s + (1.0 - b2) * g_s**2
        m_hat_s = m_s / (1.0 - b1**step)
        v_hat_s = v_s / (1.0 - b2**step)
        s = s - eta_gda * m_hat_s / (v_hat_s.sqrt() + eps_adam)

        if eval_callback is not None:
            eval_callback(step, s, u)

    wall_time = time.perf_counter() - t0
    return s, u, wall_time


def run_uzawa(
    A: torch.Tensor,
    B: torch.Tensor,
    F: torch.Tensor,
    K_max: int = 2000,
    eta_u: float = 1.0e-2,
    rho: float = 1.0e-6,
    eval_callback: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Uzawa iteration with Cholesky factorization."""

    dim_s = A.shape[0]
    dim_u = B.shape[1]

    A_reg = A + rho * torch.eye(dim_s, dtype=DTYPE, device=DEVICE)
    L = torch.linalg.cholesky(A_reg)

    s = torch.zeros(dim_s, dtype=DTYPE, device=DEVICE)
    u = torch.zeros(dim_u, dtype=DTYPE, device=DEVICE)

    t0 = time.perf_counter()
    for step in range(1, K_max + 1):
        rhs = -(B @ u)
        s = torch.cholesky_solve(rhs.unsqueeze(1), L).squeeze(1)
        u = u + eta_u * (B.T @ s + F)

        if eval_callback is not None:
            eval_callback(step, s, u)

    wall_time = time.perf_counter() - t0
    return s, u, wall_time


def run_arrow_hurwicz(
    A: torch.Tensor,
    B: torch.Tensor,
    F: torch.Tensor,
    K_max: int = 2000,
    eta_s: float = 1.0,
    eta_u: float = 1.0e-2,
    rho: float = 1.0e-6,
    eval_callback: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Arrow-Hurwicz with diagonal preconditioner J = diag(A + rho I)^(-1)."""

    dim_s = A.shape[0]
    A_reg = A + rho * torch.eye(dim_s, dtype=DTYPE, device=DEVICE)
    J_diag = 1.0 / A_reg.diag()

    s = torch.zeros(dim_s, dtype=DTYPE, device=DEVICE)
    u = torch.zeros(B.shape[1], dtype=DTYPE, device=DEVICE)

    t0 = time.perf_counter()
    for step in range(1, K_max + 1):
        g_s = A_reg @ s + B @ u
        s = s - eta_s * J_diag * g_s
        u = u + eta_u * (B.T @ s + F)

        if eval_callback is not None:
            eval_callback(step, s, u)

    wall_time = time.perf_counter() - t0
    return s, u, wall_time


# ---------------------------------------------------------------------------
# Validation and orchestration helpers
# ---------------------------------------------------------------------------
def validate_algorithms_to_run(algorithm_ids: list[str]) -> list[str]:
    """Validate configured algorithm ids before any expensive work starts."""

    if not algorithm_ids:
        raise ValueError(
            "Config.algorithms_to_run must contain at least one algorithm id."
        )

    unknown_ids = [
        algorithm_id
        for algorithm_id in algorithm_ids
        if algorithm_id not in ALGORITHM_SPECS
    ]
    if unknown_ids:
        valid_ids = ", ".join(ALGORITHM_SPECS)
        raise ValueError(
            "Unknown algorithm ids in Config.algorithms_to_run: "
            f"{unknown_ids}. Valid ids: [{valid_ids}]"
        )

    seen: set[str] = set()
    duplicate_ids: list[str] = []
    for algorithm_id in algorithm_ids:
        if algorithm_id in seen and algorithm_id not in duplicate_ids:
            duplicate_ids.append(algorithm_id)
        seen.add(algorithm_id)

    if duplicate_ids:
        raise ValueError(
            "Duplicate algorithm ids in Config.algorithms_to_run: "
            f"{duplicate_ids}"
        )

    return list(algorithm_ids)


def validate_config(cfg: Config, selected_algorithm_ids: list[str]) -> None:
    """Validate configuration before expensive work starts."""

    if cfg.E <= 0.0:
        raise ValueError("Config.E must be positive.")
    if not (-1.0 < cfg.nu < 0.5):
        raise ValueError("Config.nu must lie in (-1, 0.5).")
    if cfg.M_s <= 0 or cfg.M_u <= 0:
        raise ValueError("Config.M_s and Config.M_u must be positive.")
    if cfg.Q_int <= 0:
        raise ValueError("Config.Q_int must be positive.")
    if cfg.Q_test <= 0:
        raise ValueError("Config.Q_test must be positive.")
    if cfg.K_max <= 0:
        raise ValueError("Config.K_max must be positive.")
    if cfg.eval_every <= 0:
        raise ValueError("Config.eval_every must be positive.")
    if cfg.body_force_batch_size <= 0:
        raise ValueError("Config.body_force_batch_size must be positive.")
    if cfg.assembly_batch_size <= 0:
        raise ValueError("Config.assembly_batch_size must be positive.")
    if cfg.sampling_method not in {"mc", "sobol"}:
        raise ValueError("Config.sampling_method must be 'mc' or 'sobol'.")
    if not math.isfinite(cfg.eigh_rtol):
        raise ValueError("Config.eigh_rtol must be finite.")
    if cfg.eigh_rtol < 0.0:
        raise ValueError("Config.eigh_rtol must be non-negative.")

    if cfg.use_penalty:
        if cfg.Q_bc < 6:
            raise ValueError("Config.Q_bc must be at least 6 when use_penalty=True.")
        if not math.isfinite(cfg.lambda_bc):
            raise ValueError("Config.lambda_bc must be finite.")
        if cfg.lambda_bc <= 0.0:
            raise ValueError("Config.lambda_bc must be positive when use_penalty=True.")


def print_result_summary(
    wall_time: float,
    r_c: float,
    r_e: float,
    rel_u: float,
    rel_sigma: float,
) -> None:
    """Print a compact per-method completion summary."""

    print(
        f"    Done in {wall_time:.2f}s, "
        f"||r_c||={r_c:.2e}, ||r_e||={r_e:.2e}, "
        f"rel_u={rel_u:.2e}, rel_sigma={rel_sigma:.2e}"
    )


def get_summary_labels(results: dict[str, dict[str, object]]) -> list[str]:
    """Return present display names in the canonical summary order."""

    return [name for name in SUMMARY_DISPLAY_ORDER if name in results]


def get_iterative_plot_data(
    results: dict[str, dict[str, object]]
) -> tuple[list[str], list[dict[str, list[float | int]]]]:
    """Return iterative labels and histories in canonical order."""

    labels = [name for name in ITERATIVE_DISPLAY_ORDER if name in results]
    histories = [results[name]["history"] for name in labels]
    return labels, histories


def get_l2_plot_data(
    results: dict[str, dict[str, object]]
) -> tuple[list[str], list[dict[str, list[float | int]]]]:
    """Return L2 plot labels and histories in the canonical summary order."""

    labels = [name for name in SUMMARY_DISPLAY_ORDER if name in results]
    histories = [results[name]["history"] for name in labels]
    return labels, histories


def run_all_algorithms(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    F: torch.Tensor,
    xi_u_active_test: torch.Tensor,
    xi_s_test: torch.Tensor,
    u_exact: torch.Tensor,
    sigma_exact: torch.Tensor,
    *,
    algorithms_to_run: list[str] | None = None,
    K_max: int = 2000,
    eval_every: int = 50,
    rho: float = 1.0e-6,
    eta_gda: float = 0.02,
    beta_adam: tuple[float, float] = (0.9, 0.98),
    eta_u_uzawa: float | None = None,
    eta_s_ah: float | None = None,
    eta_u_ah: float | None = None,
    xi_s_train: torch.Tensor | None = None,
    xi_u_active_train: torch.Tensor | None = None,
    u_exact_train: torch.Tensor | None = None,
    sigma_exact_train: torch.Tensor | None = None,
    eigh_rtol: float = 1.0e-12,
) -> dict[str, dict[str, object]]:
    """Run the configured algorithms on one assembled system."""

    selected_algorithm_ids = validate_algorithms_to_run(
        list(DEFAULT_ALGORITHM_IDS)
        if algorithms_to_run is None
        else algorithms_to_run
    )
    results: dict[str, dict[str, object]] = {}
    penalty_active = bool(C.abs().max().item() > 0.0)
    mixed_equilibrium_residuals = (
        penalty_active
        and any(
            algorithm_id in KKT_RESIDUAL_ALGORITHM_IDS
            for algorithm_id in selected_algorithm_ids
        )
        and any(
            algorithm_id not in KKT_RESIDUAL_ALGORITHM_IDS
            for algorithm_id in selected_algorithm_ids
        )
    )

    if mixed_equilibrium_residuals:
        print(
            "  Note: with use_penalty=True, Eigh/Lstsq report "
            "||r_e|| = ||B^T s + C u + F||, while Projection and iterative "
            "methods report ||r_e|| = ||B^T s + F||."
        )

    if "projection" in selected_algorithm_ids:
        if (
            xi_s_train is None
            or xi_u_active_train is None
            or u_exact_train is None
            or sigma_exact_train is None
        ):
            raise RuntimeError("Projection requires exact training data.")

        print("  Running Projection...")
        s, u, wall_time = run_projection(
            xi_s_train,
            xi_u_active_train,
            u_exact_train,
            sigma_exact_train,
        )
        r_c, r_e, rel_u, rel_sigma = evaluate_result(
            A,
            B,
            F,
            xi_u_active_test,
            xi_s_test,
            u_exact,
            sigma_exact,
            s,
            u,
        )
        print_result_summary(wall_time, r_c, r_e, rel_u, rel_sigma)
        results["Projection"] = {
            "s": s,
            "u": u,
            "history": build_single_step_history(r_c, r_e, rel_u, rel_sigma),
            "wall_time": wall_time,
        }

    iterative_ids = [
        algorithm_id
        for algorithm_id in selected_algorithm_ids
        if ALGORITHM_SPECS[algorithm_id].is_iterative
    ]
    uzawa_eta = eta_u_uzawa if eta_u_uzawa is not None else 1.0e-2
    ah_eta_s = eta_s_ah if eta_s_ah is not None else 1.0
    ah_eta_u = eta_u_ah if eta_u_ah is not None else 1.0e-2

    if iterative_ids:
        need_schur = (
            ("uzawa" in iterative_ids and eta_u_uzawa is None)
            or ("arrow_hurwicz" in iterative_ids and eta_u_ah is None)
        )
        need_jacobi = eta_s_ah is None and "arrow_hurwicz" in iterative_ids

        if need_schur:
            print("  Estimating Schur complement spectral radius...")
            lam_max, _ = estimate_schur_spectral_radius(A, B, rho=rho)
            if not math.isfinite(lam_max) or lam_max <= 0.0:
                raise ValueError(
                    "Automatic eta_u requires a positive finite Schur-complement "
                    f"spectral radius. Got lambda_max(S)={lam_max:.4e}."
                )
            eta_u_safe = 1.5 / lam_max
            print(f"    lambda_max(S) = {lam_max:.4e}, safe eta_u = {eta_u_safe:.4e}")
            if "uzawa" in iterative_ids and eta_u_uzawa is None:
                eta_u_uzawa = eta_u_safe
            if "arrow_hurwicz" in iterative_ids and eta_u_ah is None:
                eta_u_ah = eta_u_safe

        if need_jacobi:
            print("  Estimating Jacobi spectral radius for Arrow-Hurwicz...")
            rho_jac = estimate_jacobi_spectral_radius(A, rho=rho)
            eta_s_safe = 1.5 / max(rho_jac, 1.0) if rho_jac > 1.0 else 1.0
            print(f"    rho(Jacobi) = {rho_jac:.4e}, safe eta_s = {eta_s_safe:.4e}")
            eta_s_ah = min(1.0, eta_s_safe)

        uzawa_eta = eta_u_uzawa if eta_u_uzawa is not None else 1.0e-2
        ah_eta_s = eta_s_ah if eta_s_ah is not None else 1.0
        ah_eta_u = eta_u_ah if eta_u_ah is not None else 1.0e-2

        if "uzawa" in iterative_ids:
            print(f"    Uzawa: eta_u={uzawa_eta:.4e}")
        if "arrow_hurwicz" in iterative_ids:
            print(f"    Arrow-Hurwicz: eta_s={ah_eta_s:.4e}, eta_u={ah_eta_u:.4e}")

    for algorithm_id in selected_algorithm_ids:
        if algorithm_id == "projection":
            continue

        spec = ALGORITHM_SPECS[algorithm_id]
        display_name = spec.display_name

        if algorithm_id == "eigh":
            print(f"  Running {display_name} ({spec.backend})...")
            s, u, wall_time = run_eigh(A, B, C, F, rtol=eigh_rtol)
            r_c, r_e, rel_u, rel_sigma = evaluate_result(
                A,
                B,
                F,
                xi_u_active_test,
                xi_s_test,
                u_exact,
                sigma_exact,
                s,
                u,
                penalty_matrix=C,
            )
            print_result_summary(wall_time, r_c, r_e, rel_u, rel_sigma)
            results[display_name] = {
                "s": s,
                "u": u,
                "history": build_single_step_history(r_c, r_e, rel_u, rel_sigma),
                "wall_time": wall_time,
            }
            continue

        if algorithm_id == "lstsq":
            print(f"  Running {display_name} ({spec.backend})...")
            s, u, wall_time = run_lstsq(A, B, C, F)
            r_c, r_e, rel_u, rel_sigma = evaluate_result(
                A,
                B,
                F,
                xi_u_active_test,
                xi_s_test,
                u_exact,
                sigma_exact,
                s,
                u,
                penalty_matrix=C,
            )
            print_result_summary(wall_time, r_c, r_e, rel_u, rel_sigma)
            results[display_name] = {
                "s": s,
                "u": u,
                "history": build_single_step_history(r_c, r_e, rel_u, rel_sigma),
                "wall_time": wall_time,
            }
            continue

        callback, history = make_eval_callback(
            A,
            B,
            F,
            xi_u_active_test,
            xi_s_test,
            u_exact,
            sigma_exact,
            eval_every=eval_every,
        )

        if algorithm_id == "gda":
            print(f"  Running {display_name} ({spec.backend})...")
            s, u, wall_time = run_gda(
                A,
                B,
                F,
                K_max=K_max,
                eta_gda=eta_gda,
                beta_adam=beta_adam,
                eval_callback=callback,
            )
        elif algorithm_id == "uzawa":
            print(f"  Running {display_name} ({spec.backend})...")
            s, u, wall_time = run_uzawa(
                A,
                B,
                F,
                K_max=K_max,
                eta_u=uzawa_eta,
                rho=rho,
                eval_callback=callback,
            )
        elif algorithm_id == "arrow_hurwicz":
            print(f"  Running {display_name} ({spec.backend})...")
            s, u, wall_time = run_arrow_hurwicz(
                A,
                B,
                F,
                K_max=K_max,
                eta_s=ah_eta_s,
                eta_u=ah_eta_u,
                rho=rho,
                eval_callback=callback,
            )
        else:
            raise RuntimeError(f"Unsupported algorithm id: {algorithm_id}")

        finalize_iterative_history(
            history,
            K_max,
            A,
            B,
            F,
            xi_u_active_test,
            xi_s_test,
            u_exact,
            sigma_exact,
            s,
            u,
        )
        r_c = history["r_c"][-1]
        r_e = history["r_e"][-1]
        rel_u = history["rel_u"][-1]
        rel_sigma = history["rel_sigma"][-1]
        print_result_summary(wall_time, r_c, r_e, rel_u, rel_sigma)
        results[display_name] = {
            "s": s,
            "u": u,
            "history": history,
            "wall_time": wall_time,
        }

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_kkt_convergence(
    histories: list[dict[str, list[float | int]]],
    labels: list[str],
    save_path: str,
) -> None:
    """Plot convergence of the tracked residuals."""

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    titles = [r"$\|r_c\|_2$", r"$\|r_e\|_2$"]
    keys = ["r_c", "r_e"]

    for ax, title, key in zip(axes, titles, keys):
        for history, label in zip(histories, labels):
            steps = np.array(history["steps"], dtype=float)
            vals = np.array(history[key], dtype=float)
            valid = np.isfinite(vals) & (vals > 0) & np.isfinite(steps)
            if not valid.any():
                continue

            style = ALGO_STYLE.get(label, {})
            n_markers = max(1, int(valid.sum()) // 10)
            ax.semilogy(
                steps[valid],
                vals[valid],
                label=label,
                linewidth=1.2,
                color=style.get("color"),
                linestyle=style.get("linestyle", "-"),
                marker=style.get("marker"),
                markevery=n_markers,
                markersize=5,
            )

        ax.set_xlabel("Iteration $k$")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_l2_convergence(
    histories: list[dict[str, list[float | int]]],
    labels: list[str],
    save_path: str,
) -> None:
    """Plot L2 error convergence for iterative curves and direct-method points."""

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    titles = [
        r"Displacement $\|u_h - u_{ex}\|_{L^2} / \|u_{ex}\|_{L^2}$",
        r"Stress $\|\sigma_h - \sigma_{ex}\|_{L^2} / \|\sigma_{ex}\|_{L^2}$",
    ]
    keys = ["rel_u", "rel_sigma"]

    for ax, title, key in zip(axes, titles, keys):
        ax.set_yscale("log")
        for history, label in zip(histories, labels):
            steps = np.array(history["steps"], dtype=float)
            vals = np.array(history[key], dtype=float)
            valid = np.isfinite(vals) & (vals > 0) & np.isfinite(steps)
            if not valid.any():
                continue

            style = ALGO_STYLE.get(label, {})
            valid_steps = steps[valid]
            valid_vals = vals[valid]
            if valid_steps.size == 1:
                ax.axhline(
                    valid_vals[0],
                    color=style.get("color"),
                    linestyle=style.get("linestyle", "-"),
                    marker=style.get("marker", "o"),
                    markevery=[0, 1],
                    linewidth=1.2,
                    markersize=5,
                    label=label,
                )
                continue

            n_markers = max(1, int(valid.sum()) // 10)
            ax.semilogy(
                valid_steps,
                valid_vals,
                label=label,
                linewidth=1.2,
                color=style.get("color"),
                linestyle=style.get("linestyle", "-"),
                marker=style.get("marker"),
                markevery=n_markers,
                markersize=5,
            )

        ax.set_xlabel("Iteration $k$")
        ax.set_ylabel("Relative $L^2$ error")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def main(cfg: Config | None = None) -> None:
    """Run the main saddle-point experiment."""

    cfg = Config() if cfg is None else cfg
    selected_algorithm_ids = validate_algorithms_to_run(cfg.algorithms_to_run)
    validate_config(cfg, selected_algorithm_ids)
    projection_enabled = "projection" in selected_algorithm_ids

    print(f"Device: {DEVICE}")
    print(
        f"Config: M_s={cfg.M_s}, M_u={cfg.M_u}, "
        f"Q_int={cfg.Q_int}, Q_bc={cfg.Q_bc}, Q_test={cfg.Q_test}, "
        f"gamma_s={cfg.gamma_s}, gamma_u={cfg.gamma_u}, "
        f"sampling={cfg.sampling_method}, "
        f"use_zeta={cfg.use_zeta}, use_penalty={cfg.use_penalty}, "
        f"lambda_bc={cfg.lambda_bc:.2e}, eigh_rtol={cfg.eigh_rtol:.2e}, "
        f"assembly_batch_size={cfg.assembly_batch_size}"
    )
    print(f"Algorithms: {selected_algorithm_ids}")
    print(f"Output: {OUTPUT_DIR}")

    mu, lam = compute_lame_constants(cfg.E, cfg.nu)
    compliance_voigt = build_compliance_matrix(cfg.E, cfg.nu)
    print(f"Material: E={cfg.E}, nu={cfg.nu}, mu={mu:.4f}, lam={lam:.4f}")

    print(f"Sampling {cfg.Q_int} training points...")
    x_train = sample_points(
        cfg.Q_int,
        method=cfg.sampling_method,
        seed=BASE_SEED + 1,
    )

    print("Computing body force...")
    f_train = compute_body_force(
        x_train,
        mu,
        lam,
        batch_size=cfg.body_force_batch_size,
    )

    u_exact_train = None
    sigma_exact_train = None
    if projection_enabled:
        print("Computing exact training fields for projection...")
        u_exact_train = eval_exact_displacement(x_train)
        sigma_exact_train = compute_stress_voigt(x_train, mu, lam)

    zeta_train = (
        eval_zeta(x_train)
        if cfg.use_zeta
        else torch.ones(x_train.shape[0], dtype=DTYPE, device=DEVICE)
    )

    print("Generating split feature spaces...")
    a_s, r_s = generate_features(cfg.M_s, seed=STRESS_SEED)
    a_u, r_u = generate_features(cfg.M_u, seed=DISP_SEED)

    xi_s_train = None
    xi_u_active_train = None
    if projection_enabled:
        xi_s_train = eval_features(x_train, a_s, r_s, cfg.gamma_s)
        xi_u_train = eval_features(x_train, a_u, r_u, cfg.gamma_u)
        xi_u_active_train = activate_displacement_features(xi_u_train, zeta_train)

    x_bc = None
    w_bc = None
    zeta_bc = None
    if cfg.use_penalty:
        print(f"Sampling {cfg.Q_bc} boundary points...")
        x_bc, w_bc = sample_boundary_points(
            cfg.Q_bc,
            method=cfg.sampling_method,
            seed=BASE_SEED + 2,
        )
        zeta_bc = (
            eval_zeta(x_bc)
            if cfg.use_zeta
            else torch.ones(x_bc.shape[0], dtype=DTYPE, device=DEVICE)
        )

    print("Assembling saddle-point system...")
    A, B, C, F = assemble_system_in_batches(
        x_train,
        f_train,
        a_s,
        r_s,
        cfg.gamma_s,
        a_u,
        r_u,
        cfg.gamma_u,
        compliance_voigt,
        zeta_train,
        cfg.assembly_batch_size,
        x_bc=x_bc,
        w_bc=w_bc,
        zeta_bc=zeta_bc,
        lambda_bc=cfg.lambda_bc if cfg.use_penalty else 0.0,
    )
    print(
        f"  A: {tuple(A.shape)}, B: {tuple(B.shape)}, "
        f"C: {tuple(C.shape)}, F: {tuple(F.shape)}"
    )

    print(f"Sampling {cfg.Q_test} test points...")
    x_test = sample_points(
        cfg.Q_test,
        method=cfg.sampling_method,
        seed=BASE_SEED + 3,
    )
    zeta_test = (
        eval_zeta(x_test)
        if cfg.use_zeta
        else torch.ones(x_test.shape[0], dtype=DTYPE, device=DEVICE)
    )
    xi_s_test = eval_features(x_test, a_s, r_s, cfg.gamma_s)
    xi_u_test = eval_features(x_test, a_u, r_u, cfg.gamma_u)
    xi_u_active_test = activate_displacement_features(xi_u_test, zeta_test)
    u_exact = eval_exact_displacement(x_test)
    sigma_exact = compute_stress_voigt(x_test, mu, lam)

    print(
        f"\n=== Main experiment (M_s={cfg.M_s}, M_u={cfg.M_u}, Q={cfg.Q_int}) ==="
    )
    results = run_all_algorithms(
        A,
        B,
        C,
        F,
        xi_u_active_test,
        xi_s_test,
        u_exact,
        sigma_exact,
        algorithms_to_run=selected_algorithm_ids,
        K_max=cfg.K_max,
        eval_every=cfg.eval_every,
        rho=cfg.rho,
        eta_gda=cfg.eta_gda,
        beta_adam=cfg.beta_adam,
        eta_u_uzawa=cfg.eta_u_uzawa,
        eta_s_ah=cfg.eta_s_ah,
        eta_u_ah=cfg.eta_u_ah,
        xi_s_train=xi_s_train if projection_enabled else None,
        xi_u_active_train=xi_u_active_train if projection_enabled else None,
        u_exact_train=u_exact_train,
        sigma_exact_train=sigma_exact_train,
        eigh_rtol=cfg.eigh_rtol,
    )

    print("\nGenerating plots...")
    iter_labels, iter_histories = get_iterative_plot_data(results)
    if iter_labels:
        l2_labels, l2_histories = get_l2_plot_data(results)
        plot_kkt_convergence(
            iter_histories,
            iter_labels,
            str(OUTPUT_DIR / "kkt-convergence.png"),
        )
        plot_l2_convergence(
            l2_histories,
            l2_labels,
            str(OUTPUT_DIR / "l2-error-convergence.png"),
        )
    else:
        print("  Skipping convergence plots because no iterative algorithms were selected.")

    print("\n=== Summary ===\n")
    print(
        f"| {'Method':<14} | {'||r_c||':>10} | {'||r_e||':>10} | "
        f"{'rel_u':>10} | {'rel_sigma':>10} | {'Time(s)':>8} |"
    )
    print(
        f"|:{'-'*15}|{'-'*11}:|{'-'*11}:|{'-'*11}:|{'-'*11}:|{'-'*9}:|"
    )
    for name in get_summary_labels(results):
        item = results[name]
        history = item["history"]
        print(
            f"| {name:<14} | {history['r_c'][-1]:>10.2e} | {history['r_e'][-1]:>10.2e} | "
            f"{history['rel_u'][-1]:>10.2e} | {history['rel_sigma'][-1]:>10.2e} | "
            f"{item['wall_time']:>8.2f} |"
        )


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    main()
