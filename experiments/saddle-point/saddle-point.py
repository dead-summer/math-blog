"""
3D Hellinger-Reissner saddle-point numerical experiments

Implements Direct, ADMM, Uzawa, and Arrow-Hurwicz iterative algorithms for the
discrete saddle-point system arising from the Hellinger-Reissner variational
principle with random neural features. Generates convergence comparison plots.
"""

import math
import os
import time
from pathlib import Path

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
# Device & seed
# ---------------------------------------------------------------------------
BASE_SEED = 42

np.random.seed(BASE_SEED)
torch.manual_seed(BASE_SEED)
torch.cuda.manual_seed_all(BASE_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Material parameters
# ---------------------------------------------------------------------------
def compute_lame_constants(E: float, nu: float):
    """Return (mu, lam) from Young's modulus and Poisson's ratio."""
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def build_compliance_matrix(E: float, nu: float) -> torch.Tensor:
    """6x6 Voigt compliance matrix S, order (11,22,33,12,23,13)."""
    S = torch.zeros(6, 6, dtype=DTYPE, device=device)
    S[0, 0] = S[1, 1] = S[2, 2] = 1.0 / E
    S[0, 1] = S[0, 2] = S[1, 0] = S[1, 2] = S[2, 0] = S[2, 1] = -nu / E
    shear = 2.0 * (1.0 + nu) / E
    S[3, 3] = S[4, 4] = S[5, 5] = shear
    return S


# ---------------------------------------------------------------------------
# Manufactured solution & body force
# ---------------------------------------------------------------------------
def eval_exact_displacement(x: torch.Tensor) -> torch.Tensor:
    """Exact displacement u_ex(x), shape (N,3) -> (N,3)."""
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    zeta = x1 * (1 - x1) * x2 * (1 - x2) * x3 * (1 - x3)
    pi = math.pi
    u1 = zeta * torch.sin(pi * x1) * torch.sin(pi * x2) * torch.sin(pi * x3)
    u2 = zeta * torch.sin(2 * pi * x1) * torch.sin(pi * x2) * torch.sin(pi * x3)
    u3 = zeta * torch.sin(pi * x1) * torch.sin(2 * pi * x2) * torch.sin(pi * x3)
    return torch.stack([u1, u2, u3], dim=1)


def eval_exact_stress_voigt(x: torch.Tensor, mu: float, lam: float) -> torch.Tensor:
    """Exact stress in Voigt form (N,6) via autograd from exact displacement.

    Voigt order: (sigma_11, sigma_22, sigma_33, sigma_12, sigma_23, sigma_13).
    """
    x_ad = x.detach().requires_grad_(True)
    u = eval_exact_displacement(x_ad)  # (N,3)

    N = x.shape[0]
    grad_u = torch.zeros(N, 3, 3, dtype=DTYPE, device=device)
    for i in range(3):
        g = torch.autograd.grad(u[:, i].sum(), x_ad, create_graph=False,
                                retain_graph=(i < 2))[0]
        grad_u[:, i, :] = g

    eps = 0.5 * (grad_u + grad_u.transpose(1, 2))
    tr_eps = eps[:, 0, 0] + eps[:, 1, 1] + eps[:, 2, 2]
    sigma = 2.0 * mu * eps
    for i in range(3):
        sigma[:, i, i] += lam * tr_eps

    sig_voigt = torch.stack([
        sigma[:, 0, 0], sigma[:, 1, 1], sigma[:, 2, 2],
        sigma[:, 0, 1], sigma[:, 1, 2], sigma[:, 0, 2],
    ], dim=1)
    return sig_voigt.detach()


def compute_body_force(x: torch.Tensor, mu: float, lam: float,
                       batch_size: int = 5000) -> torch.Tensor:
    """Body force f = -div(sigma(u_ex)), computed via two-layer autograd."""
    N = x.shape[0]
    f_all = torch.zeros(N, 3, dtype=DTYPE, device=device)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        xb = x[start:end].detach().requires_grad_(True)
        u = eval_exact_displacement(xb)

        grad_u_list = []
        for i in range(3):
            g = torch.autograd.grad(u[:, i].sum(), xb, create_graph=True)[0]
            grad_u_list.append(g)
        grad_u = torch.stack(grad_u_list, dim=1)

        eps = 0.5 * (grad_u + grad_u.transpose(1, 2))
        tr_eps = eps[:, 0, 0] + eps[:, 1, 1] + eps[:, 2, 2]

        sigma = 2.0 * mu * eps
        for i in range(3):
            sigma[:, i, i] = sigma[:, i, i] + lam * tr_eps

        for i in range(3):
            div_sig_i = torch.zeros(end - start, dtype=DTYPE, device=device)
            for j in range(3):
                is_last = (i == 2 and j == 2)
                g = torch.autograd.grad(
                    sigma[:, i, j].sum(), xb,
                    create_graph=False, retain_graph=not is_last,
                )[0]
                div_sig_i = div_sig_i + g[:, j]
            f_all[start:end, i] = -div_sig_i.detach()

    return f_all


# ---------------------------------------------------------------------------
# Neural feature space
# ---------------------------------------------------------------------------
def generate_features(M: int, seed: int):
    """Generate random feature parameters (a, r).

    a: (M,3) unit vectors, r: (M,) uniform offsets in [0,1].
    """
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    raw = torch.randn(M, 3, generator=rng, dtype=DTYPE)
    norms = raw.norm(dim=1, keepdim=True).clamp_min(1e-12)
    a = (raw / norms).to(device)
    r = torch.rand(M, generator=rng, dtype=DTYPE).to(device)
    return a, r


def eval_features(x: torch.Tensor, a: torch.Tensor, r: torch.Tensor,
                  gamma: float) -> torch.Tensor:
    """Evaluate xi_0=1, xi_m=tanh(gamma*(a_m^T x + r_m)). Returns (Q, M+1)."""
    Q = x.shape[0]
    z = gamma * (x @ a.T + r.unsqueeze(0))
    xi = torch.tanh(z)
    ones = torch.ones(Q, 1, dtype=DTYPE, device=device)
    return torch.cat([ones, xi], dim=1)


def eval_feature_grads(x: torch.Tensor, a: torch.Tensor, r: torch.Tensor,
                       gamma: float) -> torch.Tensor:
    """Evaluate gradients of features. Returns (Q, M+1, 3). grad(xi_0)=0."""
    Q = x.shape[0]
    z = gamma * (x @ a.T + r.unsqueeze(0))
    dtanh = 1.0 - torch.tanh(z) ** 2
    grad_xi = gamma * dtanh.unsqueeze(2) * a.unsqueeze(0)
    zeros = torch.zeros(Q, 1, 3, dtype=DTYPE, device=device)
    return torch.cat([zeros, grad_xi], dim=1)


# ---------------------------------------------------------------------------
# Matrix assembly (Kronecker structure)
# ---------------------------------------------------------------------------
def assemble_system(xi: torch.Tensor, grad_xi: torch.Tensor,
                    S: torch.Tensor, f_vals: torch.Tensor,
                    zeta: torch.Tensor):
    """Assemble A, B, F matrices for the saddle-point system.

    Displacement features are multiplied by ζ(x) to enforce homogeneous
    Dirichlet BCs: ψ_m(x) = ζ(x) ξ_m(x), so u_h = Σ u_{m,i} ψ_m e_i = 0 on ∂Ω.

    Args:
        xi: (Q, M+1) feature values
        grad_xi: (Q, M+1, 3) feature gradients
        S: (6,6) compliance matrix
        f_vals: (Q, 3) body force values
        zeta: (Q,) envelope function ζ(x) = x1(1-x1)x2(1-x2)x3(1-x3)

    Returns:
        A: (6*(M+1), 6*(M+1))
        B: (6*(M+1), 3*(M+1))
        F: (3*(M+1),)
    """
    Q = xi.shape[0]
    Mp1 = xi.shape[1]

    # Displacement features: ψ = ζ * ξ
    psi = zeta.unsqueeze(1) * xi  # (Q, Mp1)

    # Gram matrix G = (1/Q) xi^T xi (stress-stress)
    G = (1.0 / Q) * (xi.T @ xi)

    # A = kron-like(G, S): A[alpha::6, beta::6] = S[alpha,beta] * G
    dim_s = 6 * Mp1
    A = torch.zeros(dim_s, dim_s, dtype=DTYPE, device=device)
    for alpha in range(6):
        for beta in range(6):
            A[alpha::6, beta::6] = S[alpha, beta] * G

    # Derivative matrices using displacement features ψ for test:
    # D_k[n, m] = (1/Q) sum_q ψ_m(x_q) * d_k ξ_n(x_q)
    D = []
    for k in range(3):
        Dk = (1.0 / Q) * (psi.T @ grad_xi[:, :, k])  # [m, n]
        D.append(Dk.T)  # [n, m]

    # B matrix: 6*(M+1) x 3*(M+1)
    dim_u = 3 * Mp1
    B = torch.zeros(dim_s, dim_u, dtype=DTYPE, device=device)

    B[0::6, 0::3] = D[0]   # beta=0(11), i=0: D0
    B[1::6, 1::3] = D[1]   # beta=1(22), i=1: D1
    B[2::6, 2::3] = D[2]   # beta=2(33), i=2: D2
    B[3::6, 0::3] = D[1]   # beta=3(12), i=0: D1
    B[3::6, 1::3] = D[0]   # beta=3(12), i=1: D0
    B[4::6, 1::3] = D[2]   # beta=4(23), i=1: D2
    B[4::6, 2::3] = D[1]   # beta=4(23), i=2: D1
    B[5::6, 0::3] = D[2]   # beta=5(13), i=0: D2
    B[5::6, 2::3] = D[0]   # beta=5(13), i=2: D0

    # F vector using displacement features ψ: F[i::3] = (1/Q) ψ^T f[:, i]
    F_mat = (1.0 / Q) * (psi.T @ f_vals)  # (Mp1, 3)
    F_vec = torch.zeros(dim_u, dtype=DTYPE, device=device)
    for i in range(3):
        F_vec[i::3] = F_mat[:, i]

    return A, B, F_vec


# ---------------------------------------------------------------------------
# KKT residuals & L2 errors
# ---------------------------------------------------------------------------
def compute_kkt_residuals(A, B, F, s, u):
    """Compute KKT residual norms: (||r_s||_2, ||r_u||_2)."""
    r_s = A @ s + B @ u
    r_u = B.T @ s + F
    return r_s.norm().item(), r_u.norm().item()


def compute_l2_errors(psi_test, s, u, u_exact, sigma_exact, xi_test):
    """Evaluate relative L2 errors for displacement and stress on test points.

    Displacement uses ψ = ζ*ξ features; stress uses ξ features.
    """
    Q_test = psi_test.shape[0]

    # Reconstruct displacement: u_h[:, i] = psi_test @ u[i::3]
    u_h = torch.zeros(Q_test, 3, dtype=DTYPE, device=device)
    for i in range(3):
        u_h[:, i] = psi_test @ u[i::3]

    # Reconstruct stress: sigma_h[:, alpha] = xi_test @ s[alpha::6]
    sigma_h = torch.zeros(Q_test, 6, dtype=DTYPE, device=device)
    for alpha in range(6):
        sigma_h[:, alpha] = xi_test @ s[alpha::6]

    w_frob = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                           dtype=DTYPE, device=device)

    u_diff = u_h - u_exact
    u_err = torch.sqrt((u_diff ** 2).sum(1).mean())
    u_ref = torch.sqrt((u_exact ** 2).sum(1).mean())
    rel_u = (u_err / u_ref).item() if u_ref > 0 else float("inf")

    sig_diff = sigma_h - sigma_exact
    sig_err = torch.sqrt((w_frob * sig_diff ** 2).sum(1).mean())
    sig_ref = torch.sqrt((w_frob * sigma_exact ** 2).sum(1).mean())
    rel_sig = (sig_err / sig_ref).item() if sig_ref > 0 else float("inf")

    return rel_u, rel_sig


def make_eval_callback(A, B, F, psi_test, xi_test, u_exact, sigma_exact,
                       eval_every=50):
    """Create a callback that records KKT residuals and L2 errors."""
    history = {
        "r_s": [], "r_u": [], "r_total": [],
        "rel_u": [], "rel_sigma": [],
        "steps": [],
    }

    def callback(k, s, u):
        rs, ru = compute_kkt_residuals(A, B, F, s, u)
        history["r_s"].append(rs)
        history["r_u"].append(ru)
        history["r_total"].append(rs + ru)
        if k % eval_every == 0 or k <= 1:
            rel_u, rel_sig = compute_l2_errors(
                psi_test, s, u, u_exact, sigma_exact, xi_test)
            history["rel_u"].append(rel_u)
            history["rel_sigma"].append(rel_sig)
            history["steps"].append(k)

    return callback, history


# ---------------------------------------------------------------------------
# Direct solve (baseline)
# ---------------------------------------------------------------------------
def run_direct_solve(A, B, F):
    """Solve the saddle-point system [A B; B^T 0] [s; u] = [0; -F] directly."""
    dim_s = A.shape[0]
    dim_u = B.shape[1]
    dim_total = dim_s + dim_u

    K = torch.zeros(dim_total, dim_total, dtype=DTYPE, device=device)
    K[:dim_s, :dim_s] = A
    K[:dim_s, dim_s:] = B
    K[dim_s:, :dim_s] = B.T

    rhs = torch.zeros(dim_total, dtype=DTYPE, device=device)
    rhs[dim_s:] = -F

    t0 = time.perf_counter()
    sol = torch.linalg.solve(K, rhs)
    wall_time = time.perf_counter() - t0

    return sol[:dim_s], sol[dim_s:], wall_time

# ---------------------------------------------------------------------------
# ADMM (Adam optimizer, full-batch)
# ---------------------------------------------------------------------------
def run_admm(A, B, F, K_max=2000, eta_admm=0.02, beta_adam=(0.9, 0.98),
             tol=1e-6, eval_callback=None):
    """ADMM with manual Adam: alternate u-ascent / s-descent."""
    dim_s = A.shape[0]
    dim_u = B.shape[1]
    eps_adam = 1e-8

    s = torch.zeros(dim_s, dtype=DTYPE, device=device)
    u = torch.zeros(dim_u, dtype=DTYPE, device=device)

    m_s = torch.zeros_like(s)
    v_s = torch.zeros_like(s)
    m_u = torch.zeros_like(u)
    v_u = torch.zeros_like(u)

    b1, b2 = beta_adam
    t0 = time.perf_counter()

    for k in range(1, K_max + 1):
        g_u = B.T @ s + F
        m_u = b1 * m_u + (1 - b1) * g_u
        v_u = b2 * v_u + (1 - b2) * g_u ** 2
        m_hat_u = m_u / (1 - b1 ** k)
        v_hat_u = v_u / (1 - b2 ** k)
        u = u + eta_admm * m_hat_u / (v_hat_u.sqrt() + eps_adam)

        g_s = A @ s + B @ u
        m_s = b1 * m_s + (1 - b1) * g_s
        v_s = b2 * v_s + (1 - b2) * g_s ** 2
        m_hat_s = m_s / (1 - b1 ** k)
        v_hat_s = v_s / (1 - b2 ** k)
        s = s - eta_admm * m_hat_s / (v_hat_s.sqrt() + eps_adam)

        if eval_callback is not None:
            eval_callback(k, s, u)

        if k % 100 == 0:
            rs, ru = compute_kkt_residuals(A, B, F, s, u)
            if rs + ru <= tol:
                break

    wall_time = time.perf_counter() - t0
    return s, u, wall_time


# ---------------------------------------------------------------------------
# Uzawa (Cholesky pre-factorization)
# ---------------------------------------------------------------------------
def run_uzawa(A, B, F, K_max=2000, eta_u=1e-2, rho=1e-6,
              tol=1e-6, eval_callback=None):
    """Uzawa iteration with Cholesky factorization."""
    dim_s = A.shape[0]
    dim_u = B.shape[1]

    A_reg = A + rho * torch.eye(dim_s, dtype=DTYPE, device=device)
    L = torch.linalg.cholesky(A_reg)

    s = torch.zeros(dim_s, dtype=DTYPE, device=device)
    u = torch.zeros(dim_u, dtype=DTYPE, device=device)

    t0 = time.perf_counter()

    for k in range(1, K_max + 1):
        rhs = -(B @ u)
        s = torch.cholesky_solve(rhs.unsqueeze(1), L).squeeze(1)
        u = u + eta_u * (B.T @ s + F)

        if eval_callback is not None:
            eval_callback(k, s, u)

        if k % 100 == 0:
            rs, ru = compute_kkt_residuals(A, B, F, s, u)
            if rs + ru <= tol:
                break

    wall_time = time.perf_counter() - t0
    return s, u, wall_time


# ---------------------------------------------------------------------------
# Arrow-Hurwicz (preconditioned gradient)
# ---------------------------------------------------------------------------
def run_arrow_hurwicz(A, B, F, K_max=2000, eta_s=1.0, eta_u=1e-2, rho=1e-6,
                      tol=1e-6, eval_callback=None):
    """Arrow-Hurwicz with diagonal preconditioner J = diag(A+rho*I)^{-1}."""
    dim_s = A.shape[0]

    A_reg = A + rho * torch.eye(dim_s, dtype=DTYPE, device=device)
    J_diag = 1.0 / A_reg.diag()

    s = torch.zeros(dim_s, dtype=DTYPE, device=device)
    u = torch.zeros(A.shape[1] if A.shape[1] != A.shape[0] else B.shape[1],
                     dtype=DTYPE, device=device)

    t0 = time.perf_counter()

    for k in range(1, K_max + 1):
        g_s = A_reg @ s + B @ u
        s = s - eta_s * J_diag * g_s
        u = u + eta_u * (B.T @ s + F)

        if eval_callback is not None:
            eval_callback(k, s, u)

        if k % 100 == 0:
            rs, ru = compute_kkt_residuals(A, B, F, s, u)
            if rs + ru <= tol:
                break

    wall_time = time.perf_counter() - t0
    return s, u, wall_time


# ---------------------------------------------------------------------------
# Helper: run all algorithms on same system
# ---------------------------------------------------------------------------
def run_all_algorithms(A, B, F, psi_test, xi_test, u_exact, sigma_exact,
                       K_max=2000, eval_every=50, rho=1e-6,
                       eta_admm=0.02, beta_adam=(0.9, 0.98),
                       eta_u_uzawa=1e-2, eta_s_ah=3e-3, eta_u_ah=1e-2):
    """Run Direct, ADMM, Uzawa, Arrow-Hurwicz and collect histories."""
    results = {}

    # --- Direct solve (baseline) ---
    print("  Running Direct solve...")
    s, u, wt = run_direct_solve(A, B, F)
    rs, ru = compute_kkt_residuals(A, B, F, s, u)
    rel_u, rel_sig = compute_l2_errors(psi_test, s, u, u_exact, sigma_exact, xi_test)
    history = {
        "r_s": [rs], "r_u": [ru], "r_total": [rs + ru],
        "rel_u": [rel_u], "rel_sigma": [rel_sig], "steps": [0],
    }
    results["Direct"] = {"s": s, "u": u, "history": history, "wall_time": wt}
    print(f"    Done in {wt:.2f}s, final KKT: "
          f"||r_s||={rs:.2e}, ||r_u||={ru:.2e}, total={rs+ru:.2e}")

    for name, runner, kwargs in [
        ("ADMM", run_admm,
         dict(K_max=K_max, eta_admm=eta_admm, beta_adam=beta_adam)),
        ("Uzawa", run_uzawa,
         dict(K_max=K_max, eta_u=eta_u_uzawa, rho=rho)),
        ("Arrow-Hurwicz", run_arrow_hurwicz,
         dict(K_max=K_max, eta_s=eta_s_ah, eta_u=eta_u_ah, rho=rho)),
    ]:
        cb, hist = make_eval_callback(A, B, F, psi_test, xi_test,
                                      u_exact, sigma_exact,
                                      eval_every=eval_every)
        print(f"  Running {name}...")
        s, u, wt = runner(A, B, F, eval_callback=cb, **kwargs)
        rs, ru = compute_kkt_residuals(A, B, F, s, u)
        print(f"    Done in {wt:.2f}s, final KKT: "
              f"||r_s||={rs:.2e}, ||r_u||={ru:.2e}, total={rs+ru:.2e}")
        results[name] = {"s": s, "u": u, "history": hist, "wall_time": wt}

    return results

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_kkt_convergence(histories, labels, save_path):
    """Plot KKT residual convergence: 1x3 subplots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = [r"$\|r_s\|_2$", r"$\|r_u\|_2$", r"$\|r_s\|_2 + \|r_u\|_2$"]
    keys = ["r_s", "r_u", "r_total"]

    for ax, title, key in zip(axes, titles, keys):
        for hist, label in zip(histories, labels):
            vals = np.array(hist[key], dtype=float)
            valid = np.isfinite(vals) & (vals > 0)
            if valid.any():
                iters = np.arange(1, len(vals) + 1)
                ax.semilogy(iters[valid], vals[valid], label=label, linewidth=1.2)
        ax.set_xlabel("Iteration $k$")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_l2_convergence(histories, labels, save_path, direct_l2=None):
    """Plot L2 error convergence: 1x2 subplots.

    If direct_l2 is provided (dict with 'rel_u' and 'rel_sigma'),
    draw horizontal dashed reference lines for Direct solve.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    titles = [
        r"Displacement $\|u_h - u_{ex}\|_{L^2} / \|u_{ex}\|_{L^2}$",
        r"Stress $\|\sigma_h - \sigma_{ex}\|_{L^2} / \|\sigma_{ex}\|_{L^2}$",
    ]
    keys = ["rel_u", "rel_sigma"]

    for ax, title, key in zip(axes, titles, keys):
        for hist, label in zip(histories, labels):
            steps = np.array(hist["steps"], dtype=float)
            vals = np.array(hist[key], dtype=float)
            valid = np.isfinite(vals) & (vals > 0) & np.isfinite(steps)
            if valid.any():
                ax.semilogy(steps[valid], vals[valid], marker="o", markersize=3,
                            label=label, linewidth=1.2)
        if direct_l2 is not None:
            ref_val = direct_l2.get(key)
            if ref_val is not None and np.isfinite(ref_val) and ref_val > 0:
                ax.axhline(y=ref_val, color="black", linestyle="--",
                           linewidth=1.0, alpha=0.7, label="Direct solve")
        ax.set_xlabel("Iteration $k$")
        ax.set_ylabel("Relative $L^2$ error")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_ablation_M(results, save_path):
    """Line plot comparing final errors across M values for all methods.

    Args:
        results: { M: { method_name: {"rel_u", "rel_sigma", "kkt_total"} } }
        save_path: output file path
    """
    M_values = sorted(results.keys())
    # Collect all method names from the first M entry
    method_names = list(results[M_values[0]].keys())

    style_map = {
        "Direct":         {"color": "black",       "marker": "s"},
        "ADMM":           {"color": "steelblue",   "marker": "o"},
        "Uzawa":          {"color": "darkorange",  "marker": "^"},
        "Arrow-Hurwicz":  {"color": "forestgreen", "marker": "D"},
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metric_keys = ["rel_u", "rel_sigma", "kkt_total"]
    titles = ["Displacement error", "Stress error", "Final KKT residual"]
    ylabels = [
        "Relative $L^2$ error",
        "Relative $L^2$ error",
        r"$\|r_s\|_2 + \|r_u\|_2$",
    ]

    for ax, mkey, title, ylabel in zip(axes, metric_keys, titles, ylabels):
        for method in method_names:
            vals = []
            for M in M_values:
                v = results[M].get(method, {}).get(mkey, float("nan"))
                vals.append(v)
            vals = np.array(vals, dtype=float)
            valid = np.isfinite(vals) & (vals > 0)
            if valid.any():
                sty = style_map.get(method, {"color": "gray", "marker": "x"})
                ax.semilogy(
                    np.array(M_values)[valid], vals[valid],
                    marker=sty["marker"], color=sty["color"],
                    label=method, linewidth=1.5, markersize=6,
                )
        ax.set_xlabel("Feature count $M$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(M_values)
        ax.set_xticklabels([str(m) for m in M_values])
        ax.legend()
        ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    print(f"Device: {device}")
    print(f"Output: {OUTPUT_DIR}")

    # --- Parameters ---
    E, nu = 1.0, 0.3
    gamma = 2.0
    M = 256
    ablation_M_list = [64, 128, 256, 512, 1024]
    Q_train = 20000
    Q_test = 10000
    K_max = 2000
    rho = 1e-6
    eta_admm = 2e-02
    beta_adam = (0.9, 0.98)
    eta_u_uzawa = 1e-02
    eta_s_ah = 3e-03
    eta_u_ah = 1e-02

    mu, lam = compute_lame_constants(E, nu)
    S = build_compliance_matrix(E, nu)
    print(f"Material: E={E}, nu={nu}, mu={mu:.4f}, lam={lam:.4f}")

    # --- Generate features (512 for ablation, truncate to M for main) ---
    a_full, r_full = generate_features(512, seed=BASE_SEED)
    a, r = a_full[:M], r_full[:M]

    # --- Training points ---
    print(f"Sampling {Q_train} training points...")
    torch.manual_seed(BASE_SEED + 1)
    x_train = torch.rand(Q_train, 3, dtype=DTYPE, device=device)

    xi_train = eval_features(x_train, a, r, gamma)
    grad_xi_train = eval_feature_grads(x_train, a, r, gamma)

    # --- Body force ---
    f_train = compute_body_force(x_train, mu, lam)

    # --- Envelope function for Dirichlet BC ---
    def zeta_fn(x):
        """ζ(x) = x1(1-x1)x2(1-x2)x3(1-x3), ensures u_h=0 on ∂Ω."""
        return x[:, 0] * (1 - x[:, 0]) * x[:, 1] * (1 - x[:, 1]) * x[:, 2] * (1 - x[:, 2])

    zeta_train = zeta_fn(x_train)

    # --- Assemble system ---
    print("Assembling saddle-point system...")
    A, B, F = assemble_system(xi_train, grad_xi_train, S, f_train, zeta_train)
    Mp1 = M + 1
    print(f"  A: {A.shape}, B: {B.shape}, F: {F.shape}")
    print(f"  A memory: {A.numel() * 8 / 1e6:.1f} MB")
    print(f"  ||A||_F = {A.norm():.4e}, ||B||_F = {B.norm():.4e}, ||F|| = {F.norm():.4e}")

    # --- Test points ---
    print(f"Sampling {Q_test} test points...")
    torch.manual_seed(BASE_SEED + 2)
    x_test = torch.rand(Q_test, 3, dtype=DTYPE, device=device)
    xi_test = eval_features(x_test, a, r, gamma)
    zeta_test = zeta_fn(x_test)
    psi_test = zeta_test.unsqueeze(1) * xi_test  # displacement features
    u_exact = eval_exact_displacement(x_test)
    sigma_exact = eval_exact_stress_voigt(x_test, mu, lam)

    # --- Run all algorithms (including Direct solve) ---
    print(f"\n=== Main experiment (M={M}, Q=20000) ===")
    results = run_all_algorithms(A, B, F, psi_test, xi_test, u_exact, sigma_exact,
                                 K_max=K_max, eval_every=50, rho=rho,
                                 eta_admm=eta_admm, beta_adam=beta_adam,
                                 eta_u_uzawa=eta_u_uzawa, eta_s_ah=eta_s_ah, eta_u_ah=eta_u_ah)

    # --- Plot KKT convergence (iterative methods only) ---
    print("\nGenerating plots...")
    iter_labels = ["ADMM", "Uzawa", "Arrow-Hurwicz"]
    iter_histories = [results[name]["history"] for name in iter_labels]

    plot_kkt_convergence(iter_histories, iter_labels,
                         str(OUTPUT_DIR / "kkt-convergence.png"))

    # --- Plot L2 convergence with Direct solve reference line ---
    direct_hist = results["Direct"]["history"]
    direct_l2 = {
        "rel_u": direct_hist["rel_u"][-1],
        "rel_sigma": direct_hist["rel_sigma"][-1],
    }
    plot_l2_convergence(iter_histories, iter_labels,
                        str(OUTPUT_DIR / "l2-error-convergence.png"),
                        direct_l2=direct_l2)

    # --- Summary table (all 4 methods) ---
    print("\n=== Summary ===")
    all_labels = ["Direct", "ADMM", "Uzawa", "Arrow-Hurwicz"]
    print(f"{'Algorithm':<16} {'||r_s||':>10} {'||r_u||':>10} "
          f"{'Total':>10} {'u_err':>10} {'sig_err':>10} {'Time(s)':>8}")
    print("-" * 76)
    for name in all_labels:
        h = results[name]["history"]
        rs_final = h["r_s"][-1]
        ru_final = h["r_u"][-1]
        rel_u = h["rel_u"][-1]
        rel_sig = h["rel_sigma"][-1]
        wt = results[name]["wall_time"]
        print(f"{name:<16} {rs_final:10.2e} {ru_final:10.2e} "
              f"{rs_final+ru_final:10.2e} {rel_u:10.2e} {rel_sig:10.2e} "
              f"{wt:8.2f}")

    # --- Ablation: M (all 4 methods) ---
    print("\n=== Ablation: feature count M ===")
    ablation_M_results = {}  # { M: { method: {rel_u, rel_sigma, kkt_total} } }

    for M_abl in ablation_M_list:
        print(f"\n--- M={M_abl} ---")
        a_abl = a_full[:M_abl]
        r_abl = r_full[:M_abl]
        xi_abl = eval_features(x_train, a_abl, r_abl, gamma)
        grad_xi_abl = eval_feature_grads(x_train, a_abl, r_abl, gamma)
        A_abl, B_abl, F_abl = assemble_system(
            xi_abl, grad_xi_abl, S, f_train, zeta_train)

        xi_test_abl = eval_features(x_test, a_abl, r_abl, gamma)
        psi_test_abl = zeta_test.unsqueeze(1) * xi_test_abl

        # Run all 4 methods; eval_every=K_max to only record final L2
        abl_results = run_all_algorithms(
            A_abl, B_abl, F_abl, psi_test_abl, xi_test_abl,
            u_exact, sigma_exact, K_max=K_max, eval_every=K_max, rho=rho,
            eta_admm=eta_admm, beta_adam=beta_adam,
            eta_u_uzawa=eta_u_uzawa, eta_s_ah=eta_s_ah, eta_u_ah=eta_u_ah,
        )

        ablation_M_results[M_abl] = {}
        for method_name in ["Direct", "ADMM", "Uzawa", "Arrow-Hurwicz"]:
            h = abl_results[method_name]["history"]
            rs_f = h["r_s"][-1]
            ru_f = h["r_u"][-1]
            rel_u_f = h["rel_u"][-1]
            rel_sig_f = h["rel_sigma"][-1]
            ablation_M_results[M_abl][method_name] = {
                "rel_u": rel_u_f,
                "rel_sigma": rel_sig_f,
                "kkt_total": rs_f + ru_f,
            }

    plot_ablation_M(ablation_M_results, str(OUTPUT_DIR / "ablation-M.png"))

    # --- Print ablation summary table ---
    print("\n=== Ablation M: Summary ===")
    print(f"{'M':>5} {'Algorithm':<16} {'u_err':>12} {'sig_err':>12}")
    print("-" * 48)
    for M_abl in ablation_M_list:
        for method in ["Direct", "ADMM", "Uzawa", "Arrow-Hurwicz"]:
            d = ablation_M_results[M_abl][method]
            print(f"{M_abl:5d} {method:<16} {d['rel_u']:12.2e} {d['rel_sigma']:12.2e}")
