from __future__ import annotations

import gc
import math
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from linear_elasticity_3d import (
    ALGO_STYLE,
    BASE_SEED,
    DEVICE,
    DTYPE,
    STRAIN_GRAD_BASES,
    AlgorithmResult,
    LeastSquaresConfig,
    SharedBenchmarkData,
    SharedFeatureSpace,
    build_compliance_matrix,
    build_shared_benchmark,
    build_shared_feature_space,
    clear_cuda_cache,
    configure_plotting,
    eval_active_displacement_basis_data,
    eval_raw_scalar_basis,
    eval_raw_scalar_basis_grads,
    print_aligned_markdown_table,
    run_experiment,
    synchronize_device,
)


METHOD_LABELS = {
    "rfm": "RFM + LS (Lstsq)",
    "pd-fm": "PD-FM + LS (Lstsq)",
    "pd-ofm": "PD-OFM + LS (Lstsq)",
}
METHOD_STYLES = {
    "RFM + LS (Lstsq)": {"color": "#264653", "marker": "s", "linestyle": "--"},
    "PD-FM + LS (Lstsq)": {"color": "#2A9D8F", "marker": "o", "linestyle": "-"},
    "PD-OFM + LS (Lstsq)": {"color": "#E76F51", "marker": "D", "linestyle": "-."},
}


@dataclass
class PdOfmAblationConfig:
    """Configuration for the 3D PD-OFM comparison experiment."""

    base_ls_cfg: LeastSquaresConfig = field(default_factory=LeastSquaresConfig)
    epochs: int = 200
    train_batch_size: int = 4_096
    lr: float = 1.0e-3
    report_every: int = 25
    lambda_c: float = 1.0
    lambda_e: float = 1.0
    lambda_orth: float = 1.0
    seed: int = BASE_SEED


@dataclass
class PretrainHistory:
    """Tracked pretraining losses reported at console checkpoints."""

    epochs: list[int] = field(default_factory=list)
    loss_total: list[float] = field(default_factory=list)
    loss_constitutive: list[float] = field(default_factory=list)
    loss_equilibrium: list[float] = field(default_factory=list)
    loss_orth: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class ComparisonRecord:
    """One completed feature-generation method compared downstream by least-squares."""

    method_id: str
    method_name: str
    pretrain_time: float
    solve_time: float
    total_time: float
    ls_result: AlgorithmResult


class SplitFeaturePretrainer(nn.Module):
    """Learned random-feature generator with disposable output coefficients."""

    def __init__(
        self,
        feature_space: SharedFeatureSpace,
        seed: int,
    ) -> None:
        super().__init__()
        self.gamma_s = feature_space.gamma_s
        self.gamma_u = feature_space.gamma_u

        a_s_init = feature_space.a_s.detach().clone().to(device=DEVICE, dtype=DTYPE)
        a_u_init = feature_space.a_u.detach().clone().to(device=DEVICE, dtype=DTYPE)
        r_s_init = feature_space.r_s.detach().clone().to(device=DEVICE, dtype=DTYPE)
        r_u_init = feature_space.r_u.detach().clone().to(device=DEVICE, dtype=DTYPE)

        self.a_s_raw = nn.Parameter(a_s_init)
        self.a_u_raw = nn.Parameter(a_u_init)
        self.r_s_raw = nn.Parameter(logit_parameter(r_s_init))
        self.r_u_raw = nn.Parameter(logit_parameter(r_u_init))

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        sigma_scale = 1.0 / math.sqrt(a_s_init.shape[0] + 1)
        u_scale = 1.0 / math.sqrt(a_u_init.shape[0] + 1)
        self.sigma_coeffs = nn.Parameter(
            sigma_scale
            * torch.randn(
                a_s_init.shape[0] + 1,
                6,
                generator=generator,
                dtype=DTYPE,
            ).to(DEVICE)
        )
        self.u_coeffs = nn.Parameter(
            u_scale
            * torch.randn(
                a_u_init.shape[0] + 1,
                3,
                generator=generator,
                dtype=DTYPE,
            ).to(DEVICE)
        )

    def normalized_feature_space(self) -> SharedFeatureSpace:
        """Export the currently learned random feature space."""

        return SharedFeatureSpace(
            a_s=normalize_rows(self.a_s_raw.detach()),
            r_s=torch.sigmoid(self.r_s_raw.detach()),
            a_u=normalize_rows(self.a_u_raw.detach()),
            r_u=torch.sigmoid(self.r_u_raw.detach()),
            gamma_s=self.gamma_s,
            gamma_u=self.gamma_u,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Evaluate the learned feature model and its PDE residual ingredients."""

        a_s = normalize_rows(self.a_s_raw)
        a_u = normalize_rows(self.a_u_raw)
        r_s = torch.sigmoid(self.r_s_raw)
        r_u = torch.sigmoid(self.r_u_raw)

        raw_sigma_basis = eval_raw_scalar_basis(x, a_s, r_s, self.gamma_s)
        raw_sigma_basis_grad = eval_raw_scalar_basis_grads(x, a_s, r_s, self.gamma_s)
        active_u_basis, active_u_basis_grad = eval_active_displacement_basis_data(
            x,
            a_u,
            r_u,
            self.gamma_u,
        )

        sigma = raw_sigma_basis @ self.sigma_coeffs
        du_dx = [
            active_u_basis_grad[:, :, dim_i] @ self.u_coeffs for dim_i in range(3)
        ]
        eps = sum(
            du_dx[dim_i] @ STRAIN_GRAD_BASES[dim_i].T for dim_i in range(3)
        )

        ds_dx = [
            raw_sigma_basis_grad[:, :, dim_i] @ self.sigma_coeffs for dim_i in range(3)
        ]
        div_sigma = torch.stack(
            [
                ds_dx[0][:, 0] + ds_dx[1][:, 3] + ds_dx[2][:, 5],
                ds_dx[0][:, 3] + ds_dx[1][:, 1] + ds_dx[2][:, 4],
                ds_dx[0][:, 5] + ds_dx[1][:, 4] + ds_dx[2][:, 2],
            ],
            dim=1,
        )

        return {
            "raw_sigma_basis": raw_sigma_basis,
            "active_u_basis": active_u_basis,
            "sigma": sigma,
            "eps": eps,
            "div_sigma": div_sigma,
        }


def logit_parameter(r: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Map bounded offsets from (0, 1) back to the unconstrained line."""

    clipped = r.clamp(min=eps, max=1.0 - eps)
    return torch.log(clipped) - torch.log1p(-clipped)


def normalize_rows(a_raw: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    """Row-normalize feature normals with a small safety floor."""

    return a_raw / a_raw.norm(dim=1, keepdim=True).clamp_min(eps)


def clone_feature_space(feature_space: SharedFeatureSpace) -> SharedFeatureSpace:
    """Deep-clone one shared random feature space."""

    return SharedFeatureSpace(
        a_s=feature_space.a_s.detach().clone(),
        r_s=feature_space.r_s.detach().clone(),
        a_u=feature_space.a_u.detach().clone(),
        r_u=feature_space.r_u.detach().clone(),
        gamma_s=feature_space.gamma_s,
        gamma_u=feature_space.gamma_u,
    )


def clear_experiment_memory() -> None:
    """Release references and cached CUDA memory between major runs."""

    gc.collect()
    clear_cuda_cache()


def validate_config(cfg: PdOfmAblationConfig) -> None:
    """Validate the PD-OFM ablation configuration."""

    if cfg.epochs <= 0:
        raise ValueError("Config.epochs must be positive.")
    if cfg.train_batch_size <= 0:
        raise ValueError("Config.train_batch_size must be positive.")
    if not math.isfinite(cfg.lr) or cfg.lr <= 0.0:
        raise ValueError("Config.lr must be finite and positive.")
    if cfg.report_every <= 0:
        raise ValueError("Config.report_every must be positive.")
    if not math.isfinite(cfg.lambda_c) or cfg.lambda_c <= 0.0:
        raise ValueError("Config.lambda_c must be finite and positive.")
    if not math.isfinite(cfg.lambda_e) or cfg.lambda_e <= 0.0:
        raise ValueError("Config.lambda_e must be finite and positive.")
    if not math.isfinite(cfg.lambda_orth) or cfg.lambda_orth < 0.0:
        raise ValueError("Config.lambda_orth must be finite and non-negative.")
    if cfg.base_ls_cfg.N_s <= 0 or cfg.base_ls_cfg.N_u <= 0:
        raise ValueError("Least-squares feature counts must be positive.")


def weighted_orthogonality_loss(
    feature_matrix: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """Measure how far one learned feature matrix is from weighted orthonormality."""

    if feature_matrix.shape[1] == 0:
        return torch.zeros((), dtype=DTYPE, device=DEVICE)

    sqrt_w = torch.sqrt(weights).unsqueeze(1)
    weighted = sqrt_w * feature_matrix
    col_norm = torch.linalg.norm(weighted, dim=0).clamp_min(eps)
    normalized = weighted / col_norm.unsqueeze(0)
    gram = normalized.T @ normalized
    identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    return (gram - identity).square().sum() / (gram.shape[0] ** 2)


def compute_pretrain_losses(
    model: SplitFeaturePretrainer,
    xb: torch.Tensor,
    wb: torch.Tensor,
    fb: torch.Tensor,
    compliance_voigt: torch.Tensor,
    lambda_c: float,
    lambda_e: float,
    lambda_orth: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Build the feature-pretraining loss on one interior mini-batch."""

    normalized_w = wb / wb.sum().clamp_min(1.0e-12)
    outputs = model(xb)

    r_c = outputs["sigma"] @ compliance_voigt.T - outputs["eps"]
    r_e = outputs["div_sigma"] + fb
    constitutive_loss = (normalized_w * r_c.square().sum(dim=1)).sum()
    equilibrium_loss = (normalized_w * r_e.square().sum(dim=1)).sum()

    orth_sigma = weighted_orthogonality_loss(
        outputs["raw_sigma_basis"][:, 1:],
        normalized_w,
    )
    orth_u = weighted_orthogonality_loss(
        outputs["active_u_basis"][:, 1:],
        normalized_w,
    )
    orth_loss = orth_sigma + orth_u

    total_loss = (
        lambda_c * constitutive_loss
        + lambda_e * equilibrium_loss
        + lambda_orth * orth_loss
    )
    metrics = {
        "loss_total": float(total_loss.detach().item()),
        "loss_constitutive": float(constitutive_loss.detach().item()),
        "loss_equilibrium": float(equilibrium_loss.detach().item()),
        "loss_orth": float(orth_loss.detach().item()),
    }
    return total_loss, metrics


def append_history(
    history: PretrainHistory,
    epoch: int,
    epoch_metrics: dict[str, float],
) -> None:
    """Append one report point into the tracked pretraining history."""

    history.epochs.append(epoch)
    history.loss_total.append(epoch_metrics["loss_total"])
    history.loss_constitutive.append(epoch_metrics["loss_constitutive"])
    history.loss_equilibrium.append(epoch_metrics["loss_equilibrium"])
    history.loss_orth.append(epoch_metrics["loss_orth"])


def print_pretrain_progress(
    method_name: str,
    epoch: int,
    total_epochs: int,
    epoch_metrics: dict[str, float],
) -> None:
    """Print one compact pretraining progress line."""

    print(
        f"    [{method_name}] epoch {epoch:>4d}/{total_epochs}, "
        f"loss={epoch_metrics['loss_total']:.4e}, "
        f"Lc={epoch_metrics['loss_constitutive']:.4e}, "
        f"Le={epoch_metrics['loss_equilibrium']:.4e}, "
        f"Lorth={epoch_metrics['loss_orth']:.4e}"
    )


def train_feature_space(
    method_id: str,
    cfg: PdOfmAblationConfig,
    benchmark: SharedBenchmarkData,
    initial_feature_space: SharedFeatureSpace,
) -> tuple[SharedFeatureSpace, float, PretrainHistory]:
    """Pretrain one learned feature space and export it back to least-squares."""

    base_ls_cfg = cfg.base_ls_cfg
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    model = SplitFeaturePretrainer(
        initial_feature_space,
        seed=cfg.seed + (17 if method_id == "pd-ofm" else 11),
    ).to(device=DEVICE, dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history = PretrainHistory()

    mu = base_ls_cfg.E / (2.0 * (1.0 + base_ls_cfg.nu))
    lam = base_ls_cfg.E * base_ls_cfg.nu / (
        (1.0 + base_ls_cfg.nu) * (1.0 - 2.0 * base_ls_cfg.nu)
    )
    compliance_voigt = build_compliance_matrix(mu, lam)
    lambda_orth = cfg.lambda_orth if method_id == "pd-ofm" else 0.0

    x_int = benchmark.x_int
    w_int = benchmark.w_int
    f_int = benchmark.f_int
    n_points = x_int.shape[0]

    synchronize_device()
    t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        permutation = torch.randperm(n_points, device=x_int.device)
        epoch_metrics = {
            "loss_total": 0.0,
            "loss_constitutive": 0.0,
            "loss_equilibrium": 0.0,
            "loss_orth": 0.0,
        }

        for start in range(0, n_points, cfg.train_batch_size):
            end = min(start + cfg.train_batch_size, n_points)
            batch_idx = permutation[start:end]
            xb = x_int[batch_idx]
            wb = w_int[batch_idx]
            fb = f_int[batch_idx]

            optimizer.zero_grad(set_to_none=True)
            loss, metrics = compute_pretrain_losses(
                model,
                xb,
                wb,
                fb,
                compliance_voigt,
                lambda_c=cfg.lambda_c,
                lambda_e=cfg.lambda_e,
                lambda_orth=lambda_orth,
            )
            loss.backward()
            optimizer.step()

            batch_fraction = (end - start) / n_points
            for key in epoch_metrics:
                epoch_metrics[key] += batch_fraction * metrics[key]

        if epoch == 1 or epoch % cfg.report_every == 0 or epoch == cfg.epochs:
            append_history(history, epoch, epoch_metrics)
            print_pretrain_progress(
                METHOD_LABELS[method_id],
                epoch,
                cfg.epochs,
                epoch_metrics,
            )

    synchronize_device()
    learned_feature_space = model.normalized_feature_space()
    elapsed = time.perf_counter() - t0
    del model
    clear_experiment_memory()
    return learned_feature_space, elapsed, history


def run_lstsq_with_feature_space(
    cfg: LeastSquaresConfig,
    benchmark: SharedBenchmarkData,
    feature_space: SharedFeatureSpace,
) -> AlgorithmResult:
    """Run the existing least-squares solver stack with one fixed feature space."""

    results = run_experiment(
        cfg=replace(cfg, algorithms_to_run=["lstsq"]),
        print_table=False,
        plot_results=False,
        benchmark=benchmark,
        feature_space=feature_space,
    )
    if len(results) != 1:
        raise RuntimeError("Expected exactly one least-squares result.")
    return results[0]


def print_comparison_table(records: list[ComparisonRecord]) -> None:
    """Print the PD-OFM comparison summary."""

    headers = (
        "Method",
        "‖Φ^u-u‖",
        "‖Φ^σ-σ‖",
        "Pretrain(s)",
        "Solve(s)",
        "Total(s)",
    )
    rows = [
        (
            record.method_name,
            f"{record.ls_result.u_l2_error:.2e}",
            f"{record.ls_result.sigma_l2_error:.2e}",
            f"{record.pretrain_time:.2f}",
            f"{record.solve_time:.2f}",
            f"{record.total_time:.2f}",
        )
        for record in records
    ]
    print_aligned_markdown_table(
        title="3D PD-OFM Comparison",
        headers=headers,
        rows=rows,
        alignments=("left", "center", "center", "center", "center", "center"),
    )


def plot_comparison(records: list[ComparisonRecord], save_path: str) -> None:
    """Plot the final least-squares displacement and stress metrics for each method."""

    if not records:
        print(f"  Skipped: {save_path} (no records to plot)")
        return

    configure_plotting()
    labels = [record.method_name for record in records]
    x_positions = np.arange(len(records), dtype=float)
    colors = [
        METHOD_STYLES.get(label, ALGO_STYLE.get(label, {"color": "#4C78A8"}))["color"]
        for label in labels
    ]
    metric_specs = [
        ("u_l2_error", r"$\|\Phi^u - u_{ex}\|_0$"),
        ("sigma_l2_error", r"$\|\Phi^\sigma - \sigma_{ex}\|_0$"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, (metric_name, title) in zip(axes, metric_specs):
        values = np.array(
            [getattr(record.ls_result, metric_name) for record in records],
            dtype=float,
        )
        valid = np.isfinite(values) & (values > 0.0)
        if valid.any():
            ax.bar(
                x_positions[valid],
                values[valid],
                width=0.65,
                color=[colors[index] for index in np.flatnonzero(valid)],
            )
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel("$L^2$ error")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(alpha=0.3, linestyle="--", axis="y")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_history(
    histories: dict[str, PretrainHistory],
    save_path: str,
) -> None:
    """Plot pretraining loss histories for PD-FM and PD-OFM."""

    if not histories:
        print(f"  Skipped: {save_path} (no histories to plot)")
        return

    configure_plotting()
    metric_specs = [
        ("loss_total", "Total loss"),
        ("loss_constitutive", r"$L_c$"),
        ("loss_equilibrium", r"$L_e$"),
        ("loss_orth", r"$L_{orth}$"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))

    for ax, (metric_name, title) in zip(axes.flat, metric_specs):
        has_line = False
        for method_id in ("pd-fm", "pd-ofm"):
            history = histories.get(method_id)
            if history is None or not history.epochs:
                continue
            epochs = np.array(history.epochs, dtype=float)
            values = np.array(getattr(history, metric_name), dtype=float)
            valid = np.isfinite(epochs) & np.isfinite(values) & (values > 0.0)
            if not valid.any():
                continue

            label = METHOD_LABELS[method_id]
            style = METHOD_STYLES[label]
            ax.semilogy(
                epochs[valid],
                values[valid],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=1.5,
                markersize=5,
                label=label,
            )
            has_line = True

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3, linestyle="--")
        if has_line:
            ax.legend()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def run_pd_ofm_ablation(
    cfg: PdOfmAblationConfig | None = None,
) -> list[ComparisonRecord]:
    """Run the fixed-3D RFM / PD-FM / PD-OFM comparison."""

    cfg = PdOfmAblationConfig() if cfg is None else cfg
    validate_config(cfg)
    base_ls_cfg = cfg.base_ls_cfg

    ablation_output_dir = (
        Path(__file__).resolve().parents[3]
        / "public"
        / "images"
        / "least-squares"
        / "linear-elasticity-3d"
        / "ablation"
        / "pd-ofm"
    )
    ablation_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Output: {ablation_output_dir}")
    print(
        f"Config: N_s={base_ls_cfg.N_s}, N_u={base_ls_cfg.N_u}, "
        f"Q_train={base_ls_cfg.Q_train}, Q_test={base_ls_cfg.Q_test}, "
        f"gamma_s={base_ls_cfg.gamma_s}, gamma_u={base_ls_cfg.gamma_u}, "
        f"sampling={base_ls_cfg.sampling_method}, "
        f"manufactured_solution={base_ls_cfg.manufactured_solution}, "
        f"epochs={cfg.epochs}, train_batch_size={cfg.train_batch_size}, "
        f"lr={cfg.lr:.2e}, lambda_orth={cfg.lambda_orth:.2e}"
    )

    print("Building shared benchmark data...")
    benchmark = build_shared_benchmark(
        E=base_ls_cfg.E,
        nu=base_ls_cfg.nu,
        Q_train=base_ls_cfg.Q_train,
        Q_test=base_ls_cfg.Q_test,
        sampling_method=base_ls_cfg.sampling_method,
        body_force_batch_size=base_ls_cfg.body_force_batch_size,
        manufactured_solution=base_ls_cfg.manufactured_solution,
    )

    print("Building shared initial random feature space...")
    initial_feature_space = build_shared_feature_space(
        N_s=base_ls_cfg.N_s,
        N_u=base_ls_cfg.N_u,
        gamma_s=base_ls_cfg.gamma_s,
        gamma_u=base_ls_cfg.gamma_u,
    )

    records: list[ComparisonRecord] = []
    histories: dict[str, PretrainHistory] = {}

    print("\n=== RFM baseline ===")
    rfm_feature_space = clone_feature_space(initial_feature_space)
    rfm_result = run_lstsq_with_feature_space(base_ls_cfg, benchmark, rfm_feature_space)
    records.append(
        ComparisonRecord(
            method_id="rfm",
            method_name=METHOD_LABELS["rfm"],
            pretrain_time=0.0,
            solve_time=rfm_result.wall_time,
            total_time=rfm_result.wall_time,
            ls_result=AlgorithmResult(
                name=METHOD_LABELS["rfm"],
                u_l2_error=rfm_result.u_l2_error,
                sigma_l2_error=rfm_result.sigma_l2_error,
                wall_time=rfm_result.wall_time,
            ),
        )
    )
    clear_experiment_memory()

    for method_id in ("pd-fm", "pd-ofm"):
        print(f"\n=== {METHOD_LABELS[method_id]} ===")
        learned_feature_space, pretrain_time, history = train_feature_space(
            method_id,
            cfg,
            benchmark,
            clone_feature_space(initial_feature_space),
        )
        histories[method_id] = history
        ls_result = run_lstsq_with_feature_space(
            base_ls_cfg,
            benchmark,
            learned_feature_space,
        )
        total_time = pretrain_time + ls_result.wall_time
        records.append(
            ComparisonRecord(
                method_id=method_id,
                method_name=METHOD_LABELS[method_id],
                pretrain_time=pretrain_time,
                solve_time=ls_result.wall_time,
                total_time=total_time,
                ls_result=AlgorithmResult(
                    name=METHOD_LABELS[method_id],
                    u_l2_error=ls_result.u_l2_error,
                    sigma_l2_error=ls_result.sigma_l2_error,
                    wall_time=ls_result.wall_time,
                ),
            )
        )
        clear_experiment_memory()

    plot_comparison(records, str(ablation_output_dir / "pd-ofm-comparison.png"))
    plot_training_history(histories, str(ablation_output_dir / "pd-ofm-training-history.png"))
    print_comparison_table(records)
    return records


def main(cfg: PdOfmAblationConfig | None = None) -> None:
    """Script entrypoint."""

    run_pd_ofm_ablation(cfg)


if __name__ == "__main__":
    main()
