"""
TransNet 求解 2D Poisson 方程

实现论文 "Transferable Neural Networks for Partial Differential Equations" 核心算法，
在 [-1,1]^2 区域上求解 -Δu = f，精确解 u(x,y) = sin(πx)sin(πy)。

将 trans.ipynb 转换而来，所有图表输出到项目的
assets/images/神经网络求解PDE入门/ 目录。
"""

import math
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "apublic" / "images" / "intro-to-neural-pde"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 设备与随机种子
# ---------------------------------------------------------------------------
BASE_SEED = 42

np.random.seed(BASE_SEED)
torch.manual_seed(BASE_SEED)
torch.cuda.manual_seed_all(BASE_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# TransNet 类
# ---------------------------------------------------------------------------
class TransNet(nn.Module):
    def __init__(self, a: torch.Tensor, r: torch.Tensor, gamma: float):
        super(TransNet, self).__init__()
        self.width, self.input_dim = a.size()

        dtype = torch.get_default_dtype()
        gamma_tensor = torch.as_tensor(float(gamma), dtype=dtype)

        self.register_buffer('a_const', a)
        self.register_buffer('r_const', r)
        self.register_buffer('gamma_const', gamma_tensor)

        self.out_layer = nn.Linear(self.width, 1)

    def forward(self, x):
        z = self.gamma_const * (torch.matmul(x, self.a_const.T) + self.r_const.T)
        return self.out_layer(torch.tanh(z))

    def get_weights(self):
        w = self.out_layer.weight.detach().reshape(-1)
        b = self.out_layer.bias.detach().reshape(-1)
        return torch.cat([w, b], dim=0)

    def set_weights(self, weight: torch.Tensor, bias: torch.Tensor):
        with torch.no_grad():
            self.out_layer.weight.copy_(weight)
            self.out_layer.bias.copy_(bias)


# ---------------------------------------------------------------------------
# 神经元生成与辅助采样
# ---------------------------------------------------------------------------
def generate_neurons(input_dim: int, width: int, radius: int = 1):
    dtype = torch.get_default_dtype()
    gaussian_samples = torch.randn(width, input_dim, dtype=dtype)
    norm = torch.norm(gaussian_samples, dim=1, keepdim=True).clamp_min(1e-12)
    locations = gaussian_samples / norm
    offsets = torch.rand(width, 1, dtype=dtype) * radius
    return locations, offsets


def sample_unit_ball(num_points: int, dim: int,
                     device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    direction = torch.randn(num_points, dim, device=device, dtype=dtype)
    direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
    radii = torch.rand(num_points, 1, device=device, dtype=dtype).pow(1.0 / dim)
    return direction * radii


def sample_gaussian_random_field(points: torch.Tensor,
                                 cor_len: float,
                                 num_features: int,
                                 sigma: float) -> torch.Tensor:
    weight = torch.randn(points.size(1), num_features,
                         device=points.device, dtype=points.dtype) / cor_len
    phase = 2 * math.pi * torch.rand(1, num_features,
                                      device=points.device, dtype=points.dtype)
    proj = torch.matmul(points, weight) + phase
    scale = sigma * math.sqrt(2.0 / num_features)
    return scale * torch.cos(proj).sum(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# gamma 搜索
# ---------------------------------------------------------------------------
def tune_gamma(a_const: torch.Tensor,
               r_const: torch.Tensor,
               cor_len: float,
               num_grfs: int = 50,
               gamma_bounds: tuple[float, float] = (0.2, 10.0),
               grid_size: int = 50,
               num_samples: Optional[int] = None,
               num_rff: int = 1024,
               sigma: float = 1.0,
               save_path: Optional[str] = None) -> float:
    """
    用 GRF 辅助函数离线搜索最优 gamma。

    Args:
        save_path: 若非 None，将误差曲线保存到该路径（替代 plt.show）。
    """
    num_neurons, input_dim = a_const.size()
    _device = a_const.device
    dtype = a_const.dtype
    if num_samples is None:
        base = 50 ** min(input_dim, 3)
        num_samples = max(int(base), num_neurons + 1)

    sample_points = sample_unit_ball(num_samples, input_dim, _device, dtype)
    pre_activation = torch.matmul(sample_points, a_const.T) + r_const.T

    gamma_values = torch.linspace(gamma_bounds[0], gamma_bounds[1],
                                  steps=grid_size, device=_device, dtype=dtype)
    mse = torch.zeros_like(gamma_values)
    ones = torch.ones(sample_points.size(0), 1, device=_device, dtype=dtype)

    for _ in range(num_grfs):
        targets = sample_gaussian_random_field(sample_points, cor_len, num_rff, sigma)
        for idx, gamma in enumerate(gamma_values):
            hidden = torch.tanh(gamma * pre_activation)
            features = torch.cat([ones, hidden], dim=1)
            coeffs = torch.linalg.lstsq(features, targets).solution
            residual = features @ coeffs - targets
            mse[idx] += torch.mean(residual.pow(2))

    mse /= num_grfs
    best_gamma = gamma_values[torch.argmin(mse)]
    best_gamma_value = float(best_gamma.item())

    if save_path is not None:
        gamma_cpu = gamma_values.detach().cpu().numpy()
        mse_cpu = mse.detach().cpu().numpy()
        plt.figure(figsize=(6, 4))
        plt.plot(gamma_cpu, mse_cpu, label='均方误差')
        plt.axvline(best_gamma_value, color='r', linestyle='--',
                    label=f'最优 gamma = {best_gamma_value:.2f}')
        plt.xlabel('gamma')
        plt.ylabel('均方误差')
        plt.title('gamma 搜索与拟合误差')
        plt.legend()
        plt.grid(alpha=0.3, linestyle='--')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    return best_gamma_value


# ---------------------------------------------------------------------------
# PDE 数据采样
# ---------------------------------------------------------------------------
def exact_solution(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(math.pi * x[:, 0]) * torch.sin(math.pi * x[:, 1])


def forcing_term(x: torch.Tensor) -> torch.Tensor:
    return 2.0 * (math.pi ** 2) * exact_solution(x)


def sample_domain(num_per_dim: int = 50) -> torch.Tensor:
    lin = torch.linspace(-1.0, 1.0, num_per_dim)
    grid = torch.cartesian_prod(lin, lin)
    return grid


def sample_boundary(num_points: int = 200) -> torch.Tensor:
    points_per_edge = max(1, num_points // 4)
    param = torch.linspace(-1.0, 1.0, points_per_edge)
    left = torch.stack([torch.full_like(param, -1.0), param], dim=1)
    right = torch.stack([torch.full_like(param, 1.0), param], dim=1)
    bottom = torch.stack([param, torch.full_like(param, -1.0)], dim=1)
    top = torch.stack([param, torch.full_like(param, 1.0)], dim=1)
    boundary = torch.cat([left, right, bottom, top], dim=0)
    remainder = num_points - boundary.size(0)
    if remainder > 0:
        extra_param = torch.linspace(-1.0, 1.0, remainder + 2)[1:-1]
        extra = torch.stack([extra_param, torch.full_like(extra_param, 1.0)], dim=1)
        boundary = torch.cat([boundary, extra], dim=0)
    return boundary.type(torch.get_default_dtype())


# ---------------------------------------------------------------------------
# 求解权重（最小二乘）
# ---------------------------------------------------------------------------
def solve_poisson_box(net: TransNet,
                      pde_points: torch.Tensor,
                      boundary_points: torch.Tensor,
                      f_pde: torch.Tensor,
                      g_boundary: torch.Tensor,
                      lambda_bc: float = 10.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    求解 box 区域的 Poisson 问题。

    Returns:
        weight: 输出层权重 (num_neurons,)
        bias: 输出层偏置 (1,)
    """
    num_pde_points = pde_points.size(0)
    num_boundary_points = boundary_points.size(0)
    num_neurons = net.width
    _device = pde_points.device

    # 计算 PDE 内部点的拉普拉斯矩阵
    laplacian_matrix = torch.zeros(num_pde_points, num_neurons + 1,
                                   device=_device, dtype=pde_points.dtype)
    laplacian_matrix[:, 0] = 0.0

    with torch.no_grad():
        z = torch.matmul(pde_points, net.a_const.T) + net.r_const.T
        hidden = torch.tanh(net.gamma_const * z)
        laplacian_matrix[:, 1:] = (-2
                                   * torch.norm(net.gamma_const * net.a_const, dim=1) ** 2
                                   * hidden * (1 - hidden ** 2))

    # 计算边界点的特征矩阵
    with torch.no_grad():
        z_boundary = torch.matmul(boundary_points, net.a_const.T) + net.r_const.T
        hidden_boundary = torch.tanh(net.gamma_const * z_boundary)
        ones = torch.ones(num_boundary_points, 1, device=_device, dtype=boundary_points.dtype)
        boundary_features = torch.cat([ones, hidden_boundary], dim=1)

    # 组装最小二乘系统
    A = -laplacian_matrix
    b_pde = f_pde
    B = boundary_features
    b_bc = g_boundary

    sqrt_lambda = torch.sqrt(torch.tensor(lambda_bc, device=A.device, dtype=A.dtype))
    A_combined = torch.cat([A, sqrt_lambda * B], dim=0)
    b_combined = torch.cat([b_pde, sqrt_lambda * b_bc], dim=0)

    solution = torch.linalg.lstsq(A_combined, b_combined, rcond=None).solution

    bias = solution[0:1].reshape(-1)
    weight = solution[1:].reshape(-1)
    return weight, bias


# ---------------------------------------------------------------------------
# 评估函数
# ---------------------------------------------------------------------------
def evaluate_poisson_box(net: TransNet,
                         num_per_dim: int = 100,
                         save_path: Optional[str] = None):
    """
    评估 TransNet 预测质量。

    Args:
        save_path: 若非 None，将 2×2 评估图保存到该路径。
    """
    num_neurons = net.width
    _device = net.get_weights().device

    x_vis = torch.linspace(-1.0, 1.0, num_per_dim)
    y_vis = torch.linspace(-1.0, 1.0, num_per_dim)
    X_grid, Y_grid = torch.meshgrid(x_vis, y_vis, indexing='ij')
    vis_points = torch.stack([X_grid.flatten(), Y_grid.flatten()], dim=1).to(_device)

    with torch.no_grad():
        u_exact_val = exact_solution(vis_points).cpu().numpy().reshape(num_per_dim, num_per_dim)
        u_pred = net(vis_points).squeeze().cpu().numpy().reshape(num_per_dim, num_per_dim)
        error = np.abs(u_exact_val - u_pred)

    # 计算 PDE 残差 |-Δu - f|
    residual_points = vis_points
    with torch.no_grad():
        z = torch.matmul(residual_points, net.a_const.T) + net.r_const.T
        hidden = torch.tanh(net.gamma_const * z)
        laplacian_u = (-2
                       * torch.norm(net.gamma_const * net.a_const, dim=1) ** 2
                       * hidden * (1 - hidden ** 2))
        laplacian_u = torch.matmul(laplacian_u, net.get_weights()[:-1])

    residual = torch.abs(
        -laplacian_u.detach() - forcing_term(residual_points).squeeze()
    ).cpu().numpy()

    if save_path is not None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        X_np = X_grid.cpu().numpy()
        Y_np = Y_grid.cpu().numpy()

        # 1. 精确解
        im1 = axes[0, 0].contourf(X_np, Y_np, u_exact_val, levels=50, cmap='viridis')
        axes[0, 0].set_title('精确解 $u(x,y) = \\sin(\\pi x)\\sin(\\pi y)$',
                             fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('$x$', fontsize=12)
        axes[0, 0].set_ylabel('$y$', fontsize=12)
        axes[0, 0].axis('equal')
        fig.colorbar(im1, ax=axes[0, 0])

        # 2. TransNet 预测解
        im2 = axes[0, 1].contourf(X_np, Y_np, u_pred, levels=50, cmap='viridis')
        axes[0, 1].set_title(f'TransNet 预测解 ({num_neurons} 个神经元)',
                             fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('$x$', fontsize=12)
        axes[0, 1].set_ylabel('$y$', fontsize=12)
        axes[0, 1].axis('equal')
        fig.colorbar(im2, ax=axes[0, 1])

        # 3. 绝对误差
        im3 = axes[1, 0].contourf(X_np, Y_np, error, levels=50, cmap='hot')
        axes[1, 0].set_title('绝对误差 $|u_{exact} - u_{pred}|$',
                             fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('$x$', fontsize=12)
        axes[1, 0].set_ylabel('$y$', fontsize=12)
        axes[1, 0].axis('equal')
        fig.colorbar(im3, ax=axes[1, 0])

        # 4. PDE 残差
        residual_points_np = residual_points.cpu().numpy()
        sc = axes[1, 1].scatter(residual_points_np[:, 0], residual_points_np[:, 1],
                                c=residual, s=10, cmap='hot', vmin=0, vmax=residual.max())
        axes[1, 1].set_title('PDE 残差 $|-\\Delta u - f|$',
                             fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('$x$', fontsize=12)
        axes[1, 1].set_ylabel('$y$', fontsize=12)
        axes[1, 1].axis('equal')
        axes[1, 1].set_xlim(-1, 1)
        axes[1, 1].set_ylim(-1, 1)
        fig.colorbar(sc, ax=axes[1, 1])

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    return error, residual


# ===========================================================================
# 主程序
# ===========================================================================
if __name__ == "__main__":
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

    print(f"Using device: {device}")
    print(f"图片输出目录: {OUTPUT_DIR}")

    # ------------------------------------------------------------------
    # 数据采样
    # ------------------------------------------------------------------
    pde_points = sample_domain().to(device)
    boundary_points = sample_boundary().to(device)

    f_pde = forcing_term(pde_points).unsqueeze(1).to(device)
    g_boundary = exact_solution(boundary_points).unsqueeze(1).to(device)

    print(f"方程采样: {pde_points.shape[0]}, 边界采样: {boundary_points.shape[0]}")

    # ------------------------------------------------------------------
    # 构建 TransNet 并搜索最优 gamma
    # ------------------------------------------------------------------
    num_neurons = 500
    radius = 1.5
    a, r = generate_neurons(2, num_neurons, radius)
    gamma = tune_gamma(a, r, 2,
                       save_path=str(OUTPUT_DIR / "trans-gamma-search.png"))

    transnet = TransNet(a, r, gamma).to(device)
    print(f"TransNet 隐藏层神经元数: {num_neurons}, gamma: {transnet.gamma_const.item():.2f}")

    # ------------------------------------------------------------------
    # 求解权重
    # ------------------------------------------------------------------
    lambda_bc = 1.0
    weight_opt, bias_opt = solve_poisson_box(
        transnet, pde_points, boundary_points, f_pde, g_boundary, lambda_bc=lambda_bc,
    )
    transnet.set_weights(weight_opt, bias_opt)

    print(f"权重范围: [{weight_opt.min().item():.4f}, {weight_opt.max().item():.4f}]")
    print(f"偏置值: {bias_opt.item():.4f}")

    # ------------------------------------------------------------------
    # 评估并输出图片
    # ------------------------------------------------------------------
    error, residual = evaluate_poisson_box(
        transnet, 100,
        save_path=str(OUTPUT_DIR / "trans-evaluate.png"),
    )

    print(f"L∞ 误差: {error.max():.6e}")
    print(f"L2 误差: {np.sqrt(np.mean(error**2)):.6e}")
    print(f"最大残差: {residual.max():.6e}")
    print(f"平均残差: {residual.mean():.6e}")
    print(f"\n图片已保存至: {OUTPUT_DIR}")
