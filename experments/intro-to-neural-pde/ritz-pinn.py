"""
深度学习方法求解一维 Poisson 方程
DeepRitzNet / PINNNet × Ritz / PINN 四种组合对比

将 ritz-pinn.ipynb 转换而来，所有图表输出到项目的
assets/images/神经网络求解PDE入门/ 目录。
"""

import math
import os
import time
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import torch as tc
import torch.nn as nn
from torch.utils import data
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
# 项目根目录：脚本位于 experments/intro-to-neural-pde/ritz-pinn.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "public" / "images" / "intro-to-neural-pde"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 设备与随机种子
# ---------------------------------------------------------------------------
device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
tc.set_default_dtype(tc.float32)

BASE_SEED = 42


def reset_random_seed(seed: int = BASE_SEED):
    """确保多次实验可复现"""
    tc.manual_seed(seed)
    npr.seed(seed)


# ---------------------------------------------------------------------------
# DeepRitzNet
# ---------------------------------------------------------------------------
class DeepRitzNet(nn.Module):
    """
    DeepRitzNet 网络类

    参数:
        dim_in (int): 输入维度
        dim_out (int): 输出维度
        depth (int): 网络深度（层数）
        width (int): 网络宽度（每层神经元数量）
        act_func (Callable): 激活函数
        scale (float, optional): 输入缩放因子
    """
    def __init__(self, dim_in, dim_out, depth, width, act_func: Callable, scale=None):
        super(DeepRitzNet, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.depth = depth
        self.width = width
        self.activation = act_func
        self.scale = scale if scale is not None else 1.0

        # Ix: 嵌入矩阵，形状 [dim_in, width]
        Ix = tc.zeros(dim_in, width, dtype=tc.get_default_dtype())
        for i in range(dim_in):
            if i < width:
                Ix[i, i] = 1.0
        self.register_buffer("Ix", Ix)

        # 构建网络层
        layers = []
        layers.append(nn.Linear(self.dim_in, self.width))
        for _ in range(1, self.depth, 2):
            layers.append(nn.Linear(self.width, self.width))
            layers.append(nn.Linear(self.width, self.width))
        self.layers = nn.ModuleList(layers)
        self.outlayer = nn.Linear(self.width, self.dim_out, bias=False)

    def forward(self, x):
        inp = self.scale * x
        s = inp @ self.Ix
        y = self.activation(self.layers[0](inp))
        y = y + s
        for i in range(1, self.depth, 2):
            s = y
            y = self.activation(self.layers[i](y))
            y = self.activation(self.layers[i + 1](y))
            y = y + s
        output = self.outlayer(y)
        return output


# ---------------------------------------------------------------------------
# PINNNet
# ---------------------------------------------------------------------------
class PINNNet(nn.Module):
    def __init__(self, dim_in, dim_out, depth, width, act_func: Callable, scale=None):
        super(PINNNet, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.depth = depth
        self.width = width
        self.activation = act_func
        self.scale = scale if scale is not None else 1.0

        Ix = tc.zeros(dim_in, width, dtype=tc.get_default_dtype())
        for i in range(dim_in):
            if i < width:
                Ix[i, i] = 1.0
        self.register_buffer("Ix", Ix)

        layers = []
        layers.append(nn.Linear(self.dim_in, self.width))
        for _ in range(1, self.depth):
            layers.append(nn.Linear(self.width, self.width))
        self.layers = nn.ModuleList(layers)
        self.outlayer = nn.Linear(self.width, self.dim_out, bias=False)

    def forward(self, x):
        inp = self.scale * x
        s = inp @ self.Ix
        y = self.activation(self.layers[0](inp))
        y = y + s
        for i in range(1, self.depth):
            s = y
            y = self.activation(self.layers[i](y))
            y = y + s
        output = self.outlayer(y)
        return output


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------
class CubeUniformPoints(Dataset):
    """在 d 维单位立方体内均匀随机采样"""
    def __init__(self, dim_in, batch_size):
        self.InnerPoints = npr.uniform(1e-16, 1.0, (batch_size, dim_in)).astype(np.float32)

    def __getitem__(self, index):
        return self.InnerPoints[index, :]

    def __len__(self):
        return self.InnerPoints.shape[0]


class GridPoints(Dataset):
    """规则网格点集，用于测试和可视化"""
    def __init__(self, dim_in, x1, x2, h):
        self.Vector = np.arange(x1, x2 + h / 2, h, dtype=np.float32)
        self.Points = self.Vector.reshape([self.Vector.shape[0], 1])
        self.Cell = self.Points
        for d in range(2, dim_in + 1):
            total_points = int(self.Vector.shape[0] ** d)
            self.Points = np.zeros([total_points, d], dtype=np.float32)
            for i in range(self.Vector.shape[0]):
                size = self.Vector.shape[0] ** (d - 1)
                start_idx = int(i * size)
                end_idx = int((i + 1) * size)
                self.Points[start_idx:end_idx, d - 1] = self.Vector[i]
                self.Points[start_idx:end_idx, 0:d - 1] = self.Cell
            self.Cell = self.Points

    def __getitem__(self, index):
        return self.Points[index, :]

    def __len__(self):
        return self.Points.shape[0]


# ---------------------------------------------------------------------------
# 解析解与源项
# ---------------------------------------------------------------------------
def u_exact(x):
    """解析解 u(x) = sin(πx) + sin(4πx)/4 - sin(8πx)/8 + sin(16πx)/16 + sin(24πx)/24"""
    if isinstance(x, np.ndarray):
        x = tc.tensor(x, dtype=tc.get_default_dtype(), device=device)
    pi = math.pi
    return (tc.sin(pi * x)
            + tc.sin(4 * pi * x) / 4.0
            - tc.sin(8 * pi * x) / 8.0
            + tc.sin(16 * pi * x) / 16.0
            + tc.sin(24 * pi * x) / 24.0)


def f_exact(x):
    """源项 f(x) = -Δu(x)"""
    if isinstance(x, np.ndarray):
        x = tc.tensor(x, dtype=tc.get_default_dtype(), device=device)
    pi = math.pi
    return ((pi ** 2) * tc.sin(pi * x)
            + (4.0 ** 2 * pi ** 2) * tc.sin(4 * pi * x) / 4.0
            - (8.0 ** 2 * pi ** 2) * tc.sin(8 * pi * x) / 8.0
            + (16.0 ** 2 * pi ** 2) * tc.sin(16 * pi * x) / 16.0
            + (24.0 ** 2 * pi ** 2) * tc.sin(24 * pi * x) / 24.0)


def f_func_for_training(points_tensor):
    """训练时使用的源项函数"""
    return f_exact(points_tensor)


# ---------------------------------------------------------------------------
# 训练函数
# ---------------------------------------------------------------------------
def Ritz_train(model, f_func, optimizer, scheduler, config_train):
    """Ritz 训练方法——基于变分原理的能量最小化"""
    model.train()
    dim_in = config_train['dim_in']
    N = config_train['N']
    epoch = config_train['epoch']
    loss_history = []

    for k in range(epoch):
        loss_value = None

        def closure():
            nonlocal loss_value
            optimizer.zero_grad()
            dataset = CubeUniformPoints(dim_in, N)
            loader = data.DataLoader(dataset=dataset, batch_size=N, shuffle=False,
                                     num_workers=0, drop_last=False)
            for _, points in enumerate(loader):
                points = points.to(device, dtype=tc.get_default_dtype()).requires_grad_(True)
                Dir_bd = tc.prod(points * (1.0 - points), dim=1, keepdim=True)
                u = Dir_bd * model(points)
                grad = tc.autograd.grad(u, points, tc.ones_like(u), create_graph=True)[0]
                f_val = f_func(points)
                loss = 0.5 * (grad ** 2).sum(dim=1, keepdim=True) - f_val * u
                loss = loss.mean()
                loss.backward()
                loss_value = loss.detach()
                return loss
            loss_value = tc.tensor(0.0, device=device)
            return loss_value

        optimizer.step(closure)
        if scheduler is not None:
            scheduler.step()

        current_loss = float(loss_value.item()) if loss_value is not None else float('nan')
        loss_history.append(current_loss)
        if (k + 1) % max(1, epoch // 10) == 0:
            print(f"[Ritz] epoch {k + 1}/{epoch} loss={current_loss:.4e}")

    return loss_history


def PINN_train(model, f_func, optimizer, scheduler, config_train):
    """PINN 训练方法——直接最小化 PDE 残差"""
    model.train()
    dim_in = config_train['dim_in']
    N = config_train['N']
    epoch = config_train['epoch']
    loss_history = []

    for k in range(epoch):
        loss_value = None

        def closure():
            nonlocal loss_value
            optimizer.zero_grad()
            dataset = CubeUniformPoints(dim_in, N)
            loader = data.DataLoader(dataset=dataset, batch_size=N, shuffle=False,
                                     num_workers=0, drop_last=False)
            for _, points in enumerate(loader):
                points = points.to(device, dtype=tc.get_default_dtype()).requires_grad_(True)
                Dir_bd = tc.prod(points * (1.0 - points), dim=1, keepdim=True)
                u = Dir_bd * model(points)
                grad = tc.autograd.grad(u, points, tc.ones_like(u), create_graph=True)[0]
                lap = tc.zeros_like(u)
                for i in range(points.shape[1]):
                    g_i = grad[:, i].reshape(-1, 1)
                    grad2 = tc.autograd.grad(g_i, points, tc.ones_like(g_i), create_graph=True)[0]
                    lap = lap + grad2[:, i].reshape(-1, 1)
                f_val = f_func(points)
                res = -lap - f_val
                loss = (res ** 2).mean()
                loss.backward()
                loss_value = loss.detach()
                return loss
            loss_value = tc.tensor(0.0, device=device)
            return loss_value

        optimizer.step(closure)
        if scheduler is not None:
            scheduler.step()

        current_loss = float(loss_value.item()) if loss_value is not None else float('nan')
        loss_history.append(current_loss)
        if (k + 1) % max(1, epoch // 10) == 0:
            print(f"[PINN] epoch {k + 1}/{epoch} loss={current_loss:.4e}")

    return loss_history


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def build_optimizer_and_scheduler(model: nn.Module):
    """为指定模型构建优化器与学习率调度器"""
    optimizer = tc.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-6)
    scheduler = tc.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)
    return optimizer, scheduler


def evaluate_model_on_grid(model: nn.Module, x_eval, u_exact_vals, config_train):
    """在网格上评估模型，返回预测值、误差和残差"""
    model.eval()
    x_tensor = tc.tensor(
        x_eval.reshape(-1, config_train['dim_in']),
        dtype=tc.get_default_dtype(),
        device=device,
        requires_grad=True,
    )
    with tc.enable_grad():
        Dir = tc.prod(x_tensor * (1.0 - x_tensor), dim=1, keepdim=True)
        u_tensor = Dir * model(x_tensor)
        grad = tc.autograd.grad(u_tensor, x_tensor, tc.ones_like(u_tensor), create_graph=True)[0]
        lap = tc.zeros_like(u_tensor)
        for i in range(x_tensor.shape[1]):
            g_i = grad[:, i].reshape(-1, 1)
            grad2 = tc.autograd.grad(g_i, x_tensor, tc.ones_like(g_i), create_graph=True)[0]
            lap = lap + grad2[:, i].reshape(-1, 1)
        residual = (-lap - f_func_for_training(x_tensor)).detach().cpu().numpy().squeeze()

    u_pred = u_tensor.detach().cpu().numpy().squeeze()
    error = np.abs(u_exact_vals - u_pred)
    max_error = float(np.max(error))
    mean_error = float(np.mean(error))
    l2_error = float(np.linalg.norm(error) / np.sqrt(error.size))

    return {
        'u_pred': u_pred,
        'error': error,
        'residual': residual,
        'max_error': max_error,
        'mean_error': mean_error,
        'l2_error': l2_error,
    }


# ===========================================================================
# 主程序
# ===========================================================================
if __name__ == "__main__":
    print(f"使用设备: {device}")
    print(f"图片输出目录: {OUTPUT_DIR}")

    reset_random_seed()

    # ------------------------------------------------------------------
    # 训练配置
    # ------------------------------------------------------------------
    config_train = {
        'dim_in': 1,
        'dim_out': 1,
        'N': 512,
        'epoch': 5000,
        'device': device,
    }

    net_depth = 6
    net_width = 64
    act_fun = tc.tanh

    model_configs = [
        {"name": "DeepRitzNet", "builder": DeepRitzNet},
        {"name": "PINNNet", "builder": PINNNet},
    ]
    method_configs = [
        {"name": "Ritz", "trainer": Ritz_train},
        {"name": "PINN", "trainer": PINN_train},
    ]

    print("=" * 60)
    print("配置信息")
    print("=" * 60)
    for key, value in config_train.items():
        print(f"{key}: {value}")

    # ------------------------------------------------------------------
    # 训练所有组合
    # ------------------------------------------------------------------
    experiment_results = []
    combo_counter = 0

    for model_cfg in model_configs:
        for method_cfg in method_configs:
            combo_counter += 1
            combo_seed = BASE_SEED + combo_counter
            reset_random_seed(combo_seed)

            print("=" * 60)
            print(f"开始训练: {model_cfg['name']} + {method_cfg['name']}")
            print(f"使用种子: {combo_seed}")
            print("=" * 60)

            model = model_cfg['builder'](
                dim_in=config_train['dim_in'],
                dim_out=config_train['dim_out'],
                depth=net_depth,
                width=net_width,
                act_func=act_fun,
            ).to(device)

            optimizer, scheduler = build_optimizer_and_scheduler(model)

            start_time = time.time()
            loss_history = method_cfg['trainer'](
                model,
                f_func=f_func_for_training,
                optimizer=optimizer,
                scheduler=scheduler,
                config_train=config_train,
            )
            end_time = time.time()

            experiment_results.append({
                'model_name': model_cfg['name'],
                'method_name': method_cfg['name'],
                'model': model,
                'training_time': end_time - start_time,
                'seed': combo_seed,
                'loss_history': loss_history,
            })

            print(f"组合 {model_cfg['name']} + {method_cfg['name']} 训练完成")
            print(f"训练用时: {end_time - start_time:.2f} 秒")

    # ------------------------------------------------------------------
    # 评估
    # ------------------------------------------------------------------
    x_eval = np.linspace(0.0, 1.0, 2048, endpoint=True).astype(np.float32)
    u_exact_vals = u_exact(x_eval).cpu().numpy().squeeze()

    print("=" * 60)
    print("评估所有组合")
    print("=" * 60)
    for exp in experiment_results:
        metrics = evaluate_model_on_grid(exp['model'], x_eval, u_exact_vals, config_train)
        exp.update(metrics)

    header = f"{'模型':<12}{'求解方法':<10}{'Max误差':>12}{'L2误差':>12}{'时间(s)':>10}"
    print(header)
    print("-" * len(header))
    for exp in experiment_results:
        print(f"{exp['model_name']:<12}{exp['method_name']:<10}"
              f"{exp['max_error']:>10.2e}{exp['l2_error']:>10.2e}"
              f"{exp['training_time']:>8.2f}")

    # ------------------------------------------------------------------
    # 可视化：设置中文字体
    # ------------------------------------------------------------------
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

    model_order = [cfg['name'] for cfg in model_configs]
    method_order = [cfg['name'] for cfg in method_configs]
    num_rows = len(model_order)
    num_cols = len(method_order)

    # --- 图1: 拟合曲线 ---
    fig, axes = plt.subplots(num_rows, num_cols,
                             figsize=(5 * num_cols, 3.5 * num_rows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    for exp in experiment_results:
        row = model_order.index(exp['model_name'])
        col = method_order.index(exp['method_name'])
        ax = axes[row, col]
        ax.plot(x_eval, u_exact_vals, label="解析解", lw=2, color='tab:blue')
        ax.plot(x_eval, exp['u_pred'], label="数值解", lw=1.5, linestyle='--', color='tab:red')
        ax.set_title(f"{exp['model_name']} + {exp['method_name']}")
        ax.text(0.02, 0.95,
                f"Max误差: {exp['max_error']:.2e}\nL2误差: {exp['l2_error']:.2e}",
                transform=ax.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        ax.grid(alpha=0.3)
        if row == num_rows - 1:
            ax.set_xlabel("x")
        if col == 0:
            ax.set_ylabel("u(x)")
        if row == 0 and col == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ritz-pinn-fitted-curve.png", dpi=150, bbox_inches='tight')
    plt.close()

    # --- 图2: 误差/残差 ---
    fig_err, axes_err = plt.subplots(num_rows, num_cols,
                                     figsize=(5 * num_cols, 3.5 * num_rows),
                                     sharex=True, sharey=True)
    axes_err = np.atleast_2d(axes_err)
    for exp in experiment_results:
        row = model_order.index(exp['model_name'])
        col = method_order.index(exp['method_name'])
        ax = axes_err[row, col]
        ax.plot(x_eval, exp['error'], label='|误差|', color='tab:green')
        ax.plot(x_eval, np.abs(exp['residual']), label='|残差|', color='tab:orange', linestyle='--')
        ax.set_title(f"误差/残差 - {exp['model_name']} + {exp['method_name']}")
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        if row == num_rows - 1:
            ax.set_xlabel("x")
        if col == 0:
            ax.set_ylabel("量值")
        if row == 0 and col == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ritz-pinn-error-residual.png", dpi=150, bbox_inches='tight')
    plt.close()

    # --- 图3: 训练损失曲线 ---
    fig_loss, axes_loss = plt.subplots(num_rows, num_cols,
                                       figsize=(5 * num_cols, 3.5 * num_rows),
                                       sharex=True, sharey=True)
    axes_loss = np.atleast_2d(axes_loss)
    for exp in experiment_results:
        row = model_order.index(exp['model_name'])
        col = method_order.index(exp['method_name'])
        ax = axes_loss[row, col]
        history = exp.get('loss_history') or []
        if history:
            epochs = np.arange(1, len(history) + 1)
            ax.plot(epochs, history, color='tab:purple', label='训练误差')
        else:
            ax.text(0.5, 0.5, '无训练记录', transform=ax.transAxes, ha='center', va='center')
        ax.set_title(f"训练误差 - {exp['model_name']} + {exp['method_name']}")
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        if row == num_rows - 1:
            ax.set_xlabel("Epoch")
        if col == 0:
            ax.set_ylabel("误差")
        if history and row == 0 and col == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ritz-pinn-training-loss.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ------------------------------------------------------------------
    # 结果摘要
    # ------------------------------------------------------------------
    best_accuracy = min(experiment_results, key=lambda item: item['l2_error'])
    best_speed = min(experiment_results, key=lambda item: item['training_time'])
    print(f"\n最小 L2 误差: {best_accuracy['model_name']} + {best_accuracy['method_name']}"
          f" -> {best_accuracy['l2_error']:.2e}")
    print(f"最快收敛: {best_speed['model_name']} + {best_speed['method_name']}"
          f" -> {best_speed['training_time']:.2f} s")
    print(f"\n图片已保存至: {OUTPUT_DIR}")
