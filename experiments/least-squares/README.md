# 线性化网络混合最小二乘实验（least-squares）

这个目录是论文 `content/article/least-squares.typ`「数值实验」一章的全部代码与数据。论文正文只陈述结论与数学，脚本名、CSV 路径、命令行与运行约定一律记在这里。

**不变量：正文出现的每个数字都有一份 CSV 出处。** 代码只写 CSV / JSON / PNG，从不生成 Typst；论文里的表格按下文的对应表手写。

## 目录结构

| 路径 | 内容 |
|---|---|
| `linear-elasticity-2d/`、`linear-elasticity-3d/`、`plane-stress/`、`plate-bending/` | 四个驱动，各含 `<model>.py` 与 `defaults.json`（单次运行的默认；研究脚本会覆盖 $N$、$Q$、$K$） |
| `rfm_core.py` | 确定性准均匀字典、退化区多项式方向的列主元 QR 选取、盒均值闭式、B 样条 Ritz 投影、系数球最小二乘的 SVD 因子与谱滤波 |
| `elasticity_common.py` | 三个线弹性类模型共享的残差装配、制造解与测试求积 |
| `ls_common.py` | 与 PDE 无关的工具：单位盒求积、流式 Householder TSQR、defaults 载入、控制台表格 |
| `system_backends.py` | direct 与 Gram 两个系统后端；两者共用同一批残差分块，均不作列归一化 |
| `solvers.py` | 系数求解器注册表（`ball` / `ridge` / `tsvd`），共享同一个谱因子 |
| `study_runner.py` | 四类重复研究（`order` / `q` / `k` / `power`），每个研究写 `records.csv` 与 `summary.csv`。`power` 只用于二维线弹性与板弯曲；近不可压缩实验使用独立入口 |
| `run_hu_zhang_3d.sh` | 三维 Hu--Zhang 最终 campaign；每个 worker 通过 `--run-offset` 使用不重叠的随机种子 |
| `plot_convergence.py` | 按模型生成论文收敛图，并写出 `results/observed-orders.csv` 的对数拟合 |
| `approximation-floors/` | 离散最佳逼近误差、实测/最佳逼近误差比值、系数范数扫描；见该目录自己的 README |
| `results/` | 当前论文口径的全部研究输出；二维与板取 `direct_rcond = 1e-14`，三维 Hu--Zhang 取 `1e-13` |
| `results-archive/rcond12-comparison/` | 被 1e-14 取代的二维 `1e-12` 运行，是截断水平敏感性分析的数据来源 |
| `results-archive/` | 更早的两批归档：`2026-08-22`（旧口径）与 `2026-08-24-plate-poly`（板仍用多项式制造解） |

## 实际求解的离散问题

论文正文只概述代数求解方式；完整实现约定记录在这里。装配后的加权残差系统为 $(\boldsymbol R_{N,Q}, \boldsymbol b_{N,Q})$，训练问题是

$$\min_{\boldsymbol c \in \mathcal C_{N,B}} \|\boldsymbol R_{N,Q}\boldsymbol c - \boldsymbol b_{N,Q}\|_{\ell^2}^2, \qquad \mathcal C_{N,B} = \{\|\boldsymbol c\|_{\ell^2} \le B/\sqrt{m_N}\}.$$

`rfm_core.L2BallLeastSquaresFactor` 持有一次 SVD，三个下游解都是同一族谱滤波 $\boldsymbol c(\mu) = \boldsymbol V_r(\boldsymbol\Sigma_r^2 + \mu \boldsymbol I_r)^{-1}\boldsymbol\Sigma_r\boldsymbol U_r^{\mathsf T}\boldsymbol b$：

| 方法 | 代码入口 | $\mu$ 的定法 |
|---|---|---|
| `ball` | `.solve(radius, rcond)` | 由约束定出：若 $\|\boldsymbol c(0)\|_{\ell^2} \le B/\sqrt{m_N}$ 则 $\mu = 0$，否则二分长期方程 $\|\boldsymbol c(\mu)\|_{\ell^2} = B/\sqrt{m_N}$（相对容差 `1e-12`，至多 100 步） |
| `tsvd` | `.solve_min_norm(rcond)` | $\mu = 0$，保留集由 `rcond` 定 |
| `ridge` | `.solve_ridge(lam)` | 固定 $\mu = \lambda^2$，不作截断 |

**论文只报告 `ball`**：只有它的 $\mu$ 由约束集 $\mathcal C_{N,B}$ 自身定出，与总误差定理所分析的离散问题一致；另外两个是同族旁支。注册表仍保留三者：

- 预算候选集含 $B = \infty$，该端点解与 `tsvd`（在同一 `rcond` 下）逐位相同，因此截断 SVD 已包含在预算选择之内。
- `defaults.json` 的 `algorithms_to_run` 保持三项，重跑才与 `results/`、`results-archive/` 的口径一致（归档 CSV 的 `algorithm` 列含三种）。只要 `ball`，加 `--algorithms ball`。
- 三维 Hu--Zhang 最终实验固定取 $B=\infty$，不在验证集上选择预算；二维与板的逐种子选择结果记在 `hyperparameters` 列。

### 字典规模口径：`--widths` 与论文的 $N$

命令行的 `--widths` 与 CSV 的 `N` 列数的是脚本铺放的参数对个数。实现另有一列显式常数特征（`rfm_core._augmented_feature_tensor` 写入的第 0 列），它与多项式区那 $\dim P_k - 1$ 对参数一起构成 $P_k(\Omega)$ 的一组基。论文的 $N$ 数的是字典元素个数，两者相差这一列：

$$N_{\text{论文}} = \texttt{width} + 1.$$

因此正文表格与图中的 $N \in \{201, 401, 601, 801, 1001\}$ 对应 `--widths 200,400,600,800,1000`，幂次实验的 $N=501$、$N=1001$ 对应 `--widths 500`、`--widths 1000`。正文把训练求积写作 $Q = r N$，代码里就是 `ratio * (width + 1)`（`study_runner.py`），两者数值相同。`plot_convergence.py` 在读入 `summary.csv` 后统一换算，作图与对数拟合都在论文口径下进行，`results/observed-orders.csv` 的 `dictionary_sizes` 列即论文的 $N$。

论文与实现选的 $P_k(\Omega)$ 基不同（论文取 $\dim P_k$ 对参数，实现取常数列加 $\dim P_k - 1$ 对参数），但张成的离散空间逐项相同，只有系数球的 $\ell^2$ 几何随基而变。

## 论文数字 → 命令 → 数据

命令都在本目录下运行；`<model>` 取 `elasticity-2d`、`elasticity-3d`、`plane-stress`、`plate`。不带 `results/` 前缀的文件位于 `approximation-floors/results/`。

| 论文位置 | 命令 | 数据文件 | 列 |
|---|---|---|---|
| 实验设计中的预算选择 | `study_runner.py order --model <model>` | `results/<model>/order/summary.csv` | `hyperparameters` |
| `tbl:ls-order-<model>`、`tbl:ls-config-order-<model>`、`fig:ls-convergence-order-<model>` | `study_runner.py order --model <model>` | `results/<model>/order/summary.csv` | `sigma_hdiv_error_{mean,std}`、`u_h1_error_{mean,std}`；板为 `M_hdivdiv_error_*`、`w_h2_error_*`；固定 $k=7$ |
| 正文中 $k=3$ 与 $k=7$ 的实际 $N$-观测阶对照 | `study_runner.py order --model <model>`；$k=3$ 加 `--output-name order-k3 --activation-power 3`，再运行 `plot_convergence.py` | `results/<model>/order{,-k3}/summary.csv`、`results/observed-orders.csv` | `observed_order` |
| `fig:ls-convergence-power-<model>`、`tbl:ls-power-{elasticity-2d,plate}`、`tbl:ls-config-power-{elasticity-2d,plate}` | `study_runner.py power --model <model>`，`<model>` 只取 `elasticity-2d` 与 `plate`，配置见下文 | `results/<model>/power/summary.csv` | 同上 |
| `fig:ls-convergence-near-incompressible-2d`、`tbl:ls-near-incompressible-{stress,u}` | `run_near_incompressible_2d.py` | `results/elasticity-2d/near-incompressible/summary.csv` | `lambda`、`sigma_hdiv_error_mean`、`u_h1_error_mean`；观测阶另见 `results/observed-orders.csv` |
| 离散误差辨识：辅助样条次数 | `run_floors.py --model elasticity-2d --powers 7 --widths 1000 --degrees 3,5,7 --suffix=-degrees` | `floors-elasticity-2d-degrees.csv` | `scalar_spline_graph`、`scalar_projected_graph` |
| 离散误差辨识：后端 | `study_runner.py order --model elasticity-2d --system-backend gram --widths 1000 --repeats 1 --output-name gram-probe` | `results/elasticity-2d/gram-probe/summary.csv` | `sigma_hdiv_error_mean` |
| 离散误差辨识：截断水平 | `study_runner.py order --model <model> --direct-rcond 1e-12` | `results-archive/rcond12-comparison/<model>/main/summary.csv` | `u_h1_error_mean` |
| 系数范数诊断（正文不展开） | `run_coefficient_saturation.py --model elasticity-2d --powers 3,7` | `coefficient-saturation-elasticity-2d.csv` | `coefficient_norm`、`budget`、`rank` |
| 实测与离散最佳逼近误差之比 | `run_solver_gap.py --study order`、`--study order-k3` | `solver-gap-order{,-k3}.csv` | `ratio` |
| 板的多项式制造解诊断 | `run_floors.py --model plate --solution default --widths 250,500,1000 --degrees 7 --suffix=-poly` | `floors-plate-poly.csv` | `tensor_raw_graph` |
| 系数球激活情况（正文不展开） | `study_runner.py order --model <model>` | `results/<model>/order/summary.csv` | `coefficient_ball_active_mean`、`coefficient_norm_mean` |

命名对应：论文的「关于 $N$ 的收敛实验」= 代码研究 `order`，「关于幂次 $k$ 的收敛实验」= 代码研究 `power`；论文标签、图片文件与研究 id 共用 `order` ↔ `power` 词干（`tbl:ls-order-*`、`tbl:ls-config-order-*`、`fig:ls-convergence-order-*`、`convergence-order-*.png`、`results/<model>/order/` ↔ `tbl:ls-power-*`、`tbl:ls-config-power-*`、`fig:ls-convergence-power-*`、`convergence-power-*.png`、`results/<model>/power/`）。`results-archive/` 与 `results-remote/` 是改名前的历史快照，其内部的 `main` / `activation` 目录与字段一律冻结不改；迁入当前 `results/` 时统一改用 `order` / `power`。

`run_floors.py` 的 `--suffix` 取值以连字符开头时必须写成 `--suffix=-poly`，否则 argparse 把它当作选项名。`run_full_campaign.sh` 里的 `--tag` 是该脚本自己的参数，转成 `study_runner.py` 的 `--output-name`。

## 复现顺序

```bash
conda activate dl
bash run_rcond14_chain.sh     # 二维、平面应力与板的 results/ 由此产生，逐步可续
bash run_hu_zhang_3d.sh       # 三维 Hu--Zhang 最终五档收敛实验
OPENBLAS_NUM_THREADS=4 LS_TORCH_THREADS=4 \
  conda run -n dl python run_near_incompressible_2d.py
```

第一条链生成二维、平面应力与板的当前结果，并保留 `results-archive/rcond12-comparison/` 中的旧截断水平证据；三维最终数据由 Hu--Zhang 专项脚本独立生成。所有正式结果迁入 `results/<model>/<study>/` 后，再运行 `plot_convergence.py` 与两次 `run_solver_gap.py`。近不可压缩专项也由 `run_full_campaign.sh` 在结果缺失时调用；上面的独立命令用于只重跑该专项。单独重跑其中一段：

```bash
bash run_full_campaign.sh                 # 只跑研究；已有 results.json 的研究自动跳过
python plot_convergence.py                # 分模型图片 + observed-orders.csv
cd approximation-floors && python run_floors.py --help   # 离散最佳逼近侧的命令见该目录 README
```

二维线弹性关于幂次 $k$ 的收敛实验（`power` 研究）采用容量探测后确定的离散配置：

```bash
OPENBLAS_NUM_THREADS=4 LS_TORCH_THREADS=4 \
python study_runner.py power --model elasticity-2d \
  --widths 500 --q-ratio 16 --ritz-degree 10 \
  --projection-samples 12808 --direct-rcond 1e-14 \
  --algorithms ball --budgets 1000,3000,10000,30000,inf \
  --repeats 10
```

它写入 `results/elasticity-2d/power/`；被替换的 $N=1000,p=7,Q=4004$
结果保存在 `results-archive/2026-08-29-elasticity-2d-activation-n1000/`。

板弯曲关于幂次 $k$ 的收敛实验（`power` 研究）采用 $N=1000$ 容量探测后确定的配置：

```bash
OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12 OMP_NUM_THREADS=12 \
LS_TORCH_THREADS=12 \
python study_runner.py power --model plate \
  --widths 1000 --q-ratio 8 --ritz-degree 10 \
  --projection-samples 12808 --direct-rcond 1e-14 \
  --algorithms ball,ridge,tsvd --repeats 10
```

该命令写入 `results/plate/power/`。不带离散参数覆盖时，
`study_runner.py power --model plate` 采用同一组模型级默认值。

## 报告与统计约定

- 单隐层字典是确定性点集：随机性只来自训练样本、Ritz 投影积分与验证求积三组种子。默认 `--repeats 10`，`summary.csv` 的每个量给出 `*_mean` 与 `*_std`（样本标准差）。
- 二维与板的 `order` 研究公共配置取训练点数 $Q = 4(N+1)$；二维线弹性的 `power` 研究取 $Q=16(N+1)=8016$，板弯曲的 `power` 研究取 $Q=8(N+1)=8008$，二维近不可压缩专项取 $Q=8(N+1)$，三维 Hu--Zhang 最终实验取 $Q=16(N+1)$。`q` 研究扫 $Q/(N+1) \in \{1,2,4,8,16,32\}$，`k` 研究扫 $K/(N+1) \in \{1.10, 2.10, 4.19\}$。
- Ritz 投影样本数由 `projection_samples` 控制：二维线弹性与平面应力的全部研究、以及板弯曲的 `power` 研究固定取 $Q_R=12808$；板弯曲的 `order` 研究未显式指定时取 $Q_R=\max(2048,4K)$；三维 Hu--Zhang 最终实验固定取 $Q_R=16384$。每条运行记录均保存实际使用值。
- 平面应力与二维线弹性同取 $E=4/3$、$\nu=1/3$。平面应力约化后的 $\lambda_{\rm ps}=E\nu/(1-\nu^2)=1/2$，不同于二维线弹性的 $\lambda=1$；驱动在 `plane-stress/defaults.json` 中显式固定这组材料参数。
- 三套抽样规则相互独立：训练、Ritz 投影、验证。测试求积是确定性张量积 Gauss–Legendre 规则，训练残差从不代替测试图误差。
- 二维模型与板的测试规则含 $128^2$ 个点，三维模型含 $32^3$ 个点；预算验证分别使用 $32^2$ 与 $16^3$ 个独立点。
- 误差评估按 `evaluation_batch_size`（默认 4096）分块累加：每个指标都是测试规则上的加权和，分块只改变求和次序。与 2026-08-26 归档数据逐位对照，`coefficient_norm`、`rank`、`condition_estimate`、`hyperparameter`、`validation_score` 完全一致，各误差量的相对偏差 $\le 5\times 10^{-10}$。
- 一次运行内验证规则与测试规则**共用同一个装配好的系统与谱分解**（训练点相同，仅测试求积不同），故每个 repeat 只装配、只分解一次。
- **数值可复现性受 BLAS 线程数限制。** 约化系统的条件数达 $10^{17}$，`direct_rcond = 1e-14` 的截断正落在算术噪声上，因此线程数不同会让图误差在 $10^{-5}$ 相对量级上漂移（实测 $N=200$、二维：`torch.set_num_threads` 取 14 / 28 / 默认三档，$\|u_N-u_*\|_{H^1}$ 相差至多 $3\times 10^{-5}$）。逐位复现归档数字需要固定线程数。
- 平均迹投影不抽样，按闭式箱均值精确构造，故 $\varepsilon_{\rm trquad} = 0$。
- 容量扫描对每个 $N$ 重新生成训练点与对应维数的 Ritz 规则，避免复用固定训练集导致名义 $Q$ 与实际样本数不一致。
- 二维近不可压缩专项采用 Li--Yang (2020) Example 1 的制造解，固定 $\mu=1$，取 $\lambda\in\{10,10^3,10^5\}$。每个 $\lambda$ 都运行完整的 $N\in\{200,400,600,800,1000\}$ 阶梯；同一 repeat 内三种材料共用字典、训练/Ritz/验证/测试规则。系数预算预先固定为 $B=3\times10^6$，不随 $\lambda$ 调参，十次重复形成配对比较。
- 系数预算只从上述有限候选集中按**验证**图误差选，选定后才在固定测试规则上报告误差。

## 离散最佳逼近诊断（不进入论文主结果）

论文正文只用完整经验最小二乘解评价 PDE 方法。`approximation-floors/` 保留离散最佳逼近、实际误差/误差地板比值和样条侧下界，作为定位字典、Ritz 空间或求解流程瓶颈的离线诊断；这些量不参与正文的性能结论。

诊断的数学定义、当前二维与板结果、双精度平台、三维脏数据状态、输出列和复现命令统一见 [`approximation-floors/README.md`](approximation-floors/README.md)。实际性能、激活幂次收益和正文观测阶一律以 `results/<model>/.../summary.csv` 中的完整算法误差为准。

## 诊断量（论文不再逐项列出）

| 列 | 含义 |
|---|---|
| `parameter_covering_radius_*`、`parameter_separation_*` | 非退化参数经 $\mathfrak P$ 归一化后的覆盖半径（固定探针集估计）与精确的两两最小测地距离，即准均匀性在有限 $N$ 处的验证 |
| `projection_gram_residual_*` | Ritz 投影 Gram 方程的相对残差 |
| `boundary_residual_*` | 边界值残差；板另含法向导数残差 |
| `trace_residual_*`、`continuous_trace_mean_*` | 离散系数上的平均迹规范残差，与确定性测试规则上的连续平均迹 |
| `coefficient_norm_*`、`coefficient_ball_active_*` | $\|\boldsymbol c\|_{\ell^2}$ 与约束是否激活；相应预算为 $\|\boldsymbol c\|_{\ell^2}\sqrt{m_N}$ |
| `rank_*`、`condition_estimate_*` | 截断后的保留个数与保留谱的条件数估计 |
| `sigma_deviatoric_l2_error_*`、`sigma_hydrostatic_l2_error_*` | 应力的偏差/球部误差，用于观察近不可压缩极限 |
| `wall_time_*` | 单次运行墙钟时间 |

### 三维大内存诊断与无效调参（不入论文主表）

这一组是 2026-08-27 在 25 核、90 GiB 的远端机器上做的单种子容量/消融诊断，原始数据统一保存在
`results-remote/2026-08-27-3d-capacity/`。除下面明确标成 Sobol 的截断探针外，训练积分全部使用 MC；
这些单种子结果用于定位瓶颈和排除无效参数，不与正文默认的 10 次重复统计混用。

| 配置（均为 $k=11$、`ball`） | 验证选出的 $B$ | $\|u_N-u_*\|_{H^1}$ | $\|\sigma_N-\sigma_*\|_{H(\mathrm{div})}$ | 平衡/散度残差 |
|---|---:|---:|---:|---:|
| $N=1000,Q=4004$ | $7.5\times10^6$ | $3.399\times10^{-3}$ | $1.134\times10^{-2}$ | $1.100\times10^{-2}$ |
| $N=1000,Q=8008$ | $3.0\times10^7$（约束不激活） | $2.591\times10^{-3}$ | $7.882\times10^{-3}$ | $7.629\times10^{-3}$ |
| $N=1500,Q=6004$ | $1.0\times10^7$ | $6.396\times10^{-4}$ | $1.237\times10^{-3}$ | $1.074\times10^{-3}$ |
| $N=1500,Q=12008$ | $1.0\times10^7$ | $4.258\times10^{-4}$ | $9.127\times10^{-4}$ | $7.804\times10^{-4}$ |

- **主瓶颈是字典宽度 $N$，不是内存，也不是单纯的 MC 点数 $Q$。** 固定 $N=1000$ 把 $Q$ 加倍只把应力图误差降低约 30%；保持 $Q=4(N+1)$ 把 $N$ 从 1000 增至 1500 则降低约 9.2 倍；在 $N=1500$ 再把 $Q$ 加倍只再降低约 26%。因此 $Q$ 是真实但次级的求积噪声来源。对应 CSV 分别位于 `mc-k11-n1000-budget-fine/`、`mc-k11-n1000-q8008-budget-fine/`、`mc-k11-n1500-q6004-budget-fine/` 与 `mc-k11-n1500-q12008-budget-fine/`。
- **应力误差几乎由散度项主导。** 例如 $N=1000,Q=4004$ 时应力 $L^2$ 误差仅 $2.725\times10^{-3}$，而平衡残差为 $1.100\times10^{-2}$；这与条件数约 $10^{14}$ 的小奇异模态放大训练求积扰动一致。
- **原预算阶梯在 $10^6$ 与 $\infty$ 之间过粗。** $k=7,N=1000,Q=4004$ 的细扫在 $B=2.0\times10^6$ 处取得应力图误差 $2.211\times10^{-2}$，比无约束端点约 $1.09\times10^{-1}$ 好约 4.9 倍；$k=11$ 的细扫则选中 $7.5\times10^6$。以后若目标是最小绝对误差，应围绕等效预算做细扫，不能用旧阶梯中“选中 $\infty$”解释为不需要正则化。证据见 `mc-k7-n1000-budget-fine/` 和 `mc-k11-n1000-budget-fine/`。论文最终 campaign 为保持既定公共配置仍使用正文所列的离散阶梯，这个细扫只作诊断。
- **固定总 $N$ 增加偏置层自由度无效且会恶化。** 把偏置层数约从 9 增至 14 会挤占球面方向；同一 $k=11,N=1000,Q=4004$ 探针的应力图误差恶化到 $5.676\times10^{-2}$。证据见 `mc-k11-n1000-bias15/` 与 `bias-layers-1.5.patch`。
- **提高 Ritz 样条次数不是当前应力瓶颈。** `ritz_degree: 7 -> 12` 只把位移图误差从 $3.399\times10^{-3}$ 降到 $3.253\times10^{-3}$，应力图误差从 $1.134\times10^{-2}$ 降到 $1.130\times10^{-2}$，不足 1%。证据见 `mc-k11-ritz12-n1000-q4004-budget-fine/`。
- **继续收紧 SVD 截断只能带来小修正。** Sobol 诊断中 `rcond=3e-15` 相对 $10^{-14}$ 的应力改善约 7%，位移反而略差，不能解释数量级差距；该 Sobol 探针只用于谱诊断，不进入 MC 论文结果。证据见 `sobol-k11-n1000-rcond/`。
- 历史容量诊断中的 `streaming_tsqr` 采用 1024 点批大小，故增大 $Q$ 主要增加运行时间；2026-08-28 的大宽度候选 campaign 曾测试 4096 点批。峰值内存主要随约化系统列数、即随 $N$ 增长。48 GiB vGPU 未被 SciPy 的 CPU TSQR/SVD 路径使用。

## 已知限制

- **三维的峰值内存与机器差异**。运行时打印的 `dense A=... GiB` 是稠密估计，只对 `direct_solver="dense"` 成立；默认的 `streaming_tsqr` 从不构造 $(36N, 9N)$。本机约 16 GiB 时，单个 `scipy.linalg.svd(gesdd)` 在 $N\ge1200$ 已会 OOM（$N=1000$ 约 2.8 GiB；$N=1200$ 的装配阶段只有约 2.4 GiB，峰值出现在 SVD 工作区）。历史上的 $N=2000,2500$ 容量候选实验使用了 120 GiB cgroup 主机；论文最终 Hu--Zhang 阶梯止于 $N=1000$。`dense A` 数字仍只用于容量预估，不能当作流式后端的实际峰值。
- **板必须用非多项式制造解。** `plate-bending/defaults.json` 取 `manufactured_solution: "trig"`，避免高幂次字典精确再现多项式制造解；相应误差地板证据见 `approximation-floors/README.md`。
- **`order` 研究的 `ritz_degree = 7` 会发出一条 RuntimeWarning**（$p + 1 = 8 < s_{\rm cap}(2) = 8.5$）。完整算法的 $K$-扫描中实际图误差变化小于 $1\%$；样条侧诊断详见 `approximation-floors/README.md`。二维线弹性的 `power` 研究已改用 $p=10$，对 $k\le9$ 满足次数匹配条件。
- **`results-archive/rcond12-comparison/` 不可重新生成后覆盖**：它是截断水平敏感性分析的唯一测量。

### 2026-08-30 三维 Hu--Zhang 最终 campaign

论文最终采用 Hu--Zhang 制造解，只运行固定 $k=7$ 的字典规模实验：

$$N\in\{200,400,600,800,1000\},\qquad Q=16(N+1).$$

Ritz 投影固定使用 $Q_R=16384$ 个独立均匀 MC 点；验证与测试规则分别为 $16^3$
与 $32^3$ 个 Gauss--Legendre 点。系数预算固定为 $B=\infty$，截断水平为
`direct_rcond = 1e-13`，每个宽度作十次独立重复。三维不做关于幂次 $k$ 的实验；
正文关于幂次的结论只来自二维线弹性与板弯曲。

当次运行的逐重复快照保存在
`results-remote/2026-08-30-3d-hu-zhang-k7-q16-qr16384/parts/main/`；`main` 是
研究重命名前的冻结目录名。完成后的 50 条记录已经迁入标准位置
`results/elasticity-3d/order/`，其中 `study` 字段同步规范为 `order`。绘图和正文数字
只读取这个标准目录，不依赖远端快照路径。

流式 TSQR 是最终后端，QR 训练块与 panel 块分别取 2048 与 128，Ritz 投影、测试
评价和制造体力的批大小分别取 4096、8192 与 5000。五点对数拟合得到应力
$\boldsymbol H(\mathrm{div})$ 与位移 $H^1$ 误差的观测阶 $1.5843$ 与 $1.3543$；
大 $N$ 端已经出现误差平台，因此正文只将其表述为完整三维算法的总体收敛结果，
不声称达到了字典项的理论阶 $8/3$。
