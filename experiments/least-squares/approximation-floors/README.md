# 最佳逼近下界（approximation-floors）

这个目录把「字典逼近能力不足」与「求解器/正则化/求积不行」彻底分开。目录中的每个数字都是一个**最佳逼近**问题的解

$$\inf_{v_N \in \mathcal{V}_N} \| v_\star - v_N \|,$$

其中不出现最小二乘求解器、训练求积、正则化参数与系数球。把 `order` 研究实测的图误差除以同一配置的下界，就得到「求解器还剩多少余地」这一唯一有意义的问句的答案。

这些量不再进入论文主结果；根目录 `../README.md` 只保留实验入口，本文件集中记录其诊断定位、瓶颈解释、当前结果与复现细节。

## 理论参考

$\rho_k$ 字典的饱和指数与逼近阶为

$$s_{\rm cap}(d) = \frac{d + 2k + 1}{2}, \qquad \beta(k, d, m) = \frac{s_{\rm cap}(d) - m}{d}.$$

由此得到两个函数类上的理论基准：误差范数每多一阶导数，$\beta$ 降低 $1/d$；激活幂次每增加一个单位，$\beta$ 提高 $1/d$。固定解析制造解、有限 $N$ 区间和离散测试范数下的诊断斜率无需与 $\beta$ 严格相等，本目录只比较其趋势与量级，不把最佳逼近拟合作为论文的理论验证。

## 度量了哪四个空间

`run_floors.py` 对每个 $(k, N, p)$ 组合测量四条下界，它们的相对大小决定瓶颈在哪一侧：

| 列 | 空间 | 含义 |
|---|---|---|
| `scalar_raw_l2` / `scalar_raw_graph` | 原始 $\rho_k$ 字典 | 位移/挠度在 $L^2$ 与 $H^m$ 下的下界；两者之比即 $N^{m/d}$ 效应 |
| `scalar_spline_graph` | 边界适配样条空间 $V_K^{(p)}$ | 与字典无关的**上界**，由 $p$ 与 $K$ 决定 |
| `scalar_projected_graph` | 字典在 $V_K^{(p)}$ 中的 Ritz 投影像 | 驱动程序实际搜索的空间 |
| `scalar_attainable_graph` | 上两者取大 | 端到端可达的地板 |
| `tensor_l2_floor` | 原始字典，逐分量 $L^2$ | 应力/弯矩的真 $L^2$ 下界 |
| `tensor_raw_graph` | 原始字典，联合图范数 | 应力 $\boldsymbol{H}({\rm div})$ / 弯矩 $\boldsymbol{H}({\rm div\,div})$ 下界 |

注意 `tensor_raw_l2` 与 `tensor_l2_floor` **不是同一个量**：前者是图范数最优拟合的 $L^2$ 分量，一旦散度项主导图范数，该分量几乎不受约束，可以随 $N$ 增大而上升；只有后者是下界，诊断报表使用 `tensor_l2_floor`。

## 方法：为什么这些数字可信

- **不用法方程。** 每个拟合都对高矮设计矩阵作分块 Householder QR（`accumulate_qr`），逐块把 $[A \mid b]$ 归约为 $R$，峰值内存是一块加 $R$。因此 $10^{-13}$ 量级的下界不是条件数平方造成的假象。
- **精确场取自主程序自身。** `load_benchmark` 调用四个驱动的 `build_shared_benchmark`，因而制造解、Voigt 约定、材料常数与测试求积与 `order` 研究逐字一致；`solution` 列记录解析出的制造解名，而不是命令行传入的覆盖值。
- **张量分量联合拟合。** $\boldsymbol{H}({\rm div})$ 最优解会牺牲 $L^2$ 精度换取更小的散度残差，这正是最小二乘泛函所做的事；散度行的右端取 $-\boldsymbol{f}$，与驱动的平衡残差同一约定。
- **统一使用整条宽度序列。** `floors.fitted_order` 将每组实验中所有有限且为正的误差对总宽度 $N$ 作对数最小二乘拟合，不按多项式补充占比或误差大小作事后筛选。高幂次在最大 $N$ 处若进入双精度误差平台，仍保留该点，并在结果解释中明确所报数值只是整条序列的整体斜率。

## 如何解释瓶颈

定义给定试探空间在测试规则上的误差地板

$$
E_{\mathrm{best}}(N)
=\inf_{v_N\in\mathcal V_N}\|v_\star-v_N\|_{\mathrm{disc}}.
$$

该问题直接用精确制造解选择系数，不包含训练积分、系数球预算和生产求解器，因而不是可执行的 PDE 算法。对完整算法输出 $v_N^{\mathrm{out}}\in\mathcal V_N$ 总有

$$
E_{\mathrm{best}}(N)
\le \|v_\star-v_N^{\mathrm{out}}\|_{\mathrm{disc}}.
$$

因此这些量只回答瓶颈问题：若完整算法误差与 `scalar_attainable_graph` 或对应张量下界同阶且比值保持常数量级，当前瓶颈在试探空间；若比值随 $N$ 增长，则训练积分、系数约束、Ritz 实现或代数求解尚未达到该空间的表示能力。这个比值只能把“空间误差”与“求解流程附加误差”分开，不能单独识别后者究竟来自 $Q$、$B$、$K$ 还是谱截断；进一步归因必须结合各参数扫描。

实际性能、激活幂次收益和论文观测阶一律以 `../results/<model>/.../summary.csv` 中的完整算法误差为准，本目录结果只作离线诊断。

## 当前诊断结果

二维线弹性位移与板挠度的拟合如下。每组所有有限且为正的数据都对总宽度 $N$ 作整条序列拟合，不按多项式补充占比、误差大小或点数作事后筛选。

| 模型 | $k$ | 理论 $\beta_{L^2}$ | 最佳逼近 $L^2$ 阶 | 理论 $\beta_{H^m}$ | 最佳逼近图范数阶 |
|---|---:|---:|---:|---:|---:|
| 二维线弹性 | 3 | 2.25 | 1.95 | 1.75 | 1.55 |
| 二维线弹性 | 5 | 3.25 | 3.38 | 2.75 | 2.84 |
| 二维线弹性 | 7 | 4.25 | 4.66 | 3.75 | 4.07 |
| 二维线弹性 | 9 | 5.25 | 6.03 | 4.75 | 5.55 |
| 板弯曲 | 3 | 2.25 | 2.28 | 1.25 | 1.24 |
| 板弯曲 | 5 | 3.25 | 3.63 | 2.25 | 2.51 |
| 板弯曲 | 7 | 4.25 | 5.13 | 3.25 | 3.84 |
| 板弯曲 | 9 | 5.25 | 6.12 | 4.25 | 4.97 |

高幂次尾部已经出现双精度平台：二维 $k=9$ 的 $L^2$ 误差在 $N=800$ 后约为 $10^{-14}$ 且不再单调下降，板 $k=9$ 的 $L^2$ 与 $H^2$ 误差在最后一级也略有回升。因此表中相应数值只是完整 $N$ 区间的整体诊断斜率，不能解释为平台段的局部渐近阶。旧三维最佳逼近文件含已弃用的脏数据，不据此形成结论；待新宽度序列完成后再生成三维诊断。

上表来自 `results/floors-elasticity-2d.csv` 与 `results/floors-plate.csv`；实际误差/误差地板比值由 `run_solver_gap.py` 写入 `results/solver-gap-order{,-k3}.csv`。

## 复现

全部命令都在本目录下运行。宽度阶梯与 campaign 的 $N$ 阶梯一致，`run_solver_gap.py` 按 $N$ 对齐，因此每个 campaign 配置都有下界可除。

```bash
# 四个模型的下界（三维不再使用旧的脏数据 N=200）
python run_floors.py --model plate         --powers 3,5,7,9 --widths 200,400,600,800,1000     --degrees 7
python run_floors.py --model elasticity-2d --powers 3,5,7,9 --widths 200,400,600,800,1000     --degrees 7
python run_floors.py --model plane-stress  --powers 3,5,7,9 --widths 200,400,600,800,1000     --degrees 7
python run_floors.py --model elasticity-3d --powers 3,5,7,9 --widths 400,500,600,800,1000     --degrees 7

# 样条次数天花板：固定 k=7，只扫 p
python run_floors.py --model elasticity-2d --powers 7 --widths 1000 --degrees 3,5,7 --suffix=-degrees

# 系数预算随 rcond 的增长（字典饱和判据）
python run_coefficient_saturation.py --model elasticity-2d --powers 3,7

# 实测/下界比值（需要 ../results/<model>/<study>/results.json 已存在）
python run_solver_gap.py --study order
python run_solver_gap.py --study order-k3

# 只重算拟合阶列，不重算下界
python run_floors.py --model elasticity-2d --refit
```

`--suffix` 的取值以连字符开头时必须写成 `--suffix=-trig` 而非 `--suffix -trig`，否则 argparse 把它当作选项名。

这里的脚本只写 CSV / JSON；诊断结果保留在本目录及 `results/`，不再手写进论文正文。

## 诊断结论 → 脚本 → 输出列

| 诊断结论 | 脚本 | 输出文件 | 判据 |
|---|---|---|---|
| 图范数下界比 $L^2$ 下界高约 $N^{m/d}$ 倍，且这是字典的内禀性质 | `run_floors.py` | `floors-<model>.csv` | `scalar_raw_graph / scalar_raw_l2`；张量侧用 `tensor_raw_graph / tensor_l2_floor` |
| 误差范数导数阶带来的诊断阶差 | `run_floors.py` | 同上 | `order_scalar_l2 - order_scalar_graph` 与 $m/d$ 比较 |
| 提高 $k$ 后试探空间容量是否改善 | `run_floors.py` | 同上 | `order_scalar_graph` 的变化趋势与 `beta_scalar_graph` 比较 |
| 求解器已达最佳逼近，调参无数量级余地 | `run_solver_gap.py` | `solver-gap-order.csv` | `ratio` 列 |
| 样条次数天花板与匹配条件 $p+1 \ge s_{\rm cap}(d)$ | `run_floors.py --degrees` | `floors-*-degrees.csv` | `scalar_spline_graph` 与 `scalar_projected_graph` 谁更大 |
| 辅助空间在三维不是瓶颈 | `run_floors.py` | `floors-elasticity-3d.csv` | `scalar_spline_graph` $\ll$ `scalar_raw_graph` |
| $\|\boldsymbol{c}\|_{\ell^2}$ 随 rcond 暴涨即字典饱和 | `run_coefficient_saturation.py` | `coefficient-saturation-<model>.csv` | `coefficient_norm`、`budget` 与 `tensor_graph` 三者的联动 |

## 已知限制

- **板必须用非多项式制造解。** 默认挠度 $x_1^2(1-x_1)^2x_2^2(1-x_2)^2$ 总次数为 $8$，其弯矩总次数为 $6$，$k \ge 7$ 时被 $P_k(\Omega)$ 精确再现，弯矩下界塌到 $7\times10^{-15}$ 且不随 $N$ 变化。`plate-bending/defaults.json` 因此取 `manufactured_solution: "trig"`（$\sin^2(\pi x_1)\sin^2(\pi x_2)$，固支）。
- **高阶误差平台仍需解释。** 二维弹性与板弯曲的 $k=9$ 误差在最大 $N$ 处不再单调下降。拟合仍使用全部 $N$，因此相应观测阶表示完整实验区间的整体趋势，不应解释为平台段的局部渐近阶。
- **三维结果等待干净数据。** 旧的 $N=200$ 数据不再使用；三维结论与阶对照需在更新后的宽度序列完成后重新生成。
- **比值应当 $\ge 1$。** 张量下界在无规范约束的原始字典上求得，而驱动程序还带零平均迹规范，其可行集更小；位移下界取 `max(scalar_projected_graph, scalar_spline_graph)`，同样是保守取法。因此 `ratio < 1` 不是「求解器超过了最佳逼近」，而是该行的 QR 已受双精度限制，应连同 `scalar_exact_norm` / `tensor_exact_norm` 一起检查。
