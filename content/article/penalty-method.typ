#import "/typ/templates/blog.typ": *

#show: main-zh.with(
  title: "边界罚化的混合弱随机特征方法及其在线弹性问题中的应用",
  author: "summer",
  desc: [提出统一的边界罚化混合弱随机特征方法，并以三维线弹性、平面应力与板弯曲作为数值算例],
  date: "2026-04-19",
  tags: (
    blog-tags.numerical-methods,
    blog-tags.pde,
  ),
  show-outline: true,
)

= 摘要

本文讨论一类面向线弹性方程的固定随机特征离散方法。与直接训练网络参数不同，本文首先固定单隐层随机特征，再仅对输出层系数求解。我们的主线方法是*边界罚化的混合弱形式*：在 Hellinger-Reissner 型混合变分框架下，用边界罚项弱施加齐次边界条件，并得到具有块对称结构的线性代数系统。作为对照，本文同时给出使用相同随机特征空间的*强形式最小二乘*离散。数值实验考虑三维线弹性、平面应力和 Kirchhoff-Love 板弯曲问题，其中前两者对应 $nabla dot$ 型耦合，后者对应 $nabla dot (nabla dot dot)$ 型耦合。结果表明：在三个算例上，边界罚化弱方法在应力上更稳健；强形式对照在部分算例中可取得更小的位移误差；而相较于通用最小二乘求解器，保留对称结构并进行谱截断的 Eigh 求解版本整体更稳定。

= 引言

近年来，利用固定随机特征或浅层神经特征近似偏微分方程解的做法受到持续关注。其核心思想是先固定隐藏层参数，再将求解过程化为有限维线性代数问题。

对于线弹性和平板弯曲这类方程，混合形式具有两个直接优势。其一，应力或弯矩未知量可被独立近似，从而避免把全部物理约束压缩到原变量的高阶导数中；其二，混合形式天然分离了本构方程与平衡方程，便于在固定随机特征空间上构造块结构清晰的离散系统。本文关注的问题正是：在不引入 PINN 训练的前提下，如何用随机特征框架求解线弹性方程。

本文采用的随机特征参数化与 @zhang2024transnet 一致：隐藏层权重随机生成后固定，仅求解输出层系数。在线性代数层面，我们比较两条路线：

1. 边界罚化的混合弱离散，即本文的主方法；
2. 带同类边界罚项的强形式最小二乘离散，即对照方法。

本文的主要贡献可概括为：

1. 给出一个统一的边界罚化混合弱随机特征框架，可同时处理 $nabla dot$ 型和 $nabla dot (nabla dot dot)$ 型力学方程；
2. 给出具有相同随机特征预算的强形式最小二乘对照离散，并在同一实验协议下比较两者；
3. 以三维线弹性、平面应力和板弯曲为三个基准算例，提炼出跨模型一致的数值现象。

= 问题描述

== 连续问题

设 $Omega subset RR^d$ 为有界区域，$partial Omega$ 为其边界。记 $bold(Sigma)$ 为对称张量场空间，$bold(U)$ 为位移空间。考虑如下连续问题：
$
  cases(
    bold(cal(A)) bold(sigma) - bold(cal(B)^*) bold(u) & = bold(0) & quad "in" Omega,
    bold(cal(B)) bold(sigma) + bold(f) & = bold(0) & quad "in" Omega,
    bold(Gamma)_j bold(u) & = bold(0) & quad "on" partial Omega\, quad j = 0\, dots\, J - 1.
  )
$
其中：$bold(sigma) in bold(Sigma)$ 表示应力，$bold(u) in bold(U)$ 表示位移。$bold(cal(A))$ 为柔度算子，$bold(cal(B))$ 为平衡算子，$bold(Gamma)_j$ 为边界迹算子。

记
$
  a(bold(sigma), bold(tau))
  := integral_Omega (bold(cal(A)) bold(sigma)) : bold(tau) dif x,
  quad
  b(bold(tau), bold(v))
  := integral_Omega (bold(cal(B)) bold(tau)) dot bold(v) dif x.
$
则对应的混合弱形式为：求 $(bold(sigma), bold(u)) in bold(Sigma) times bold(U)$，使得
$
  cases(
    a(bold(sigma), bold(tau)) + b(bold(tau), bold(u)) & = 0 & quad forall bold(tau) in bold(Sigma),
    b(bold(sigma), bold(v)) & = - (bold(f), bold(v)) & quad forall bold(v) in bold(U).
  )
$

具体而言，对于如下三个模型：

1. 三维线弹性：$bold(cal(B)) = nabla dot$，$bold(u)$ 为三维位移向量；
2. 平面应力：$bold(cal(B)) = nabla dot$，$bold(u)$ 为二维面内位移；
3. 板弯曲：$bold(cal(B)) = nabla dot (nabla dot dot)$，此时向量 $bold(u)$ 退化为标量挠度 $u$。

== 边界罚项

固定随机特征空间时，离散位移空间一般不自动满足齐次边界条件。为此，本文采用统一的边界罚化双线性形式
$
  c (bold(u), bold(v))
  := sum_(j=0)^(J-1) lambda_j (bold(Gamma)_j bold(u), bold(Gamma)_j bold(v))_(partial Omega),
$
其中 $lambda_j > 0$ 为罚参数。于是本文的离散目标不是直接强制 $bold(Gamma)_j bold(u) = 0$，而是用罚项把边界残差纳入离散方程。

#proposition(title: [边界罚项的一致性])[
  设 $(bold(sigma), bold(u))$ 为连续问题的精确解，且满足齐次边界条件
  $
    bold(Gamma)_j bold(u) = 0, quad j = 0, dots, J - 1.
  $
  则对任意 $bold(v) in bold(U)$，
  $
    c (bold(u), bold(v)) = 0.
  $
]

#proof[
  由 $bold(Gamma)_j bold(u) = 0$ 可得
  $
    (bold(Gamma)_j bold(u), bold(Gamma)_j bold(v))_(partial Omega) = 0, quad j = 0, dots, J - 1.
  $
  将其代入 $c$ 的定义即得结论。证毕。
]

因此，边界罚项不会改变原连续解；它只在离散层面上起作用，使得不满足边界约束的随机特征近似被拉回到物理可接受的边界状态附近。

= 固定随机特征离散

== 标量随机特征空间

本文采用单隐层随机特征
$
  xi_m (x) = rho(gamma (bold(a)_m^T x + r_m)), quad m = 1, dots, M,
$
并记常数特征 $xi_0 = 1$。这里 $rho$ 为激活函数，本文实验中取为 $tanh$。隐藏层参数采用重参数化方式生成 @zhang2024transnet：
$
  bold(w)_m = gamma bold(a)_m, quad b_m = gamma r_m,
$
其中 $bold(a)_m$ 是单位法向量，$r_m$ 是随机截距，$gamma > 0$ 为形状参数。采样策略为
$
  bold(a)_m = bold(X)_m / norm(bold(X)_m)_2, quad r_m = U_m,
$
其中 $bold(X)_m in RR^d$ 是从标准正态分布采样的随机向量，$U_m in RR$ 是从 $[0, 1]$ 均匀分布采样的随机数。

由此得到标量随机特征空间
$
  Xi_M := span { xi_0, xi_1, dots, xi_M }.
$

== 张量化离散空间

记 ${bold(E)_alpha}_1^(n_bold(sigma))$ 为目标对称张量空间的标准基，${bold(e)_i}_1^(n_bold(u))$ 为原变量值域的标准基。于是离散空间写为
$
  bold(Sigma)_M
  := span { xi^(bold(sigma))_m bold(E)_alpha : 0 <= m <= M, 1 <= alpha <= n_bold(sigma) },
$
以及
$
  bold(U)_M
  := span { xi^(bold(u))_m bold(e)_i : 0 <= m <= M, 1 <= i <= n_bold(u) }.
$

其中：

- 三维线弹性：$n_bold(sigma) = 6$，$n_bold(u) = 3$；
- 平面应力：$n_bold(sigma) = 3$，$n_bold(u) = 2$；
- 板弯曲：$n_bold(sigma) = 3$，$n_bold(u) = 1$。

离散未知量分别展开为
$
  bold(Phi)^(bold(sigma)) &= sum_(m = 0)^M sum_(alpha = 1)^(n_(bold(sigma))) phi^(bold(sigma))_(m, alpha) xi^(bold(sigma))_m bold(E)_alpha, \
  bold(Phi)^(bold(u)) &= sum_(m = 0)^M sum_(i = 1)^(n_(bold(u))) phi^(bold(u))_(m, i) xi^(bold(u))_m bold(e)_i.
$

#figure(
  neural-net(d: 3, n: 6, y: "bold(Phi)^(bold(sigma))"),
  caption: [神经网络结构：蓝色实线表示神经网络参数已随机初始化并固定，红色虚线则为需要求解的系数向量],
)

== 边界罚化的混合弱离散

本文考虑的问题如下：求 $(bold(Phi)^(bold(sigma)), bold(Phi)^(bold(u))) in bold(Sigma)_M times bold(U)_M$，使得
$
  cases(
    a(bold(Phi)^(bold(sigma)), bold(Phi)^(bold(tau))) + b(bold(Phi)^(bold(tau)), bold(Phi)^(bold(u))) & = 0 & quad forall bold(Phi)^(bold(tau)) in bold(Sigma)_M,
    b(bold(Phi)^(bold(sigma)), bold(Phi)^(bold(v))) + c(bold(Phi)^(bold(u)), bold(Phi)^(bold(v))) & = - (bold(f), bold(Phi)^(bold(v))) & quad forall bold(Phi)^(bold(v)) in bold(U)_M.
  )
$

记
$
  A_((n, beta), (m, alpha))
  := a(xi^(bold(sigma))_n bold(E)_beta, xi^(bold(sigma))_m bold(E)_alpha),
  quad & B_((n, beta), (m, i))
         := b(xi^(bold(sigma))_n bold(E)_beta, xi^(bold(u))_m bold(e)_i), \
  C_((n, j), (m, i))
  := c(xi^(bold(u))_n bold(e)_j, xi^(bold(u))_m bold(e)_i),
  quad & F_((n, j)) := (bold(f), xi^(bold(u))_n bold(e)_j).
$
则系数向量 $bold(phi)^(bold(sigma)) = (phi^(bold(sigma))_(m, alpha))$ 与 $bold(phi)^(bold(u)) = (phi^(bold(u))_(m, i))$ 满足块对称线性系统
$
  mat(bold(A), bold(B); bold(B)^T, bold(C)) mat(bold(phi)^(bold(sigma)); bold(phi)^(bold(u))) = mat(0; -F).
$

这里 $bold(A)$ 反映本构部分，$bold(B)$ 反映平衡耦合，$bold(C))$ 反映边界罚项。

= 强形式

本文引入强形式最小二乘对照。记
$
  bold(r)_"c" := bold(cal(A)) bold(Phi)^(bold(sigma)) - bold(cal(B)^*) bold(Phi)^(bold(u)),
  quad
  bold(r)_"e" := bold(cal(B)) bold(Phi)^(bold(sigma)) + bold(f),
  quad
  bold(r)_("b",j) := bold(Gamma)_j bold(Phi)^(bold(u)).
$
则对应的离散泛函定义为
$
  L_"strong" (bold(Phi)^(bold(sigma)), bold(Phi)^(bold(u)))
  := norm(bold(r)_"c")_(L^2(Omega))^2
  + norm(bold(r)_"e")_(L^2(Omega))^2
  + sum_(j=0)^(J-1) lambda_j norm(bold(r)_("b",j))_(L^2(partial Omega))^2.
$

即将本构残差、平衡残差与边界残差全部压缩到一个单一目标泛函中。将 $(bold(Phi)^(bold(sigma)), bold(Phi)^(bold(u)))$ 在固定随机特征基上展开后，$L_"strong"$ 对系数向量是一个二次泛函，因此最小化问题等价于一个对称线性系统。

本文关注的是：在相同随机特征空间、相同采样点与相同边界罚参数量级下，边界罚化弱方法与强形式对照究竟呈现出怎样稳定且可复现的数值差异。

= 数值实验

== 实验设定

实验算例均采用制造解基准。除非另有说明，所有实验统一使用：

#figure(
  three-line-table(
    columns: 2,
    align: (right, left),
  )[
    | 参数 | 设置 |
    |---|---|
    | 杨氏模量 | $E = 1.0$  |
    | 泊松比   | $nu = 0.3$ |
    | 激活函数 | $tanh$ |
    | 特征生成 | $gamma = 2.0$ |
    | 采样方式 | Sobol 采样 |
    | 实验特征数 | $M = 300$ |
  ],
  caption: [三类算例的公共实验设置],
)

比较的方法包括 Weak (Eigh)、Weak (Lstsq)、Strong (Eigh) 与 Strong (Lstsq)。其中 Eigh 表示对对称线性系统采用特征值分解并做相对阈值截断，Lstsq 则调用通用最小二乘求解器。
- Weak 两种算法通过离散 Hellinger-Reissner 系统求解系数
- Strong 两种算法则通过强形式残差最小二乘得到法方程

== 三维线弹性

该算例在立方体区域 $Omega = [0, 1]^3$ 上求解对称应力张量 $bold(sigma)$ 与位移向量 $bold(u)$。连续模型采用标准三维线弹性强形式
$
  cases(
    bold(cal(A)) : bold(sigma) - bold(epsilon)(bold(u)) & = 0 & quad "in" Omega,
    nabla dot bold(sigma) + bold(f) & = 0 & quad "in" Omega,
    bold(u) & = 0 & quad "on" partial Omega.
  )
$
其中
$
  bold(epsilon)(bold(u)) = 1/2 (nabla bold(u) + (nabla bold(u))^T)
$
为线性应变张量，边界条件为齐次位移约束。也就是说，该算例同时要求位移满足几何约束，并要求应力场满足内部平衡。详细推导见附录 @app:3d。

三维线弹性实验参数设置如下：

#figure(
  three-line-table(
    columns: 2,
    align: (right, left),
  )[
    | 项目         | 具体说明 |
    |:-------------|:---------|
    | 算例名称     | 三维线弹性 |
    | 计算区域     | $[0, 1]^3$ |
    | 内部采样点   | $Q_"int" = (2^6)^3$ |
    | 边界采样点   | $Q_"bc" = 6 (2^5)^2$ |
    | 测试采样点   | $Q_"test" = (2^5)^3$ |
    | 罚参数（弱式） | $lambda_"bc" = 1$ |
    | 罚参数（强式） | $lambda_"bc" = 10$ |
  ],
  caption: [三维线弹性实验设置],
)

制造解取为位移场
$
  bold(u)_"ex" (x)
  =
  mat(
    sin(pi x_1) sin(pi x_2) sin(pi x_3);
    sin(2 pi x_1) sin(pi x_2) sin(pi x_3);
    sin(pi x_1) sin(2 pi x_2) sin(pi x_3)
  ).
$
由于正弦因子在边界上为零，故该制造解满足 $bold(u)_"ex" = 0$ 于 $partial Omega$。精确应力 $bold(sigma)_"ex"$ 由各向同性本构关系计算，体力由
$
  bold(f)_"ex" = - nabla dot bold(sigma)(bold(u)_"ex")
$
通过自动微分生成，因此 $(bold(sigma)_"ex", bold(u)_"ex")$ 构成该算例的精确解。

对三维线弹性，位移误差定义为
$
  e_bold(u)
  :=
  frac(
    sqrt(frac(1, Q_"test") sum_(p=1)^(Q_"test") norm(bold(Phi)^bold(u) (bold(x)_p) - bold(u)_"ex" (bold(x)_p))_2^2),
    sqrt(frac(1, Q_"test") sum_(p=1)^(Q_"test") norm(bold(u)_"ex" (bold(x)_p))_2^2)
  ),
$
而在 Voigt 顺序 $(11, 22, 33, 12, 23, 13)$ 下，对应力采用权重 $bold(w)^"V" = (1, 1, 1, 2, 2, 2)^T$，并定义
$
  e_bold(sigma)
  :=
  frac(
    sqrt(frac(1, Q_"test") sum_(p=1)^(Q_"test") sum_(alpha=1)^6 w^"V"_alpha ((bold(Phi)^(bold(sigma)) (bold(x)_p))_alpha - (bold(sigma)_"ex" (bold(x)_p))_alpha)^2),
    sqrt(frac(1, Q_"test") sum_(p=1)^(Q_"test") sum_(alpha=1)^6 w^"V"_alpha ((bold(sigma)_"ex" (bold(x)_p))_alpha)^2)
  ).
$

=== 实验结果

实验结果见 @tb:3d-main。

#figure(
  three-line-table(
    columns: 4,
    align: (left, right, right, right),
  )[
    | 方法 | 位移误差 | 应力误差 | Time(s) |
    |:-----|---------:|---------:|--------:|
    | Weak (Eigh)        |   1.10e-02 |   1.01e-02 |     0.69 |
    | Weak (Lstsq)       |   3.92e-02 |   3.85e-02 |     0.20 |
    | Strong (Eigh)      |   3.23e-03 |   1.40e-02 |     0.62 |
    | Strong (Lstsq)     |   1.15e-01 |   7.61e-01 |     0.20 |
  ],
  caption: [三维线弹性主实验结果（$M = 300$）],
)<tb:3d-main>

#figure(
  image("/public/images/penalty-method/linear-elasticity-3d/l2-error-summary.png"),
  caption: [三维线弹性主实验结果（$M = 300$）],
)

在该算例中，Strong (Eigh) 取得了最小的位移误差，而 Weak (Eigh) 在应力误差上更优。这说明当制造解较为平滑且内部平衡残差可被强形式直接压缩时，强形式对位移场具有一定优势；但在应力恢复上，混合弱形式仍更稳健。Strong (Lstsq) 的应力误差明显劣化，说明在三维问题上通用最小二乘对病态性的敏感性尤为突出。

=== 特征数消融

特征数消融结果见 @tb:3d-ablation。

#figure(
  three-line-table(
    columns: 5,
    align: (right, left, right, right, right),
  )[
    | $M$ | 方法 | 位移误差 | 应力误差 |  Time(s) |
    |-------:|:-----------------|-------------:|-------------:|---------:|
    |    200 | Weak (Eigh)      |     4.33e-02 |     3.87e-02 |     0.41 |
    |    200 | Weak (Lstsq)     |     1.37e-01 |     1.29e-01 |     0.08 |
    |    200 | Strong (Eigh)    |     1.03e-02 |     3.73e-02 |     0.20 |
    |    200 | Strong (Lstsq)   |     9.71e-03 |     3.38e-02 |     0.07 |
    |    400 | Weak (Eigh)      |     4.76e-03 |     4.36e-03 |     1.31 |
    |    400 | Weak (Lstsq)     |     5.27e-02 |     5.37e-02 |     0.41 |
    |    400 | Strong (Eigh)    |     1.75e-03 |     7.38e-03 |     1.32 |
    |    400 | Strong (Lstsq)   |     4.34e-03 |     2.92e-02 |     0.41 |
    |    600 | Weak (Eigh)      |     1.41e-03 |     1.04e-03 |     3.72 |
    |    600 | Weak (Lstsq)     |     3.96e-02 |     3.91e-02 |     1.21 |
    |    600 | Strong (Eigh)    |     5.76e-04 |     2.76e-03 |     3.70 |
    |    600 | Strong (Lstsq)   |     1.06e-02 |     7.76e-02 |     1.21 |
    |    800 | Weak (Eigh)      |     5.88e-04 |     4.69e-04 |     7.95 |
    |    800 | Weak (Lstsq)     |     3.59e-03 |     3.94e-03 |     2.68 |
    |    800 | Strong (Eigh)    |     2.71e-04 |     1.25e-03 |     7.95 |
    |    800 | Strong (Lstsq)   |     5.53e-03 |     4.44e-02 |     2.68 |
    |   1000 | Weak (Eigh)      |     4.74e-04 |     3.83e-04 |    14.93 |
    |   1000 | Weak (Lstsq)     |     1.50e-02 |     1.83e-02 |     5.08 |
    |   1000 | Strong (Eigh)    |     2.13e-04 |     9.94e-04 |    14.91 |
    |   1000 | Strong (Lstsq)   |     6.99e-03 |     5.99e-02 |     5.09 |
  ],
  caption: [三维线弹性特征数 $M$ 的消融实验],
)<tb:3d-ablation>

#figure(
  image("/public/images/penalty-method/linear-elasticity-3d/ablation/M/ablation-M.png"),
  caption: [三维线弹性特征数 $M$ 的消融实验],
)

可以看到，随着 $M$ 增大，两个 Eigh 版本的误差整体下降，其中 Strong (Eigh) 的位移误差下降最快，而 Weak (Eigh) 的应力误差始终更小。相较之下，两个 Lstsq 版本随特征维度增加表现出明显非单调性，说明求解器病态性在三维问题中是主要瓶颈之一。

== 平面应力

该算例考虑二维薄板的面内受力，在平面应力假设下有
$
  sigma_(13) = sigma_(23) = sigma_(33) = 0,
$
因此只保留面内位移 $bold(u) = (u_1, u_2)^T$ 与二维对称应力张量 $bold(sigma)$。对应的强形式为
$
  cases(
    bold(cal(A)) : bold(sigma) - bold(epsilon)(bold(u)) & = 0 & quad "in" Omega,
    nabla dot bold(sigma) + bold(f) & = 0 & quad "in" Omega,
    bold(u) & = 0 & quad "on" partial Omega.
  )
$
其中 $bold(epsilon)(bold(u)) = 1/2 (nabla bold(u) + (nabla bold(u))^T)$ 为二维线性应变张量，边界上施加齐次面内位移条件。也就是说，该算例关注的是面内位移与面内应力的联合恢复。详细推导见附录 @app:plane-stress。

平面应力实验参数设置如下：

#figure(
  three-line-table(
    columns: 2,
    align: (right, left),
  )[
    | 项目         | 具体说明 |
    |:-------------|:---------|
    | 算例名称     | 平面应力 |
    | 计算区域     | $[0, 1]^2$ |
    | 内部采样点   | $Q_"int" = (2^8)^2$ |
    | 边界采样点   | $Q_"bc" = 4 (2^7)$ |
    | 测试采样点   | $Q_"test" = (2^7)^2$ |
    | 罚参数（弱式） | $lambda_"bc" = 1$ |
    | 罚参数（强式） | $lambda_"bc" = 10$ |
  ],
  caption: [平面应力实验设置],
)

制造解取为面内位移场
$
  bold(u)_"ex" (x)
  =
  mat(
    sin(pi x_1) sin(pi x_2);
    sin(2 pi x_1) sin(pi x_2)
  ).
$
同样地，该制造解在边界上满足 $bold(u)_"ex" = 0$。精确应力 $bold(sigma)_"ex"$ 由平面应力本构计算，体力仍由
$
  bold(f)_"ex" = - nabla dot bold(sigma)(bold(u)_"ex")
$
自动生成，因此可直接用来评估位移与应力误差。

对平面应力，仍使用同样的位移误差定义：
$
  e_bold(u)
  :=
  frac(
    sqrt(frac(1, Q_"test") sum_(p=1)^(Q_"test") norm(bold(Phi)^bold(u) (bold(x)_p) - bold(u)_"ex" (bold(x)_p))_2^2),
    sqrt(frac(1, Q_"test") sum_(p=1)^(Q_"test") norm(bold(u)_"ex" (bold(x)_p))_2^2)
  ),
$
而在 Voigt 顺序 $(11, 22, 12)$ 下，对应力采用权重 $bold(w)^"V" = (1, 1, 2)^T$：
$
  e_bold(sigma)
  :=
  frac(
    sqrt(frac(1, Q_"test") sum_(p=1)^(Q_"test") sum_(alpha=1)^3 w^"V"_alpha ((bold(Phi)^bold(sigma) (bold(x)_p))_alpha - (bold(sigma)_"ex" (bold(x)_p))_alpha)^2),
    sqrt(frac(1, Q_"test") sum_(p=1)^(Q_"test") sum_(alpha=1)^3 w^"V"_alpha ((bold(sigma)_"ex" (bold(x)_p))_alpha)^2)
  ).
$

=== 实验结果

实验结果见 @tb:ps-main。

#figure(
  three-line-table(
    columns: 4,
    align: (left, right, right, right),
  )[
    | 方法 | 位移误差 | 应力误差 | Time(s) |
    |:-----|---------:|---------:|--------:|
    | Weak (Eigh)   | 7.68e-05 | 6.26e-05 | 0.54 |
    | Weak (Lstsq)  | 9.70e-05 | 8.52e-05 | 0.08 |
    | Strong (Eigh) | 3.23e-05 | 1.78e-04 | 0.14 |
    | Strong (Lstsq)| 3.38e-05 | 3.46e-04 | 0.05 |
  ],
  caption: [平面应力主实验结果（$M = 300$）],
)<tb:ps-main>

#figure(
  image("/public/images/penalty-method/plane-stress/l2-error-summary.png"),
  caption: [平面应力主实验结果（$M = 300$）],
)

在平面应力算例中，Strong (Eigh) 给出了最小位移误差，而 Weak (Eigh) 给出了最小应力误差。这一现象与三维线弹性保持一致，但更为清晰：强形式对原变量更友好，弱形式对共轭张量场更友好。与此同时，Lstsq 版本虽然更快，但强形式对应力场的退化更明显。

=== 特征数消融

特征数消融结果见 @tb:ps-ablation。

#figure(
  three-line-table(
    columns: 5,
    align: (right, left, right, right, right),
  )[
    | $M$ | 方法 | 位移误差 | 应力误差 |  Time(s) |
    |----:|:-----|---------:|---------:|---------:|
    |    100 | Weak (Eigh)      |     7.03e-04 |     7.41e-04 |     0.13 |
    |    100 | Weak (Lstsq)     |     7.50e-04 |     7.55e-04 |     0.01 |
    |    100 | Strong (Eigh)    |     3.50e-04 |     1.82e-03 |     0.02 |
    |    100 | Strong (Lstsq)   |     2.38e-03 |     2.23e-02 |     0.01 |
    |    150 | Weak (Eigh)      |     1.58e-04 |     1.34e-04 |     0.03 |
    |    150 | Weak (Lstsq)     |     6.29e-04 |     7.98e-04 |     0.01 |
    |    150 | Strong (Eigh)    |     5.91e-05 |     3.42e-04 |     0.03 |
    |    150 | Strong (Lstsq)   |     1.91e-04 |     1.71e-03 |     0.01 |
    |    200 | Weak (Eigh)      |     1.34e-04 |     9.02e-05 |     0.05 |
    |    200 | Weak (Lstsq)     |     1.62e-04 |     1.91e-04 |     0.02 |
    |    200 | Strong (Eigh)    |     4.15e-05 |     2.42e-04 |     0.05 |
    |    200 | Strong (Lstsq)   |     2.59e-04 |     2.33e-03 |     0.02 |
    |    250 | Weak (Eigh)      |     8.04e-05 |     6.08e-05 |     0.08 |
    |    250 | Weak (Lstsq)     |     1.05e-04 |     1.05e-04 |     0.03 |
    |    250 | Strong (Eigh)    |     3.46e-05 |     1.58e-04 |     0.08 |
    |    250 | Strong (Lstsq)   |     1.69e-04 |     1.55e-03 |     0.03 |
    |    300 | Weak (Eigh)      |     7.68e-05 |     6.26e-05 |     0.12 |
    |    300 | Weak (Lstsq)     |     9.70e-05 |     8.52e-05 |     0.05 |
    |    300 | Strong (Eigh)    |     3.23e-05 |     1.78e-04 |     0.12 |
    |    300 | Strong (Lstsq)   |     3.38e-05 |     3.46e-04 |     0.05 |
  ],
  caption: [平面应力特征数消融实验],
)<tb:ps-ablation>

#figure(
  image("/public/images/penalty-method/plane-stress/ablation/M/ablation-M.png"),
  caption: [平面应力特征数消融实验],
)

当 $M$ 增大时，两种 Eigh 方法整体呈稳定下降趋势。其中 Strong (Eigh) 在整个区间内维持最小或近最小的位移误差，而 Weak (Eigh) 在应力误差上始终最好。`Weak (Lstsq)` 在 $M = 150$ 之后出现明显波动，说明即使在二维问题中，通用最小二乘求解器仍然可能成为限制精度的主因。

== 板弯曲

该算例考虑 Kirchhoff-Love 薄板弯曲，未知量为弯矩张量 $bold(cal(M))$ 与标量挠度 $u$。在区域 $Omega = [0, 1]^2$ 上，其强形式写为
$
  cases(
    bold(cal(A)) : bold(cal(M)) - bold(cal(K))(u) & = 0 & quad "in" Omega,
    nabla dot (nabla dot bold(cal(M))) + f & = 0 & quad "in" Omega,
    u & = 0 & quad "on" partial Omega,
    partial_n u & = 0 & quad "on" partial Omega.
  )
$
这里 $bold(cal(K))(u)$ 表示曲率张量，边界条件为固支边界，即同时约束挠度和法向转角。与前两个二阶模型不同，该问题的平衡方程含有四阶导数，因此更能体现混合变量分离后的优势。详细推导见附录 @app:plate。

板弯曲问题实验参数设置如下：

#figure(
  three-line-table(
    columns: 2,
    align: (right, left),
  )[
    | 项目         | 具体说明 |
    |:-------------|:---------|
    | 算例名称     | 板弯曲 |
    | 计算区域     | $[0, 1]^2$ |
    | 板厚度       | $h = 1$    |
    | 内部采样点   | $Q_"int" = (2^8)^2$ |
    | 边界采样点   | $Q_"bc" = 4 (2^7)$ |
    | 测试采样点   | $Q_"test" = (2^7)^2$ |
    | 罚参数（弱式） | $lambda_0 = lambda_1 = 1$ |
    | 罚参数（强式） | $lambda_0 = lambda_1 = 10$ |
  ],
  caption: [板弯曲实验设置],
)

算例采用固支制造解
$
  u_"ex" (x_1, x_2) = p(x_1) p(x_2),
  quad
  p(t) = t^2 (1 - t)^2.
$
由于 $p(0) = p(1) = p'(0) = p'(1) = 0$，故 $u_"ex" = 0$ 且 $partial_n u_"ex" = 0$ 在边界上成立。精确弯矩 $bold(cal(M))_"ex"$ 由弯矩-曲率关系给出，荷载满足
$
  f_"ex" = D Delta^2 u_"ex",
$
其中 $D = E h^3 \/ (12 (1 - nu^2))$ 为弯曲刚度。

对板弯曲，挠度误差定义为
$
  e_u
  :=
  frac(
    sqrt(frac(1, Q_"test") sum_(p=1)^(Q_"test") abs(bold(Phi)^u (bold(x)_p) - u_"ex" (bold(x)_p))^2),
    sqrt(frac(1, Q_"test") sum_(p=1)^(Q_"test") abs(u_"ex" (bold(x)_p))^2)
  ),
$
而在 Voigt 顺序 $(11, 22, 12)$ 下，对弯矩采用同样的权重 $bold(w)^"V" = (1, 1, 2)^T$：
$
  e_bold(cal(M))
  :=
  frac(
    sqrt(frac(1, Q_"test") sum_(p=1)^(Q_"test") sum_(alpha=1)^3 w^"V"_alpha ((bold(Phi)^(bold(cal(M))) (bold(x)_p))_alpha - (bold(cal(M))_"ex" (bold(x)_p))_alpha)^2),
    sqrt(frac(1, Q_"test") sum_(p=1)^(Q_"test") sum_(alpha=1)^3 w^"V"_alpha ((bold(cal(M))_"ex" (bold(x)_p))_alpha)^2)
  ).
$

=== 实验结果

实验结果见 @tb:pb-main。

#figure(
  three-line-table(
    columns: 4,
    align: (left, right, right, right),
  )[
    | 方法 | 挠度误差 | 弯矩误差 | Time(s) |
    |:-----|---------:|---------:|--------:|
    | Weak (Eigh)   | 1.02e-04 | 5.68e-04 | 0.16 |
    | Weak (Lstsq)  | 5.29e-04 | 3.73e-03 | 0.03 |
    | Strong (Eigh) | 1.05e-03 | 8.09e-03 | 0.08 |
    | Strong (Lstsq)| 2.45e-03 | 3.27e-02 | 0.03 |
  ],
  caption: [板弯曲主实验结果（$M = 300$）],
)<tb:pb-main>

#figure(
  image("/public/images/penalty-method/plate-bending/l2-error-summary.png"),
  caption: [板弯曲主实验结果（$M = 300$）],
)

与前两个算例不同，在板弯曲问题中，边界罚化混合弱方法在挠度和弯矩两项指标上都明显优于强形式对照。这表明当模型含有更高阶几何量且边界条件同时包含 $u$ 与 $partial_n u$ 两种约束时，弱式分离本构与平衡的优势更加明显。

=== 特征数消融

特征数消融结果见 @tb:pb-ablation。

#figure(
  three-line-table(
    columns: 5,
    align: (right, left, right, right, right),
  )[
    | $M$ | 方法 | 挠度误差 | 弯矩误差 |  Time(s) |
    |----:|:-----|---------:|---------:|---------:|
    |    100 | Weak (Eigh)      |     8.96e-04 |     3.92e-03 |     0.13 |
    |    100 | Weak (Lstsq)     |     1.46e-03 |     9.89e-03 |     0.01 |
    |    100 | Strong (Eigh)    |     5.20e-03 |     2.45e-02 |     0.01 |
    |    100 | Strong (Lstsq)   |     1.39e-02 |     8.50e-02 |     0.01 |
    |    150 | Weak (Eigh)      |     2.02e-04 |     9.80e-04 |     0.02 |
    |    150 | Weak (Lstsq)     |     4.20e-04 |     2.87e-03 |     0.01 |
    |    150 | Strong (Eigh)    |     1.94e-03 |     1.04e-02 |     0.02 |
    |    150 | Strong (Lstsq)   |     2.95e-03 |     2.51e-02 |     0.01 |
    |    200 | Weak (Eigh)      |     1.98e-04 |     1.03e-03 |     0.03 |
    |    200 | Weak (Lstsq)     |     1.60e-03 |     7.16e-03 |     0.01 |
    |    200 | Strong (Eigh)    |     1.63e-03 |     1.08e-02 |     0.03 |
    |    200 | Strong (Lstsq)   |     3.72e-03 |     4.19e-02 |     0.01 |
    |    250 | Weak (Eigh)      |     1.12e-04 |     8.79e-04 |     0.05 |
    |    250 | Weak (Lstsq)     |     3.27e-04 |     1.55e-03 |     0.02 |
    |    250 | Strong (Eigh)    |     1.05e-03 |     7.78e-03 |     0.05 |
    |    250 | Strong (Lstsq)   |     2.37e-03 |     2.66e-02 |     0.02 |
    |    300 | Weak (Eigh)      |     1.02e-04 |     5.68e-04 |     0.07 |
    |    300 | Weak (Lstsq)     |     5.29e-04 |     3.73e-03 |     0.03 |
    |    300 | Strong (Eigh)    |     1.05e-03 |     8.09e-03 |     0.07 |
    |    300 | Strong (Lstsq)   |     2.45e-03 |     3.27e-02 |     0.03 |
  ],
  caption: [板弯曲特征数消融实验],
)<tb:pb-ablation>

#figure(
  image("/public/images/penalty-method/plate-bending/ablation/M/ablation-M.png"),
  caption: [板弯曲特征数消融实验],
)

在整个特征数区间内，Weak (Eigh) 都保持了最好的综合表现。Strong (Eigh) 虽然随 $M$ 增大也有改善，但改善幅度明显小于弱式方法；两个 Lstsq 版本则始终存在数量级上的误差劣势。这说明对于 $nabla dot (nabla dot dot)$ 型问题，保留混合弱结构比直接在强形式残差上做最小二乘更合适。

= 结语

本文提出了一个边界罚化的混合弱随机特征框架，并以强形式最小二乘作为对照方法。在三维线弹性问题、二维平面应力问题以及 Kirchhoff-Love 板弯曲问题上的数值实验说明：当关注位移或挠度等原变量时，强形式在部分低阶模型上可以取得更小误差；当关注应力、弯矩等共轭张量场时，边界罚化混合弱方法整体更稳健，且这一优势在高阶板弯曲问题中更加明显。换言之，将本构关系与平衡关系在弱形式下显式分离，更适合在固定随机特征空间中恢复辅助物理量，并更好地保留问题本身的力学结构。

另一方面，求解器选择并非实现层面的次要问题，而是影响方法表现的关键因素。三个算例都表明：保留块对称结构并结合谱截断的 Eigh 求解版本整体稳定性最好；相较之下，直接使用通用最小二乘求解器时，误差会随着特征维度增大出现更明显的非单调性，尤其在三维问题和板弯曲问题中更为突出。因此，若希望将随机特征方法用于更复杂的连续介质模型，仅有合适的离散形式还不够，还需要匹配结构保持的线性代数求解策略。

本文工作仍有若干值得继续推进的方向，例如边界罚参数的自适应选取、离散误差与条件数之间关系的理论分析，以及面向更高维、更复杂几何区域的采样与预条件设计。本文的数值结果表明，边界罚化混合弱随机特征方法为在线性力学问题中构造低训练成本、结构清晰且物理量恢复稳定的近似框架提供了一条可行路径。


#bibliography("/public/reference/penalty-method.bib")

#set heading(numbering: "附录 A.1", supplement: [附录])
#counter(heading).update(0)

= 三维线弹性问题的具体推导 <app:3d>

== 连续模型与混合弱形式

设 $Omega subset RR^3$，未知量为对称应力张量 $bold(sigma)$ 与位移向量 $bold(u)$。三维线弹性的混合强形式为
$
  cases(
    bold(cal(A)) : bold(sigma) - bold(epsilon)(bold(u)) & = 0 & quad "in" Omega,
    nabla dot bold(sigma) + bold(f) & = 0 & quad "in" Omega,
    bold(u) & = 0 & quad "on" partial Omega.
  )
$
其中
$
  bold(epsilon)(bold(u)) = 1/2 (nabla bold(u) + (nabla bold(u))^T).
$
取
$
  bold(Sigma) := bold(H)(div, Omega; SS^3),
  quad
  bold(U) := (L^2(Omega))^3.
$
定义双线性形式
$
  a(bold(sigma), bold(tau))
  := integral_Omega (bold(cal(A)) : bold(sigma)) : bold(tau) dif x,
$
以及
$
  b(bold(tau), bold(v))
  := integral_Omega (nabla dot bold(tau)) dot bold(v) dif x.
$
对本构方程与测试函数 $bold(tau) in bold(Sigma)$ 做内积并积分，再利用齐次位移边界条件分部积分，可得
$
  a(bold(sigma), bold(tau)) + b(bold(tau), bold(u)) = 0.
$
对平衡方程与 $bold(v) in bold(U)$ 做内积并积分，得到
$
  b(bold(sigma), bold(v)) = - (bold(f), bold(v)).
$
因此三维线弹性的混合弱形式为：求 $(bold(sigma), bold(u)) in bold(Sigma) times bold(U)$，使得
$
  cases(
    a(bold(sigma), bold(tau)) + b(bold(tau), bold(u)) & = 0 & quad forall bold(tau) in bold(Sigma),
    b(bold(sigma), bold(v)) & = - (bold(f), bold(v)) & quad forall bold(v) in bold(U).
  )
$

== 边界罚项

离散位移空间不天然满足 $bold(u)=0$，因此引入边界罚项
$
  c (bold(u), bold(v))
  := lambda_"bc" integral_(partial Omega) bold(u) dot bold(v) dif s.
$
于是罚化弱离散写为
$
  cases(
    a(bold(Phi)^(bold(sigma)), bold(Phi)^(bold(tau))) + b(bold(Phi)^(bold(tau)), bold(Phi)^(bold(u))) & = 0 & quad forall bold(Phi)^(bold(tau)) in bold(Sigma)_M,
    b(bold(Phi)^(bold(sigma)), bold(Phi)^(bold(v))) + c (bold(Phi)^(bold(u)), bold(Phi)^(bold(v))) & = - (bold(f), bold(Phi)^(bold(v))) & quad forall bold(Phi)^(bold(v)) in bold(U)_M.
  )
$

== 随机特征离散

记 $Xi_M^(bold(sigma))$ 与 $Xi_M^(bold(u))$ 分别为应力和位移所用的标量随机特征空间，并令 ${bold(E)_alpha}_(alpha=1)^6$ 为对称矩阵标准基，按 $(11,22,33,12,23,13)$ 排列，${bold(e)_i}_(i=1)^3$ 为 $RR^3$ 标准基。定义
$
  bold(Sigma)_M
  := span { xi^(bold(sigma))_m bold(E)_alpha : 0 <= m <= M, 1 <= alpha <= 6 },
$
以及
$
  bold(U)_M
  := span { xi^(bold(u))_m bold(e)_i : 0 <= m <= M, 1 <= i <= 3 }.
$
将离散解展开为
$
  bold(Phi)^(bold(sigma)) = sum_(m=0)^M sum_(alpha=1)^6 phi^(bold(sigma))_(m, alpha) xi^(bold(sigma))_m bold(E)_alpha,
$
以及
$
  bold(Phi)^(bold(u)) = sum_(m=0)^M sum_(i=1)^3 phi^(bold(u))_(m, i) xi^(bold(u))_m bold(e)_i.
$

== 从变分方程到块线性系统

取测试函数为同一组基函数：
$
  bold(Phi)^(bold(tau)) = xi^(bold(sigma))_n bold(E)_beta, quad 0 <= n <= M, 1 <= beta <= 6,
$
以及
$
  bold(Phi)^(bold(v)) = xi^(bold(u))_n bold(e)_j, quad 0 <= n <= M, 1 <= j <= 3.
$
这里约定第一组下标对应测试函数，也就是矩阵块的行指标；第二组下标对应展开系数，也就是矩阵块的列指标。

定义块矩阵
$
  A_((n, beta), (m, alpha))
  := a(xi^(bold(sigma))_n bold(E)_beta, xi^(bold(sigma))_m bold(E)_alpha),
$
$
  B_((n, beta), (m, i))
  := b(xi^(bold(sigma))_n bold(E)_beta, xi^(bold(u))_m bold(e)_i),
$
$
  C_((n, j), (m, i))
  := c (xi^(bold(u))_n bold(e)_j, xi^(bold(u))_m bold(e)_i),
$
$
  F_((n, j)) := (bold(f), xi^(bold(u))_n bold(e)_j).
$

对任意固定的 $(n, beta)$，取测试函数 $bold(Phi)^(bold(tau)) = xi^(bold(sigma))_n bold(E)_beta$。第一条离散变分方程为
$
  0 = a(bold(Phi)^(bold(sigma)), xi^(bold(sigma))_n bold(E)_beta) + b(xi^(bold(sigma))_n bold(E)_beta, bold(Phi)^(bold(u))).
$
将
$
  bold(Phi)^(bold(sigma)) = sum_(m=0)^M sum_(alpha=1)^6 phi^(bold(sigma))_(m, alpha) xi^(bold(sigma))_m bold(E)_alpha
$
与
$
  bold(Phi)^(bold(u)) = sum_(m=0)^M sum_(i=1)^3 phi^(bold(u))_(m, i) xi^(bold(u))_m bold(e)_i
$
代入，并利用双线性性，可得
$
  0
  = sum_(m=0)^M sum_(alpha=1)^6 phi^(bold(sigma))_(m, alpha) a(xi^(bold(sigma))_n bold(E)_beta, xi^(bold(sigma))_m bold(E)_alpha)
  + sum_(m=0)^M sum_(i=1)^3 phi^(bold(u))_(m, i) b(xi^(bold(sigma))_n bold(E)_beta, xi^(bold(u))_m bold(e)_i).
$
将两项分别识别为矩阵元素，即
$
  sum_(m=0)^M sum_(alpha=1)^6 A_((n, beta), (m, alpha)) phi^(bold(sigma))_(m, alpha)
  + sum_(m=0)^M sum_(i=1)^3 B_((n, beta), (m, i)) phi^(bold(u))_(m, i) = 0.
$
这正对应于
$
  A bold(phi)^(bold(sigma)) + B bold(phi)^(bold(u)) = 0
$
的第 $(n, beta)$ 个分量。

同理，对任意固定的 $(n, j)$，取测试函数 $bold(Phi)^(bold(v)) = xi^(bold(u))_n bold(e)_j$。第二条离散变分方程为
$
  0 = b(bold(Phi)^(bold(sigma)), xi^(bold(u))_n bold(e)_j) + c (bold(Phi)^(bold(u)), xi^(bold(u))_n bold(e)_j) + (bold(f), xi^(bold(u))_n bold(e)_j).
$
代入展开式可得
$
  0
  = sum_(m=0)^M sum_(alpha=1)^6 phi^(bold(sigma))_(m, alpha) b(xi^(bold(sigma))_m bold(E)_alpha, xi^(bold(u))_n bold(e)_j)
  + sum_(m=0)^M sum_(i=1)^3 phi^(bold(u))_(m, i) c (xi^(bold(u))_m bold(e)_i, xi^(bold(u))_n bold(e)_j)
  + F_((n, j)).
$
注意到
$
  b(xi^(bold(sigma))_m bold(E)_alpha, xi^(bold(u))_n bold(e)_j) = B_((m, alpha), (n, j)),
$
以及
$
  c (xi^(bold(u))_m bold(e)_i, xi^(bold(u))_n bold(e)_j) = C_((n, j), (m, i)),
$
因此可写为
$
  sum_(m=0)^M sum_(alpha=1)^6 B_((m, alpha), (n, j)) phi^(bold(sigma))_(m, alpha)
  + sum_(m=0)^M sum_(i=1)^3 C_((n, j), (m, i)) phi^(bold(u))_(m, i)
  = - F_((n, j)).
$
这正对应于
$
  B^T bold(phi)^(bold(sigma)) + C bold(phi)^(bold(u)) = -F
$
的第 $(n, j)$ 个分量。

若将 $A$、$B$、$C$ 视为 $(M+1) times (M+1)$ 块矩阵，$F$ 视为 $(M+1) times 1$ 块向量，则
$
  A =
  mat(
    A_(0, 0), dots.h, A_(0, M);
    dots.v, dots.down, dots.v;
    A_(M, 0), dots.h, A_(M, M)
  ), quad & B =
            mat(
              B_(0, 0), dots.h, B_(0, M);
              dots.v, dots.down, dots.v;
              B_(M, 0), dots.h, B_(M, M)
            ), \
  C =
  mat(
    C_(0, 0), dots.h, C_(0, M);
    dots.v, dots.down, dots.v;
    C_(M, 0), dots.h, C_(M, M)
  ), quad & F =
            mat(
              F_0;
              dots.v;
              F_M
            ).
$
其中块内指标满足
$
  A_(n, m) in RR^(6 times 6), quad (A_(n, m))_(beta, alpha) = A_((n, beta), (m, alpha)),
$
$
  B_(n, m) in RR^(6 times 3), quad (B_(n, m))_(beta, i) = B_((n, beta), (m, i)),
$
$
  C_(n, m) in RR^(3 times 3), quad (C_(n, m))_(j, i) = C_((n, j), (m, i)),
$
$
  F_n in RR^3, quad (F_n)_j = F_((n, j)).
$
若再记
$
  bold(phi)^(bold(sigma))_m := (phi^(bold(sigma))_(m, 1), dots.c, phi^(bold(sigma))_(m, 6))^T in RR^6,
  quad
  bold(phi)^(bold(u))_m := (phi^(bold(u))_(m, 1), dots.c, phi^(bold(u))_(m, 3))^T in RR^3,
$
则 $bold(phi)^(bold(sigma)) = (bold(phi)^(bold(sigma))_0, dots.c, bold(phi)^(bold(sigma))_M)^T$，$bold(phi)^(bold(u)) = (bold(phi)^(bold(u))_0, dots.c, bold(phi)^(bold(u))_M)^T$，并且两条方程可按块写成
$
  sum_(m=0)^M A_(n, m) bold(phi)^(bold(sigma))_m + sum_(m=0)^M B_(n, m) bold(phi)^(bold(u))_m = 0,
$
以及
$
  sum_(m=0)^M B_(m, n)^T bold(phi)^(bold(sigma))_m + sum_(m=0)^M C_(n, m) bold(phi)^(bold(u))_m = -F_n, quad 0 <= n <= M.
$

== $A$ 块元素

由定义
$
  A_((n, beta), (m, alpha))
  := a(xi^(bold(sigma))_n bold(E)_beta, xi^(bold(sigma))_m bold(E)_alpha)
  = integral_Omega (bold(cal(A)) : (xi^(bold(sigma))_n bold(E)_beta)) : (xi^(bold(sigma))_m bold(E)_alpha) dif x.
$
由于 $xi^(bold(sigma))_n$ 与 $xi^(bold(sigma))_m$ 为标量函数，$bold(E)_beta$ 与 $bold(E)_alpha$ 为常矩阵，可得
$
  A_((n, beta), (m, alpha))
  = integral_Omega xi^(bold(sigma))_n (bold(x)) xi^(bold(sigma))_m (bold(x))
  ((bold(cal(A))(bold(x)) : bold(E)_beta) : bold(E)_alpha) dif x.
$
若将双重收缩写成坐标形式，则
$
  ((bold(cal(A)) : bold(E)_beta) : bold(E)_alpha)
  = sum_(i=1)^3 sum_(j=1)^3 sum_(k=1)^3 sum_(l=1)^3
  bold(cal(A))_(i j k l) (bold(E)_beta)_(k l) (bold(E)_alpha)_(i j).
$
因此
$
  A_((n, beta), (m, alpha))
  = integral_Omega xi^(bold(sigma))_n xi^(bold(sigma))_m
  sum_(i=1)^3 sum_(j=1)^3 sum_(k=1)^3 sum_(l=1)^3
  bold(cal(A))_(i j k l) (bold(E)_beta)_(k l) (bold(E)_alpha)_(i j) dif x.
$

== $B$ 块元素

由定义
$
  B_((n, beta), (m, i))
  := b(xi^(bold(sigma))_n bold(E)_beta, xi^(bold(u))_m bold(e)_i)
  = integral_Omega (nabla dot (xi^(bold(sigma))_n bold(E)_beta)) dot (xi^(bold(u))_m bold(e)_i) dif x.
$
利用张量散度的分量定义 $(nabla dot bold(tau))_p = tau_(p k, k)$，令 $bold(tau) = xi^(bold(sigma))_n bold(E)_beta$，则
$
  (nabla dot (xi^(bold(sigma))_n bold(E)_beta))_i
  = (xi^(bold(sigma))_n (bold(E)_beta)_(i k))_(,k)
  = (bold(E)_beta)_(i k) partial_k xi^(bold(sigma))_n.
$
也就是说
$
  nabla dot (xi^(bold(sigma))_n bold(E)_beta) = bold(E)_beta nabla xi^(bold(sigma))_n.
$
于是
$
  B_((n, beta), (m, i))
  = integral_Omega xi^(bold(u))_m (bold(E)_beta nabla xi^(bold(sigma))_n)_i dif x
  = integral_Omega xi^(bold(u))_m sum_(k=1)^3 (bold(E)_beta)_(i k) partial_k xi^(bold(sigma))_n dif x.
$
记
$
  nabla xi^(bold(sigma))_n
  = (partial_1 xi^(bold(sigma))_n, partial_2 xi^(bold(sigma))_n, partial_3 xi^(bold(sigma))_n)^T,
$
则按对称基顺序 $(11,22,33,12,23,13)$ 有
$
  cases(
    bold(E)_1 nabla xi^(bold(sigma))_n = (partial_1 xi^(bold(sigma))_n, 0, 0)^T,
    bold(E)_2 nabla xi^(bold(sigma))_n = (0, partial_2 xi^(bold(sigma))_n, 0)^T,
    bold(E)_3 nabla xi^(bold(sigma))_n = (0, 0, partial_3 xi^(bold(sigma))_n)^T,
    bold(E)_4 nabla xi^(bold(sigma))_n = (partial_2 xi^(bold(sigma))_n, partial_1 xi^(bold(sigma))_n, 0)^T,
    bold(E)_5 nabla xi^(bold(sigma))_n = (0, partial_3 xi^(bold(sigma))_n, partial_2 xi^(bold(sigma))_n)^T,
    bold(E)_6 nabla xi^(bold(sigma))_n = (partial_3 xi^(bold(sigma))_n, 0, partial_1 xi^(bold(sigma))_n)^T.
  )
$

对于 $n >= 1$ 的随机特征
$
  xi^(bold(sigma))_n (bold(x)) = rho(bold(w)_n^T bold(x) + b_n),
$
其梯度可直接写为
$
  nabla xi^(bold(sigma))_n (bold(x))
  = rho'(bold(w)_n^T bold(x) + b_n) bold(w)_n.
$
若采用重参数化 $bold(w)_n = gamma bold(a)_n$ 与 $b_n = gamma r_n$，则等价地
$
  nabla xi^(bold(sigma))_n (bold(x))
  = rho'(gamma (bold(a)_n^T bold(x) + r_n)) gamma bold(a)_n.
$
特别地，$xi^(bold(sigma))_0 = 1$，故 $nabla xi^(bold(sigma))_0 = 0$，从而所有以 $n=0$ 为测试函数的 $B$ 行元素都为 $0$。

== $C$ 块元素

由定义
$
  C_((n, j), (m, i))
  := c (xi^(bold(u))_n bold(e)_j, xi^(bold(u))_m bold(e)_i)
  = lambda_"bc" integral_(partial Omega) (xi^(bold(u))_n bold(e)_j) dot (xi^(bold(u))_m bold(e)_i) dif s.
$
由于 $bold(e)_j dot bold(e)_i = delta_(j i)$，故
$
  C_((n, j), (m, i))
  = lambda_"bc" integral_(partial Omega)
  xi^(bold(u))_n (bold(x)) xi^(bold(u))_m (bold(x)) delta_(j i) dif s.
$

== $F$ 载荷向量元素

由定义
$
  F_((n, j))
  := (bold(f), xi^(bold(u))_n bold(e)_j)
  = integral_Omega bold(f) dot (xi^(bold(u))_n bold(e)_j) dif x.
$
按分量写开即
$
  F_((n, j)) = integral_Omega f_j (bold(x)) xi^(bold(u))_n (bold(x)) dif x.
$


= 平面应力问题的具体推导 <app:plane-stress>

== 由平面应力假设到混合弱形式

设 $Omega subset RR^2$，薄板上下表面自由，并满足平面应力假设
$
  sigma_(13) = sigma_(23) = sigma_(33) = 0.
$
于是只保留板面内位移
$
  bold(u) = (u_1, u_2)^T : Omega -> RR^2.
$
二维线性应变张量为
$
  bold(epsilon)(bold(u)) = 1/2 (nabla bold(u) + (nabla bold(u))^T).
$
对各向同性材料，平面应力本构可写为
$
  sigma_(alpha beta)
  = 2 mu epsilon_(alpha beta)(bold(u))
  + lambda_"ps" epsilon_(gamma gamma)(bold(u)) delta_(alpha beta),
  quad alpha, beta, gamma = 1, 2,
$
其中
$
  mu = E / (2 (1 + nu)),
  quad
  lambda_"ps" = E nu / (1 - nu^2).
$
记柔度张量为 $bold(cal(A)) = bold(cal(C))^(-1)$，则混合强形式写为
$
  cases(
    bold(cal(A)) : bold(sigma) - bold(epsilon)(bold(u)) & = 0 & quad "in" Omega,
    nabla dot bold(sigma) + bold(f) & = 0 & quad "in" Omega,
    bold(u) & = 0 & quad "on" partial Omega.
  )
$
取
$
  bold(Sigma) := bold(H)(div, Omega; SS^2),
  quad
  bold(U) := (L^2(Omega))^2.
$
定义
$
  a(bold(sigma), bold(tau))
  := integral_Omega (bold(cal(A)) : bold(sigma)) : bold(tau) dif x,
$
以及
$
  b(bold(tau), bold(v))
  := integral_Omega (nabla dot bold(tau)) dot bold(v) dif x.
$
与三维情形相同，经一次分部积分后可得混合弱形式
$
  cases(
    a(bold(sigma), bold(tau)) + b(bold(tau), bold(u)) & = 0 & quad forall bold(tau) in bold(Sigma),
    b(bold(sigma), bold(v)) & = - (bold(f), bold(v)) & quad forall bold(v) in bold(U).
  )
$

== 边界罚项

边界罚项取为
$
  c (bold(u), bold(v))
  := lambda_"bc" integral_(partial Omega) bold(u) dot bold(v) dif s.
$
于是离散弱式写为
$
  cases(
    a(bold(Phi)^(bold(sigma)), bold(Phi)^(bold(tau))) + b(bold(Phi)^(bold(tau)), bold(Phi)^(bold(u))) & = 0 & quad forall bold(Phi)^(bold(tau)) in bold(Sigma)_M,
    b(bold(Phi)^(bold(sigma)), bold(Phi)^(bold(v))) + c (bold(Phi)^(bold(u)), bold(Phi)^(bold(v))) & = - (bold(f), bold(Phi)^(bold(v))) & quad forall bold(Phi)^(bold(v)) in bold(U)_M.
  )
$

== 随机特征离散

取 ${bold(E)_alpha}_(alpha=1)^3$ 为二维对称基，按 $(11,22,12)$ 排列，${bold(e)_i}_(i=1)^2$ 为 $RR^2$ 标准基。定义
$
  bold(Sigma)_M
  := span { xi^(bold(sigma))_m bold(E)_alpha : 0 <= m <= M, 1 <= alpha <= 3 },
$
以及
$
  bold(U)_M
  := span { xi^(bold(u))_m bold(e)_i : 0 <= m <= M, 1 <= i <= 2 }.
$
将离散解展开为
$
  bold(Phi)^(bold(sigma)) = sum_(m=0)^M sum_(alpha=1)^3 phi^(bold(sigma))_(m, alpha) xi^(bold(sigma))_m bold(E)_alpha,
$
以及
$
  bold(Phi)^(bold(u)) = sum_(m=0)^M sum_(i=1)^2 phi^(bold(u))_(m, i) xi^(bold(u))_m bold(e)_i.
$

== 从变分方程到块线性系统

取测试函数为同一组基函数：
$
  bold(Phi)^(bold(tau)) = xi^(bold(sigma))_n bold(E)_beta, quad 0 <= n <= M, 1 <= beta <= 3,
$
以及
$
  bold(Phi)^(bold(v)) = xi^(bold(u))_n bold(e)_j, quad 0 <= n <= M, 1 <= j <= 2.
$
这里约定第一组下标对应测试函数，也就是矩阵块的行指标；第二组下标对应展开系数，也就是矩阵块的列指标。

定义块矩阵
$
  A_((n, beta), (m, alpha))
  := a(xi^(bold(sigma))_n bold(E)_beta, xi^(bold(sigma))_m bold(E)_alpha),
$
$
  B_((n, beta), (m, i))
  := b(xi^(bold(sigma))_n bold(E)_beta, xi^(bold(u))_m bold(e)_i),
$
$
  C_((n, j), (m, i))
  := c (xi^(bold(u))_n bold(e)_j, xi^(bold(u))_m bold(e)_i),
$
$
  F_((n, j)) := (bold(f), xi^(bold(u))_n bold(e)_j).
$

对任意固定的 $(n, beta)$，取测试函数 $bold(Phi)^(bold(tau)) = xi^(bold(sigma))_n bold(E)_beta$。第一条离散变分方程为
$
  0 = a(bold(Phi)^(bold(sigma)), xi^(bold(sigma))_n bold(E)_beta) + b(xi^(bold(sigma))_n bold(E)_beta, bold(Phi)^(bold(u))).
$
将
$
  bold(Phi)^(bold(sigma)) = sum_(m=0)^M sum_(alpha=1)^3 phi^(bold(sigma))_(m, alpha) xi^(bold(sigma))_m bold(E)_alpha, quad
  bold(Phi)^(bold(u)) = sum_(m=0)^M sum_(i=1)^2 phi^(bold(u))_(m, i) xi^(bold(u))_m bold(e)_i
$
代入，并利用双线性性，得到
$
  0
  = sum_(m=0)^M sum_(alpha=1)^3 phi^(bold(sigma))_(m, alpha) a(xi^(bold(sigma))_n bold(E)_beta, xi^(bold(sigma))_m bold(E)_alpha)
  + sum_(m=0)^M sum_(i=1)^2 phi^(bold(u))_(m, i) b(xi^(bold(sigma))_n bold(E)_beta, xi^(bold(u))_m bold(e)_i).
$
将两项分别识别为矩阵元素，即
$
  sum_(m=0)^M sum_(alpha=1)^3 A_((n, beta), (m, alpha)) phi^(bold(sigma))_(m, alpha)
  + sum_(m=0)^M sum_(i=1)^2 B_((n, beta), (m, i)) phi^(bold(u))_(m, i) = 0.
$
这正对应于
$
  A bold(phi)^(bold(sigma)) + B bold(phi)^(bold(u)) = 0
$
的第 $(n, beta)$ 个分量。

同理，对任意固定的 $(n, j)$，取测试函数 $bold(Phi)^(bold(v)) = xi^(bold(u))_n bold(e)_j$。第二条离散变分方程为
$
  0 = b(bold(Phi)^(bold(sigma)), xi^(bold(u))_n bold(e)_j) + c (bold(Phi)^(bold(u)), xi^(bold(u))_n bold(e)_j) + (bold(f), xi^(bold(u))_n bold(e)_j).
$
代入展开式可得
$
  0
  = sum_(m=0)^M sum_(alpha=1)^3 phi^(bold(sigma))_(m, alpha) b(xi^(bold(sigma))_m bold(E)_alpha, xi^(bold(u))_n bold(e)_j)
  + sum_(m=0)^M sum_(i=1)^2 phi^(bold(u))_(m, i) c (xi^(bold(u))_m bold(e)_i, xi^(bold(u))_n bold(e)_j)
  + F_((n, j)).
$
注意到
$
  b(xi^(bold(sigma))_m bold(E)_alpha, xi^(bold(u))_n bold(e)_j) = B_((m, alpha), (n, j)),
$
以及
$
  c (xi^(bold(u))_m bold(e)_i, xi^(bold(u))_n bold(e)_j) = C_((n, j), (m, i)),
$
因此可写为
$
  sum_(m=0)^M sum_(alpha=1)^3 B_((m, alpha), (n, j)) phi^(bold(sigma))_(m, alpha)
  + sum_(m=0)^M sum_(i=1)^2 C_((n, j), (m, i)) phi^(bold(u))_(m, i)
  = - F_((n, j)).
$
这正对应于
$
  B^T bold(phi)^(bold(sigma)) + C bold(phi)^(bold(u)) = -F
$
的第 $(n, j)$ 个分量。

若将 $A$、$B$、$C$ 视为 $(M+1) times (M+1)$ 块矩阵，$F$ 视为 $(M+1) times 1$ 块向量，则
$
  A =
  mat(
    A_(0, 0), dots.h, A_(0, M);
    dots.v, dots.down, dots.v;
    A_(M, 0), dots.h, A_(M, M)
  ), quad & B =
            mat(
              B_(0, 0), dots.h, B_(0, M);
              dots.v, dots.down, dots.v;
              B_(M, 0), dots.h, B_(M, M)
            ), \
  C =
  mat(
    C_(0, 0), dots.h, C_(0, M);
    dots.v, dots.down, dots.v;
    C_(M, 0), dots.h, C_(M, M)
  ), quad & F =
            mat(
              F_0;
              dots.v;
              F_M
            ).
$
其中块内指标满足
$
  A_(n, m) in RR^(3 times 3), quad (A_(n, m))_(beta, alpha) = A_((n, beta), (m, alpha)),
$
$
  B_(n, m) in RR^(3 times 2), quad (B_(n, m))_(beta, i) = B_((n, beta), (m, i)),
$
$
  C_(n, m) in RR^(2 times 2), quad (C_(n, m))_(j, i) = C_((n, j), (m, i)),
$
$
  F_n in RR^2, quad (F_n)_j = F_((n, j)).
$
若再记
$
  bold(phi)^(bold(sigma))_m := (phi^(bold(sigma))_(m, 1), dots.c, phi^(bold(sigma))_(m, 3))^T in RR^3,
  quad
  bold(phi)^(bold(u))_m := (phi^(bold(u))_(m, 1), dots.c, phi^(bold(u))_(m, 2))^T in RR^2,
$
则 $bold(phi)^(bold(sigma)) = (bold(phi)^(bold(sigma))_0, dots.c, bold(phi)^(bold(sigma))_M)^T$，$bold(phi)^(bold(u)) = (bold(phi)^(bold(u))_0, dots.c, bold(phi)^(bold(u))_M)^T$，并且两条方程可按块写成
$
  sum_(m=0)^M A_(n, m) bold(phi)^(bold(sigma))_m + sum_(m=0)^M B_(n, m) bold(phi)^(bold(u))_m = 0,
$
以及
$
  sum_(m=0)^M B_(m, n)^T bold(phi)^(bold(sigma))_m + sum_(m=0)^M C_(n, m) bold(phi)^(bold(u))_m = -F_n, quad 0 <= n <= M.
$

== $A$ 块元素

由定义
$
  A_((n, beta), (m, alpha))
  := a(xi^(bold(sigma))_n bold(E)_beta, xi^(bold(sigma))_m bold(E)_alpha)
  = integral_Omega (bold(cal(A)) : (xi^(bold(sigma))_n bold(E)_beta)) : (xi^(bold(sigma))_m bold(E)_alpha) dif x.
$
由于 $xi^(bold(sigma))_n$ 与 $xi^(bold(sigma))_m$ 为标量函数，$bold(E)_beta$ 与 $bold(E)_alpha$ 为常矩阵，可得
$
  A_((n, beta), (m, alpha))
  = integral_Omega xi^(bold(sigma))_n (bold(x)) xi^(bold(sigma))_m (bold(x))
  ((bold(cal(A))(bold(x)) : bold(E)_beta) : bold(E)_alpha) dif x.
$
若将双重收缩写成坐标形式，则
$
  ((bold(cal(A)) : bold(E)_beta) : bold(E)_alpha)
  = sum_(i=1)^2 sum_(j=1)^2 sum_(k=1)^2 sum_(l=1)^2
  bold(cal(A))_(i j k l) (bold(E)_beta)_(k l) (bold(E)_alpha)_(i j).
$
因此
$
  A_((n, beta), (m, alpha))
  = integral_Omega xi^(bold(sigma))_n xi^(bold(sigma))_m
  sum_(i=1)^2 sum_(j=1)^2 sum_(k=1)^2 sum_(l=1)^2
  bold(cal(A))_(i j k l) (bold(E)_beta)_(k l) (bold(E)_alpha)_(i j)
  dif x.
$

== $B$ 块元素

由定义
$
  B_((n, beta), (m, i))
  := b(xi^(bold(sigma))_n bold(E)_beta, xi^(bold(u))_m bold(e)_i)
  = integral_Omega (nabla dot (xi^(bold(sigma))_n bold(E)_beta)) dot (xi^(bold(u))_m bold(e)_i) dif x.
$
利用张量散度的分量定义 $(nabla dot bold(tau))_p = tau_(p k, k)$，令 $bold(tau) = xi^(bold(sigma))_n bold(E)_beta$，则
$
  (nabla dot (xi^(bold(sigma))_n bold(E)_beta))_i
  = (xi^(bold(sigma))_n (bold(E)_beta)_(i k))_(,k)
  = (bold(E)_beta)_(i k) partial_k xi^(bold(sigma))_n.
$
也就是说
$
  nabla dot (xi^(bold(sigma))_n bold(E)_beta) = bold(E)_beta nabla xi^(bold(sigma))_n.
$
于是
$
  B_((n, beta), (m, i))
  = integral_Omega xi^(bold(u))_m (bold(E)_beta nabla xi^(bold(sigma))_n)_i dif x
  = integral_Omega xi^(bold(u))_m sum_(k=1)^2 (bold(E)_beta)_(i k) partial_k xi^(bold(sigma))_n dif x.
$
记
$
  nabla xi^(bold(sigma))_n
  = (partial_1 xi^(bold(sigma))_n, partial_2 xi^(bold(sigma))_n)^T,
$
则按对称基顺序 $(11,22,12)$ 有
$
  cases(
    bold(E)_1 nabla xi^(bold(sigma))_n = (partial_1 xi^(bold(sigma))_n, 0)^T,
    bold(E)_2 nabla xi^(bold(sigma))_n = (0, partial_2 xi^(bold(sigma))_n)^T,
    bold(E)_3 nabla xi^(bold(sigma))_n = (partial_2 xi^(bold(sigma))_n, partial_1 xi^(bold(sigma))_n)^T.
  )
$

对于 $n >= 1$ 的随机特征
$
  xi^(bold(sigma))_n (bold(x)) = rho(bold(w)_n^T bold(x) + b_n),
$
其梯度可直接写为
$
  nabla xi^(bold(sigma))_n (bold(x))
  = rho'(bold(w)_n^T bold(x) + b_n) bold(w)_n.
$
若采用重参数化 $bold(w)_n = gamma bold(a)_n$ 与 $b_n = gamma r_n$，则等价地
$
  nabla xi^(bold(sigma))_n (bold(x))
  = rho'(gamma (bold(a)_n^T bold(x) + r_n)) gamma bold(a)_n.
$
特别地，$xi^(bold(sigma))_0 = 1$，故 $nabla xi^(bold(sigma))_0 = 0$，从而所有以 $n=0$ 为测试函数的 $B$ 行元素都为 $0$。

== $C$ 块元素

由定义
$
  C_((n, j), (m, i))
  := c (xi^(bold(u))_n bold(e)_j, xi^(bold(u))_m bold(e)_i)
  = lambda_"bc" integral_(partial Omega) (xi^(bold(u))_n bold(e)_j) dot (xi^(bold(u))_m bold(e)_i) dif s.
$
由于 $bold(e)_j dot bold(e)_i = delta_(j i)$，故
$
  C_((n, j), (m, i))
  = lambda_"bc" integral_(partial Omega)
  xi^(bold(u))_n (bold(x)) xi^(bold(u))_m (bold(x)) delta_(j i) dif s.
$

== $F$ 载荷向量元素

由定义
$
  F_((n, j))
  := (bold(f), xi^(bold(u))_n bold(e)_j)
  = integral_Omega bold(f) dot (xi^(bold(u))_n bold(e)_j) dif x.
$
按分量写开即
$
  F_((n, j)) = integral_Omega f_j (bold(x)) xi^(bold(u))_n (bold(x)) dif x.
$

= Kirchhoff-Love 板弯曲问题的具体推导 <app:plate>

== 从 Kirchhoff-Love 假设到混合弱形式

设 $Omega subset RR^2$ 为板中面区域，板厚为 $h$，挠度记为 $u: Omega -> RR$。Kirchhoff-Love 假设给出三维位移场
$
  cases(
    u_alpha (x_1, x_2, x_3) = - x_3 partial_alpha u(x_1, x_2) & quad alpha = 1\, 2,
    u_3 (x_1, x_2, x_3) = u(x_1, x_2).
  )
$
于是可定义中面曲率张量
$
  bold(cal(K))(u) := - nabla^2 u.
$
对各向同性材料，弯矩-曲率关系写为
$
  bold(cal(M)) = bold(cal(C)) : bold(cal(K))(u),
  quad
  D = E h^3 / (12 (1 - nu^2)),
$
其中 $D$ 为弯曲刚度。引入柔度张量 $bold(cal(A)) = bold(cal(C))^(-1)$ 后，板弯曲的混合强形式为
$
  cases(
    bold(cal(A)) : bold(cal(M)) - bold(cal(K))(u) & = 0 & quad "in" Omega,
    nabla dot (nabla dot bold(cal(M))) + f & = 0 & quad "in" Omega,
    u & = 0 & quad "on" partial Omega,
    partial_n u & = 0 & quad "on" partial Omega.
  )
$
取
$
  bold(Sigma) := bold(H)(div div, Omega; SS^2),
  quad
  U := L^2(Omega).
$
定义
$
  a(bold(cal(M)), bold(tau))
  := integral_Omega (bold(cal(A)) : bold(cal(M))) : bold(tau) dif x,
$
以及
$
  b(bold(tau), v)
  := integral_Omega (nabla dot (nabla dot bold(tau))) v dif x.
$
对本构方程做两次分部积分，并利用固支边界条件 $u = 0$ 与 $partial_n u = 0$，可得混合弱形式
$
  cases(
    a(bold(cal(M)), bold(tau)) + b(bold(tau), u) & = 0 & quad forall bold(tau) in bold(Sigma),
    b(bold(cal(M)), v) & = - (f, v) & quad forall v in U.
  )
$

== 边界双罚项

由于离散挠度空间一般不自动满足固支边界条件，本文引入两个罚项
$
  c_0(u, v) := lambda_0 integral_(partial Omega) u v dif s,
$
以及
$
  c_1(u, v) := lambda_1 integral_(partial Omega) partial_n u partial_n v dif s.
$
于是罚化弱离散为
$
  cases(
    a(bold(Phi)^(bold(cal(M))), bold(Phi)^(bold(tau))) + b(bold(Phi)^(bold(tau)), Phi^(u)) & = 0 & quad forall bold(Phi)^(bold(tau)) in bold(Sigma)_M,
    b(bold(Phi)^(bold(cal(M))), Phi^(v)) + c_0(Phi^(u), Phi^(v)) + c_1(Phi^(u), Phi^(v)) & = - (f, Phi^(v)) & quad forall Phi^(v) in U_M.
  )
$

== 随机特征离散

取 ${bold(E)_alpha}_(alpha=1)^3$ 为二维对称基，按 $(11,22,12)$ 排列。定义
$
  bold(Sigma)_M
  := span { xi^(bold(cal(M)))_m bold(E)_alpha : 0 <= m <= M, 1 <= alpha <= 3 },
$
以及
$
  U_M := span { xi^(u)_m : 0 <= m <= M }.
$
将离散解展开为
$
  bold(Phi)^(bold(cal(M))) = sum_(m=0)^M sum_(alpha=1)^3 phi^(bold(cal(M)))_(m, alpha) xi^(bold(cal(M)))_m bold(E)_alpha,
$
以及
$
  Phi^(u) = sum_(m=0)^M phi^(u)_m xi^(u)_m.
$

== 从变分方程到块线性系统

取测试函数为同一组基函数：
$
  bold(Phi)^(bold(tau)) = xi^(bold(cal(M)))_n bold(E)_beta, quad 0 <= n <= M, 1 <= beta <= 3,
$
以及
$
  Phi^(v) = xi^(u)_n, quad 0 <= n <= M.
$
这里约定第一组下标对应测试函数，也就是矩阵块的行指标；第二组下标对应展开系数，也就是矩阵块的列指标。

定义矩阵块与离散载荷向量
$
  A_((n, beta), (m, alpha))
  := a(xi^(bold(cal(M)))_n bold(E)_beta, xi^(bold(cal(M)))_m bold(E)_alpha),
$
$
  B_((n, beta), m)
  := b(xi^(bold(cal(M)))_n bold(E)_beta, xi^(u)_m),
$
$
  (C_0)_(n, m)
  := c_0(xi^(u)_n, xi^(u)_m),
  quad
  (C_1)_(n, m)
  := c_1(xi^(u)_n, xi^(u)_m),
$
$
  F_n := (f, xi^(u)_n).
$

对任意固定的 $(n, beta)$，取测试函数 $bold(Phi)^(bold(tau)) = xi^(bold(cal(M)))_n bold(E)_beta$。第一条离散变分方程为
$
  0 = a(bold(Phi)^(bold(cal(M))), xi^(bold(cal(M)))_n bold(E)_beta) + b(xi^(bold(cal(M)))_n bold(E)_beta, Phi^(u)).
$
将
$
  bold(Phi)^(bold(cal(M))) = sum_(m=0)^M sum_(alpha=1)^3 phi^(bold(cal(M)))_(m, alpha) xi^(bold(cal(M)))_m bold(E)_alpha
$
与
$
  Phi^(u) = sum_(m=0)^M phi^(u)_m xi^(u)_m
$
代入，并利用双线性性，可得
$
  0
  = sum_(m=0)^M sum_(alpha=1)^3 phi^(bold(cal(M)))_(m, alpha) a(xi^(bold(cal(M)))_n bold(E)_beta, xi^(bold(cal(M)))_m bold(E)_alpha)
  + sum_(m=0)^M phi^(u)_m b(xi^(bold(cal(M)))_n bold(E)_beta, xi^(u)_m).
$
将两项分别识别为矩阵元素，即
$
  sum_(m=0)^M sum_(alpha=1)^3 A_((n, beta), (m, alpha)) phi^(bold(cal(M)))_(m, alpha)
  + sum_(m=0)^M B_((n, beta), m) phi^(u)_m = 0.
$
这正对应于
$
  A bold(phi)^(bold(cal(M))) + B bold(phi)^(u) = 0
$
的第 $(n, beta)$ 个分量。

同理，对任意固定的 $n$，取测试函数 $Phi^(v) = xi^(u)_n$。第二条离散变分方程为
$
  0 = b(bold(Phi)^(bold(cal(M))), xi^(u)_n) + c_0(Phi^(u), xi^(u)_n) + c_1(Phi^(u), xi^(u)_n) + (f, xi^(u)_n).
$
代入展开式可得
$
  0
  = sum_(m=0)^M sum_(alpha=1)^3 phi^(bold(cal(M)))_(m, alpha) b(xi^(bold(cal(M)))_m bold(E)_alpha, xi^(u)_n)
  + sum_(m=0)^M phi^(u)_m c_0(xi^(u)_m, xi^(u)_n)
  + sum_(m=0)^M phi^(u)_m c_1(xi^(u)_m, xi^(u)_n)
  + F_n.
$
注意到
$
  b(xi^(bold(cal(M)))_m bold(E)_alpha, xi^(u)_n) = B_((m, alpha), n),
$
并且由 $c_0, c_1$ 的对称性可知
$
  c_l(xi^(u)_m, xi^(u)_n) = (C_l)_(n, m), quad l = 0, 1.
$
因此可写为
$
  sum_(m=0)^M sum_(alpha=1)^3 B_((m, alpha), n) phi^(bold(cal(M)))_(m, alpha)
  + sum_(m=0)^M ((C_0)_(n, m) + (C_1)_(n, m)) phi^(u)_m
  = - F_n.
$
这正对应于
$
  B^T bold(phi)^(bold(cal(M))) + (C_0 + C_1) bold(phi)^(u) = -F
$
的第 $n$ 个分量。

若将 $A$、$B$、$C_0$、$C_1$ 视为 $(M+1) times (M+1)$ 块矩阵，$F$ 视为 $(M+1) times 1$ 块向量，则
$
  A =
  mat(
    A_(0, 0), dots.h, A_(0, M);
    dots.v, dots.down, dots.v;
    A_(M, 0), dots.h, A_(M, M)
  ), quad
  B =
  mat(
    B_(0, 0), dots.h, B_(0, M);
    dots.v, dots.down, dots.v;
    B_(M, 0), dots.h, B_(M, M)
  ),
$
$
  C_l =
  mat(
    (C_l)_(0, 0), dots.h, (C_l)_(0, M);
    dots.v, dots.down, dots.v;
    (C_l)_(M, 0), dots.h, (C_l)_(M, M)
  ), quad
  F =
  mat(
    F_0;
    dots.v;
    F_M
  ), quad l = 0, 1.
$
其中块内指标满足
$
  A_(n, m) in RR^(3 times 3), quad (A_(n, m))_(beta, alpha) = A_((n, beta), (m, alpha)),
$
$
  B_(n, m) in RR^(3 times 1), quad (B_(n, m))_beta = B_((n, beta), m),
$
$
  (C_l)_(n, m) in RR, quad l = 0, 1,
$
$
  F_n in RR.
$
若再记
$
  bold(phi)^(bold(cal(M)))_k := (phi^(bold(cal(M)))_(k, 1), dots.c, phi^(bold(cal(M)))_(k, 3))^T in RR^3,
$
则 $bold(phi)^(bold(cal(M))) = (bold(phi)^(bold(cal(M)))_0, dots.c, bold(phi)^(bold(cal(M)))_M)^T$，$bold(phi)^(u) = (phi^(u)_0, dots.c, phi^(u)_M)^T$，并且两条方程可按块写成
$
  sum_(k=0)^M A_(n, k) bold(phi)^(bold(cal(M)))_k + sum_(k=0)^M B_(n, k) phi^(u)_k = 0,
$
以及
$
  sum_(k=0)^M B_(k, n)^T bold(phi)^(bold(cal(M)))_k + sum_(k=0)^M ((C_0)_(n, k) + (C_1)_(n, k)) phi^(u)_k = -F_n, quad 0 <= n <= M.
$
于是离散块系统写为
$
  mat(A, B; B^T, C_0 + C_1) mat(bold(phi)^(bold(cal(M))); bold(phi)^(u)) = mat(0; -F).
$
其中
$
  A in RR^(3(M+1) times 3(M+1)), quad
  B in RR^(3(M+1) times (M+1)), quad
  C_0, C_1 in RR^((M+1) times (M+1)), quad
  F in RR^(M+1).
$

== $A$ 块元素

由定义
$
  A_((n, beta), (m, alpha))
  := a(xi^(bold(cal(M)))_n bold(E)_beta, xi^(bold(cal(M)))_m bold(E)_alpha)
  = integral_Omega (bold(cal(A)) : (xi^(bold(cal(M)))_n bold(E)_beta)) : (xi^(bold(cal(M)))_m bold(E)_alpha) dif x.
$
由于 $xi^(bold(cal(M)))_n$ 与 $xi^(bold(cal(M)))_m$ 为标量函数，$bold(E)_beta$ 与 $bold(E)_alpha$ 为常矩阵，可得
$
  A_((n, beta), (m, alpha))
  = integral_Omega xi^(bold(cal(M)))_n (x) xi^(bold(cal(M)))_m (x)
  ((bold(cal(A))(x) : bold(E)_beta) : bold(E)_alpha) dif x.
$
若将双重收缩写成坐标形式，则
$
  ((bold(cal(A)) : bold(E)_beta) : bold(E)_alpha)
  = sum_(i=1)^2 sum_(j=1)^2 sum_(k=1)^2 sum_(l=1)^2
  bold(cal(A))_(i j k l) (bold(E)_beta)_(k l) (bold(E)_alpha)_(i j).
$
因此
$
  A_((n, beta), (m, alpha))
  = integral_Omega xi^(bold(cal(M)))_n xi^(bold(cal(M)))_m
  sum_(i=1)^2 sum_(j=1)^2 sum_(k=1)^2 sum_(l=1)^2
  bold(cal(A))_(i j k l) (bold(E)_beta)_(k l) (bold(E)_alpha)_(i j)
  dif x.
$

== $B$ 块元素

由定义
$
  B_((n, beta), m)
  := b(xi^(bold(cal(M)))_n bold(E)_beta, xi^(u)_m)
  = integral_Omega (nabla dot (nabla dot (xi^(bold(cal(M)))_n bold(E)_beta))) xi^(u)_m dif x.
$
利用双散度的分量定义，
$
  nabla dot (nabla dot (xi^(bold(cal(M)))_n bold(E)_beta))
  = (xi^(bold(cal(M)))_n (bold(E)_beta)_(i j))_(, i j)
  = (bold(E)_beta)_(i j) partial_(i j) xi^(bold(cal(M)))_n.
$
也就是说
$
  nabla dot (nabla dot (xi^(bold(cal(M)))_n bold(E)_beta))
  = bold(E)_beta : nabla^2 xi^(bold(cal(M)))_n,
$
其中 $nabla^2 xi^(bold(cal(M)))_n$ 表示 Hessian 矩阵。因此
$
  B_((n, beta), m)
  = integral_Omega xi^(u)_m (bold(E)_beta : nabla^2 xi^(bold(cal(M)))_n) dif x.
$

对 3 个对称基，$bold(E)_beta : nabla^2 xi^(bold(cal(M)))_n$ 可完全写成二阶导数的线性组合。记
$
  nabla^2 xi^(bold(cal(M)))_n
  = mat(
    partial_11 xi^(bold(cal(M)))_n, partial_12 xi^(bold(cal(M)))_n;
    partial_12 xi^(bold(cal(M)))_n, partial_22 xi^(bold(cal(M)))_n
  ),
$
则
$
  cases(
    bold(E)_1 : nabla^2 xi^(bold(cal(M)))_n = partial_11 xi^(bold(cal(M)))_n,
    bold(E)_2 : nabla^2 xi^(bold(cal(M)))_n = partial_22 xi^(bold(cal(M)))_n,
    bold(E)_3 : nabla^2 xi^(bold(cal(M)))_n = 2 partial_12 xi^(bold(cal(M)))_n.
  )
$

对于 $n >= 1$ 的弯矩特征函数
$
  xi^(bold(cal(M)))_n (x) = rho(bold(w)_n^T x + b_n),
$
其 Hessian 可直接写为
$
  nabla^2 xi^(bold(cal(M)))_n (x)
  = rho''(bold(w)_n^T x + b_n) bold(w)_n bold(w)_n^T.
$
若采用重参数化 $bold(w)_n = gamma bold(a)_n$ 与 $b_n = gamma r_n$，则等价地
$
  nabla^2 xi^(bold(cal(M)))_n (x)
  = rho''(gamma (bold(a)_n^T x + r_n)) gamma^2 bold(a)_n bold(a)_n^T.
$
特别地，$xi^(bold(cal(M)))_0 = 1$，故 $nabla^2 xi^(bold(cal(M)))_0 = 0$，从而所有以 $n=0$ 为测试函数的 $B$ 行元素都为 $0$。

== $C_0$ 与 $C_1$ 块元素

由定义
$
  (C_0)_(n, m)
  := c_0(xi^(u)_n, xi^(u)_m)
  = lambda_0 integral_(partial Omega) xi^(u)_n (x) xi^(u)_m (x) dif s.
$

同理，
$
  (C_1)_(n, m)
  := c_1(xi^(u)_n, xi^(u)_m)
  = lambda_1 integral_(partial Omega) partial_n xi^(u)_n (x) partial_n xi^(u)_m (x) dif s,
$
其中
$
  partial_n xi^(u)_n = nabla xi^(u)_n dot bold(n).
$
对于 $n >= 1$ 的位移特征函数，有
$
  nabla xi^(u)_n (x) = rho'(bold(w)_n^T x + b_n) bold(w)_n.
$
若采用重参数化，则等价地
$
  nabla xi^(u)_n (x) = rho'(gamma (bold(a)_n^T x + r_n)) gamma bold(a)_n,
$
从而
$
  partial_n xi^(u)_n (x)
  = rho'(gamma (bold(a)_n^T x + r_n)) gamma bold(a)_n dot bold(n).
$
特别地，$xi^(u)_0 = 1$，故 $partial_n xi^(u)_0 = 0$，因此 $(C_1)_(0, m) = (C_1)_(m, 0) = 0$。

== $F$ 载荷向量元素

由定义
$
  F_n
  := (f, xi^(u)_n)
  = integral_Omega f(x) xi^(u)_n (x) dif x.
$
