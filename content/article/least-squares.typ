#import "/typ/templates/blog.typ": *
#import fletcher: diagram, edge, node
#import "@preview/lovelace:0.3.0": pseudocode-list

#show: main-zh.with(
  title: "线性化网络混合最小二乘方法及其误差分析",
  author: "summer",
  desc: [证明线弹性与 Kirchhoff--Love 板弯曲的混合最小二乘泛函与相应图范数平方双边等价，等价常数可关于 Lamé 常数一致选取。本文还以准均匀单隐层 ReLU 幂字典的 Sobolev 逼近结果为理论输入，给出两类方程的线性化网络误差分析与完整算法的数值收敛结果],
  date: "2026-04-20",
  tags: (
    blog-tags.numerical-methods,
    blog-tags.pde,
  ),
  show-outline: true,
)

= 引言

线性化网络方法预先取定并冻结单隐层参数，随后只求输出层系数。冻结参数生成的特征族在逼近论中称为字典（dictionary）。给定字典后，偏微分方程离散仍是有限维凸二次问题。本文沿用 @SiegelHongJinHaoXu2023 的方向--偏置参数化，并以 @LiuMaoXu2025 的准均匀字典逼近结果作为确定性逼近输入。本文将这一结构用于线弹性应力--位移系统与 Kirchhoff--Love 板弯曲弯矩--挠度系统，连续问题采用经典的混合最小二乘泛函。
本文的核心结果是两条残差稳定性定理。其一，线弹性最小二乘泛函与 $bold(H)(div) times H^1$ 图范数平方双边等价（@thm:elasticity-stability）。其二，板弯曲最小二乘泛函与 $bold(H)(div div) times H^2$ 图范数平方双边等价（@thm:plate-stability）。两条定理的等价常数都可关于 Lamé 常数 $lambda$ 一致选取，因而覆盖近不可压缩极限 $lambda -> oo$。两类问题的一致性机制并不相同：线弹性依赖偏差--体积分解与零平均迹下的迹估计，板弯曲依赖柔度算子的一致椭圆性与 $H_0^2$ 上的 Poincaré 不等式。经典应力--位移最小二乘稳定性可参见 @CaiStarke2003、@CaiStarke2004 与 @DuanLin2005。

本文的另一项工作是将单隐层线性化 $rho_k$ 网络用于两类方程的数值求解，并给出误差分析：总误差分为字典逼近误差、本质边界投影的空间截断与积分误差、训练泛函的 Monte Carlo 一致偏差及线性代数求解误差四类。Ritz 离散用于本质边界投影的空间截断，训练泛函采用独立的 Monte Carlo 样本近似，其一致偏差由 Rademacher 复杂度控制。激活幂次 $k$ 通过字典的饱和指数 $s_"cap" (d) = (d + 2k + 1)\/2$ 进入逼近误差界，因而改变关于字典规模 $N$ 的理论收敛指数。数值实验统一报告完全离散最小二乘算法在独立测试规则上的图范数误差，并考察字典规模、激活幂次与近不可压缩极限的影响。

= 连续混合最小二乘问题

== 记号

设 $Omega subset RR^d$ 为有界 Lipschitz 区域，$d in {2,3}$。以 $op("Sym")(d)$ 表示 $d times d$ 实对称矩阵空间，并定义
$
  bold(H)(div, Omega; op("Sym")(d))
  := {bold(tau) in L^2(Omega; op("Sym")(d)) :
    div bold(tau) in L^2(Omega; RR^d)}.
$
对二维板问题定义
$
  bold(H)(div div, Omega; op("Sym")(2))
  := {bold(tau) in L^2(Omega; op("Sym")(2)) :
    div div bold(tau) in L^2(Omega)}.
$
相应图范数分别记为
$
      norm(bold(tau))_(bold(H)(div))^2 & := norm(bold(tau))_(L^2)^2 + norm(div bold(tau))_(L^2)^2, \
  norm(bold(tau))_(bold(H)(div div))^2 & := norm(bold(tau))_(L^2)^2 + norm(div div bold(tau))_(L^2)^2.
$
本文以下标 $"LE"$（linear elasticity，线弹性）与 $"KL"$（Kirchhoff--Love 板弯曲）区分两类问题的算子、泛函与函数空间。以 $(bold(sigma), bold(u))$ 表示线弹性未知量，以 $(bold(M), w)$ 表示板弯曲未知量。$bold(T)_alpha$ 表示对称张量基。对非负量 $A$ 与 $B$，记 $A lt.tilde B$ 表示存在常数 $C > 0$ 使 $A <= C B$，其中 $C$ 不依赖于 Lamé 常数 $lambda$ 与任何离散参数，具体依赖关系在上下文中给出。$A gt.tilde B$ 即 $B lt.tilde A$。记 $A tilde.eq B$ 表示 $A lt.tilde B$ 与 $B lt.tilde A$ 同时成立，即 $A$ 与 $B$ 同阶，其中两侧常数满足同一约定。以 $bb(P)$ 与 $EE$ 分别表示概率与期望，$EE$ 的下标区分取期望所对的样本组。

== 线弹性

线弹性的应力--位移系统为
$
  cases(
    bold(cal(A))_"LE" bold(sigma) - bold(epsilon)(bold(u)) = bold(0) & "in" Omega,
    div bold(sigma) + bold(f) = bold(0) & "in" Omega,
    bold(u) = bold(0) & "on" partial Omega.
  )
$
其中未知量为应力 $bold(sigma)$ 与位移 $bold(u)$，$bold(f) in L^2(Omega; RR^d)$ 为给定体力。第一个方程为本构关系，第二个方程为平衡方程，第三个方程为纯位移边界条件。$bold(epsilon)$ 为对称梯度算子，
$
  bold(epsilon)(bold(v)) := 1/2 (nabla bold(v) + nabla bold(v)^T).
$
$bold(cal(A))_"LE":op("Sym")(d)->op("Sym")(d)$ 为柔度算子，即各向同性刚度算子 $bold(tau) |-> 2 mu bold(tau) + lambda tr(bold(tau)) bold(I)$ 之逆，其中 $mu > 0$ 与 $lambda >= 0$ 为 Lamé 常数，$tr(bold(tau))$ 为迹，$bold(I)$ 为单位张量。显式地，
$
  bold(cal(A))_"LE" bold(tau)
  := 1/(2 mu) (
    bold(tau) - lambda/(2 mu+d lambda) tr(bold(tau)) bold(I)
  ).
$
定义
$
  bold(Sigma)_"LE" & := {bold(tau) in bold(H)(div, Omega; op("Sym")(d)):
                       integral_Omega tr(bold(tau)) dif x=0}, \
      bold(U)_"LE" & := H_0^1(Omega; RR^d), \
      bold(X)_"LE" & := bold(Sigma)_"LE" times bold(U)_"LE".
$
沿用 @CaiStarke2003、@CaiStarke2004 与 @DuanLin2005 提出的应力--位移最小二乘提法，定义泛函
$
  cal(J)_"LE" (bold(tau),bold(v);bold(f))
  := norm(bold(cal(A))_"LE" bold(tau)-bold(epsilon)(bold(v)))_(L^2)^2
  + norm(div bold(tau)+bold(f))_(L^2)^2.
$

对应双线性形式记为
$
  a_"LE" ((bold(sigma), bold(u)), (bold(tau), bold(v)))
  := (bold(cal(A))_"LE" bold(sigma) - bold(epsilon)(bold(u)),
    bold(cal(A))_"LE" bold(tau) - bold(epsilon)(bold(v)))_(L^2(Omega)) + (div bold(sigma), div bold(tau))_(L^2(Omega)),
$
右端线性泛函为
$
  ell_"LE" (bold(tau), bold(v))
  := - (bold(f), div bold(tau))_(L^2(Omega)).
$

下述定理表明，强形式、极小化 $cal(J)_"LE"$ 与变分问题三种提法彼此等价，因此可将 $cal(J)_"LE"$ 作为离散化的目标泛函。

#theorem(title: [线弹性三种提法的等价性])[
  若线弹性强形式在 $bold(Sigma)_"LE" times bold(U)_"LE"$ 中存在解，则下列三个问题彼此等价：

  1. 线弹性强形式。
  2. 最小化 $cal(J)_"LE"$。
  3. 求 $(bold(sigma), bold(u)) in bold(Sigma)_"LE" times bold(U)_"LE"$，使得
    $
      a_"LE" ((bold(sigma), bold(u)), (bold(tau), bold(v)))
      = ell_"LE" (bold(tau), bold(v)),
      quad forall (bold(tau), bold(v)) in bold(Sigma)_"LE" times bold(U)_"LE".
    $
]<thm:elasticity-equivalence>

证明见附录 @app:elasticity。

@thm:elasticity-equivalence 只表明零残差刻画精确解，并未量化残差对误差的控制。下述稳定性定理补足这一环节，也是后文误差分析的出发点。

#theorem(title: [线弹性残差稳定性])[
  对任意 $(bold(tau), bold(v)) in bold(Sigma)_"LE" times bold(U)_"LE"$ 都有
  $
    norm(bold(tau))_(bold(H)(div))^2
    + norm(bold(v))_(H^1(Omega))^2
    lt.tilde cal(J)_"LE" (bold(tau), bold(v); bold(0))
    lt.tilde norm(bold(tau))_(bold(H)(div))^2
    + norm(bold(v))_(H^1(Omega))^2,
  $
  其中两侧隐含常数仅依赖于 $Omega$、$mu$ 与 $d$。
]<thm:elasticity-stability>

#proof[
  先证上界。令 $bold(f) = bold(0)$。由柔度算子的显式公式
  $
    bold(cal(A))_"LE" bold(tau)
    = 1/(2 mu) (
      bold(tau) - lambda/(2 mu + d lambda) tr(bold(tau)) bold(I)
    ),
  $
  以及估计 $norm(tr(bold(tau)) bold(I))_(L^2(Omega)) <= d norm(bold(tau))_(L^2(Omega))$，可得
  $
    norm(bold(cal(A))_"LE" bold(tau))_(L^2(Omega)) & <= 1/(2 mu) norm(bold(tau))_(L^2(Omega))
                                                     + lambda/(2 mu (2 mu + d lambda))
                                                     norm(tr(bold(tau)) bold(I))_(L^2(Omega)) \
                                                   & <= (
                                                       1/(2 mu) + d lambda/(2 mu (2 mu + d lambda))
                                                     ) norm(bold(tau))_(L^2(Omega)) \
                                                   & <= 1/mu norm(bold(tau))_(L^2(Omega)).
  $
  再由 $(a + b)^2 <= 2 a^2 + 2 b^2$ 与 $norm(bold(epsilon)(bold(v)))_(L^2(Omega)) <= norm(nabla bold(v))_(L^2(Omega))$，可得
  $
    cal(J)_"LE" (bold(tau), bold(v); bold(0)) & <= 2 norm(bold(cal(A))_"LE" bold(tau))_(L^2(Omega))^2
                                                + 2 norm(bold(epsilon)(bold(v)))_(L^2(Omega))^2
                                                + norm(div bold(tau))_(L^2(Omega))^2 \
                                              & lt.tilde norm(bold(tau))_(bold(H)(div))^2
                                                + norm(bold(v))_(H^1(Omega))^2.
  $
  其中隐含常数仅依赖于 $mu$。

  再证下界，即关于 $lambda$ 一致有界的强制性。以下仅在下界证明中引入偏差投影
  $
    bold(tau)^"D" := bold(tau) - 1/d tr(bold(tau)) bold(I),
    quad
    bold(epsilon)^"D" (bold(v)) := bold(epsilon)(bold(v)) - 1/d (div bold(v)) bold(I),
  $
  以及各向同性柔度算子的偏差--体积分解
  $
    bold(cal(A))_"LE" bold(tau) = 1/(2 mu) bold(tau)^"D" + 1/(d(2 mu + d lambda)) tr(bold(tau)) bold(I).
  $
  由偏差部分与球张量部分在 $L^2(Omega; op("Sym")(d))$ 中的正交性，
  $
    norm(bold(cal(A))_"LE" bold(tau) - bold(epsilon)(bold(v)))_(L^2(Omega))^2
    = norm(1/(2 mu) bold(tau)^"D" - bold(epsilon)^"D" (bold(v)))_(L^2(Omega))^2
    + 1/d norm(div bold(v) - 1/(2 mu + d lambda) tr(bold(tau)))_(L^2(Omega))^2.
  $
  因此记三个残差
  $
    bold(r)_"dev" := 1/(2 mu) bold(tau)^"D" - bold(epsilon)^"D" (bold(v)),
    quad
    r_"vol" := 1/sqrt(d) (div bold(v) - 1/(2 mu + d lambda) tr(bold(tau))),
    quad
    bold(r)_"equ" := div bold(tau).
  $
  便有
  $
    cal(J)_"LE" (bold(tau), bold(v); bold(0))
    = norm(bold(r)_"dev")_(L^2)^2
    + norm(r_"vol")_(L^2)^2
    + norm(bold(r)_"equ")_(L^2)^2.
  $

  首先处理位移部分。由偏差投影的定义，
  $
    bold(epsilon)(bold(v)) = bold(epsilon)^"D" (bold(v)) + 1/d (div bold(v)) bold(I).
  $
  由于 $bold(epsilon)^"D" (bold(v)) = 1/(2 mu) bold(tau)^"D" - bold(r)_"dev"$，以及 $div bold(v) = 1/(2 mu + d lambda) tr(bold(tau)) + sqrt(d) r_"vol"$，有
  $
    bold(epsilon)(bold(v)) = 1/(2 mu) bold(tau)^"D" - bold(r)_"dev"
    + 1/d (1/(2 mu + d lambda) tr(bold(tau)) + sqrt(d) r_"vol") bold(I).
  $
  由 Korn 第一不等式（二维情形参见 @BrennerScott2008 推论 11.2.25，三维推广见其注 11.2.27），
  $
    norm(bold(v))_(H^1(Omega)) & lt.tilde norm(bold(epsilon)(bold(v)))_(L^2(Omega)) \
                               & lt.tilde 1/(2 mu) norm(bold(tau)^"D")_(L^2)
                                 + norm(bold(r)_"dev")_(L^2)
                                 + 1/(sqrt(d) (2 mu + d lambda)) norm(tr(bold(tau)))_(L^2)
                                 + norm(r_"vol")_(L^2).
  $
  <eq:korn>

  接下来控制应力。由分解 $bold(tau) = bold(tau)^"D" + 1/d tr(bold(tau)) bold(I)$ 以及 $bold(tau)^"D" : bold(I) = tr(bold(tau)^"D") = 0$，
  $
    integral_Omega (1/(2 mu) bold(tau)^"D") : bold(tau) dif x
    = 1/(2 mu) integral_Omega bold(tau)^"D" : (bold(tau)^"D" + 1/d tr(bold(tau)) bold(I)) dif x
    = 1/(2 mu) norm(bold(tau)^"D")_(L^2(Omega))^2.
  $
  由于 $bold(tau)$ 对称且 $bold(v) in H_0^1(Omega; RR^d)$，分部积分给出
  $
    integral_Omega bold(epsilon)^"D" (bold(v)) : bold(tau) dif x & = integral_Omega bold(epsilon)(bold(v)) : bold(tau) dif x
                                                                   - 1/d integral_Omega (div bold(v)) tr(bold(tau)) dif x \
                                                                 & = - integral_Omega bold(v) dot (div bold(tau)) dif x
                                                                   - 1/d integral_Omega (div bold(v)) tr(bold(tau)) dif x.
  $
  于是
  $
    integral_Omega (1/(2 mu) bold(tau)^"D") : bold(tau) dif x
    &= integral_Omega (bold(r)_"dev" + bold(epsilon)^"D" (bold(v))) : bold(tau) dif x
    \
    &= integral_Omega bold(r)_"dev" : bold(tau) dif x
    - integral_Omega bold(v) dot (div bold(tau)) dif x
    - 1/d integral_Omega (div bold(v)) tr(bold(tau)) dif x.
  $
  因此
  $
    1/(2 mu) norm(bold(tau)^"D")_(L^2)^2
    <= norm(bold(r)_"dev")_(L^2) norm(bold(tau))_(L^2)
    + norm(bold(v))_(L^2) norm(bold(r)_"equ")_(L^2)
    + 1/d norm(1/(2 mu + d lambda) tr(bold(tau)) + sqrt(d) r_"vol")_(L^2) norm(tr(bold(tau)))_(L^2).
  $
  由迹与偏差的关系：$norm(bold(tau))_(L^2)^2 = norm(bold(tau)^"D")_(L^2)^2 + 1/d norm(tr(bold(tau)))_(L^2)^2$。利用 @eq:korn 及 Young 不等式逐项调节，可将含 $norm(bold(tau))^2$ 的项吸收至左端，得到
  $
    norm(bold(tau)^"D")_(L^2(Omega))^2
    lt.tilde norm(bold(r)_"dev")_(L^2(Omega))^2
    + norm(r_"vol")_(L^2(Omega))^2
    + norm(bold(r)_"equ")_(L^2(Omega))^2.
  $
  <eq:tau-dev-bound>

  对于应力的迹部分，由估计（参见 @CarstensenDolzmann1998，引理 4.1），
  $
    norm(tr(bold(tau)))_(L^2(Omega))
    lt.tilde norm(bold(tau)^"D")_(L^2(Omega)) + norm(div bold(tau))_(L^2(Omega)),
    quad forall bold(tau) in bold(Sigma)_"LE".
  $
  该式中零平均迹条件是必要的，否则常数球应力 $bold(tau) = c bold(I)$ 会使右端为零而左端非零。
  结合 @eq:tau-dev-bound 即得
  $
    norm(tr(bold(tau)))_(L^2(Omega))^2
    lt.tilde norm(bold(r)_"dev")_(L^2(Omega))^2
    + norm(bold(r)_"equ")_(L^2(Omega))^2
    + norm(r_"vol")_(L^2(Omega))^2.
  $
  因此 $norm(bold(tau))_(L^2)^2$ 由残差平方和控制。将应力范数估计代回 @eq:korn，得到 $norm(bold(v))_(H^1)^2$ 的类似估计。最后结合
  $
    norm(bold(tau))_(bold(H)(div))^2
    = norm(bold(tau))_(L^2(Omega))^2
    + norm(bold(r)_"equ")_(L^2(Omega))^2
  $
  即得下界，其隐含常数仅依赖于 $Omega$、$mu$ 与 $d$。证毕。
]

由 @thm:elasticity-stability，双线性形式 $a_"LE"$ 在 $bold(Sigma)_"LE" times bold(U)_"LE"$ 上连续且强制，其中连续性与强制性常数均可关于 Lamé 常数 $lambda$ 一致选取。特别地，近不可压缩极限 $lambda -> oo$ 下稳定常数不退化。只要离散空间包含于 $bold(Sigma)_"LE" times bold(U)_"LE"$ 且其逼近误差本身保持稳定，离散解不会因该极限而出现锁定。平均迹规范排除了常值球应力核。

== Kirchhoff--Love 板弯曲

以下令 $d=2$。Kirchhoff--Love 板的弯矩--挠度系统为
$
  cases(
    bold(cal(A))_"KL" bold(M)-bold(kappa)(w)=bold(0) & "in" Omega,
    div div bold(M)+f=0 & "in" Omega,
    w=0 \, quad partial_n w=0 & "on" partial Omega.
  )
$
其中未知量为弯矩 $bold(M)$ 与挠度 $w$，$f in L^2(Omega)$ 为给定横向载荷。第一个方程为本构关系，第二个方程为平衡方程，第三个方程为固支边界条件。曲率张量固定记为
$
  bold(kappa)(w):=-nabla^2 w.
$
$bold(cal(A))_"KL":op("Sym")(2)->op("Sym")(2)$ 为各向同性板柔度算子，
$
  bold(cal(A))_"KL" bold(tau)
  := 1/(D(1-nu)) bold(tau)
  - nu/(D(1-nu)(1+nu)) tr(bold(tau)) bold(I),
$
其中 $E$、$nu$ 与 $h$ 分别为杨氏模量、泊松比与板厚，
$
  D = (E h^3)/(12(1 - nu^2)),
  quad
  lambda = (E nu)/((1 + nu) (1 - 2 nu)),
  quad
  mu = E/(2(1 + nu))
$
为弯曲刚度与对应三维各向同性材料的 Lamé 参数。下文固定 $mu > 0$ 与 $h > 0$，并只讨论物理参数区间 $lambda >= 0$，等价地，$0 <= nu < 1/2$。
定义
$
  bold(Sigma)_"KL" & := bold(H)(div div,Omega;op("Sym")(2)), \
            U_"KL" & := H_0^2(Omega), \
      bold(X)_"KL" & := bold(Sigma)_"KL" times U_"KL".
$
有关 $bold(H)(div div)$ 弯矩--挠度模型可参见 @FuhrerHeuerNiemi2019 与 @FuhrerHeuer2025。板弯曲的残差型最小二乘离散还可参见 @PontazaReddy2004。本文采用的最小二乘泛函为
$
  cal(J)_"KL" (bold(tau),v;f)
  := norm(bold(cal(A))_"KL" bold(tau)-bold(kappa)(v))_(L^2)^2
  + norm(div div bold(tau)+f)_(L^2)^2.
$

双线性形式为
$
  a_"KL" ((bold(M), w), (bold(tau), v))
  := (bold(cal(A))_"KL" bold(M) - bold(kappa)(w),
    bold(cal(A))_"KL" bold(tau) - bold(kappa)(v))_(L^2(Omega)) + (div div bold(M), div div bold(tau))_(L^2(Omega)),
$
右端泛函为
$
  ell_"KL" (bold(tau), v)
  := - (f, div div bold(tau))_(L^2(Omega)).
$

与线弹性情形平行，三种提法的等价性由下述定理给出。

#theorem(title: [板弯曲三种提法的等价性])[
  若 Kirchhoff--Love 板弯曲强形式在 $bold(Sigma)_"KL" times U_"KL"$ 中存在解，则下列三个问题彼此等价：

  1. Kirchhoff--Love 板弯曲强形式。
  2. 最小化 $cal(J)_"KL"$。
  3. 求 $(bold(M), w) in bold(Sigma)_"KL" times U_"KL"$，使得
    $
      a_"KL" ((bold(M), w), (bold(tau), v))
      = ell_"KL" (bold(tau), v),
      quad forall (bold(tau), v) in bold(Sigma)_"KL" times U_"KL".
    $
]<thm:plate-equivalence>

证明见附录 @app:plate。

与 @thm:elasticity-stability 平行，板弯曲的最小二乘泛函也双边控制相应图范数。一致性机制的差别见证明后的讨论。

#theorem(title: [板弯曲残差稳定性])[
  在上述各向同性薄板参数假设下，对任意 $(bold(tau), v) in bold(Sigma)_"KL" times U_"KL"$ 都有
  $
    norm(bold(tau))_(bold(H)(div div))^2
    + norm(v)_(H^2(Omega))^2
    lt.tilde cal(J)_"KL" (bold(tau), v; 0)
    lt.tilde norm(bold(tau))_(bold(H)(div div))^2
    + norm(v)_(H^2(Omega))^2,
  $
  其中两侧隐含常数仅依赖于 $Omega$、$mu$ 与 $h$。
]<thm:plate-stability>

#proof[
  先将柔度算子写成偏差--迹分解。对任意 $bold(tau) in op("Sym")(2)$，记
  $
    bold(tau)^"D" := bold(tau) - 1/2 tr(bold(tau)) bold(I),
    quad
    bold(tau) = bold(tau)^"D" + 1/2 tr(bold(tau)) bold(I).
  $
  将该分解代入各向同性薄板柔度公式，得到
  $
    bold(cal(A))_"KL" bold(tau)
    = 1/(D(1-nu)) bold(tau)^"D"
    + 1/(2 D (1+nu)) tr(bold(tau)) bold(I).
  $
  由 $mu = E/(2(1+nu))$ 与 $nu = lambda/(2(lambda + mu))$，可化为
  $
    1/(D(1-nu)) = 6/(mu h^3),
    quad
    1/(D(1+nu)) = 6 (lambda + 2 mu)/(mu (3 lambda + 2 mu) h^3).
  $
  在 $lambda >= 0$ 时，进一步有
  $
    2/(mu h^3)
    <= 1/(D(1+nu))
    <= 6/(mu h^3).
  $
  再利用偏差部分与球张量部分在 $L^2(Omega; op("Sym")(2))$ 中的正交性以及
  $
    norm(bold(tau))_(L^2(Omega))^2
    = norm(bold(tau)^"D")_(L^2(Omega))^2
    + 1/2 norm(tr(bold(tau)))_(L^2(Omega))^2,
  $
  得到
  $
    norm(bold(cal(A))_"KL" bold(tau))_(L^2(Omega))^2
    = (1/(D(1-nu)))^2 norm(bold(tau)^"D")_(L^2(Omega))^2
    + 1/(2 D^2 (1+nu)^2) norm(tr(bold(tau)))_(L^2(Omega))^2
    <= 36/(mu^2 h^6) norm(bold(tau))_(L^2(Omega))^2,
  $
  以及
  $
    2/(mu h^3) norm(bold(tau))_(L^2(Omega))^2
    <= integral_Omega (bold(cal(A))_"KL" bold(tau)) : bold(tau) dif x
    <= 6/(mu h^3) norm(bold(tau))_(L^2(Omega))^2.
  $
  为简记，以下记
  $
    c_A := 2/(mu h^3),
    quad
    C_A := 6/(mu h^3).
  $

  上界由上式直接得到：
  $
    norm(bold(cal(A))_"KL" bold(tau) - bold(kappa)(v))_(L^2(Omega))^2
    <= 2 C_A^2 norm(bold(tau))_(L^2(Omega))^2
    + 2 norm(bold(kappa)(v))_(L^2(Omega))^2.
  $
  又因 $bold(kappa)(v) = - nabla^2 v$，故
  $
    norm(bold(kappa)(v))_(L^2(Omega))
    = norm(nabla^2 v)_(L^2(Omega))
    <= norm(v)_(H^2(Omega)).
  $
  再加上
  $
    norm(div div bold(tau))_(L^2(Omega))^2
    <= norm(bold(tau))_(bold(H)(div div))^2
  $
  即得上界。

  再证下界。记
  $
    bold(r)_"c" := bold(cal(A))_"KL" bold(tau) - bold(kappa)(v),
    quad
    r_"e" := div div bold(tau).
  $
  则
  $
    bold(kappa)(v) = bold(cal(A))_"KL" bold(tau) - bold(r)_"c".
  $
  由于 $v in H_0^2(Omega)$，有 $v in H_0^1(Omega)$ 且 $partial_i v in H_0^1(Omega)$，$i = 1, 2$。由 Poincaré 不等式（参见 @BrennerScott2008 命题 5.3.5），存在常数 $C_P > 0$，使得
  $
    norm(z)_(H^1(Omega))
    <= C_P abs(z)_(H^1(Omega)),
    quad forall z in H_0^1(Omega).
  $
  分别取 $z = v$ 与 $z = partial_i v$，得到
  $
    norm(v)_(H^1(Omega))
    <= C_P norm(nabla v)_(L^2(Omega)),
    quad
    norm(partial_i v)_(H^1(Omega))
    <= C_P norm(nabla partial_i v)_(L^2(Omega)).
  $
  因此
  $
    norm(v)_(H^2(Omega))^2 & = norm(v)_(L^2(Omega))^2
                             + norm(nabla v)_(L^2(Omega))^2 + norm(nabla^2 v)_(L^2(Omega))^2 \
                           & <= (C_P^4 + C_P^2 + 1) norm(nabla^2 v)_(L^2(Omega))^2,
  $
  令 $tilde(C)_P = C_P^4 + C_P^2 + 1$，则 $tilde(C)_P$ 仅依赖于 $C_P$ 与 $Omega$。于是
  $
    norm(v)_(H^2(Omega))
    <= tilde(C)_P norm(nabla^2 v)_(L^2(Omega))
    = tilde(C)_P norm(bold(kappa)(v))_(L^2(Omega))
    <= tilde(C)_P (
      norm(bold(r)_"c")_(L^2(Omega))
      + C_A norm(bold(tau))_(L^2(Omega))
    ).
  $
  <eq:plate-h2>

  另一方面，由椭圆性，
  $
    c_A norm(bold(tau))_(L^2(Omega))^2
    <= integral_Omega (bold(cal(A))_"KL" bold(tau)) : bold(tau) dif x.
  $
  又由于 $v in H_0^2(Omega)$，对 $bold(kappa)(v) = - nabla^2 v$ 作两次分部积分可得
  $
    integral_Omega bold(kappa)(v) : bold(tau) dif x
    = - integral_Omega v div div bold(tau) dif x
    = - integral_Omega v r_"e" dif x.
  $
  因而
  $
    c_A norm(bold(tau))_(L^2(Omega))^2
    <= norm(v)_(H^2(Omega)) norm(r_"e")_(L^2(Omega))
    + norm(bold(r)_"c")_(L^2(Omega)) norm(bold(tau))_(L^2(Omega)).
  $
  将 @eq:plate-h2 代入上式，并对右端应用 Young 不等式吸收含 $norm(bold(tau))_(L^2)^2$ 的项，可得
  $
    norm(bold(tau))_(L^2(Omega))^2
    lt.tilde norm(bold(r)_"c")_(L^2(Omega))^2
    + norm(r_"e")_(L^2(Omega))^2.
  $
  再代回 @eq:plate-h2，得到
  $
    norm(v)_(H^2(Omega))^2
    lt.tilde norm(bold(r)_"c")_(L^2(Omega))^2
    + norm(r_"e")_(L^2(Omega))^2.
  $
  最后结合
  $
    norm(bold(tau))_(bold(H)(div div))^2
    = norm(bold(tau))_(L^2(Omega))^2
    + norm(r_"e")_(L^2(Omega))^2
  $
  即得下界。由于 $c_A$ 与 $C_A$ 仅依赖于 $mu$ 与 $h$，两侧隐含常数仅依赖于 $Omega$、$mu$ 与 $h$。证毕。
]

由 @thm:plate-stability，双线性形式 $a_"KL"$ 在 $bold(Sigma)_"KL" times U_"KL"$ 上连续且强制，其中常数可关于 Lamé 常数 $lambda$ 一致选取。值得对比的是：线弹性中柔度算子的体积部分在 $lambda -> oo$ 时退化，一致稳定性依赖偏差--体积分解与迹估计。板弯曲中柔度算子本身即一致椭圆，一致稳定性转而依赖 Poincaré 不等式从曲率恢复挠度范数。

= 线性化网络离散

本节定义离散模型并给出其逼近性质：先定义单隐层特征与参数域，再转述准均匀字典的 Sobolev 逼近结果，随后给出参数点集的确定性构造，最后按物理分量组装张量与向量离散空间及系数向量，并定义施加平均迹规范与本质边界条件的投影。

== 单隐层字典

=== 特征与参数域

定义 ReLU 幂次激活
$
  rho_k (t):=max(t, 0)^k,
$
其中 $k >= 1$ 为整数。易知 $rho_k in C^(k-1)(RR)$，且在任意有界区间 $I$ 上 $rho_k in W^(k,oo)(I)$。本文的离散模型是以 $rho_k$ 为激活函数的双层网络
$
  u_N (bold(x)) := sum_(j=1)^N c_j rho_k (bold(omega)_j dot bold(x) + b_j),
$
其中 $(bold(omega)_j, b_j)$ 为隐藏层第 $j$ 个神经元的参数，$(c_j)_(j=1)^N$ 为输出层系数。隐藏层参数预先取定并冻结，不参与求解。此时模型关于输出层系数是线性的，故称线性化网络。代入凸的最小二乘泛函后，离散问题仍是有限维凸二次问题。隐藏层单个神经元给出的函数称为单隐层特征，即 $rho_k$ 与仿射函数的复合
$
  bold(x) |-> rho_k (bold(omega) dot bold(x) + b),
  quad bold(omega) in RR^d without {bold(0)},
  quad b in RR,
$
其中 $bold(omega)$ 称为方向，$b$ 称为偏置，两者即被冻结的隐藏层参数。该函数沿 $bold(omega)$ 变化，在与 $bold(omega)$ 正交的方向上取常值，这类函数称为脊函数（ridge function）。给定激活函数 $rho$ 与紧参数域 $Theta subset RR^(d+1)$，定义脊字典
$
  bb(D)_rho (Theta)
  := {bold(x) |-> rho(bold(omega) dot bold(x) + b) : (bold(omega), b) in Theta}.
$
<eq:ridge-dictionary>
有限冻结参数集生成该字典的有限子族，网络则是这些特征的线性组合。记
$
  R_Omega
  := sup_(bold(omega) in bb(S)^(d-1), bold(x) in overline(Omega)) abs(bold(omega) dot bold(x))
  = sup_(bold(x) in overline(Omega)) norm(bold(x))_(ell^2),
$
其中 $bb(S)^(d-1) subset RR^d$ 为单位球面。由 $Omega$ 有界，$R_Omega < oo$。方向--偏置的参数域取柱面 $bb(S)^(d-1) times [-c_b, c_b]$，其中偏置半径 $c_b > R_Omega$ 为取定常数。本文所用的完整脊字典简记为
$
  bb(D)_(rho_k)
  := bb(D)_(rho_k) (bb(S)^(d-1) times [-c_b, c_b]).
$
后文各物理变量的冻结特征族 ${xi_(chi,j)}_(j=1)^N$ 均为 $bb(D)_(rho_k)$ 的有限子族。记 $k$ 次多项式空间为
$
  P_k (Omega)
  := span {bold(x)^bold(alpha): bold(alpha) in NN_0^d, abs(bold(alpha)) <= k},
  quad dim P_k (Omega) = binom(d+k, k).
$
由 $abs(bold(omega) dot bold(x)) <= R_Omega$，柱面两端的特征具有简单形式：
$
  rho_k (bold(omega) dot bold(x) + b)|_Omega =
  cases(
    0 \, & quad b <= -R_Omega,
    (bold(omega) dot bold(x) + b)^k \, & quad b >= R_Omega.
  )
$
三段参数对字典的贡献因而不同。

- $bb(S)^(d-1) times [-c_b, -R_Omega]$ 一段：特征在 $Omega$ 上恒为零，不提供任何张成方向。
- $bb(S)^(d-1) times [R_Omega, c_b]$ 一段：特征都是 $k$ 次多项式，其张成恰为固定维数的 $P_k (Omega)$，用固定大小的生成组即可覆盖。验证如下：由二项展开
  $
    (bold(omega) dot bold(x) + b)^k
    = sum_(j=0)^k binom(k, j) b^(k-j) (bold(omega) dot bold(x))^j
  $
  知该段特征都是 $k$ 次多项式，其张成含于 $P_k (Omega)$。固定 $b$ 时 $(bold(omega) dot bold(x))^j$ 随 $bold(omega)$ 变化张成 $j$ 次齐次多项式。再取 $k + 1$ 个互不相同的 $b$ 作 Vandermonde 型组合，即可逐次提取各次齐次分量，故该段特征张成整个 $P_k (Omega)$。
- $bb(S)^(d-1) times [-R_Omega, R_Omega] =: cal(P)_Omega$ 一段：特征既非零函数也非多项式，随 $N$ 增长的非多项式逼近自由度只需配置在这一段。

下文据此作参数分解：在 $cal(P)_Omega$ 上配置随 $N$ 增长的点集，另从 $b > R_Omega$ 的多项式区固定选取有限个参数补足 $P_k (Omega)$。同一分解在 @LiuMaoXu2025 注记 2.1 中以球面坐标给出。该分解所达到的字典逼近阶由 @thm:band-dictionary-rate 给出，本文点集构造满足其假设的验证见 @prop:layered-quasi-uniform。

=== 球面参数化与字典逼近性质

特征只通过仿射自变量依赖于参数：
$
  bold(omega) dot bold(x) + b = (bold(omega), b) dot tilde(bold(x)),
  quad tilde(bold(x)) := (bold(x), 1) in RR^(d+1),
$
故参数的实质是向量 $(bold(omega), b) in RR^(d+1)$。由 $rho_k$ 的正 $k$ 次齐次性，该向量与其任意正数倍给出同一个特征（至多相差一个正因子），因此参数只按方向计，归一化到单位球面 $bb(S)^d subset RR^(d+1)$ 不损失任何字典元素。

柱面参数域 $bb(S)^(d-1) times [-c_b, c_b]$ 是同一族特征的另一套坐标。两套坐标由归一化映射
$
  frak(P): bb(S)^(d-1) times [-c_b, c_b] & -> bb(S)^d, \
                        (bold(omega), b) & |-> ((bold(omega), b))/sqrt(1 + b^2)
$
转换：将 $(bold(omega), b)$ 视为 $RR^(d+1)$ 中的向量并除以其欧氏范数，由 $norm(bold(omega))_(ell^2) = 1$ 该范数即 $sqrt(1 + b^2)$。该映射双向 Lipschitz 地映为 $bb(S)^d$ 的赤道带 ${bold(theta) in bb(S)^d : abs(theta_(d+1)) <= c_b\/sqrt(1+c_b^2)}$，Lipschitz 常数仅依赖于 $c_b$。具体地，对任意 $(bold(omega), b) in bb(S)^(d-1) times [-c_b, c_b]$，
$
  rho_k (bold(omega) dot bold(x) + b)
  = (1+b^2)^(k\/2) rho_k (frak(P)(bold(omega), b) dot tilde(bold(x))),
$
两种参数化因而张成同一空间。

#definition(title: [准均匀参数点集])[
  设 $Theta subset cal(P)_Omega$ 为有限点集，其覆盖半径为
  $
    h(Theta) := max_(bold(theta) in frak(P)(cal(P)_Omega))
    min_(bold(p) in Theta) op("dist")(bold(theta), frak(P)(bold(p))),
  $
  其中 $op("dist")$ 为 $bb(S)^d$ 上的测地距离。称 $Theta$ 关于规模参数 $N in NN$ 准均匀，若其在 $frak(P)(cal(P)_Omega)$ 上良分布，
  $
    h(Theta) lt.tilde N^(-1\/d),
  $
  且覆盖半径与最小间距同阶，从而排除点的聚集，
  $
    h(Theta) lt.tilde min_(bold(p) != bold(q) in Theta)
    op("dist")(frak(P)(bold(p)), frak(P)(bold(q))).
  $
  两处隐含常数都与 $N$ 无关。
]<def:quasi-uniform>

以下积分表示刻画字典可利用的光滑性上限。

#theorem(title: [球面积分表示与饱和指数（@LiuMaoXu2025 定理 2.3--2.4）])[
  记
  $
    s_"cap" (d) := (d+2k+1)/2.
  $
  则 $H^(s_"cap" (d))(Omega)$ 恰由具有球面积分表示
  $
    f(bold(x))
    = integral_(bb(S)^d) rho_k (bold(theta) dot tilde(bold(x))) psi(bold(theta)) dif bold(theta),
    quad bold(x) in Omega
  $
  且 $psi in L^2(bb(S)^d)$ 的函数 $f$ 组成。这样的 $psi$ 一般不唯一，而
  $
    norm(f)_(H^(s_"cap" (d))(Omega))
    tilde.eq inf {norm(psi)_(L^2(bb(S)^d)) :
      f(bold(x))
      = integral_(bb(S)^d) rho_k (bold(theta) dot tilde(bold(x))) psi(bold(theta)) dif bold(theta),
      quad bold(x) in Omega}.
  $
  该刻画充分必要：在某个与字典规模无关的系数预算下可被准均匀字典按 $L^2$ 逼近的函数，恰为 $H^(s_"cap" (d))(Omega)$ 的元素。
]<thm:sphere-representation>

称 $s_"cap" (d)$ 为字典的饱和指数：它来自 $rho_k (bold(theta) dot tilde(bold(x)))$ 关于 $bold(theta) in bb(S)^d$ 的球谐展开中 Legendre 系数的衰减率，度量特征族可利用的光滑性，并不限制方程解自身的正则性。因此对物理变量只需假设 Sobolev 正则性，无需单独的积分表示假设。

@thm:sphere-representation 只刻画了可被逼近函数的范围，并未给出具体点集上的收敛率。下述定理表明，在参数带 $cal(P)_Omega = bb(S)^(d-1) times [-R_Omega, R_Omega]$ 上配置准均匀点集，并从多项式区 $bb(S)^(d-1) times (R_Omega, c_b]$ 固定选取有限个参数补足 $P_k (Omega)$，两部分合并为总规模为 $N$ 的单一参数点集，其生成的字典对不超过饱和指数 $s_"cap" (d)$ 的一切正则性都给出相应的收敛率，恰好覆盖 @thm:sphere-representation 刻画的范围，并同时控制实现该收敛率的系数。它由 @LiuMaoXu2025 定理 2.2 结合其注记 2.1 的参数带分解得到。

#theorem(title: [准均匀字典的 Sobolev 逼近与系数控制（@LiuMaoXu2025 定理 2.2 与注记 2.1）])[
  设 $Theta_N subset bb(S)^(d-1) times [-c_b, c_b]$ 为有限参数点集，记 $N := abs(Theta_N)$，并有不交分解
  $
    Theta_N = Theta^"band" union.sq Theta^"poly",
  $
  其中 $Theta^"band" subset cal(P)_Omega$ 关于 $N$ 满足 @def:quasi-uniform，$h(Theta^"band")$ 为其覆盖半径。$Theta^"poly" subset bb(S)^(d-1) times (R_Omega, c_b]$ 为固定的 $n_P$ 对参数，$n_P := dim P_k (Omega)$，其生成的特征构成 $P_k (Omega)$ 的一组基。将全部参数特征枚举为字典
  $
    phi_j (bold(x)) := rho_k (bold(omega)_j dot bold(x) + b_j),
    quad 1 <= j <= N,
  $
  其中 $(bold(omega)_j, b_j)$ 枚举 $Theta_N$。则对任意 $0 <= m <= k$、$s in [m, s_"cap" (d)]$ 与 $v in H^s (Omega)$，存在系数 $(c_j)_(j=1)^N$ 使
  $
    v_N := sum_(j=1)^N c_j phi_j
  $
  满足
  $
    norm(v - v_N)_(H^m (Omega)) lt.tilde h(Theta^"band")^(s-m) norm(v)_(H^s (Omega)),
  $
  且
  $
    (sum_(j=1)^N c_j^2)^(1\/2)
    lt.tilde h(Theta^"band")^(-(s_"cap" (d)-s)) N^(-1\/2)
    norm(v)_(H^s (Omega)).
  $
  隐含常数与 $N$、$v$ 无关。逼近误差随覆盖半径以 $h(Theta^"band")^(s-m)$ 衰减。系数界随 $s$ 与饱和指数之差 $s_"cap" (d) - s$ 的增大而放大，在 $s = s_"cap" (d)$ 处与 $h(Theta^"band")$ 无关。
]<thm:band-dictionary-rate>

@thm:band-dictionary-rate 以覆盖半径 $h(Theta^"band")$ 度量逼近阶，由于 @def:quasi-uniform 给出 $h(Theta^"band") lt.tilde N^(-1\/d)$，可将覆盖半径的估计转换为关于点集规模 $N$ 的估计，从而得到下述推论。

#corollary(title: [准均匀字典关于点集规模的逼近阶])[
  在 @thm:band-dictionary-rate 的假设与其字典枚举 $(phi_j)_(j=1)^N$ 下，对任意 $0 <= m <= k$、$s in [m, s_"cap" (d)]$ 与 $v in H^s (Omega)$，存在系数 $(c_j)_(j=1)^N$ 使 $v_N := sum_(j=1)^N c_j phi_j$ 满足
  $
    norm(v - v_N)_(H^m (Omega))
    lt.tilde N^(-(s-m)\/d) norm(v)_(H^s (Omega)),
  $
  且
  $
    (sum_(j=1)^N c_j^2)^(1\/2)
    lt.tilde N^((s_"cap" (d)-s)\/d-1\/2)
    norm(v)_(H^s (Omega)).
  $
  两处隐含常数由 @thm:band-dictionary-rate 与 @def:quasi-uniform 中的常数决定，与 $N$、$v$ 无关。
]<cor:quasi-uniform-rate>

#proof[
  由 @def:quasi-uniform，$h(Theta^"band") lt.tilde N^(-1\/d)$。反之，$frak(P)(cal(P)_Omega)$ 是 $bb(S)^d$ 上测度为正的固定区域，以 $Theta^"band"$ 的 $abs(Theta^"band")$ 个像点为心、$h(Theta^"band")$ 为半径的测地球将其覆盖，故 $abs(Theta^"band") h(Theta^"band")^d gt.tilde 1$。又因 $abs(Theta^"band") <= N$，故 $h(Theta^"band") gt.tilde N^(-1\/d)$。于是 $h(Theta^"band") tilde.eq N^(-1\/d)$。代入 @thm:band-dictionary-rate 的两个估计：由 $s >= m$ 得 $h(Theta^"band")^(s-m) lt.tilde N^(-(s-m)\/d)$，由 $s <= s_"cap" (d)$ 得 $h(Theta^"band")^(-(s_"cap" (d)-s)) lt.tilde N^((s_"cap" (d)-s)\/d)$。证毕。
]

=== 参数点集构造

以 $chi$ 标记物理变量：线弹性中 $chi in {bold(sigma), bold(u)}$，板弯曲中 $chi in {bold(M), w}$。对每个物理变量，取含 $N$ 对 $(bold(omega), b)$ 的确定性点集，并作不交分解
$
  Theta_(chi,N) = Theta^"poly" union.sq Theta_(chi,N)^"band",
  quad Theta^"poly" inter Theta_(chi,N)^"band" = emptyset.
$
其中 $Theta^"poly" subset bb(S)^(d-1) times (R_Omega,c_b]$ 是固定的 $n_P$ 对参数：前文的二项展开已说明多项式区特征张成 $P_k (Omega)$，而张成组必含一组基，故可取出这样的 $n_P$ 对参数，使其生成的特征构成 $P_k (Omega)$ 的一组基。$Theta_(chi,N)^"band" subset cal(P)_Omega$ 含其余 $N-n_P$ 对参数。故 $abs(Theta_(chi,N)^"band") tilde.eq N$，多项式补充的大小 $n_P = dim P_k (Omega) =binom(d+k, k)$ 只依赖于 $d$、$k$，与 $N$ 无关。记 $Theta_(chi,N)$ 的元素为 $(bold(omega)_(chi,j),b_(chi,j))$，$1<=j<=N$，并定义
$
  xi_(chi,j) (bold(x)) := rho_k (bold(omega)_(chi,j) dot bold(x) + b_(chi,j)),
  quad 1<=j<=N.
$
常数函数含于 $P_k (Omega)$，已由 $Theta^"poly"$ 生成的特征张成，故本文的离散模型不另设输出偏置，字典元素与参数一一对应。这与 @LiuMaoXu2025 注记 2.1 的取法一致：该注记在多项式区取恰好 $binom(k+d, d)=n_P$ 个参数使其 $k$ 次幂线性无关，同样不单列常数特征。

参数带中的点按乘积坐标分层构造。令
$
  t(b) := (b+R_Omega)/(2R_Omega) in [0,1],
  quad b(t) := -R_Omega+2R_Omega t.
$
将 $[0,1]$ 等分为 $n_2 tilde.eq N^(1\/d)$ 段并取各段中点 $t_l := (l-1\/2)\/n_2$。第 $l$ 层配置准均匀方向集 $Lambda_l subset bb(S)^(d-1)$，各层点数至多相差 $1$，均为 $n_1 tilde.eq N^((d-1)\/d)$，并取
$
  Theta_(chi,N)^"band"
  := union.big_(l=1)^(n_2) {(bold(omega),b(t_l)):bold(omega) in Lambda_l}.
$
方向集按维数取：

- $d = 2$ 取等距角网格，其覆盖半径为 $pi\/n_1$，该界直接得到。
- $d = 3$ 取 $bb(S)^2$ 上的 Fibonacci 格，其准均匀性见 @HardinMichaelsSaff2016。

#figure(
  code-image(class: "center", theme => diagram(
    spacing: (24mm, 12mm),
    node((0, 0), $bb(S)^(d-1) times [0, 1]$, name: <prod>),
    node((1, 0), $cal(P)_Omega$, name: <cyl>),
    node((2, 0), $frak(P)(cal(P)_Omega) subset bb(S)^d$, name: <sph>),
    edge(<prod>, <cyl>, $(bold(omega), t) |-> (bold(omega), b(t))$, "->", stroke: 0.6pt + theme.main-color),
    edge(<cyl>, <sph>, $frak(P)$, "->", stroke: 0.6pt + theme.main-color),
    edge(<prod>, <sph>, [双向 Lipschitz], "<->", bend: 38deg, stroke: 0.6pt + theme.main-color),
    node((0, 0.45), text(0.8em)[点集构造 $Lambda_l times {t_l}$]),
    node((1, 0.45), text(0.8em)[柱面坐标]),
    node((2, 0.45), text(0.8em)[准均匀性在此度量]),
  )),
  caption: [参数域的三套坐标：点集在乘积坐标中构造，准均匀性则按 @def:quasi-uniform 在 $bb(S)^d$ 上度量。两段映射均双向 Lipschitz，故覆盖半径与最小间距在三套坐标中各差至多一个与 $N$ 无关的常数因子],
)<fig:least-squares-param-charts>

#proposition(title: [分层点集的准均匀性])[
  上述 $Theta_(chi,N)^"band"$ 关于 $N$ 满足 @def:quasi-uniform。
]<prop:layered-quasi-uniform>

#proof[
  仿射映射 $(bold(omega),t) |-> (bold(omega),-R_Omega+2R_Omega t)$ 将 $bb(S)^(d-1) times [0,1]$ 双向 Lipschitz 地映到 $cal(P)_Omega$，$frak(P)$ 在有界柱面上同样双向 Lipschitz。故只需在乘积度量下验证两条要求。

  在乘积度量下，层距为 $n_2^(-1) tilde.eq N^(-1\/d)$。由 $Lambda_l$ 的准均匀性，其在 $bb(S)^(d-1)$ 上的覆盖半径与最小间距均为 $n_1^(-1\/(d-1)) tilde.eq N^(-1\/d)$ 量级。$bb(S)^(d-1) times [0,1]$ 中任一点到最近层的距离不超过层距的一半，到该层内最近点的距离不超过 $Lambda_l$ 的覆盖半径，故 $Theta_(chi,N)^"band"$ 的覆盖半径 $lt.tilde N^(-1\/d)$。其最小间距在同层内为 $Lambda_l$ 的最小间距，跨层则不小于层距，故 $gt.tilde N^(-1\/d)$，从而不小于覆盖半径的常数倍。证毕。
]

=== 张量与向量字典

以上是标量字典，本文考虑的两类问题未知量为对称张量场与向量场（板弯曲的挠度为标量），故逐分量张成。记 $n_s=d(d+1)/2$，取 $op("Sym")(d)$ 的固定正交基
${bold(T)_alpha}_(alpha=1)^(n_s)$ 与 $RR^d$ 的标准基
${bold(e)_i}_(i=1)^d$。线弹性的原始空间为
$
  hat(bold(Sigma))_("LE",N) & := span {xi_(bold(sigma),j) bold(T)_alpha:
                                1<=j<=N, 1<=alpha<=n_s}, \
      hat(bold(U))_("LE",N) & := span {xi_(bold(u),j) bold(e)_i:
                                1<=j<=N, 1<=i<=d}.
$
板弯曲的原始空间为
$
  hat(bold(Sigma))_("KL",N) & := span {xi_(bold(M),j) bold(T)_alpha:
                                1<=j<=N, 1<=alpha<=3}, \
            hat(U)_("KL",N) & := span {xi_(w,j):1<=j<=N}.
$
本文以上标 $hat(dot)$ 统一标记这类尚未施加平均迹规范或边界投影的原始字典空间，以及其中的原始近似与定义在其上的泛函（如后文的 $hat(v)_N$ 与 $hat(cal(J))_"LE"$）。

== 系数向量 <sec:coeff-vector>

将给定模型的全部独立输出系数依固定顺序排成向量 $bold(c) in RR^(m_N)$，其中 $m_N$ 是离散系数总数，即上述原始字典空间的维数之和。

线弹性中 $m_N=(n_s+d)N$，系数分应力块 $(c_(j,alpha))_(1<=j<=N, 1<=alpha<=n_s)$ 与位移块 $(c_(j,i))_(1<=j<=N, 1<=i<=d)$，相应映射为
$
  hat(bold(sigma))_N (bold(c)) := sum_(j,alpha) c_(j,alpha) xi_(bold(sigma),j) bold(T)_alpha,
  quad
  hat(bold(u))_N (bold(c)) := sum_(j,i) c_(j,i) xi_(bold(u),j) bold(e)_i.
$
板弯曲为二维问题，$n_s=3$，$m_N=4N$，系数分弯矩块 $(c_(j,alpha))_(1<=j<=N, 1<=alpha<=3)$ 与挠度块 $(c_j)_(1<=j<=N)$，相应映射为
$
  hat(bold(M))_N (bold(c)) := sum_(j,alpha) c_(j,alpha) xi_(bold(M),j) bold(T)_alpha,
  quad
  hat(w)_N (bold(c)) := sum_(j=1)^N c_j xi_(w,j).
$

== 连续正交投影

两条残差稳定性定理都只在约束空间上陈述：@thm:elasticity-stability 要求 $(bold(tau), bold(v)) in bold(X)_"LE"$，@thm:plate-stability 要求挠度属于 $H_0^2(Omega)$。两条约束作用于不同物理分量且彼此独立，分别由应力的平均迹投影与位移（挠度）的本质边界投影施加。原始字典空间保持不变，投影复合进被最小化的泛函。

原始特征是 $rho_k$ 与仿射函数的复合，由 $rho_k$ 在有界区间上属于 $W^(k,oo)$，原始字典空间的元素逐分量属于 $W^(k,oo)(Omega) subset H^k (Omega)$。线弹性的图范数只含一阶导数，因此 $k >= 1$ 已保证 $hat(bold(Sigma))_("LE",N) subset bold(H)(div)$ 与 $hat(bold(U))_("LE",N) subset H^1(Omega; RR^d)$。板弯曲的图范数含二阶导数，以下对板弯曲设 $k >= 2$，从而 $hat(bold(Sigma))_("KL",N) subset bold(H)(div div)$ 与 $hat(U)_("KL",N) subset H^2(Omega)$。

平均迹规范由平均迹投影施加。对 $bold(tau) in bold(H)(div, Omega; op("Sym")(d))$ 定义
$
  Pi_"tr" bold(tau)
  := bold(tau)
  - 1/(d abs(Omega))
  (integral_Omega tr(bold(tau)) dif x) bold(I).
$
由 $(bold(tau), bold(I))_(L^2(Omega)) = integral_Omega tr(bold(tau)) dif x$ 与 $norm(bold(I))_(L^2(Omega))^2 = d abs(Omega)$，上式即
$
  Pi_"tr" bold(tau)
  = bold(tau)
  - ((bold(tau), bold(I))_(L^2(Omega)))/(norm(bold(I))_(L^2(Omega))^2) bold(I),
$
故 $Pi_"tr"$ 是 $L^2(Omega; op("Sym")(d))$ 中向 $span {bold(I)}$ 的正交补（即零平均迹子空间）的 $L^2$ 正交投影，$L^2$ 范数不增。被减去的常数球张量散度为零，散度不变，故 $Pi_"tr"$ 保持 $bold(H)(div)$ 正则性且在图范数下非扩张。若 $bold(tau) in bold(Sigma)_"LE"$，其平均迹已为零，故 $Pi_"tr" bold(tau) = bold(tau)$，从而对任意 $bold(sigma)_star in bold(Sigma)_"LE"$ 与 $hat(bold(tau)) in bold(H)(div)$ 有
$
  norm(bold(sigma)_star - Pi_"tr" hat(bold(tau)))_(bold(H)(div))
  = norm(Pi_"tr" (bold(sigma)_star - hat(bold(tau))))_(bold(H)(div))
  <= norm(bold(sigma)_star - hat(bold(tau)))_(bold(H)(div)).
$
<eq:projection-transfer-tr>

$Pi_"tr"$ 有显式公式，其中只有区域积分需要计算。

本质边界条件由椭圆正交投影施加。对 $m in {1,2}$，令 $Pi_D^(m)$ 为 $H^m (Omega)$ 到闭子空间 $H_0^m (Omega)$ 的正交投影（存在性、最佳逼近性与非扩张性参见 @BrennerScott2008 命题 2.3.1）：
$
  (Pi_D^(m) v,z)_(H^m)
  =(v,z)_(H^m),
  quad forall z in H_0^m (Omega).
$
<eq:boundary-projection>
由于正交投影的算子范数为 $1$，
$
  norm(Pi_D^(m) v)_(H^m)<=norm(v)_(H^m).
$
若 $v_star in H_0^m (Omega)$，则 $Pi_D^(m) v_star = v_star$，从而对任意 $hat(v)_N in H^m (Omega)$ 有
$
  norm(v_star-Pi_D^(m) hat(v)_N)_(H^m)
  =norm(Pi_D^(m)(v_star-hat(v)_N))_(H^m)
  <=norm(v_star-hat(v)_N)_(H^m).
$
<eq:projection-transfer>

两条传递估计的结构相同：投影以约束空间中的场为不动点且范数不增，故不放大字典逼近误差，所需假设仅是物理变量本身的 Sobolev 正则性。两个投影的差别在于可计算性：与平均迹投影不同，$Pi_D^(m)$ 没有显式形式，实际计算中必须近似求解其投影问题。

因此离散模型的未知量仍是原始字典空间的系数，物理残差在投影后的场上计算。将正交投影复合到冻结神经字典，并研究其离散实现，是本文采用的构造。

== 投影的离散实现

上一节的两个连续投影中，$Pi_"tr"$ 有显式公式，$Pi_D^(m)$ 则没有。本节先说明 $Pi_"tr"$ 在字典空间上归结为有限个脊特征的区域积分，并给出使这些积分具有闭式的区域类，在该类区域上平均迹投影不经任何离散即可精确实现。对于本质边界投影 $Pi_D^(m)$，则通过有限维协调空间中的 Ritz 投影近似求解，并采用 Monte Carlo 求积计算投影方程中的内积。

=== 平均迹投影

对系数向量 $bold(c)$，由 $tr(bold(T)_alpha)$ 为常数，
$
  1/abs(Omega) integral_Omega tr(hat(bold(sigma))_N (bold(c))) dif x
  = sum_(j,alpha) c_(j,alpha) tr(bold(T)_alpha) overline(xi)_(bold(sigma),j),
  quad
  overline(xi)_(bold(sigma),j) := 1/abs(Omega) integral_Omega xi_(bold(sigma),j) dif x,
$
<eq:trace-means>
故 $Pi_"tr"$ 在应力字典空间 $hat(bold(Sigma))_("LE",N)$ 上只需 $N$ 个脊特征的区域平均值 $overline(xi)_(bold(sigma),j)$。

为精确计算这些平均值，下面考虑具有有限单纯形剖分的区域。利用积分的可加性，区域积分可分解为各单纯形上的积分。而在每个单纯形上，脊特征的积分可由仿射函数在顶点处的取值通过一维差商表示。

#definition(title: [单纯形剖分])[
  设 $Omega subset RR^d$ 为有界 Lipschitz 区域。若由有限个非退化闭 $d$ 维单纯形组成的集合 $cal(T)$ 满足
  $
    overline(Omega) = union.big_(S in cal(T)) S,
    quad
    op("int")(S) inter op("int")(S') = emptyset
    quad (S != S'),
  $
  其中 $S, S' in cal(T)$，$op("int")(S)$ 表示 $S$ 在 $RR^d$ 中的内部，则称 $cal(T)$ 为 $Omega$ 的一个单纯形剖分。
]<def:admissible-domain>

差商按重节点约定：对 $g in C^d (RR)$ 与节点 $t_0, dots, t_d$，$g[t_0, dots, t_d]$ 为 $g$ 在这些节点上的 Hermite 插值多项式的 $d$ 次项系数，节点两两不同时即通常的差商。下述引理把脊特征在单纯形上的积分化为 $rho_(k+d)$ 在顶点值处的差商。

#lemma(title: [单纯形剖分上的脊特征积分])[
  设 $Omega$ 具有有限单纯形剖分 $cal(T) = {S_i}_(i=1)^M$，$S_i$ 的顶点为 $bold(v)_(i,0), dots, bold(v)_(i,d)$。则对任意整数 $k >= 1$、$bold(omega) in RR^d$ 与 $b in RR$，
  $
    integral_Omega rho_k (bold(omega) dot bold(x) + b) dif x
    = (d! thin k!)/((k+d)!) sum_(i=1)^M abs(S_i) thin
    rho_(k+d) [bold(omega) dot bold(v)_(i,0) + b, dots, bold(omega) dot bold(v)_(i,d) + b].
  $
  <eq:simplex-mean>
]<lem:simplex-mean>

#proof[
  对 $g in C^d (RR)$ 与任意节点 $t_0, dots, t_d$，Hermite--Genocchi 公式（@deBoor2005 第 9 节）给出
  $
    g[t_0, dots, t_d]
    = integral_(Delta_d) g^((d)) (sum_(l=0)^d lambda_l t_l) dif bold(lambda),
    quad
    Delta_d := {bold(lambda) in RR^d : lambda_l >= 0, sum_(l=1)^d lambda_l <= 1},
    quad
    lambda_0 := 1 - sum_(l=1)^d lambda_l.
  $
  取 $g := (k!)/((k+d)!) rho_(k+d)$。由 $rho_(k+d) in C^(k+d-1)(RR)$ 与 $k >= 1$ 有 $g in C^d (RR)$，逐次求导得 $g^((d)) = rho_k$。固定 $i$，仿射映射
  $
    bold(lambda) |-> bold(x)(bold(lambda))
    := bold(v)_(i,0) + sum_(l=1)^d lambda_l (bold(v)_(i,l) - bold(v)_(i,0))
  $
  把 $Delta_d$ 映为 $S_i$，其 Jacobi 行列式的绝对值为 $d! abs(S_i)$，且 $bold(omega) dot bold(x)(bold(lambda)) + b = sum_(l=0)^d lambda_l (bold(omega) dot bold(v)_(i,l) + b)$。换元得
  $
    integral_(S_i) rho_k (bold(omega) dot bold(x) + b) dif x
    &= d! abs(S_i) integral_(Delta_d) g^((d)) (sum_(l=0)^d lambda_l (bold(omega) dot bold(v)_(i,l) + b)) dif bold(lambda) \
    &= d! abs(S_i) thin g[bold(omega) dot bold(v)_(i,0) + b, dots, bold(omega) dot bold(v)_(i,d) + b].
  $
  各单纯形内部两两不交，对 $i$ 求和即得 @eq:simplex-mean。证毕。
]

由 @lem:simplex-mean，在具有有限单纯形剖分的区域上，每个 $overline(xi)_(bold(sigma),j)$ 与 $abs(Omega) = sum_(i=1)^M abs(S_i)$ 都是顶点数据的有限闭式，$Pi_"tr"$ 在 $hat(bold(Sigma))_("LE",N)$ 上按 @eq:trace-means 精确实现，不引入截断或求积误差。

#example(title: [单位盒])[
  $Omega = (0,1)^d$ 具有有限单纯形剖分。对 ${1, dots, d}$ 的每个置换 $pi$，$S_pi := {bold(x) in [0,1]^d : x_(pi(1)) <= dots.c <= x_(pi(d))}$ 是体积为 $1\/d!$ 的单纯形，$d!$ 个 $S_pi$ 内部两两不交且并为 $[0,1]^d$。盒的乘积结构还给出不经剖分的闭式。设 $bold(omega) in RR^d without {bold(0)}$，令 $i_1, dots, i_n$ 为 $bold(omega)$ 的非零分量指标，$a_l := abs(omega_(i_l))$，$tilde(b) := b + sum_(omega_i < 0) omega_i$。则
  $
    integral_Omega rho_k (bold(omega) dot bold(x) + b) dif x
    = (k!)/((k+n)! product_(l=1)^n a_l)
    sum_(bold(v) in {0,1}^n) (-1)^(n - sum_(l=1)^n v_l)
    rho_(k+n) (tilde(b) + sum_(l=1)^n a_l v_l).
  $
  <eq:box-mean>
]<ex:box-domain>

#proof[
  对每个 $omega_i < 0$ 的坐标作变量替换 $x_i = 1-y_i$，其余坐标令 $x_i=y_i$。该变换保持单位盒不变，且 Jacobian 行列式的绝对值为 $1$，因而
  $
    bold(omega) dot bold(x) + b
    = tilde(b) + sum_(l=1)^n a_l y_(i_l).
  $
  变换后的被积函数仅依赖于 $y_(i_1), dots, y_(i_n)$。由 Fubini 定理，先对其余坐标积分；由于这些坐标的积分区间均为 $(0,1)$，相应因子均为 $1$。将保留下来的坐标依次记为 $t_1, dots, t_n$，得到
  $
    integral_Omega rho_k (bold(omega) dot bold(x) + b) dif x
    = integral_0^1 dots integral_0^1
      rho_k (tilde(b) + sum_(l=1)^n a_l t_l)
      dif t_n dots dif t_1.
  $

  对任意 $a>0$、$c in RR$ 与整数 $q>=1$，由 $rho_(q+1)'=(q+1)rho_q$（几乎处处成立）及微积分基本定理，有
  $
    integral_0^1 rho_q(c+a t) dif t
    = (rho_(q+1)(c+a)-rho_(q+1)(c))/((q+1)a).
  $
  对上述重积分逐次应用这一恒等式。每次积分均产生上端点值与下端点值之差，并将幂次增加 $1$；经过 $n$ 次积分后，公共系数为
  $
    product_(l=1)^n 1/((k+l)a_l)
    = (k!)/((k+n)! product_(l=1)^n a_l).
  $
  展开各次端点差所得的 $2^n$ 项，以 $bold(v) in {0,1}^n$ 标记所选端点。端点 $bold(v)$ 对应的函数值为 $rho_(k+n)(tilde(b)+sum_(l=1)^n a_l v_l)$，其中每选取一个下端点便产生一个负号，故该项的符号为 $(-1)^(n-sum_(l=1)^n v_l)$。对所有端点求和即得 @eq:box-mean。证毕。
]

=== 本质边界投影

由 @eq:boundary-projection，计算 $Pi_D^(m)v$ 需要在 $H_0^m (Omega)$ 中求解投影方程。为得到有限阶线性方程组，将该方程的试探空间与检验空间同时限制为有限维协调子空间。由于所得函数仍属于 $H_0^m (Omega)$，这一近似保留了齐次本质边界条件。具体地，对 $m in {1,2}$，取
$
  V_K^(m) subset H_0^m (Omega),
  quad dim V_K^(m)=K,
$
并定义采用精确积分的 Ritz 投影 $Pi_(D,K)^(m):H^m (Omega)->V_K^(m)$：
$
  (Pi_(D,K)^(m)v,z_K)_(H^m)
  =(v,z_K)_(H^m),
  quad forall z_K in V_K^(m).
$
冻结字典函数经投影后的像构成实际试探空间：

- 线弹性取 $m=1$：$Pi_(D,K)^(1)$ 逐分量作用于位移字典空间 $hat(bold(U))_("LE",N)$，像空间逐分量含于 $V_K^(1)$。
- 板弯曲取 $m=2$：$Pi_(D,K)^(2)$ 作用于挠度字典空间 $hat(U)_("KL",N)$，像空间含于 $V_K^(2) subset H_0^2(Omega)$。

$Pi_(D,K)^(m)$ 是线性映射，故两个试探空间的维数都不超过相应字典空间的维数。协调空间只用于投影方程的离散化，不引入新的模型未知量。

$Pi_(D,K)^(m)$ 的定义方程以 $H^m$ 内积为双线性形式，故 $v-Pi_(D,K)^(m) v$ 与 $V_K^(m)$ 在该内积下正交。$V_K^(m)$ 有限维从而闭，$Pi_(D,K)^(m)$ 因而是 $H^m (Omega)$ 到 $V_K^(m)$ 的正交投影。由该正交性，对任意 $z_K in V_K^(m)$，
$
  norm(v-z_K)_(H^m (Omega))^2
  = norm(v-Pi_(D,K)^(m) v)_(H^m (Omega))^2
  + norm(Pi_(D,K)^(m) v-z_K)_(H^m (Omega))^2.
$
取 $z_K=0$ 得 $norm(Pi_(D,K)^(m) v)_(H^m (Omega))<=norm(v)_(H^m (Omega))$，而 $Pi_(D,K)^(m)$ 在 $V_K^(m)$ 上为恒等，故算子范数为 $1$。上式右端第二项非负，故 $norm(v-z_K)_(H^m (Omega))$ 在 $z_K=Pi_(D,K)^(m) v$ 处取到最小值，即 $Pi_(D,K)^(m) v$ 是 $v$ 在 $V_K^(m)$ 中关于 $H^m$ 范数的最佳逼近。记该空间的最佳逼近量为
$
  cal(E)_(m,K)(v)
  := inf_(z_K in V_K^(m))
  norm(v-z_K)_(H^m (Omega))
  = norm(v-Pi_(D,K)^(m)v)_(H^m (Omega)),
  quad v in H^m (Omega).
$
<eq:aux-best-approx>
向量场以分量乘积空间及相应乘积 $H^m$ 范数定义同一记号。下述引理说明 $Pi_(D,K)^(m)$ 恰是连续投影 $Pi_D^(m)$ 向有限维协调空间的截断，并将截断误差归结为最佳逼近量。

#lemma(title: [Ritz 投影是连续投影的截断])[
  对任意 $v in H^m (Omega)$，
  $
    Pi_(D,K)^(m) v = Pi_(D,K)^(m) Pi_D^(m) v,
    quad
    norm(Pi_D^(m) v - Pi_(D,K)^(m) v)_(H^m (Omega))
    = cal(E)_(m,K)(Pi_D^(m) v).
  $
]<lem:ritz-truncation>

#proof[
  由 $Pi_D^(m)$ 的定义，$v - Pi_D^(m) v$ 与整个 $H_0^m (Omega)$ 正交，而 $V_K^(m) subset H_0^m (Omega)$，故对任意 $z_K in V_K^(m)$，
  $
    (Pi_(D,K)^(m)(v - Pi_D^(m) v), z_K)_(H^m)
    = (v - Pi_D^(m) v, z_K)_(H^m)
    = 0.
  $
  $Pi_(D,K)^(m)(v - Pi_D^(m) v)$ 属于 $V_K^(m)$ 且与 $V_K^(m)$ 正交，故为零，由线性即得第一式。移项得
  $
    Pi_D^(m) v - Pi_(D,K)^(m) v
    = (I - Pi_(D,K)^(m)) Pi_D^(m) v.
  $
  由 @eq:aux-best-approx，右端的范数即 $cal(E)_(m,K)(Pi_D^(m) v)$。证毕。
]

@lem:ritz-truncation 把截断误差归结为 $Pi_D^(m) v$ 在 $V_K^(m)$ 中的最佳逼近量，其阶取决于 $Pi_D^(m) v$ 的正则性。$v$ 取原始字典场 $hat(v)_N$ 时，$rho_k$ 特征经 $Pi_D^(m)$ 投影后的 Sobolev 正则性没有现成估计。以 $H_0^m (Omega)$ 中的场 $v_star$ 为比较对象则不需要这一正则性：对任意 $v_star in H_0^m (Omega)$ 与 $hat(v)_N in H^m (Omega)$，由三角不等式、@eq:aux-best-approx 与 $Pi_(D,K)^(m)$ 的非扩张性，
$
  norm(v_star-Pi_(D,K)^(m)hat(v)_N)_(H^m) & <= norm(v_star-Pi_(D,K)^(m)v_star)_(H^m)
                                            + norm(Pi_(D,K)^(m)(v_star-hat(v)_N))_(H^m) \
                                          & <= cal(E)_(m,K)(v_star)
                                            + norm(v_star-hat(v)_N)_(H^m).
$
<eq:finite-projection-transfer>
这是连续投影传递估计 @eq:projection-transfer 的有限维形式。第一项只刻画协调空间截断，且只涉及 $v_star$ 自身的正则性。第二项完整保留无约束字典的逼近能力。

上述 Ritz 投影将问题限制在有限维空间中，但其方程仍包含精确的 $H^m$ 内积。选定 $V_K^(m)$ 的一组基后，组装系数矩阵和右端向量都需要计算区域积分。为实现这些计算，取 $Omega$ 上的独立均匀样本 ${bold(x)_("R",r)}_(r=1)^(Q_"R")$，称为 Ritz 投影构造样本，并用其作 Monte Carlo 求积以近似这些内积。

记多重指标个数为 $n_(bold(alpha)) := binom(d+m, m)$，将满足 $abs(bold(alpha)) <= m$ 的多重指标按固定次序排为 $bold(alpha)_1, dots, bold(alpha)_(n_(bold(alpha)))$，并定义至 $m$ 阶导数的点值算子
$
  cal(L)_m (bold(x)): H^m (Omega) -> RR^(n_(bold(alpha))),
  quad
  cal(L)_m (bold(x))v
  := (partial^(bold(alpha)_i) v(bold(x)))_(i=1)^(n_(bold(alpha))).
$
<eq:jet-operator>
后文凡以 $bold(alpha)$ 为指标的向量与矩阵均按该次序排列。$H^m$ 内积是该算子逐点内积在 $Omega$ 上的积分，
$
  (v,z)_(H^m)
  = sum_(abs(bold(alpha)) <= m) integral_Omega
  partial^bold(alpha) v partial^bold(alpha) z dif x
  = integral_Omega cal(L)_m (bold(x))v dot cal(L)_m (bold(x))z dif x,
$
以独立均匀样本作 Monte Carlo 估计即得求积内积
$
  (v,z)_(H^m,Q_"R")
  := abs(Omega)/Q_"R" sum_(r=1)^(Q_"R")
  cal(L)_m (bold(x)_("R",r))v dot cal(L)_m (bold(x)_("R",r))z.
$
据此，定义完全离散的投影 $tilde(Pi)_(D,K,Q_"R")^(m)v in V_K^(m)$，使其满足
$
  (tilde(Pi)_(D,K,Q_"R")^(m)v,z_K)_(H^m,Q_"R")
  = (v,z_K)_(H^m,Q_"R"),
  quad forall z_K in V_K^(m).
$
当求积内积在 $V_K^(m)$ 上正定时，该方程有唯一解。由于解仍取值于 $V_K^(m)$，数值积分也保留了齐次本质边界条件。至此，本质边界投影的计算归结为有限维空间中的求积与线性方程组求解，其误差分别来自空间截断和内积的数值积分。

= 完全离散最小二乘与误差分析

本章先给出训练问题，再按误差来源逐项估计。训练泛函一节把系数向量映到采用数值积分构造的投影场，在该场上以独立的 Monte Carlo 样本离散最小二乘泛函。其后各节依次估计系数球内的字典 Sobolev 逼近误差、投影的空间截断与积分误差，并以 Rademacher 复杂度控制训练泛函的一致偏差，最后合并为总误差估计。

== 训练泛函 <sec:train-functional>

=== 字典场与投影场

给定字典与系数向量 $bold(c)$，记原始字典场为 $hat(bold(z))_N (bold(c))$，连续投影场为 $bold(z)_N (bold(c))$，采用精确积分的 Ritz 投影场为 $bold(z)_(N,K)(bold(c))$，数值积分后的投影场为 $tilde(bold(z))_(N,K)(bold(c))$。这四类场分别对应原始字典展开、连续约束投影、投影空间截断以及投影积分的数值近似。

从 $hat(bold(z))_N (bold(c))$ 出发，施加本质边界投影 $Pi_D^(m)$，并在线弹性中施加平均迹投影 $Pi_"tr"$，得到 $bold(z)_N (bold(c))$。将本质边界投影的检验空间截断为 $V_K^(m)$，保留精确的 $H^m$ 内积，即以 Ritz 投影 $Pi_(D,K)^(m)$ 代替 $Pi_D^(m)$，得到 $bold(z)_(N,K)(bold(c))$。进一步用 Monte Carlo 求积近似 Ritz 内积，得到 $tilde(bold(z))_(N,K)(bold(c))$。

若 $bold(c)^"out"$ 为求解所得的系数向量，则计算解为 $bold(z)_(N,Q,K) := tilde(bold(z))_(N,K)(bold(c)^"out")$，其中 $Q$ 为训练点数。四类场及计算解的关系如 @fig:least-squares-field-relations 所示。

#figure(
  code-image(class: "center", theme => diagram(
    spacing: (46mm, 30mm),
    node((0, 0), [$hat(bold(z))_N (bold(c))$ \ #text(0.75em)[原始字典场]], name: <z-raw>),
    node((1, 0), [$bold(z)_N (bold(c))$ \ #text(0.75em)[连续投影场]], name: <z-cont>),
    node((2, 0), [$bold(z)_(N,K)(bold(c))$ \ #text(0.75em)[Ritz 投影场 \ （精确积分）]], name: <z-ritz>),
    node((1, 1), [$tilde(bold(z))_(N,K)(bold(c))$ \ #text(0.75em)[投影场 \ （数值积分）]], name: <z-emp>),
    node((2, 1), [$bold(z)_(N,Q,K)$ \ #text(0.75em)[计算解]], name: <z-out>),
    edge(<z-raw>, <z-cont>, [#text(0.8em)[连续投影]], "->", stroke: 0.6pt + theme.main-color),
    edge(<z-cont>, <z-ritz>, [#text(0.8em)[投影空间截断]], "->", stroke: 0.6pt + theme.main-color),
    edge(
      <z-raw>,
      <z-emp>,
      [#text(0.8em)[投影的数值积分实现]],
      "->",
      label-side: right,
      stroke: 0.6pt + theme.main-color,
    ),
    edge(
      <z-ritz>,
      <z-emp>,
      [#text(0.8em)[投影求积误差]],
      "<->",
      stroke: (paint: theme.main-color, thickness: 0.6pt, dash: "dashed"),
    ),
    edge(<z-emp>, <z-out>, $bold(c)=bold(c)^"out"$, "->", stroke: 0.6pt + theme.main-color),
  )),
  caption: [字典场、投影场与计算解的关系。实线表示由起点场构造终点场的映射，虚线标记采用精确积分的 Ritz 投影场与数值积分后的投影场之间的投影求积误差。],
)<fig:least-squares-field-relations>

对于线弹性问题，给定系数向量 $bold(c) in RR^((n_s+d)N)$，定义应力--位移字典场及其投影场为
$
       hat(bold(z))_N (bold(c)) & := (hat(bold(sigma))_N (bold(c)), hat(bold(u))_N (bold(c))), \
            bold(z)_N (bold(c)) & := (Pi_"tr" hat(bold(sigma))_N (bold(c)), Pi_D^(1) hat(bold(u))_N (bold(c))), \
         bold(z)_(N,K)(bold(c)) & := (Pi_"tr" hat(bold(sigma))_N (bold(c)), Pi_(D,K)^(1) hat(bold(u))_N (bold(c))), \
  tilde(bold(z))_(N,K)(bold(c)) & := (
                                    Pi_"tr" hat(bold(sigma))_N (bold(c)),
                                    tilde(Pi)_(D,K,Q_"R")^(1) hat(bold(u))_N (bold(c))
                                  ).
$

对于 Kirchhoff--Love 板弯曲问题，给定系数向量 $bold(c) in RR^(4N)$，相应的弯矩--挠度字典场及其投影场定义为
$
       hat(bold(z))_N (bold(c)) & := (hat(bold(M))_N (bold(c)), hat(w)_N (bold(c))), \
            bold(z)_N (bold(c)) & := (hat(bold(M))_N (bold(c)), Pi_D^(2) hat(w)_N (bold(c))), \
         bold(z)_(N,K)(bold(c)) & := (hat(bold(M))_N (bold(c)), Pi_(D,K)^(2) hat(w)_N (bold(c))), \
  tilde(bold(z))_(N,K)(bold(c)) & := (hat(bold(M))_N (bold(c)), tilde(Pi)_(D,K,Q_"R")^(2) hat(w)_N (bold(c))).
$

原始特征逐分量属于 $W^(k,oo)(Omega) subset H^k (Omega)$。因而在线弹性情形下，$k >= 1$ 保证上述各场属于 $bold(H)(div) times H^1(Omega; RR^d)$。对于板弯曲，$k >= 2$ 则保证各场属于 $bold(H)(div div) times H^2(Omega)$。

=== 离散泛函

令 ${bold(x)_ell}_(ell=1)^Q$ 为 $Omega$ 上独立均匀样本，与 Ritz 投影构造样本相互独立，并设 $abs(Omega)$ 为区域测度，两个模型的离散泛函分别如下。

对线弹性，取 $bold(z)=(bold(sigma),bold(u)) in bold(H)(div) times H^1(Omega; RR^d)$，将本构残差与平衡残差依次记为
$
  bold(r)_("LE","c") (bold(z))
  :=bold(cal(A))_"LE" bold(sigma)-bold(epsilon)(bold(u)),
  quad
  bold(r)_("LE","e") (bold(z))
  :=div bold(sigma)+bold(f).
$
离散泛函为
$
  cal(J)_("LE",Q)(bold(z))
  :=abs(Omega)/Q sum_(ell=1)^Q (
    norm(bold(r)_("LE","c") (bold(z))(bold(x)_ell))_F^2
    +norm(bold(r)_("LE","e") (bold(z))(bold(x)_ell))_(ell^2)^2
  ).
$
对板弯曲，取 $bold(z)=(bold(M),w) in bold(H)(div div) times H^2(Omega)$，其中第二分量为标量挠度，相应残差为
$
  bold(r)_("KL","c") (bold(z))
  :=bold(cal(A))_"KL" bold(M)-bold(kappa)(w),
  quad
  r_("KL","e") (bold(z)):=div div bold(M)+f,
$
此时平衡残差是标量，离散泛函为
$
  cal(J)_("KL",Q)(bold(z))
  :=abs(Omega)/Q sum_(ell=1)^Q (
    norm(bold(r)_("KL","c") (bold(z))(bold(x)_ell))_F^2
    +abs(r_("KL","e") (bold(z))(bold(x)_ell))^2
  ).
$
两式分别是连续泛函 $cal(J)_"LE" (bold(z); bold(f))$ 与 $cal(J)_"KL" (bold(z); f)$ 的 Monte Carlo 估计。

训练泛函取 $bold(z) = tilde(bold(z))_(N,K)(bold(c))$。把各样本点处残差的全部分量乘以权 $sqrt(abs(Omega)\/Q)$ 并按 $bold(c)$ 展开，该二次多项式即写成设计矩阵 $bold(A)_(N,Q)$ 与右端向量 $bold(b)_(N,Q)$ 的最小二乘形式。

对于线弹性问题，两个残差在每个样本点共有 $n_s + d$ 个分量，故 $bold(A)_("LE",N,Q) in RR^((n_s+d)Q times m_N), bold(b)_("LE",N,Q) in RR^((n_s+d)Q)$。对于板弯曲问题，共有 $3 + 1$ 个分量，因此 $bold(A)_("KL",N,Q) in RR^(4Q times m_N), bold(b)_("KL",N,Q) in RR^(4Q)$。离散泛函因此可写成如下形式：
$
  cal(J)_("LE",Q)(tilde(bold(z))_(N,K)(bold(c))) & = norm(bold(A)_("LE",N,Q) bold(c)-bold(b)_("LE",N,Q))_(ell^2)^2, \
  cal(J)_("KL",Q)(tilde(bold(z))_(N,K)(bold(c))) & = norm(bold(A)_("KL",N,Q) bold(c)-bold(b)_("KL",N,Q))_(ell^2)^2,
                                                   quad
$
<eq:train-objective>

设计矩阵的每一列对应一个经数值积分构造的投影特征，由其齐次残差分量在样本点处的加权取值组成，右端向量由载荷在样本点处的取值组成，本构残差对应的右端行为零。以下不区分模型时省略模型下标，写作 $bold(A)_(N,Q)$ 与 $bold(b)_(N,Q)$。

=== 系数约束与训练问题

上一章的本质边界投影及本节的离散泛函均采用 Monte Carlo 求积，其数值实现都引入了积分误差。误差分析须控制由此产生的求积误差，且该控制须对训练问题可能输出的一切系数一致成立。以下说明这样的一致控制在整个 $RR^(m_N)$ 上不可能成立，训练问题因而必须把系数限制在有界集内。

对于离散泛函，@eq:train-objective 是 $bold(c)$ 上的二次多项式，将同一个场代入连续泛函，$cal(J)(tilde(bold(z))_(N,K)(bold(c)); bold(f))$ 也是 $bold(c)$ 上的二次多项式。两者之差
$
  cal(J)(tilde(bold(z))_(N,K)(bold(c)); bold(f))
  - cal(J)_Q(tilde(bold(z))_(N,K)(bold(c)))
$
仍是 $bold(c)$ 的二次多项式。故该差有界当且仅当其二次与一次部分为零，即样本均值对上述全部乘积都与精确积分相等，这对有限个随机样本一般不成立。因此连续泛函与离散泛函的一致偏差只能在有界系数集上取。

对 Ritz 投影也有同样的结论。以 $hat(v)_N (bold(c))$ 记位移或挠度字典场，则
$
  (tilde(Pi)_(D,K,Q_"R")^(m)-Pi_(D,K)^(m)) hat(v)_N (bold(c))
$
是关于 $bold(c)$ 的线性映射。只要该映射不恒为零，它在整个 $RR^(m_N)$ 上无界，因此投影求积误差的一致上确界也只能在有界系数集上取。

据此，训练问题的可行集取为欧氏球。对给定预算 $B>0$，定义系数球
$
  cal(C)_(N,B)
  := {bold(c) in RR^(m_N):sqrt(m_N) norm(bold(c))_(ell^2)<=B}
  = {bold(c) in RR^(m_N):norm(bold(c))_(ell^2)<=B\/sqrt(m_N)}.
$
<eq:coeff-ball>
训练问题即在该球上极小化 @eq:train-objective：
$
  min_(bold(c) in cal(C)_(N,B))
  norm(bold(A)_(N,Q) bold(c)-bold(b)_(N,Q))_(ell^2)^2.
$
<eq:train-problem>
这是欧氏球上的凸最小二乘问题。可行集缩小为 $cal(C)_(N,B)$ 后，训练问题的解至多与球内元素一样接近精确解。预算 $B$ 因此不能任意取小，它必须大到使球内仍含有以字典逼近阶接近精确解的系数。

== 系数球内的字典逼近

本节估计总误差的第一项：系数球内的原始字典场逼近精确解所能达到的阶，并定出达到该阶所需的充分预算。逼近误差以线弹性的 $bold(H)(div) times H^1$ 与板弯曲的 $bold(H)(div div) times H^2$ 乘积图范数度量，其阶由 @cor:quasi-uniform-rate 的逐分量估计按固定基组装得到。实现该阶的系数记为比较系数 $bold(c)^"cmp"$，总误差证明以它与求解器输出的系数作比较。

记 $m$ 为相应图范数所含的最高导数阶：线弹性 $m = 1$，板弯曲 $m = 2$。对精确解的每个物理变量 $chi$，分别假设其各分量属于 $H^(s_chi)(Omega)$，其中 $s_chi in [m, s_"cap" (d)]$。解更光滑时取 $s_chi = s_"cap" (d)$，逼近阶在该处饱和。整体逼近阶由最不光滑的变量决定，记
$
  beta_"LE"
  := min_(chi in {bold(sigma),bold(u)}) (s_chi-1)/d,
  quad
  beta_"KL"
  := min_(chi in {bold(M),w}) (s_chi-2)/2,
$
即 $(s_chi - m)\/d$ 在两个模型下的最小值。

以下命题在上述正则性假设下构造两类模型的比较系数，并给出使其落入系数球的充分预算。

#proposition(title: [球内字典逼近与系数控制])[
  设各物理变量的标量字典均满足 @thm:band-dictionary-rate 的假设，且 $k >= m$。在上述正则性假设下，对每个 $N$，两类模型分别存在仅由精确解与冻结字典决定的系数 $bold(c)^"cmp" in RR^(m_N)$，满足以下逼近误差界与系数界。

  线弹性中，设精确解 $(bold(sigma)_star, bold(u)_star)$ 的各分量分别属于 $H^(s_(bold(sigma)))(Omega)$ 与 $H^(s_(bold(u)))(Omega)$，$s_chi in [1, s_"cap" (d)]$。则存在 $bold(c)^"cmp"$，使
  $
    norm(bold(sigma)_star - hat(bold(sigma))_N (bold(c)^"cmp"))_(bold(H)(div))^2
    + norm(bold(u)_star - hat(bold(u))_N (bold(c)^"cmp"))_(H^1(Omega))^2
    lt.tilde N^(-2 beta_"LE"),
  $
  且
  $
    sqrt(m_N) norm(bold(c)^"cmp")_(ell^2)
    lt.tilde N^((s_"cap" (d) - min(s_(bold(sigma)), s_(bold(u))))\/d).
  $

  板弯曲中，设精确解 $(bold(M)_star, w_star)$ 的各分量分别属于 $H^(s_(bold(M)))(Omega)$ 与 $H^(s_w)(Omega)$，$s_chi in [2, s_"cap" (2)]$。则存在 $bold(c)^"cmp"$，使
  $
    norm(bold(M)_star - hat(bold(M))_N (bold(c)^"cmp"))_(bold(H)(div div))^2
    + norm(w_star - hat(w)_N (bold(c)^"cmp"))_(H^2(Omega))^2
    lt.tilde N^(-2 beta_"KL"),
  $
  且
  $
    sqrt(m_N) norm(bold(c)^"cmp")_(ell^2)
    lt.tilde N^((s_"cap" (2) - min(s_(bold(M)), s_w))\/2).
  $
  两类模型的比较系数分别选取。隐含常数依赖于固定字典参数、$Omega$、$d$ 与精确解各分量的 Sobolev 范数，与 $N$ 无关。

  记 $bar(s):=min_chi s_chi$。若预算满足
  $
    B gt.tilde N^((s_"cap" (d)-bar(s))\/d),
  $
  <eq:budget>
  其中隐含常数取得足够大，则上述系数属于 $cal(C)_(N,B)$，相应原始字典场分别以 $beta_"LE"$ 与 $beta_"KL"$ 阶逼近精确解。
]<prop:comparison-field>

#proof[
  在固定基 ${bold(T)_alpha}$ 与 ${bold(e)_i}$ 下把每个物理变量逐分量分解。对变量 $chi$ 的每个标量分量 $v in H^(s_chi)(Omega)$，在字典 ${xi_(chi,j)}_(j=1)^N$ 上应用 @cor:quasi-uniform-rate，得到分量近似 $hat(v)_N = sum_(j=1)^N c_j xi_(chi,j)$，满足
  $
    norm(v - hat(v)_N)_(H^m (Omega)) lt.tilde N^(-(s_chi - m)\/d) norm(v)_(H^(s_chi)(Omega)),
    quad
    (sum_(j=1)^N c_j^2)^(1\/2) lt.tilde N^((s_"cap" (d) - s_chi)\/d - 1\/2) norm(v)_(H^(s_chi)(Omega)).
  $
  各分量近似按基组装即得 $hat(bold(z))_N (bold(c)^"cmp")$。图范数受逐分量 $H^m$ 范数控制：由 $norm(div bold(tau))_(L^2(Omega)) lt.tilde norm(bold(tau))_(H^1(Omega))$ 得 $norm(bold(tau))_(bold(H)(div)) lt.tilde norm(bold(tau))_(H^1(Omega))$，由 $norm(div div bold(tau))_(L^2(Omega)) lt.tilde norm(bold(tau))_(H^2(Omega))$ 得 $norm(bold(tau))_(bold(H)(div div)) lt.tilde norm(bold(tau))_(H^2(Omega))$。各分量误差中最慢的阶为 $N^(-min_chi (s_chi - m)\/d) = N^(-beta)$，即得误差界。系数向量由分量数固定的有限个块拼成，$norm(bold(c)^"cmp")_(ell^2)$ 不超过各块 $ell^2$ 范数之和，其中最大的指数为 $(s_"cap" (d) - min_chi s_chi)\/d - 1\/2$，再由 $m_N tilde.eq N$ 吸收因子 $N^(-1\/2)$ 即得系数界。

  在 @eq:budget 中将隐含常数取为不小于上述系数界的常数，即有 $sqrt(m_N) norm(bold(c)^"cmp")_(ell^2) <= B$。逐分量逼近只依赖于精确解与冻结字典，故所选系数与投影求积规则及训练样本无关。证毕。
]

充分预算条件 @eq:budget 中的隐含常数依赖于各物理分量的 Sobolev 范数及固定的字典参数，在最不光滑分量达到饱和正则性时，$B$ 可取与 $N$ 无关的常数。球定义中的因子 $sqrt(m_N)$ 正好吸收了标量系数估计中的 $N^(-1\/2)$，该缩放与 @LiuMaoXu2025 的线性化网络类系数约束一致。

在解足够光滑、逼近阶在饱和处取到的情形下，将 $s_chi = s_"cap" (d)$ 代入 $beta$ 即得本文反复使用的显式收敛率
$
  beta(k, d, m) = (s_"cap" (d) - m)/d = (d + 2k + 1 - 2m)/(2 d).
$
<eq:beta-rate>
该式给出两个理论基准：
1. $k$ 每增加一个单位，指数 $beta$ 提高 $1\/d$。
2. 误差范数每多一阶导数，指数降低 $1\/d$。

== 投影离散与求积误差

本节估计总误差中由本质边界投影的空间截断与数值积分引入的两类误差。本质边界投影先因检验空间截断产生 Ritz 逼近误差，再因 $H^m$ 内积的 Monte Carlo 离散产生求积误差。记投影所用协调空间 $V_K^(m) subset H_0^m (Omega)$ 的维数为 $K$，两类误差界对 $K$ 的依赖分别由协调空间族的逼近估计与逆估计决定。

=== 容许协调空间族

Ritz 投影 $Pi_(D,K)^(m)$ 对任意有限维协调空间 $V_K^(m) subset H_0^m (Omega)$ 都有定义，但空间截断误差的衰减和后续求积方程的谱稳定性都取决于空间族 ${V_K^(m)}_K$ 随 $K$ 加密的方式。为分别提供所需的逼近估计与逆估计，本文考虑下述的容许协调空间族。

#definition(title: [容许协调空间族])[
  设 $m in {1,2}$。称有限维空间族 ${V_K^(m)}_K$ 为容许协调空间族，若每个 $V_K^(m) subset H_0^m (Omega)$，$dim V_K^(m)=K$，存在 $r_m>m$ 及与 $K$ 无关的常数 $C_"apx"$、$C_"inv"$，使下述两条对族中每个 $K$ 成立。

  + 逼近率：对任意 $s>=m$ 与 $v in H^s (Omega) inter H_0^m (Omega)$，
    $
      cal(E)_(m,K)(v)
      <= C_"apx" K^(-(min(s, r_m)-m)\/d)
      norm(v)_(H^s (Omega)),
    $
    <eq:aux-approx-rate>
    其中 $C_"apx"$ 可依赖于 $s$，与 $v$ 无关。
  + 逆估计：对任意 $z_K in V_K^(m)$，
    $
      max_(abs(bold(alpha)) <= m)
      norm(partial^bold(alpha) z_K)_(L^oo (Omega))
      <= C_"inv" K^(1\/2) norm(z_K)_(H^m (Omega)).
    $
    <eq:aux-inverse>

  称 $r_m$ 为该族的逼近饱和指数。
]<def:aux-admissible>

下述两例在单位盒上给出这样的空间族，二者共用同一组张量积 B 样条空间。

#definition(title: [开放均匀 B 样条基])[
  设 $Omega=(0,1)^d$、$h^(-1) in NN$ 且 $p>=1$，记
  $
    n_h:=h^(-1)+p,
    quad
    (upsilon_ell)_(ell=0)^(n_h+p)
    :=(underbrace(0\, dots.c\, 0, p+1),h,2h,dots,1-h,
      underbrace(1\, dots.c\, 1, p+1)).
  $
  该节点向量的端点各重复 $p+1$ 次，内部节点单重。为避免与系数预算 $B$ 混淆，以直立体 $upright(B)_(i,q)$ 记 B 样条。对 $0<=i<=n_h+p-1$ 定义
  $
    upright(B)_(i,0)(x)
    := cases(
      1 & quad upsilon_i<=x<upsilon_(i+1),
      0 & quad "otherwise",
    )
  $
  并对 $1<=q<=p$、$0<=i<=n_h+p-q-1$ 作 Cox--de Boor 递推
  $
    upright(B)_(i,q)(x)
    := (x-upsilon_i)/(upsilon_(i+q)-upsilon_i)
    upright(B)_(i,q-1)(x)
    + (upsilon_(i+q+1)-x)/(upsilon_(i+q+1)-upsilon_(i+1))
    upright(B)_(i+1,q-1)(x).
  $
  分母为零的项约定为零，最终得到的 $p$ 次基函数在 $x=1$ 处取左极限值。相应的 $d$ 维张量积空间为
  $
    bold(S)_(p,h)^d
    :=(span {upright(B)_(i,p):0<=i<=n_h-1})^(times.o d).
  $
]<def:bspline-basis>

开放端点性质给出
$
             upright(B)_(i,p)|_(partial(0, 1))=0 & arrow.l.r.double 1<=i<=n_h-2, \
  upright(B)_(i,p)|_(partial(0, 1))
  =partial_x upright(B)_(i,p)|_(partial(0, 1))=0 & arrow.l.r.double 2<=i<=n_h-3.
$
因而，对 $m in {1,2}$、$p>=m$ 且 $n_h>=2m+1$，
$
        bold(S)_(p,h)^d inter H_0^m (Omega) & =(span {upright(B)_(i,p):m<=i<=n_h-1-m})^(times.o d), \
  dim (bold(S)_(p,h)^d inter H_0^m (Omega)) & =(n_h-2m)^d.
$
<eq:bspline-conforming-subspaces>

@fig:ls-bspline-basis 以 $d=1$、$p=2$、$h=1\/4$ 为例展示 @def:bspline-basis，此时 $n_h=6$。图中以 $upright(B)_i:=upright(B)_(i,2)$ 简记六个二次 B 样条，轴下方堆叠的短线标出节点重数。

#figure(
  code-image(class: "center", theme => canvas({
    import draw: *

    // 具体例子: p = 2, h = 1/4, 节点 (0,0,0, 1/4, 1/2, 3/4, 1,1,1)
    let W = 7.2
    let H = 2.2

    let col-keep = theme.main-color
    let col-h1 = rgb("#b3261e")
    let col-h2 = rgb("#1a5fb4")
    let col-grid = rgb("#9aa0a6")

    // 六个二次 B 样条
    let B0(x) = if x <= 0.25 { calc.pow(1 - 4 * x, 2) } else { 0 }
    let B1(x) = if x <= 0.25 { 8 * x - 24 * x * x } else if x <= 0.5 { 2 - 8 * x + 8 * x * x } else { 0 }
    let B2(x) = if x <= 0.25 { 8 * x * x } else if x <= 0.5 { -16 * x * x + 12 * x - 1.5 } else if x <= 0.75 {
      8 * x * x - 12 * x + 4.5
    } else { 0 }
    let B3(x) = B2(1 - x)
    let B4(x) = B1(1 - x)
    let B5(x) = if x >= 0.75 { calc.pow(4 * x - 3, 2) } else { 0 }
    let Bs = (B0, B1, B2, B3, B4, B5)
    let cols = (col-h1, col-h2, col-keep, col-keep, col-h2, col-h1)

    // 左：单变量基
    for k in (1, 2, 3) {
      line((W * k / 4, 0), (W * k / 4, H), stroke: (paint: col-grid, thickness: 0.4pt, dash: "dashed"))
    }
    line((0, 0), (W, 0), stroke: 0.7pt + col-keep)

    let n-samples = 48
    for (i, Bi) in Bs.enumerate() {
      let pts = ()
      for s in range(n-samples + 1) {
        let x = s / n-samples
        pts.push((W * x, H * Bi(x)))
      }
      line(..pts, stroke: 1.1pt + cols.at(i))
    }

    // 曲线峰值旁标注直立体 B_i
    let label-pos = (
      (0.065 * W, 0.88 * H),
      (0.125 * W + 0.18, 0.72 * H),
      (0.375 * W, 0.86 * H),
      (0.625 * W, 0.86 * H),
      (0.875 * W - 0.18, 0.72 * H),
      (0.935 * W, 0.88 * H),
    )
    for i in range(6) {
      content(
        label-pos.at(i),
        text(0.75em, fill: cols.at(i), eval("$upright(B)_" + str(i) + "$", mode: "markup")),
      )
    }

    // 节点重数：轴下方堆叠短横线
    for (xk, m) in ((0, 3), (0.25, 1), (0.5, 1), (0.75, 1), (1, 3)) {
      for r in range(m) {
        line(
          (W * xk - 0.09, -0.10 - r * 0.055),
          (W * xk + 0.09, -0.10 - r * 0.055),
          stroke: 1.2pt + col-grid,
        )
      }
    }
    let knot-labels = ((0, $0$), (0.25, $1\/4$), (0.5, $1\/2$), (0.75, $3\/4$), (1, $1$))
    for (xk, lab) in knot-labels {
      content((W * xk, -0.42), text(0.7em, fill: col-grid, lab))
    }

    // 图例
    let ly = -0.85
    line((0.2, ly), (0.55, ly), stroke: 1.1pt + col-h1)
    content((1.05, ly), text(0.7em)[迹非零])
    line((2.1, ly), (2.45, ly), stroke: 1.1pt + col-h2)
    content((3.5, ly), text(0.7em)[仅法向导数非零])
    line((4.8, ly), (5.15, ly), stroke: 1.1pt + col-keep)
    content((6.45, ly), text(0.7em)[迹与法向导数均为零])
  })),
  caption: [$d=1$ 开放均匀节点向量上的单变量 B 样条基（$p=2$，$h=1\/4$）],
)<fig:ls-bspline-basis>

#example(title: [单位盒上的 $H_0^1$ B 样条空间])[
  取 $p>=1$ 且 $n_h>=3$。按 @eq:bspline-conforming-subspaces 筛除迹非零的张量积基函数，得
  $
    V_h^(1)
    :=(span {upright(B)_(i,p):1<=i<=n_h-2})^(times.o d)
    =bold(S)_(p,h)^d inter H_0^1(Omega),
    quad K:=dim V_h^(1).
  $
  随 $h$ 减小而逐次加密网格，所得空间满足 @def:aux-admissible，构成容许协调空间族，其逼近饱和指数为 $r_1=p+1$。
]

#proof[
  固定 $p$。内部节点单重，结合 @eq:bspline-conforming-subspaces 得
  $
    bold(S)_(p,h)^d & subset C^(p-1)(overline(Omega)) inter H^p(Omega)
                      subset H^1(Omega), \
            V_h^(1) & =bold(S)_(p,h)^d inter H_0^1(Omega)
                      subset H_0^1(Omega), \
                  K & =(n_h-2)^d=(h^(-1)+p-2)^d
                      tilde.eq h^(-d).
  $
  保齐次边界的样条准插值估计与逐单元逆估计分别给出（参见 @Schumaker2007 与 @TakacsTakacs2016）
  $
    inf_(z_h in V_h^(1)) norm(v-z_h)_(H^1(Omega))
    &lt.tilde h^(min(s, p+1)-1) norm(v)_(H^s(Omega)),
    && s>=1,
    quad v in H^s(Omega) inter H_0^1(Omega), \
    max_(abs(bold(alpha))<=1)
    norm(partial^bold(alpha) z_h)_(L^oo(Omega))
    &lt.tilde h^(-d\/2) norm(z_h)_(H^1(Omega)),
    && z_h in V_h^(1).
  $
  代入 $h tilde.eq K^(-1\/d)$ 即得 @eq:aux-approx-rate、@eq:aux-inverse 与 $r_1=p+1$。证毕。
]

#example(title: [单位盒上的 $H_0^2$ B 样条空间])[
  取 $p>=2$ 且 $n_h>=5$。按 @eq:bspline-conforming-subspaces 筛除迹或法向导数非零的张量积基函数，得
  $
    V_h^(2)
    :=(span {upright(B)_(i,p):2<=i<=n_h-3})^(times.o d)
    =bold(S)_(p,h)^d inter H_0^2(Omega),
    quad K:=dim V_h^(2).
  $
  随 $h$ 减小而逐次加密网格，所得空间满足 @def:aux-admissible，构成容许协调空间族，其逼近饱和指数为 $r_2=p+1$。
]

#proof[
  固定 $p$。内部节点单重，结合 @eq:bspline-conforming-subspaces 得
  $
    bold(S)_(p,h)^d & subset C^(p-1)(overline(Omega)) inter H^p(Omega)
                      subset H^2(Omega), \
            V_h^(2) & =bold(S)_(p,h)^d inter H_0^2(Omega)
                      subset H_0^2(Omega), \
                  K & =(n_h-4)^d=(h^(-1)+p-4)^d
                      tilde.eq h^(-d).
  $
  保齐次边界的样条准插值估计与逐单元逆估计分别给出（参见 @Schumaker2007 与 @TakacsTakacs2016）
  $
    inf_(z_h in V_h^(2)) norm(v-z_h)_(H^2(Omega))
    &lt.tilde h^(min(s, p+1)-2) norm(v)_(H^s(Omega)),
    && s>=2,
    quad v in H^s(Omega) inter H_0^2(Omega), \
    max_(abs(bold(alpha))<=2)
    norm(partial^bold(alpha) z_h)_(L^oo(Omega))
    &lt.tilde h^(-d\/2) norm(z_h)_(H^2(Omega)),
    && z_h in V_h^(2).
  $
  代入 $h tilde.eq K^(-1\/d)$ 即得 @eq:aux-approx-rate、@eq:aux-inverse 与 $r_2=p+1$。证毕。
]

上述两例直接说明 @def:aux-admissible 所定义的空间族非空。

协调空间维数 $K$ 与字典规模 $N$ 可分别选取。考虑 $K tilde.eq N$ 的配置，对正则指数为 $s in [m,s_"cap" (d)]$ 的位移或挠度分量，@cor:quasi-uniform-rate 给出的字典逼近阶为 $N^(-(s-m)\/d)$，而 @eq:aux-approx-rate 给出
$
  cal(E)_(m,K)(v_star)
  lt.tilde N^(-(min(s, r_m)-m)\/d) norm(v_star)_(H^s (Omega)).
$
由传递估计 @eq:finite-projection-transfer，投影后的逼近误差受这两项之和控制。对给定解，为使 Ritz 截断误差不降低整体字典逼近阶 $N^(-beta)$，其中 $beta$ 为相应模型的 $beta_"LE"$ 或 $beta_"KL"$，只需令上述截断误差上界的衰减指数满足
$
  (min(s, r_m)-m)/d >= beta,
$
即 $min(s, r_m) >= m+d beta$。由 $beta$ 的定义，$m+d beta=min_chi s_chi$，而位移或挠度的正则指数 $s>=min_chi s_chi$，故上述指数条件等价于
$
  r_m >= m+d beta.
$
因此，保持给定解的整体字典逼近阶，只需协调空间的逼近饱和指数达到各物理变量中最低的正则指数。

若希望协调空间的选取不依赖于具体解的正则性，可利用 $m+d beta=min_chi s_chi<=s_"cap" (d)$，采用更强的统一充分条件
$
  r_m >= s_"cap" (d).
$
<eq:degree-match-abstract>
这一条件对允许的全部正则性配置都保证上述要求。当所有物理变量的正则指数均取 $s_"cap" (d)$ 时，两种条件一致。

对上述两个单位盒 B 样条族，$r_m=p+1$，故统一充分条件具体化为
$
  p+1 >= s_"cap" (d)=(d+2k+1)/2.
$
<eq:degree-match>

=== Ritz 投影的 Monte Carlo 积分误差

前述协调空间的逼近估计已经给出 Ritz 空间截断误差的 $K$ 阶。完全离散算法进一步以 $tilde(Pi)_(D,K,Q_"R")^(m)$ 代替精确积分的 $Pi_(D,K)^(m)$，因此还须利用该空间的逆估计控制投影方程的 Monte Carlo 积分误差。

对于线弹性问题，定义位移场在系数球上的一致 Ritz 投影积分误差为
$
  epsilon_("LE","Rquad")
  := sup_(bold(c) in cal(C)_(N,B))
  norm(
    (tilde(Pi)_(D,K,Q_"R")^(1) - Pi_(D,K)^(1))
    hat(bold(u))_N (bold(c))
  )_(H^1(Omega));
$
对于板弯曲问题，相应的挠度投影积分误差定义为
$
  epsilon_("KL","Rquad")
  := sup_(bold(c) in cal(C)_(N,B))
  norm(
    (tilde(Pi)_(D,K,Q_"R")^(2) - Pi_(D,K)^(2))
    hat(w)_N (bold(c))
  )_(H^2(Omega)).
$
下述 Ritz 投影估计对两种误差均成立，其中将相应误差简记为 $epsilon_"Rquad"$。

取 $V_K^(m)$ 的任意 $H^m$ 正交归一基
${phi_k}_(k=1)^K$，令求积 Gram 矩阵与其谱稳定事件为
$
  (bold(G)_(K,Q_"R"))_(i j)
  := (phi_i,phi_j)_(H^m,Q_"R"),
  quad
  cal(E)_("R",delta)
  := {norm(bold(G)_(K,Q_"R")-bold(I)_K)_2 <= delta}.
$
记 $EE_"R" [dot]$ 为对 Ritz 投影构造样本 ${bold(x)_("R",r)}_(r=1)^(Q_"R")$ 取的期望，$EE_"R" [dot | cal(E)_("R",delta)]$ 为条件于谱稳定事件的条件期望。
定义
$
  C_("R",K)^(m)
  := abs(Omega) op("ess sup")_(bold(x) in Omega)
  sup_(0 != z_K in V_K^(m))
  norm(cal(L)_m (bold(x))z_K)_(ell^2)^2 / norm(z_K)_(H^m (Omega))^2.
$
若以 $sqrt(abs(Omega)) cal(L)_m$ 为采样算子，上式即 @Adcock2025 定义 9.1 与式 (9.3) 中广义 Christoffel 函数的本质上确界。

#theorem(title: [Ritz 投影的 Monte Carlo 积分误差])[
  设 $0<delta<1$、$0<eta<1$。若
  $
    Q_"R"
    >= C_delta C_("R",K)^(m) log(2 K/eta),
  $
  其中 $C_delta>0$ 仅依赖于 $delta$，则谱稳定事件满足
  $
    bb(P)(cal(E)_("R",delta)) >= 1-eta,
  $
  此外，条件于谱稳定事件 $cal(E)_("R",delta)$，Ritz 投影积分误差的均方根满足
  $
    (
      EE_"R" [epsilon_"Rquad"^2 | cal(E)_("R",delta)]
    )^(1\/2)
    lt.tilde B
    sqrt(C_("R",K)^(m)/Q_"R").
  $
  隐含常数依赖于 $delta$、$eta$、$m$、$Omega$ 与原始特征的一致 $H^m$ 上界，不依赖于 $lambda$、$N$、$K$、$Q_"R"$ 与 $B$。其中 $delta$ 与 $eta$ 作为固定的稳定性和置信度参数。
]<thm:rquad-rate>

#proof[
  以下只在 $cal(E)_("R",delta)$ 上使用 $tilde(Pi)_(D,K,Q_"R")^(m)$。在其补集上如何定义投影差不影响条件期望。

  第一步：Gram 谱稳定性。标量点值采样下的分析见 @CohenDavenportLeviatan2013 定理 1、式 (1.2)。本文采用 @Adcock2025 第 9.1 节的向量值采样框架，其第 9.2 节例 (ii) 给出了函数值与梯度采样的例子。对本文的至 $m$ 阶导数采样，定义矩阵
  $
    (bold(A)(bold(x)))_(i j)
    := sqrt(abs(Omega)) partial^(bold(alpha)_i) phi_j (bold(x)),
    quad 1 <= i <= n_(bold(alpha)), quad 1 <= j <= K,
  $
  其行按 @eq:jet-operator 固定的多重指标次序排列。于是对任意 $bold(a) in RR^K$ 与 $z_K = sum_(j=1)^K a_j phi_j$，
  $
    bold(A)(bold(x)) bold(a) = sqrt(abs(Omega)) cal(L)_m (bold(x)) z_K.
  $
  <eq:sampling-matrix>
  由均匀采样与基的 $H^m$ 正交归一性，
  $
    bold(G)_(K,Q_"R")
    = 1/Q_"R" sum_(r=1)^(Q_"R")
    bold(A)(bold(x)_("R",r))^T bold(A)(bold(x)_("R",r)),
    quad
    EE_"R" [bold(A)(bold(x)_("R",1))^T bold(A)(bold(x)_("R",1))]
    = bold(I)_K.
  $
  由同一正交归一性，$norm(z_K)_(H^m)=norm(bold(a))_(ell^2)$。代入 @eq:sampling-matrix 并用 $C_("R",K)^(m)$ 的定义，几乎处处有
  $
    norm(bold(A)(bold(x))^T bold(A)(bold(x)))_2
    = sup_(bold(a) != bold(0))
    norm(bold(A)(bold(x))bold(a))_(ell^2)^2/norm(bold(a))_(ell^2)^2
    = abs(Omega) sup_(0 != z_K in V_K^(m))
    norm(cal(L)_m (bold(x))z_K)_(ell^2)^2/norm(z_K)_(H^m (Omega))^2
    <= C_("R",K)^(m).
  $
  各样本矩阵相互独立且半正定，故可应用 @Adcock2025 附录定理 A.1 的矩阵 Chernoff 界，分别估计最大、最小特征值并取并集，得到
  $
    bb(P)(cal(E)_("R",delta)^c)
    <= 2 K exp(-c_delta Q_"R"/C_("R",K)^(m)), \
    c_delta := min {
      delta+(1-delta)log(1-delta),
      (1+delta)log(1+delta)-delta
    } > 0.
  $
  取 $C_delta=c_delta^(-1)$，定理中的样本条件即保证 $bb(P)(cal(E)_("R",delta)) >= 1-eta$。

  第二步：固定场的投影差二阶矩。沿用 @CohenDavenportLeviatan2013 第 2 节定理 2 证明中的正交残差论证（第 826--827 页，关键方差计算见第 827 页），以下将其用于本文的 $H^m$ 向量值采样。对固定原始标量场 $hat(v)$，记
  $r:=(I-Pi_(D,K)^(m))hat(v)$。连续正交性给出
  $(r,phi_k)_(H^m)=0$。将 $tilde(Pi)_(D,K,Q_"R")^(m) hat(v)$ 与 $Pi_(D,K)^(m) hat(v)$ 在基 ${phi_k}_(k=1)^K$ 下的系数向量分别记为 $bold(a)^"quad"$ 与 $bold(a)^"ex"$，则
  $
    bold(a)^"quad"-bold(a)^"ex"
    = bold(G)_(K,Q_"R")^(-1) bold(d)_(Q_"R"),
    quad
    (bold(d)_(Q_"R"))_k
    := (r,phi_k)_(H^m,Q_"R"),
  $
  且 $EE_"R" bold(d)_(Q_"R")=bold(0)$。由样本独立性与连续正交性，交叉项的期望为零。利用第一步定义的 $bold(A)(bold(x))$ 及其算子范数界，得到
  $
    EE_"R" norm(bold(d)_(Q_"R"))_(ell^2)^2 & = 1/Q_"R" integral_Omega
                                             norm(bold(A)(bold(x))^T cal(L)_m (bold(x))r)_(ell^2)^2 dif x \
                                           & <= C_("R",K)^(m)/Q_"R" norm(r)_(H^m (Omega))^2.
  $

  第三步：系数球上的一致估计与条件期望。记需投影场的分量数为 $n_v$：线弹性中 $n_v=d$，板弯曲中 $n_v=1$。令 $xi_(v,j)$ 为相应原始标量特征。用系数球的半径 $B/sqrt(m_N)$、矩阵算子范数不超过 Frobenius 范数，并在 $cal(E)_("R",delta)$ 上使用
  $norm(bold(G)_(K,Q_"R")^(-1))_2 <= (1-delta)^(-1)$，得到
  $
    EE_"R" [epsilon_"Rquad"^2 1_(cal(E)_("R",delta))]
    <= B^2 C_("R",K)^(m) /(
    m_N (1-delta)^2 Q_"R"
    )
    sum_(i=1)^(n_v) sum_(j=1)^N
    norm((I-Pi_(D,K)^(m))xi_(v,j))_(H^m (Omega))^2.
  $
  由于正交投影非扩张，右端每个残差不超过
  $norm(xi_(v,j))_(H^m)$。$rho_k$ 的参数域有界，故这些特征的 $H^m$ 范数关于 $j$ 与 $N$ 一致有界。再用 $n_v N<=m_N$，并以
  $bb(P)(cal(E)_("R",delta)) >= 1-eta$ 除以事件概率，即得条件二阶矩界。

  证毕。
]

对本文采用的容许协调空间族，@thm:rquad-rate 中的 $C_("R",K)^(m)$ 及相应结论可进一步具体化如下。

#corollary(title: [容许协调空间上的 Ritz 投影积分误差])[
  设 ${V_K^(m)}_K$ 为满足 @def:aux-admissible 的容许协调空间族，则
  $
    K/n_(bold(alpha))
    <= C_("R",K)^(m)
    <= n_(bold(alpha)) abs(Omega) C_"inv"^2 K.
  $
  <eq:leverage-bound>
  因此，在 @thm:rquad-rate 的其余假设下，只要
  $
    Q_"R"
    >= C_delta n_(bold(alpha)) abs(Omega) C_"inv"^2
    K log(2K/eta),
  $
  则谱稳定事件满足 $bb(P)(cal(E)_("R",delta)) >= 1-eta$。此外，条件于谱稳定事件 $cal(E)_("R",delta)$，Ritz 投影积分误差的均方根满足
  $
    (EE_"R" [epsilon_"Rquad"^2 | cal(E)_("R",delta)])^(1\/2)
    lt.tilde B sqrt(K/Q_"R").
  $
  换言之，固定 $delta$ 与 $eta$ 后，充分的样本规模为
  $Q_"R" gt.tilde K log(2K\/eta)$，而条件均方根误差为
  $O(B sqrt(K/Q_"R"))$。
]<cor:rquad-admissible>

#proof[
  由逆估计 @eq:aux-inverse，对任意 $z_K in V_K^(m)$ 与几乎处处的 $bold(x) in Omega$ 有
  $
    abs(Omega) norm(cal(L)_m (bold(x))z_K)_(ell^2)^2
    <= n_(bold(alpha)) abs(Omega) C_"inv"^2 K
    norm(z_K)_(H^m (Omega))^2,
  $
  这给出 @eq:leverage-bound 的上界。

  再取 $V_K^(m)$ 的任意 $H^m$ 正交归一基 ${phi_k}_(k=1)^K$。矩阵
  $[cal(L)_m (bold(x)) phi_1, dots, cal(L)_m (bold(x)) phi_K]$
  的秩不超过 $n_(bold(alpha))$，故其谱范数平方不小于 Frobenius 范数平方的 $1\/n_(bold(alpha))$。后者在 $Omega$ 上的积分为
  $sum_(k=1)^K norm(phi_k)_(H^m (Omega))^2=K$，从而
  $C_("R",K)^(m)>=K\/n_(bold(alpha))$。最后将上界代入 @thm:rquad-rate，即得样本条件、谱稳定概率与条件误差估计。证毕。
]

== Monte Carlo 训练泛函的一致偏差 <sec:train-mc>

本节研究训练泛函采用 Monte Carlo 积分时，在系数球 $cal(C)_(N,B)$ 上的一致偏差。先固定 Ritz 投影构造样本，从而固定系数到投影场 $tilde(bold(z))_(N,K)(bold(c))$ 的线性映射。随后利用与投影构造样本独立的均匀训练样本，比较同一投影场上的连续泛函与经验泛函。投影构造对这一估计的影响通过投影特征及其导数的一致上界体现。

以下分析同时适用于线弹性与板弯曲，省略模型下标及固定的载荷参数，将相应的连续泛函与经验泛函分别记为 $cal(J)$ 与 $cal(J)_Q$。记 $cal(G)_"proj"$ 为 Ritz 投影构造样本 ${bold(x)_("R",r)}_(r=1)^(Q_"R")$ 生成的 $sigma$-代数。训练样本 ${bold(x)_ell}_(ell=1)^Q$ 与 $cal(G)_"proj"$ 独立。下文对可积或非负随机变量使用条件期望记号
$
  EE_"train" [dot] := EE [dot | cal(G)_"proj"].
$
<eq:projection-conditioning>
下述条件估计在投影映射良定义的样本实现上陈述。条件于 $cal(G)_"proj"$ 后，投影特征构成确定的有限族，$EE_"train"$ 仅对训练样本取期望。

映射 $bold(c) |-> tilde(bold(z))_(N,K)(bold(c))$ 是线性的，故零系数向量 $bold(0) in RR^(m_N)$ 对应的投影场 $tilde(bold(z))_(N,K)(bold(0))$ 为零场。以此为基准，定义关于系数的中心化泛函
$
    cal(J)^circle.stroked.tiny (bold(c)) & := cal(J)(tilde(bold(z))_(N,K)(bold(c)))
                                           - cal(J)(tilde(bold(z))_(N,K)(bold(0))), \
  cal(J)_Q^circle.stroked.tiny (bold(c)) & := cal(J)_Q (tilde(bold(z))_(N,K)(bold(c)))
                                           - cal(J)_Q (tilde(bold(z))_(N,K)(bold(0))).
$
<eq:centered-risk>
被扣除的两项分别为载荷模平方的积分及其 Monte Carlo 估计，均与 $bold(c)$ 无关。这一平移不改变各泛函的极小点，也不改变离散目标值的次优性。以下估计中心化后的一致偏差
$
  EE_"train" sup_(bold(c) in cal(C)_(N,B))
  abs(cal(J)^circle.stroked.tiny (bold(c))-cal(J)_Q^circle.stroked.tiny (bold(c))).
$

采用 @SiegelHongJinHaoXu2023 第 7.1 节的 Rademacher 方法，对称化将上述一致偏差归结为损失函数类的复杂度，二次损失收缩再将其归结为残差类的复杂度与一致上界。

=== 投影特征与导数一致界

用 Rademacher 复杂度度量投影特征所生成函数类的采样偏差。

#definition(title: [Rademacher 复杂度])[
  设 $cal(F)$ 为 $Omega$ 上的非空实值可测函数类，${bold(x)_ell}_(ell=1)^Q$ 为 $Omega$ 上的独立均匀样本。取相互独立的 Rademacher 随机变量 ${epsilon_ell}_(ell=1)^Q$，即每个变量均满足
  $
    bb(P)(epsilon_ell = 1) = bb(P)(epsilon_ell = -1) = 1/2,
    quad ell = 1, dots, Q,
  $
  并令整组随机变量独立于全部训练样本和投影构造样本。定义 $cal(F)$ 关于均匀分布的 Rademacher 复杂度为
  $
    frak(R)_Q (cal(F))
    := EE_(bold(x)_ell) EE_(epsilon_ell)
    sup_(h in cal(F)) 1/Q sum_(ell=1)^Q epsilon_ell h(bold(x)_ell).
  $
]<def:rademacher-complexity>

#lemma(title: [双侧对称化])[
  设 $cal(F)$ 为 $Omega$ 上包含零函数的实值可积函数类，${bold(x)_ell}_(ell=1)^Q$ 为 $Omega$ 上的独立均匀样本，并假设所涉上确界可测。则
  $
    EE sup_(h in cal(F))
    abs(1/Q sum_(ell=1)^Q h(bold(x)_ell) - 1/abs(Omega) integral_Omega h dif x)
    <= 4 frak(R)_Q (cal(F)).
  $
]<lem:symmetrization>

#proof[
  记 $P h := 1/abs(Omega) integral_Omega h dif x$ 与 $P_Q h := 1/Q sum_(ell=1)^Q h(bold(x)_ell)$，并用一组与训练样本独立同分布的样本 ${bold(x)'_ell}_(ell=1)^Q$ 定义样本均值 $P'_Q h$。由 Jensen 不等式与两组样本的交换对称性，
  $
    EE sup_(h in cal(F)) (P_Q h-P h) & <= EE sup_(h in cal(F)) (P_Q h-P'_Q h) \
                                     & = EE_(bold(x)_ell, bold(x)'_ell, epsilon_ell)
                                       sup_(h in cal(F)) 1/Q sum_(ell=1)^Q
                                       epsilon_ell (h(bold(x)_ell)-h(bold(x)'_ell)) \
                                     & <= 2 frak(R)_Q (cal(F)).
  $
  最后一步分别估计两组样本的上确界，并利用随机符号的对称性。同理，$EE sup_(h in cal(F)) (P h-P_Q h) <= 2 frak(R)_Q (cal(F))$。由于 $0 in cal(F)$，两种单侧上确界均非负，故
  $
    sup_(h in cal(F)) abs(P_Q h-P h)
    <= sup_(h in cal(F)) (P_Q h-P h)
    + sup_(h in cal(F)) (P h-P_Q h).
  $
  对训练样本取期望即得结论。证毕。
]

由对称化估计，一致训练偏差可归结为损失类的复杂度。损失由投影场的残差构成，因此先明确两类模型的投影特征。分块由各模型的投影作用范围给出：线弹性的 $Pi_"tr"$ 只作用于应力块，板弯曲的弯矩块保持原始字典。边界投影 $tilde(Pi)_(D,K,Q_"R")^(m)$ 在线弹性中逐分量作用于位移（$m=1$），在板弯曲中作用于挠度（$m=2$），故每个模型都只需投影 $N$ 个标量特征。据此将投影基场的全部标量分量分成两族，线弹性为
$
  cal(B)_("LE",N,K) & := {(Pi_"tr" (xi_(bold(sigma),j) bold(T)_alpha))_(beta gamma):
                        1 <= j <= N, 1 <= alpha <= n_s, 1 <= beta, gamma <= d} \
                    & quad union {tilde(Pi)_(D,K,Q_"R")^(1) xi_(bold(u),j): 1 <= j <= N},
$
板弯曲相应为
$
  cal(B)_("KL",N,K) & := {(xi_(bold(M),j) bold(T)_alpha)_(beta gamma):
                        1 <= j <= N, 1 <= alpha <= 3, 1 <= beta, gamma <= 2} \
                    & quad union {tilde(Pi)_(D,K,Q_"R")^(2) xi_(w,j): 1 <= j <= N}.
$
张量块由脊字典及常数修正生成，边界投影后的位移（挠度）特征一般不再具有脊函数结构。两类特征均可按固定的有限线性特征族估计。为此需要特征的点态导数上界：$H^m$ 范数控制积分意义下的导数，而残差中的最高阶导数仅有 $L^2$ 控制尚不足以给出样本点上的一致界。下面定义这些导数共同满足的一致上界。

省略模型下标，将上述特征族统一记为 $cal(B)_(N,K)$。在本节的条件化约定下，投影为确定性线性算子，$cal(B)_(N,K)$ 为确定的有限集。取其至多 $m$ 阶的导数族
$
  cal(D)_(N,K)
  := {partial^bold(alpha) g:
    g in cal(B)_(N,K), bold(alpha) in NN_0^d, abs(bold(alpha)) <= m},
$
其中 $m$ 为相应图范数所含的最高导数阶（线弹性 $m=1$，板弯曲 $m=2$）。残差用到的是其中的特定组合：线弹性为 $div bold(tau)$ 与 $bold(epsilon)(bold(v))$ 中的一阶导数，板弯曲为 $div div bold(tau)$ 与 $nabla^2 v$ 中的二阶导数。按全部 $abs(bold(alpha)) <= m$ 取族把这些组合的分量一并纳入。将该族的一致 $L^oo$ 上界记为
$
  C_(Pi,K) := sup_(g in cal(D)_(N,K))
  norm(g)_(L^oo (Omega)).
$
该定义表示：每个投影特征及其至多 $m$ 阶导数的绝对值，在 $Omega$ 上几乎处处不超过同一个常数 $C_(Pi,K)$。这里“一致”是指该上界适用于族中的所有特征与导数。先以该上界陈述一般训练偏差估计，再利用本文投影构造的性质给出关于 $K$ 的显式界。

导数一致界同时控制各齐次残差分量的取值范围，因而可通过收缩估计，将残差类的复杂度转化为二次损失类的复杂度。所需估计如下，参见 @SiegelHongJinHaoXu2023 定理 5 的证明。

#lemma(title: [二次损失收缩])[
  设标量函数类 $cal(F)$ 满足 $sup_(g in cal(F)) norm(g)_(L^oo (Omega)) <= A$，$f in L^oo (Omega)$ 为固定函数，并令
  $
    cal(H):={g^2+2 f g:g in cal(F)}.
  $
  则
  $
    frak(R)_Q (cal(H))
    lt.tilde (A+norm(f)_(L^oo (Omega))) frak(R)_Q (cal(F)).
  $
]<lem:quadratic-structural>

对两类模型的各标量残差分量分别应用上述估计，再由 Rademacher 复杂度的次可加性求和，即可得到相应损失函数类的复杂度界。

=== 一致偏差界

#theorem(title: [Monte Carlo 训练泛函的一致偏差])[
  设系数取自 $cal(C)_(N,B)$，投影映射良定义且投影特征的导数一致界满足 $C_(Pi,K)<oo$，中心化泛函按 @eq:centered-risk 定义。训练样本为 $Omega$ 上与投影构造样本独立的 $Q$ 个独立均匀样本。下述条件期望 $EE_"train"$ 按 @eq:projection-conditioning 定义，泛函、系数球与导数一致界均取自相应模型。

  线弹性中，取 $m=1$，设 $k >= 2$、$bold(f) in L^oo (Omega; RR^d)$，则
  $
    EE_"train" sup_(bold(c) in cal(C)_(N,B))
    abs(cal(J)^circle.stroked.tiny (bold(c))-cal(J)_Q^circle.stroked.tiny (bold(c)))
    lt.tilde (
      B^2 C_(Pi,K)^2
      + norm(bold(f))_(L^oo (Omega)) B C_(Pi,K)
    ) Q^(-1\/2).
  $

  板弯曲中，取 $m=2$，设 $k >= 3$、$f in L^oo (Omega)$，则
  $
    EE_"train" sup_(bold(c) in cal(C)_(N,B))
    abs(cal(J)^circle.stroked.tiny (bold(c))-cal(J)_Q^circle.stroked.tiny (bold(c)))
    lt.tilde (
      B^2 C_(Pi,K)^2
      + norm(f)_(L^oo (Omega)) B C_(Pi,K)
    ) Q^(-1\/2).
  $
  两式在满足上述假设的投影样本实现上几乎处处成立。隐含常数仅依赖于 $Omega$、$d$、$mu$ 及板厚 $h$（板弯曲情形），不依赖于 $N$、$Q$、$K$、$B$、$lambda$ 与投影样本的具体实现。投影构造的影响由 $C_(Pi,K)$ 体现。
]<thm:train-generalization>

#proof[
  以下均条件于 $cal(G)_"proj"$，将投影特征视为固定函数。

  第一步：齐次残差类的复杂度。固定本构残差或扣除载荷后的平衡残差中的任一标量分量，在本证明中将其记为 $g_(bold(c))$。由于投影映射与残差算子均线性，该分量可写成
  $
    g_(bold(c)) (bold(x))
    = sum_(p=1)^(m_N) c_p a_p (bold(x))
    = bold(c) dot bold(a)_"res" (bold(x)),
    quad bold(a)_"res" := (a_p)_(p=1)^(m_N).
  $
  这里 $a_p$、$bold(a)_"res"$ 与下文的 $cal(F)_"res"$ 均为本证明中针对所选残差分量的局部记号。每个 $a_p$ 是 $cal(D)_(N,K)$ 中有限多个元素的线性组合。组合中的导数指标与分量个数仅依赖于 $d$，柔度算子的分量在线弹性中以 $1\/mu$ 为界，在板弯曲中以 $6\/(mu h^3)$ 为界，均关于 $lambda$ 一致。因此
  $
    norm(a_p)_(L^oo (Omega)) lt.tilde C_(Pi,K),
    quad
    norm(bold(a)_"res" (bold(x)))_(ell^2) lt.tilde sqrt(m_N) C_(Pi,K)
  $
  几乎处处成立。这些界对所选残差分量一致。令
  $cal(F)_"res" := {g_(bold(c)):bold(c) in cal(C)_(N,B)}$。
  由系数球半径为 $B/sqrt(m_N)$，该残差类的一致上界满足
  $
    sup_(g in cal(F)_"res") norm(g)_(L^oo (Omega)) lt.tilde B C_(Pi,K).
  $
  对固定训练样本，使用线性类的 Rademacher 估计（@ShalevShwartzBenDavid2014 第 26.2 节），并由 Cauchy--Schwarz 与 Rademacher 变量的独立性，
  $
    EE_(epsilon_ell) sup_(bold(c) in cal(C)_(N,B))
    1/Q sum_(ell=1)^Q epsilon_ell g_(bold(c)) (bold(x)_ell)
    &= B/(sqrt(m_N) Q) EE_(epsilon_ell)
    norm(sum_(ell=1)^Q epsilon_ell bold(a)_"res" (bold(x)_ell))_(ell^2) \
    &<= B/(sqrt(m_N) Q)
    (sum_(ell=1)^Q norm(bold(a)_"res" (bold(x)_ell))_(ell^2)^2)^(1\/2) \
    &lt.tilde B C_(Pi,K) Q^(-1\/2).
  $
  再对训练样本取期望，得 $frak(R)_Q (cal(F)_"res") lt.tilde B C_(Pi,K) Q^(-1\/2)$。系数球的缩放抵消了特征维数带来的 $sqrt(m_N)$ 因子，但该界仍可通过 $B$ 与 $C_(Pi,K)$ 依赖于 $N$。

  第二步：中心化损失类的复杂度。按 @eq:centered-risk 扣除载荷平方项后，中心化泛函的被积函数为齐次残差分量的平方和，加上平衡残差与载荷的交叉项。记这些被积函数组成的类为 $cal(H)_"loss"$。对各标量残差类应用 @lem:quadratic-structural，并利用 Rademacher 复杂度的次可加性求和，在线弹性中得到
  $
    frak(R)_Q (cal(H)_"loss") lt.tilde (
      B C_(Pi,K) + norm(bold(f))_(L^oo (Omega))
    ) B C_(Pi,K) Q^(-1\/2).
  $
  板弯曲中相应为
  $
    frak(R)_Q (cal(H)_"loss") lt.tilde (
      B C_(Pi,K) + norm(f)_(L^oo (Omega))
    ) B C_(Pi,K) Q^(-1\/2).
  $

  第三步：对称化。训练样本与投影构造样本独立，故条件于 $cal(G)_"proj"$ 后仍为独立均匀样本。零系数向量对应恒为零的中心化被积函数，故 $0 in cal(H)_"loss"$。对固定的损失类 $cal(H)_"loss"$ 应用 @lem:symmetrization，并计入区域测度，得到
  $
    EE_"train" sup_(bold(c) in cal(C)_(N,B))
    abs(cal(J)^circle.stroked.tiny (bold(c))-cal(J)_Q^circle.stroked.tiny (bold(c)))
    <= 4 abs(Omega) frak(R)_Q (cal(H)_"loss").
  $
  代入第二步的两条复杂度界即得结论。证毕。
]

=== 协调空间维数的显式依赖

@thm:train-generalization 以投影特征的导数一致界 $C_(Pi,K)$ 为输入。对容许协调空间族，该界在谱稳定事件上是 $K$ 的显式函数：逆估计 @eq:aux-inverse 把导数点值归结为 $H^m$ 范数，Gram 矩阵的谱稳定性则给出求积 Ritz 投影的一致 $H^m$ 稳定性。

#lemma(title: [投影特征的导数一致界])[
  设容许协调空间族满足 @def:aux-admissible，$k >= m$。则在 @thm:rquad-rate 的谱稳定事件 $cal(E)_("R",delta)$ 上，对投影构造样本的每个实现有
  $
    C_(Pi,K) lt.tilde K^(1\/2),
  $
  <eq:envelope-rate>
  隐含常数仅依赖于 $delta$、$m$、$d$、$Omega$、$C_"inv"$ 与原始特征至 $m$ 阶导数的一致 $L^oo$ 上界，与 $N$、$K$、$Q_"R"$、$B$ 及 $lambda$ 无关。
]<lem:envelope-rate>

#proof[
  $cal(B)_(N,K)$ 中未经 Ritz 投影的特征是原始特征与常数之差：板弯曲的弯矩特征就是原始特征，线弹性的应力特征被减去的常数不超过原始应力特征的一致 $L^oo$ 上界。原始特征逐分量属于 $W^(k,oo)(Omega)$ 且参数域有界，故这些特征至 $m$ 阶导数的 $L^oo$ 范数关于 $j$ 与 $N$ 一致有界。

  对 Ritz 投影特征 $g := tilde(Pi)_(D,K,Q_"R")^(m) xi$，其中 $xi$ 为原始位移或挠度标量特征，记 $g$ 在 $H^m$ 正交归一基 ${phi_k}_(k=1)^K$ 下的系数向量为 $bold(a)$，并记求积半范数 $norm(v)_(H^m,Q_"R") := (v,v)_(H^m,Q_"R")^(1\/2)$。求积投影方程以 $g$ 为检验函数给出 $(g,g)_(H^m,Q_"R") = (xi,g)_(H^m,Q_"R")$。在 $cal(E)_("R",delta)$ 上 $norm(bold(G)_(K,Q_"R")-bold(I)_K)_2 <= delta$，故 $bold(G)_(K,Q_"R")$ 的特征值落在 $[1-delta,1+delta]$ 内，从而
  $
    norm(g)_(H^m (Omega))^2
    = norm(bold(a))_(ell^2)^2
    <= (bold(a)^T bold(G)_(K,Q_"R") bold(a))/(1-delta)
    = ((xi,g)_(H^m,Q_"R"))/(1-delta)
    <= sqrt(1+delta)/(1-delta)
    norm(xi)_(H^m,Q_"R") norm(g)_(H^m (Omega)),
  $
  最后一步对半正定双线性形式 $(dot,dot)_(H^m,Q_"R")$ 用 Cauchy--Schwarz 不等式，并用 $norm(g)_(H^m,Q_"R")^2 = bold(a)^T bold(G)_(K,Q_"R") bold(a) <= (1+delta) norm(bold(a))_(ell^2)^2$。求积半范数由导数点值控制，
  $
    norm(xi)_(H^m,Q_"R")^2
    = abs(Omega)/Q_"R" sum_(r=1)^(Q_"R")
    norm(cal(L)_m (bold(x)_("R",r)) xi)_(ell^2)^2
    <= n_(bold(alpha)) abs(Omega)
    max_(abs(bold(alpha)) <= m) norm(partial^bold(alpha) xi)_(L^oo (Omega))^2,
  $
  右端关于 $j$、$N$ 与样本实现一致有界，故 $norm(g)_(H^m (Omega))$ 一致有界。再由逆估计 @eq:aux-inverse，
  $
    max_(abs(bold(alpha)) <= m) norm(partial^bold(alpha) g)_(L^oo (Omega))
    <= C_"inv" K^(1\/2) norm(g)_(H^m (Omega))
    lt.tilde K^(1\/2).
  $
  两族特征的界合并即得结论。证毕。
]

#corollary(title: [训练泛函一致偏差关于协调空间维数的估计])[
  在 @thm:train-generalization 的假设下，设容许协调空间族满足 @def:aux-admissible，则在 $cal(E)_("R",delta)$ 上，线弹性满足
  $
    EE_"train" sup_(bold(c) in cal(C)_(N,B))
    abs(cal(J)^circle.stroked.tiny (bold(c))-cal(J)_Q^circle.stroked.tiny (bold(c)))
    lt.tilde (B^2 K+norm(bold(f))_(L^oo (Omega)) B sqrt(K)) Q^(-1\/2),
  $
  板弯曲满足
  $
    EE_"train" sup_(bold(c) in cal(C)_(N,B))
    abs(cal(J)^circle.stroked.tiny (bold(c))-cal(J)_Q^circle.stroked.tiny (bold(c)))
    lt.tilde (B^2 K+norm(f)_(L^oo (Omega)) B sqrt(K)) Q^(-1\/2).
  $
  隐含常数还依赖于 @lem:envelope-rate 的常数，对该事件内的投影样本实现一致。
]<cor:train-generalization-k>

#proof[
  将 @lem:envelope-rate 代入 @thm:train-generalization 即得结论。证毕。
]

上述 $Q^(-1\/2)$ 刻画训练泛函的 Monte Carlo 一致偏差，$K$ 的出现来自投影特征的导数上界。由于该上界在谱稳定事件上对投影样本一致，可在总误差分析中进一步对投影样本取期望。下一节将训练偏差代入离散目标的极小性比较，再结合残差稳定性与投影误差估计。在图范数均方根误差中，训练积分项相应为 $Q^(-1\/4)$。

== 总误差估计

令 $bold(z)_star$ 表示连续精确解，并记计算解为 $bold(z)_(N,Q,K) := tilde(bold(z))_(N,K)(bold(c)^"out")$，其中 $bold(c)^"out" in cal(C)_(N,B)$ 为求解所得的系数向量。下标 $N$、$Q$ 与 $K$ 分别表示字典特征数、训练点数与 Ritz 协调空间维数。计算解对训练样本的依赖通过 $bold(c)^"out"$ 体现，对系数预算及投影求积规则的依赖从略。以 $epsilon_("LE","opt")$ 与 $epsilon_("KL","opt")$ 分别表示两类模型在离散目标值意义下的代数求解次优性，不区分模型时记为 $epsilon_"opt"$，即对一切 $bold(c) in cal(C)_(N,B)$，
$
  cal(J)_Q (tilde(bold(z))_(N,K)(bold(c)^"out"))
  <= cal(J)_Q (tilde(bold(z))_(N,K)(bold(c))) + epsilon_"opt".
$
<eq:suboptimality>

前面各节的估计都不涉及 $bold(c)^"out"$，计算解与它们之间的唯一联系是 @eq:suboptimality。本节先把该极小性转化为一条与模型无关的拟最优性引理，再代入各阶得到两类模型的总误差。

=== 拟最优性

以下同时处理两类模型，省略模型下标。记 $bold(X)$ 为相应模型的约束空间 $bold(X)_"LE"$ 或 $bold(X)_"KL"$，$bold(z)_star in bold(X)$ 为精确解，并对 $bold(w) = (bold(tau), bold(v))$ 记乘积图范数
$
  norm(bold(w))_(bold(X))^2
  := norm(bold(tau))_(bold(H)(div))^2 + norm(bold(v))_(H^1(Omega))^2
  quad "或" quad
  norm(bold(w))_(bold(X))^2
  := norm(bold(tau))_(bold(H)(div div))^2 + norm(v)_(H^2(Omega))^2.
$
两个模型的数值积分投影场都落在约束空间中。线弹性中 $Pi_"tr" hat(bold(sigma))_N (bold(c))$ 的平均迹为零，且 $tilde(Pi)_(D,K,Q_"R")^(1) hat(bold(u))_N (bold(c)) in V_K^(1) subset H_0^1(Omega; RR^d)$。板弯曲中弯矩分量不受约束，且 $tilde(Pi)_(D,K,Q_"R")^(2) hat(w)_N (bold(c)) in V_K^(2) subset H_0^2(Omega)$。故只要投影映射良定义，对一切 $bold(c) in RR^(m_N)$ 有 $tilde(bold(z))_(N,K)(bold(c)) in bold(X)$。

对 $bold(w) in bold(X)$ 记
$
  abs(bold(w))_(cal(J)) := cal(J)(bold(w); 0)^(1\/2).
$
载荷只出现在平衡残差的常数项中，两个残差都是场的仿射函数且在精确解处同时为零，因此对任意 $bold(z) in bold(X)$，
$
  cal(J)(bold(z); f) = abs(bold(z) - bold(z)_star)_(cal(J))^2.
$
<eq:residual-shift>
@thm:elasticity-stability 与 @thm:plate-stability 即
$
  norm(bold(w))_(bold(X))^2
  lt.tilde abs(bold(w))_(cal(J))^2
  lt.tilde norm(bold(w))_(bold(X))^2,
  quad forall bold(w) in bold(X),
$
<eq:residual-equivalence>
其中两侧常数在线弹性中可关于 $lambda$ 一致选取。

下述引理是总误差分析的核心：它把计算解的误差归结为系数球内数值积分投影场的最佳逼近误差，加上训练求积与代数求解两项。引理对固定的样本实现逐点成立，不涉及任何收敛阶。

#lemma(title: [离散极小点的拟最优性])[
  固定 Ritz 投影构造样本与训练样本的一个实现，设投影映射良定义。设 $bold(c)^"out" in cal(C)_(N,B)$ 满足 @eq:suboptimality，并按 @eq:centered-risk 记训练泛函在系数球上的中心化一致偏差为
  $
    delta_Q := sup_(bold(c) in cal(C)_(N,B))
    abs(cal(J)^circle.stroked.tiny (bold(c)) - cal(J)_Q^circle.stroked.tiny (bold(c))).
  $
  则
  $
    norm(bold(z)_star - tilde(bold(z))_(N,K)(bold(c)^"out"))_(bold(X))^2
    lt.tilde inf_(bold(c) in cal(C)_(N,B))
    norm(bold(z)_star - tilde(bold(z))_(N,K)(bold(c)))_(bold(X))^2
    + delta_Q + epsilon_"opt",
  $
  隐含常数只由 @eq:residual-equivalence 的两个常数合成，不依赖于 $N$、$Q$、$K$、$B$ 与样本实现。
]<lem:quasi-optimality>

#proof[
  任取 $bold(c) in cal(C)_(N,B)$。中心化 @eq:centered-risk 扣除的两项与系数无关，在 $bold(c)^"out"$ 与 $bold(c)$ 处的两个偏差相减时抵消，故由 $delta_Q$ 的定义与 @eq:suboptimality，
  $
    cal(J)(tilde(bold(z))_(N,K)(bold(c)^"out"); f)
    & <= cal(J)_Q (tilde(bold(z))_(N,K)(bold(c)^"out"))
    + cal(J)(tilde(bold(z))_(N,K)(bold(c)); f)
    - cal(J)_Q (tilde(bold(z))_(N,K)(bold(c))) + 2 delta_Q \
    & <= cal(J)(tilde(bold(z))_(N,K)(bold(c)); f) + 2 delta_Q + epsilon_"opt".
  $
  <eq:empirical-chain>
  $tilde(bold(z))_(N,K)(bold(c)^"out")$、$tilde(bold(z))_(N,K)(bold(c))$ 与 $bold(z)_star$ 都属于 $bold(X)$。对左端用 @eq:residual-shift 与 @eq:residual-equivalence 的下界，对右端用 @eq:residual-shift 与 @eq:residual-equivalence 的上界，得
  $
    norm(tilde(bold(z))_(N,K)(bold(c)^"out") - bold(z)_star)_(bold(X))^2
    lt.tilde cal(J)(tilde(bold(z))_(N,K)(bold(c)^"out"); f),
    quad
    cal(J)(tilde(bold(z))_(N,K)(bold(c)); f)
    lt.tilde norm(tilde(bold(z))_(N,K)(bold(c)) - bold(z)_star)_(bold(X))^2.
  $
  联立三式并对 $bold(c)$ 取下确界即得结论。证毕。
]

=== 两类模型的总误差

线弹性中令 $bold(z)_star=(bold(sigma)_star,bold(u)_star)$，并采用 @sec:train-functional 定义的应力--位移离散场。

#theorem(title: [线弹性线性化网络完全离散最小二乘误差])[
  设 $Omega$ 具有 @def:admissible-domain 所定义的有限单纯形剖分，假设 @thm:elasticity-stability 成立，激活幂次 $k >= 2$，应力与位移的参数带点集 $Theta_(bold(sigma),N)^"band"$ 与 $Theta_(bold(u),N)^"band"$ 均满足 @def:quasi-uniform，位移协调空间族 ${V_K^(1)}_K$ 满足 @def:aux-admissible。设 $bold(sigma)_star$ 与 $bold(u)_star$ 的各分量分别属于 $H^(s_(bold(sigma)))(Omega)$ 与 $H^(s_(bold(u)))(Omega)$，$s_chi in [1, s_"cap" (d)]$，载荷 $bold(f) in L^oo (Omega; RR^d)$，且预算 $B$ 满足 @eq:budget。固定 $0<delta,eta<1$，令 $Q_"R" >= C K log(2K/eta)$，其中 $C$ 足够大。期望对训练样本与条件于 $cal(E)_("R",delta)$ 的 Ritz 投影构造样本取，则
  $
    (EE norm(bold(z)_star - bold(z)_(N,Q,K))_(bold(X)_"LE")^2)^(1\/2)
    lt.tilde & N^(-beta_"LE") \
             & + K^(-(min(s_(bold(u)), r_1)-1)\/d) \
             & + B sqrt(K/Q_"R") \
             & + (B^2 K
                 + norm(bold(f))_(L^oo (Omega)) B sqrt(K))^(1\/2) Q^(-1\/4) \
             & + epsilon_("LE","opt")^(1\/2).
  $
  隐含常数依赖于 $Omega$、$mu$、$d$、字典参数、固定的谱稳定参数、协调空间的逼近与逆估计常数及精确解各分量的 Sobolev 范数，不依赖于 $N$、$Q$、$K$、$B$ 与 $lambda$。
]<thm:total-error-le>

#proof[
  第一步：比较系数。由 @prop:comparison-field 与预算条件 @eq:budget，存在 $bold(c)^"cmp" in cal(C)_(N,B)$，使原始字典场以 $N^(-beta_"LE")$ 阶逼近精确解。应力分量由传递估计 @eq:projection-transfer-tr 控制。位移分量先由三角不等式分出 Ritz 求积差，再由传递估计 @eq:finite-projection-transfer 控制：
  $
    norm(bold(u)_star - tilde(Pi)_(D,K,Q_"R")^(1) hat(bold(u))_N (bold(c)^"cmp"))_(H^1(Omega))
    <= cal(E)_(1,K)(bold(u)_star)
    + norm(bold(u)_star - hat(bold(u))_N (bold(c)^"cmp"))_(H^1(Omega))
    + epsilon_("LE","Rquad").
  $
  两分量平方相加，并由 @eq:aux-approx-rate 估计最佳逼近量，得
  $
    norm(bold(z)_star - tilde(bold(z))_(N,K)(bold(c)^"cmp"))_(bold(X)_"LE")^2
    & lt.tilde N^(-2 beta_"LE") + cal(E)_(1,K)(bold(u)_star)^2 + epsilon_("LE","Rquad")^2, \
    cal(E)_(1,K)(bold(u)_star)
    & lt.tilde K^(-(min(s_(bold(u)), r_1)-1)\/d)
    norm(bold(u)_star)_(H^(s_(bold(u)))(Omega)).
  $
  <eq:comparison-bound-le>
  比较系数与最佳逼近量由字典、协调空间与精确解决定，与全部样本无关。

  第二步：拟最优性。固定满足 $cal(E)_("R",delta)$ 的 Ritz 投影构造样本与训练样本，在 @lem:quasi-optimality 中取 $bold(c) = bold(c)^"cmp"$，代入 @eq:comparison-bound-le，得
  $
    norm(bold(z)_star - bold(z)_(N,Q,K))_(bold(X)_"LE")^2
    lt.tilde N^(-2 beta_"LE")
    + cal(E)_(1,K)(bold(u)_star)^2
    + epsilon_("LE","Rquad")^2
    + delta_("LE",Q)
    + epsilon_("LE","opt").
  $

  第三步：逐层取期望。右端只有 $delta_("LE",Q)$ 依赖于训练样本。按 @eq:projection-conditioning 对训练样本取条件期望，@cor:train-generalization-k 给出
  $
    EE_"train" delta_("LE",Q)
    lt.tilde (B^2 K + norm(bold(f))_(L^oo (Omega)) B sqrt(K)) Q^(-1\/2),
  $
  其常数在 $cal(E)_("R",delta)$ 上对投影构造样本一致。再对条件于 $cal(E)_("R",delta)$ 的 Ritz 投影构造样本取期望，由 @cor:rquad-admissible，$EE_"R" [epsilon_("LE","Rquad")^2 | cal(E)_("R",delta)] lt.tilde B^2 K Q_"R"^(-1)$。两组样本相互独立，由条件期望的塔式性质合并两层期望，开方即得结论。证毕。
]

板弯曲中令 $bold(z)_star=(bold(M)_star,w_star)$，并采用 @sec:train-functional 定义的弯矩--挠度离散场。

#theorem(title: [板弯曲线性化网络完全离散最小二乘误差])[
  假设 @thm:plate-stability 成立，激活幂次 $k >= 3$，弯矩与挠度的参数带点集 $Theta_(bold(M),N)^"band"$ 与 $Theta_(w,N)^"band"$ 均满足 @def:quasi-uniform，挠度协调空间族 ${V_K^(2)}_K$ 满足 @def:aux-admissible。设 $bold(M)_star$ 与 $w_star$ 的各分量分别属于 $H^(s_(bold(M)))(Omega)$ 与 $H^(s_w)(Omega)$，$s_chi in [2, s_"cap" (2)]$，载荷 $f in L^oo (Omega)$，且预算 $B$ 满足 @eq:budget。固定 $0<delta,eta<1$，令 $Q_"R" >= C K log(2K/eta)$，其中 $C$ 足够大。期望对训练样本与条件于 $cal(E)_("R",delta)$ 的 Ritz 投影构造样本取，则
  $
    (EE norm(bold(z)_star - bold(z)_(N,Q,K))_(bold(X)_"KL")^2)^(1\/2)
    lt.tilde & N^(-beta_"KL") \
             & + K^(-(min(s_w, r_2)-2)\/2) \
             & + B sqrt(K/Q_"R") \
             & + (B^2 K
                 + norm(f)_(L^oo (Omega)) B sqrt(K))^(1\/2) Q^(-1\/4) \
             & + epsilon_("KL","opt")^(1\/2).
  $
  隐含常数依赖于 $Omega$、$mu$、$h$、字典参数、固定的谱稳定参数、协调空间的逼近与逆估计常数及精确解各分量的 Sobolev 范数，不依赖于 $N$、$Q$、$K$ 与 $B$。
]<thm:total-error-kl>

#proof[
  与 @thm:total-error-le 的证明平行。由 @prop:comparison-field（板弯曲情形）与 @eq:budget 取 $bold(c)^"cmp" in cal(C)_(N,B)$。弯矩分量不经投影，挠度分量由三角不等式、传递估计 @eq:finite-projection-transfer 与 @eq:aux-approx-rate 控制，得
  $
    norm(bold(z)_star - tilde(bold(z))_(N,K)(bold(c)^"cmp"))_(bold(X)_"KL")^2
    & lt.tilde N^(-2 beta_"KL") + cal(E)_(2,K)(w_star)^2 + epsilon_("KL","Rquad")^2, \
    cal(E)_(2,K)(w_star)
    & lt.tilde K^(-(min(s_w, r_2)-2)\/2) norm(w_star)_(H^(s_w)(Omega)).
  $
  固定满足 $cal(E)_("R",delta)$ 的 Ritz 投影构造样本与训练样本，@lem:quasi-optimality 给出
  $
    norm(bold(z)_star - bold(z)_(N,Q,K))_(bold(X)_"KL")^2
    lt.tilde N^(-2 beta_"KL")
    + cal(E)_(2,K)(w_star)^2
    + epsilon_("KL","Rquad")^2
    + delta_("KL",Q)
    + epsilon_("KL","opt").
  $
  先对训练样本取条件期望，$delta_("KL",Q)$ 由 @cor:train-generalization-k（板弯曲情形）控制。再对条件于 $cal(E)_("R",delta)$ 的 Ritz 投影构造样本取期望，$epsilon_("KL","Rquad")$ 由 @cor:rquad-admissible 控制。由塔式性质合并两层期望，开方即得结论。证毕。
]

由 @prop:layered-quasi-uniform，采用前述分层参数点集时，@thm:total-error-le 与 @thm:total-error-kl 关于参数带点集的假设自动成立。

代数误差 $epsilon_"opt"$ 是球约束凸最小二乘问题的离散目标次优性，其与谱截断及乘子方程求解容差的关系见 @sec:ball-solve。两类模型的总误差常数关于 $lambda$ 一致，还要求系数预算、精确解的 Sobolev 范数与投影稳定性界均能关于 $lambda$ 一致控制。

= 数值实验

本节首先固定激活幂次 $k=7$，考察完整离散解关于字典规模 $N$ 的观测收敛阶。随后固定 $N$，比较 $k in {3,5,7,9}$ 下完全离散最小二乘解的实际误差。最后在二维散度非零制造解上考察近不可压缩极限的稳定性。

== 实验设计

=== 实验配置与误差指标

四个模型共用 @tbl:ls-design 的配置。材料参数、激活幂次、字典规模、协调空间的次数与维数、两组求积的规模与系数预算取值序列逐模型取定。两类实验的配置表分开给出：关于 $N$ 的收敛实验列于各节开头，关于幂次 $k$ 的收敛实验另立一表。凡逐模型配置表中重复出现的配置项，均以该表为准。

图表报告独立测试规则上各分量图范数误差的十次均值 $bar(e)=1\/10 sum_(j=1)^10 e_j$，误差棒为样本标准差。由 $EE e <= (EE e^2)^(1\/2)$，两条定理也给出相应的条件平均误差界。

#figure(
  three-line-table(
    columns: 2,
    align: (left, left),
  )[
    | 配置项 | 取值 |
    |---|---|
    | 计算区域 | $Omega=(0,1)^d$，平均迹按 @eq:box-mean 精确计算 |
    | 参数点集 | 单隐层参数取确定性分层点集，偏置半径 $c_b=2$ |
    | 抽样规则 | 训练求积与 Ritz 投影求积均取均匀 Monte Carlo 点，两组样本独立抽取 |
    | 系统装配 | 约化系统由流式 Householder QR 装配 |
    | 代数求解 | 截断奇异值分解。二维与板取 $epsilon_"cut"=10^(-14)$，三维取 $10^(-13)$ |
    | 验证规则 | 张量积 Gauss--Legendre，二维 $32^2$ 点、三维 $16^3$ 点。需要扫描系数预算时按其上的图误差选出 |
    | 测试规则 | 张量积 Gauss--Legendre，二维 $128^2$ 点、三维 $32^3$ 点 |
    | 独立重复 | 10 次，每次重抽训练样本与投影求积点 |
  ],
  caption: [数值实验的公共配置],
)<tbl:ls-design>

== 系数球约束问题的求解 <sec:ball-solve>

记系数球半径为 $R_B:=B\/sqrt(m_N)$，则 @eq:train-problem 的约束集即 @eq:coeff-ball 的 $cal(C)_(N,B)={bold(c) in RR^(m_N):norm(bold(c))_(ell^2)<=R_B}$。对其中的加权残差矩阵作奇异值分解
$
  bold(A)_(N,Q)
  = bold(Y) bold(Gamma) bold(Z)^T,
  quad
  bold(Gamma)=op("diag")(gamma_1,dots.h,gamma_(m_N)),
  quad
  gamma_1 >= dots.h >= gamma_(m_N) >= 0.
$
给定相对截断水平 $epsilon_"cut"$，由奇异值已按降序排列，保留的指标恰为前
$
  n_"cut" := max {i:gamma_i>epsilon_"cut" gamma_1}
$
个，相应的截断因子记为 $bold(Y)_"cut"$、$bold(Gamma)_"cut"$ 与 $bold(Z)_"cut"$。二维模型取 $epsilon_"cut"=10^(-14)$。由于矩阵列不作归一化，约束 $norm(bold(c))_(ell^2)<=R_B$ 直接作用于表示物理场的系数。

在保留的奇异子空间上，KKT 条件给出谱滤波族
$
  bold(c)(zeta)
  := bold(Z)_"cut"
  (bold(Gamma)_"cut"^2+zeta bold(I)_(n_"cut"))^(-1)
  bold(Gamma)_"cut" bold(Y)_"cut"^T bold(b)_(N,Q),
  quad zeta>=0,
$
<eq:ball-filter>
其中 $zeta$ 为系数球约束的 Lagrange 乘子。取 $zeta=0$ 得到截断系统的最小范数解
$
  bold(c)(0)
  = bold(Z)_"cut" bold(Gamma)_"cut"^(-1)
  bold(Y)_"cut"^T bold(b)_(N,Q).
$

记右端在保留的左奇异基下的坐标为 $bold(g):=bold(Y)_"cut"^T bold(b)_(N,Q)$，则
$
  norm(bold(c)(zeta))_(ell^2)^2
  = sum_(i=1)^(n_"cut")
  (gamma_i^2 g_i^2)/(gamma_i^2+zeta)^2
  =: Psi(zeta).
$
<eq:ball-secular>
$Psi$ 在 $[0,oo)$ 上连续且单调递减。在非退化情形下，
$
  Psi'(zeta)
  = -2 sum_(i=1)^(n_"cut")
  (gamma_i^2 g_i^2)/(gamma_i^2+zeta)^3
  <0.
$
因此，当 $norm(bold(c)(0))_(ell^2)>R_B$ 时，方程
$
  Psi(zeta)=R_B^2
$
<eq:ball-secular-root>
存在唯一正根 $zeta_star$，可由二分法求得。最终输出为
$
  bold(c)^"out"
  = cases(
    bold(c)(0) \, & quad norm(bold(c)(0))_(ell^2)<=R_B,
    bold(c)(zeta_star) \, & quad norm(bold(c)(0))_(ell^2)>R_B.
  )
$
<eq:ball-output>
第一种情形中约束不激活。第二种情形中 $norm(bold(c)(zeta_star))_(ell^2)=R_B$。本文只报告 @eq:ball-output 的系数球解，因该解与 @thm:total-error-le、@thm:total-error-kl 所分析的离散问题一致。

由 @eq:train-objective，@eq:train-problem 的目标即离散泛函在系数上的表示，省略模型下标，代数次优性为
$
  epsilon_"opt"
  := cal(J)_Q(tilde(bold(z))_(N,K)(bold(c)^"out"))
  - inf_(bold(c) in cal(C)_(N,B)) cal(J)_Q(tilde(bold(z))_(N,K)(bold(c))).
$
<eq:opt-error>
若不作谱截断且精确求得 $zeta_star$，则 $epsilon_"opt"=0$。实际计算中，奇异值截断与 @eq:ball-secular-root 的有限求解精度共同贡献该项。求解流程汇总于 @alg:ball-solve。

#figure(
  kind: "algorithm",
  supplement: [算法],
  code-image(class: "center", theme => pseudocode-list(
    booktabs: true,
    stroke: 0.7pt + theme.main-color,
    booktabs-stroke: 1pt + theme.main-color,
    title: [*输入：* $bold(A)_(N,Q)$、$bold(b)_(N,Q)$、$B$、$m_N$、截断水平 $epsilon_"cut"$ 与求根容差 $epsilon_"sec"$],
  )[
    + 计算 $bold(A)_(N,Q)=bold(Y)bold(Gamma)bold(Z)^T$，并令 $n_"cut"=max {i:gamma_i>epsilon_"cut" gamma_1}$。
    + 取 $bold(Y)_"cut"$、$bold(Gamma)_"cut"$、$bold(Z)_"cut"$，计算 $bold(g)=bold(Y)_"cut"^T bold(b)_(N,Q)$、$R_B=B\/sqrt(m_N)$ 与 $bold(c)(0)=bold(Z)_"cut" bold(Gamma)_"cut"^(-1)bold(g)$。
    + 若 $B=oo$ 或 $norm(bold(c)(0))_(ell^2)<=R_B$，则返回 $bold(c)^"out"=bold(c)(0)$。
    + 按 @eq:ball-secular 定义 $Psi$。取 $zeta_-=0$、$zeta_+=max(gamma_1^2, 1)$，并逐次令 $zeta_+ arrow.r.long 2zeta_+$，直至 $Psi(zeta_+)<=R_B^2$。
    + 在 $[zeta_-,zeta_+]$ 上二分：令 $zeta=(zeta_-+zeta_+)\/2$。若 $Psi(zeta)>R_B^2$，置 $zeta_-=zeta$，否则置 $zeta_+=zeta$。重复至 $abs(Psi(zeta)-R_B^2)<=epsilon_"sec" max(R_B^2, 1)$。
    + 由 @eq:ball-filter 计算并返回 $bold(c)^"out"=bold(c)(zeta)$。
    - *输出：* 系数球约束解 $bold(c)^"out"$。
  ]),
  caption: [截断 SVD 系数球最小二乘法],
)<alg:ball-solve>

== 二维线弹性 <sec:main-results>

二维位移制造解取
$
  bold(u)_star = vec(
    e^(x_1 - x_2) x_1 (1-x_1) x_2 (1-x_2),
    sin(pi x_1) sin(pi x_2)
  ).
$
<eq:sol-elasticity-2d>

本模型的材料参数与关于 $N$ 的收敛实验的离散配置见 @tbl:ls-config-order-elasticity-2d，关于幂次 $k$ 的收敛实验的离散配置另见 @tbl:ls-config-power-elasticity-2d。其余配置项取 @tbl:ls-design 的公共值。

#figure(
  three-line-table(
    columns: 2,
    align: (left, left),
  )[
    | 配置项 | 取值 |
    |---|---|
    | 杨氏模量 | $E=4\/3$ |
    | 泊松比 | $nu=1\/3$ |
    | 剪切模量 | $mu=1\/2$ |
    | Lamé 常数 | $lambda=1$ |
    | 激活幂次 | $k=7$ |
    | 字典规模 | $N in {201,401,601,801,1001}$ |
    | 训练求积 | $Q=4N$ |
    | 协调空间 | $p=7$，$K in {441,841,1225,1681,2025}$ |
    | Ritz 投影求积 | $Q_"R"=12808$ |
    | 系数预算 | $B in {10^2,10^3,10^4,10^5,10^6,oo}$ |
    | $k=3$ 对照实验 | 除激活幂次取 $k=3$ 外，配置同上 |
  ],
  caption: [关于 $N$ 的收敛实验：二维线弹性的配置],
)<tbl:ls-config-order-elasticity-2d>

=== 关于 $N$ 的收敛实验

@fig:ls-convergence-order-elasticity-2d 给出固定 $k=7$ 时应力与位移图范数误差随 $N$ 的变化，@tbl:ls-order-elasticity-2d 列出各 $N$ 处的误差均值。

#figure(
  image("/public/images/least-squares/campaign/convergence-order-elasticity-2d.png"),
  caption: [固定 $k=7$ 时二维线弹性的应力与位移收敛曲线。虚线仅给出无约束原始字典逼近指数 @eq:beta-rate 的诊断参考斜率，不是 $p=7$ 完整算法的保证阶。误差棒为十次独立重复的样本标准差],
)<fig:ls-convergence-order-elasticity-2d>

#figure(
  three-line-table(
    columns: 3,
    align: (right,) * 3,
  )[
    | $N$ | 应力 $bold(H)(div)$ 误差 | 位移 $H^1$ 误差 |
    |---|---|---|
    | $201$ | $3.56 times 10^(-6)$ | $6.41 times 10^(-7)$ |
    | $401$ | $1.76 times 10^(-7)$ | $3.76 times 10^(-8)$ |
    | $601$ | $5.22 times 10^(-8)$ | $1.13 times 10^(-8)$ |
    | $801$ | $1.01 times 10^(-8)$ | $4.26 times 10^(-9)$ |
    | $1001$ | $2.79 times 10^(-9)$ | $5.79 times 10^(-10)$ |
  ],
  caption: [固定 $k=7$ 时二维线弹性在各 $N$ 处的误差均值],
)<tbl:ls-order-elasticity-2d>

$k=7$ 时应力与位移实际误差的观测阶分别为 $4.33$ 与 $4.06$，均高于字典逼近指数给出的参考值 $beta(7, 2, 1)=3.75$。

=== 关于幂次 $k$ 的收敛实验

本实验对 $k in {3,5,7,9}$ 分别运行完全离散最小二乘算法，字典规模固定为 $N=501$。全部离散配置见 @tbl:ls-config-power-elasticity-2d。

#figure(
  three-line-table(
    columns: 2,
    align: (left, left),
  )[
    | 配置项 | 取值 |
    |---|---|
    | 材料参数 | 同 @tbl:ls-config-order-elasticity-2d |
    | 激活幂次 | $k in {3,5,7,9}$ |
    | 字典规模 | $N=501$ |
    | 训练求积 | $Q=16N=8016$ |
    | 协调空间 | $p=10$，$K=1024$ |
    | Ritz 投影求积 | $Q_"R"=12808$ |
    | 系数预算 | $B in {10^3,3 times 10^3,10^4,3 times 10^4,oo}$ |
  ],
  caption: [关于幂次 $k$ 的收敛实验：二维线弹性的配置],
)<tbl:ls-config-power-elasticity-2d>

#figure(
  image("/public/images/least-squares/campaign/convergence-power-elasticity-2d.png"),
  caption: [固定 $N=501$ 时二维线弹性完全离散最小二乘解的图范数误差随激活幂次的变化。误差棒表示十次独立重复的样本标准差，虚线是忽略依赖于 $k$ 的常数后的 $N^(-(beta(k)-beta(3)))$ 等常数启发线，不是固定 $N$ 下的定量预测],
)<fig:ls-convergence-power-elasticity-2d>

#figure(
  three-line-table(
    columns: 3,
    align: (right,) * 3,
  )[
    | $k$ | 应力 $bold(H)(div)$ 误差 | 位移 $H^1$ 误差 |
    |---|---|---|
    | $3$ | $2.11 times 10^(-3)$ | $3.82 times 10^(-4)$ |
    | $5$ | $8.69 times 10^(-6)$ | $2.09 times 10^(-6)$ |
    | $7$ | $3.99 times 10^(-8)$ | $1.48 times 10^(-8)$ |
    | $9$ | $1.35 times 10^(-9)$ | $3.13 times 10^(-10)$ |
  ],
  caption: [固定 $N=501$ 时二维线弹性在各 $k$ 处的图范数误差均值],
)<tbl:ls-power-elasticity-2d>

@fig:ls-convergence-power-elasticity-2d 给出固定 $N=501$ 时应力与位移图范数误差随幂次 $k$ 的变化，@tbl:ls-power-elasticity-2d 列出各幂次处的误差均值。相邻幂次的位移均值分别下降约 $182$、$142$ 与 $47$ 倍，应力均值分别下降约 $242$、$218$ 与 $30$ 倍。

== 平面应力

平面应力实验采用与 @eq:sol-elasticity-2d 相同的位移制造解与材料参数，并将本构关系替换为平面应力柔度。在 $E=4/3$、$nu=1/3$ 下，平面应力约化参数为 $mu=1/2$ 与 $lambda_"ps"=(E nu)/(1-nu^2)=1/2$。@tbl:ls-config-order-elasticity-2d 中二维线弹性的 Lamé 常数为 $lambda=1$，故两者的本构张量不同。由于该实验用于考察本构关系改变是否造成额外退化，只报告 $k=7$ 时关于 $N$ 的收敛结果。本模型的配置见 @tbl:ls-config-order-plane-stress，其余配置项取 @tbl:ls-design 的公共值。

#figure(
  three-line-table(
    columns: 2,
    align: (left, left),
  )[
    | 配置项 | 取值 |
    |---|---|
    | 杨氏模量 | $E=4\/3$ |
    | 泊松比 | $nu=1\/3$ |
    | 本构模型 | 平面应力柔度 |
    | 平面应力 Lamé 参数 | $lambda_"ps"=1\/2$ |
    | 激活幂次 | $k=7$ |
    | 字典规模 | $N in {201,401,601,801,1001}$ |
    | 训练求积 | $Q=4N$ |
    | 协调空间 | $p=7$，$K in {441,841,1225,1681,2025}$ |
    | Ritz 投影求积 | $Q_"R"=12808$ |
    | 系数预算 | $B in {10^2,10^3,10^4,10^5,10^6,oo}$ |
  ],
  caption: [关于 $N$ 的收敛实验：平面应力的配置],
)<tbl:ls-config-order-plane-stress>

@fig:ls-convergence-order-plane-stress 给出固定 $k=7$ 时应力与位移图范数误差随 $N$ 的变化，@tbl:ls-order-plane-stress 列出各 $N$ 处的误差均值。

#figure(
  image("/public/images/least-squares/campaign/convergence-order-plane-stress.png"),
  caption: [固定 $k=7$ 时平面应力的应力与位移收敛曲线。虚线仅给出无约束原始字典逼近指数 @eq:beta-rate 的诊断参考斜率，不是 $p=7$ 完整算法的保证阶。误差棒为十次独立重复的样本标准差],
)<fig:ls-convergence-order-plane-stress>

#figure(
  three-line-table(
    columns: 3,
    align: (right,) * 3,
  )[
    | $N$ | 应力 $bold(H)(div)$ 误差 | 位移 $H^1$ 误差 |
    |---|---|---|
    | $201$ | $2.82 times 10^(-6)$ | $6.42 times 10^(-7)$ |
    | $401$ | $1.41 times 10^(-7)$ | $3.79 times 10^(-8)$ |
    | $601$ | $4.26 times 10^(-8)$ | $1.15 times 10^(-8)$ |
    | $801$ | $7.82 times 10^(-9)$ | $4.26 times 10^(-9)$ |
    | $1001$ | $2.15 times 10^(-9)$ | $5.78 times 10^(-10)$ |
  ],
  caption: [固定 $k=7$ 时平面应力在各 $N$ 处的误差均值],
)<tbl:ls-order-plane-stress>

应力与位移误差的观测阶分别为 $4.35$ 与 $4.06$，各 $N$ 处的误差也与二维线弹性接近，因而在所考察的参数范围内，没有观察到本构张量改变引起的附加退化。

== 三维线弹性

为与四面体网格上的 Hu--Zhang 对称混合有限元使用同一测试问题，三维实验采用 @HuZhang2015 第 4 节式 (4.1) 的制造解。在 $Omega=(0,1)^3$ 上令
$
  bold(u)_star=vec(16, 32, 64) product_(i=1)^3 x_i (1-x_i).
$
<eq:sol-elasticity-3d>
该解满足齐次位移边界条件，但不是无散场。事实上，记 $g_i=x_i(1-x_i)$，则
$
  div bold(u)_star
  =16(1-2x_1)g_2g_3+32g_1(1-2x_2)g_3+64g_1g_2(1-2x_3)
  !=0.
$
取 $bold(sigma)_star=2mu bold(epsilon)(bold(u)_star)+lambda div(bold(u)_star)bold(I)$，再由平衡方程制造体力 $bold(f)=-div bold(sigma)_star$。三维实验的配置见 @tbl:ls-config-order-elasticity-3d。与前述二维模型一致，本实验以 $N$ 为唯一自变量，训练点数 $Q$ 与 Ritz 协调空间维数 $K$ 按固定比例随 $N$ 配套变化。

#figure(
  three-line-table(
    columns: 2,
    align: (left, left),
  )[
    | 配置项 | 取值 |
    |---|---|
    | 杨氏模量 | $E=4\/3$ |
    | 泊松比 | $nu=1\/3$ |
    | 剪切模量 | $mu=1\/2$ |
    | Lamé 常数 | $lambda=1$ |
    | 激活幂次 | $k=7$ |
    | 字典规模 | $N in {201,401,601,801,1001}$ |
    | 训练求积 | $Q=16N$ 个均匀 Monte Carlo 点 |
    | 协调空间 | $p=7$，目标维数 $2N$，实际 $K in {512,1000,1331,1728,2197}$ |
    | Ritz 投影求积 | 固定 $Q_"R"=16384$ 个独立均匀 Monte Carlo 点 |
    | 验证与测试规则 | 分别固定为 $4096$ 与 $32768$ 个点 |
    | 系数预算 | 固定 $B=oo$ |
    | 代数求解 | 流式 Householder QR 后作 SVD，$epsilon_"cut"=10^(-13)$ |
    | 独立重复 | 每个 $N$ 作 10 次，重复编号对应相同的随机种子规则 |
  ],
  caption: [关于 $N$ 的收敛实验：三维线弹性的配置],
)<tbl:ls-config-order-elasticity-3d>

@fig:ls-convergence-order-elasticity-3d 给出三维应力与位移图范数误差随 $N$ 的变化，@tbl:ls-order-elasticity-3d 列出十次独立重复的误差均值。

#figure(
  image("/public/images/least-squares/campaign/convergence-order-elasticity-3d.png"),
  caption: [固定 $k=7$ 时三维线弹性的应力与位移收敛曲线。虚线仅给出无约束原始字典逼近指数 @eq:beta-rate 的诊断参考斜率，不是 $p=7$ 完整算法的保证阶。误差棒为十次独立重复的样本标准差],
)<fig:ls-convergence-order-elasticity-3d>

#figure(
  three-line-table(
    columns: 3,
    align: (right,) * 3,
  )[
    | $N$ | 应力 $bold(H)(div)$ 误差 | 位移 $H^1$ 误差 |
    |---|---|---|
    | $201$ | $3.49 times 10^(-7)$ | $1.73 times 10^(-8)$ |
    | $401$ | $2.63 times 10^(-7)$ | $2.98 times 10^(-9)$ |
    | $601$ | $9.07 times 10^(-8)$ | $1.06 times 10^(-9)$ |
    | $801$ | $4.03 times 10^(-8)$ | $2.63 times 10^(-9)$ |
    | $1001$ | $3.40 times 10^(-8)$ | $1.87 times 10^(-9)$ |
  ],
  caption: [固定 $k=7$ 时三维线弹性在各 $N$ 处的误差均值],
)<tbl:ls-order-elasticity-3d>

五点对数最小二乘拟合给出的应力与位移观测阶分别为 $1.59$ 与 $1.36$。两条曲线在大 $N$ 端均不再保持参考线的斜率，其中位移误差从 $N=601$ 起处于 $10^(-9)$ 量级的平台。因此这组完整算法结果验证了三维离散格式随字典加密总体收敛，但不能据此声称达到了字典逼近指数给出的参考值 $beta(7, 3, 1)=8/3$。


== 近不可压缩极限

参见 @CaiStarke2004 第 6 节与图 6.3，以及 @CaiKorsaweStarke2005 表 I--II 与图 4，本文以不同 Lamé 参数下离散加密曲线的观测阶与误差常数是否保持稳定作为近不可压缩锁定的数值判据。具体地，对每个 $lambda$ 采用相同的 $N$ 取值序列，若所得曲线具有一致的衰减阶，且未随 $lambda$ 增大而整体上移，则认为离散误差关于 Lamé 参数保持一致。

制造解取自 @LiYang2020 例 1（亦见 @GrieshaberMcBrideReddy2015）。在 $Omega=(0,1)^2$ 上令
$
  bold(u)_(star,lambda)(x_1,x_2)
  := vec(
    sin(2 pi x_2)(-1+cos(2 pi x_1))
    + (sin(pi x_1) sin(pi x_2))/(1+lambda),
    sin(2 pi x_1)(1-cos(2 pi x_2))
    + (sin(pi x_1) sin(pi x_2))/(1+lambda)
  ).
$
<eq:sol-near-incompressible-2d>
前一向量场严格无散，第二项则给出
$
  div bold(u)_(star,lambda)
  = pi/(1+lambda)
  (cos(pi x_1) sin(pi x_2) + sin(pi x_1) cos(pi x_2)),
$
故位移在边界上为零，散度不恒为零，且 $lambda div bold(u)_(star,lambda)=O(1)$。固定 $mu=1$，定义
$
  bold(sigma)_(star,lambda)
  := 2 bold(epsilon)(bold(u)_(star,lambda))
  + lambda div(bold(u)_(star,lambda)) bold(I),
  quad
  bold(f)_lambda := -div bold(sigma)_(star,lambda).
$

三组材料参数及其余离散配置见 @tbl:ls-config-near-incompressible-2d。

#figure(
  three-line-table(
    columns: 2,
    align: (left, left),
  )[
    | 配置项 | 取值 |
    |---|---|
    | 剪切模量 | $mu=1$ |
    | Lamé 常数 | $lambda in {10,10^3,10^5}$ |
    | 杨氏模量 | $E in {32\/11,3002\/1001,300002\/100001}$ |
    | 泊松比 | $nu in {5\/11,500\/1001,50000\/100001}$ |
    | 字典规模 | $N in {201,401,601,801,1001}$ |
    | 激活幂次 | $k=7$ |
    | 协调空间 | 次数 $p=9$，目标维数 $2N$，实际 $K in {441,841,1225,1681,2025}$ |
    | 训练求积 | $Q=8N$ 个均匀 Monte Carlo 点 |
    | Ritz 投影求积 | $Q_"R"=12808$ 个独立均匀 Monte Carlo 点 |
    | 验证规则 | 张量积 Gauss--Legendre，$64^2=4096$ 点 |
    | 测试规则 | 张量积 Gauss--Legendre，$128^2=16384$ 点 |
    | 系数预算 | 对所有 $N$、$lambda$ 固定 $B=3 times 10^6$，不逐材料调参 |
    | 代数求解 | 系数球法，流式 Householder QR，$epsilon_"cut"=10^(-14)$ |
    | 重复与共享 | 10 次配对重复，同一 $(N,r)$ 内三种 $lambda$ 共用字典及训练、投影、验证与测试规则 |
  ],
  caption: [近不可压缩实验的配置],
)<tbl:ls-config-near-incompressible-2d>

#figure(
  image("/public/images/least-squares/campaign/convergence-near-incompressible-elasticity-2d.png"),
  caption: [散度非零制造解在三种 Lamé 参数下的完整 $N$ 收敛曲线。点为十次配对重复的均值，误差棒为样本标准差，所有材料使用同一固定预算],
)<fig:ls-convergence-near-incompressible-2d>

#figure(
  three-line-table(
    columns: 4,
    align: (right,) * 4,
  )[
    | $N$ | $lambda=10$ | $lambda=10^3$ | $lambda=10^5$ |
    |---|---|---|---|
    | $201$ | $2.591 times 10^(-3)$ | $2.600 times 10^(-3)$ | $2.600 times 10^(-3)$ |
    | $401$ | $8.940 times 10^(-5)$ | $8.960 times 10^(-5)$ | $8.960 times 10^(-5)$ |
    | $601$ | $2.223 times 10^(-5)$ | $2.232 times 10^(-5)$ | $2.232 times 10^(-5)$ |
    | $801$ | $6.648 times 10^(-6)$ | $6.659 times 10^(-6)$ | $6.658 times 10^(-6)$ |
    | $1001$ | $1.508 times 10^(-6)$ | $1.514 times 10^(-6)$ | $1.514 times 10^(-6)$ |
  ],
  caption: [近不可压缩实验的应力 $bold(H)(div)$ 误差均值],
)<tbl:ls-near-incompressible-stress>

#figure(
  three-line-table(
    columns: 4,
    align: (right,) * 4,
  )[
    | $N$ | $lambda=10$ | $lambda=10^3$ | $lambda=10^5$ |
    |---|---|---|---|
    | $201$ | $3.394 times 10^(-4)$ | $3.432 times 10^(-4)$ | $3.433 times 10^(-4)$ |
    | $401$ | $1.221 times 10^(-5)$ | $1.232 times 10^(-5)$ | $1.232 times 10^(-5)$ |
    | $601$ | $4.098 times 10^(-6)$ | $4.123 times 10^(-6)$ | $4.123 times 10^(-6)$ |
    | $801$ | $7.103 times 10^(-7)$ | $7.152 times 10^(-7)$ | $7.161 times 10^(-7)$ |
    | $1001$ | $2.330 times 10^(-7)$ | $2.396 times 10^(-7)$ | $2.404 times 10^(-7)$ |
  ],
  caption: [近不可压缩实验的位移 $H^1$ 误差均值],
)<tbl:ls-near-incompressible-u>

@fig:ls-convergence-near-incompressible-2d 中三条加密曲线几乎重合，相同 $N$ 下三种材料的应力误差均值最大相差 $0.43\%$，位移误差均值最大相差 $3.17\%$。因此误差常数与观测阶均未随 $lambda$ 增大而退化，从而本实验未观察到近不可压缩锁定，并为 @thm:elasticity-stability 的 Lamé 一致性提供了数值佐证。

== Kirchhoff--Love 板弯曲

板挠度制造解取
$
  w_star = sin^2(pi x_1) sin^2(pi x_2).
$
<eq:sol-plate>
其挠度与法向导数在 $partial Omega$ 上同时为零，故满足固支条件。本模型的材料参数与关于 $N$ 的收敛实验的离散配置见 @tbl:ls-config-order-plate，关于幂次 $k$ 的收敛实验的离散配置另见 @tbl:ls-config-power-plate。其余配置项取 @tbl:ls-design 的公共值。

#figure(
  three-line-table(
    columns: 2,
    align: (left, left),
  )[
    | 配置项 | 取值 |
    |---|---|
    | 杨氏模量 | $E=1$ |
    | 泊松比 | $nu=0.3$ |
    | 板厚 | $h=1$ |
    | 激活幂次 | $k=7$ |
    | 字典规模 | $N in {201,401,601,801,1001}$ |
    | 训练求积 | $Q=4N$ |
    | 协调空间 | $p=7$，$K in {441,841,1225,1681,2025}$ |
    | Ritz 投影求积 | $Q_"R"=max(2048, 4K)$ |
    | 系数预算 | $B in {10^2,10^3,10^4,10^5,10^6,oo}$ |
    | $k=3$ 对照实验 | 除激活幂次取 $k=3$ 外，配置同上 |
  ],
  caption: [关于 $N$ 的收敛实验：Kirchhoff--Love 板弯曲的配置],
)<tbl:ls-config-order-plate>

=== 关于 $N$ 的收敛实验

@fig:ls-convergence-order-plate 给出固定 $k=7$ 时弯矩与挠度图范数误差随 $N$ 的变化，@tbl:ls-order-plate 列出各 $N$ 处的误差均值。

#figure(
  image("/public/images/least-squares/campaign/convergence-order-plate.png"),
  caption: [固定 $k=7$ 时 Kirchhoff--Love 板弯曲的弯矩与挠度收敛曲线。虚线仅给出无约束原始字典逼近指数 @eq:beta-rate 的诊断参考斜率，不是 $p=7$ 完整算法的保证阶。误差棒为十次独立重复的样本标准差],
)<fig:ls-convergence-order-plate>

#figure(
  three-line-table(
    columns: 3,
    align: (right,) * 3,
  )[
    | $N$ | 弯矩 $bold(H)(div div)$ 误差 | 挠度 $H^2$ 误差 |
    |---|---|---|
    | $201$ | $1.44 times 10^(-2)$ | $2.18 times 10^(-3)$ |
    | $401$ | $7.89 times 10^(-4)$ | $1.13 times 10^(-4)$ |
    | $601$ | $2.60 times 10^(-4)$ | $4.23 times 10^(-5)$ |
    | $801$ | $7.92 times 10^(-5)$ | $1.25 times 10^(-5)$ |
    | $1001$ | $2.22 times 10^(-5)$ | $3.21 times 10^(-6)$ |
  ],
  caption: [固定 $k=7$ 时板弯曲在各 $N$ 处的误差均值],
)<tbl:ls-order-plate>

$k=7$ 时弯矩与挠度实际误差的观测阶分别为 $3.88$ 与 $3.87$，均高于板图范数的理论基准 $beta(7, 2, 2)=3.25$。

=== 关于幂次 $k$ 的收敛实验

本实验对 $k in {3,5,7,9}$ 分别运行完全离散最小二乘算法，字典规模固定为 $N=1001$。全部离散配置见 @tbl:ls-config-power-plate。

#figure(
  three-line-table(
    columns: 2,
    align: (left, left),
  )[
    | 配置项 | 取值 |
    |---|---|
    | 材料参数 | 同 @tbl:ls-config-order-plate |
    | 激活幂次 | $k in {3,5,7,9}$ |
    | 字典规模 | $N=1001$ |
    | 训练求积 | $Q=8N=8008$ |
    | 协调空间 | $p=10$，$K=2025$ |
    | Ritz 投影求积 | $Q_"R"=12808$ |
    | 系数预算 | $B in {10^2,10^3,10^4,10^5,10^6,oo}$ |
  ],
  caption: [关于幂次 $k$ 的收敛实验：Kirchhoff--Love 板弯曲的配置],
)<tbl:ls-config-power-plate>

#figure(
  image("/public/images/least-squares/campaign/convergence-power-plate.png"),
  caption: [固定 $N=1001$ 时板弯曲完全离散最小二乘解的图范数误差随激活幂次的变化。误差棒表示十次独立重复的样本标准差，虚线是忽略依赖于 $k$ 的常数后的 $N^(-(beta(k)-beta(3)))$ 等常数启发线，不是固定 $N$ 下的定量预测],
)<fig:ls-convergence-power-plate>

#figure(
  three-line-table(
    columns: 3,
    align: (right,) * 3,
  )[
    | $k$ | 弯矩 $bold(H)(div div)$ 误差 | 挠度 $H^2$ 误差 |
    |---|---|---|
    | $3$ | $3.11 times 10^(-1)$ | $5.00 times 10^(-2)$ |
    | $5$ | $2.60 times 10^(-3)$ | $3.46 times 10^(-4)$ |
    | $7$ | $1.92 times 10^(-5)$ | $2.98 times 10^(-6)$ |
    | $9$ | $3.05 times 10^(-6)$ | $8.58 times 10^(-7)$ |
  ],
  caption: [固定 $N=1001$ 时板弯曲在各 $k$ 处的图范数误差均值],
)<tbl:ls-power-plate>

@fig:ls-convergence-power-plate 给出固定 $N=1001$ 时弯矩与挠度图范数误差随幂次 $k$ 的变化，@tbl:ls-power-plate 列出各幂次处的误差均值。相邻幂次的弯矩均值分别下降约 $119$、$135$ 与 $6.3$ 倍，挠度均值分别下降约 $145$、$116$ 与 $3.5$ 倍。四个幂次下实际误差持续下降。此外，$k=3$ 与 $k=7$ 下 $N$ 的完整取值序列分别给出 $1.22$ 与 $3.87$ 的挠度实际观测阶。

= 结语

本文的主要结果是线弹性与板弯曲的两条残差稳定性定理：混合最小二乘泛函与相应图范数平方双边等价，且等价常数关于 Lamé 常数 $lambda$ 一致，在近不可压缩极限下不退化。在此基础上，本文以准均匀单隐层字典的 Sobolev 逼近率与系数控制（@thm:band-dictionary-rate）为输入，给出两类方程的线性化网络误差分析。误差界对参数带中任意准均匀点集成立，本文的分层构造是其一个实例。字典逼近项是确定性的 $N^(-beta)$，没有独立抽样带来的对数修正，而有限维计算必须显式计入 Ritz 截断与投影积分误差。

误差界同时展示了各离散参数的不同作用：$N$ 控制单隐层逼近，$K$ 控制投影空间，$Q_"R"$ 控制投影构造中的积分误差，$Q$ 控制训练泛函的 Monte Carlo 一致偏差，而 $B$ 连接逼近表示与泛化控制。特别地，固定其余离散参数时，中心化训练泛函一致偏差的条件期望上界按 $Q^(-1\/2)$ 衰减。由离散极小化比较与残差稳定性得到的图范数均方根误差界中，相应的训练积分项按 $Q^(-1\/4)$ 衰减。

激活幂次 $k$ 通过饱和指数 $s_"cap" (d) = (d+2k+1)\/2$ 改变字典逼近项的理论指数。完整算法关于幂次 $k$ 的收敛实验表明，在离散资源与最高幂次相匹配后，二维线弹性与板弯曲在 $k=3,5,7,9$ 上都保持实际误差下降。两类模型的 $k=7$ 实际观测阶也均高于 $k=3$。

#bibliography("/public/reference/least-squares/least-squares.bib")

#set heading(numbering: "附录 A.1", supplement: [Appendix])
#counter(heading).update(0)

= 线弹性等价性定理证明 <app:elasticity>

本附录证明 @thm:elasticity-equivalence。

#proof[
  先证强形式与最小二乘极小化等价。若 $(bold(sigma), bold(u))$ 满足线弹性强形式，则两个残差均为零，从而
  $
    cal(J)_"LE" (bold(sigma), bold(u); bold(f)) = 0.
  $
  由于 $cal(J)_"LE"$ 是两个 $L^2$ 范数平方之和，必有 $cal(J)_"LE" >= 0$，因此该解是全局极小点。

  另一方面，假设线弹性方程组在 $bold(Sigma)_"LE" times bold(U)_"LE"$ 中存在解 $(bold(sigma)^*, bold(u)^*)$，则
  $
    cal(J)_"LE" (bold(sigma)^*, bold(u)^*; bold(f)) = 0,
  $
  从而 $cal(J)_"LE"$ 的全局最小值为 $0$。因此任意全局极小点 $(bold(sigma), bold(u))$ 都满足
  $
    cal(J)_"LE" (bold(sigma), bold(u); bold(f)) = 0.
  $
  于是
  $
    bold(cal(A))_"LE" bold(sigma) - bold(epsilon)(bold(u)) = 0,
    quad
    div bold(sigma) + bold(f) = 0
  $
  在 $Omega$ 中几乎处处成立，而 $bold(u) in bold(U)_"LE"$ 已自动满足齐次位移边界条件，因此 $(bold(sigma), bold(u))$ 满足原强形式。

  下面证明最小二乘极小化与变分问题等价。将 $cal(J)_"LE"$ 展开可得
  $
    cal(J)_"LE" (bold(tau), bold(v); bold(f))
    = a_"LE" ((bold(tau), bold(v)), (bold(tau), bold(v)))
    - 2 ell_"LE" (bold(tau), bold(v))
    + norm(bold(f))_(L^2(Omega))^2.
  $

  若 $(bold(sigma), bold(u))$ 是 $cal(J)_"LE"$ 的全局极小点，则对任意 $(bold(eta), bold(w)) in bold(Sigma)_"LE" times bold(U)_"LE"$，函数
  $
    phi(t) := cal(J)_"LE" (bold(sigma) + t bold(eta), bold(u) + t bold(w); bold(f))
  $
  在 $t = 0$ 处取极小值。由上述二次展开对 $t$ 求导并取 $t = 0$，得到
  $
    a_"LE" ((bold(sigma), bold(u)), (bold(eta), bold(w)))
    = ell_"LE" (bold(eta), bold(w)),
    quad forall (bold(eta), bold(w)) in bold(Sigma)_"LE" times bold(U)_"LE",
  $
  即得到变分问题。

  反之，若 $(bold(sigma), bold(u))$ 满足变分问题，则对任意 $(bold(tau), bold(v)) in bold(Sigma)_"LE" times bold(U)_"LE"$，记
  $
    (bold(eta), bold(w))
    := (bold(tau) - bold(sigma), bold(v) - bold(u)).
  $
  利用二次展开与变分等式，有
  $
    cal(J)_"LE" (bold(tau), bold(v); bold(f))
    - cal(J)_"LE" (bold(sigma), bold(u); bold(f))
    = a_"LE" ((bold(eta), bold(w)), (bold(eta), bold(w)))
    >= 0.
  $
  因此 $(bold(sigma), bold(u))$ 是 $cal(J)_"LE"$ 的全局极小点。证毕。
]

= Kirchhoff--Love 板弯曲等价性定理证明 <app:plate>

本附录证明 @thm:plate-equivalence。

#proof[
  若 $(bold(M), w)$ 满足板弯曲混合强形式，则两个残差同时为零，从而
  $
    cal(J)_"KL" (bold(M), w; f) = 0.
  $
  由于 $cal(J)_"KL"$ 是两个 $L^2$ 范数平方之和，必有 $cal(J)_"KL" >= 0$，因此该解是全局极小点。

  另一方面，设混合强形式在 $bold(Sigma)_"KL" times U_"KL"$ 中存在解 $(bold(M)^*, w^*)$，则 $cal(J)_"KL" (bold(M)^*, w^*; f) = 0$，从而 $cal(J)_"KL"$ 的全局最小值为 $0$。因此任意全局极小点 $(bold(M), w)$ 都满足
  $
    bold(cal(A))_"KL" bold(M) - bold(kappa)(w) = 0,
    quad
    div div bold(M) + f = 0
  $
  在 $Omega$ 中几乎处处成立，而 $w in H_0^2(Omega)$ 已自动满足固支边界条件，因此 $(bold(M), w)$ 满足混合强形式。

  接下来证明最小二乘极小化与变分问题等价。将 $cal(J)_"KL"$ 展开可得
  $
    cal(J)_"KL" (bold(tau), v; f)
    = a_"KL" ((bold(tau), v), (bold(tau), v))
    - 2 ell_"KL" (bold(tau), v)
    + norm(f)_(L^2(Omega))^2.
  $
  若 $(bold(M), w)$ 是 $cal(J)_"KL"$ 的全局极小点，则对任意 $(bold(eta), z) in bold(Sigma)_"KL" times U_"KL"$，函数
  $
    phi(t) := cal(J)_"KL" (bold(M) + t bold(eta), w + t z; f)
  $
  在 $t = 0$ 处取极小值。由上述二次展开对 $t$ 求导并取 $t = 0$，得到
  $
    a_"KL" ((bold(M), w), (bold(eta), z))
    = ell_"KL" (bold(eta), z),
    quad forall (bold(eta), z) in bold(Sigma)_"KL" times U_"KL",
  $
  即得到变分问题。

  反之，若 $(bold(M), w)$ 满足变分问题，则对任意 $(bold(tau), v) in bold(Sigma)_"KL" times U_"KL"$，记
  $
    (bold(eta), z)
    := (bold(tau) - bold(M), v - w).
  $
  利用二次展开与变分等式，有
  $
    cal(J)_"KL" (bold(tau), v; f)
    - cal(J)_"KL" (bold(M), w; f)
    = a_"KL" ((bold(eta), z), (bold(eta), z))
    >= 0.
  $
  因此 $(bold(M), w)$ 是 $cal(J)_"KL"$ 的全局极小点。证毕。
]
