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
本文的核心结果是两条残差稳定性定理。其一，线弹性最小二乘泛函与 $bold(H)(div) times H^1$ 图范数平方双边等价（@thm:elasticity-stability）；其二，板弯曲最小二乘泛函与 $bold(H)(div div) times H^2$ 图范数平方双边等价（@thm:plate-stability）。两条定理的等价常数都可关于 Lamé 常数 $lambda$ 一致选取，因而覆盖近不可压缩极限 $lambda -> oo$。两类问题的一致性机制并不相同：线弹性依赖偏差--体积分解与零平均迹下的迹估计，板弯曲依赖柔度算子的一致椭圆性与 $H_0^2$ 上的 Poincaré 不等式。经典应力--位移最小二乘稳定性可参见 @CaiStarke2003、@CaiStarke2004 与 @DuanLin2005。

本文的另一项工作是将单隐层线性化 $rho_k$ 网络用于两类方程的数值求解，并给出误差分析：总误差分为字典逼近误差、有限维 Ritz 及其积分误差、训练泛函的 Monte Carlo 误差与线性代数求解误差四类。激活幂次 $k$ 通过字典的饱和指数 $s_"cap" (d) = (d + 2k + 1)\/2$ 进入逼近误差界，因而改变关于字典规模 $N$ 的理论收敛指数。数值实验统一报告完整经验最小二乘算法在独立测试规则上的图范数误差，并考察字典规模、激活幂次与近不可压缩极限的影响。

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
本文以下标 $"LE"$（linear elasticity，线弹性）与 $"KL"$（Kirchhoff--Love 板弯曲）区分两类问题的算子、泛函与函数空间。以 $(bold(sigma), bold(u))$ 表示线弹性未知量，以 $(bold(M), w)$ 表示板弯曲未知量。$bold(T)_alpha$ 表示对称张量基。对非负量 $A$ 与 $B$，记 $A lt.tilde B$ 表示存在常数 $C > 0$ 使 $A <= C B$，其中 $C$ 不依赖于 Lamé 常数 $lambda$ 与任何离散参数，具体依赖关系在上下文中给出。$A gt.tilde B$ 即 $B lt.tilde A$。记 $A tilde.eq B$ 表示 $A lt.tilde B$ 与 $B lt.tilde A$ 同时成立，即 $A$ 与 $B$ 同阶，其中两侧常数满足同一约定。

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

  1. 线弹性强形式；
  2. 最小化 $cal(J)_"LE"$；
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

  1. Kirchhoff--Love 板弯曲强形式；
  2. 最小化 $cal(J)_"KL"$；
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

由 @thm:plate-stability，双线性形式 $a_"KL"$ 在 $bold(Sigma)_"KL" times U_"KL"$ 上连续且强制，其中常数可关于 Lamé 常数 $lambda$ 一致选取。值得对比的是：线弹性中柔度算子的体积部分在 $lambda -> oo$ 时退化，一致稳定性依赖偏差--体积分解与迹估计；板弯曲中柔度算子本身即一致椭圆，一致稳定性转而依赖 Poincaré 不等式从曲率恢复挠度范数。

= 线性化网络离散

本节定义离散模型并给出其逼近性质：先定义单隐层特征与参数域，再转述准均匀字典的 Sobolev 逼近结果，随后给出参数点集的确定性构造，最后按物理分量组装张量与向量离散空间。

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
其中 $bold(omega)$ 称为方向，$b$ 称为偏置，两者即被冻结的隐藏层参数。该函数沿 $bold(omega)$ 变化，在与 $bold(omega)$ 正交的方向上取常值，这类函数称为脊函数（ridge function）。冻结参数生成的特征族称为字典。记
$
  R_Omega
  := sup_(bold(omega) in bb(S)^(d-1), bold(x) in overline(Omega)) abs(bold(omega) dot bold(x))
  = sup_(bold(x) in overline(Omega)) norm(bold(x))_(ell^2),
$
其中 $bb(S)^(d-1) subset RR^d$ 为单位球面；由 $Omega$ 有界，$R_Omega < oo$。方向--偏置的参数域取柱面 $bb(S)^(d-1) times [-c_b, c_b]$，其中偏置半径 $c_b > R_Omega$ 为取定常数。记 $k$ 次多项式空间为
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
  其中 $Theta^"band" subset cal(P)_Omega$ 关于 $N$ 满足 @def:quasi-uniform，$h(Theta^"band")$ 为其覆盖半径；$Theta^"poly" subset bb(S)^(d-1) times (R_Omega, c_b]$ 为固定的 $n_P$ 对参数，$n_P := dim P_k (Omega)$，其生成的特征构成 $P_k (Omega)$ 的一组基。将全部参数特征枚举为字典
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
  由 @def:quasi-uniform，$h(Theta^"band") lt.tilde N^(-1\/d)$。反之，$frak(P)(cal(P)_Omega)$ 是 $bb(S)^d$ 上测度为正的固定区域，以 $Theta^"band"$ 的 $abs(Theta^"band")$ 个像点为心、$h(Theta^"band")$ 为半径的测地球将其覆盖，故 $abs(Theta^"band") h(Theta^"band")^d gt.tilde 1$；而 $abs(Theta^"band") <= N$，故 $h(Theta^"band") gt.tilde N^(-1\/d)$。于是 $h(Theta^"band") tilde.eq N^(-1\/d)$。代入 @thm:band-dictionary-rate 的两个估计：由 $s >= m$ 得 $h(Theta^"band")^(s-m) lt.tilde N^(-(s-m)\/d)$，由 $s <= s_"cap" (d)$ 得 $h(Theta^"band")^(-(s_"cap" (d)-s)) lt.tilde N^((s_"cap" (d)-s)\/d)$。证毕。
]

=== 参数点集构造

以 $chi$ 标记物理变量：线弹性中 $chi in {bold(sigma), bold(u)}$，板弯曲中 $chi in {bold(M), w}$。对每个物理变量，取含 $N$ 对 $(bold(omega), b)$ 的确定性点集，并作不交分解
$
  Theta_(chi,N) = Theta^"poly" union.sq Theta_(chi,N)^"band",
  quad Theta^"poly" inter Theta_(chi,N)^"band" = emptyset.
$
其中 $Theta^"poly" subset bb(S)^(d-1) times (R_Omega,c_b]$ 是固定的 $n_P$ 对参数：前文的二项展开已说明多项式区特征张成 $P_k (Omega)$，而张成组必含一组基，故可取出这样的 $n_P$ 对参数，使其生成的特征构成 $P_k (Omega)$ 的一组基；$Theta_(chi,N)^"band" subset cal(P)_Omega$ 含其余 $N-n_P$ 对参数。故 $abs(Theta_(chi,N)^"band") tilde.eq N$，多项式补充的大小 $n_P=binom(d+k, k)$ 只依赖于 $d$、$k$，与 $N$ 无关。记 $Theta_(chi,N)$ 的元素为 $(bold(omega)_(chi,j),b_(chi,j))$，$1<=j<=N$，并定义
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

- $d = 2$ 取等距角网格，其覆盖半径为 $pi\/n_1$，该界直接得到；
- $d = 3$ 取 $bb(S)^2$ 上的 Fibonacci 格，其准均匀性见 @HardinMichaelsSaff2016。

#figure(
  code-image(class: "center", theme => diagram(
    spacing: (24mm, 12mm),
    node((0, 0), $bb(S)^(d-1) times [0, 1]$, name: <prod>),
    node((1, 0), $cal(P)_Omega$, name: <cyl>),
    node((2, 0), $frak(P)(cal(P)_Omega) subset bb(S)^d$, name: <sph>),
    edge(<prod>, <cyl>, $(bold(omega), t) |-> (bold(omega), b(t))$, "->", stroke: 0.6pt + theme.main-color),
    edge(<cyl>, <sph>, $frak(P)$, "->", stroke: 0.6pt + theme.main-color),
    edge(<prod>, <sph>, [双向 Lipschitz], "->", bend: 38deg, stroke: 0.6pt + theme.main-color),
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

于是对每个物理变量 $chi$，@thm:band-dictionary-rate 中的 $Theta_N$ 即取为 $Theta_(chi,N)$：带部分 $Theta_(chi,N)^"band"$ 关于 $N$ 的准均匀性由 @prop:layered-quasi-uniform 保证，多项式部分即固定的 $Theta^"poly"$，定理的字典枚举 $(phi_j)_(j=1)^N$ 逐字等于特征族 ${xi_(chi,j)}_(j=1)^N$，@cor:quasi-uniform-rate 的两个估计随之直接成立。

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
本文以上标 $hat(dot)$ 统一标记这类尚未施加平均迹规范或边界投影的原始字典空间，以及其中的原始近似与定义在其上的泛函（如后文的 $hat(v)_N$ 与 $hat(cal(J))_"LE"$）。经过约束投影的物理量不带该记号。

== 系数约束

将给定模型的全部独立输出系数依固定顺序排成向量 $bold(c) in RR^(m_N)$，其中 $m_N$ 是离散系数总数，即上述原始字典空间的维数之和：线弹性为 $m_N=(n_s+d)N$，板弯曲为 $m_N=4N$。记 $bold(c)$ 的应力（板弯曲为弯矩）块系数为 $(c_(j,alpha))_(1<=j<=N, 1<=alpha<=n_s)$、位移块系数为 $(c_(j,i))_(1<=j<=N, 1<=i<=d)$，并将 $bold(c)$ 在原始空间中确定的场记为
$
  hat(bold(tau))_bold(c) := sum_(j,alpha) c_(j,alpha) xi_(bold(sigma),j) bold(T)_alpha,
  quad
  hat(bold(v))_bold(c) := sum_(j,i) c_(j,i) xi_(bold(u),j) bold(e)_i;
$
板弯曲为二维问题，此时 $n_s=3$，将 $xi_(bold(sigma),j)$ 换成 $xi_(bold(M),j)$，位移块换成标量挠度 $hat(w)_bold(c) := sum_j c_j xi_(w,j)$。

对给定预算 $B>0$，定义系数球
$
  cal(C)_(N,B)
  := {bold(c) in RR^(m_N):sqrt(m_N) norm(bold(c))_(ell^2)<=B}
  = {bold(c) in RR^(m_N):norm(bold(c))_(ell^2)<=B\/sqrt(m_N)}.
$
<eq:coeff-ball>
该缩放即 @LiuMaoXu2025 线性化网络类中的系数约束，与 ReLU 幂网络的有界表示相容，并避免同一物理函数仅因特征数改变而获得不同的系数尺度。系数约束施加在未作列归一化的物理系数上。以下凡出现 $sup_(bold(c) in cal(C)_(N,B))$ 的估计，均是在半径为 $B\/sqrt(m_N)$ 的球上取一致上确界。

== 连续正交投影

平均迹投影 $Pi_"tr"$ 定义为
$
  Pi_"tr" bold(tau)
  := bold(tau)
  - 1/(d abs(Omega))
  (integral_Omega tr(bold(tau)) dif x) bold(I).
$
该投影是从 $bold(H)(div)$ 到 $bold(Sigma)_"LE"$ 的有界投影。对 $m in {1,2}$，令 $Pi_D^(m)$ 为 $H^m (Omega)$ 到闭子空间 $H_0^m (Omega)$ 的正交投影（存在性与正交性参见 @BrennerScott2008 命题 2.3.1）：
$
  (Pi_D^(m) v,z)_(H^m (Omega))
  =(v,z)_(H^m (Omega)),
  quad forall z in H_0^m (Omega).
$
向量情形逐分量作用。由于正交投影的算子范数为 $1$，
$
  norm(Pi_D^(m) v)_(H^m)<=norm(v)_(H^m).
$

原始特征是 $rho_k$ 与仿射函数的复合，由 $rho_k$ 在有界区间上属于 $W^(k,oo)$，原始字典空间的元素逐分量属于 $W^(k,oo)(Omega) subset H^k(Omega)$。线弹性的图范数只含一阶导数，$k >= 1$ 已保证 $hat(bold(Sigma))_("LE",N) subset bold(H)(div)$ 与 $hat(bold(U))_("LE",N) subset H^1(Omega; RR^d)$。板弯曲的图范数含二阶导数，以下对板弯曲设 $k >= 2$，从而 $hat(bold(Sigma))_("KL",N) subset bold(H)(div div)$ 与 $hat(U)_("KL",N) subset H^2(Omega)$。

本文不改变原始空间，而将投影写进被最小化的泛函。对
$hat(bold(z))=(hat(bold(tau)),hat(bold(v)))$ 定义
$
  hat(cal(J))_"LE" (hat(bold(z));bold(f))
  :=cal(J)_"LE" (
    Pi_"tr" hat(bold(tau)),
    Pi_D^(1) hat(bold(v));bold(f)
  ),
$
并对板问题定义
$
  hat(cal(J))_"KL" (hat(bold(tau)),hat(v);f)
  :=cal(J)_"KL" (
    hat(bold(tau)),Pi_D^(2) hat(v);f
  ).
$
离散未知量直接记作
$
  (bold(sigma)_N,bold(u)_N) & :=(Pi_"tr" hat(bold(sigma))_N,
                                Pi_D^(1) hat(bold(u))_N), \
            (bold(M)_N,w_N) & :=(hat(bold(M))_N,
                                Pi_D^(2) hat(w)_N).
$
因此求解变量仍是原始简单张成空间中的系数，物理残差则在投影后的场上计算。在最小二乘框架下强加而非弱化本质边界条件是有限元文献的标准做法（@BochevGunzburger2009）。神经网络方法则更常用边界软残差（@RaissiPerdikarisKarniadakis2019、@ChenChiEYang2022），或以构造性乘子与距离函数精确满足边界条件（@LagarisLikasFotiadis1998、@SukumarSrivastava2022）。@Liu2026 通过一阶系统重写规避二阶提法中的余法向迹（conormal trace）与逆迹（inverse-trace）障碍，但仍以 $L^2(partial Omega)$ 边界残差处理 Dirichlet 数据，且其结论在诱导的最小二乘范数中成立。本文采用投影，因此没有边界软残差及其权重选择问题。若要使含边界残差的最小二乘泛函与完整图范数等价，边界失配通常须以 $H^(1\/2)$ 型分数阶范数或相应加权范数度量，参见 @BochevGunzburger2009 与 @MonsuurSmeetsStevenson2025。该构造也不损失字典的逼近阶：对 $m in {1,2}$，若 $v_star in H_0^m (Omega)$，则 $Pi_D^(m) v_star = v_star$，从而对任意 $hat(v)_N in H^m (Omega)$ 有
$
  norm(v_star-Pi_D^(m) hat(v)_N)_(H^m)
  =norm(Pi_D^(m)(v_star-hat(v)_N))_(H^m)
  <=norm(v_star-hat(v)_N)_(H^m).
$
<eq:projection-transfer>

该传递估计表明，所需假设仅是物理变量本身的 Sobolev 正则性。椭圆型投影带来的理论收益正是这一算子范数为 $1$ 的稳定传递估计。代价是实际计算中必须近似求解投影问题。

== 有限维 Ritz 实现

以下有限维 Ritz 构造及其显式逼近阶限于数值实验采用的张量积盒状区域，具体取 $Omega=(0,1)^d$。

对 $m in {1,2}$，取边界适配的张量积 $p$ 次 B 样条空间（$p >= max(m, 2) + 1$）
$
  V_K^(m) subset H_0^m (Omega),
  quad dim V_K^(m)=K,
$
并定义精确 Ritz 投影 $Pi_(D,K)^(m):H^m (Omega)->V_K^(m)$：
$
  (Pi_(D,K)^(m)v,z_K)_(H^m)
  =(v,z_K)_(H^m),
  quad forall z_K in V_K^(m).
$
对 $m=1$，删除在边界取非零值的端点 B 样条；对 $m=2$，进一步删除具有非零边界法向导数的端点模态。开放节点向量下，每端只有第一个 B 样条在端点取非零值，只有前两个有非零端点导数，与 $p$ 无关，因此辅助空间对任意 $p$ 都精确满足零迹或固支迹条件。

对任意 $v_star in H_0^m (Omega)$ 与原始近似 $hat(v)_N$，正交性给出
$
  norm(v_star-Pi_(D,K)^(m)hat(v)_N)_(H^m) & <=norm((I-Pi_(D,K)^(m))v_star)_(H^m) \
                                          & quad +norm(v_star-hat(v)_N)_(H^m).
$
第一项是辅助 Ritz 空间误差，第二项保留原始字典逼近阶。注意 $Pi_(D,K)^(m)$ 不是连续投影 $Pi_D^(m)$。即使二者都满足边界条件，也不能在分析中混为同一算子。

辅助 Ritz 空间误差有显式阶。设张量积网格尺寸为 $h_K tilde.eq K^(-1\/d)$。$p$ 次 B 样条的阶数为 $p+1$。在单位区间上取保持相应齐次端点条件的 $p$ 次样条拟插值，再逐方向作张量积，可得 $v in H^s (Omega) inter H_0^m (Omega)$（$s >= m$）的边界适配逼近，其误差为 $h_K^(min(s, p+1)-m) norm(v)_(H^s (Omega))$ 量级（样条逼近参见 @Schumaker2007）。由于 $Pi_(D,K)^(m)$ 是 $V_K^(m)$ 上的 $H^m$ 正交投影（投影存在性与最佳逼近性质参见 @BrennerScott2008 命题 2.3.1），故
$
  norm((I-Pi_(D,K)^(m))v)_(H^m)
  lt.tilde K^(-(min(s, p+1)-m)\/d) norm(v)_(H^s (Omega)).
$
<eq:ritz-rate>
上式中 $s$ 是对被投影函数的正则性假设，指数在 $s = p+1$ 处饱和：$p+1$ 是样条侧的饱和指数，地位与后文字典侧的 $s_"cap" (d)$ 相同，解更光滑时阶不再提高。

两个饱和指数必须匹配。取 $K tilde.eq N$，则 Ritz 项与字典项分别按 $N^(-(min(s, p+1)-m)\/d)$ 与 $N^(-(min(s, s_"cap" (d))-m)\/d)$ 衰减，故只要
$
  p + 1 >= s_"cap" (d) = (d+2k+1)/2,
$
<eq:degree-match>
辅助空间就不会成为渐近瓶颈。反之，无论字典多强，总误差都被 Ritz 项限制在较低的阶。这一条件对 $k=3$ 只要求 $p >= 4$，对 $k=7$ 则要求 $p >= 8$（$d=2$）或 $p >= 9$（$d=3$），因而在提高激活幂次时必须同步提高样条次数。关于 $N$ 的收敛实验取 $p = 7$。为使二维线弹性与板弯曲关于幂次 $k$ 的收敛实验覆盖到 $k=9$，这两个实验均取 $p=10$，从而满足 $p+1 >= s_"cap"(2)=10.5$。

数值实验采用一组与训练点独立的均匀样本
${bold(x)_r^"R"}_(r=1)^(Q_"R")$ 离散 $H^m$ 内积。记
$
  cal(L)_m (bold(x))v
  := (partial^bold(alpha) v(bold(x)))_(abs(bold(alpha)) <= m),
$
并定义
$
  (v,z)_(H^m,Q_"R")
  := abs(Omega)/Q_"R" sum_(r=1)^(Q_"R")
  cal(L)_m (bold(x)_r^"R")v dot cal(L)_m (bold(x)_r^"R")z.
$
由该经验内积定义的投影记作 $tilde(Pi)_(D,K,Q_"R")^(m)$。平均迹也不使用训练样本估计，而是另取一组与训练规则、Ritz 规则均独立的均匀 Monte Carlo 点
${bold(y)_r}_(r=1)^(Q_"tr")$，定义
$
  tilde(Pi)_("tr",Q_"tr") bold(tau)
  := bold(tau)-1/d (
    1/Q_"tr" sum_(r=1)^(Q_"tr") tr(bold(tau)(bold(y)_r))
  ) bold(I).
$
实现中先计算每个原始应力特征在这组点上的样本均值，再消去一个常数球应力自由度。因此离散系数对该独立规则精确满足零平均迹规范，与连续规范之间的偏差则由下述 $epsilon_"trquad"$ 度量。记 Ritz 内积误差为 $epsilon_"Rquad"$，平均迹积分误差为 $epsilon_"trquad"$，并合记
$
  epsilon_"projquad"^2
  := epsilon_"Rquad"^2+epsilon_"trquad"^2.
$
其中 $epsilon_"trquad"$ 具体控制系数球上一致的投影差
$
  sup_(bold(c) in cal(C)_(N,B))
  norm(
    (tilde(Pi)_("tr",Q_"tr")-Pi_"tr")
    hat(bold(tau))_bold(c)
  )_(bold(H)(div));
$
该差是常数球张量，故其散度为零，只需控制独立 Monte Carlo 样本均值与连续平均值之差。分析时先将经验投影场与精确 $Pi_"tr"$ 投影场比较，再对后者应用 @thm:elasticity-stability。这正是平均迹积分误差必须进入 $epsilon_"projquad"$ 的原因。

#theorem(title: [平均迹投影积分误差])[
  在系数约束 $cal(C)_(N,B)$ 下，平均迹投影积分误差满足
  $
    (EE_"tr" epsilon_"trquad"^2)^(1\/2)
    <= C_"tr" B Q_"tr"^(-1\/2),
  $
  其中期望只对平均迹规则取，$C_"tr"$ 仅依赖于 $abs(Omega)$、$d$ 与原始应力特征的一致 $L^oo$ 上界，不依赖于 $N$、$K$、$Q_"tr"$ 与 $B$。
]<thm:trquad-rate>

#proof[
  两投影只在被减去的常数球张量上不同：
  $
    (tilde(Pi)_("tr",Q_"tr") - Pi_"tr") hat(bold(tau))_(bold(c))
    = - 1/d Delta(bold(c)) bold(I),
    quad
    Delta(bold(c))
    := 1/Q_"tr" sum_(r=1)^(Q_"tr") tr(hat(bold(tau))_(bold(c))(bold(y)_r))
    - 1/abs(Omega) integral_Omega tr(hat(bold(tau))_(bold(c))) dif x.
  $
  常数球张量的散度为零，且 $norm(bold(I))_(L^2(Omega)) = sqrt(d abs(Omega))$，故投影差的 $bold(H)(div)$ 范数就是 $L^2$ 范数：
  $
    norm((tilde(Pi)_("tr",Q_"tr") - Pi_"tr") hat(bold(tau))_(bold(c)))_(bold(H)(div))
    = 1/d abs(Delta(bold(c))) norm(bold(I))_(L^2(Omega))
    = sqrt(abs(Omega)/d) abs(Delta(bold(c))),
  $
  从而 $epsilon_"trquad" = sqrt(abs(Omega)\/d) sup_(bold(c) in cal(C)_(N,B)) abs(Delta(bold(c)))$。由应力块的展开 $tr hat(bold(tau))_(bold(c)) = sum_(j,alpha) c_(j,alpha) tr(bold(T)_alpha) xi_(bold(sigma),j)$，$Delta$ 关于 $bold(c)$ 线性：
  $
    Delta(bold(c)) = sum_(j,alpha) c_(j,alpha) tr(bold(T)_alpha) Delta_j,
    quad
    Delta_j := 1/Q_"tr" sum_(r=1)^(Q_"tr") xi_(bold(sigma),j)(bold(y)_r)
    - 1/abs(Omega) integral_Omega xi_(bold(sigma),j) dif x,
  $
  即每个应力特征的迹均值误差 $Delta_j$ 与系数的内积。对该内积用 Cauchy--Schwarz，并注意基 ${bold(T)_alpha}$ 正交归一给出 $sum_alpha tr(bold(T)_alpha)^2 = sum_alpha (bold(T)_alpha, bold(I))_F^2 = norm(bold(I))_F^2 = d$，于是在系数球上
  $
    sup_(bold(c) in cal(C)_(N,B)) abs(Delta(bold(c)))
    <= B/sqrt(m_N) (sum_(j,alpha) tr(bold(T)_alpha)^2 Delta_j^2)^(1\/2)
    = B sqrt(d/m_N) (sum_(j=1)^N Delta_j^2)^(1\/2).
  $
  最后对每个 $Delta_j$ 作方差估计：${bold(y)_r}$ 为独立均匀样本，故 $EE Delta_j = 0$，且
  $
    EE Delta_j^2
    = 1/Q_"tr" op("Var")(xi_(bold(sigma),j)(bold(y)_1))
    <= C_xi^2/Q_"tr",
    quad
    C_xi := max_(1<=j<=N) norm(xi_(bold(sigma),j))_(L^oo(Omega)),
  $
  其中 $C_xi <= (sup_(bold(x) in overline(Omega)) norm(bold(x))_(ell^2) + c_b)^k$ 是与 $N$ 无关的常数。综合三式，并用应力特征数 $N <= m_N$，得对平均迹规则的二阶矩界
  $
    (EE epsilon_"trquad"^2)^(1\/2)
    <= C_xi sqrt(abs(Omega) N/m_N) B Q_"tr"^(-1\/2)
    lt.tilde B Q_"tr"^(-1\/2),
  $
  即得结论。证毕。
]

@thm:trquad-rate 适用于一般区域。本文的计算区域是单位盒，$rho_k$ 的脊结构又使每个原始应力特征的盒上均值有闭式，因此该项可精确消去而非仅仅控制其阶。

为区分盒上均值的计算分支，对 $bold(omega) in bb(S)^(d-1)$ 局部记
$
  b_-(bold(omega)) := - max_(bold(x) in overline(Omega)) bold(omega) dot bold(x),
  quad
  b_+(bold(omega)) := - min_(bold(x) in overline(Omega)) bold(omega) dot bold(x),
$
并记 $cal(P)_Omega^"nd" := {(bold(omega),b):b_-(bold(omega))<b<b_+(bold(omega))}$。这些记号只用于以下积分公式：区间内部仿射函数在盒上变号，两端则分别给出零函数与普通多项式。

#corollary(title: [盒状区域上的精确平均迹投影])[
  设 $Omega = (0,1)^d$、$d in {2,3}$。对 $(bold(omega), b) in cal(P)_Omega^"nd"$，记 $n$ 为 $bold(omega)$ 中非零分量的个数、$tilde(b) := b + sum_(omega_i < 0) omega_i$ 为反射后的偏置，则
  $
    integral_Omega rho_k (bold(omega) dot bold(x) + b) dif bold(x)
    = (k!)/((k+n)! product_(omega_i != 0) abs(omega_i))
    sum_(bold(v) in {0,1}^n) (-1)^(n - abs(bold(v)))
    rho_1 (tilde(b) + sum_(i) abs(omega_i) v_i)^(k+n),
  $
  <eq:box-mean>
  其中 $rho_1 (t) = max(t, 0)$ 为 $k = 1$ 时的激活。退化参数的两种情形分别取 $k$ 次多项式在盒上的精确均值与零。以 @eq:box-mean 的精确均值构造 $tilde(Pi)_("tr",Q_"tr")$ 中被减去的常数球张量，则
  $
    tilde(Pi)_"tr" = Pi_"tr" quad "在" hat(bold(Sigma))_("LE",N) "上",
    quad
    epsilon_"trquad" = 0.
  $
]<cor:exact-trace>

#proof[
  先设 $(bold(omega), b) in cal(P)_Omega^"nd"$。对 $omega_i < 0$ 的坐标作反射 $x_i |-> 1 - x_i$，盒不变而仿射函数化为 $sum_i abs(omega_i) x_i + tilde(b)$。零分量对应的坐标不出现在被积函数中，积分消去后只剩 $n$ 个活动方向。对这 $n$ 个方向逐一取原函数：每积一次使 $rho_k$ 的幂次增加 $1$，并除以相应的 $abs(omega_i)$ 与新幂次，$n$ 次之后得到幂次 $k+n$、分母 $product abs(omega_i) dot (k+n)!\/k!$，而按盒的顶点取值作容斥即得 @eq:box-mean 的交错和。

  再考虑其余参数。由 $b_-$、$b_+$ 的定义，$b >= b_+(bold(omega))$ 时特征等于 $k$ 次多项式 $(bold(omega) dot bold(x) + b)^k$；$b <= b_-(bold(omega))$ 时特征在 $Omega$ 上恒为零，均值为零。第一种情形当然也可代入 @eq:box-mean，但该式中的交错和在被积函数不变号时会出现严重相消。实现中改用每轴 $ceil((k+1)\/2)$ 点的张量 Gauss--Legendre 规则，该规则对 $k$ 次多项式精确，故均值仍无求积误差。

  最后由 $tr hat(bold(tau))_bold(c) = sum_(j,alpha) c_(j,alpha) tr(bold(T)_alpha) xi_(bold(sigma),j)$ 关于 $bold(c)$ 线性，@thm:trquad-rate 证明中的 $Delta(bold(c)) = sum_(j,alpha) c_(j,alpha) tr(bold(T)_alpha) Delta_j$ 在每个 $Delta_j$ 为零时对一切 $bold(c)$ 为零，故两个投影在 $hat(bold(Sigma))_("LE",N)$ 上重合，$epsilon_"trquad" = sqrt(abs(Omega)\/d) sup_bold(c) abs(Delta(bold(c))) = 0$。证毕。
]

下面估计经验 Ritz 投影误差。以 $hat(v)_bold(c)$ 统一表示需要施加本质边界投影的原始场：线弹性中为逐分量投影的位移场且 $m=1$，板弯曲中为挠度且 $m=2$。定义
$
  epsilon_"Rquad"
  := sup_(bold(c) in cal(C)_(N,B))
  norm(
    (tilde(Pi)_(D,K,Q_"R")^(m) - Pi_(D,K)^(m))
    hat(v)_bold(c)
  )_(H^m(Omega)).
$
取 $V_K^(m)$ 的任意 $H^m$ 正交归一基
${phi_k}_(k=1)^K$，令经验 Gram 矩阵与其谱稳定事件为
$
  (bold(G)_(K,Q_"R"))_(i j)
  := (phi_i,phi_j)_(H^m,Q_"R"),
  quad
  cal(E)_("R",delta)
  := {norm(bold(G)_(K,Q_"R")-bold(I)_K)_2 <= delta}.
$
相应的广义杠杆包络记为
$
  C_("R",K)^(m)
  := abs(Omega) op("ess sup")_(bold(x) in Omega)
  sup_(0 != z_K in V_K^(m))
  norm(cal(L)_m (bold(x))z_K)_(ell^2)^2 / norm(z_K)_(H^m(Omega))^2.
$
此处使用 $C_("R",K)^(m)$ 而非通常的 Christoffel 函数记号，以避免与后文其他数学量冲突。

#theorem(title: [经验 Ritz 投影积分误差])[
  设 $0<delta<1$、$0<eta<1$。若
  $
    Q_"R"
    >= C_delta C_("R",K)^(m) log(2 K/eta),
  $
  其中 $C_delta>0$ 仅依赖于 $delta$，则
  $
    bb(P)(cal(E)_("R",delta)) >= 1-eta,
  $
  且条件于该谱稳定事件有
  $
    (
      EE_"R" [epsilon_"Rquad"^2 | cal(E)_("R",delta)]
    )^(1\/2)
    <= C_(delta,eta,xi,m) B
    sqrt(C_("R",K)^(m)/Q_"R").
  $
  对本文的准均匀张量积 $p$ 次 B 样条空间，固定 $m in {1,2}$ 时
  $
    C_("R",K)^(m) <= C_("inv",m) h_K^(-d) lt.tilde K,
  $
  从而
  $
    (
      EE_"R" [epsilon_"Rquad"^2 | cal(E)_("R",delta)]
    )^(1\/2)
    lt.tilde B sqrt(K/Q_"R").
  $
  隐含常数不依赖于 $N$、$K$、$Q_"R"$ 与 $B$。
]<thm:rquad-rate>

#proof[
  以下只在 $cal(E)_("R",delta)$ 上使用经验投影。在其补集上如何定义投影差不影响条件期望。
  随机离散最小二乘投影的 Gram 谱稳定性由最大 Christoffel 或杠杆函数控制，标量点值采样可参见 @CohenDavenportLeviatan2013，任意向量值线性采样与 Sobolev 梯度采样的统一框架可参见 @Adcock2025。将后者的采样算子取为 $cal(L)_m (bold(x))$，矩阵浓缩即给出所述样本条件与
  $bb(P)(cal(E)_("R",delta)) >= 1-eta$。

  以下给出投影差的二阶矩估计。对固定原始场 $hat(v)$，记
  $r:=(I-Pi_(D,K)^(m))hat(v)$。连续正交性给出
  $(r,phi_k)_(H^m)=0$。将 $tilde(Pi)_(D,K,Q_"R")^(m) hat(v)$ 与 $Pi_(D,K)^(m) hat(v)$ 在基 ${phi_k}_(k=1)^K$ 下的系数向量分别记为 $bold(a)^"emp"$ 与 $bold(a)^"ex"$，则
  $
    bold(a)^"emp"-bold(a)^"ex"
    = bold(G)_(K,Q_"R")^(-1) bold(d)_(Q_"R"),
    quad
    (bold(d)_(Q_"R"))_k
    := (r,phi_k)_(H^m,Q_"R"),
  $
  且 $EE_"R" bold(d)_(Q_"R")=bold(0)$。按 $C_("R",K)^(m)$ 的定义与样本独立性，
  $
    EE_"R" norm(bold(d)_(Q_"R"))_(ell^2)^2
    <= C_("R",K)^(m)/Q_"R" norm(r)_(H^m(Omega))^2.
  $
  这正是 @CohenDavenportLeviatan2013 定理 2 证明中经验投影与连续投影之差的估计对 $H^m$ 向量值采样的对应形式。

  再对系数球取一致上确界。记需投影场的分量数为 $n_v$：线弹性中 $n_v=d$，板弯曲中 $n_v=1$。令 $xi_(v,j)$ 为相应原始标量特征。用系数球的半径 $B/sqrt(m_N)$、矩阵算子范数不超过 Frobenius 范数，并在 $cal(E)_("R",delta)$ 上使用
  $norm(bold(G)_(K,Q_"R")^(-1))_2 <= (1-delta)^(-1)$，得到
  $
    EE_"R" [epsilon_"Rquad"^2 1_(cal(E)_("R",delta))]
    <= B^2 C_("R",K)^(m) /(
    m_N (1-delta)^2 Q_"R"
    )
    sum_(i=1)^(n_v) sum_(j=1)^N
    norm((I-Pi_(D,K)^(m))xi_(v,j))_(H^m(Omega))^2.
  $
  由于正交投影非扩张，右端每个残差不超过
  $norm(xi_(v,j))_(H^m)$。$rho_k$ 的参数域有界，故这些特征的 $H^m$ 范数关于 $j$ 与 $N$ 一致有界。再用 $n_v N<=m_N$，并以
  $bb(P)(cal(E)_("R",delta)) >= 1-eta$ 除以事件概率，即得条件二阶矩界。

  最后，固定次数、准均匀网格上的张量积 B 样条满足局部 Sobolev 逆估计（参见 @Schumaker2007 与 @TakacsTakacs2016）
  $
    abs(Omega) norm(cal(L)_m (bold(x))z_K)_(ell^2)^2
    <= C_("inv",m) h_K^(-d) norm(z_K)_(H^m(Omega))^2.
  $
  因此 $C_("R",K)^(m) lt.tilde h_K^(-d) tilde.eq K$，代回即得最后一式。证毕。
]

= 经验最小二乘与误差分析

== 训练泛函

令 ${bold(x)_ell}_(ell=1)^Q$ 为 $Omega$ 上独立均匀样本，并设 $abs(Omega)$ 为区域测度，两个模型的经验泛函分别如下。

对线弹性，取 $bold(z)=(bold(tau),bold(v)) in bold(X)_"LE"$，将本构残差与平衡残差依次记为
$
  bold(r)_("LE","c") (bold(z))
  :=bold(cal(A))_"LE" bold(tau)-bold(epsilon)(bold(v)),
  quad
  bold(r)_("LE","e") (bold(z))
  :=div bold(tau)+bold(f).
$
经验泛函为
$
  cal(J)_("LE",Q)(bold(z))
  :=abs(Omega)/Q sum_(ell=1)^Q (
    norm(bold(r)_("LE","c") (bold(z))(bold(x)_ell))_F^2
    +norm(bold(r)_("LE","e") (bold(z))(bold(x)_ell))_(ell^2)^2
  ).
$
对板弯曲，取 $bold(z)=(bold(tau),v) in bold(X)_"KL"$，其中第二分量为标量挠度，相应残差为
$
  bold(r)_("KL","c") (bold(z))
  :=bold(cal(A))_"KL" bold(tau)-bold(kappa)(v),
  quad
  r_("KL","e") (bold(z)):=div div bold(tau)+f,
$
此时平衡残差是标量，经验泛函为
$
  cal(J)_("KL",Q)(bold(z))
  :=abs(Omega)/Q sum_(ell=1)^Q (
    norm(bold(r)_("KL","c") (bold(z))(bold(x)_ell))_F^2
    +abs(r_("KL","e") (bold(z))(bold(x)_ell))^2
  ).
$
两类模型各自将全部加权残差写成设计矩阵 $bold(A)_(N,Q)$ 与右端
$bold(b)_(N,Q)$ 后，训练问题是
$
  min_(bold(c) in cal(C)_(N,B))
  norm(bold(A)_(N,Q) bold(c)-bold(b)_(N,Q))_(ell^2)^2.
$
<eq:train-problem>
其中 $bold(c) in RR^(m_N)$ 是该模型的输出系数向量，约束集为 @eq:coeff-ball 定义的系数球
$cal(C)_(N,B)={bold(c) in RR^(m_N):norm(bold(c))_(ell^2)<=B\/sqrt(m_N)}$。
因此 @eq:train-problem 是欧氏球上的凸最小二乘问题，其求解见 @sec:ball-solve。

== 单隐层 Sobolev 逼近误差

本节设各物理变量的参数带点集满足 @def:quasi-uniform，并对其标量字典逐分量应用 @cor:quasi-uniform-rate。其中的饱和指数为
$
  s_"cap" (d) = (d+2k+1)/2.
$

对每个物理变量 $chi$，取正则指数 $s_chi in [m, s_"cap" (d)]$ 使 $chi$ 的各分量属于 $H^(s_chi)(Omega)$，其中 $m$ 为相应图范数所含的最高导数阶：线弹性 $m = 1$，板弯曲 $m = 2$。解更光滑时取 $s_chi = s_"cap" (d)$，逼近阶在该处饱和。对线弹性定义
$
  beta_"LE"
  := min_(chi in {bold(sigma),bold(u)}) (s_chi-1)/d,
$
对二维板问题定义
$
  beta_"KL"
  := min_(chi in {bold(M),w}) (s_chi-2)/2.
$
上式中减去的 $1$ 或 $2$ 即 $m$。由 $norm(div bold(tau))_(L^2(Omega)) lt.tilde norm(bold(tau))_(H^1(Omega))$ 与 $norm(div div bold(tau))_(L^2(Omega)) lt.tilde norm(bold(tau))_(H^2(Omega))$，两类图范数分别受逐分量 $H^1$ 与 $H^2$ 范数控制，逐分量 $H^m$ 逼近率因此直接给出图范数逼近率。

将 @cor:quasi-uniform-rate 施于变量 $chi$ 的每个标量分量 $v_chi in H^(s_chi)(Omega)$，即得原始字典近似 $hat(v)_N := sum_(j=1)^N c_j xi_(chi,j)$ 满足
$
  norm(v_chi - hat(v)_N)_(H^m (Omega)) & lt.tilde N^(-(s_chi - m)\/d) norm(v_chi)_(H^(s_chi)(Omega)), \
            (sum_(j=1)^N c_j^2)^(1\/2) & lt.tilde N^((s_"cap" (d) - s_chi)\/d - 1\/2)
                                         norm(v_chi)_(H^(s_chi)(Omega)),
$
其中 $(c_j)_(j=1)^N$ 是单个标量分量在字典 ${xi_(chi,j)}_(j=1)^N$ 上的系数。对各物理变量取最慢的收敛阶，整体字典逼近误差因而为 $N^(-beta)$ 阶，其中 $beta=beta_"LE"$ 或 $beta_"KL"$。在解足够光滑、逼近阶在饱和处取到的情形下，将 $s_chi = s_"cap" (d)$ 代入即得本文反复使用的显式收敛率
$
  beta(k, d, m) = (s_"cap" (d) - m)/d = (d + 2k + 1 - 2m)/(2 d).
$
<eq:beta-rate>
该式给出两个理论基准：
1. $k$ 每增加一个单位，指数 $beta$ 提高 $1\/d$；
2. 误差范数每多一阶导数，指数降低 $1\/d$。

因此 $L^2$ 与 $H^m$ 逼近误差界的指数相差 $m\/d$。这些关系属于函数类上的理论上界，该上界是确定性的：既无独立抽样带来的对数修正，也无需对参数取期望。

由上述系数界，预算取 $B gt.tilde N^((s_"cap" (d) - min_chi s_chi)\/d)$ 即足以容纳达到该逼近阶的比较函数。当最不光滑分量满足 $s_chi = s_"cap" (d)$ 时，$B$ 可取与 $N$ 无关的常数。

== Monte Carlo 训练误差

本节的误差分解与 Rademacher 估计遵循 @SiegelHongJinHaoXu2023。@LiuMaoXu2025 第 7 节对本文所用的同一线性化网络类（同样的 $ell^2$ 系数球）给出了能量泛函情形的对应结论。与两者的差别在于本文对本质边界条件采用投影：投影后的位移与挠度特征不再是脊函数，脊函数字典的标准 Rademacher 界（@lem:ridge-rademacher）不能直接施于该部分，需分块处理。

脊结构的失效只妨碍训练泛化误差，不妨碍确定性逼近项：真解本身属于 $H_0^m (Omega)$，故 $Pi_D^(m) v_star = v_star$，@eq:projection-transfer 的传递估计即以连续投影的非扩张性保证投影后的比较函数保留原始字典逼近阶。一致偏差则须对系数球中的一切 $bold(c)$ 成立，而经验泛函取的是投影后特征的至多 $m$ 阶导数在样本点处的逐点值，故须一致控制投影后特征的点态导数类，这不能仅由 $H^m$ 非扩张性得到。障碍有两层：
- 其一，投影前后的两个函数类一般并不互相接近。取参数带中任一使仿射函数 $bold(omega) dot bold(x)+b$ 在 $overline(Omega)$ 上变号的特征，其最大值为正；由于梯度非零，最大值只在 $partial Omega$ 上取到。于是该特征 $xi$ 在 $partial Omega$ 的某点严格为正，迹不恒为零。$Omega$ 为 Lipschitz 区域，$H_0^m (Omega) subset H_0^1 (Omega)$ 中的元素迹为零，故 $xi in.not H_0^m (Omega)$，而 $H_0^m (Omega)$ 闭，从而
  $
    norm(xi - Pi_D^(m) xi)_(H^m (Omega))
    = op("dist")_(H^m (Omega)) (xi, H_0^m (Omega))
    > 0.
  $
  该距离由 $(bold(omega),b)$ 与 $Omega$ 决定，故投影不能作为原脊函数类上的恒等扰动处理。已投影块随之不再是脊函数族的像，@lem:ridge-rademacher 证明所依赖的参数覆盖数论证在其上无从施行。参数带中可能同时含有零函数或多项式特征，这不影响上述非多项式子族已经给出的障碍。
- 其二，$H^m$ 范数不控制点态值。投影的算子范数为 $1$ 只给出 $norm(Pi_D^(m) xi_(chi,j))_(H^m (Omega)) lt.tilde 1$，即仅知投影后的特征落在一个固定半径的 $H^m$ 球内。而两类残差取的都是 $m$ 阶导数的点值，$m$ 阶导数只属于 $L^2(Omega)$，该球对此不提供任何控制。取 $phi in C_c^oo(B_1)$ 使某个 $abs(bold(alpha)) = m$ 有 $partial^bold(alpha) phi(bold(0)) != 0$，以样本点 $bold(x)_ell$ 为中心作尺度化
  $
    v_eta (bold(x)) := eta^(m - d\/2) phi((bold(x) - bold(x)_ell)\/eta),
  $
  则换元给出 $norm(partial^bold(beta) v_eta)_(L^2(Omega)) = eta^(m - abs(bold(beta))) norm(partial^bold(beta) phi)_(L^2(B_1))$，故 $norm(v_eta)_(H^m (Omega)) <= norm(phi)_(H^m (B_1))$ 关于 $eta in (0,1]$ 一致，而 $partial^bold(alpha) v_eta (bold(x)_ell) = eta^(-d\/2) partial^bold(alpha) phi(bold(0))$ 随 $eta -> 0$ 无界。$eta$ 充分小时 $v_eta$ 的支集含于 $Omega$ 且不含其余样本点，故该球上的 Rademacher 复杂度对每个样本都是 $+oo$，仅以 $H^m$ 范数为依据的估计必然空洞。又 $v_eta in C_c^oo(Omega) subset H_0^m (Omega)$，边界约束不改善这一点。因此已投影块须另配至多 $m$ 阶导数族的一致 $L^oo$ 界，即下文的包络 $C_(Pi,K)$，其阶由 B 样条逆估计定出，这也是训练误差界依赖 $K$ 的来源。

本节用到三条外部估计，连同其在本文记号下的陈述转录于此，证明见所引文献。先定义 Rademacher 复杂度：对 $Omega$ 上的函数类 $cal(F)$，
$
  frak(R)_Q (cal(F))
  := EE_(bold(x)_ell) EE_(epsilon_ell)
  sup_(h in cal(F)) 1/Q sum_(ell=1)^Q epsilon_ell h(bold(x)_ell),
$
其中 ${bold(x)_ell}$ 为独立均匀样本，${epsilon_ell}$ 为独立 Rademacher 变量（各以 $1\/2$ 概率取 $plus.minus 1$）。

#lemma(title: [对称化（@SiegelHongJinHaoXu2023 定理 4）])[
  对 $Omega$ 上的函数类 $cal(F)$ 与独立均匀样本，
  $
    EE sup_(h in cal(F))
    abs(1/Q sum_(ell=1)^Q h(bold(x)_ell) - 1/abs(Omega) integral_Omega h dif x)
    <= 2 frak(R)_Q (cal(F)).
  $
]<lem:symmetrization>

#lemma(title: [脊字典导数族的 Rademacher 界（@SiegelHongJinHaoXu2023 定理 6）])[
  设 $sigma in W^(m+1,oo)(RR)$，参数域 $Theta subset RR^(d+1)$ 紧，脊字典为 $bb(D)_sigma := {sigma(bold(omega) dot bold(x) + b) : (bold(omega), b) in Theta}$。则其各阶导数族满足
  $
    frak(R)_Q (partial^bold(alpha) bb(D)_sigma) lt.tilde Q^(-1\/2),
    quad abs(bold(alpha)) <= m,
  $
  隐含常数与 $Q$ 无关。证明的关键是导数族保持脊结构：$partial^bold(alpha) sigma(bold(omega) dot bold(x) + b) = bold(omega)^bold(alpha) sigma^((abs(bold(alpha))))(bold(omega) dot bold(x) + b)$，其中 $bold(omega)^bold(alpha)$ 有界、$sigma^((abs(bold(alpha))))$ Lipschitz，函数类的覆盖数由紧参数域的覆盖数控制。
]<lem:ridge-rademacher>

#lemma(title: [二次损失类的结构性界（@SiegelHongJinHaoXu2023 定理 5）])[
  设损失类由字典 $bb(D)$ 上系数预算为 $M$ 的线性模型经二次损失复合而成：损失对模型输出及其至多 $m$ 阶导数是二次的，二次系数函数的 $L^oo$ 范数以 $K$ 为界，另含一个以右端 $bold(f)$ 为系数的线性项。若 $sup_(g in bb(D)) norm(g)_(W^(m,oo)(Omega)) <= C$，则损失类的 Rademacher 复杂度满足
  $
    frak(R)_Q (cal(L)_M)
    lt.tilde C K M sum_(abs(bold(alpha)) <= m) frak(R)_Q (partial^bold(alpha) bb(D))
    + norm(bold(f))_(L^oo(Omega)) M frak(R)_Q (bb(D)).
  $
  即损失类的复杂度由字典各阶导数族的复杂度装配而成：平方复合经 Lipschitz 收缩，为每项引入一个由 $C M$ 定界的因子。
]<lem:quadratic-structural>

分块由两个投影的作用范围给出：$tilde(Pi)_("tr",Q_"tr")$ 只作用于应力（弯矩）块，$tilde(Pi)_(D,K,Q_"R")^(m)$ 只作用于位移（挠度）块且逐分量作用，故后者只生成 $N$ 个标量函数，与分量指标无关。据此将投影基场的全部标量分量分成两族，线弹性为
$
  cal(B)_("LE",N,K) & := {(tilde(Pi)_("tr",Q_"tr") (xi_(bold(sigma),j) bold(T)_alpha))_(beta gamma):
                        1 <= j <= N, 1 <= alpha <= n_s, 1 <= beta, gamma <= d} \
                    & quad union {tilde(Pi)_(D,K,Q_"R")^(1) xi_(bold(u),j): 1 <= j <= N},
$
板弯曲相应为
$
  cal(B)_("KL",N,K) & := {(xi_(bold(M),j) bold(T)_alpha)_(beta gamma):
                        1 <= j <= N, 1 <= alpha <= 3, 1 <= beta, gamma <= 2} \
                    & quad union {tilde(Pi)_(D,K,Q_"R")^(2) xi_(w,j): 1 <= j <= N}.
$
以下以 $cal(B)_(N,K)$ 统记两者。条件于平均迹规则与 Ritz 规则，两个经验投影是确定性线性算子，$cal(B)_(N,K)$ 随之是确定的有限集。取其至多 $m$ 阶的导数族
$
  cal(D)_(N,K)
  := {partial^bold(alpha) g:
    g in cal(B)_(N,K), bold(alpha) in NN_0^d, abs(bold(alpha)) <= m},
$
其中 $m$ 为相应图范数所含的最高导数阶（线弹性 $m=1$，板弯曲 $m=2$）。残差用到的是其中的特定组合：线弹性为 $div bold(tau)$ 与 $bold(epsilon)(bold(v))$ 中的一阶导数，板弯曲为 $div div bold(tau)$ 与 $nabla^2 v$ 中的二阶导数。按全部 $abs(bold(alpha)) <= m$ 取族把这些组合的分量一并纳入，并与 @thm:rquad-rate 所用采样算子 $cal(L)_m$ 的导数指标集一致。该族的一致 $L^oo$ 界即包络
$
  C_(Pi,K) := sup_(g in cal(D)_(N,K))
  norm(g)_(L^oo(Omega)).
$
$cal(D)_(N,K)$ 亦有限，故该上确界即最大值。其取值随 $N$ 与 $K$ 共同变化，记号只标出 $K$，因为 @eq:envelope-rate 以 $K$ 定阶，其隐含常数关于 $N$ 一致。

应力（弯矩）块经 @lem:ridge-rademacher 估计，该引理要求激活的 $m$ 阶导数 Lipschitz。$rho_k$ 的 $k-1$ 阶导数 Lipschitz 而 $k$ 阶导数不连续，该要求对 $rho_k$ 即 $k >= m+1$，线弹性为 $k >= 2$，板弯曲为 $k >= 3$。

#theorem(title: [训练泛化误差])[
  沿用 @thm:rquad-rate 的谱稳定事件 $cal(E)_("R",delta)$，并条件于平均迹规则与 Ritz 规则。设 $k >= m+1$，$bold(f) in L^oo$（板问题为 $f in L^oo$），系数取自 $cal(C)_(N,B)$。则包络满足
  $
    C_(Pi,K) lt.tilde K^(1\/2),
  $
  <eq:envelope-rate>
  且平方风险层面的一致偏差满足
  $
    EE sup_(bold(c) in cal(C)_(N,B))
    abs(cal(J)(bold(c))-cal(J)_Q (bold(c)))
    lt.tilde (
      B^2 K + norm(bold(f))_(L^oo(Omega)) B sqrt(K)
    ) Q^(-1\/2),
  $
  隐含常数不依赖于 $N$、$Q$、$K$、$B$ 与 $lambda$。
]<thm:train-generalization>

#proof[
  第一步：分块。条件于两组投影规则后，$cal(D)_(N,K)$ 按 $cal(B)_(N,K)$ 的两个并项分为两族。未投影块由应力（弯矩）分量的至多 $m$ 阶导数组成：$tilde(Pi)_("tr",Q_"tr")$ 只减去一个由独立样本定出的常数球张量，条件化后是固定平移，其散度为零且不改变任何导数，故该族仍是脊函数族 $bb(D) = {rho_k (bold(omega) dot bold(x) + b) : (bold(omega), b) in bb(S)^(d-1) times [-c_b, c_b]}$ 的像；此处用到字典元素与参数一一对应，即每个 $xi_(chi,j)$ 都是 $bb(D)$ 的元素，无例外的常数特征。已投影块是位移（$m=1$）与挠度（$m=2$）特征，经 $tilde(Pi)_(D,K,Q_"R")^(m)$ 后落在 $V_K^(m)$ 中，是 B 样条而非脊函数。

  第二步：未投影块。由 $abs(bold(omega) dot bold(x)) <= R_Omega$ 与 $abs(b) <= c_b$，仿射自变量 $bold(omega) dot bold(x) + b$ 在 $overline(Omega)$ 上的取值含于有界区间 $I := [-(R_Omega + c_b), R_Omega + c_b]$。$rho_k in W^(k,oo)(I)$ 且 $k >= m+1$，故可取 $sigma in W^(m+1,oo)(RR)$ 使 $sigma = rho_k$ 于 $I$ 上，例如以一个在 $I$ 上恒为 $1$ 的 $C_c^oo(RR)$ 函数乘 $rho_k$。$sigma$ 与 $rho_k$ 在 $Omega$ 上生成同一字典 $bb(D)$，@lem:ridge-rademacher 的假设成立：
  $
    frak(R)_Q (partial^bold(alpha) bb(D)) lt.tilde Q^(-1\/2),
    quad abs(bold(alpha)) <= m,
  $
  隐含常数与 $Q$、$N$ 无关（但依赖于 $k$，因 $rho_k$ 各阶导数在 $I$ 上的 Lipschitz 常数随 $k$ 增大）。该块在两个残差中以柔度算子的分量与 ${bold(T)_alpha}$ 的分量为系数出现，前者以 $1\/mu$（板弯曲为 $6\/(mu h^3)$）为界且关于 $lambda$ 一致，故取有限个这样的线性组合不改变上述阶。

  第三步：已投影块。上式的推导对 B 样条无效，改用线性类的 Rademacher 界（@ShalevShwartzBenDavid2014 第 26.2 节）。该块在两个残差中只经 $bold(epsilon)$（板弯曲为 $nabla^2$）出现，故残差的每个标量分量在该块上形如 $scripts(sum)_p c_p a_p$，其中 $p$ 遍历该块的系数指标，$a_p$ 是 $cal(D)_(N,K)$ 中有限多个元素的线性组合，个数与组合系数都只依赖于 $d$，因而 $norm(a_p)_(L^oo(Omega)) lt.tilde C_(Pi,K)$。该块的假设类即向量 $(a_p (bold(x)))_p$ 与系数在半径 $B\/sqrt(m_N)$ 的 $ell^2$ 球上的内积，而该向量的 $ell^2$ 范数不超过 $C_(Pi,K) sqrt(m_N)$ 的常数倍。于是
  $
    frak(R)_Q lt.tilde (B/sqrt(m_N)) dot (C_(Pi,K) sqrt(m_N)) Q^(-1\/2)
    = B C_(Pi,K) Q^(-1\/2),
  $
  与 $N$ 无关：系数球的 $sqrt(m_N)$ 缩放恰好抵消特征维数的增长。

  第四步：包络定阶。设 $chi$ 为需投影的标量特征。由 $rho_k$ 的参数域有界，$norm(xi_(chi,j))_(H^m(Omega))$ 关于 $j$ 与 $N$ 一致有界。正交投影非扩张给出 $norm(Pi_(D,K)^(m) xi_(chi,j))_(H^m(Omega)) lt.tilde 1$，在 $cal(E)_("R",delta)$ 上经验投影亦然，相差因子 $(1-delta)^(-1)$。再用 @thm:rquad-rate 证明末尾的张量积 B 样条逆估计
  $
    abs(Omega) norm(cal(L)_m (bold(x)) z_K)_(ell^2)^2
    <= C_("inv",m) h_K^(-d) norm(z_K)_(H^m(Omega))^2,
  $
  取 $z_K = tilde(Pi)_(D,K,Q_"R")^(m) xi_(chi,j)$，得 $C_(Pi,K) lt.tilde h_K^(-d\/2) tilde.eq K^(1\/2)$，即 @eq:envelope-rate。未投影块的包络与 $K$ 无关，不改变该阶。

  第五步：装配。经验泛函在系数上是二次项加一个以 $bold(f)$ 为系数的线性项，具有 @lem:quadratic-structural 所处理的损失结构：二次系数函数由柔度算子分量与 ${bold(T)_alpha}$ 的分量组成，其 $L^oo$ 界关于 $lambda$ 一致；模型输出即各残差分量，其点值由包络与系数半径之积 $B C_(Pi,K)$ 一致控制（见第三步），该乘积即结构性界中的 $C M$。分别代入第二、三步的两个复杂度界，得
  $
    frak(R)_Q lt.tilde (
      B C_(Pi,K) + norm(bold(f))_(L^oo(Omega))
    ) B C_(Pi,K) Q^(-1\/2).
  $
  最后由 @lem:symmetrization 将 $frak(R)_Q$ 转为一致偏差的期望，并代入 @eq:envelope-rate，即得结论。逆估计常数 $C_("inv",m)$ 与 B 样条次数、网格准均匀性有关，与 Lamé 常数无关，故隐含常数关于 $lambda$ 一致。证毕。
]

因此，当最终误差以图范数的均方根表示时，仅由经验积分产生的项是 $Q^(-1\/4)$，而不是 $Q^(-1\/2)$。两种指数对应不同层次的量，必须加以区分。

== 总误差估计

令 $bold(z)_star$ 表示连续精确解。$bold(z)_(N,Q,K)$ 表示用 $N$ 个字典特征、$Q$ 个训练点与 $K$ 维 Ritz 辅助空间得到的物理解。以 $epsilon_"opt"$ 表示经验目标值意义下的代数求解次优性。下述两条定理分别对线弹性与板弯曲将四类误差合并为总误差估计：Ritz 截断、Ritz 投影积分、平均迹积分与训练泛化误差分别由 @eq:ritz-rate、@thm:rquad-rate、@thm:trquad-rate 与 @thm:train-generalization 取显式阶，余下的离散常数只有代数次优性 $epsilon_"opt"$。两类问题共享同一误差分解框架，差别仅在稳定性定理的来源与平均迹投影的有无。

#theorem(title: [线弹性线性化网络经验最小二乘误差])[
  假设 @thm:elasticity-stability 成立，激活幂次 $k >= 2$，应力与位移的参数带点集 $Theta_(bold(sigma),N)^"band"$ 与 $Theta_(bold(u),N)^"band"$ 均满足 @def:quasi-uniform。设 $bold(sigma)_star$ 与 $bold(u)_star$ 的各分量分别属于 $H^(s_(bold(sigma)))(Omega)$ 与 $H^(s_(bold(u)))(Omega)$，$s_chi in [1, s_"cap" (d)]$，且预算 $B$ 足以容纳一个达到所述逼近阶的比较函数。固定 $0<delta,eta<1$，并设 $Q_"R"$ 满足 @thm:rquad-rate 的谱稳定样本条件。期望对训练样本、平均迹规则与 Ritz 规则取，其中 Ritz 规则条件于 $cal(E)_("R",delta)$，则
  $
    EE norm(bold(z)_star - bold(z)_(N,Q,K))_(bold(X)_"LE")^2
    lt.tilde & N^(-2 beta_"LE") \
             & + K^(-2(min(s_(bold(u)), p+1)-1)\/d) \
             & + B^2 K Q_"R"^(-1) \
             & + B^2 Q_"tr"^(-1) \
             & + (B^2 K + norm(bold(f))_(L^oo(Omega)) B sqrt(K)) Q^(-1\/2) \
             & + epsilon_"opt".
  $
]<thm:total-error-le>

其中 $B^2 Q_"tr"^(-1)$ 一项刻画一般区域上的平均迹积分误差。本文数值实验取 $Omega = (0,1)^d$ 并按 @cor:exact-trace 使用闭式盒上均值，故该项恒为零。

#proof[
  先条件于 Ritz 与平均迹两组相互独立的投影规则，以下确定性比较与训练样本期望均在该条件下进行。最后再分别对两组投影规则取期望，其中 Ritz 规则条件于 $cal(E)_("R",delta)$。

  第一步：系数到场的三个映射。对系数向量 $bold(c) in RR^(m_N)$，记原始字典场为 $hat(bold(z))(bold(c)) = (hat(bold(tau))_(bold(c)), hat(bold(v))_(bold(c)))$，并定义精确投影场与经验投影场
  $
    bold(z)(bold(c)) := (Pi_"tr" hat(bold(tau))_(bold(c)), Pi_(D,K)^(1) hat(bold(v))_(bold(c))),
    quad
    tilde(bold(z))(bold(c)) := (
      tilde(Pi)_("tr",Q_"tr") hat(bold(tau))_(bold(c)),
      tilde(Pi)_(D,K,Q_"R")^(1) hat(bold(v))_(bold(c))
    ).
  $
  记 $bold(c)^"out" in cal(C)_(N,B)$ 为求解器输出的系数，则计算所得物理解为 $bold(z)_(N,Q,K) = tilde(bold(z))(bold(c)^"out")$。原始特征逐分量属于 $W^(k,oo)(Omega) subset H^1(Omega)$，故上述诸场均属于 $bold(H)(div) times H^1(Omega; RR^d)$。下文将乘积图范数
  $
    norm(bold(w))_(bold(X))^2
    := norm(bold(tau))_(bold(H)(div))^2 + norm(bold(v))_(H^1(Omega))^2,
    quad bold(w) = (bold(tau), bold(v)),
  $
  视为整个 $bold(H)(div) times H^1$ 上的范数，$bold(X)_"LE"$ 是其带平均迹规范与边界约束的闭子空间。$Pi_"tr" hat(bold(tau))_(bold(c))$ 的平均迹为零，且 $Pi_(D,K)^(1) hat(bold(v))_(bold(c)) in V_K^(1) subset H_0^1(Omega; RR^d)$，故对一切 $bold(c)$ 有 $bold(z)(bold(c)) in bold(X)_"LE"$。经验投影场则一般不落在 $bold(X)_"LE"$ 中：$tilde(Pi)_(D,K,Q_"R")^(1)$ 的经验性只体现在 Gram 方程的数值积分上，输出仍是 $V_K^(1)$ 的元素，边界条件不受影响。但 $tilde(Pi)_("tr",Q_"tr") hat(bold(tau))_(bold(c))$ 与 $Pi_"tr" hat(bold(tau))_(bold(c))$ 相差一个常数球张量，其连续平均迹一般非零。这正是不能对 $tilde(bold(z))(bold(c)^"out")$ 直接引用 @thm:elasticity-stability、必须经由 $bold(z)(bold(c)^"out")$ 过渡的原因。在 @cor:exact-trace 的条件下该差为零，应力块无条件落在 $bold(Sigma)_"LE"$ 中，过渡随之多余。以下论证保持一般性。

  #figure(
    code-image(class: "center", theme => diagram(
      spacing: (44mm, 22mm),
      node((0, 0.5), [$hat(bold(z))(bold(c))$ \ #text(0.75em)[原始字典场]], name: <raw>),
      node((1, 0), [$bold(z)(bold(c)) in bold(X)_"LE"$ \ #text(0.75em)[精确投影场]], name: <ex>),
      node((1, 1), [$tilde(bold(z))(bold(c))$ \ #text(0.75em)[经验投影场]], name: <emp>),
      edge(<raw>, <ex>, $(Pi_"tr", Pi_(D,K)^(1))$, "->", stroke: 0.6pt + theme.main-color),
      edge(
        <raw>,
        <emp>,
        $(tilde(Pi)_("tr",Q_"tr"), tilde(Pi)_(D,K,Q_"R")^(1))$,
        "->",
        label-side: right,
        stroke: 0.6pt + theme.main-color,
      ),
      edge(
        <ex>,
        <emp>,
        $norm(dot)_(bold(X)) <= epsilon_"projquad"$,
        "<->",
        label-side: right,
        stroke: (paint: theme.main-color, thickness: 0.6pt, dash: "dashed"),
      ),
    )),
    caption: [同一系数 $bold(c)$ 到场的两条路径。上路施加精确平均迹投影与精确 Ritz 投影，其像落在 $bold(X)_"LE"$ 内，可对其引用 @thm:elasticity-stability；下路施加两个经验投影，得到训练泛函实际作用的场，其平均迹一般非零，故一般不落在 $bold(X)_"LE"$ 内。两路不交换，其差由 @eq:proj-gap 在系数球上一致控制。$bold(c) = bold(c)^"out"$ 时下路的像即计算所得物理解 $bold(z)_(N,Q,K)$，这正是必须经由 $bold(z)(bold(c)^"out")$ 过渡才能使用稳定性定理的原因],
  )<fig:least-squares-projection-paths>

  第二步：残差半范数及其双边控制。对 $bold(w) = (bold(tau), bold(v)) in bold(H)(div) times H^1$ 定义
  $
    abs(bold(w))_(cal(J))
    := cal(J)_"LE" (bold(w); bold(0))^(1\/2)
    = (
      norm(bold(cal(A))_"LE" bold(tau) - bold(epsilon)(bold(v)))_(L^2(Omega))^2
      + norm(div bold(tau))_(L^2(Omega))^2
    )^(1\/2).
  $
  映射 $bold(w) |-> (bold(cal(A))_"LE" bold(tau) - bold(epsilon)(bold(v)), div bold(tau))$ 是线性的，故 $abs(dot)_(cal(J))$ 是半范数，满足三角不等式。本构残差是场的线性函数，平衡残差是仿射函数，二者在精确解 $bold(z)_star = (bold(sigma)_star, bold(u)_star)$ 处同时为零，因此对任意 $bold(z) in bold(H)(div) times H^1$，
  $
    cal(J)_"LE" (bold(z); bold(f))
    = abs(bold(z) - bold(z)_star)_(cal(J))^2.
  $
  <eq:residual-shift>
  @thm:elasticity-stability 上界部分的证明只用到柔度算子的范数界与 $norm(bold(epsilon)(bold(v)))_(L^2(Omega)) <= norm(nabla bold(v))_(L^2(Omega))$，并未使用平均迹规范或边界条件，因此连续性在全空间成立：
  $
    abs(bold(w))_(cal(J)) lt.tilde norm(bold(w))_(bold(X)),
    quad forall bold(w) in bold(H)(div) times H^1(Omega; RR^d);
  $
  <eq:residual-continuity>
  强制性则按原定理只在约束子空间上成立：
  $
    norm(bold(w))_(bold(X)) lt.tilde abs(bold(w))_(cal(J)),
    quad forall bold(w) in bold(X)_"LE".
  $
  <eq:residual-coercivity>
  两式的适用范围不同，故不能合并为一个双边等价。以下隐含常数均由这两式的常数装配而成，其比值即 @thm:elasticity-stability 的稳定性比，按该定理可关于 $lambda$ 一致选取。

  第三步：投影积分误差的一致界。按前文约定，$epsilon_"trquad"$ 控制系数球上一致的平均迹投影差。与之对应，取 $epsilon_"Rquad"$ 为系数球上一致的经验 Ritz 投影差，即
  $
    sup_(bold(c) in cal(C)_(N,B))
    norm((tilde(Pi)_("tr",Q_"tr") - Pi_"tr") hat(bold(tau))_(bold(c)))_(bold(H)(div))
    <= epsilon_"trquad",
    quad
    sup_(bold(c) in cal(C)_(N,B))
    norm((tilde(Pi)_(D,K,Q_"R")^(1) - Pi_(D,K)^(1)) hat(bold(v))_(bold(c)))_(H^1(Omega))
    <= epsilon_"Rquad".
  $
  两个差分别只出现在应力分量与位移分量上，按乘积范数平方相加并用 $epsilon_"projquad"^2 = epsilon_"Rquad"^2 + epsilon_"trquad"^2$，得
  $
    sup_(bold(c) in cal(C)_(N,B))
    norm(tilde(bold(z))(bold(c)) - bold(z)(bold(c)))_(bold(X))
    <= epsilon_"projquad";
  $
  <eq:proj-gap>
  结合连续性 @eq:residual-continuity 又得一致的残差差
  $
    sup_(bold(c) in cal(C)_(N,B))
    abs(tilde(bold(z))(bold(c)) - bold(z)(bold(c)))_(cal(J))
    lt.tilde epsilon_"projquad".
  $
  <eq:proj-gap-residual>

  第四步：比较函数。由正则性假设，$bold(sigma)_star$ 的各分量属于 $H^(s_(bold(sigma)))(Omega)$，$bold(u)_star$ 的各分量属于 $H^(s_(bold(u)))(Omega)$，$s_chi in [1, s_"cap" (d)]$。在固定基 ${bold(T)_alpha}$ 与 ${bold(e)_i}$ 下将两个场逐分量分解，并对每个标量分量应用 @cor:quasi-uniform-rate（取 $m = 1$），得到原始字典场 $hat(bold(tau))^"cmp" in hat(bold(Sigma))_("LE",N)$ 与 $hat(bold(v))^"cmp" in hat(bold(U))_("LE",N)$，满足
  $
    norm(bold(sigma)_star - hat(bold(tau))^"cmp")_(H^1(Omega))
    lt.tilde N^(-(s_(bold(sigma))-1)\/d) norm(bold(sigma)_star)_(H^(s_(bold(sigma)))(Omega)),
    quad
    norm(bold(u)_star - hat(bold(v))^"cmp")_(H^1(Omega))
    lt.tilde N^(-(s_(bold(u))-1)\/d) norm(bold(u)_star)_(H^(s_(bold(u)))(Omega)),
  $
  其合并系数向量 $bold(c)^"cmp"$ 满足
  $
    norm(bold(c)^"cmp")_(ell^2)
    lt.tilde (
      N^((s_"cap"(d)-s_(bold(sigma)))\/d-1\/2)
      norm(bold(sigma)_star)_(H^(s_(bold(sigma)))(Omega))
      + N^((s_"cap"(d)-s_(bold(u)))\/d-1\/2)
      norm(bold(u)_star)_(H^(s_(bold(u)))(Omega))
    ).
  $
  又因 $m_N tilde.eq N$，上述系数界给出
  $
    sqrt(m_N) norm(bold(c)^"cmp")_(ell^2)
    lt.tilde N^((s_"cap" (d) - min(s_(bold(sigma)), s_(bold(u))))\/d)
    (
      norm(bold(sigma)_star)_(H^(s_(bold(sigma)))(Omega))
      + norm(bold(u)_star)_(H^(s_(bold(u)))(Omega))
    ),
  $
  预算假设正是保证 $B$ 不小于该量级，故 $bold(c)^"cmp" in cal(C)_(N,B)$。$bold(c)^"cmp"$ 由确定性字典与精确解决定，与训练样本无关。

  比较场取精确投影场 $bold(z)(bold(c)^"cmp")$，两个分量分别估计。先估计应力。由 $(bold(tau), bold(I))_(L^2(Omega)) = integral_Omega tr(bold(tau)) dif x$ 与 $norm(bold(I))_(L^2(Omega))^2 = d abs(Omega)$，
  $
    Pi_"tr" bold(tau)
    = bold(tau)
    - ((bold(tau), bold(I))_(L^2(Omega)))/(norm(bold(I))_(L^2(Omega))^2) bold(I),
  $
  即 $Pi_"tr"$ 是 $L^2(Omega; op("Sym")(d))$ 中向 $span {bold(I)}$ 的正交补作正交投影，$L^2$ 范数不增。被减去的常数球张量散度为零，故散度不变，$Pi_"tr"$ 在 $bold(H)(div)$ 图范数下也非扩张。$bold(sigma)_star in bold(Sigma)_"LE"$ 的平均迹为零，故 $Pi_"tr" bold(sigma)_star = bold(sigma)_star$，从而
  $
    norm(bold(sigma)_star - Pi_"tr" hat(bold(tau))^"cmp")_(bold(H)(div))
    = norm(Pi_"tr" (bold(sigma)_star - hat(bold(tau))^"cmp"))_(bold(H)(div))
    <= norm(bold(sigma)_star - hat(bold(tau))^"cmp")_(bold(H)(div))
    lt.tilde norm(bold(sigma)_star - hat(bold(tau))^"cmp")_(H^1(Omega)),
  $
  最后一步用了 $norm(div bold(w))_(L^2(Omega)) lt.tilde norm(bold(w))_(H^1(Omega))$。再估计位移。取精确有限维 Ritz 误差为
  $
    epsilon_"Ritz" (K) := norm((I - Pi_(D,K)^(1)) bold(u)_star)_(H^1(Omega)),
  $
  由 $Pi_(D,K)^(1)$ 是 $H^1$ 正交投影、算子范数为 $1$，
  $
    norm(bold(u)_star - Pi_(D,K)^(1) hat(bold(v))^"cmp")_(H^1(Omega))
    <= epsilon_"Ritz" (K)
    + norm(bold(u)_star - hat(bold(v))^"cmp")_(H^1(Omega)),
  $
  这是连续投影传递估计 @eq:projection-transfer 的有限维形式。两个分量平方相加并代入字典逼近率，得
  $
    norm(bold(z)(bold(c)^"cmp") - bold(z)_star)_(bold(X))^2
    lt.tilde N^(-2 beta_"LE") + epsilon_"Ritz" (K)^2,
  $
  <eq:comparison-bound>
  其中隐含常数依赖于 $Omega$、$c_b$、$d$ 与精确解分量的 Sobolev 范数，不依赖于 $N$、$Q$、$K$。

  第五步：经验极小性与一致 Monte Carlo 偏差。记条件于投影规则的一致偏差
  $
    delta_Q := sup_(bold(c) in cal(C)_(N,B))
    abs(
      cal(J)_"LE" (tilde(bold(z))(bold(c)); bold(f))
      - cal(J)_("LE",Q) (tilde(bold(z))(bold(c)))
    ).
  $
  经验泛函逐点求值所需的有界性由投影字典的点态导数包络提供。在 $bold(f)$ 有界与系数预算假设下，@thm:train-generalization 给出
  $
    EE delta_Q lt.tilde (
      B^2 K + norm(bold(f))_(L^oo(Omega)) B sqrt(K)
    ) Q^(-1\/2).
  $
  训练问题恰以 $bold(c) |-> cal(J)_("LE",Q)(tilde(bold(z))(bold(c)))$ 为目标在 $cal(C)_(N,B)$ 上极小化，而 $epsilon_"opt"$ 的定义即经验目标值的次优性，故对一切 $bold(c) in cal(C)_(N,B)$ 有 $cal(J)_("LE",Q)(tilde(bold(z))(bold(c)^"out")) <= cal(J)_("LE",Q)(tilde(bold(z))(bold(c))) + epsilon_"opt"$。特别取 $bold(c) = bold(c)^"cmp"$，这可行，因第四步已验证 $bold(c)^"cmp" in cal(C)_(N,B)$。于是
  $
    cal(J)_"LE" (tilde(bold(z))(bold(c)^"out"); bold(f))
    & <= cal(J)_("LE",Q) (tilde(bold(z))(bold(c)^"out")) + delta_Q \
    & <= cal(J)_("LE",Q) (tilde(bold(z))(bold(c)^"cmp")) + epsilon_"opt" + delta_Q \
    & <= cal(J)_"LE" (tilde(bold(z))(bold(c)^"cmp"); bold(f)) + 2 delta_Q + epsilon_"opt".
  $
  <eq:empirical-chain>

  第六步：装配。先控制 @eq:empirical-chain 右端的总体泛函。由 @eq:residual-shift、半范数的三角不等式、@eq:proj-gap-residual、@eq:residual-continuity 与 @eq:comparison-bound，
  $
    cal(J)_"LE" (tilde(bold(z))(bold(c)^"cmp"); bold(f)) & = abs(tilde(bold(z))(bold(c)^"cmp") - bold(z)_star)_(cal(J))^2 \
                                                         & <= (
                                                             abs(bold(z)(bold(c)^"cmp") - bold(z)_star)_(cal(J))
                                                             + abs(tilde(bold(z))(bold(c)^"cmp") - bold(z)(bold(c)^"cmp"))_(cal(J))
                                                           )^2 \
                                                         & lt.tilde norm(bold(z)(bold(c)^"cmp") - bold(z)_star)_(bold(X))^2
                                                           + epsilon_"projquad"^2 \
                                                         & lt.tilde N^(-2 beta_"LE")
                                                           + epsilon_"Ritz" (K)^2
                                                           + epsilon_"projquad"^2.
  $
  再从左端恢复图范数误差。$bold(z)(bold(c)^"out")$ 与 $bold(z)_star$ 都属于 $bold(X)_"LE"$，故 @eq:residual-coercivity 可用于其差。再由三角不等式、@eq:proj-gap-residual 与 @eq:residual-shift，
  $
    norm(bold(z)(bold(c)^"out") - bold(z)_star)_(bold(X))^2
    lt.tilde abs(bold(z)(bold(c)^"out") - bold(z)_star)_(cal(J))^2
    lt.tilde cal(J)_"LE" (tilde(bold(z))(bold(c)^"out"); bold(f))
    + epsilon_"projquad"^2.
  $
  联立上两式与 @eq:empirical-chain，并由 @eq:proj-gap 将精确投影场换回计算解本身，得
  $
    norm(bold(z)_star - bold(z)_(N,Q,K))_(bold(X))^2 & lt.tilde norm(bold(z)(bold(c)^"out") - bold(z)_star)_(bold(X))^2
                                                       + epsilon_"projquad"^2 \
                                                     & lt.tilde N^(-2 beta_"LE")
                                                       + epsilon_"Ritz" (K)^2
                                                       + epsilon_"projquad"^2
                                                       + delta_Q
                                                       + epsilon_"opt".
  $
  $bold(c)^"cmp"$、$epsilon_"Ritz" (K)$ 与 $epsilon_"projquad"$ 都与训练样本无关。将 $epsilon_"opt"$ 视为求解器给定的次优性上界。对训练样本取期望并代入第五步的 $EE delta_Q$ 界，先得条件于两组投影规则的估计
  $
    EE norm(bold(z)_star - bold(z)_(N,Q,K))_(bold(X))^2
    lt.tilde N^(-2 beta_"LE")
    + epsilon_"Ritz" (K)^2
    + epsilon_"Rquad"^2
    + epsilon_"trquad"^2
    + (B^2 K + norm(bold(f))_(L^oo(Omega)) B sqrt(K)) Q^(-1\/2)
    + epsilon_"opt",
  $
  隐含常数仅依赖于 $Omega$、$mu$、$d$、$c_b$ 与精确解各分量的 Sobolev 范数，不依赖于 $N$、$Q$、$K$、$B$ 与 $lambda$。由 $bold(u)_star$ 各分量属于 $H^(s_(bold(u)))(Omega)$ 与 @eq:ritz-rate，
  $
    epsilon_"Ritz" (K)^2
    lt.tilde K^(-2(min(s_(bold(u)), p+1)-1)\/d)
    norm(bold(u)_star)_(H^(s_(bold(u)))(Omega))^2.
  $
  最后依次对平均迹规则与条件于 $cal(E)_("R",delta)$ 的 Ritz 规则取期望，分别应用 @thm:trquad-rate 与 @thm:rquad-rate，得到
  $
    EE epsilon_"trquad"^2 lt.tilde B^2 Q_"tr"^(-1),
    quad
    EE_"R" [epsilon_"Rquad"^2 | cal(E)_("R",delta)]
    lt.tilde B^2 K Q_"R"^(-1).
  $
  代回条件估计即得定理。证毕。
]

板弯曲情形与线弹性逐步平行且更简单：无平均迹投影，故 $B^2 Q_"tr"^(-1)$ 一项不出现。

#theorem(title: [板弯曲线性化网络经验最小二乘误差])[
  假设 @thm:plate-stability 成立，激活幂次 $k >= 3$，弯矩与挠度的参数带点集 $Theta_(bold(M),N)^"band"$ 与 $Theta_(w,N)^"band"$ 均满足 @def:quasi-uniform。设 $bold(M)_star$ 与 $w_star$ 的各分量分别属于 $H^(s_(bold(M)))(Omega)$ 与 $H^(s_w)(Omega)$，$s_chi in [2, s_"cap" (2)]$，且预算 $B$ 足以容纳一个达到所述逼近阶的比较函数。固定 $0<delta,eta<1$，并设 $Q_"R"$ 满足 @thm:rquad-rate 的谱稳定样本条件。期望对训练样本与条件于 $cal(E)_("R",delta)$ 的 Ritz 规则取，则
  $
    EE norm(bold(z)_star - bold(z)_(N,Q,K))_(bold(X)_"KL")^2
    lt.tilde & N^(-2 beta_"KL") \
             & + K^(-(min(s_w, p+1)-2)) \
             & + B^2 K Q_"R"^(-1) \
             & + (B^2 K + norm(f)_(L^oo(Omega)) B sqrt(K)) Q^(-1\/2) \
             & + epsilon_"opt".
  $
]<thm:total-error-kl>

#proof[
  逐步重复 @thm:total-error-le 的证明，作如下替换。应力分量不设平均迹投影，取
  $
    bold(z)(bold(c)) := (hat(bold(tau))_(bold(c)), Pi_(D,K)^(2) hat(w)_(bold(c))),
    quad
    tilde(bold(z))(bold(c)) := (hat(bold(tau))_(bold(c)), tilde(Pi)_(D,K,Q_"R")^(2) hat(w)_(bold(c)));
  $
  由 $k >= 3$，原始特征逐分量属于 $W^(k,oo)(Omega) subset H^2(Omega)$，故诸场属于 $bold(H)(div div) times H^2(Omega)$。即 @fig:least-squares-projection-paths 中两条路径只作用于挠度分量，弯矩块沿两路都保持为 $hat(bold(tau))_(bold(c))$。$bold(Sigma)_"KL"$ 不带规范约束且 $V_K^(2) subset H_0^2(Omega)$，故精确投影场自动属于 $bold(X)_"KL"$。第二步以 @thm:plate-stability 替代 @thm:elasticity-stability 后逐字重复，其上界证明同样未用边界条件，连续性在全空间 $bold(H)(div div) times H^2$ 成立。第三步只出现 $epsilon_"Rquad" <= epsilon_"projquad"$。第四步删去 $Pi_"tr"$ 一段，弯矩图范数由逐分量 $H^2$ 范数经 $norm(div div bold(w))_(L^2(Omega)) lt.tilde norm(bold(w))_(H^2(Omega))$ 控制，Ritz 误差取 $epsilon_"Ritz" (K) := norm((I - Pi_(D,K)^(2)) w_star)_(H^2(Omega))$，由 $w_star in H^(s_w)(Omega)$ 与 @eq:ritz-rate 得 $epsilon_"Ritz" (K) lt.tilde K^(-(min(s_w, p+1)-2)\/2)$。@cor:quasi-uniform-rate（取 $m = 2$）给出字典逼近项 $N^(-2 beta_"KL")$。第五、六步不变。最后对条件于 $cal(E)_("R",delta)$ 的 Ritz 规则取期望并应用 @thm:rquad-rate，以 $B^2 K Q_"R"^(-1)$ 替换条件估计中的 $epsilon_"Rquad"^2$，即得结论。证毕。
]

由 @prop:layered-quasi-uniform，采用前述分层参数点集时，@thm:total-error-le 与 @thm:total-error-kl 关于参数带点集的假设自动成立。

若用均方根图误差表述 @thm:total-error-le 与 @thm:total-error-kl，则各项开方后分别具有量级
$
  N^(-beta),
  quad K^(-(min(s, p+1)-m)\/d),
  quad B sqrt(K/Q_"R"),
  quad B Q_"tr"^(-1\/2),
  quad (B^2 K + norm(bold(f))_(L^oo) B sqrt(K))^(1\/2) Q^(-1\/4),
  quad epsilon_"opt"^(1\/2).
$
其中线弹性取 $beta = beta_"LE"$、$(s, m) = (s_(bold(u)), 1)$，板弯曲取 $beta = beta_"KL"$、$(s, m) = (s_w, 2)$、$bold(f)$ 换作 $f$ 且无 $B Q_"tr"^(-1\/2)$ 一项。线弹性在盒状区域上取闭式盒上均值时该项亦为零（@cor:exact-trace）。Ritz 投影积分阶以 $Q_"R" >= C_delta C_("R",K)^(m) log(2K/eta)$ 的 Gram 谱稳定样本条件为前提。对准均匀 $p$ 次 B 样条，$C_("R",K)^(m) lt.tilde K$。训练项中的 $K$ 来自投影字典包络 $C_(Pi,K) lt.tilde K^(1\/2)$（@eq:envelope-rate），与 Ritz 投影积分项中的 $K$ 同源于同一条 B 样条逆估计。至此四类误差全部定阶，余下只有 $epsilon_"opt"$，即球约束凸最小二乘的经验次优性，在 @sec:ball-solve 的求解格式下主要由谱截断水平与乘子方程的求解容差产生。线弹性中的 Lamé 一致性还要求 $B$、目标各分量的 Sobolev 范数以及 $cal(E)_("R",delta)$ 上的 Ritz 谱稳定常数都能关于 $lambda$ 一致控制。包络 $C_(Pi,K)$ 的上述定阶只依赖于 B 样条次数与网格准均匀性，本身已关于 $lambda$ 一致。连续最小二乘稳定性本身的一致性不能自动推出其余离散统计量的一致性。

= 数值实验

本节首先固定激活幂次 $k=7$，考察完整离散解关于字典规模 $N$ 的观测收敛阶。随后固定 $N$，比较 $k in {3,5,7,9}$ 下完整经验最小二乘解的实际误差。最后在二维散度非零制造解上考察近不可压缩极限的稳定性。

== 实验设计

四个模型共用 @tbl:ls-design 的配置。材料参数、激活幂次、字典规模、辅助空间的次数与维数、两组求积的规模与系数预算取值序列逐模型取定。两类实验的配置表分开给出：关于 $N$ 的收敛实验列于各节开头，关于幂次 $k$ 的收敛实验另立一表。凡逐模型配置表中重复出现的配置项，均以该表为准。

#figure(
  three-line-table(
    columns: 2,
    align: (left, left),
  )[
    | 配置项 | 取值 |
    |---|---|
    | 计算区域 | $Omega=(0,1)^d$ |
    | 参数点集 | 单隐层参数取确定性分层点集，偏置半径 $c_b=2$ |
    | 抽样规则 | 训练求积与 Ritz 投影求积均取均匀 Monte Carlo 点，两组样本独立抽取 |
    | 系统装配 | 约化系统由流式 Householder QR 装配 |
    | 代数求解 | 截断奇异值分解；二维与板取 $epsilon_"cut"=10^(-14)$，三维取 $10^(-13)$ |
    | 验证规则 | 张量积 Gauss--Legendre，二维 $32^2$ 点、三维 $16^3$ 点；需要扫描系数预算时按其上的图误差选出 |
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

@eq:train-problem 的目标即经验泛函在系数上的表示，
$
  norm(bold(A)_(N,Q) bold(c)-bold(b)_(N,Q))_(ell^2)^2
  = cal(J)_("LE",Q)(tilde(bold(z))(bold(c))),
$
板弯曲将下标 $"LE"$ 换作 $"KL"$。因此代数次优性为
$
  epsilon_"opt"
  := cal(J)_("LE",Q)(tilde(bold(z))(bold(c)^"out"))
  - inf_(bold(c) in cal(C)_(N,B)) cal(J)_("LE",Q)(tilde(bold(z))(bold(c))).
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
    | 辅助空间 | $p=7$，$K in {441,841,1225,1681,2025}$ |
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
  caption: [固定 $k=7$ 时二维线弹性的应力与位移收敛曲线。虚线给出字典逼近指数 @eq:beta-rate 对应的参考斜率，误差棒为十次独立重复的样本标准差],
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

本实验对 $k in {3,5,7,9}$ 分别运行完整经验最小二乘算法，字典规模固定为 $N=501$。训练求积与辅助空间次数按 @eq:degree-match 与最高幂次 $k=9$ 相匹配，材料参数与关于 $N$ 的收敛实验相同，全部离散配置见 @tbl:ls-config-power-elasticity-2d。

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
    | 辅助空间 | $p=10$，$K=1024$ |
    | Ritz 投影求积 | $Q_"R"=12808$ |
    | 系数预算 | $B in {10^3,3 times 10^3,10^4,3 times 10^4,oo}$ |
  ],
  caption: [关于幂次 $k$ 的收敛实验：二维线弹性的配置],
)<tbl:ls-config-power-elasticity-2d>

#figure(
  image("/public/images/least-squares/campaign/convergence-power-elasticity-2d.png"),
  caption: [固定 $N=501$ 时二维线弹性完整经验最小二乘解的图范数误差随激活幂次的变化。误差棒表示十次独立重复的样本标准差，虚线是忽略依赖于 $k$ 的常数后的 $N^(-(beta(k)-beta(3)))$ 等常数启发线，不是固定 $N$ 下的定量预测],
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
    | 辅助空间 | $p=7$，$K in {441,841,1225,1681,2025}$ |
    | Ritz 投影求积 | $Q_"R"=12808$ |
    | 系数预算 | $B in {10^2,10^3,10^4,10^5,10^6,oo}$ |
  ],
  caption: [关于 $N$ 的收敛实验：平面应力的配置],
)<tbl:ls-config-order-plane-stress>

@fig:ls-convergence-order-plane-stress 给出固定 $k=7$ 时应力与位移图范数误差随 $N$ 的变化，@tbl:ls-order-plane-stress 列出各 $N$ 处的误差均值。

#figure(
  image("/public/images/least-squares/campaign/convergence-order-plane-stress.png"),
  caption: [固定 $k=7$ 时平面应力的应力与位移收敛曲线。虚线给出字典逼近指数 @eq:beta-rate 对应的参考斜率，误差棒为十次独立重复的样本标准差],
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
取 $bold(sigma)_star=2mu bold(epsilon)(bold(u)_star)+lambda div(bold(u)_star)bold(I)$，再由平衡方程制造体力 $bold(f)=-div bold(sigma)_star$。三维实验的配置见 @tbl:ls-config-order-elasticity-3d。与前述二维模型一致，本实验以 $N$ 为唯一自变量，训练点数 $Q$ 与 Ritz 辅助空间维数 $K$ 按固定比例随 $N$ 配套变化。

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
    | 辅助空间 | $p=7$，目标维数 $2N$，实际 $K in {512,1000,1331,1728,2197}$ |
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
  caption: [固定 $k=7$ 时三维线弹性的应力与位移收敛曲线。虚线给出字典逼近指数 @eq:beta-rate 对应的参考斜率，误差棒为十次独立重复的样本标准差],
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

五点对数最小二乘拟合给出的应力与位移观测阶分别为 $1.59$ 与 $1.36$。两条曲线在大 $N$ 端均不再保持参考线的斜率，其中位移误差从 $N=601$ 起处于 $10^(-9)$ 量级的平台。因此这组完整算法结果验证了三维离散流程随字典加密总体收敛，但不能据此声称达到了字典逼近指数给出的参考值 $beta(7, 3, 1)=8/3$。


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
    | 字典规模 | $N_sigma=N_u=N in {201,401,601,801,1001}$ |
    | 激活幂次 | $k=7$ |
    | 辅助空间 | 次数 $p=9$，目标维数 $2N$，实际 $K in {441,841,1225,1681,2025}$ |
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
    | 辅助空间 | $p=7$，$K in {441,841,1225,1681,2025}$ |
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
  caption: [固定 $k=7$ 时 Kirchhoff--Love 板弯曲的弯矩与挠度收敛曲线。虚线给出字典逼近指数 @eq:beta-rate 对应的参考斜率，误差棒为十次独立重复的样本标准差],
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

本实验对 $k in {3,5,7,9}$ 分别运行完整经验最小二乘算法，字典规模固定为 $N=1001$。训练求积与辅助空间次数按 @eq:degree-match 与最高幂次 $k=9$ 相匹配，材料参数与关于 $N$ 的收敛实验相同，全部离散配置见 @tbl:ls-config-power-plate。

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
    | 辅助空间 | $p=10$，$K=2025$ |
    | Ritz 投影求积 | $Q_"R"=12808$ |
    | 系数预算 | $B in {10^2,10^3,10^4,10^5,10^6,oo}$ |
  ],
  caption: [关于幂次 $k$ 的收敛实验：Kirchhoff--Love 板弯曲的配置],
)<tbl:ls-config-power-plate>

#figure(
  image("/public/images/least-squares/campaign/convergence-power-plate.png"),
  caption: [固定 $N=1001$ 时板弯曲完整经验最小二乘解的图范数误差随激活幂次的变化。误差棒表示十次独立重复的样本标准差，虚线是忽略依赖于 $k$ 的常数后的 $N^(-(beta(k)-beta(3)))$ 等常数启发线，不是固定 $N$ 下的定量预测],
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

误差界同时展示了 $N$、$Q$、$K$、$B$ 与代数求解精度的不同作用：$N$ 控制单隐层逼近，$K$ 控制投影空间，$Q$ 控制经验泛函，而 $B$ 连接逼近表示与泛化控制。特别地，平方风险中的 Monte Carlo 项是 $Q^(-1\/2)$，对应图范数均方根误差中的 $Q^(-1\/4)$。

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
