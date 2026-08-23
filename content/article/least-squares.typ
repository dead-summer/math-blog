#import "/typ/templates/blog.typ": *
#import fletcher: diagram, node, edge

#show: main-zh.with(
  title: "线性化网络混合最小二乘方法及其误差分析",
  author: "summer",
  desc: [证明线弹性与 Kirchhoff--Love 板弯曲混合最小二乘泛函与相应图范数平方双边等价，等价常数可关于 Lamé 常数一致选取；并将参数取准均匀点集的单隐层线性化 ReLU 三次幂网络用于两类方程的求解，给出误差分析],
  date: "2026-04-20",
  tags: (
    blog-tags.numerical-methods,
    blog-tags.pde,
  ),
  show-outline: true,
)

= 引言

线性化网络方法预先取定并冻结单隐层参数，随后只求输出层系数；冻结参数生成的特征族在逼近论中称为字典（dictionary），给定字典以后，偏微分方程离散仍是有限维凸二次问题。本文沿用 @SiegelHongJinHaoXu2023 的字典参数化，把单隐层参数取为方向--偏置参数域上的确定性准均匀点集，而非随机抽样。本文把这一结构用于线弹性应力--位移系统和 Kirchhoff--Love 板弯曲弯矩--挠度系统，连续问题采用经典的混合最小二乘泛函。

本文的核心结果是两条残差稳定性定理。其一，线弹性最小二乘泛函与 $bold(H)(div) times H^1$ 图范数平方双边等价（@thm:elasticity-stability）；其二，板弯曲最小二乘泛函与 $bold(H)(div div) times H^2$ 图范数平方双边等价（@thm:plate-stability）。两条定理的等价常数都可关于 Lamé 常数 $lambda$ 一致选取，因而覆盖近不可压缩极限 $lambda -> oo$。两类问题的一致性机制并不相同：线弹性依赖偏差--体积分解与零平均迹下的迹估计，板弯曲依赖柔度算子的一致椭圆性与 $H_0^2$ 上的 Poincaré 不等式。经典应力--位移最小二乘稳定性可参见 @CaiStarke2003、@CaiStarke2004 与 @DuanLin2005。

本文的另一项工作是将单隐层线性化 $"ReLU"^3$ 网络用于两类方程的数值求解，并给出误差分析：总误差分为字典逼近误差、有限维 Ritz 及其积分误差、训练泛函的 Monte Carlo 误差和线性代数求解误差四类。

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
本文以下标 $"LE"$（linear elasticity，线弹性）与 $"KL"$（Kirchhoff--Love 板弯曲）区分两类问题的算子、泛函与函数空间。使用 $(bold(sigma), bold(u))$ 表示线弹性未知量，使用 $(bold(M), w)$ 表示板弯曲未知量；$bold(T)_alpha$ 表示对称张量基。对非负量 $A$ 与 $B$，记 $A lt.tilde B$ 表示存在常数 $C > 0$ 使 $A <= C B$，其中 $C$ 不依赖于 Lamé 常数 $lambda$ 与任何离散参数，具体依赖在上下文说明；$A gt.tilde B$ 即 $B lt.tilde A$；记 $A tilde.eq B$ 表示 $A lt.tilde B$ 与 $B lt.tilde A$ 同时成立，即 $A$ 与 $B$ 同阶，其中两侧常数满足同一约定。

== 线弹性

线弹性的应力--位移系统为
$
  cases(
    bold(cal(A))_"LE" bold(sigma) - bold(epsilon)(bold(u)) = bold(0) & "in" Omega,
    div bold(sigma) + bold(f) = bold(0) & "in" Omega,
    bold(u) = bold(0) & "on" partial Omega.
  )
$
其中未知量为应力 $bold(sigma)$ 与位移 $bold(u)$，$bold(f) in L^2(Omega; RR^d)$ 为给定体力；第一个方程为本构关系，第二个方程为平衡方程，第三个方程为纯位移边界条件。$bold(epsilon)$ 为对称梯度算子，
$
  bold(epsilon)(bold(v)) := 1/2 (nabla bold(v) + nabla bold(v)^T).
$
$bold(cal(A))_"LE":op("Sym")(d)->op("Sym")(d)$ 为柔度算子，即各向同性刚度算子 $bold(tau) |-> 2 mu bold(tau) + lambda tr(bold(tau)) bold(I)$ 之逆，其中 $mu > 0$ 与 $lambda >= 0$ 为 Lamé 常数，$tr(bold(tau))$ 为迹，$bold(I)$ 为单位张量；显式地，
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

下述定理说明，强形式、极小化 $cal(J)_"LE"$ 与变分问题三种提法彼此等价，因此可以把 $cal(J)_"LE"$ 作为离散化的目标泛函。

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

@thm:elasticity-equivalence 只说明零残差刻画精确解，并未量化残差对误差的控制。下述稳定性定理补足这一环节，也是后文误差分析的出发点。

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

  再证下界，即关于 $lambda$ 一致有界的强制性。这里只在下界证明中引入偏差投影
  $
    bold(tau)^"D" := bold(tau) - 1/d tr(bold(tau)) bold(I),
    quad
    bold(epsilon)^"D" (bold(v)) := bold(epsilon)(bold(v)) - 1/d (div bold(v)) bold(I),
  $
  以及各向同性柔度算子的偏差-体积分解
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
  由 Korn 第一不等式（二维情形参考 @BrennerScott2008 推论 11.2.25，三维推广见其注 11.2.27），
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
  这里零平均迹条件是必要的；否则常数球应力 $bold(tau) = c bold(I)$ 会使右端为零而左端非零。
  结合 @eq:tau-dev-bound 即得
  $
    norm(tr(bold(tau)))_(L^2(Omega))^2
    lt.tilde norm(bold(r)_"dev")_(L^2(Omega))^2
    + norm(bold(r)_"equ")_(L^2(Omega))^2
    + norm(r_"vol")_(L^2(Omega))^2.
  $
  因此 $norm(bold(tau))_(L^2)^2$ 被残差平方和控制。将应力范数估计代回 @eq:korn，得到 $norm(bold(v))_(H^1)^2$ 的类似估计。最后结合
  $
    norm(bold(tau))_(bold(H)(div))^2
    = norm(bold(tau))_(L^2(Omega))^2
    + norm(bold(r)_"equ")_(L^2(Omega))^2
  $
  即得下界，其隐含常数仅依赖于 $Omega$、$mu$ 与 $d$。证毕。
]

由 @thm:elasticity-stability，双线性形式 $a_"LE"$ 在 $bold(Sigma)_"LE" times bold(U)_"LE"$ 上连续且强制，其中连续性与强制性常数均可关于 Lamé 常数 $lambda$ 一致选取。特别地，近不可压缩极限 $lambda -> oo$ 下稳定常数不退化；只要离散空间包含于 $bold(Sigma)_"LE" times bold(U)_"LE"$ 且其逼近误差本身保持稳定，离散解不会因该极限而产生锁死。平均迹规范排除了常值球应力核。

== Kirchhoff--Love 板弯曲

以下令 $d=2$。Kirchhoff--Love 板的弯矩--挠度系统为
$
  cases(
    bold(cal(A))_"KL" bold(M)-bold(kappa)(w)=bold(0) & "in" Omega,
    div div bold(M)+f=0 & "in" Omega,
    w=0 \, quad partial_n w=0 & "on" partial Omega.
  )
$
其中未知量为弯矩 $bold(M)$ 与挠度 $w$，$f in L^2(Omega)$ 为给定横向载荷；第一个方程为本构关系，第二个方程为平衡方程，第三个方程为固支边界条件。曲率张量固定记为
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
为弯曲刚度与对应三维各向同性材料的 Lamé 参数。下文固定 $mu > 0$ 与 $h > 0$，并只讨论物理参数区间 $lambda >= 0$；等价地，$0 <= nu < 1/2$。
定义
$
  bold(Sigma)_"KL" & := bold(H)(div div,Omega;op("Sym")(2)), \
            U_"KL" & := H_0^2(Omega), \
      bold(X)_"KL" & := bold(Sigma)_"KL" times U_"KL".
$
有关 $bold(H)(div div)$ 弯矩--挠度模型可参见 @FuhrerHeuerNiemi2019 与 @FuhrerHeuer2025；板弯曲的残差型最小二乘离散还可参见 @PontazaReddy2004。本文采用的最小二乘泛函为
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

与 @thm:elasticity-stability 平行，板弯曲的最小二乘泛函也双边控制相应图范数；一致性机制的差别见证明后的讨论。

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
  先将柔度算子写成偏差-迹分解。对任意 $bold(tau) in op("Sym")(2)$，记
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
  又因为 $bold(kappa)(v) = - nabla^2 v$，故
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
  由于 $v in H_0^2(Omega)$，有 $v in H_0^1(Omega)$ 且 $partial_i v in H_0^1(Omega)$，$i = 1, 2$。由 Poincaré 不等式（参考 @BrennerScott2008 命题 5.3.5），存在常数 $C_P > 0$，使得
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
  把 @eq:plate-h2 代入上式，并对右端使用 Young 不等式吸收含 $norm(bold(tau))_(L^2)^2$ 的项，可得
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

由 @thm:plate-stability，双线性形式 $a_"KL"$ 在 $bold(Sigma)_"KL" times U_"KL"$ 上连续且强制，其中常数可关于 Lamé 常数 $lambda$ 一致选取。值得对比的是：线弹性中柔度算子的体积部分在 $lambda -> oo$ 时退化，一致稳定性依赖偏差-体积分解与迹估计；板弯曲中柔度算子本身即一致椭圆，一致稳定性转而依赖 Poincaré 不等式从曲率恢复挠度范数。

= 线性化网络离散

== 单隐层字典

定义 ReLU 三次幂激活
$
  rho_3(t):=max(t, 0)^3.
$
方向--偏置的参数域取柱面 $bb(S)^(d-1) times [-c_b, c_b]$，其中 $bb(S)^(d-1) subset RR^d$ 为单位球面，偏置半径 $c_b$ 为取定常数，满足
$
  sup_(bold(omega) in bb(S)^(d-1), bold(x) in overline(Omega)) abs(bold(omega) dot bold(x))
  = sup_(bold(x) in overline(Omega)) norm(bold(x))_(ell^2) < c_b,
$
由 $Omega$ 有界，这样的 $c_b$ 存在，以下 $lt.tilde$ 中的常数允许依赖 $c_b$。记 $overline(Omega)$ 在方向 $bold(omega)$ 上的支撑区间端点为
$
  b_-(bold(omega)) := - max_(bold(x) in overline(Omega)) bold(omega) dot bold(x),
  quad
  b_+(bold(omega)) := - min_(bold(x) in overline(Omega)) bold(omega) dot bold(x),
$
并定义非退化参数集
$
  cal(P)_Omega := {(bold(omega), b) in bb(S)^(d-1) times [-c_b, c_b]:
    b_-(bold(omega)) < b < b_+(bold(omega))}.
$
$(bold(omega), b) in cal(P)_Omega$ 当且仅当仿射函数 $bold(omega) dot bold(x) + b$ 在 $overline(Omega)$ 上变号，等价地超平面 ${bold(omega) dot bold(x) + b = 0}$ 与 $overline(Omega)$ 相交。其余参数的特征在 $Omega$ 上退化：$b >= b_+(bold(omega))$ 时它等于三次多项式 $(bold(omega) dot bold(x) + b)^3$，$b <= b_-(bold(omega))$ 时它恒为零。记三次多项式空间为
$
  P_3 (Omega)
  := span {bold(x)^bold(alpha): bold(alpha) in NN_0^d, abs(bold(alpha)) <= 3},
  quad dim P_3 (Omega) = binom(d+3, 3),
$
退化参数的特征都落在其中。

以 $chi$ 标记物理变量：线弹性中 $chi in {bold(sigma), bold(u)}$，板弯曲中 $chi in {bold(M), w}$。每个物理变量独自生成单隐层参数，即各自取含 $N$ 个参数的确定性点集 $Theta_(chi,N) subset bb(S)^(d-1) times [-c_b, c_b]$，并按上述二分作不交分解
$
  Theta_(chi,N) = Theta^"poly" union.sq Theta_(chi,N)^"act", quad Theta^"poly" inter Theta_(chi,N)^"act" = emptyset,
$
其中 $Theta^"poly"$ 是退化区中固定的 $dim P_3 (Omega) - 1$ 个多项式方向，与 $chi$ 及 $N$ 都无关，与常数特征一起张成 $P_3 (Omega)$；$Theta_(chi,N)^"act" subset cal(P)_Omega$ 是其余 $N - dim P_3 (Omega) + 1$ 个参数，其构造在下文给出。因此 $abs(Theta_(chi,N)^"act") tilde.eq N$，下文以 $N$ 的幂次表述的量级换用 $abs(Theta_(chi,N)^"act")$ 表述时只差与 $N$ 无关的常数。把 $Theta_(chi,N)$ 的元素记为 $(bold(omega)_(chi,j), b_(chi,j))$，$1 <= j <= N$，变量 $chi$ 的特征及常数特征分别为
$
  xi_(chi,j) (bold(x)) := rho_3(bold(omega)_(chi,j) dot bold(x) + b_(chi,j)),
  quad xi_(chi,0) := 1.
$

现考虑 $Theta_(chi,N)$ 的构造。注意到特征 $xi_(chi,j)$ 只通过仿射自变量依赖参数：
$
  bold(omega) dot bold(x) + b = (bold(omega), b) dot tilde(bold(x)),
  quad tilde(bold(x)) := (bold(x), 1) in RR^(d+1),
$
故参数的实质是向量 $(bold(omega), b) in RR^(d+1)$。由 $rho_3$ 的正三次齐次性，该向量与它的任意正数倍给出同一个特征（至多相差一个正因子），因此参数只按方向计，归一化到单位球面 $bb(S)^d subset RR^(d+1)$ 不损失任何字典元素。@LiuMaoXu2025 的积分表示理论即建立在该球面上：$rho_3(bold(theta) dot tilde(bold(x)))$ 作为 $bold(theta) in bb(S)^d$ 的函数有球谐展开，其 Legendre 系数的衰减率给出后文的饱和指数 $s_"cap" (d)$，准均匀性也是对 $bb(S)^d$ 上的点集陈述的。

柱面参数域 $bb(S)^(d-1) times [-c_b, c_b]$ 是同一族特征的另一套坐标，便于刻画 $Omega$ 的几何：支撑端点 $b_(plus.minus)$ 与非退化参数集 $cal(P)_Omega$ 都以方向为自变量。两套坐标由归一化映射
$
  frak(P)(bold(omega), b) := ((bold(omega), b))/sqrt(1 + b^2)
$
转换；由 $norm(bold(omega))_(ell^2) = 1$ 有 $norm((bold(omega), b))_(ell^2) = sqrt(1 + b^2)$，故 $frak(P)$ 就是上述归一化在柱面上的限制。它将 $bb(S)^(d-1) times [-c_b, c_b]$ 双向 Lipschitz 地映为 $bb(S)^d$ 的赤道带 ${bold(theta) in bb(S)^d : abs(theta_(d+1)) <= c_b\/sqrt(1+c_b^2)}$，Lipschitz 常数仅依赖 $c_b$。具体地，
$
  xi_(chi,j) (bold(x)) = (1+b_(chi,j)^2)^(3/2) rho_3(frak(P)(bold(omega)_(chi,j), b_(chi,j)) dot tilde(bold(x))),
$
两种参数化张成同一空间，故 @LiuMaoXu2025 中球面上的结论可以逐条移到柱面上。

另一方面，@LiuMaoXu2025 注 2.1 把 $bb(S)^d$ 的两个极冠分离出来：其上的特征在 $Omega$ 上分别退化为三次多项式与零，故球面积分表示在极冠上的贡献落在 $P_3 (Omega)$ 中，由有限个多项式方向补足即可，只有中间带需要良分布点。论文中极冠由 $abs(theta_(d+1)) >= "diam"(Omega)\/sqrt(1 + "diam"(Omega)^2)$ 划定，与方向无关；本文的 $cal(P)_Omega$ 则逐方向取精确的非退化区间，其补集严格大于上述两个极冠。下述引理说明这一收紧不改变结论。

#lemma(title: [退化区分解])[
  记 $cal(G) := bb(S)^d without frak(P)(cal(P)_Omega)$。则对任意 $bold(theta) in cal(G)$，$rho_3(bold(theta) dot tilde(bold(x)))|_Omega in P_3 (Omega)$；从而对任意 $psi in L^2(bb(S)^d)$，
  $
    integral_(cal(G)) rho_3(bold(theta) dot tilde(bold(x))) psi(bold(theta)) dif bold(theta)
    in P_3 (Omega).
  $
]<lem:degenerate-split>

#proof[
  设 $bold(theta) = frak(P)(bold(omega), b) in cal(G)$。由 $frak(P)$ 是双射且 $rho_3$ 正三次齐次，只需就 $(bold(omega), b) in.not cal(P)_Omega$ 讨论。若 $b >= b_+(bold(omega))$，则对一切 $bold(x) in overline(Omega)$ 有 $bold(omega) dot bold(x) + b >= bold(omega) dot bold(x) - min_(bold(y) in overline(Omega)) bold(omega) dot bold(y) >= 0$，故 $rho_3(bold(omega) dot bold(x) + b) = (bold(omega) dot bold(x) + b)^3 in P_3 (Omega)$；若 $b <= b_-(bold(omega))$，则同理 $bold(omega) dot bold(x) + b <= 0$，故该特征在 $Omega$ 上恒为零。两种情形都落在 $P_3 (Omega)$ 中。

  于是 $bold(theta) |-> rho_3(bold(theta) dot tilde(bold(x)))|_Omega$ 在 $cal(G)$ 上是取值于 $P_3 (Omega)$ 的映射，且被 $(sup_(bold(x) in overline(Omega)) norm(bold(x))_(ell^2) + c_b)^3$ 一致控制。$P_3 (Omega)$ 是 $L^2(Omega)$ 的有限维子空间，因而闭；由 Bochner 积分保持闭子空间，上述积分仍属于 $P_3 (Omega)$。证毕。
]

因此把 @LiuMaoXu2025 定理 2.2 的球面积分表示按 $bb(S)^d = frak(P)(cal(P)_Omega) union.sq cal(G)$ 分解后，由 @lem:degenerate-split，$cal(G)$ 上的贡献由固定的 $dim P_3 (Omega)$ 个特征张成的 $P_3 (Omega)$ 精确吸收，覆盖半径只需在 $frak(P)(cal(P)_Omega)$ 上要求，不影响下文的逼近阶。

这 $dim P_3 (Omega)$ 个固定特征，本文的取法与该注不同：常数特征 $xi_(chi,0)$ 本身不是字典元素，而 @LiuMaoXu2025 注 2.1 是取 $dim P_3 (Omega)$ 个退化区中的脊特征张成 $P_3 (Omega)$，本文改取 $dim P_3 (Omega) - 1$ 个脊特征与 $xi_(chi,0)$，张成的空间相同。该文定理 2.2 的构造只要求这组固定特征张成 $P_3 (Omega)$；两组特征都是 $P_3 (Omega)$ 的基，其间的过渡矩阵由多项式方向确定且与 $N$ 无关，故下文比较函数（见 @thm:total-error-le 证明第四步）的系数界至多相差一个与 $N$ 无关的常数因子。

#definition(title: [准均匀参数点集])[
  设 $Theta subset cal(P)_Omega$ 为有限点集，其覆盖半径为
  $
    h(Theta) := max_(bold(theta) in frak(P)(cal(P)_Omega))
    min_(bold(p) in Theta) op("dist")(bold(theta), frak(P)(bold(p))),
  $
  其中 $op("dist")$ 为 $bb(S)^d$ 上的测地距离。称 $Theta$ 关于 $N$ 准均匀，若它在 $frak(P)(cal(P)_Omega)$ 上良分布，
  $
    h(Theta) lt.tilde N^(-1/d),
  $
  且覆盖半径与最小间距同阶，从而排除点的聚集，
  $
    h(Theta) lt.tilde min_(bold(p) != bold(q) in Theta)
    op("dist")(frak(P)(bold(p)), frak(P)(bold(q))).
  $
  两处隐含常数都与 $N$ 无关。
]<def:quasi-uniform>

本文取每个 $Theta_(chi,N)^"act"$ 关于 $N$ 准均匀，并记 $h_(chi,N) := h(Theta_(chi,N)^"act")$。下面构造满足 @def:quasi-uniform 的 $Theta_(chi,N)^"act"$：把偏置换成它在非退化区间 $(b_-(bold(omega)), b_+(bold(omega)))$ 中的相对位置
$
  t(bold(omega), b)
  := (b - b_-(bold(omega)))/(b_+(bold(omega)) - b_-(bold(omega))) in (0, 1),
$
把 $(0,1)$ 等分为 $n_2 tilde.eq N^(1/d)$ 段并取各段中点 $t_k := (k - 1\/2)\/n_2$，$1 <= k <= n_2$。第 $k$ 层配置各自的准均匀方向集 $Lambda_k subset bb(S)^(d-1)$，各层点数至多相差一。方向 $bold(omega) in Lambda_k$ 所配的偏置由 $t(bold(omega), b) = t_k$ 反解，即 $b(bold(omega), t_k) := b_-(bold(omega)) + t_k (b_+(bold(omega)) - b_-(bold(omega)))$，第 $k$ 层的参数为 ${(bold(omega), b(bold(omega), t_k)) : bold(omega) in Lambda_k}$。

由 $b = b_-(bold(omega)) + t (b_+(bold(omega)) - b_-(bold(omega)))$，映射 $(bold(omega), t) |-> (bold(omega), b)$ 关于 $t$ 仿射，斜率为方向宽度 $b_+(bold(omega)) - b_-(bold(omega))$。该宽度关于 $bold(omega)$ 有正的上下界（$Omega$ 含一个半径 $r > 0$ 的球时它不小于 $2 r$，又不超过 $"diam"(Omega)$；$Omega = (0,1)^d$ 时它等于 $norm(bold(omega))_(ell^1) in [1, sqrt(d)]$），而 $b_+$ 与 $-b_-$ 分别是 $overline(Omega)$ 关于 $-bold(omega)$ 与 $bold(omega)$ 的支撑函数，以 $max_(bold(x) in overline(Omega)) norm(bold(x))_(ell^2)$ 为 Lipschitz 常数，故该映射是 $bb(S)^(d-1) times [0,1]$ 到 $overline(cal(P))_Omega$ 的双向 Lipschitz 同胚。与 $frak(P)$ 复合后，$bb(S)^(d-1) times [0,1]$ 到 $frak(P)(overline(cal(P))_Omega)$ 的映射仍双向 Lipschitz，常数只依赖 $Omega$ 与 $c_b$。双向 Lipschitz 映射把覆盖半径与最小间距同时改变至多一个常数因子，故 @def:quasi-uniform 的两条要求只需在乘积度量下验证（@fig:least-squares-param-charts）。

#figure(
  diagram(
    spacing: (24mm, 12mm),
    node((0, 0), $bb(S)^(d-1) times [0, 1]$, name: <prod>),
    node((1, 0), $overline(cal(P))_Omega$, name: <cyl>),
    node((2, 0), $frak(P)(overline(cal(P))_Omega) subset bb(S)^d$, name: <sph>),
    edge(<prod>, <cyl>, $(bold(omega), t) |-> (bold(omega), b)$, "->"),
    edge(<cyl>, <sph>, $frak(P)$, "->"),
    edge(<prod>, <sph>, [双向 Lipschitz], "->", bend: 38deg),
    node((0, 0.45), text(0.8em)[点集构造 $Lambda_k times {t_k}$]),
    node((1, 0.45), text(0.8em)[柱面坐标]),
    node((2, 0.45), text(0.8em)[准均匀性在此度量]),
  ),
  caption: [参数域的三套坐标：点集在乘积坐标中构造，准均匀性则按 @def:quasi-uniform 在 $bb(S)^d$ 上度量。两截映射都双向 Lipschitz，常数只依赖 $Omega$ 与 $c_b$，故覆盖半径与最小间距在两端各差至多一个常数因子],
)<fig:least-squares-param-charts>

满足 @def:quasi-uniform 的点集存在。@LiuMaoXu2025 推论 2.1 取 $bb(S)^(d-1)$ 上含 $n_1 tilde.eq N^((d-1)\/d)$ 个准均匀方向的 $Lambda$ 与 $n_2 tilde.eq N^(1/d)$ 个均匀偏置作张量积 $Lambda times {t_1, dots.h, t_(n_2)}$，其覆盖半径为 $N^(-1/d)$ 量级。本文的点集则为 $union.big_(k=1)^(n_2) Lambda_k times {t_k}$，各层的方向集可以不同。上述计数只要求每个 $Lambda_k$ 在 $bb(S)^(d-1)$ 上准均匀、层距仍为 $n_2^(-1)$，故覆盖半径的量级不变，最小间距同样是该量级，同层内为 $Lambda_k$ 的球面间距，跨层则不小于层距。方向集按维数取：

- $d = 2$ 取等距角网格，其覆盖半径为 $pi\/n_1$，该界直接得到；
- $d = 3$ 取 $bb(S)^2$ 上的 Fibonacci 格，其准均匀性见 @HardinMichaelsSaff2016。

以上是标量字典，本文考虑的两类问题未知量为对称张量场与向量场（板弯曲的挠度为标量），故逐分量张成。记 $n_s=d(d+1)/2$，取 $op("Sym")(d)$ 的固定正交基
${bold(T)_alpha}_(alpha=1)^(n_s)$ 与 $RR^d$ 的标准基
${bold(e)_i}_(i=1)^d$。线弹性的原始空间为
$
  hat(bold(Sigma))_("LE",N) & := span {xi_(bold(sigma),j) bold(T)_alpha:
                                0<=j<=N, 1<=alpha<=n_s}, \
      hat(bold(U))_("LE",N) & := span {xi_(bold(u),j) bold(e)_i:
                                0<=j<=N, 1<=i<=d}.
$
板弯曲的原始空间为
$
  hat(bold(Sigma))_("KL",N) & := span {xi_(bold(M),j) bold(T)_alpha:
                                0<=j<=N, 1<=alpha<=3}, \
            hat(U)_("KL",N) & := span {xi_(w,j):0<=j<=N}.
$
这里用上标 $hat(dot)$ 统一标记这类尚未施加平均迹规范或边界投影的原始字典空间，以及其中的原始近似与定义在其上的泛函（如后文的 $hat(v)_N$ 与 $hat(cal(J))_"LE"$）；经过约束投影的物理量不带该记号。

== 连续正交投影

平均迹投影 $Pi_"tr"$ 定义为
$
  Pi_"tr" bold(tau)
  := bold(tau)
  - 1/(d abs(Omega))
  (integral_Omega tr(bold(tau)) dif x) bold(I).
$
它是从 $bold(H)(div)$ 到 $bold(Sigma)_"LE"$ 的有界投影。对 $m in {1,2}$，令 $Pi_D^(m)$ 为 $H^m (Omega)$ 到闭子空间 $H_0^m (Omega)$ 的正交投影（存在性与正交性参考 @BrennerScott2008 命题 2.3.1）：
$
  (Pi_D^(m) v,z)_(H^m (Omega))
  =(v,z)_(H^m (Omega)),
  quad forall z in H_0^m (Omega).
$
向量情形逐分量作用。因为正交投影的算子范数为 $1$，
$
  norm(Pi_D^(m) v)_(H^m)<=norm(v)_(H^m).
$

我们不改变原始空间，而把投影写进被最小化的泛函。对
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
因此求解变量仍是原始简单张成空间中的系数，物理残差则在投影后的场上计算。在最小二乘框架下强加而非弱化本质边界条件是有限元文献的标准做法（@BochevGunzburger2009）；神经网络方法则更常用边界软残差（@RaissiPerdikarisKarniadakis2019、@ChenChiEYang2022），或以构造性乘子与距离函数精确满足边界条件（@LagarisLikasFotiadis1998、@SukumarSrivastava2022）。@Liu2026 通过一阶系统重写规避二阶提法中的 conormal trace 与 inverse-trace 障碍，但仍以 $L^2(partial Omega)$ 边界残差处理 Dirichlet 数据，且其结论是在诱导的最小二乘范数中成立。本文采用投影，因此没有边界软残差及其权重选择问题；若要使含边界残差的最小二乘泛函与完整图范数等价，边界失配通常须以 $H^(1\/2)$ 型分数阶范数或相应加权范数度量，参见 @BochevGunzburger2009 与 @MonsuurSmeetsStevenson2025。该构造也不损失字典的逼近阶：若 $v_star in H_0^m (Omega)$、$m in {1,2}$，则 $Pi_D^(m) v_star = v_star$，从而对任意 $hat(v)_N in H^m (Omega)$ 有
$
  norm(v_star-Pi_D^(m) hat(v)_N)_(H^m)
  =norm(Pi_D^(m)(v_star-hat(v)_N))_(H^m)
  <=norm(v_star-hat(v)_N)_(H^m).
$
<eq:projection-transfer>

这个传递估计说明，所需假设仅是物理变量本身的 Sobolev 正则性。椭圆型投影带来的理论收益正是这一范数为一的稳定传递。代价是实际计算中必须近似求解投影问题。

== 有限维 Ritz 实现

以下有限维 Ritz 构造及其显式逼近阶限于数值实验采用的张量积盒状区域，具体取 $Omega=(0,1)^d$。

对 $m in {1,2}$，取边界适配的张量积三次 B 样条空间
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
对 $m=1$，删除在边界取非零值的端点 B 样条；对 $m=2$，进一步删除具有非零边界法向导数的端点模态。因此辅助空间本身精确满足零迹或固支迹条件。

对任意 $v_star in H_0^m (Omega)$ 与原始近似 $hat(v)_N$，正交性给出
$
  norm(v_star-Pi_(D,K)^(m)hat(v)_N)_(H^m) & <=norm((I-Pi_(D,K)^(m))v_star)_(H^m) \
                                          & quad +norm(v_star-hat(v)_N)_(H^m).
$
第一项是辅助 Ritz 空间误差，第二项保留原始字典逼近阶。注意 $Pi_(D,K)^(m)$ 不是连续投影 $Pi_D^(m)$；即使二者都满足边界条件，也不能在分析中混为同一算子。

辅助 Ritz 空间误差有显式阶。设张量积网格尺寸为 $h_K tilde.eq K^(-1/d)$。三次 B 样条的阶数为 $4$；在单位区间上取保持相应齐次端点条件的三次样条拟插值，再逐方向作张量积，可得对 $v in H^s (Omega) inter H_0^m (Omega)$、$s >= m$ 的边界适配逼近，其误差为 $h_K^(min(s, 4)-m) norm(v)_(H^s (Omega))$ 量级（样条逼近参考 @Schumaker2007）。由于 $Pi_(D,K)^(m)$ 是 $V_K^(m)$ 上的 $H^m$ 正交投影，投影存在性与最佳逼近性质参考 @BrennerScott2008 命题 2.3.1，故
$
  norm((I-Pi_(D,K)^(m))v)_(H^m)
  lt.tilde K^(-(min(s, 4)-m)/d) norm(v)_(H^s (Omega)).
$
<eq:ritz-rate>
这里 $s$ 是对被投影函数的正则性假设，指数在 $s = 4$ 处饱和：$4$ 是样条侧的饱和指数，地位与后文字典侧的 $s_"cap" (d)$ 相同，解更光滑时阶不再提高。

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
由该经验内积定义的投影记作 $tilde(Pi)_(D,K,Q_"R")^(m)$。平均迹也不使用训练样本估计，而是另取一组与训练规则及 Ritz 规则都独立的均匀 Monte Carlo 点
${bold(y)_r}_(r=1)^(Q_"tr")$，定义
$
  tilde(Pi)_("tr",Q_"tr") bold(tau)
  := bold(tau)-1/d (
    1/Q_"tr" sum_(r=1)^(Q_"tr") tr(bold(tau)(bold(y)_r))
  ) bold(I).
$
实现中先计算每个原始应力特征在这组点上的样本均值，再消去一个常数球应力自由度；因此离散系数对该独立规则精确满足零平均迹规范，与连续规范之间的偏差则由下述 $epsilon_"trquad"$ 度量。记 Ritz 内积误差为 $epsilon_"Rquad"$，平均迹积分误差为 $epsilon_"trquad"$，并合记
$
  epsilon_"projquad"^2
  := epsilon_"Rquad"^2+epsilon_"trquad"^2.
$
这里的 $epsilon_"trquad"$ 具体控制系数球上一致的投影差
$
  sup_(bold(c) in cal(C)_(N,B))
  norm(
    (tilde(Pi)_("tr",Q_"tr")-Pi_"tr")
    hat(bold(tau))_bold(c)
  )_(bold(H)(div));
$
该差是常数球张量，故其散度为零，只需控制独立 MC 样本均值与连续平均值之差。分析时先把经验投影场与精确 $Pi_"tr"$ 投影场比较，再对后者使用 @thm:elasticity-stability；这正是平均迹积分误差必须进入 $epsilon_"projquad"$ 的原因。

#theorem(title: [平均迹投影积分误差])[
  在系数约束 $cal(C)_(N,B)$ 下，平均迹投影积分误差满足
  $
    (EE_"tr" epsilon_"trquad"^2)^(1/2)
    <= C_"tr" B Q_"tr"^(-1/2),
  $
  其中期望只对平均迹规则取，$C_"tr"$ 仅依赖 $abs(Omega)$、$d$ 与原始应力特征的一致 $L^oo$ 上界，不依赖 $N$、$K$、$Q_"tr"$ 与 $B$。
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
从而 $epsilon_"trquad" = sqrt(abs(Omega)\/d) sup_(bold(c) in cal(C)_(N,B)) abs(Delta(bold(c)))$。记 $bold(c)$ 中的应力块系数为 $(c_(j,alpha))_(0<=j<=N, 1<=alpha<=n_s)$，则 $tr hat(bold(tau))_(bold(c)) = sum_(j,alpha) c_(j,alpha) tr(bold(T)_alpha) xi_(bold(sigma),j)$，$Delta$ 关于 $bold(c)$ 线性：
$
  Delta(bold(c)) = sum_(j,alpha) c_(j,alpha) tr(bold(T)_alpha) Delta_j,
  quad
  Delta_j := 1/Q_"tr" sum_(r=1)^(Q_"tr") xi_(bold(sigma),j)(bold(y)_r)
  - 1/abs(Omega) integral_Omega xi_(bold(sigma),j) dif x,
$
即每个应力特征的迹均值误差 $Delta_j$ 与系数的内积。对该内积用 Cauchy--Schwarz，并注意基 ${bold(T)_alpha}$ 正交归一给出 $sum_alpha tr(bold(T)_alpha)^2 = sum_alpha (bold(T)_alpha, bold(I))_F^2 = norm(bold(I))_F^2 = d$，于是在系数球上
$
  sup_(bold(c) in cal(C)_(N,B)) abs(Delta(bold(c)))
  <= B/sqrt(m_N) (sum_(j,alpha) tr(bold(T)_alpha)^2 Delta_j^2)^(1/2)
  = B sqrt(d/m_N) (sum_(j=0)^N Delta_j^2)^(1/2).
$
最后对每个 $Delta_j$ 作方差估计：${bold(y)_r}$ 为独立均匀样本，故 $EE Delta_j = 0$，且
$
  EE Delta_j^2
  = 1/Q_"tr" op("Var")(xi_(bold(sigma),j)(bold(y)_1))
  <= C_xi^2/Q_"tr",
  quad
  C_xi := max_(0<=j<=N) norm(xi_(bold(sigma),j))_(L^oo(Omega)),
$
其中 $C_xi <= (sup_(bold(x) in overline(Omega)) norm(bold(x))_(ell^2) + c_b)^3$ 是与 $N$ 无关的常数。三式串联并用应力特征数 $N + 1 <= m_N$，得对平均迹规则的二阶矩界
$
  (EE epsilon_"trquad"^2)^(1/2)
  <= C_xi sqrt(abs(Omega) (N+1)/m_N) B Q_"tr"^(-1/2)
  lt.tilde B Q_"tr"^(-1/2),
$
即得结论。证毕。
]

@thm:trquad-rate 覆盖一般区域。本文的计算区域是单位盒，$rho_3$ 的脊结构又使每个原始应力特征的箱均值有闭式，因此该项可以精确消去而非仅仅控制其阶。

#corollary(title: [盒状区域上的精确平均迹投影])[
  设 $Omega = (0,1)^d$、$d in {2,3}$。对 $(bold(omega), b) in cal(P)_Omega$，记 $n$ 为 $bold(omega)$ 中非零分量的个数、$tilde(b) := b + sum_(omega_i < 0) omega_i$ 为反射后的偏置，则
  $
    integral_Omega rho_3(bold(omega) dot bold(x) + b) dif bold(x)
    = (3!)/((3+n)! product_(omega_i != 0) abs(omega_i))
    sum_(bold(v) in {0,1}^n) (-1)^(n - abs(bold(v)))
    rho_1 (tilde(b) + sum_(i) abs(omega_i) v_i)^(3+n),
  $
  <eq:box-mean>
  其中 $rho_1(t) := max(t,0)$；退化参数的两种情形分别取三次多项式的显式三阶矩与零。以 @eq:box-mean 的精确均值构造 $tilde(Pi)_("tr",Q_"tr")$ 中被减去的常数球张量，则
  $
    tilde(Pi)_"tr" = Pi_"tr" quad "在" hat(bold(Sigma))_("LE",N) "上",
    quad
    epsilon_"trquad" = 0.
  $
]<cor:exact-trace>

#proof[
  先设 $(bold(omega), b) in cal(P)_Omega$。把 $omega_i < 0$ 的坐标作反射 $x_i |-> 1 - x_i$，盒不变而仿射函数化为 $sum_i abs(omega_i) x_i + tilde(b)$；零分量对应的坐标不出现在被积函数中，积分掉后只剩 $n$ 个活动方向。对这 $n$ 个方向逐一取原函数：每积一次使 $rho_3$ 的幂次升一、并除以相应的 $abs(omega_i)$ 与新幂次，$n$ 次之后得到幂次 $3+n$、分母 $product abs(omega_i) dot (3+n)!\/3!$，而按盒的顶点取值作容斥即得 @eq:box-mean 的交错和。

  再看退化参数。由 @lem:degenerate-split 的两种情形：$b >= b_+(bold(omega))$ 时特征等于三次多项式 $(bold(omega) dot bold(x) + b)^3$，其均值由 $X_i tilde U(0,1)$ 的中心三阶矩为零、方差为 $sum_i omega_i^2\/12$ 直接给出；$b <= b_-(bold(omega))$ 时特征在 $Omega$ 上恒为零，均值为零。

  最后由 $tr hat(bold(tau))_bold(c) = sum_(j,alpha) c_(j,alpha) tr(bold(T)_alpha) xi_(bold(sigma),j)$ 关于 $bold(c)$ 线性，@thm:trquad-rate 证明中的 $Delta(bold(c)) = sum_(j,alpha) c_(j,alpha) tr(bold(T)_alpha) Delta_j$ 在每个 $Delta_j$ 为零时对一切 $bold(c)$ 为零，故两个投影在 $hat(bold(Sigma))_("LE",N)$ 上重合，$epsilon_"trquad" = sqrt(abs(Omega)\/d) sup_bold(c) abs(Delta(bold(c))) = 0$。证毕。
]

下面估计经验 Ritz 投影误差。以 $hat(v)_bold(c)$ 统一表示需要施加本质边界投影的原始场：线弹性中它是逐分量投影的位移场且 $m=1$，板弯曲中它是挠度且 $m=2$。定义
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
这里使用 $C_("R",K)^(m)$ 而非通常的 Christoffel 函数记号，以避免与后文其他数学量冲突。

#theorem(title: [经验 Ritz 投影积分误差])[
  设 $0<delta<1$、$0<eta<1$。若
  $
    Q_"R"
    >= C_delta C_("R",K)^(m) log(2 K/eta),
  $
  其中 $C_delta>0$ 仅依赖 $delta$，则
  $
    bb(P)(cal(E)_("R",delta)) >= 1-eta,
  $
  且条件于该谱稳定事件有
  $
    (
      EE_"R" [epsilon_"Rquad"^2 | cal(E)_("R",delta)]
    )^(1/2)
    <= C_(delta,eta,xi,m) B
    sqrt(C_("R",K)^(m)/Q_"R").
  $
  对本文的准均匀张量积三次 B 样条空间，固定 $m in {1,2}$ 时
  $
    C_("R",K)^(m) <= C_("inv",m) h_K^(-d) lt.tilde K,
  $
  从而
  $
    (
      EE_"R" [epsilon_"Rquad"^2 | cal(E)_("R",delta)]
    )^(1/2)
    lt.tilde B sqrt(K/Q_"R").
  $
  隐含常数不依赖 $N$、$K$、$Q_"R"$ 与 $B$。
]<thm:rquad-rate>

#proof[
以下只在 $cal(E)_("R",delta)$ 上使用经验投影；在其补集如何定义投影差不影响条件期望。
随机离散最小二乘投影的 Gram 谱稳定性由最大 Christoffel 或杠杆函数控制，标量点值采样可参考 @CohenDavenportLeviatan2013，任意向量值线性采样及 Sobolev 梯度采样的统一框架可参考 @Adcock2025。把后者的采样算子取为 $cal(L)_m (bold(x))$，矩阵浓缩即给出所述样本条件与
$bb(P)(cal(E)_("R",delta)) >= 1-eta$。

以下给出投影差的二阶矩估计。对固定原始场 $hat(v)$，记
$r:=(I-Pi_(D,K)^(m))hat(v)$。连续正交性给出
$(r,phi_k)_(H^m)=0$。经验投影与连续投影的系数差满足
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
这正是 @CohenDavenportLeviatan2013 定理 2 证明中经验投影减连续投影估计对 $H^m$ 向量值采样的对应形式。

再对系数球取一致上确界。记需投影场的分量数为 $n_v$；线弹性中 $n_v=d$，板弯曲中 $n_v=1$。令 $xi_(v,j)$ 为相应原始标量特征。用系数球的半径 $B/sqrt(m_N)$、矩阵算子范数不超过 Frobenius 范数，并在 $cal(E)_("R",delta)$ 上使用
$norm(bold(G)_(K,Q_"R")^(-1))_2 <= (1-delta)^(-1)$，得到
$
  EE_"R" [epsilon_"Rquad"^2 1_(cal(E)_("R",delta))]
  <= B^2 C_("R",K)^(m) /(
    m_N (1-delta)^2 Q_"R"
  )
  sum_(i=1)^(n_v) sum_(j=0)^N
  norm((I-Pi_(D,K)^(m))xi_(v,j))_(H^m(Omega))^2.
$
由于正交投影非扩张，右端每个残差不超过
$norm(xi_(v,j))_(H^m)$。$rho_3$ 的参数域有界，故这些特征的 $H^m$ 范数关于 $j$ 与 $N$ 一致有界；再用 $n_v(N+1)<=m_N$，并以
$bb(P)(cal(E)_("R",delta)) >= 1-eta$ 除去事件概率，即得条件二阶矩界。

最后，固定次数、准均匀网格上的张量积 B 样条满足局部 Sobolev 逆估计（参考 @Schumaker2007 与 @TakacsTakacs2016）
$
  abs(Omega) norm(cal(L)_m (bold(x))z_K)_(ell^2)^2
  <= C_("inv",m) h_K^(-d) norm(z_K)_(H^m(Omega))^2.
$
因此 $C_("R",K)^(m) lt.tilde h_K^(-d) tilde.eq K$，代回即得最后一式。证毕。
]

== 系数约束

把给定模型的全部独立输出系数依固定顺序排成向量 $bold(c) in RR^(m_N)$，其中 $m_N$ 是离散系数总数。对给定预算 $B>0$，定义
$
  cal(C)_(N,B)
  := {bold(c) in RR^(m_N):sqrt(m_N) norm(bold(c))_(ell^2)<=B}.
$
这个缩放即 @LiuMaoXu2025 线性化网络类中的系数约束，与 ReLU 幂网络的有界表示相容，并避免同一物理函数仅因特征数改变而获得不同的系数尺度。系数约束施加在未作列归一化的物理系数上。

= 经验最小二乘与误差分析

== 训练泛函

令 ${bold(x)_ell}_(ell=1)^Q$ 为 $Omega$ 上独立均匀样本，并设 $abs(Omega)$ 为区域测度。对 $bold(z)=(bold(tau),bold(v)) in bold(X)_"LE"$，把线弹性的两个残差依次记为
$
  bold(r)_"c" (bold(z))
  :=bold(cal(A))_"LE" bold(tau)-bold(epsilon)(bold(v)),
  quad
  bold(r)_"e" (bold(z))
  :=div bold(tau)+bold(f).
$
经验泛函为
$
  cal(J)_("LE",Q)(bold(z))
  :=abs(Omega)/Q sum_(ell=1)^Q (
    norm(bold(r)_"c" (bold(z))(bold(x)_ell))_F^2
    +norm(bold(r)_"e" (bold(z))(bold(x)_ell))_(ell^2)^2
  ).
$
对板问题，取
$
  bold(r)_"c" (bold(z))
  :=bold(cal(A))_"KL" bold(tau)-bold(kappa)(v),
  quad
  r_"e" (bold(z)):=div div bold(tau)+f,
$
经验泛函为
$
  cal(J)_("KL",Q)(bold(z))
  :=abs(Omega)/Q sum_(ell=1)^Q (
    norm(bold(r)_"c" (bold(z))(bold(x)_ell))_F^2
    +abs(r_"e" (bold(z))(bold(x)_ell))^2
  ).
$
把所有加权残差写成设计矩阵 $bold(R)_(N,Q)$ 与右端
$bold(b)_(N,Q)$ 后，训练问题是
$
  min_(bold(c) in cal(C)_(N,B))
  norm(bold(R)_(N,Q) bold(c)-bold(b)_(N,Q))_(ell^2)^2.
$
这是欧氏球上的凸最小二乘问题。

== 单隐层 Sobolev 逼近误差

字典逼近误差由 @LiuMaoXu2025 的积分表示理论给出，出发点是激活函数的球面谱衰减：$rho_3$ 在 $bb(S)^d$ 上的非零 Legendre 系数按球谐次数的 $-(d+2k+1)\/2$ 次幂精确衰减，取 $k = 3$ 并记
$
  s_"cap" (d) := (d+7)/2.
$
由 Funk--Hecke 公式，叠加算子
$
  psi |-> integral_(bb(S)^d) rho_3(bold(theta) dot tilde(bold(x))) psi(bold(theta)) dif bold(theta),
  quad psi in L^2(bb(S)^d)
$
在每个球谐频带上以这些系数为乘子，恰好光滑化 $s_"cap" (d)$ 阶；经 $rho_3$ 的正三次齐次性限制到 $Omega$，该算子把 $L^2(bb(S)^d)$ 映满 $H^(s_"cap" (d))(Omega)$，且 $H^(s_"cap" (d))(Omega)$ 范数与最小表示密度的 $L^2(bb(S)^d)$ 范数等价（@LiuMaoXu2025 定理 2.3）。该刻画双向成立：在某个与 $N$ 无关的系数预算下可被准均匀字典按 $L^2$ 逼近的函数，恰为 $H^(s_"cap" (d))(Omega)$ 的元素（@LiuMaoXu2025 定理 2.4）。因此 $s_"cap" (d)$ 是字典的饱和指数，度量特征族可利用的光滑性，并不限制方程解自身的正则性；对物理变量只需假设 Sobolev 正则性，无需单独的积分表示假设。

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
这里减去的 $1$ 或 $2$ 即 $m$；由 $norm(div bold(tau))_(L^2(Omega)) lt.tilde norm(bold(tau))_(H^1(Omega))$ 与 $norm(div div bold(tau))_(L^2(Omega)) lt.tilde norm(bold(tau))_(H^2(Omega))$，两类图范数分别受逐分量 $H^1$ 与 $H^2$ 范数控制，逐分量 $H^m$ 逼近率因此直接给出图范数逼近率。

@LiuMaoXu2025 定理 2.2 给出线性化 ReLU 幂网络的确定性 Sobolev 逼近率：设分量 $v_chi in H^(s_chi)(Omega)$，$m <= s_chi <= s_"cap" (d)$，变量 $chi$ 的参数点集覆盖半径为 $h_(chi,N)$，则存在系数满足
$
  norm(bold(a))_(ell^2)
  lt.tilde h_(chi,N)^(-(s_"cap" (d)-s_chi)) norm(v_chi)_(H^(s_chi)(Omega)) N^(-1/2)
$
的线性化网络 $v_N$，使得
$
  norm(v_chi - v_N)_(H^m (Omega))
  lt.tilde h_(chi,N)^(s_chi - m) norm(v_chi)_(H^(s_chi)(Omega)).
$
定理另要求误差范数的导数阶不超过激活幂次，即 $m <= 3$，在 $m in {1,2}$ 时自动满足。代入字典构造的覆盖半径 $h_(chi,N) lt.tilde N^(-1/d)$，得
$
  epsilon_"app" (N)
  lt.tilde N^(-beta),
$
其中 $beta=beta_"LE"$ 或 $beta_"KL"$。该上界是确定性的：既无独立抽样带来的对数修正，也无需对参数取期望。系数界源于构造中的带限反卷积：把球谐展开截断到次数 $tilde.eq h_(chi,N)^(-1)$，再逐频带除以 $rho_3$ 的 Legendre 系数；当 $s_chi < s_"cap" (d)$ 时该步骤把系数范数放大至多 $h_(chi,N)^(-(s_"cap" (d)-s_chi))$ 倍。系数规模决定预算：当最不光滑分量满足 $s_chi = s_"cap" (d)$ 时，预算 $B$ 可取与 $N$ 无关的常数；一般情形需把 $B$ 取到 $N^((s_"cap" (d) - s_chi)\/d)$ 的量级。

== Monte Carlo 训练误差

本节的误差分解与 Rademacher 估计遵循 @SiegelHongJinHaoXu2023；@LiuMaoXu2025 第 7 节对本文所用的同一线性化网络类（同样的 $ell^2$ 系数球）给出了能量泛函情形的对应结论。与两者的差别在于本文对本质边界条件采用投影：投影后的位移与挠度特征不再是脊函数，@SiegelHongJinHaoXu2023 定理 6 不能直接施于该部分，需分块处理。

记 $cal(D)_(N,K)$ 为构成两个物理残差的所有投影特征及其所需导数，并设
$
  C_(Pi,K) := sup_(g in cal(D)_(N,K))
  norm(g)_(L^oo(Omega)).
$
对线弹性，这一包络涉及应力的一阶导数和位移的一阶导数；对板问题则涉及弯矩与挠度的二阶导数。

#theorem(title: [训练泛化误差])[
  沿用 @thm:rquad-rate 的谱稳定事件 $cal(E)_("R",delta)$，并条件于平均迹规则与 Ritz 规则。设 $bold(f) in L^oo$（板问题为 $f in L^oo$），系数取自 $cal(C)_(N,B)$。则包络满足
  $
    C_(Pi,K) lt.tilde K^(1/2),
  $
  <eq:envelope-rate>
  且平方风险层面的一致偏差满足
  $
    EE sup_(bold(c) in cal(C)_(N,B))
    abs(cal(J)(bold(c))-cal(J)_Q (bold(c)))
    lt.tilde (
      B^2 K + norm(bold(f))_(L^oo(Omega)) B sqrt(K)
    ) Q^(-1/2),
  $
  隐含常数不依赖 $N$、$Q$、$K$、$B$ 与 $lambda$。
]<thm:train-generalization>

#proof[
  第一步：分块。条件于两组投影规则后，$cal(D)_(N,K)$ 分为两族。未投影块是应力（弯矩）特征及其所需导数：$tilde(Pi)_("tr",Q_"tr")$ 只减去一个由独立样本定出的常数球张量，条件化后是固定平移，其散度为零且不改变任何导数，故该族仍是脊函数族 $bb(D) = {rho_3(bold(omega) dot bold(x) + b) : (bold(omega), b) in bb(S)^(d-1) times [-c_b, c_b]}$ 的像。已投影块是位移（$m=1$）与挠度（$m=2$）特征，经 $tilde(Pi)_(D,K,Q_"R")^(m)$ 后落在 $V_K^(m)$ 中，是 B 样条而非脊函数。

  第二步：未投影块。参数域紧，$rho_3 in W^(3,oo)$（其二阶导数 Lipschitz），故对 $m in {1,2}$ 有 $rho_3 in W^(m+1,oo)$，@SiegelHongJinHaoXu2023 定理 6 适用：
  $
    R_Q (partial^bold(alpha) bb(D)) lt.tilde Q^(-1/2),
    quad abs(bold(alpha)) <= m,
  $
  隐含常数与 $Q$、$N$ 无关。该定理的证明只用到 $partial^bold(alpha) bb(D) = {bold(omega)^bold(alpha) rho_3^((bold(alpha)))(bold(omega) dot bold(x) + b)}$ 与 Lipschitz 复合规则，正是脊结构所在。

  第三步：已投影块。上式的推导对 B 样条无效，改用线性类的 Rademacher 界（@ShalevShwartzBenDavid2014 第 26.2 节）。该块的假设类是系数的线性映射在半径 $B\/sqrt(m_N)$ 的 $ell^2$ 球上的像；把该块的投影特征及所需导数在点 $bold(x)$ 处的取值排成向量，其分量被 $C_(Pi,K)$ 控制，故该向量的 $ell^2$ 范数不超过 $C_(Pi,K) sqrt(m_N)$。于是
  $
    R_Q lt.tilde (B/sqrt(m_N)) dot (C_(Pi,K) sqrt(m_N)) Q^(-1/2)
    = B C_(Pi,K) Q^(-1/2),
  $
  与 $N$ 无关：系数球的 $sqrt(m_N)$ 缩放恰好抵消特征维数的增长。

  第四步：包络定阶。设 $chi$ 为需投影的标量特征。由 $rho_3$ 的参数域有界，$norm(xi_(chi,j))_(H^m(Omega))$ 关于 $j$ 与 $N$ 一致有界；正交投影非扩张给 $norm(Pi_(D,K)^(m) xi_(chi,j))_(H^m(Omega)) lt.tilde 1$，在 $cal(E)_("R",delta)$ 上经验投影亦然，相差因子 $(1-delta)^(-1)$。再用 @thm:rquad-rate 证明末尾的张量积 B 样条逆估计
  $
    abs(Omega) norm(cal(L)_m (bold(x)) z_K)_(ell^2)^2
    <= C_("inv",m) h_K^(-d) norm(z_K)_(H^m(Omega))^2,
  $
  取 $z_K = tilde(Pi)_(D,K,Q_"R")^(m) xi_(chi,j)$，得 $C_(Pi,K) lt.tilde h_K^(-d/2) tilde.eq K^(1/2)$，即 @eq:envelope-rate。未投影块的包络与 $K$ 无关，不改变该阶。

  第五步：装配。经验泛函在系数上是二次项加一个以 $bold(f)$ 为系数的线性项，与 @SiegelHongJinHaoXu2023 定理 5 所处理的损失同型；对该定理的结构性界分别代入第二、三步的两个复杂度界，并以 $B C_(Pi,K)$ 作为其中的 $C M$，得
  $
    R_Q lt.tilde (
      B C_(Pi,K) + norm(bold(f))_(L^oo(Omega))
    ) B C_(Pi,K) Q^(-1/2).
  $
  最后由对称化（@SiegelHongJinHaoXu2023 定理 4）把 $R_Q$ 转为一致偏差的期望，并代入 @eq:envelope-rate，即得结论。逆估计常数 $C_("inv",m)$ 与 B 样条次数、网格准均匀性有关，与 Lamé 常数无关，故隐含常数关于 $lambda$ 一致。证毕。
]

因此，当最终误差以图范数的均方根表示时，仅由经验积分产生的项是 $Q^(-1/4)$，而不是 $Q^(-1/2)$。两种指数对应不同层次的量，必须加以区分。

== 总误差估计

令 $bold(z)_star$ 表示连续精确解；$bold(z)_(N,Q,K)$ 表示用 $N$ 个字典特征、$Q$ 个训练点和 $K$ 维 Ritz 辅助空间得到的物理解。以 $epsilon_"opt"$ 表示经验目标值意义下的代数求解次优性。下述两条定理分别对线弹性与板弯曲把四类误差合并为总误差估计：Ritz 截断、Ritz 投影积分、平均迹积分与训练泛化误差分别由 @eq:ritz-rate、@thm:rquad-rate、@thm:trquad-rate 与 @thm:train-generalization 取显式阶，余下的离散常数只有代数次优性 $epsilon_"opt"$。两类问题共享同一误差分解框架，差别仅在稳定性定理的来源与平均迹投影的有无。

#theorem(title: [线弹性线性化网络经验最小二乘误差])[
  假设 @thm:elasticity-stability 成立；单隐层参数取上述准均匀字典；$bold(sigma)_star$ 与 $bold(u)_star$ 的各分量分别属于 $H^(s_(bold(sigma)))(Omega)$ 与 $H^(s_(bold(u)))(Omega)$，$s_chi in [1, s_"cap" (d)]$；预算 $B$ 足以容纳一个达到所述逼近阶的比较函数。固定 $0<delta,eta<1$，并设 $Q_"R"$ 满足 @thm:rquad-rate 的谱稳定样本条件。期望对训练样本、平均迹规则与 Ritz 规则取，其中 Ritz 规则条件于 $cal(E)_("R",delta)$，则
  $
    EE norm(bold(z)_star - bold(z)_(N,Q,K))_(bold(X)_"LE")^2
    lt.tilde N^(-2 beta_"LE")
    + K^(-2(min(s_(bold(u)), 4)-1)/d)
    + B^2 K Q_"R"^(-1)
    + B^2 Q_"tr"^(-1)
    + (B^2 K + norm(bold(f))_(L^oo(Omega)) B sqrt(K)) Q^(-1/2)
    + epsilon_"opt".
  $
]<thm:total-error-le>

其中 $B^2 Q_"tr"^(-1)$ 一项刻画一般区域上的平均迹积分误差；本文数值实验取 $Omega = (0,1)^d$ 并按 @cor:exact-trace 使用闭式箱均值，故该项恒为零。

#proof[
  先条件于 Ritz 与平均迹两组相互独立的投影规则，以下确定性比较与训练样本期望均在该条件下进行；最后再分别对两组投影规则取期望，其中 Ritz 规则条件于 $cal(E)_("R",delta)$。

  第一步：系数到场的三个映射。对系数向量 $bold(c) in RR^(m_N)$，记原始字典场为 $hat(bold(z))(bold(c)) = (hat(bold(tau))_(bold(c)), hat(bold(v))_(bold(c)))$，并定义精确投影场与经验投影场
  $
    bold(z)(bold(c)) := (Pi_"tr" hat(bold(tau))_(bold(c)), Pi_(D,K)^(1) hat(bold(v))_(bold(c))),
    quad
    tilde(bold(z))(bold(c)) := (
      tilde(Pi)_("tr",Q_"tr") hat(bold(tau))_(bold(c)),
      tilde(Pi)_(D,K,Q_"R")^(1) hat(bold(v))_(bold(c))
    ).
  $
  记 $bold(c)^"out" in cal(C)_(N,B)$ 为求解器输出的系数，则计算所得物理解为 $bold(z)_(N,Q,K) = tilde(bold(z))(bold(c)^"out")$。由 $rho_3 in C^2$ 且其二阶导数 Lipschitz，原始特征逐分量属于 $W^(3,oo)(Omega) subset H^3(Omega)$，故上述诸场均属于 $bold(H)(div) times H^1(Omega; RR^d)$。下文把乘积图范数
  $
    norm(bold(w))_(bold(X))^2
    := norm(bold(tau))_(bold(H)(div))^2 + norm(bold(v))_(H^1(Omega))^2,
    quad bold(w) = (bold(tau), bold(v)),
  $
  视为整个 $bold(H)(div) times H^1$ 上的范数，$bold(X)_"LE"$ 是其带平均迹规范与边界约束的闭子空间。$Pi_"tr" hat(bold(tau))_(bold(c))$ 的平均迹为零，且 $Pi_(D,K)^(1) hat(bold(v))_(bold(c)) in V_K^(1) subset H_0^1(Omega; RR^d)$，故对一切 $bold(c)$ 有 $bold(z)(bold(c)) in bold(X)_"LE"$。经验投影场则一般不落在 $bold(X)_"LE"$ 中：$tilde(Pi)_(D,K,Q_"R")^(1)$ 的经验性只体现在 Gram 方程的数值积分上，输出仍是 $V_K^(1)$ 的元素，边界条件不受影响；但 $tilde(Pi)_("tr",Q_"tr") hat(bold(tau))_(bold(c))$ 与 $Pi_"tr" hat(bold(tau))_(bold(c))$ 相差一个常数球张量，其连续平均迹一般非零。这正是不能对 $tilde(bold(z))(bold(c)^"out")$ 直接引用 @thm:elasticity-stability、必须经由 $bold(z)(bold(c)^"out")$ 中转的原因。在 @cor:exact-trace 的条件下该差为零，应力块无条件落在 $bold(Sigma)_"LE"$ 中，中转随之多余；以下论证保持一般性。

  第二步：残差半范数及其双边控制。对 $bold(w) = (bold(tau), bold(v)) in bold(H)(div) times H^1$ 定义
  $
    abs(bold(w))_(cal(J))
    := cal(J)_"LE" (bold(w); bold(0))^(1/2)
    = (
      norm(bold(cal(A))_"LE" bold(tau) - bold(epsilon)(bold(v)))_(L^2(Omega))^2
      + norm(div bold(tau))_(L^2(Omega))^2
    )^(1/2).
  $
  映射 $bold(w) |-> (bold(cal(A))_"LE" bold(tau) - bold(epsilon)(bold(v)), div bold(tau))$ 是线性的，故 $abs(dot)_(cal(J))$ 是半范数，满足三角不等式。本构残差是场的线性函数，平衡残差是仿射函数，二者在精确解 $bold(z)_star = (bold(sigma)_star, bold(u)_star)$ 处同时为零，因此对任意 $bold(z) in bold(H)(div) times H^1$，
  $
    cal(J)_"LE" (bold(z); bold(f))
    = abs(bold(z) - bold(z)_star)_(cal(J))^2.
  $
  <eq:residual-shift>
  @thm:elasticity-stability 上界部分的证明只用到柔度算子的范数界与 $norm(bold(epsilon)(bold(v)))_(L^2(Omega)) <= norm(nabla bold(v))_(L^2(Omega))$，并未使用平均迹规范或边界条件，因此连续性在全空间成立：
  $
    abs(bold(w))_(cal(J)) <= sqrt(C_1) norm(bold(w))_(bold(X)),
    quad forall bold(w) in bold(H)(div) times H^1(Omega; RR^d);
  $
  <eq:residual-continuity>
  强制性则按原定理只在约束子空间上成立：
  $
    sqrt(c_1) norm(bold(w))_(bold(X)) <= abs(bold(w))_(cal(J)),
    quad forall bold(w) in bold(X)_"LE".
  $
  <eq:residual-coercivity>
  其中 $0 < c_1 <= C_1 < oo$ 为 @thm:elasticity-stability 两侧的隐含常数，按 $lt.tilde$ 的规范可关于 $lambda$ 一致选取。

  第三步：投影积分误差的一致界。按前文约定，$epsilon_"trquad"$ 控制系数球上一致的平均迹投影差；与之对应，取 $epsilon_"Rquad"$ 为系数球上一致的经验 Ritz 投影差，即
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
    <= sqrt(C_1) epsilon_"projquad".
  $
  <eq:proj-gap-residual>

  第四步：比较函数。由正则性假设，$bold(sigma)_star$ 的各分量属于 $H^(s_(bold(sigma)))(Omega)$，$bold(u)_star$ 的各分量属于 $H^(s_(bold(u)))(Omega)$，$s_chi in [1, s_"cap" (d)]$。在固定基 ${bold(T)_alpha}$ 与 ${bold(e)_i}$ 下把两个场逐分量分解，对每个标量分量在相应变量的字典（含常数特征与多项式方向补足）上应用 @LiuMaoXu2025 定理 2.2，得到原始字典场 $hat(bold(tau))^"cmp" in hat(bold(Sigma))_("LE",N)$ 与 $hat(bold(v))^"cmp" in hat(bold(U))_("LE",N)$，满足
  $
    norm(bold(sigma)_star - hat(bold(tau))^"cmp")_(H^1(Omega))
    lt.tilde h_(bold(sigma),N)^(s_(bold(sigma)) - 1) norm(bold(sigma)_star)_(H^(s_(bold(sigma)))(Omega)),
    quad
    norm(bold(u)_star - hat(bold(v))^"cmp")_(H^1(Omega))
    lt.tilde h_(bold(u),N)^(s_(bold(u)) - 1) norm(bold(u)_star)_(H^(s_(bold(u)))(Omega)),
  $
  其合并系数向量 $bold(c)^"cmp"$ 满足
  $
    norm(bold(c)^"cmp")_(ell^2)
    lt.tilde N^(-1/2) (
      h_(bold(sigma),N)^(-(s_"cap" (d) - s_(bold(sigma))))
      norm(bold(sigma)_star)_(H^(s_(bold(sigma)))(Omega))
      + h_(bold(u),N)^(-(s_"cap" (d) - s_(bold(u))))
      norm(bold(u)_star)_(H^(s_(bold(u)))(Omega))
    ).
  $
  按 @def:quasi-uniform，除上界 $h_(chi,N) lt.tilde N^(-1/d)$ 外，$Theta_(chi,N)^"act"$ 的 $tilde.eq N$ 个点在 $d$ 维参数集 $cal(P)_Omega$ 上的覆盖半径也有一般下界 $h_(chi,N) gt.tilde N^(-1/d)$。代入即得逼近率 $N^(-(s_chi - 1)/d)$ 与系数量级
  $
    sqrt(m_N) norm(bold(c)^"cmp")_(ell^2)
    lt.tilde N^((s_"cap" (d) - min(s_(bold(sigma)), s_(bold(u))))/d)
    (
      norm(bold(sigma)_star)_(H^(s_(bold(sigma)))(Omega))
      + norm(bold(u)_star)_(H^(s_(bold(u)))(Omega))
    ),
  $
  其中用了 $m_N lt.tilde N$。预算假设正是保证 $B$ 不小于该量级，故 $bold(c)^"cmp" in cal(C)_(N,B)$；且 $bold(c)^"cmp"$ 由确定性字典与精确解决定，与训练样本无关。

  比较场取精确投影场 $bold(z)(bold(c)^"cmp")$，两个分量分别估计。先看应力。由 $(bold(tau), bold(I))_(L^2(Omega)) = integral_Omega tr(bold(tau)) dif x$ 与 $norm(bold(I))_(L^2(Omega))^2 = d abs(Omega)$，
  $
    Pi_"tr" bold(tau)
    = bold(tau)
    - ((bold(tau), bold(I))_(L^2(Omega)))/(norm(bold(I))_(L^2(Omega))^2) bold(I),
  $
  即 $Pi_"tr"$ 是 $L^2(Omega; op("Sym")(d))$ 中向 $span {bold(I)}$ 的正交补作正交投影，$L^2$ 范数不增；被减去的常数球张量散度为零，故散度不变，$Pi_"tr"$ 在 $bold(H)(div)$ 图范数下也非扩张。$bold(sigma)_star in bold(Sigma)_"LE"$ 的平均迹为零，故 $Pi_"tr" bold(sigma)_star = bold(sigma)_star$，从而
  $
    norm(bold(sigma)_star - Pi_"tr" hat(bold(tau))^"cmp")_(bold(H)(div))
    = norm(Pi_"tr" (bold(sigma)_star - hat(bold(tau))^"cmp"))_(bold(H)(div))
    <= norm(bold(sigma)_star - hat(bold(tau))^"cmp")_(bold(H)(div))
    lt.tilde norm(bold(sigma)_star - hat(bold(tau))^"cmp")_(H^1(Omega)),
  $
  末一步用了 $norm(div bold(w))_(L^2(Omega)) lt.tilde norm(bold(w))_(H^1(Omega))$。再看位移。取精确有限维 Ritz 误差为
  $
    epsilon_"Ritz" (K) := norm((I - Pi_(D,K)^(1)) bold(u)_star)_(H^1(Omega)),
  $
  由 $Pi_(D,K)^(1)$ 是 $H^1$ 正交投影、算子范数为一，
  $
    norm(bold(u)_star - Pi_(D,K)^(1) hat(bold(v))^"cmp")_(H^1(Omega))
    <= epsilon_"Ritz" (K)
    + norm(bold(u)_star - hat(bold(v))^"cmp")_(H^1(Omega)),
  $
  这是连续投影传递估计 @eq:projection-transfer 的有限维形式。两个分量平方相加并代入字典逼近率，得
  $
    norm(bold(z)(bold(c)^"cmp") - bold(z)_star)_(bold(X))^2
    <= C_2 (N^(-2 beta_"LE") + epsilon_"Ritz" (K)^2),
  $
  <eq:comparison-bound>
  其中 $C_2$ 依赖 $Omega$、$c_b$、$d$ 与精确解分量的 Sobolev 范数，不依赖 $N$、$Q$、$K$。

  第五步：经验极小性与一致 Monte Carlo 偏差。记条件于投影规则的一致偏差
  $
    delta_Q := sup_(bold(c) in cal(C)_(N,B))
    abs(
      cal(J)_"LE" (tilde(bold(z))(bold(c)); bold(f))
      - cal(J)_("LE",Q) (tilde(bold(z))(bold(c)))
    ).
  $
  经验泛函逐点求值所需的有界性由投影字典的点态导数包络提供；在 $bold(f)$ 有界与系数预算假设下，@thm:train-generalization 给出
  $
    EE delta_Q lt.tilde (
      B^2 K + norm(bold(f))_(L^oo(Omega)) B sqrt(K)
    ) Q^(-1/2).
  $
  训练问题恰以 $bold(c) |-> cal(J)_("LE",Q)(tilde(bold(z))(bold(c)))$ 为目标在 $cal(C)_(N,B)$ 上极小化，而 $epsilon_"opt"$ 的定义即经验目标值的次优性，故对一切 $bold(c) in cal(C)_(N,B)$ 有 $cal(J)_("LE",Q)(tilde(bold(z))(bold(c)^"out")) <= cal(J)_("LE",Q)(tilde(bold(z))(bold(c))) + epsilon_"opt"$；特别取 $bold(c) = bold(c)^"cmp"$，这可行因为第四步已验证 $bold(c)^"cmp" in cal(C)_(N,B)$。于是
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
                                                         & <= 2 C_1 norm(bold(z)(bold(c)^"cmp") - bold(z)_star)_(bold(X))^2
                                                           + 2 C_1 epsilon_"projquad"^2 \
                                                         & <= 2 C_1 C_2 (N^(-2 beta_"LE") + epsilon_"Ritz" (K)^2)
                                                           + 2 C_1 epsilon_"projquad"^2.
  $
  再从左端恢复图范数误差。$bold(z)(bold(c)^"out")$ 与 $bold(z)_star$ 都属于 $bold(X)_"LE"$，故 @eq:residual-coercivity 可用于其差；再由三角不等式、@eq:proj-gap-residual 与 @eq:residual-shift，
  $
    c_1 norm(bold(z)(bold(c)^"out") - bold(z)_star)_(bold(X))^2
    <= abs(bold(z)(bold(c)^"out") - bold(z)_star)_(cal(J))^2
    <= 2 cal(J)_"LE" (tilde(bold(z))(bold(c)^"out"); bold(f))
    + 2 C_1 epsilon_"projquad"^2.
  $
  串联上两式与 @eq:empirical-chain，
  $
    norm(bold(z)(bold(c)^"out") - bold(z)_star)_(bold(X))^2
    <= (4 C_1)/(c_1) (
      C_2 (N^(-2 beta_"LE") + epsilon_"Ritz" (K)^2)
      + epsilon_"projquad"^2
    )
    + 2/(c_1) (2 delta_Q + epsilon_"opt")
    + (2 C_1)/(c_1) epsilon_"projquad"^2.
  $
  最后由 @eq:proj-gap 把精确投影场换回计算解本身：
  $
    norm(bold(z)_star - bold(z)_(N,Q,K))_(bold(X))^2
    <= 2 norm(bold(z)_star - bold(z)(bold(c)^"out"))_(bold(X))^2
    + 2 epsilon_"projquad"^2.
  $
  $bold(c)^"cmp"$、$epsilon_"Ritz" (K)$ 与 $epsilon_"projquad"$ 都与训练样本无关；把 $epsilon_"opt"$ 视为求解器给定的次优性上界。对训练样本取期望并代入第五步的 $EE delta_Q$ 界，先得条件于两组投影规则的估计
  $
    EE norm(bold(z)_star - bold(z)_(N,Q,K))_(bold(X))^2
    lt.tilde N^(-2 beta_"LE")
    + epsilon_"Ritz" (K)^2
    + epsilon_"Rquad"^2
    + epsilon_"trquad"^2
    + (B^2 K + norm(bold(f))_(L^oo(Omega)) B sqrt(K)) Q^(-1/2)
    + epsilon_"opt",
  $
  隐含常数只依赖 $c_1$、$C_1$ 与 $C_2$，不依赖 $N$、$Q$、$K$ 与 $B$。由 $bold(u)_star$ 各分量属于 $H^(s_(bold(u)))(Omega)$ 与 @eq:ritz-rate，
  $
    epsilon_"Ritz" (K)^2
    lt.tilde K^(-2(min(s_(bold(u)), 4)-1)\/d)
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
  假设 @thm:plate-stability 成立；单隐层参数取上述准均匀字典；$bold(M)_star$ 与 $w_star$ 的各分量分别属于 $H^(s_(bold(M)))(Omega)$ 与 $H^(s_w)(Omega)$，$s_chi in [2, s_"cap" (2)]$；预算 $B$ 足以容纳一个达到所述逼近阶的比较函数。固定 $0<delta,eta<1$，并设 $Q_"R"$ 满足 @thm:rquad-rate 的谱稳定样本条件。期望对训练样本与条件于 $cal(E)_("R",delta)$ 的 Ritz 规则取，则
  $
    EE norm(bold(z)_star - bold(z)_(N,Q,K))_(bold(X)_"KL")^2
    lt.tilde N^(-2 beta_"KL")
    + K^(-(min(s_w, 4)-2))
    + B^2 K Q_"R"^(-1)
    + (B^2 K + norm(f)_(L^oo(Omega)) B sqrt(K)) Q^(-1/2)
    + epsilon_"opt".
  $
]<thm:total-error-kl>

#proof[
  逐步重复 @thm:total-error-le 的证明，作如下替换。应力分量不设平均迹投影，取
  $
    bold(z)(bold(c)) := (hat(bold(tau))_(bold(c)), Pi_(D,K)^(2) hat(w)_(bold(c))),
    quad
    tilde(bold(z))(bold(c)) := (hat(bold(tau))_(bold(c)), tilde(Pi)_(D,K,Q_"R")^(2) hat(w)_(bold(c)));
  $
  原始特征逐分量属于 $H^3(Omega)$，故诸场属于 $bold(H)(div div) times H^2(Omega)$；$bold(Sigma)_"KL"$ 不带规范约束且 $V_K^(2) subset H_0^2(Omega)$，故精确投影场自动属于 $bold(X)_"KL"$。第二步以 @thm:plate-stability 两侧的隐含常数替代 $c_1$、$C_1$ 后逐字重复，其上界证明同样未用边界条件，连续性在全空间 $bold(H)(div div) times H^2$ 成立；第三步只出现 $epsilon_"Rquad" <= epsilon_"projquad"$；第四步删去 $Pi_"tr"$ 一段，弯矩图范数由逐分量 $H^2$ 范数经 $norm(div div bold(w))_(L^2(Omega)) lt.tilde norm(bold(w))_(H^2(Omega))$ 控制，Ritz 误差取 $epsilon_"Ritz" (K) := norm((I - Pi_(D,K)^(2)) w_star)_(H^2(Omega))$，由 $w_star in H^(s_w)(Omega)$ 与 @eq:ritz-rate 得 $epsilon_"Ritz" (K) lt.tilde K^(-(min(s_w, 4)-2)\/2)$；字典逼近率给出 $N^(-2 beta_"KL")$，其中 $m = 2 <= 3$ 满足 @LiuMaoXu2025 定理 2.2 的导数阶限制；第五、六步不变。最后对条件于 $cal(E)_("R",delta)$ 的 Ritz 规则取期望并应用 @thm:rquad-rate，以 $B^2 K Q_"R"^(-1)$ 替换条件估计中的 $epsilon_"Rquad"^2$，即得结论。证毕。
]

若用均方根图误差表述 @thm:total-error-le 与 @thm:total-error-kl，则各项开方后分别具有量级
$
  N^(-beta),
  quad K^(-(min(s, 4)-m)/d),
  quad B sqrt(K/Q_"R"),
  quad B Q_"tr"^(-1/2),
  quad (B^2 K + norm(bold(f))_(L^oo) B sqrt(K))^(1/2) Q^(-1/4),
  quad epsilon_"opt"^(1/2).
$
其中线弹性取 $beta = beta_"LE"$、$(s, m) = (s_(bold(u)), 1)$，板弯曲取 $beta = beta_"KL"$、$(s, m) = (s_w, 2)$、$bold(f)$ 换作 $f$ 且无 $B Q_"tr"^(-1/2)$ 一项；线弹性在盒状区域上取闭式箱均值时该项亦为零（@cor:exact-trace）。Ritz 投影积分阶以 $Q_"R" >= C_delta C_("R",K)^(m) log(2K/eta)$ 的 Gram 谱稳定样本条件为前提；对准均匀三次 B 样条，$C_("R",K)^(m) lt.tilde K$。训练项中的 $K$ 来自投影字典包络 $C_(Pi,K) lt.tilde K^(1/2)$（@eq:envelope-rate），与 Ritz 投影积分项中的 $K$ 同源于同一条 B 样条逆估计。至此四类误差全部定阶，余下只有 $epsilon_"opt"$：它是球约束凸最小二乘的经验次优性，direct 后端解至求解器容差时该项为机器精度量级，rcond 截断或岭正则解则引入真实的非零次优性，后文的求解策略对照正是对该项的直接观测。线弹性中的 Lamé 一致性还要求 $B$、目标各分量的 Sobolev 范数以及 $cal(E)_("R",delta)$ 上的 Ritz 谱稳定常数都能关于 $lambda$ 一致控制；包络 $C_(Pi,K)$ 的上述定阶只依赖 B 样条次数与网格准均匀性，本身已关于 $lambda$ 一致。连续最小二乘稳定性本身的一致性不能自动推出其余离散统计量的一致性。

= 数值实现与实验方案

== 投影与抽样

计算区域取单位方形或单位立方体 $Omega = (0,1)^d$，偏置半径取 $c_b = 2$；由 $sup_(bold(x) in overline(Omega)) norm(bold(x))_(ell^2) = sqrt(d) <= sqrt(3) < 2$，$d in {2, 3}$ 时字典参数域的包含条件成立。所有原始特征使用 $rho_3$，单隐层参数为上述确定性点集：相对偏置取等距中点层，每层配置各自的 $bb(S)^(d-1)$ 准均匀方向集，$d=2$ 取等距角网格，$d=3$ 取 Fibonacci 球面格；此外固定保留 $dim P_3 (Omega) - 1$ 个由列主元 QR 选出的多项式方向。Fibonacci 球面格的准均匀性见 @HardinMichaelsSaff2016，不由 @LiuMaoXu2025 推论 2.1 的存在性构造给出；程序仍对每个实际使用的 $N$ 报告该点集的覆盖半径与最小间距，作为该性质在有限 $N$ 处的验证。训练积分使用 $Q$ 个独立均匀 Monte Carlo 点；Ritz 投影使用与训练规则独立的 $Q_"R"$ 个均匀 Monte Carlo 点；平均迹投影不用抽样规则，而按 @cor:exact-trace 的闭式箱均值精确构造；测试误差使用与上述规则都无关的张量积 Gauss--Legendre 求积。线弹性位移采用 $H^1$ Ritz 投影，板挠度采用 $H^2$ Ritz 投影。程序报告以下诊断量：

- 非退化参数经 $frak(P)$ 归一化后的覆盖半径与最小间距，前者在 $frak(P)(cal(P)_Omega)$ 上以固定探针集估计，后者为这些参数的精确两两最小测地距离；
- 辅助空间的实际维数 $K$；
- 投影 Gram 方程相对残差；
- 边界值残差，以及板问题中的法向导数残差；
- 离散系数上的平均迹规范残差（应为舍入量级）、确定性测试规则上的连续平均迹，以及应力误差的偏差与球分量。

训练矩阵不作列归一化。约束半径为 $B/sqrt(m_N)$，并记录系数球是否激活。后端由配置中的 $"system_backend" in {"direct","gram"}$ 显式选择，默认取 direct 后端；$"direct_solver" in {"dense","streaming_tsqr"}$ 默认取流式 TSQR，在不保留完整加权残差矩阵的条件下逐批压缩为等价的小型最小二乘问题。dense 方式保留作小规模高精度校验，Gram 后端保留作算法对照，二者均不作静默切换。

== 主实验与独立重复

主收敛实验取
$
  N in {200,400,600,800,1000},
  quad Q=4(N+1).
$
单隐层字典为确定性点集，不含特征随机性；每个配置使用十组相互独立的训练样本、Ritz 投影积分与平均迹积分种子。测试求积固定为确定性规则。每个指标报告十次独立结果的均值与样本标准差，并同时保存逐次运行的 JSON 与 CSV 数据；不能把一次随机运行当作最终表格。

系数预算从预先给定的有限候选集选择。对制造解实验，每个候选 $B$ 只在与训练集、测试集均分离的验证求积上比较图误差；没有精确解时则比较验证残差。选定 $B$ 后才在固定测试规则上报告误差。近不可压缩参数扫描对所有 $nu$ 使用同一个已选预算，避免随材料参数重新调参掩盖可能的退化。

== 消融实验

为区分三个离散尺度的作用，设置以下消融：

- 训练样本消融：固定 $N=400$，取
  $Q/(N+1) in {1,2,4,8,16,32}$；
- Ritz 维数消融：对二维线弹性和板弯曲固定 $N=400$，取
  $K/(N+1) in {1,2,4}$；
- 近不可压缩消融：在三维线弹性中取
  $nu in {0.49,0.499,0.4999,0.49999,0.499999}$，保持特征、预算选择规则与测试求积一致。

线弹性主要报告
$
  norm(bold(sigma)_star-bold(sigma)_N)_(bold(H)(div)),
  quad
  norm(bold(u)_star-bold(u)_N)_(H^1),
$
及本构、平衡连续残差。板问题主要报告
$
  norm(bold(M)_star-bold(M)_N)_(bold(H)(div div)),
  quad
  norm(w_star-w_N)_(H^2),
$
及对应两项连续残差。$L^2$ 分量误差作为辅助指标保留。所有误差都在确定性测试求积上计算；训练残差不能代替测试图误差。

== 默认配置的单次诊断

下列图比较四个主程序在各自默认配置下的一次运行。它们用于检查系数球、岭正则与截断 SVD 三种线性代数求解策略在同一个 direct + streaming TSQR 残差系统上的行为，不替代主收敛实验所要求的十次独立重复及均值--标准差汇总。

#figure(
  image("/public/images/least-squares/linear-elasticity-2d/graph-error-summary.png"),
  caption: [二维线弹性默认配置下的图范数误差诊断],
)<fig:least-squares-elasticity-2d-diagnostic>

#figure(
  image("/public/images/least-squares/linear-elasticity-3d/graph-error-summary.png"),
  caption: [三维线弹性默认配置下的图范数误差诊断],
)<fig:least-squares-elasticity-3d-diagnostic>

#figure(
  image("/public/images/least-squares/plane-stress/graph-error-summary.png"),
  caption: [平面应力默认配置下的图范数误差诊断],
)<fig:least-squares-plane-stress-diagnostic>

#figure(
  image("/public/images/least-squares/plate-bending/graph-error-summary.png"),
  caption: [Kirchhoff--Love 板弯曲默认配置下的图范数误差诊断],
)<fig:least-squares-plate-diagnostic>

== 实验程序对应关系

四个主程序分别位于二维线弹性、三维线弹性、平面应力与板弯曲目录。它们共享同一个确定性准均匀特征生成器、B 样条 Ritz 投影器、均匀 Monte Carlo 规则、物理系数球最小二乘求解器与 direct/Gram 后端；线弹性类程序还共享 @cor:exact-trace 的闭式平均迹投影。容量消融脚本对每个 $N$ 重新生成相应的训练点及对应维数的 Ritz 规则，避免复用固定训练集导致名义上的 $Q$ 与实际样本数不一致。

完整尺度的十次重复与两组消融计算量显著高于单次主程序。论文表格应由聚合文件自动生成；在这些计算实际完成以前，不沿用旧激活函数、旧边界因子或随机抽样参数产生的数值作为本文结果。

= 结语

本文的主要结果是线弹性与板弯曲的两条残差稳定性定理：混合最小二乘泛函与相应图范数平方双边等价，且等价常数关于 Lamé 常数 $lambda$ 一致，在近不可压缩极限下不退化。在此基础上，本文将参数准均匀的单隐层线性化 ReLU 三次幂网络用于两类方程的数值求解，并给出误差分析；字典逼近项是确定性的 $N^(-beta)$，没有独立抽样带来的对数修正，有限维计算必须显式计入 Ritz 截断和投影积分误差。

误差界同时展示了 $N$、$Q$、$K$、$B$ 与代数求解精度的不同作用：$N$ 控制单隐层逼近，$K$ 控制投影空间，$Q$ 控制经验泛函，而 $B$ 连接逼近表示与泛化控制。特别地，平方风险中的 Monte Carlo 项是 $Q^(-1/2)$，对应图范数均方根误差中的 $Q^(-1/4)$。这一区分也是数值消融应分别改变 $N$、$Q$ 与 $K$ 的原因。

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
  在 $t = 0$ 处取极小值。由上面的二次展开对 $t$ 求导并取 $t = 0$，得到
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
  在 $t = 0$ 处取极小值。由上面的二次展开对 $t$ 求导并取 $t = 0$，得到
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
