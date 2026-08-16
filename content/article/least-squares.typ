#import "/typ/templates/blog.typ": *

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

本文的另一项工作是将单隐层线性化 ReLU 三次幂网络用于两类方程的数值求解，并给出误差分析：总误差分为字典逼近误差、有限维 Ritz 及其积分误差、训练泛函的 Monte Carlo 误差和线性代数求解误差四类。准均匀参数下线性化 ReLU 幂网络的最优 Sobolev 逼近率取自 @LiuMaoXu2025；字典参数化与经验风险分解框架参考 @SiegelHongJinHaoXu2023。系数约束同时用于保持表示有界并控制经验泛函的泛化误差。

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
本文以下标 $"LE"$（linear elasticity，线弹性）与 $"KL"$（Kirchhoff--Love 板弯曲）区分两类问题的算子、泛函与函数空间。使用 $(bold(sigma), bold(u))$ 表示线弹性未知量，使用 $(bold(M), w)$ 表示板弯曲未知量；$bold(T)_alpha$ 表示对称张量基。对非负量 $A$ 与 $B$，记 $A lt.tilde B$ 表示存在常数 $C > 0$ 使 $A <= C B$，其中 $C$ 不依赖于 Lamé 常数 $lambda$ 与任何离散参数，具体依赖在上下文说明。

== 线弹性

线弹性的应力--位移系统为
$
  cases(
    cal(A)_"LE" bold(sigma) - bold(epsilon)(bold(u)) = bold(0) & "in" Omega,
    div bold(sigma) + bold(f) = bold(0) & "in" Omega,
    bold(u) = bold(0) & "on" partial Omega.
  )
$
其中未知量为应力 $bold(sigma)$ 与位移 $bold(u)$，$bold(f) in L^2(Omega; RR^d)$ 为给定体力；第一个方程为本构关系，第二个方程为平衡方程，第三个方程为纯位移边界条件。$bold(epsilon)$ 为对称梯度算子，
$
  bold(epsilon)(bold(v)) := 1/2 (nabla bold(v) + nabla bold(v)^T).
$
$cal(A)_"LE":op("Sym")(d)->op("Sym")(d)$ 为柔度算子，即各向同性刚度算子 $bold(tau) |-> 2 mu bold(tau) + lambda tr(bold(tau)) bold(I)$ 之逆，其中 $mu > 0$ 与 $lambda >= 0$ 为 Lamé 常数，$tr(bold(tau))$ 为迹，$bold(I)$ 为单位张量；显式地，
$
  cal(A)_"LE" bold(tau)
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
  := norm(cal(A)_"LE" bold(tau)-bold(epsilon)(bold(v)))_(L^2)^2
  + norm(div bold(tau)+bold(f))_(L^2)^2.
$

对应双线性形式记为
$
  a_"LE" ((bold(sigma), bold(u)), (bold(tau), bold(v)))
  := (cal(A)_"LE" bold(sigma) - bold(epsilon)(bold(u)),
    cal(A)_"LE" bold(tau) - bold(epsilon)(bold(v)))_(L^2(Omega)) + (div bold(sigma), div bold(tau))_(L^2(Omega)),
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
  存在常数 $0 < c <= C < oo$，仅依赖于 $Omega$、$mu$ 与 $d$，且可关于 Lamé 常数 $lambda$ 一致选取，使得对任意 $(bold(tau), bold(v)) in bold(Sigma)_"LE" times bold(U)_"LE"$ 都有
  $
    c (
      norm(bold(tau))_(bold(H)(div))^2
      + norm(bold(v))_(H^1(Omega))^2
    )
    <= cal(J)_"LE" (bold(tau), bold(v); bold(0))
    <= C (
      norm(bold(tau))_(bold(H)(div))^2
      + norm(bold(v))_(H^1(Omega))^2
    ).
  $
]<thm:elasticity-stability>

#proof[
  先证上界。令 $bold(f) = bold(0)$。由柔度算子的显式公式
  $
    cal(A)_"LE" bold(tau)
    = 1/(2 mu) (
      bold(tau) - lambda/(2 mu + d lambda) tr(bold(tau)) bold(I)
    ),
  $
  以及估计 $norm(tr(bold(tau)) bold(I))_(L^2(Omega)) <= d norm(bold(tau))_(L^2(Omega))$，可得
  $
    norm(cal(A)_"LE" bold(tau))_(L^2(Omega)) & <= 1/(2 mu) norm(bold(tau))_(L^2(Omega))
                                               + lambda/(2 mu (2 mu + d lambda))
                                               norm(tr(bold(tau)) bold(I))_(L^2(Omega)) \
                                             & <= (
                                                 1/(2 mu) + d lambda/(2 mu (2 mu + d lambda))
                                               ) norm(bold(tau))_(L^2(Omega)) \
                                             & <= 1/mu norm(bold(tau))_(L^2(Omega)).
  $
  再由 $(a + b)^2 <= 2 a^2 + 2 b^2$ 与 $norm(bold(epsilon)(bold(v)))_(L^2(Omega)) <= norm(nabla bold(v))_(L^2(Omega))$，可得
  $
    cal(J)_"LE" (bold(tau), bold(v); bold(0)) & <= 2 norm(cal(A)_"LE" bold(tau))_(L^2(Omega))^2
                                                + 2 norm(bold(epsilon)(bold(v)))_(L^2(Omega))^2
                                                + norm(div bold(tau))_(L^2(Omega))^2 \
                                              & <= C (
                                                  norm(bold(tau))_(bold(H)(div))^2
                                                  + norm(bold(v))_(H^1(Omega))^2
                                                ).
  $
  其中上界常数 $C$ 不依赖于 $lambda$。

  再证下界，即关于 $lambda$ 一致有界的强制性。这里只在下界证明中引入偏差投影
  $
    bold(tau)^"D" := bold(tau) - 1/d tr(bold(tau)) bold(I),
    quad
    bold(epsilon)^"D" (bold(v)) := bold(epsilon)(bold(v)) - 1/d (div bold(v)) bold(I),
  $
  以及各向同性柔度算子的偏差-体积分解
  $
    cal(A)_"LE" bold(tau) = 1/(2 mu) bold(tau)^"D" + 1/(d(2 mu + d lambda)) tr(bold(tau)) bold(I).
  $
  由偏差部分与球张量部分在 $L^2(Omega; op("Sym")(d))$ 中的正交性，
  $
    norm(cal(A)_"LE" bold(tau) - bold(epsilon)(bold(v)))_(L^2(Omega))^2
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
  由 Korn 第一不等式（参考 @BrennerScott2008 推论 11.2.25），
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
    integral_Omega bold(epsilon)^"D" (bold(v)) : bold(tau) dif x
    &= integral_Omega bold(epsilon)(bold(v)) : bold(tau) dif x
    - 1/d integral_Omega (div bold(v)) tr(bold(tau)) dif x
    \
    &= - integral_Omega bold(v) dot (div bold(tau)) dif x
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
  即得下界，且下界常数 $c$ 也可不依赖于 $lambda$。证毕。
]

由 @thm:elasticity-stability，双线性形式 $a_"LE"$ 在 $bold(Sigma)_"LE" times bold(U)_"LE"$ 上连续且强制，其中连续性与强制性常数均可关于 Lamé 常数 $lambda$ 一致选取。特别地，近不可压缩极限 $lambda -> oo$ 下稳定常数不退化；只要离散空间包含于 $bold(Sigma)_"LE" times bold(U)_"LE"$ 且其逼近误差本身保持稳定，离散解不会因该极限而产生锁死。平均迹规范排除了常值球应力核。

== Kirchhoff--Love 板弯曲

以下令 $d=2$。Kirchhoff--Love 板的弯矩--挠度系统为
$
  cases(
    cal(A)_"KL" bold(M)-bold(kappa)(w)=bold(0) & "in" Omega,
    div div bold(M)+f=0 & "in" Omega,
    w=0 \, quad partial_n w=0 & "on" partial Omega.
  )
$
其中未知量为弯矩 $bold(M)$ 与挠度 $w$，$f in L^2(Omega)$ 为给定横向载荷；第一个方程为本构关系，第二个方程为平衡方程，第三个方程为固支边界条件。曲率张量固定记为
$
  bold(kappa)(w):=-nabla^2 w.
$
$cal(A)_"KL":op("Sym")(2)->op("Sym")(2)$ 为各向同性板柔度算子，
$
  cal(A)_"KL" bold(tau)
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
有关 $bold(H)(div div)$ 弯矩--挠度模型可参见 @FuhrerHeuerNiemi2019 与 @FuhrerHeuer2024；相应最小二乘泛函为
$
  cal(J)_"KL" (bold(tau),v;f)
  := norm(cal(A)_"KL" bold(tau)-bold(kappa)(v))_(L^2)^2
  + norm(div div bold(tau)+f)_(L^2)^2.
$

双线性形式为
$
  a_"KL" ((bold(M), w), (bold(tau), v))
  := (cal(A)_"KL" bold(M) - bold(kappa)(w),
    cal(A)_"KL" bold(tau) - bold(kappa)(v))_(L^2(Omega)) + (div div bold(M), div div bold(tau))_(L^2(Omega)),
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
  在上述各向同性薄板参数假设下，存在常数 $0 < c <= C < oo$，仅依赖于 $Omega$、$mu$ 与 $h$，且可关于 Lamé 常数 $lambda$ 一致选取，使得对任意 $(bold(tau), v) in bold(Sigma)_"KL" times U_"KL"$ 都有
  $
    c (
      norm(bold(tau))_(bold(H)(div div))^2
      + norm(v)_(H^2(Omega))^2
    )
    <= cal(J)_"KL" (bold(tau), v; 0)
    <= C (
      norm(bold(tau))_(bold(H)(div div))^2
      + norm(v)_(H^2(Omega))^2
    ).
  $
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
    cal(A)_"KL" bold(tau)
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
    norm(cal(A)_"KL" bold(tau))_(L^2(Omega))^2
    = (1/(D(1-nu)))^2 norm(bold(tau)^"D")_(L^2(Omega))^2
    + 1/(2 D^2 (1+nu)^2) norm(tr(bold(tau)))_(L^2(Omega))^2
    <= 36/(mu^2 h^6) norm(bold(tau))_(L^2(Omega))^2,
  $
  以及
  $
    2/(mu h^3) norm(bold(tau))_(L^2(Omega))^2
    <= integral_Omega (cal(A)_"KL" bold(tau)) : bold(tau) dif x
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
    norm(cal(A)_"KL" bold(tau) - bold(kappa)(v))_(L^2(Omega))^2
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
    bold(r)_"c" := cal(A)_"KL" bold(tau) - bold(kappa)(v),
    quad
    r_"e" := div div bold(tau).
  $
  则
  $
    bold(kappa)(v) = cal(A)_"KL" bold(tau) - bold(r)_"c".
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
    <= integral_Omega (cal(A)_"KL" bold(tau)) : bold(tau) dif x.
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
  即得下界。由于 $c_A$ 与 $C_A$ 仅依赖于 $mu$ 与 $h$，故上、下界常数 $c$ 与 $C$ 也仅依赖于 $Omega$、$mu$ 与 $h$，并可关于 $lambda$ 一致选取。证毕。
]

由 @thm:plate-stability，双线性形式 $a_"KL"$ 在 $bold(Sigma)_"KL" times U_"KL"$ 上连续且强制，其中常数可关于 Lamé 常数 $lambda$ 一致选取。值得对比的是：线弹性中柔度算子的体积部分在 $lambda -> oo$ 时退化，一致稳定性依赖偏差-体积分解与迹估计；板弯曲中柔度算子本身即一致椭圆，一致稳定性转而依赖 Poincaré 不等式从曲率恢复挠度范数。

= 线性化网络离散

== 单隐层字典

令 $tilde(bold(x)):=(bold(x),1) in RR^(d+1)$，并定义 ReLU 三次幂激活
$
  rho_3(t):=max(t,0)^3.
$
字典的参数域取方向--偏置柱面 $bb(S)^(d-1) times [-c_b, c_b]$，其中 $bb(S)^(d-1) subset RR^d$ 为单位球面，偏置半径固定为 $c_b = 2$；由 $Omega = (0,1)^d$ 有
$
  sup_(bold(omega) in bb(S)^(d-1), bold(x) in overline(Omega)) abs(bold(omega) dot bold(x)) = sqrt(d) < c_b,
$
因此与 $overline(Omega)$ 相交的仿射超平面 ${bold(omega) dot bold(x) + b = 0}$ 的参数都落在参数域内部。以 $chi$ 标记物理变量：线弹性中 $chi in {bold(sigma), bold(u)}$，板弯曲中 $chi in {bold(M), w}$。每个物理变量独自生成单隐层参数，即各自取该域上的确定性张量点集
$
  Theta_(chi,N) := {(bold(omega)_(chi,j), b_(chi,j))}_(j=1)^N subset bb(S)^(d-1) times [-c_b, c_b]:
$
偏置取 $n_2 tilde.eq N^(1/d)$ 个等距区间的中点，每个偏置层配置各自的 $bb(S)^(d-1)$ 准均匀方向集（$d=2$ 为等距角网格，$d=3$ 为 Fibonacci 球面格），各层点数至多相差一且总数为 $N$。变量 $chi$ 的特征及常数特征分别为
$
  xi_(chi,j) (bold(x)) := rho_3(bold(omega)_(chi,j) dot bold(x) + b_(chi,j)),
  quad xi_(chi,0) := 1.
$

归一化映射
$
  frak(P)(bold(omega), b) := ((bold(omega), b))/sqrt(1 + b^2)
$
把 $bb(S)^(d-1) times [-c_b, c_b]$ 双向 Lipschitz 地映为增广球面 $bb(S)^d subset RR^(d+1)$ 的赤道带 ${bold(theta) in bb(S)^d : abs(theta_(d+1)) <= c_b\/sqrt(1+c_b^2)}$，Lipschitz 常数仅依赖 $c_b$；由 $rho_3$ 的正三次齐次性，
$
  xi_(chi,j) (bold(x)) = (1+b_(chi,j)^2)^(3/2) rho_3(frak(P)(bold(omega)_(chi,j), b_(chi,j)) dot tilde(bold(x))),
$
两种参数化张成同一空间。张量构造保证每个 $frak(P)(Theta_(chi,N))$ 在该赤道带上的覆盖半径满足（参见 @LiuMaoXu2025 推论 2.1；$d=3$ 时 Fibonacci 球面格的覆盖半径由程序数值验证）
$
  h_(chi,N) := max_(bold(theta) in frak(P)(bb(S)^(d-1) times [-c_b, c_b]))
  min_(1<=j<=N) op("dist")(bold(theta), frak(P)(bold(omega)_(chi,j), b_(chi,j)))
  lt.tilde N^(-1/d),
$
其中 $op("dist")$ 为 $bb(S)^d$ 上的测地距离。带外极冠方向的特征在 $Omega$ 上退化为三次多项式或零，按 @LiuMaoXu2025 注 2.1 由有限个多项式方向补足，不影响下文的逼近阶。

记 $n_s=d(d+1)/2$，取 $op("Sym")(d)$ 的固定正交基
${bold(T)_alpha}_(alpha=1)^(n_s)$ 与 $RR^d$ 的标准基
${bold(e)_i}_(i=1)^d$。线弹性的原始空间是简单的逐分量张成空间
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
它是从 $bold(H)(div)$ 到 $bold(Sigma)_"LE"$ 的有界投影。对 $m in {1,2}$，令 $Pi_D^(m)$ 为 $H^m (Omega)$ 到闭子空间 $H_0^m (Omega)$ 的正交投影：
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
因此求解变量仍是原始简单张成空间中的系数，物理残差则在投影后的场上计算。这里没有边界软残差及其权重选择问题。该构造也不损失字典的逼近阶：若 $v_star in H_0^m (Omega)$、$m in {1,2}$，则 $Pi_D^(m) v_star = v_star$，从而对任意 $hat(v)_N in H^m (Omega)$ 有
$
  norm(v_star-Pi_D^(m) hat(v)_N)_(H^m)
  =norm(Pi_D^(m)(v_star-hat(v)_N))_(H^m)
  <=norm(v_star-hat(v)_N)_(H^m).
$
<eq:projection-transfer>

这个传递估计说明，不需要研究 $v_star$ 除以某个边界消失因子后的正则性；所需假设仅是物理变量本身的 Sobolev 正则性。椭圆型投影带来的理论收益正是这一范数为一的稳定传递。代价是实际计算中必须近似求解投影问题。

== 有限维 Ritz 实现

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

数值实验采用一组与训练点独立的 $Q_"R"$ 个均匀样本离散 $H^m$ 内积。由此得到的经验投影记作
$tilde(Pi)_(D,K,Q_"R")^(m)$。平均迹也不使用训练样本估计，而是另取一组与训练规则及 Ritz 规则都独立的均匀 Monte Carlo 点
${bold(y)_r}_(r=1)^(Q_"tr")$，定义
$
  tilde(Pi)_("tr",Q_"tr") bold(tau)
  := bold(tau)-1/d (
    1/Q_"tr" sum_(r=1)^(Q_"tr") tr(bold(tau)(bold(y)_r))
  ) bold(I).
$
实现中先计算每个原始应力特征在这组点上的样本均值，再消去一个常数球应力自由度；因此离散系数对该独立规则精确满足零平均迹规范，而不会把训练样本的积分波动固化为应力中的常数球分量。记 Ritz 内积误差为 $epsilon_"Rquad"$，平均迹积分误差为 $epsilon_"trquad"$，并合记
$
  epsilon_"projquad"^2
  := epsilon_"Rquad"^2+epsilon_"trquad"^2.
$
这里的 $epsilon_"trquad"$ 具体控制系数球上一致的投影差
$
  sup_(bold(c) in cal(C)_(N,B))
  norm((tilde(Pi)_("tr",Q_"tr")-Pi_"tr")
  hat(bold(tau))_bold(c))_(bold(H)(div));
$
该差是常数球张量，故其散度为零，只需控制独立 MC 样本均值与连续平均值之差。分析时先把经验投影场与精确 $Pi_"tr"$ 投影场比较，再对后者使用 @thm:elasticity-stability；这正是平均迹积分误差必须进入 $epsilon_"projquad"$ 的原因。

== 系数约束

把给定模型的全部独立输出系数依固定顺序排成向量 $bold(c) in RR^(m_N)$，其中 $m_N$ 是离散系数总数。对给定预算 $B>0$，定义
$
  cal(C)_(N,B)
  := {bold(c) in RR^(m_N):sqrt(m_N) norm(bold(c))_2<=B}.
$
这个缩放即 @LiuMaoXu2025 线性化网络类中的系数约束，与 ReLU 幂网络的有界表示相容，并避免同一物理函数仅因特征数改变而获得不同的系数尺度。系数约束施加在未作列归一化的物理系数上。

= 经验最小二乘与误差分析

== 训练泛函

令 ${bold(x)_ell}_(ell=1)^Q$ 为 $Omega$ 上独立均匀样本，并设 $abs(Omega)$ 为区域测度。对 $bold(z)=(bold(tau),bold(v)) in bold(X)_"LE"$，把线弹性的两个残差依次记为
$
  bold(r)_"c" (bold(z))
  :=cal(A)_"LE" bold(tau)-bold(epsilon)(bold(v)),
  quad
  bold(r)_"e" (bold(z))
  :=div bold(tau)+bold(f).
$
经验泛函为
$
  cal(J)_("LE",Q)(bold(z))
  :=abs(Omega)/Q sum_(ell=1)^Q (
    abs(bold(r)_"c" (bold(z))(bold(x)_ell))_F^2
    +abs(bold(r)_"e" (bold(z))(bold(x)_ell))_2^2
  ).
$
对板问题，取
$
  bold(r)_"c" (bold(z))
  :=cal(A)_"KL" bold(tau)-bold(kappa)(v),
  quad
  r_"e" (bold(z)):=div div bold(tau)+f,
$
经验泛函为
$
  cal(J)_("KL",Q)(bold(z))
  :=abs(Omega)/Q sum_(ell=1)^Q (
    abs(bold(r)_"c" (bold(z))(bold(x)_ell))_F^2
    +abs(r_"e" (bold(z))(bold(x)_ell))^2
  ).
$
把所有加权残差写成设计矩阵 $bold(R)_(N,Q)$ 与右端
$bold(b)_(N,Q)$ 后，训练问题是
$
  min_(bold(c) in cal(C)_(N,B))
  norm(bold(R)_(N,Q) bold(c)-bold(b)_(N,Q))_2^2.
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
  norm(bold(a))_2
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
其中 $beta=beta_"LE"$ 或 $beta_"KL"$。该上界是确定性的：既无独立抽样带来的对数修正，也无需对参数取期望。系数界源于构造中的带限反卷积：把球谐展开截断到次数 $tilde.eq h_(chi,N)^(-1)$，再逐频带除以 $rho_3$ 的 Legendre 系数；当 $s_chi < s_"cap" (d)$ 时该步骤把系数范数放大至多 $h_(chi,N)^(-(s_"cap" (d)-s_chi))$ 倍。系数规模决定预算：当最不光滑分量满足 $s_chi = s_"cap" (d)$ 时，预算 $B$ 可取与 $N$ 无关的常数；一般情形需把 $B$ 取到 $N^((s_"cap" (d) - s_chi)\/d)$ 的量级。@SiegelHongJinHaoXu2023 对单隐层 PDE 离散中的逼近、积分与优化误差分解提供了另一种分析框架。

== Monte Carlo 训练误差

为了对经验泛函作一致控制，需要对投影后的字典作明确假设。记 $cal(D)_(N,K)$ 为构成两个物理残差的所有投影特征及其所需导数。假设存在有限常数 $C_(Pi,K)$，使
$
  sup_(g in cal(D)_(N,K))
  norm(g)_(L^oo(Omega))<=C_(Pi,K).
$
对线弹性，这一包络涉及应力的一阶导数和位移的一阶导数；对板问题则涉及弯矩与挠度的二阶导数。有限维经验 Ritz 投影一般会使 $C_(Pi,K)$ 依赖 $K$ 与投影 Gram 矩阵的稳定性，因而不能无条件把泛化常数写成与 $K$ 无关。

在上述包络、数据有界性和系数预算下，Rademacher 或经验过程估计给出平方风险层面的界
$
  EE sup_(bold(c) in cal(C)_(N,B))
  abs(cal(J)(bold(c))-cal(J)_Q(bold(c)))
  <= C_(B,bold(f),Pi,K) Q^(-1/2).
$
因此，当最终误差以图范数的均方根表示时，仅由经验积分产生的项是 $Q^(-1/4)$，而不是 $Q^(-1/2)$。两种指数对应不同层次的量，必须加以区分。

== 总误差估计

令 $bold(z)_star$ 表示连续精确解；$bold(z)_(N,Q,K)$ 表示用 $N$ 个字典特征、$Q$ 个训练点和 $K$ 维 Ritz 辅助空间得到的物理解。以
$epsilon_"Ritz" (K)$ 表示精确有限维 Ritz 误差，以
$epsilon_"projquad"$ 表示投影内积及平均迹积分的离散误差，以
$epsilon_"opt"$ 表示经验目标值意义下的代数求解次优性。下述定理把四类误差合并为总误差估计。

#theorem(title: [线性化网络经验最小二乘误差])[
  假设相应连续残差稳定性成立；单隐层参数取上述准均匀字典；精确解各分量满足上述 Sobolev 正则性；投影字典满足点态导数包络；预算 $B$ 足以容纳一个达到所述逼近阶的比较函数。则
  $
    EE norm(bold(z)_star-bold(z)_(N,Q,K))_bold(X)^2
    <= C (N^(-2 beta)
    +epsilon_"Ritz" (K)^2
    +epsilon_"projquad"^2
  $
  $
    +C_(B,bold(f),Pi,K) Q^(-1/2)
    +epsilon_"opt").
  $
  在线弹性中取 $bold(X)=bold(X)_"LE"$、$beta=beta_"LE"$；在板问题中取 $bold(X)=bold(X)_"KL"$、$beta=beta_"KL"$。
]<thm:total-error>

#proof[
  先由 @thm:elasticity-stability 或 @thm:plate-stability 把图范数误差控制为总体最小二乘残差。以达到字典逼近率且满足预算的比较函数代入经验极小性；再在经验泛函与总体泛函之间插入一致 Monte Carlo 偏差。投影传递估计 @eq:projection-transfer 把原始字典的 Sobolev 逼近误差传递给连续投影，有限维 Ritz 恒等式产生 $epsilon_"Ritz"$，经验投影与平均迹数值积分产生 $epsilon_"projquad"$。最后加入经验问题未被精确求解时的 $epsilon_"opt"$。字典参数是确定性的；上述期望可理解为条件于两组独立投影规则、只对训练样本取，若再对投影规则取期望，则相应随机性由 $epsilon_"projquad"$ 的矩界吸收。
]

若用均方根图误差表述 @thm:total-error，则各项开方后分别具有量级
$
  N^(-beta),
  quad epsilon_"Ritz" (K),
  quad epsilon_"projquad",
  quad Q^(-1/4),
  quad epsilon_"opt"^(1/2).
$
线弹性中的 Lamé 一致性还要求 $B$、目标各分量的 Sobolev 范数、投影字典包络 $C_(Pi,K)$ 以及 Ritz 稳定常数都能关于 $lambda$ 一致控制。连续最小二乘稳定性本身的一致性不能自动推出这些离散统计量的一致性。

= 数值实现与实验方案

== 投影与抽样

计算区域取单位方形或单位立方体。所有原始特征使用 $rho_3$，单隐层参数为上述确定性张量点集：偏置取 $[-2,2]$ 上的等距中点层，每层配置各自的 $bb(S)^(d-1)$ 准均匀方向集（$d=2$ 等距角网格，$d=3$ Fibonacci 球面格）。训练积分使用 $Q$ 个独立均匀 Monte Carlo 点；Ritz 投影与平均迹投影分别使用相互独立的 $Q_"R"$ 与 $Q_"tr"$ 个均匀 Monte Carlo 点；测试误差使用与这三组规则都无关的张量积 Gauss--Legendre 求积。线弹性位移采用 $H^1$ Ritz 投影，板挠度采用 $H^2$ Ritz 投影。程序报告以下诊断量：

- 参数点集经 $frak(P)$ 归一化后的覆盖半径与最小间距；
- 辅助空间的实际维数 $K$；
- 投影 Gram 方程相对残差；
- 边界值残差，以及板问题中的法向导数残差；
- 独立 $Q_"tr"$ 规则上的规范残差、确定性测试规则上的连续平均迹，以及应力误差的偏差与球分量。

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

四个主程序分别位于二维线弹性、三维线弹性、平面应力与板弯曲目录。它们共享同一个确定性准均匀特征生成器、B 样条 Ritz 投影器、均匀 Monte Carlo 规则、物理系数球最小二乘求解器与 direct/Gram 后端；线弹性类程序还共享独立平均迹投影。容量消融脚本对每个 $N$ 重新生成相应的训练点及对应维数的 Ritz 与平均迹规则，避免复用固定训练集导致名义上的 $Q$ 与实际样本数不一致。

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
    cal(A)_"LE" bold(sigma) - bold(epsilon)(bold(u)) = 0,
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
    cal(A)_"KL" bold(M) - bold(kappa)(w) = 0,
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
