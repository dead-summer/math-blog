#import "/typ/templates/blog.typ": *

#show: main-zh.with(
  title: "面向弹性问题的固定随机特征混合最小二乘离散方法",
  author: "summer",
  desc: [提出一种面向弹性问题的固定随机特征混合最小二乘方法，分别建立线弹性与 Kirchhoff-Love 板弯曲问题的理论分析，并在三维线弹性、二维线弹性、近不可压缩三维线弹性、平面应力与板弯曲算例中进行验证],
  date: "2026-04-20",
  tags: (
    blog-tags.numerical-methods,
    blog-tags.pde,
  ),
  show-outline: true,
)

= 引言

在固定随机特征方法中，隐藏层参数首先随机生成并冻结，随后仅对输出层系数求解。由此，偏微分方程的近似被转化为有限维线性代数问题。对于弹性问题而言，混合表示与模型结构相匹配：本构关系与平衡关系可以被显式分离，应力、弯矩等辅助变量也可以被独立近似。

本文构造一种面向弹性问题的固定随机特征混合最小二乘方法，并分别研究线弹性与 Kirchhoff-Love 板弯曲两个模型。两类问题对应不同的混合未知量、函数空间与平衡算子：线弹性采用应力-位移混合形式，板弯曲采用弯矩-挠度混合形式；相应的稳定性分析、保形随机特征离散与准最优误差估计也分别在各自的函数空间中建立。

在具体分析上，本文主要建立两类模型的稳定性与离散误差估计。其一，在线弹性问题中证明应力-位移混合最小二乘泛函与 $bold(H)(div, Omega; SS^d) times H_0^1(Omega)$ 范数等价，且稳定常数可关于 Lamé 常数一致选取，从而覆盖近不可压缩情形。其二，在 Kirchhoff-Love 板弯曲问题中证明弯矩-挠度混合最小二乘泛函与 $bold(H)(div div, Omega; SS^2) times H_0^2(Omega)$ 范数等价，明确四阶问题的稳定性机制。离散层面则采用保形固定随机特征子空间，并在三维线弹性、二维线弹性、近不可压缩三维线弹性、平面应力与板弯曲算例中进行验证。

= 线弹性问题

== 应力-位移混合形式

设 $Omega subset RR^d, d in {2, 3}$ 为有界、连通且边界 Lipschitz 的区域，未知量为对称应力张量 $bold(sigma): Omega -> SS^d$ 与位移向量 $bold(u): Omega -> RR^d$。数值实验章还将单独讨论与之同属二阶系统的二维平面应力模型。考虑如下方程组
$
  cases(
    bold(cal(A)) : bold(sigma) - bold(epsilon)(bold(u)) & = 0 & quad "in" Omega,
    nabla dot bold(sigma) + bold(f) & = 0 & quad "in" Omega,
    bold(u) & = 0 & quad "on" partial Omega,
  )
$
其中
$
  bold(cal(A)) : bold(sigma) = 1/(2 mu) (bold(sigma) - lambda / (2 mu + d lambda) tr(bold(sigma)) bold(I)),
$
这里 $bold(cal(A))$ 为 $SS^d$ 上有界且一致正定的柔度算子，即存在常数 $0 < c_A <= C_A < oo$ 使得
$
  c_A norm(bold(tau))_(L^2(Omega))^2
  <= integral_Omega (bold(cal(A)) : bold(tau)) : bold(tau) dif x
  <= C_A norm(bold(tau))_(L^2(Omega))^2,
  quad forall bold(tau) in L^2(Omega; SS^d).
$
线性应变张量 $bold(epsilon)(bold(u))$ 为：
$
  bold(epsilon)(bold(u)) = 1/2 (nabla bold(u) + (nabla bold(u))^T)
$
定义空间
$
  bold(Sigma) & := { bold(tau) in bold(H)(div, Omega; SS^d) :
                  integral_Omega tr(bold(tau)) dif x = 0 }, \
      bold(U) & := { bold(u) in (H^1(Omega))^d: bold(u) (bold(x)) = bold(0), bold(x) in partial Omega}.
$

== 最小二乘泛函与变分

定义线弹性问题的最小二乘泛函
$
  cal(J) ((bold(tau), bold(v)); bold(f))
  := norm(bold(cal(A)) : bold(tau) - bold(epsilon)(bold(v)))_(L^2(Omega))^2
  + norm(nabla dot bold(tau) + bold(f))_(L^2(Omega))^2.
$
该泛函的两项分别度量本构关系残差与平衡方程残差。

对应双线性形式记为
$
  a ((bold(sigma), bold(u)), (bold(tau), bold(v)))
  := (bold(cal(A)) : bold(sigma) - bold(epsilon)(bold(u)),
    bold(cal(A)) : bold(tau) - bold(epsilon)(bold(v)))_(L^2(Omega)) + (nabla dot bold(sigma), nabla dot bold(tau))_(L^2(Omega)).
$
右端线性泛函为
$
  ell (bold(tau), bold(v))
  := - (bold(f), nabla dot bold(tau))_(L^2(Omega)).
$

#theorem(title: [线弹性强形式、最小二乘极小化与变分问题的等价性])[
  若线弹性强形式在 $bold(Sigma) times bold(U)$ 中存在解，则下列三个问题彼此等价：

  1. 线弹性强形式；
  2. 最小化 $cal(J)$；
  3. 求 $(bold(sigma), bold(u)) in bold(Sigma) times bold(U)$，使得
    $
      a ((bold(sigma), bold(u)), (bold(tau), bold(v)))
      = ell (bold(tau), bold(v)),
      quad forall (bold(tau), bold(v)) in bold(Sigma) times bold(U).
    $
]<thm:elasticity-equivalence>

证明见附录 @app:elasticity。

== 残差范数等价性

下述引理建立了泛函与 $bold(H)(div, Omega; SS^d) times H^1(Omega)$ 范数平方的双边等价，且常数关于 Lamé 常数 $lambda$ 一致。

#theorem(title: [线弹性问题的强制性])[
  存在常数 $0 < c <= C < oo$，仅依赖于 $Omega$、$mu$ 与 $d$，且可关于 Lamé 常数 $lambda$ 一致选取，使得对任意 $(bold(tau), bold(v)) in bold(Sigma) times bold(U)$ 都有
  $
    c (
      norm(bold(tau))_(bold(H)(div))^2
      + norm(bold(v))_(H^1(Omega))^2
    )
    <= cal(J) ((bold(tau), bold(v)); bold(0))
    <= C (
      norm(bold(tau))_(bold(H)(div))^2
      + norm(bold(v))_(H^1(Omega))^2
    ).
  $
]<theorem:elasticity-coercive>

#proof[
  先证上界。令 $bold(f) = bold(0)$。由柔度算子的显式公式
  $
    bold(cal(A)) : bold(tau)
    = 1/(2 mu) (
      bold(tau) - lambda/(2 mu + d lambda) tr(bold(tau)) bold(I)
    ),
  $
  以及估计 $norm(tr(bold(tau)) bold(I))_(L^2(Omega)) <= d norm(bold(tau))_(L^2(Omega))$，可得
  $
    norm(bold(cal(A)) : bold(tau))_(L^2(Omega)) & <= 1/(2 mu) norm(bold(tau))_(L^2(Omega))
                                                  + lambda/(2 mu (2 mu + d lambda))
                                                  norm(tr(bold(tau)) bold(I))_(L^2(Omega)) \
                                                & <= (
                                                    1/(2 mu) + d lambda/(2 mu (2 mu + d lambda))
                                                  ) norm(bold(tau))_(L^2(Omega)) \
                                                & <= 1/mu norm(bold(tau))_(L^2(Omega)).
  $
  再由 $(a + b)^2 <= 2 a^2 + 2 b^2$ 与 $norm(bold(epsilon)(bold(v)))_(L^2(Omega)) <= norm(nabla bold(v))_(L^2(Omega))$，可得
  $
    cal(J) ((bold(tau), bold(v)); bold(0)) & <= 2 norm(bold(cal(A)) : bold(tau))_(L^2(Omega))^2
                                             + 2 norm(bold(epsilon)(bold(v)))_(L^2(Omega))^2
                                             + norm(nabla dot bold(tau))_(L^2(Omega))^2 \
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
    bold(epsilon)^"D" (bold(v)) := bold(epsilon)(bold(v)) - 1/d (nabla dot bold(v)) bold(I),
  $
  以及各向同性柔度算子的偏差-体积分解
  $
    bold(cal(A)) : bold(tau) = 1/(2 mu) bold(tau)^"D" + 1/(d(2 mu + d lambda)) tr(bold(tau)) bold(I).
  $
  由偏差部分与球张量部分在 $L^2(Omega; SS^d)$ 中的正交性，
  $
    norm(bold(cal(A)) : bold(tau) - bold(epsilon)(bold(v)))_(L^2(Omega))^2
    = norm(1/(2 mu) bold(tau)^"D" - bold(epsilon)^"D" (bold(v)))_(L^2(Omega))^2
    + 1/d norm(nabla dot bold(v) - 1/(2 mu + d lambda) tr(bold(tau)))_(L^2(Omega))^2.
  $
  因此记三个残差
  $
    bold(r)_"dev" := 1/(2 mu) bold(tau)^"D" - bold(epsilon)^"D" (bold(v)),
    quad
    r_"vol" := 1/sqrt(d) (nabla dot bold(v) - 1/(2 mu + d lambda) tr(bold(tau))),
    quad
    bold(r)_"equ" := nabla dot bold(tau).
  $
  便有
  $
    cal(J) ((bold(tau), bold(v)); bold(0))
    = norm(bold(r)_"dev")_(L^2)^2
    + norm(r_"vol")_(L^2)^2
    + norm(bold(r)_"equ")_(L^2)^2.
  $

  首先处理位移部分。由偏差投影的定义，
  $
    bold(epsilon)(bold(v)) = bold(epsilon)^"D" (bold(v)) + 1/d (nabla dot bold(v)) bold(I).
  $
  由于 $bold(epsilon)^"D" (bold(v)) = 1/(2 mu) bold(tau)^"D" - bold(r)_"dev"$，以及 $nabla dot bold(v) = 1/(2 mu + d lambda) tr(bold(tau)) + sqrt(d) r_"vol"$，有
  $
    bold(epsilon)(bold(v)) = 1/(2 mu) bold(tau)^"D" - bold(r)_"dev"
    + 1/d (1/(2 mu + d lambda) tr(bold(tau)) + sqrt(d) r_"vol") bold(I).
  $
  由 Korn 第一不等式（参考 @BrennerScott2008 推论 11.2.25），存在常数 $C_K > 0$，使得
  $
    norm(bold(v))_(H^1(Omega)) & <= C_K norm(bold(epsilon)(bold(v)))_(L^2(Omega)) \
                               & <= C_K (
                                   1/(2 mu) norm(bold(tau)^"D")_(L^2)
                                   + norm(bold(r)_"dev")_(L^2)
                                   + 1/(sqrt(d) (2 mu + d lambda)) norm(tr(bold(tau)))_(L^2)
                                   + norm(r_"vol")_(L^2)
                                 ).
  $
  <eq:korn>

  接下来控制应力。由分解 $bold(tau) = bold(tau)^"D" + 1/d tr(bold(tau)) bold(I)$ 以及 $bold(tau)^"D" : bold(I) = tr(bold(tau)^"D") = 0$，
  $
    integral_Omega (1/(2 mu) bold(tau)^"D") : bold(tau) dif x
    = 1/(2 mu) integral_Omega bold(tau)^"D" : (bold(tau)^"D" + 1/d tr(bold(tau)) bold(I)) dif x
    = 1/(2 mu) norm(bold(tau)^"D")_(L^2(Omega))^2.
  $
  由于 $bold(tau)$ 对称且 $bold(v) in (H_0^1(Omega))^d$，分部积分给出
  $
    integral_Omega bold(epsilon)^"D" (bold(v)) : bold(tau) dif x
    &= integral_Omega bold(epsilon)(bold(v)) : bold(tau) dif x
    - 1/d integral_Omega (nabla dot bold(v)) tr(bold(tau)) dif x
    \
    &= - integral_Omega bold(v) dot (nabla dot bold(tau)) dif x
    - 1/d integral_Omega (nabla dot bold(v)) tr(bold(tau)) dif x.
  $
  于是
  $
    integral_Omega (1/(2 mu) bold(tau)^"D") : bold(tau) dif x
    &= integral_Omega (bold(r)_"dev" + bold(epsilon)^"D" (bold(v))) : bold(tau) dif x
    \
    &= integral_Omega bold(r)_"dev" : bold(tau) dif x
    - integral_Omega bold(v) dot (nabla dot bold(tau)) dif x
    - 1/d integral_Omega (nabla dot bold(v)) tr(bold(tau)) dif x.
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
    <= C_1 (
      norm(bold(r)_"dev")_(L^2(Omega))^2
      + norm(r_"vol")_(L^2(Omega))^2
      + norm(bold(r)_"equ")_(L^2(Omega))^2
    ).
  $
  <eq:tau-dev-bound>

  对于应力的迹部分，由估计（参见 @CarstensenDolzmann1998，引理 4.1），存在只依赖于 $Omega$ 的常数 $C_"tr" > 0$，使得
  $
    norm(tr(bold(tau)))_(L^2(Omega))
    <= C_"tr" (norm(bold(tau)^"D")_(L^2(Omega)) + norm(nabla dot bold(tau))_(L^2(Omega))),
    quad forall bold(tau) in bold(Sigma).
  $
  这里零平均迹条件是必要的；否则常数球应力 $bold(tau) = c bold(I)$ 会使右端为零而左端非零。
  结合 @eq:tau-dev-bound 即得
  $
    norm(tr(bold(tau)))_(L^2(Omega))^2
    <= C_2 (
      norm(bold(r)_"dev")_(L^2(Omega))^2
      + norm(bold(r)_"equ")_(L^2(Omega))^2
      + norm(r_"vol")_(L^2(Omega))^2
    ).
  $
  因此 $norm(bold(tau))_(L^2)^2$ 被残差平方和控制。将应力范数估计代回 @eq:korn，得到 $norm(bold(v))_(H^1)^2$ 的类似估计。最后结合
  $
    norm(bold(tau))_(bold(H)(div))^2
    = norm(bold(tau))_(L^2(Omega))^2
    + norm(bold(r)_"equ")_(L^2(Omega))^2
  $
  即得下界，且下界常数 $c$ 也可不依赖于 $lambda$。证毕。
]

由 @theorem:elasticity-coercive，双线性形式 $a$ 在 $bold(Sigma) times bold(U)$ 上连续且强制，其中连续性与强制性常数均可关于 Lamé 常数 $lambda$ 一致选取。特别地，近不可压缩极限 $lambda -> oo$ 下稳定常数不退化；只要保形离散空间的逼近误差本身保持稳定，离散解不会因该极限而产生锁死。

= Kirchhoff-Love 板弯曲问题

== 弯矩-挠度混合形式

这一情形以弯矩与挠度作为混合变量。与线弹性不同，平衡算子为 $nabla dot (nabla dot dot.c)$，因此弯矩变量需要置于 $bold(H)(div div, Omega; SS^2)$ 中。设 $Omega subset RR^2$ 为板中面区域，未知量为弯矩张量 $bold(cal(M)): Omega -> SS^2$ 与标量挠度 $u: Omega -> RR$。运动学算子取曲率张量
$
  bold(cal(K))(u) := - nabla^2 u.
$
于是 Kirchhoff-Love 板弯曲的混合系统写为
$
  cases(
    bold(cal(A)) : bold(cal(M)) - bold(cal(K))(u) & = 0 & quad "in" Omega,
    nabla dot (nabla dot bold(cal(M))) + f & = 0 & quad "in" Omega,
    u & = 0 & quad "on" partial Omega,
    partial_n u & = 0 & quad "on" partial Omega.
  )
$
这里 $bold(cal(A))$ 为弯矩柔度张量；对各向同性薄板，其作用可写为
$
  bold(cal(A)) : bold(tau)
  = 1/(D(1-nu)) bold(tau)
  - nu/(D(1-nu)(1+nu)) tr(bold(tau)) bold(I),
$
其中
$
  D = (E h^3)/(12(1 - nu^2)),
  quad
  lambda = (E nu)/((1 + nu) (1 - 2 nu)),
  quad
  mu = E/(2(1 + nu))
$
为弯曲刚度与对应三维各向同性材料的 Lamé 参数。下文固定 $mu > 0$ 与 $h > 0$，并只讨论物理参数区间 $lambda >= 0$；等价地，$0 <= nu < 1/2$。
定义函数空间
$
  bold(Sigma) := bold(H)(div div, Omega; SS^2),
  quad
  U := H_0^2(Omega),
$
并记
$
  norm(bold(tau))_(bold(H)(div div))^2
  := norm(bold(tau))_(L^2(Omega))^2
  + norm(nabla dot (nabla dot bold(tau)))_(L^2(Omega))^2.
$

== 最小二乘泛函与变分

定义板弯曲问题的最小二乘泛函：
$
  cal(J) ((bold(tau), v); f)
  := norm(bold(cal(A)) : bold(tau) - bold(cal(K))(v))_(L^2(Omega))^2 + norm(nabla dot (nabla dot bold(tau)) + f)_(L^2(Omega))^2,
$
双线性形式：
$
  a ((bold(cal(M)), u), (bold(tau), v))
  := (bold(cal(A)) : bold(cal(M)) - bold(cal(K))(u),
    bold(cal(A)) : bold(tau) - bold(cal(K))(v))_(L^2(Omega)) + (nabla dot (nabla dot bold(cal(M))), nabla dot (nabla dot bold(tau)))_(L^2(Omega)),
$
与右端泛函：
$
  ell (bold(tau), v)
  := - (f, nabla dot (nabla dot bold(tau)))_(L^2(Omega)).
$

#theorem(title: [板弯曲强形式、最小二乘极小化与变分问题的等价性])[
  若 Kirchhoff-Love 板弯曲强形式在 $bold(Sigma) times U$ 中存在解，则下列三个问题彼此等价：

  1. Kirchhoff-Love 板弯曲强形式；
  2. 最小化 $cal(J)$；
  3. 求 $(bold(cal(M)), u) in bold(Sigma) times U$，使得
    $
      a ((bold(cal(M)), u), (bold(tau), v))
      = ell (bold(tau), v),
      quad forall (bold(tau), v) in bold(Sigma) times U.
    $
]<thm:plate-equivalence>

证明见附录 @app:plate。

== 残差范数等价性

与线弹性问题的机制不同：此处柔度算子 $bold(cal(A))$ 本身即关于 $lambda$ 一致，故稳定性主要借助 Poincaré 不等式从曲率控制挠度，而不依赖迹估计。

#theorem(title: [板弯曲问题的强制性])[
  在上述各向同性薄板参数假设下，存在常数 $0 < c <= C < oo$，仅依赖于 $Omega$、$mu$ 与 $h$，且可关于 Lamé 常数 $lambda$ 一致选取，使得对任意 $(bold(tau), v) in bold(Sigma) times U$ 都有
  $
    c (
      norm(bold(tau))_(bold(H)(div div))^2
      + norm(v)_(H^2(Omega))^2
    )
    <= cal(J) ((bold(tau), v); 0)
    <= C (
      norm(bold(tau))_(bold(H)(div div))^2
      + norm(v)_(H^2(Omega))^2
    ).
  $
]<lem:4th-coercive>

#proof[
  先将柔度算子写成偏差-迹分解。对任意 $bold(tau) in SS^2$，记
  $
    bold(tau)^"D" := bold(tau) - 1/2 tr(bold(tau)) bold(I),
    quad
    bold(tau) = bold(tau)^"D" + 1/2 tr(bold(tau)) bold(I).
  $
  将该分解代入各向同性薄板柔度公式，得到
  $
    bold(cal(A)) : bold(tau)
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
  再利用偏差部分与球张量部分在 $L^2(Omega; SS^2)$ 中的正交性以及
  $
    norm(bold(tau))_(L^2(Omega))^2
    = norm(bold(tau)^"D")_(L^2(Omega))^2
    + 1/2 norm(tr(bold(tau)))_(L^2(Omega))^2,
  $
  得到
  $
    norm(bold(cal(A)) : bold(tau))_(L^2(Omega))^2
    = (1/(D(1-nu)))^2 norm(bold(tau)^"D")_(L^2(Omega))^2
    + 1/(2 D^2 (1+nu)^2) norm(tr(bold(tau)))_(L^2(Omega))^2
    <= 36/(mu^2 h^6) norm(bold(tau))_(L^2(Omega))^2,
  $
  以及
  $
    2/(mu h^3) norm(bold(tau))_(L^2(Omega))^2
    <= integral_Omega (bold(cal(A)) : bold(tau)) : bold(tau) dif x
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
    norm(bold(cal(A)) : bold(tau) - bold(cal(K))(v))_(L^2(Omega))^2
    <= 2 C_A^2 norm(bold(tau))_(L^2(Omega))^2
    + 2 norm(bold(cal(K))(v))_(L^2(Omega))^2.
  $
  又因为 $bold(cal(K))(v) = - nabla^2 v$，故
  $
    norm(bold(cal(K))(v))_(L^2(Omega))
    = norm(nabla^2 v)_(L^2(Omega))
    <= norm(v)_(H^2(Omega)).
  $
  再加上
  $
    norm(nabla dot (nabla dot bold(tau)))_(L^2(Omega))^2
    <= norm(bold(tau))_(bold(H)(div div))^2
  $
  即得上界。

  再证下界。记
  $
    bold(r)_"c" := bold(cal(A)) : bold(tau) - bold(cal(K))(v),
    quad
    r_"e" := nabla dot (nabla dot bold(tau)).
  $
  则
  $
    bold(cal(K))(v) = bold(cal(A)) : bold(tau) - bold(r)_"c".
  $
  由于 $v in H_0^2(Omega)$，有 $v in H_0^1(Omega)$ 且 $partial_i v in H_0^1(Omega)$，$i = 1, 2$。由 Poincaré 不等式（参考 @BrennerScott2008 命题 5.3.5），存在常数 $C_P > 0$，使得
  $
    norm(w)_(H^1(Omega))
    <= C_P abs(w)_(H^1(Omega)),
    quad forall w in H_0^1(Omega).
  $
  分别取 $w = v$ 与 $w = partial_i v$，得到
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
    = tilde(C)_P norm(bold(cal(K))(v))_(L^2(Omega))
    <= tilde(C)_P (
      norm(bold(r)_"c")_(L^2(Omega))
      + C_A norm(bold(tau))_(L^2(Omega))
    ).
  $
  <eq:plate-h2>

  另一方面，由椭圆性，
  $
    c_A norm(bold(tau))_(L^2(Omega))^2
    <= integral_Omega (bold(cal(A)) : bold(tau)) : bold(tau) dif x.
  $
  又由于 $v in H_0^2(Omega)$，对 $bold(cal(K))(v) = - nabla^2 v$ 作两次分部积分可得
  $
    integral_Omega bold(cal(K))(v) : bold(tau) dif x
    = - integral_Omega v nabla dot (nabla dot bold(tau)) dif x
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
    <= C_1 (
      norm(bold(r)_"c")_(L^2(Omega))^2
      + norm(r_"e")_(L^2(Omega))^2
    ).
  $
  再代回 @eq:plate-h2，得到
  $
    norm(v)_(H^2(Omega))^2
    <= C_2 (
      norm(bold(r)_"c")_(L^2(Omega))^2
      + norm(r_"e")_(L^2(Omega))^2
    ).
  $
  最后结合
  $
    norm(bold(tau))_(bold(H)(div div))^2
    = norm(bold(tau))_(L^2(Omega))^2
    + norm(r_"e")_(L^2(Omega))^2
  $
  即得下界。由于 $c_A$ 与 $C_A$ 仅依赖于 $mu$ 与 $h$，故上、下界常数 $c$ 与 $C$ 也仅依赖于 $Omega$、$mu$ 与 $h$，并可关于 $lambda$ 一致选取。证毕。
]

由 @lem:4th-coercive，板弯曲双线性形式在 $bold(Sigma) times U$ 上连续且强制，其中常数可关于 Lamé 常数 $lambda$ 一致选取。值得对比的是：线弹性中柔度算子的体积部分在 $lambda -> oo$ 时退化，一致稳定性依赖偏差-体积分解与迹估计；板弯曲中柔度算子本身即一致椭圆，一致稳定性转而依赖 Poincaré 不等式从曲率恢复挠度范数。

= 离散空间逼近

== 随机特征空间

本文对每个未知量分别生成一组相互独立的原始随机基。记
$
  xi_n^q (x) = rho(gamma ((bold(a)_n^q)^T x + r_n^q)),
  quad n = 1, dots, N,
$
并记常数基函数 $xi_0^q = 1$。这里 $q$ 表示相应未知量，例如线弹性中的 $bold(sigma)$、$bold(u)$，以及板弯曲中的 $bold(cal(M))$、$u$。参数 $gamma > 0$ 为全局形状参数，随机方向与随机截距按 @zhang2024transnet 的重参数化方式生成：
$
  bold(w)_n^q = gamma bold(a)_n^q, quad
  b_n^q = gamma r_n^q,
$
其中 $bold(a)_n^q$ 表示超平面法向量，$r_n^q$ 表示截距。采样方式如下：
$
  bold(a)_n^q = bold(X)_n^q / norm(bold(X)_n^q)_2,
  quad
  r_n^q = U_n^q.
$
这里 $bold(X)_n^q$ 服从标准正态分布，$U_n^q$ 服从有界区间上的均匀分布。由于激活函数 $rho$ 光滑（实验取 $rho = tanh$），每个原始基函数均属于 $C^oo (overline(Omega))$。因此由这些基函数与有限维分量基逐分量张成的向量场或张量场也逐分量光滑，满足 $bold(H)(div, Omega; SS^d)$ 与 $bold(H)(div div, Omega; SS^2)$ 所需的正则性。

为从原始随机基得到满足本质边界条件的基函数，引入迹消去算子
$
  cal(P)^(0): C^oo (overline(Omega)) -> H_0^1(Omega),
  quad
  gamma_0 cal(P)^(0) g = 0,
$
以及
$
  cal(P)^(1): C^oo (overline(Omega)) -> H_0^2(Omega),
  quad
  gamma_0 cal(P)^(1) g = 0,
  quad
  gamma_1 cal(P)^(1) g = 0.
$
这里 $gamma_0$ 表示零阶迹，$gamma_1$ 表示一阶法向迹。相应保形基定义为
$
  zeta_n^(0, q) := cal(P)^(0) xi_n^q,
  quad
  zeta_n^(1, q) := cal(P)^(1) xi_n^q.
$
因此 ${zeta_n^(0, bold(u))}$ 用于构造线弹性的 $H_0^1$-保形位移基，${zeta_n^(1, u)}$ 用于构造板弯曲的 $H_0^2$-保形挠度基。

又以零平均迹泛函
$
  ell_"tr" (bold(tau)) := integral_Omega tr(bold(tau)) dif x
$
记之。线弹性应力变量的平均迹约束直接作为有限维系数的线性约束施加。也就是说，先写出由原始基与对称分量基组成的应力展开式，再要求 $ell_"tr" (bold(tau)_N) = 0$；满足该约束的展开式张成最终应力随机特征空间。

== 线弹性离散空间

设 $d in {2, 3}$，区域取盒型区域 $Omega = product_(i=1)^d [0, 1]$。本文采用的零阶迹消去算子可具体写为
$
  cal(P)^(0) g (x) := product_(i=1)^d x_i (1 - x_i) g(x),
  quad
  zeta_n^(0, bold(u)) := cal(P)^(0) xi_n^(bold(u)).
$
以及 $SS^d$ 的对称基 ${bold(E)_alpha}_(alpha = 1)^(d(d+1)/2)$ 与 $RR^d$ 的标准基 ${bold(e)_i}_(i = 1)^d$。线弹性随机特征空间取为
$
  bold(Sigma)_N & := {
                    sum_(n=0)^N sum_(alpha=1)^(d(d+1)/2)
                    phi^(bold(sigma))_(n, alpha) xi_n^(bold(sigma)) bold(E)_alpha :
                    integral_Omega tr(
                      sum_(n=0)^N sum_(alpha=1)^(d(d+1)/2)
                      phi^(bold(sigma))_(n, alpha) xi_n^(bold(sigma)) bold(E)_alpha
                    ) dif x = 0
                  }, \
      bold(U)_N & := span { zeta_n^(0, bold(u)) bold(e)_i : 0 <= n <= N, 1 <= i <= d }.
$
由于应力基函数逐分量光滑且满足零平均迹约束，同时 $zeta_n^(0, bold(u)) in H_0^1(Omega)$，故 $bold(Sigma)_N subset bold(Sigma)$、$bold(U)_N subset bold(U)$。平面应力只改变 $d = 2$ 时的本构参数与应力分量数，仍属于同一线弹性离散框架。

离散未知量写为
$
  bold(Phi)^(bold(sigma))
  = sum_(n=0)^N sum_(alpha=1)^(d(d+1)/2)
  phi^(bold(sigma))_(n, alpha) xi_n^(bold(sigma)) bold(E)_alpha,
$
以及
$
  bold(Phi)^(bold(u))
  = sum_(n=0)^N sum_(i=1)^d
  phi^(bold(u))_(n, i) zeta_n^(0, bold(u)) bold(e)_i.
$
其中应力系数满足零平均迹约束
$
  sum_(n=0)^N (integral_Omega xi_n^(bold(sigma)) dif x)
  sum_(i=1)^d phi^(bold(sigma))_(n, i) = 0,
$
这里前 $d$ 个对称基分量对应对角应力分量。

离散问题为：求 $(bold(Phi)^(bold(sigma)), bold(Phi)^(bold(u))) in bold(Sigma)_N times bold(U)_N$，使得
$
  a ((bold(Phi)^(bold(sigma)), bold(Phi)^(bold(u))),
    (bold(Phi)^(bold(tau)), bold(Phi)^(bold(v))))
  = ell (bold(Phi)^(bold(tau)), bold(Phi)^(bold(v))),
  quad forall (bold(Phi)^(bold(tau)), bold(Phi)^(bold(v))) in bold(Sigma)_N times bold(U)_N.
$
具体系数展开与约束后的加权残差最小二乘问题见附录 @app:elasticity。

== 线弹性准最优估计

#theorem(title: [线弹性离散问题的存在唯一性与稳定性])[
  对任意保形有限维子空间 $bold(Sigma)_N subset bold(Sigma)$ 与 $bold(U)_N subset bold(U)$，线弹性离散问题存在唯一解。记离散解为 $(bold(Phi)^(bold(sigma)), bold(Phi)^(bold(u)))$，则有
  $
    norm(bold(Phi)^(bold(sigma)))_(bold(H)(div))
    + norm(bold(Phi)^(bold(u)))_(H^1(Omega))
    <= sqrt(2)/c norm(bold(f))_(L^2(Omega)),
  $
  其中 $c$ 来源于 @theorem:elasticity-coercive 的强制性常数。
]

#proof[
  由于 $bold(Sigma)_N subset bold(Sigma)$ 且 $bold(U)_N subset bold(U)$，@theorem:elasticity-coercive 的强制性可直接限制到离散子空间上，且常数不随 $N$ 改变。右端泛函满足
  $
    abs(ell (bold(Phi)^(bold(tau)), bold(Phi)^(bold(v))))
    = abs((bold(f), nabla dot bold(Phi)^(bold(tau)))_(L^2(Omega)))
    <= norm(bold(f))_(L^2(Omega)) norm(bold(Phi)^(bold(tau)))_(bold(H)(div)).
  $
  因此由 Lax-Milgram 定理，离散解存在唯一。取测试函数为离散解本身，并记
  $
    C_N := (
      norm(bold(Phi)^(bold(sigma)))_(bold(H)(div))^2
      + norm(bold(Phi)^(bold(u)))_(H^1(Omega))^2
    )^(1/2),
  $
  则
  $
    c C_N^2
    <= a ((bold(Phi)^(bold(sigma)), bold(Phi)^(bold(u))),
      (bold(Phi)^(bold(sigma)), bold(Phi)^(bold(u))))
    = ell (bold(Phi)^(bold(sigma)), bold(Phi)^(bold(u)))
    <= norm(bold(f))_(L^2(Omega)) C_N.
  $
  若 $C_N = 0$，结论显然；否则两端同除以 $C_N$，再用
  $
    norm(bold(Phi)^(bold(sigma)))_(bold(H)(div))
    + norm(bold(Phi)^(bold(u)))_(H^1(Omega))
    <= sqrt(2) C_N
  $
  即得稳定性估计。证毕。
]

#theorem(title: [线弹性的准最优误差估计])[
  设连续解 $(bold(sigma), bold(u)) in bold(Sigma) times bold(U)$ 存在，离散解 $(bold(Phi)^(bold(sigma)), bold(Phi)^(bold(u))) in bold(Sigma)_N times bold(U)_N$ 由上式给出。则有
  $
    norm(bold(sigma) - bold(Phi)^(bold(sigma)))_(bold(H)(div))
    + norm(bold(u) - bold(Phi)^(bold(u)))_(H^1(Omega))
    <= C_"le"
    inf_((bold(Phi)^(bold(tau)), bold(Phi)^(bold(v))) in bold(Sigma)_N times bold(U)_N)
    (
      norm(bold(sigma) - bold(Phi)^(bold(tau)))_(bold(H)(div))
      + norm(bold(u) - bold(Phi)^(bold(v)))_(H^1(Omega))
    ),
  $
  其中 $C_"le"$ 可关于 Lamé 常数 $lambda$ 一致选取。
]

#proof[
  连续解满足连续变分问题，离散解满足离散变分问题。由于离散空间保形，二者相减可得 Galerkin 正交性：
  $
    a (
      (bold(sigma) - bold(Phi)^(bold(sigma)), bold(u) - bold(Phi)^(bold(u))),
      (bold(Phi)^(bold(tau)), bold(Phi)^(bold(v)))
    )
    = 0,
    quad forall (bold(Phi)^(bold(tau)), bold(Phi)^(bold(v))) in bold(Sigma)_N times bold(U)_N.
  $
  记误差
  $
    (bold(e)_sigma, bold(e)_u)
    := (bold(sigma) - bold(Phi)^(bold(sigma)), bold(u) - bold(Phi)^(bold(u))).
  $
  对任意 $(bold(Phi)^(bold(tau)), bold(Phi)^(bold(v))) in bold(Sigma)_N times bold(U)_N$，有
  $
    (bold(e)_sigma, bold(e)_u)
    = (bold(sigma) - bold(Phi)^(bold(tau)), bold(u) - bold(Phi)^(bold(v)))
    + (bold(Phi)^(bold(tau)) - bold(Phi)^(bold(sigma)), bold(Phi)^(bold(v)) - bold(Phi)^(bold(u))),
  $
  其中第二项属于离散空间。由强制性、Galerkin 正交性与连续性，
  $
       & c (
           norm(bold(e)_sigma)_(bold(H)(div))^2
           + norm(bold(e)_u)_(H^1(Omega))^2
         ) \
    <= & a ((bold(e)_sigma, bold(e)_u), (bold(e)_sigma, bold(e)_u)) \
     = & a ((bold(e)_sigma, bold(e)_u),
           (bold(sigma) - bold(Phi)^(bold(tau)), bold(u) - bold(Phi)^(bold(v)))) \
    <= & C (
           norm(bold(e)_sigma)_(bold(H)(div))^2
           + norm(bold(e)_u)_(H^1(Omega))^2
         )^(1/2) \
       & quad times (
           norm(bold(sigma) - bold(Phi)^(bold(tau)))_(bold(H)(div))^2
           + norm(bold(u) - bold(Phi)^(bold(v)))_(H^1(Omega))^2
         )^(1/2).
  $
  若误差为零，结论显然；否则两端约去误差范数，并对所有离散函数取下确界，即得所述估计。证毕。
]

== 板弯曲离散空间

板弯曲情形取 $Omega = [0, 1]^2$。本文采用的消去至一阶迹的算子可具体写为
$
  cal(P)^(1) g (x, y) := [x (1 - x) y (1 - y)]^2 g(x, y),
  quad
  zeta_n^(1, u) := cal(P)^(1) xi_n^(u).
$
于是 $gamma_0 zeta_n^(1, u) = gamma_1 zeta_n^(1, u) = 0$。取 ${bold(E)_alpha}_(alpha = 1)^3 subset SS^2$ 按 Voigt 顺序 $(11, 22, 12)$ 排列。弯矩空间无迹约束，故
$
  bold(Sigma)_N & := span { xi_n^(bold(cal(M))) bold(E)_alpha : 0 <= n <= N, 1 <= alpha <= 3 }, \
            U_N & := span { zeta_n^(1, u) : 0 <= n <= N }.
$
由于弯矩基函数逐分量光滑，且 $zeta_n^(1, u) in H_0^2(Omega)$，故 $bold(Sigma)_N subset bold(H)(div div, Omega; SS^2)$、$U_N subset H_0^2(Omega)$。

离散未知量写为
$
  bold(Phi)^(bold(cal(M)))
  = sum_(n=0)^N sum_(alpha=1)^3
  phi^(bold(cal(M)))_(n, alpha) xi_n^(bold(cal(M))) bold(E)_alpha,
$
以及
$
  Phi^(u)
  = sum_(n=0)^N phi_n^(u) zeta_n^(1, u).
$
离散问题为：求 $(bold(Phi)^(bold(cal(M))), Phi^(u)) in bold(Sigma)_N times U_N$，使得
$
  a ((bold(Phi)^(bold(cal(M))), Phi^(u)),
    (bold(Phi)^(bold(tau)), Phi^(v)))
  = ell (bold(Phi)^(bold(tau)), Phi^(v)),
  quad forall (bold(Phi)^(bold(tau)), Phi^(v)) in bold(Sigma)_N times U_N.
$
具体系数展开与加权残差最小二乘问题见附录 @app:plate。

== 板弯曲准最优估计

#theorem(title: [板弯曲离散问题的存在唯一性与稳定性])[
  对任意保形有限维子空间 $bold(Sigma)_N subset bold(Sigma)$ 与 $U_N subset U$，板弯曲离散问题存在唯一解。记离散解为 $(bold(Phi)^(bold(cal(M))), Phi^(u))$，则有
  $
    norm(bold(Phi)^(bold(cal(M))))_(bold(H)(div div))
    + norm(Phi^(u))_(H^2(Omega))
    <= sqrt(2)/c norm(f)_(L^2(Omega)),
  $
  其中 $c$ 来源于 @lem:4th-coercive 的强制性常数。
]

#proof[
  由于 $bold(Sigma)_N subset bold(Sigma)$ 且 $U_N subset U$，@lem:4th-coercive 的强制性可直接限制到离散子空间上，且常数不随 $N$ 改变。右端泛函满足
  $
    abs(ell (bold(Phi)^(bold(tau)), Phi^(v)))
    = abs((f, nabla dot (nabla dot bold(Phi)^(bold(tau))))_(L^2(Omega)))
    <= norm(f)_(L^2(Omega)) norm(bold(Phi)^(bold(tau)))_(bold(H)(div div)).
  $
  因此由 Lax-Milgram 定理，离散解存在唯一。取测试函数为离散解本身，并记
  $
    C_N := (
      norm(bold(Phi)^(bold(cal(M))))_(bold(H)(div div))^2
      + norm(Phi^(u))_(H^2(Omega))^2
    )^(1/2),
  $
  则
  $
    c C_N^2
    <= a ((bold(Phi)^(bold(cal(M))), Phi^(u)),
      (bold(Phi)^(bold(cal(M))), Phi^(u)))
    = ell (bold(Phi)^(bold(cal(M))), Phi^(u))
    <= norm(f)_(L^2(Omega)) C_N.
  $
  若 $C_N = 0$，结论显然；否则两端同除以 $C_N$，再用两项范数和与平方和范数之间的关系即可得到稳定性估计。证毕。
]

#theorem(title: [板弯曲的准最优误差估计])[
  设连续解 $(bold(cal(M)), u) in bold(Sigma) times U$ 存在，离散解 $(bold(Phi)^(bold(cal(M))), Phi^(u)) in bold(Sigma)_N times U_N$ 由上式给出。则有
  $
       & norm(bold(cal(M)) - bold(Phi)^(bold(cal(M))))_(bold(H)(div div))
         + norm(u - Phi^(u))_(H^2(Omega)) \
    <= & C_"pl"
         inf_((bold(Phi)^(bold(tau)), Phi^(v)) in bold(Sigma)_N times U_N)
         (
           norm(bold(cal(M)) - bold(Phi)^(bold(tau)))_(bold(H)(div div))
           + norm(u - Phi^(v))_(H^2(Omega))
         ),
  $
  其中 $C_"pl"$ 可关于 Lamé 常数 $lambda$ 一致选取。
]

#proof[
  连续解满足连续变分问题，离散解满足离散变分问题。由于离散空间保形，二者相减可得 Galerkin 正交性：
  $
    a (
      (bold(cal(M)) - bold(Phi)^(bold(cal(M))), u - Phi^(u)),
      (bold(Phi)^(bold(tau)), Phi^(v))
    )
    = 0,
    quad forall (bold(Phi)^(bold(tau)), Phi^(v)) in bold(Sigma)_N times U_N.
  $
  记误差
  $
    (bold(e)_M, e_u)
    := (bold(cal(M)) - bold(Phi)^(bold(cal(M))), u - Phi^(u)).
  $
  对任意 $(bold(Phi)^(bold(tau)), Phi^(v)) in bold(Sigma)_N times U_N$，有
  $
    (bold(e)_M, e_u)
    = (bold(cal(M)) - bold(Phi)^(bold(tau)), u - Phi^(v))
    + (bold(Phi)^(bold(tau)) - bold(Phi)^(bold(cal(M))), Phi^(v) - Phi^(u)),
  $
  其中第二项属于离散空间。由强制性、Galerkin 正交性与连续性，
  $
       & c (
           norm(bold(e)_M)_(bold(H)(div div))^2
           + norm(e_u)_(H^2(Omega))^2
         ) \
    <= & a ((bold(e)_M, e_u), (bold(e)_M, e_u)) \
     = & a ((bold(e)_M, e_u),
           (bold(cal(M)) - bold(Phi)^(bold(tau)), u - Phi^(v))) \
    <= & C (
           norm(bold(e)_M)_(bold(H)(div div))^2
           + norm(e_u)_(H^2(Omega))^2
         )^(1/2) \
       & quad times (
           norm(bold(cal(M)) - bold(Phi)^(bold(tau)))_(bold(H)(div div))^2
           + norm(u - Phi^(v))_(H^2(Omega))^2
         )^(1/2).
  $
  若误差为零，结论显然；否则两端约去误差范数，并对所有离散函数取下确界，即得所述估计。证毕。
]

= 数值实验

== 实验设定

实验算例均采用制造解基准。除非另有说明，所有实验共同采用：

#figure(
  three-line-table(
    columns: 2,
    align: (right, left),
  )[
    | 参数 | 设置 |
    |---|---|
    | 数据类型 | `torch.float64` |
    | 激活函数 | $tanh$ |
    | 求积方式 | Gauss--Legendre 张量积求积 |
    | 特征种子 | 应力/弯矩特征为 $42$，位移/挠度特征为 $1042$ |
  ],
  caption: [数值算例的公共实验设置],
)

本文所有数值结果均记为 LS。对训练求积点上的本构残差与平衡残差分别乘以求积权重的平方根，并按行堆叠为残差矩阵 $bold(R)_N$ 与右端 $bold(b)_N$，随后求解
$
  bold(phi)_N
  := arg min_(bold(psi) in RR^(K_N))
  norm(bold(R)_N bold(psi) - bold(b)_N)_2.
$
求解前对 $bold(R)_N$ 的各列作二范数缩放，再使用 LAPACK GELSD，截断参数取 $10^(-14)$；三维大规模系统先用流式 TSQR 压缩增广残差矩阵，再求解保持同一极小点的约化问题。具体残差块与求解过程见附录 @app:elasticity、@app:plate 和 @app:solver。以下记测试求积点与权重为 ${(bold(x)_p, omega_p^"test")}_(p=1)^(Q_"test")$。

== 三维线弹性

该算例用于检验本文方法在三维线弹性、三维对称应力空间中的表现。实验参数设置如下：

#figure(
  three-line-table(
    columns: 2,
    align: (right, left),
  )[
    | 项目         | 具体说明 |
    |-------------|---------|
    | 算例名称     | 三维线弹性 |
    | 杨氏模量     | $E = 4/3$ |
    | 泊松比       | $nu = 1/3$ |
    | 计算区域     | $[0, 1]^3$ |
    | 形状参数     | $gamma = 2.0$ |
    | 超平面法向量  | $bold(X) tilde.op cal(N)(0, bold(I)_3)$ |
    | 截距        | $U_n tilde.op cal(U)(0, 1)$ |
    | 训练采样点   | $Q_"train" = (2^4)^3 = 4096$ |
    | 测试采样点   | $Q_"test" = (2^4)^3 = 4096$ |
  ],
  caption: [三维线弹性实验设置],
)

参考 @hu2015tetrahedral 中精确解，设
$
  bold(u)_"ex" (x)
  =
  mat(
    2^4 x_1 (1 - x_1) x_2 (1 - x_2) x_3 (1 - x_3);
    2^5 x_1 (1 - x_1) x_2 (1 - x_2) x_3 (1 - x_3);
    2^6 x_1 (1 - x_1) x_2 (1 - x_2) x_3 (1 - x_3)
  ).
$
这里取 $E = 4/3$、$nu = 1/3$，则 Lamé 参数
$
  mu = E / (2 (1 + nu)) = 1/2,
  quad
  lambda = (E nu) / ((1 + nu)(1 - 2 nu)) = 1,
$
与文献中的三维线弹性系数一致。由于公共因子 $x_1 (1 - x_1) x_2 (1 - x_2) x_3 (1 - x_3)$ 在边界上为零，故该制造解满足 $bold(u)_"ex" = 0$ 于 $partial Omega$。精确应力 $bold(sigma)_"ex"$ 由各向同性本构关系计算，体力由
$
  bold(f)_"ex" = - nabla dot bold(sigma)(bold(u)_"ex")
$
通过自动微分生成。

对三维线弹性，位移误差定义为
$
  norm(bold(Phi)^bold(u) - bold(u)_"ex")_0
  := sqrt(sum_(p=1)^(Q_"test") omega_p^"test" sum_(i = 1)^3 [(bold(Phi)^bold(u) (bold(x)_p))_i - (bold(u)_"ex" (bold(x)_p))_i]^2),
$
而在 Voigt 顺序 $(11, 22, 33, 12, 23, 13)$ 下，对应力采用权重 $bold(w)^"V" = (1, 1, 1, 2, 2, 2)^T$，并定义
$
  norm(bold(Phi)^bold(sigma) - bold(sigma)_"ex")_0
  := sqrt(sum_(p=1)^(Q_"test") omega_p^"test" sum_(alpha=1)^6 w^"V"_alpha [(bold(Phi)^(bold(sigma)) (bold(x)_p))_alpha - (bold(sigma)_"ex" (bold(x)_p))_alpha]^2).
$

=== 实验结果

不同特征数量下的数值结果见 @tb:3d-ablation-n。表中 DOF 表示约束后线性系统的未知系数个数；对三维线弹性，
$
  "DOF" = (6(N + 1) - 1) + 3(N + 1) = 9 N + 8.
$

#figure(
  three-line-table(
    columns: 6,
    align: (left, center, center, center, center, center),
  )[
    | 方法 | $N$ | DOF | $norm(bold(Phi)^bold(u) - bold(u)_"ex")_0$ | $norm(bold(Phi)^bold(sigma) - bold(sigma)_"ex")_0$ | Time(s) |
    |:-----------|:----:|:----:|:--------:|:--------:|:-------:|
    | LS | 200  | 1808 | 4.05e-04 | 2.27e-02 |  2.87   |
    | LS | 400  | 3608 | 2.16e-05 | 1.47e-03 |  13.41  |
    | LS | 600  | 5408 | 3.93e-06 | 3.13e-04 |  39.55  |
    | LS | 800  | 7208 | 7.22e-07 | 6.41e-05 |  89.87  |
    | LS | 1000 | 9008 | 2.30e-07 | 2.31e-05 | 166.60  |
  ],
  caption: [三维线弹性在不同特征数量下的数值结果],
)<tb:3d-ablation-n>

#figure(
  image("/public/images/least-squares/linear-elasticity-3d/ablation/N/ablation-N.png"),
  caption: [三维线弹性在不同特征数量下的数值结果],
)

由 @tb:3d-ablation-n 可见，随着 $N$ 从 $200$ 增至 $1000$，位移误差由 $4.05 times 10^(-4)$ 降至 $2.30 times 10^(-7)$，应力误差由 $2.27 times 10^(-2)$ 降至 $2.31 times 10^(-5)$。两项误差均随特征数量稳定下降，说明加权残差 LS 在该三维随机空间上保持了清晰的收敛趋势；相应代价是残差矩阵规模增大后求解时间明显上升。

为给出同一制造解下的有限元参照，@hu2015tetrahedral 中 $P_4$ Hu-Zhang 元的结果列于 @tb:3d-hu-zhang。这里 DOF 按文中初始网格及均匀二分细化后的 Kuhn 六四面体剖分推算，统计为 $dim bold(Sigma)_h + dim V_h$。

#figure(
  three-line-table(
    columns: 5,
    align: (center, center, center, center, center),
  )[
    | 方法 | level | DOF | $norm(bold(u) - bold(u)_h)_0$ | $norm(bold(sigma) - bold(sigma)_h)_0$ |
    |:-----------:|:-----:|:----:|:--------:|:--------:|
    | Hu-Zhang ($P_4$) | 1 | 1215  | 6.13e-02 | 1.99e-01 |
    | Hu-Zhang ($P_4$) | 2 | 8472  | 7.15e-03 | 8.05e-03 |
    | Hu-Zhang ($P_4$) | 3 | 63666 | 4.91e-04 | 2.91e-04 |
  ],
  caption: [@hu2015tetrahedral 中三维 Hu-Zhang $P_4$ 元的数值结果],
)<tb:3d-hu-zhang>

== 二维线弹性

该算例用于检验本文方法在二维线弹性、二维对称应力空间中的表现，并与 @hu2014triangle 中的二维线弹性结果作对比。实验参数设置如下：

#figure(
  three-line-table(
    columns: 2,
    align: (right, left),
  )[
    | 项目         | 具体说明 |
    |:-------------|:---------|
    | 算例名称     | 二维线弹性 |
    | 杨氏模量     | $E = 4/3$ |
    | 泊松比       | $nu = 1/3$ |
    | 计算区域     | $[0, 1]^2$ |
    | 形状参数     | $gamma = 2.0$ |
    | 超平面法向量  | $bold(X) tilde.op cal(N)(0, bold(I)_2)$ |
    | 截距        | $U_n tilde.op cal(U)(0, 1)$ |
    | 训练采样点   | $Q_"train" = (42)^2 = 1764$ |
    | 测试采样点   | $Q_"test" = (64)^2 = 4096$ |
  ],
  caption: [二维线弹性实验设置],
)

为便于与 @hu2014triangle 的二维线弹性算例对比，取制造解
$
  bold(u)_"ex" (x)
  =
  mat(
    e^(x_1 - x_2) x_1 (1 - x_1) x_2 (1 - x_2);
    sin(pi x_1) sin(pi x_2)
  ).
$
这里取 $E = 4/3$、$nu = 1/3$，则 Lamé 参数
$
  mu = E / (2 (1 + nu)) = 1/2,
  quad
  lambda = (E nu) / ((1 + nu)(1 - 2 nu)) = 1,
$
与文献中的二维线弹性系数一致。由于两个分量在边界上均为零，故该制造解满足 $bold(u)_"ex" = 0$ 于 $partial Omega$。精确应力 $bold(sigma)_"ex"$ 由二维各向同性线弹性本构关系计算，体力由
$
  bold(f)_"ex" = - nabla dot bold(sigma)(bold(u)_"ex")
$
通过自动微分生成。

对二维线弹性，位移误差定义为
$
  norm(bold(Phi)^bold(u) - bold(u)_"ex")_0
  := sqrt(sum_(p=1)^(Q_"test") omega_p^"test" sum_(i = 1)^2 [(bold(Phi)^bold(u) (bold(x)_p))_i - (bold(u)_"ex" (bold(x)_p))_i]^2),
$
而在 Voigt 顺序 $(11, 22, 12)$ 下，对应力采用权重 $bold(w)^"V" = (1, 1, 2)^T$，并定义
$
  norm(bold(Phi)^bold(sigma) - bold(sigma)_"ex")_0
  := sqrt(sum_(p=1)^(Q_"test") omega_p^"test" sum_(alpha=1)^3 w^"V"_alpha [(bold(Phi)^(bold(sigma)) (bold(x)_p))_alpha - (bold(sigma)_"ex" (bold(x)_p))_alpha]^2).
$

=== 实验结果

不同特征数量下的数值结果见 @tb:2d-ablation-n。表中 DOF 表示约束后线性系统的未知系数个数；对二维线弹性，
$
  "DOF" = (3(N + 1) - 1) + 2(N + 1) = 5 N + 4.
$

#figure(
  three-line-table(
    columns: 6,
    align: (left, center, center, center, center, center),
  )[
    | 方法 | $N$ | DOF | $norm(bold(Phi)^bold(u) - bold(u)_"ex")_0$ | $norm(bold(Phi)^bold(sigma) - bold(sigma)_"ex")_0$ | Time(s) |
    |:-----------|:----:|:----:|:--------:|:--------:|:-------:|
    | LS | 200  | 1004 | 5.96e-07 | 6.33e-05 |  0.37   |
    | LS | 400  | 2004 | 1.23e-09 | 2.06e-07 |  0.96   |
    | LS | 600  | 3004 | 2.02e-11 | 3.99e-09 |  3.34   |
    | LS | 800  | 4004 | 1.11e-12 | 2.36e-10 |  9.19   |
    | LS | 1000 | 5004 | 1.99e-13 | 3.43e-11 |  21.03  |
  ],
  caption: [二维线弹性在不同特征数量下的数值结果],
)<tb:2d-ablation-n>

#figure(
  image("/public/images/least-squares/linear-elasticity-2d/ablation/N/ablation-N.png"),
  caption: [二维线弹性在不同特征数量下的数值结果],
)

二维算例的误差随 $N$ 增大快速下降。在 $N = 1000$ 时，位移与应力误差分别达到 $1.99 times 10^(-13)$ 和 $3.43 times 10^(-11)$，表明当前固定随机特征空间与直接残差 LS 可以在该光滑二维问题上达到接近机器精度的位移逼近。

作为有限元参照，@hu2014triangle 中 $P_4$ Hu-Zhang 元的结果列于 @tb:2d-hu-zhang。

#figure(
  three-line-table(
    columns: 5,
    align: (center, center, center, center, center),
  )[
    | 方法 | level | DOF | $norm(bold(u) - bold(u)_h)_0$ | $norm(bold(sigma) - bold(sigma)_h)_0$ |
    |:-----------:|:-----:|:----:|:--------:|:--------:|
    | Hu-Zhang ($P_4$) | 1 | 118  | 4.85e-02 | 1.63e-01 |
    | Hu-Zhang ($P_4$) | 2 | 427  | 2.89e-03 | 5.69e-03 |
    | Hu-Zhang ($P_4$) | 3 | 1627 | 1.91e-04 | 1.99e-04 |
    | Hu-Zhang ($P_4$) | 4 | 6355 | 1.21e-05 | 7.00e-06 |
  ],
  caption: [@hu2014triangle 中二维 Hu-Zhang $P_4$ 元的数值结果],
)<tb:2d-hu-zhang>

== 近不可压缩三维线弹性

该算例用于检验 @theorem:elasticity-coercive 中关于 Lamé 常数一致稳定性的数值表现。与前述固定材料参数算例不同，本节固定离散参数与制造位移场，通过泊松比 $nu -> 1/2$ 的参数扫描考察近不可压缩情形下的误差表现。

实验参数设置如下：

#figure(
  three-line-table(
    columns: 2,
    align: (right, left),
  )[
    | 项目         | 具体说明 |
    |:-------------|:---------|
    | 算例名称     | 近不可压缩三维线弹性 |
    | 杨氏模量     | $E = 4/3$ |
    | 计算区域     | $[0, 1]^3$ |
    | 特征数量     | $N_s = N_u = 1000$ |
    | 形状参数     | $gamma_s = gamma_u = 2.0$ |
    | 超平面法向量  | $bold(X) tilde.op cal(N)(0, bold(I)_3)$ |
    | 截距        | $U_n tilde.op cal(U)(0, 1)$ |
    | 训练采样点   | $Q_"train" = (16)^3 = 4096$ |
    | 测试采样点   | $Q_"test" = (16)^3 = 4096$ |
  ],
  caption: [近不可压缩三维线弹性实验设置],
)

制造位移场取为
$
  bold(u)_"ex" (x)
  =
  mat(
    -1 / pi (1 - cos(2 pi x_1)) sin(2 pi x_2) sin(2 pi x_3);
    1 / (2 pi) sin(2 pi x_1) (1 - cos(2 pi x_2)) sin(2 pi x_3);
    1 / (2 pi) sin(2 pi x_1) sin(2 pi x_2) (1 - cos(2 pi x_3))
  ).
$
直接计算可得
$
  nabla dot bold(u)_"ex" = 0,
  quad
  bold(u)_"ex" = bold(0) quad "on" partial Omega.
$
因此该制造解没有体积应变项，精确应力中不会出现随近不可压缩极限放大的 $lambda (nabla dot bold(u)_"ex") bold(I)$。体力仍由
$
  bold(f)_"ex" = - nabla dot bold(sigma)(bold(u)_"ex")
$
通过自动微分生成。位移误差、应力误差与 Voigt 权重的定义沿用三维线弹性算例。

=== 实验结果

不同泊松比下的数值结果见 @tb:3d-ablation-nu。

#figure(
  three-line-table(
    columns: 5,
    align: (left, left, center, center, center),
  )[
    | 方法 | $nu$ | $norm(bold(Phi)^bold(u) - bold(u)_"ex")_0$ | $norm(bold(Phi)^bold(sigma) - bold(sigma)_"ex")_0$ | Time(s) |
    |:-----------|:---------|:--------:|:--------:|:-------:|
    | LS |   0.49   | 2.17e-06 | 1.38e-04 | 169.34  |
    | LS |  0.499   | 2.26e-06 | 1.40e-04 | 168.62  |
    | LS |  0.4999  | 2.27e-06 | 1.40e-04 | 167.73  |
    | LS | 0.49999  | 2.27e-06 | 1.40e-04 | 166.12  |
    | LS | 0.499999 | 2.27e-06 | 1.40e-04 | 168.29  |
  ],
  caption: [近不可压缩三维线弹性在不同泊松比下的数值结果],
)<tb:3d-ablation-nu>

#figure(
  image("/public/images/least-squares/linear-elasticity-3d/ablation/nu/ablation-nu.png"),
  caption: [三维线弹性在不同泊松比下的数值结果],
)

当 $nu$ 从 $0.49$ 增至 $0.499999$ 时，位移误差始终约为 $2.2 times 10^(-6)$，应力误差始终约为 $1.4 times 10^(-4)$，且各次求解耗时接近。这与 @theorem:elasticity-coercive 的 Lamé 参数一致稳定性相符，数值结果未表现出随近不可压缩极限恶化的锁死现象。

== 平面应力

实验参数设置如下：

#figure(
  three-line-table(
    columns: 2,
    align: (right, left),
  )[
    | 项目         | 具体说明 |
    |:-------------|:---------|
    | 算例名称     | 平面应力 |
    | 杨氏模量     | $E = 3/2$ |
    | 泊松比       | $nu = 1/2$ |
    | 计算区域     | $[0, 1]^2$ |
    | 形状参数     | $gamma = 3.0$ |
    | 超平面法向量  | $bold(X) tilde.op cal(N)(0, bold(I)_2)$ |
    | 截距        | $U_n tilde.op cal(U)(0, 1)$ |
    | 内部采样点   | $Q_"train" = (42)^2 = 1764$ |
    | 测试采样点   | $Q_"test" = (64)^2 = 4096$ |
  ],
  caption: [平面应力实验设置],
)

这里取面内位移场
$
  bold(u)_"ex" (x)
  =
  mat(
    e^(x_1 - x_2) x_1 (1 - x_1) x_2 (1 - x_2);
    sin(pi x_1) sin(pi x_2)
  ).
$
这里取 $E = 3/2$、$nu = 1/2$，则平面应力的有效参数满足
$
  mu = E / (2 (1 + nu)) = 1/2,
  quad
  lambda = (E nu) / (1 - nu^2) = 1,
$
体力仍由
$
  bold(f)_"ex" = - nabla dot bold(sigma)(bold(u)_"ex")
$
自动生成，因此可直接用来评估位移与应力误差。

对平面应力，位移误差定义为
$
  norm(bold(Phi)^bold(u) - bold(u)_"ex")_0
  := sqrt(sum_(p=1)^(Q_"test") omega_p^"test" sum_(i = 1)^2 [(bold(Phi)^bold(u) (bold(x)_p))_i - (bold(u)_"ex" (bold(x)_p))_i]^2),
$
而在 Voigt 顺序 $(11, 22, 12)$ 下，对应力采用权重 $bold(w)^"V" = (1, 1, 2)^T$，并定义
$
  norm(bold(Phi)^bold(sigma) - bold(sigma)_"ex")_0
  := sqrt(sum_(p=1)^(Q_"test") omega_p^"test" sum_(alpha=1)^3 w^"V"_alpha [(bold(Phi)^(bold(sigma)) (bold(x)_p))_alpha - (bold(sigma)_"ex" (bold(x)_p))_alpha]^2).
$

=== 实验结果

不同特征数量下的数值结果见 @tb:ps-ablation。

#figure(
  three-line-table(
    columns: 5,
    align: (left, center, center, center, center),
  )[
    | 方法 | $N$ | $norm(bold(Phi)^bold(u) - bold(u)_"ex")_0$ | $norm(bold(Phi)^bold(sigma) - bold(sigma)_"ex")_0$ | Time(s) |
    |:-----------|:----:|:--------:|:--------:|:-------:|
    | LS | 200  | 5.96e-07 | 6.33e-05 |  0.32   |
    | LS | 400  | 1.23e-09 | 2.06e-07 |  0.85   |
    | LS | 600  | 2.03e-11 | 3.99e-09 |  2.88   |
    | LS | 800  | 1.21e-12 | 2.25e-10 |  9.15   |
    | LS | 1000 | 1.68e-13 | 2.96e-11 |  19.55  |
  ],
  caption: [平面应力在不同特征数量下的数值结果],
)<tb:ps-ablation>

#figure(
  image("/public/images/least-squares/plane-stress/ablation/N/ablation-N.png"),
  caption: [平面应力在不同特征数量下的数值结果],
)

平面应力算例与二维线弹性呈现相同的快速收敛趋势。在 $N = 1000$ 时，位移误差为 $1.68 times 10^(-13)$，应力误差为 $2.96 times 10^(-11)$；从 $N = 200$ 到 $N = 1000$，两项误差均下降了多个数量级。

== 板弯曲

该算例对应前述 Kirchhoff-Love 板弯曲问题，以弯矩张量 $bold(cal(M))$ 与标量挠度 $u$ 为未知量。与前三个线弹性类算例不同，它要求 $H_0^2$-保形挠度空间，并同时满足固支边界条件；具体离散推导见附录 @app:plate。

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
    | 形状参数     | $gamma = 2.0$ |
    | 超平面法向量  | $bold(X) tilde.op cal(N)(0, bold(I)_2)$ |
    | 截距        | $U_n tilde.op cal(U)(0, 1)$ |
    | 内部采样点   | $Q_"train" = (32)^2 = 1024$ |
    | 测试采样点   | $Q_"test" = (32)^2 = 1024$ |
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
  norm(Phi^u - u_"ex")_0
  := sqrt(sum_(p=1)^(Q_"test") omega_p^"test" abs(Phi^u (bold(x)_p) - u_"ex" (bold(x)_p))^2),
$
而在 Voigt 顺序 $(11, 22, 12)$ 下，对弯矩采用权重 $bold(w)^"V" = (1, 1, 2)^T$，其绝对误差定义为
$
  norm(bold(Phi)^(bold(cal(M))) - bold(cal(M))_"ex")_0
  := sqrt(sum_(p=1)^(Q_"test") omega_p^"test" sum_(alpha=1)^3 w^"V"_alpha ((bold(Phi)^(bold(cal(M))) (bold(x)_p))_alpha - (bold(cal(M))_"ex" (bold(x)_p))_alpha)^2).
$

=== 实验结果

不同特征数量下的数值结果见 @tb:pb-ablation。

#figure(
  three-line-table(
    columns: 5,
    align: (left, right, right, right, right),
  )[
    | 方法 | $N$ | $norm(Phi^u - u_"ex")_0$ | $norm(bold(Phi)^(bold(cal(M))) - bold(cal(M))_"ex")_0$ |  Time(s) |
    |:-------------------|-------:|-----------:|-----------:|---------:|
    | LS | 200  | 1.60e-10 | 1.15e-07 |  0.14   |
    | LS | 400  | 3.12e-13 | 2.12e-10 |  0.39   |
    | LS | 600  | 3.72e-14 | 2.71e-11 |  1.06   |
    | LS | 800  | 4.18e-14 | 1.71e-11 |  5.22   |
    | LS | 1000 | 1.25e-14 | 1.32e-11 |  9.09   |
  ],
  caption: [板弯曲在不同特征数量下的数值结果],
)<tb:pb-ablation>

#figure(
  image("/public/images/least-squares/plate-bending/ablation/N/ablation-N.png"),
  caption: [板弯曲在不同特征数量下的数值结果],
)

板弯曲的绝对误差同样随特征数量总体下降。挠度误差从 $N = 200$ 时的 $1.60 times 10^(-10)$ 降至 $N = 1000$ 时的 $1.25 times 10^(-14)$；弯矩误差则由 $1.15 times 10^(-7)$ 降至 $1.32 times 10^(-11)$。$N = 600$ 之后挠度误差已接近浮点精度平台，因而局部的轻微非单调不影响整体收敛判断。

= 结语

本文提出了一种面向弹性问题的固定随机特征混合最小二乘方法。其基本思路是先根据混合模型中的本构关系与平衡关系构造最小二乘泛函，再将离散残差直接组装为加权矩形矩阵并求解相应的 LS 问题；在随机特征层面，则通过逐分量随机基、迹消去算子与线性约束获得保形性与齐次边界条件。对于线弹性问题，虽然泛函直接采用本构残差与平衡残差的两项写法，但借助柔度算子的偏差-体积分解仍可得到关于 Lamé 常数 $lambda$ 一致的连续性与强制性估计，从而可以同时处理可压缩与近不可压缩弹性问题。

在具体理论上，本文分别建立了该方法在线弹性应力-位移混合形式与 Kirchhoff-Love 板弯曲弯矩-挠度混合形式下的连续与离散分析。理论结果表明，只要离散空间与相应连续问题的函数空间保持一致，连续稳定性、离散稳定性以及准最优误差估计便可为该方法提供一致的分析基础。数值实验进一步在三维线弹性、二维线弹性、近不可压缩三维线弹性、平面应力与 Kirchhoff-Love 板弯曲算例上验证了该方法的可行性：各特征数量扫描均呈现稳定的总体收敛趋势，近不可压缩扫描中的误差对 $nu -> 1/2$ 保持稳定，二维线弹性、平面应力与板弯曲还获得了很高的绝对 $L^2$ 精度。

#bibliography("/public/reference/least-squares.bib")

#set heading(numbering: "附录 A.1", supplement: [Appendix])
#counter(heading).update(0)

= 线弹性模型 <app:elasticity>

== 等价性定理证明

#proof[
  先证强形式与最小二乘极小化等价。若 $(bold(sigma), bold(u))$ 满足线弹性强形式，则两个残差均为零，从而
  $
    cal(J) ((bold(sigma), bold(u)); bold(f)) = 0.
  $
  由于 $cal(J)$ 是两个 $L^2$ 范数平方之和，必有 $cal(J) >= 0$，因此该解是全局极小点。

  另一方面，假设线弹性方程组在 $bold(Sigma) times bold(U)$ 中存在解 $(bold(sigma)^*, bold(u)^*)$，则
  $
    cal(J) ((bold(sigma)^*, bold(u)^*); bold(f)) = 0,
  $
  从而 $cal(J)$ 的全局最小值为 $0$。因此任意全局极小点 $(bold(sigma), bold(u))$ 都满足
  $
    cal(J) ((bold(sigma), bold(u)); bold(f)) = 0.
  $
  于是
  $
    bold(cal(A)) : bold(sigma) - bold(epsilon)(bold(u)) = 0,
    quad
    nabla dot bold(sigma) + bold(f) = 0
  $
  在 $Omega$ 中几乎处处成立，而 $bold(u) in bold(U)$ 已自动满足齐次位移边界条件，因此 $(bold(sigma), bold(u))$ 满足原强形式。

  下面证明最小二乘极小化与变分问题等价。将 $cal(J)$ 展开可得
  $
    cal(J) ((bold(tau), bold(v)); bold(f))
    = a ((bold(tau), bold(v)), (bold(tau), bold(v)))
    - 2 ell (bold(tau), bold(v))
    + norm(bold(f))_(L^2(Omega))^2.
  $

  若 $(bold(sigma), bold(u))$ 是 $cal(J)$ 的全局极小点，则对任意 $(bold(eta), bold(w)) in bold(Sigma) times bold(U)$，函数
  $
    phi(t) := cal(J) ((bold(sigma) + t bold(eta), bold(u) + t bold(w)); bold(f))
  $
  在 $t = 0$ 处取极小值。由上面的二次展开对 $t$ 求导并取 $t = 0$，得到
  $
    a ((bold(sigma), bold(u)), (bold(eta), bold(w)))
    = ell (bold(eta), bold(w)),
    quad forall (bold(eta), bold(w)) in bold(Sigma) times bold(U),
  $
  即得到变分问题。

  反之，若 $(bold(sigma), bold(u))$ 满足变分问题，则对任意 $(bold(tau), bold(v)) in bold(Sigma) times bold(U)$，记
  $
    (bold(eta), bold(w))
    := (bold(tau) - bold(sigma), bold(v) - bold(u)).
  $
  利用二次展开与变分等式，有
  $
    cal(J) ((bold(tau), bold(v)); bold(f))
    - cal(J) ((bold(sigma), bold(u)); bold(f))
    = a ((bold(eta), bold(w)), (bold(eta), bold(w)))
    >= 0.
  $
  因此 $(bold(sigma), bold(u))$ 是 $cal(J)$ 的全局极小点。证毕。
]

== 具体离散推导

线弹性模型包括三维线弹性、二维线弹性以及平面应力情形。三者共享同一应力-位移混合最小二乘结构；区别只在于维数 $d$、对称张量空间 $SS^d$ 的独立分量数以及柔度算子 $bold(cal(A))$ 的具体材料参数。下述推导统一取 $d in {2, 3}$。

取 ${bold(E)_alpha}_(alpha = 1)^(d(d+1)/2)$ 为 $SS^d$ 的标准对称基，按 Voigt 顺序排列；令 ${bold(e)_i}_(i = 1)^d$ 为 $RR^d$ 的标准基。令
$
  zeta_n^(0, bold(u)) := cal(P)^(0) xi_n^(bold(u)).
$
于是线弹性的离散空间定义为
$
  bold(Sigma)_N & := {
                    sum_(n=0)^N sum_(alpha=1)^(d(d+1)/2)
                    phi^(bold(sigma))_(n, alpha) xi_n^(bold(sigma)) bold(E)_alpha :
                    integral_Omega tr(
                      sum_(n=0)^N sum_(alpha=1)^(d(d+1)/2)
                      phi^(bold(sigma))_(n, alpha) xi_n^(bold(sigma)) bold(E)_alpha
                    ) dif x = 0
                  }, \
      bold(U)_N & := span { zeta_n^(0, bold(u)) bold(e)_i : 0 <= n <= N, 1 <= i <= d }.
$
将离散未知量展开为
$
  bold(Phi)^(bold(sigma))
  = sum_(n=0)^N sum_(alpha=1)^(d(d+1)/2)
  phi^(bold(sigma))_(n, alpha) xi_n^(bold(sigma)) bold(E)_alpha,
$
以及
$
  bold(Phi)^(bold(u))
  = sum_(n=0)^N sum_(i=1)^d
  phi^(bold(u))_(n, i) zeta_n^(0, bold(u)) bold(e)_i.
$
其中应力系数满足
$
  sum_(n=0)^N (integral_Omega xi_n^(bold(sigma)) dif x)
  sum_(i=1)^d phi^(bold(sigma))_(n, i) = 0.
$

离散最小二乘问题等价于：求
$
  (bold(Phi)^(bold(sigma)), bold(Phi)^(bold(u)))
  in bold(Sigma)_N times bold(U)_N,
$
使得
$
  a ((bold(Phi)^(bold(sigma)), bold(Phi)^(bold(u))),
    (bold(Phi)^(bold(tau)), bold(Phi)^(bold(v))))
  = ell (bold(Phi)^(bold(tau)), bold(Phi)^(bold(v))),
  quad forall
  (bold(Phi)^(bold(tau)), bold(Phi)^(bold(v))) in bold(Sigma)_N times bold(U)_N.
$

为与数值实现一致，取训练求积点与权重 ${(bold(x)_p, omega_p)}_(p=1)^Q$，并令 $s = d(d+1)/2$。各张量残差按代码中的 Voigt 分量顺序逐行排列。先不施加零平均迹约束，将本构残差与平衡残差的加权矩阵块定义为
$
  (bold(R)_"c,sigma")_((p, beta), (n, alpha))
  & := sqrt(omega_p)
  (bold(cal(A)) : (xi_n^(bold(sigma))(bold(x)_p) bold(E)_alpha))_beta, \
  (bold(R)_"c,u")_((p, beta), (n, i))
  & := - sqrt(omega_p)
  (bold(epsilon)(zeta_n^(0, bold(u)) bold(e)_i)(bold(x)_p))_beta, \
  (bold(R)_"e,sigma")_((p, j), (n, alpha))
  & := sqrt(omega_p)
  (nabla dot (xi_n^(bold(sigma)) bold(E)_alpha)(bold(x)_p))_j.
$
其中 $1 <= beta, alpha <= s$、$1 <= i, j <= d$。右端在本构残差行上为零，在平衡残差行上定义为
$
  (bold(b)_e)_((p, j)) := - sqrt(omega_p) f_j(bold(x)_p).
$
于是未约束的加权残差矩阵、右端和系数向量分别为
$
  tilde(bold(R))_N
  := mat(
    bold(R)_"c,sigma", bold(R)_"c,u";
    bold(R)_"e,sigma", 0
  ),
  quad
  bold(b)_N := mat(0; bold(b)_e),
  quad
  tilde(bold(phi)) := mat(bold(phi)^(bold(sigma)); bold(phi)^(bold(u))).
$
若 $bold(c)^T bold(phi)^(bold(sigma)) = 0$ 表示零平均迹约束，取 $bold(Z)$ 的列张成完整系数空间中该约束的零空间，并写成 $tilde(bold(phi)) = bold(Z) hat(bold(phi))$。代码实际求解的问题即为
$
  hat(bold(phi))_N
  := arg min_(hat(bold(psi)))
  norm(tilde(bold(R))_N bold(Z) hat(bold(psi)) - bold(b)_N)_2.
$
该目标函数正是连续最小二乘泛函的 Gauss--Legendre 求积离散。由逐分量光滑性、零平均迹约束以及 $zeta_n^(0, bold(u)) in H_0^1(Omega)$ 可知，$bold(Sigma)_N subset bold(Sigma)$、$bold(U)_N subset bold(U)$ 保形，故正文中的稳定性与准最优估计仍适用于所得离散函数。

= Kirchhoff-Love 板弯曲模型 <app:plate>

== 等价性定理证明

#proof[
  若 $(bold(cal(M)), u)$ 满足板弯曲混合强形式，则两个残差同时为零，从而
  $
    cal(J) ((bold(cal(M)), u); f) = 0.
  $
  由于 $cal(J)$ 是两个 $L^2$ 范数平方之和，必有 $cal(J) >= 0$，因此该解是全局极小点。

  另一方面，设混合强形式在 $bold(Sigma) times U$ 中存在解 $(bold(cal(M))^*, u^*)$，则 $cal(J) ((bold(cal(M))^*, u^*); f) = 0$，从而 $cal(J)$ 的全局最小值为 $0$。因此任意全局极小点 $(bold(cal(M)), u)$ 都满足
  $
    bold(cal(A)) : bold(cal(M)) - bold(cal(K))(u) = 0,
    quad
    nabla dot (nabla dot bold(cal(M))) + f = 0
  $
  在 $Omega$ 中几乎处处成立，而 $u in H_0^2(Omega)$ 已自动满足固支边界条件，因此 $(bold(cal(M)), u)$ 满足混合强形式。

  接下来证明最小二乘极小化与变分问题等价。将 $cal(J)$ 展开可得
  $
    cal(J) ((bold(tau), v); f)
    = a ((bold(tau), v), (bold(tau), v))
    - 2 ell (bold(tau), v)
    + norm(f)_(L^2(Omega))^2.
  $
  若 $(bold(cal(M)), u)$ 是 $cal(J)$ 的全局极小点，则对任意 $(bold(eta), w) in bold(Sigma) times U$，函数
  $
    phi(t) := cal(J) ((bold(cal(M)) + t bold(eta), u + t w); f)
  $
  在 $t = 0$ 处取极小值。由上面的二次展开对 $t$ 求导并取 $t = 0$，得到
  $
    a ((bold(cal(M)), u), (bold(eta), w))
    = ell (bold(eta), w),
    quad forall (bold(eta), w) in bold(Sigma) times U,
  $
  即得到变分问题。

  反之，若 $(bold(cal(M)), u)$ 满足变分问题，则对任意 $(bold(tau), v) in bold(Sigma) times U$，记
  $
    (bold(eta), w)
    := (bold(tau) - bold(cal(M)), v - u).
  $
  利用二次展开与变分等式，有
  $
    cal(J) ((bold(tau), v); f)
    - cal(J) ((bold(cal(M)), u); f)
    = a ((bold(eta), w), (bold(eta), w))
    >= 0.
  $
  因此 $(bold(cal(M)), u)$ 是 $cal(J)$ 的全局极小点。证毕。
]

== 具体离散推导

取 ${bold(E)_alpha}_(alpha = 1)^3$ 为 $SS^2$ 的标准对称基，按 Voigt 顺序 $(11, 22, 12)$ 排列。令
$
  zeta_n^(1, u) := cal(P)^(1) xi_n^(u).
$
于是板弯曲的保形离散空间取为
$
  bold(Sigma)_N
  := span { xi_n^(bold(cal(M))) bold(E)_alpha : 0 <= n <= N, 1 <= alpha <= 3 },
$
以及
$
  U_N
  := span { zeta_n^(1, u) : 0 <= n <= N }.
$
将离散未知量展开为
$
  bold(Phi)^(bold(cal(M)))
  = sum_(n=0)^N sum_(alpha=1)^3
  phi^(bold(cal(M)))_(n, alpha) xi_n^(bold(cal(M))) bold(E)_alpha,
$
以及
$
  Phi^(u)
  = sum_(n=0)^N phi_n^(u) zeta_n^(1, u).
$

离散最小二乘问题等价于：求 $(bold(Phi)^(bold(cal(M))), Phi^(u)) in bold(Sigma)_N times U_N$，使得
$
  a ((bold(Phi)^(bold(cal(M))), Phi^(u)),
    (bold(Phi)^(bold(tau)), Phi^(v)))
  = ell (bold(Phi)^(bold(tau)), Phi^(v)),
  quad forall
  (bold(Phi)^(bold(tau)), Phi^(v)) in bold(Sigma)_N times U_N.
$
取训练求积点与权重 ${(bold(x)_p, omega_p)}_(p=1)^Q$，并令 $bold(w)^"V" = (1, 1, 2)^T$。本构残差与平衡残差的加权矩阵块定义为
$
  (bold(R)_"c,M")_((p, beta), (n, alpha))
  & := sqrt(omega_p w_beta^"V")
  (bold(cal(A)) : (xi_n^(bold(cal(M)))(bold(x)_p) bold(E)_alpha))_beta, \
  (bold(R)_"c,u")_((p, beta), n)
  & := - sqrt(omega_p w_beta^"V")
  (bold(cal(K))(zeta_n^(1, u))(bold(x)_p))_beta, \
  (bold(R)_"e,M")_(p, (n, alpha))
  & := sqrt(omega_p)
  (nabla dot (nabla dot (xi_n^(bold(cal(M))) bold(E)_alpha)))(bold(x)_p), \
  (bold(b)_e)_p & := - sqrt(omega_p) f(bold(x)_p).
$
将系数记为 $bold(phi) = mat(bold(phi)^(bold(cal(M))); phi^(u))$，并令
$
  bold(R)_N
  := mat(
    bold(R)_"c,M", bold(R)_"c,u";
    bold(R)_"e,M", 0
  ),
  quad
  bold(b)_N := mat(0; bold(b)_e).
$
则代码求解的板弯曲离散问题为
$
  bold(phi)_N
  := arg min_(bold(psi))
  norm(bold(R)_N bold(psi) - bold(b)_N)_2.
$
该目标函数是板弯曲最小二乘泛函的 Gauss--Legendre 求积离散；由于弯矩与挠度基分别属于 $bold(H)(div div, Omega; SS^2)$ 和 $H_0^2(Omega)$，所得离散函数保持保形。

= 加权残差最小二乘问题的求解 <app:solver>

四类算例最终均写成
$
  bold(phi)_N
  := arg min_(bold(psi) in RR^(K_N))
  norm(bold(R)_N bold(psi) - bold(b)_N)_2.
$
为减小不同残差列尺度的影响，令
$
  s_j := max(norm((bold(R)_N)_(:, j))_2, epsilon s_"max"),
  quad
  s_"max" := max_j norm((bold(R)_N)_(:, j))_2,
  quad
  bold(S) := diag(s_1, dots, s_(K_N)),
$
其中 $epsilon$ 为机器精度。代码先求解缩放后的问题
$
  bold(y)_N
  := arg min_(bold(y))
  norm(bold(R)_N bold(S)^(-1) bold(y) - bold(b)_N)_2,
  quad
  bold(phi)_N = bold(S)^(-1) bold(y)_N,
$
再由 LAPACK GELSD 给出秩揭示的最小二乘解，截断参数取 $10^(-14)$。

当三维残差矩阵不适合整体存储时，代码按点批量组装增广矩阵 $[bold(R)_N, bold(b)_N]$，并用流式 TSQR 得到约化矩阵 $[hat(bold(R))_N, hat(bold(b))_N]$。Householder 正交变换保证存在与 $bold(psi)$ 无关的常数 $C$，使得
$
  norm(bold(R)_N bold(psi) - bold(b)_N)_2^2
  = norm(hat(bold(R))_N bold(psi) - hat(bold(b))_N)_2^2 + C.
$
因此约化问题与原问题具有相同的极小点，二者在本文中均称为 LS。
