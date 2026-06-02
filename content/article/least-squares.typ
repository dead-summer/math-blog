#import "/typ/templates/blog.typ": *

#show: main-zh.with(
  title: "面向弹性问题的固定随机特征混合最小二乘离散方法",
  author: "summer",
  desc: [构造面向弹性问题的混合最小二乘泛函、对应变分问题及随机特征离散方法，并在线弹性、平面应力与 Kirchhoff-Love 板弯曲算例中进行验证],
  date: "2026-04-20",
  tags: (
    blog-tags.numerical-methods,
    blog-tags.pde,
  ),
  show-outline: true,
)

= 引言

在固定随机特征方法中，隐藏层参数首先随机生成并冻结，随后仅对输出层系数求解。这样一来，偏微分方程的近似被转化为有限维线性代数问题。对于弹性问题而言，混合表示尤其自然：本构关系与平衡关系可以被显式分离，应力、弯矩等辅助变量也可以被独立近似。

本文的核心不是把弹性问题方程组本身作为主要对象，而是面向一类弹性问题构造混合最小二乘泛函，并由其一阶最优性条件得到对应的变分问题，进而建立固定随机特征离散方法。在这一框架下，离散设计的关键是按照连续模型的自然空间构造保形试探空间，并通过包络函数强制施加齐次边界条件。连续稳定性、离散稳定性以及准最优误差估计为该离散构造提供理论支撑；在线弹性、平面应力与 Kirchhoff-Love 板弯曲三个算例中的数值实验则用于检验这一方法在不同模型下的实际表现。

= 二阶线弹性模型

本节建立二阶线弹性模型理论。设 $Omega subset RR^d, d in {2, 3}$ 为有界、连通且边界 Lipschitz 的区域，未知量为对称应力张量 $bold(sigma): Omega -> SS^d$ 与位移向量 $bold(u): Omega -> RR^d$。考虑如下方程组
$
  cases(
    bold(cal(A)) : bold(sigma) - bold(epsilon)(bold(u)) & = 0 & quad "in" Omega,
    nabla dot bold(sigma) + bold(f) & = 0 & quad "in" Omega,
    bold(u) & = 0 & quad "on" partial Omega,
  )
$
其中
$
  bold(epsilon)(bold(u)) = 1/2 (nabla bold(u) + (nabla bold(u))^T)
$
为线性应变张量，$bold(cal(A))$ 为在 $SS^d$ 上有界且一致正定的柔度算子，即存在常数 $0 < c_A <= C_A < oo$ 使得
$
  c_A norm(bold(tau))_(L^2(Omega))^2
  <= integral_Omega (bold(cal(A)) : bold(tau)) : bold(tau) dif x
  <= C_A norm(bold(tau))_(L^2(Omega))^2,
  quad forall bold(tau) in (L^2(Omega))^(d times d) inter SS^d.
$
这里 $bold(cal(A))$ 可视为四阶张量；$bold(cal(A)) : bold(tau)$ 表示它对二阶对称张量 $bold(tau)$ 的双重缩并，结果仍为二阶对称张量。
定义
$
  bold(Sigma) := bold(H)(div, Omega; SS^d),
  quad
  bold(U) := (H_0^1(Omega))^d.
$

== 最小二乘泛函与变分

定义二阶线弹性的最小二乘泛函
$
  cal(J) ((bold(tau), bold(v)); bold(f))
  := norm(bold(cal(A)) : bold(tau) - bold(epsilon)(bold(v)))_(L^2(Omega))^2
  + norm(nabla dot bold(tau) + bold(f))_(L^2(Omega))^2.
$
相应双线性形式记为
$
  a ((bold(sigma), bold(u)), (bold(tau), bold(v)))
  := (bold(cal(A)) : bold(sigma) - bold(epsilon)(bold(u)),
    bold(cal(A)) : bold(tau) - bold(epsilon)(bold(v)))_(L^2(Omega))
  + (nabla dot bold(sigma), nabla dot bold(tau))_(L^2(Omega)).
$
右端线性泛函为
$
  ell (bold(tau), bold(v))
  := - (bold(f), nabla dot bold(tau))_(L^2(Omega)).
$

可以证明：当连续强形式解在上述空间中存在时，线弹性方程组、最小二乘泛函极小化与变分问题三者等价。

#theorem(title: [二阶线弹性强形式、最小二乘极小化与变分问题的等价性])[
  若二阶线弹性的强形式在 $bold(Sigma) times bold(U)$ 中解存在，则下列三个问题彼此等价：

  1. 二阶线弹性的强形式；
  2. 最小化 $cal(J)$；
  3. 求 $(bold(sigma), bold(u)) in bold(Sigma) times bold(U)$，使得
    $
      a ((bold(sigma), bold(u)), (bold(tau), bold(v)))
      = ell (bold(tau), bold(v)),
      quad forall (bold(tau), bold(v)) in bold(Sigma) times bold(U).
    $
]

#proof[
  先证强形式与最小二乘极小化等价。若 $(bold(sigma), bold(u))$ 满足强形式，则两个残差都为零，从而
  $
    cal(J) ((bold(sigma), bold(u)); bold(f)) = 0.
  $
  由于 $cal(J)$ 是两个 $L^2$ 范数平方之和，必有 $cal(J) >= 0$，因此该解是全局极小点。

  另一方面，假设二阶线弹性方程组在 $bold(Sigma) times bold(U)$ 中存在解 $(bold(sigma)^*, bold(u)^*)$，则
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
  在 $Omega$ 中几乎处处成立，而 $bold(u) in (H_0^1(Omega))^d$ 已自动满足齐次位移边界条件，因此 $(bold(sigma), bold(u))$ 满足原强形式。

  下面证明最小二乘极小化与变分问题等价。将 $cal(J)$ 展开可得
  $
    cal(J) ((bold(tau), bold(v)); bold(f))
    = a ((bold(tau), bold(v)), (bold(tau), bold(v)))
    - 2 ell (bold(tau), bold(v))
    + norm(bold(f))_(L^2(Omega))^2.
  $

  若 $(bold(sigma), bold(u))$ 是 $cal(J)$ 的全局极小点，则对任意 $(bold(eta), bold(w)) in bold(Sigma) times bold(U)$，函数
  $
    phi(t) :=& cal(J) ((bold(sigma) + t bold(eta), bold(u) + t bold(w)); bold(f)) \
    =& a((bold(sigma) + t bold(eta), bold(u) + t bold(w)), (bold(sigma) + t bold(eta), bold(u) + t bold(w)))
    \
    &- 2 ell (bold(sigma) + t bold(eta), bold(u) + t bold(w)) + norm(f)_(L^2(Omega))^2 \
    =& a((bold(sigma), bold(u)), (bold(sigma), bold(u))) + 2 t a((bold(sigma), bold(u)), (bold(eta), bold(w))) + t^2 a((bold(eta), bold(w)), (bold(eta), bold(w))) \
    &- 2 ell(bold(sigma), bold(u)) - 2 t ell (bold(eta), bold(w)) + norm(f)_(L^2(Omega))^2,
  $
  在 $t = 0$ 处取极小值。对上式求导，得到
  $
    phi'(t) = 2 a((bold(sigma), bold(u)), (bold(eta), bold(w))) + 2t a((bold(eta), bold(w)), (bold(eta), bold(w))) - 2 ell (bold(eta), bold(w)),
  $
  令 $t = 0$，则
  $
    0 = phi'(0)
    = 2 a ((bold(sigma), bold(u)), (bold(eta), bold(w)))
    - 2 ell (bold(eta), bold(w)).
  $
  因此
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
  利用上面的二次展开与变分等式，有
  $
    & cal(J) ((bold(tau), bold(v)); bold(f))
      - cal(J) ((bold(sigma), bold(u)); bold(f)) \
    & = 2 a ((bold(sigma), bold(u)), (bold(eta), bold(w)))
      + a ((bold(eta), bold(w)), (bold(eta), bold(w)))
      - 2 ell (bold(eta), bold(w)) \
    & = a ((bold(eta), bold(w)), (bold(eta), bold(w)))
      >= 0.
  $
  因此 $(bold(sigma), bold(u))$ 是 $cal(J)$ 的全局极小点。证毕。
]

== 连续稳定性

#theorem(title: [二阶线弹性连续稳定性的双边估计])[
  存在常数 $0 < c <= C < oo$，仅依赖于 $Omega$ 与材料参数，使得对任意 $(bold(tau), bold(v)) in bold(Sigma) times bold(U)$ 都有
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
]<thm:continuous-stability>

#proof[
  先证上界。由 $bold(cal(A))$ 的有界性与
  $
    norm(bold(epsilon)(bold(v)))_(L^2(Omega))
    <= norm(nabla bold(v))_(L^2(Omega))
    <= norm(bold(v))_(H^1(Omega))
  $
  可得
  $
    norm(bold(cal(A)) : bold(tau) - bold(epsilon)(bold(v)))_(L^2(Omega))^2
    <= 2 C_A^2 norm(bold(tau))_(L^2(Omega))^2
    + 2 norm(bold(v))_(H^1(Omega))^2.
  $
  再加上
  $
    norm(nabla dot bold(tau))_(L^2(Omega))^2
    <= norm(bold(tau))_(bold(H)(div))^2,
  $
  即得上界。

  再证下界。记
  $
    bold(r)_"c" := bold(cal(A)) : bold(tau) - bold(epsilon)(bold(v)),
    quad
    bold(r)_"e" := nabla dot bold(tau).
  $
  则
  $
    bold(epsilon)(bold(v)) = bold(cal(A)) : bold(tau) - bold(r)_"c".
  $
  由 Korn 第一不等式（参考 @BrennerScott2008 推论 11.2.25），存在常数 $C_K > 0$，使得
  $
    norm(bold(v))_(H^1(Omega))
    <= C_K norm(bold(epsilon)(bold(v)))_(L^2(Omega))
    <= C_K (
      norm(bold(r)_"c")_(L^2(Omega))
      + C_A norm(bold(tau))_(L^2(Omega))
    ).
  $
  <eq:korn-est>

  另一方面，由柔度算子的椭圆性，
  $
    c_A norm(bold(tau))_(L^2(Omega))^2
    <= integral_Omega (bold(cal(A)) : bold(tau)) : bold(tau) dif x.
  $
  又因为 $bold(tau)$ 对称且 $bold(v) in (H_0^1(Omega))^d$，分部积分给出
  $
    integral_Omega bold(epsilon)(bold(v)) : bold(tau) dif x
    = - integral_Omega bold(v) dot (nabla dot bold(tau)) dif x
    = - integral_Omega bold(v) dot bold(r)_"e" dif x.
  $
  于是
  $
    c_A norm(bold(tau))_(L^2(Omega))^2
    <= norm(bold(v))_(H^1(Omega)) norm(bold(r)_"e")_(L^2(Omega))
    + norm(bold(r)_"c")_(L^2(Omega)) norm(bold(tau))_(L^2(Omega)).
  $
  将 @eq:korn-est 代入，对右端逐项使用 Young 不等式并把含 $norm(bold(tau))_(L^2)^2$ 的项吸收到左端，可得
  $
    norm(bold(tau))_(L^2(Omega))^2
    <= C_1 (
      norm(bold(r)_"c")_(L^2(Omega))^2
      + norm(bold(r)_"e")_(L^2(Omega))^2
    ).
  $
  再代回 @eq:korn-est，即得
  $
    norm(bold(v))_(H^1(Omega))^2
    <= C_2 (
      norm(bold(r)_"c")_(L^2(Omega))^2
      + norm(bold(r)_"e")_(L^2(Omega))^2
    ).
  $
  最后利用
  $
    norm(bold(tau))_(bold(H)(div))^2
    = norm(bold(tau))_(L^2(Omega))^2
    + norm(bold(r)_"e")_(L^2(Omega))^2
  $
  即得下界，证毕。
]

#corollary(title: [二阶线弹性双线性形式 $a$ 的连续性])[
  对任意 $(bold(sigma), bold(u)), (bold(tau), bold(v)) in bold(Sigma) times bold(U)$，
  $
    abs(a ((bold(sigma), bold(u)), (bold(tau), bold(v))))
    <= C (
      norm(bold(sigma))_(bold(H)(div))^2
      + norm(bold(u))_(H^1(Omega))^2
    )^(1/2)
    (
      norm(bold(tau))_(bold(H)(div))^2
      + norm(bold(v))_(H^1(Omega))^2
    )^(1/2),
  $
  其中常数 $C$ 来源于 @thm:continuous-stability。
]<cor:a-continuity>

#proof[
  令 $bold(f) = bold(0)$。由最小二乘泛函的定义，
  $
    cal(J) ((bold(tau), bold(v)); bold(0))
    = norm(bold(cal(A)) : bold(tau) - bold(epsilon)(bold(v)))_(L^2(Omega))^2
    + norm(nabla dot bold(tau))_(L^2(Omega))^2.
  $
  因此 $a$ 正是该二次型对应的极化双线性形式。对任意
  $(bold(sigma), bold(u)), (bold(tau), bold(v)) in bold(Sigma) times bold(U)$，
  由 Cauchy-Schwarz 不等式，
  $
    abs(a ((bold(sigma), bold(u)), (bold(tau), bold(v))))
    & <=
    norm(bold(cal(A)) : bold(sigma) - bold(epsilon)(bold(u)))_(L^2(Omega))
    norm(bold(cal(A)) : bold(tau) - bold(epsilon)(bold(v)))_(L^2(Omega))
    \
    & quad
    + norm(nabla dot bold(sigma))_(L^2(Omega))
    norm(nabla dot bold(tau))_(L^2(Omega)) \
    & <=
    cal(J) ((bold(sigma), bold(u)); bold(0))^(1/2)
    cal(J) ((bold(tau), bold(v)); bold(0))^(1/2).
  $
  再由 @thm:continuous-stability 的上界，
  $
    cal(J) ((bold(sigma), bold(u)); bold(0))
    <= C (
      norm(bold(sigma))_(bold(H)(div))^2
      + norm(bold(u))_(H^1(Omega))^2
    ),
  $
  且对 $(bold(tau), bold(v))$ 同理。代入上式可得
  $
    abs(a ((bold(sigma), bold(u)), (bold(tau), bold(v))))
    <= C (
      norm(bold(sigma))_(bold(H)(div))^2
      + norm(bold(u))_(H^1(Omega))^2
    )^(1/2)
    (
      norm(bold(tau))_(bold(H)(div))^2
      + norm(bold(v))_(H^1(Omega))^2
    )^(1/2).
  $
  即 $a$ 在 $bold(Sigma) times bold(U)$ 上连续。证毕。
]

上述定理说明，二阶最小二乘双线性形式 $a$ 在 $bold(H)(div, Omega; SS^d) times (H_0^1(Omega))^d$ 上连续（上界）且强制（下界），因此对应离散系统自然导出对称正定结构。

== 离散稳定性与准最优误差估计

设 $bold(Sigma)_M subset bold(Sigma)$、$bold(U)_M subset bold(U)$ 为任意有限维保形子空间。离散最小二乘问题为：求 $(bold(sigma)_M, bold(u)_M) in bold(Sigma)_M times bold(U)_M$，使得
$
  a ((bold(sigma)_M, bold(u)_M), (bold(tau)_M, bold(v)_M))
  = ell (bold(tau)_M, bold(v)_M),
  quad forall (bold(tau)_M, bold(v)_M) in bold(Sigma)_M times bold(U)_M.
$

在任意保形有限维子空间上仍可建立一致稳定性。

#theorem(title: [二阶线弹性离散问题的存在唯一性与稳定性])[
  对任意保形有限维子空间 $bold(Sigma)_M subset bold(Sigma)$ 与 $bold(U)_M subset bold(U)$，离散问题存在唯一解。记离散问题的解为 $(bold(sigma)_M, bold(u)_M)$，则有
  $
    norm(bold(sigma)_M)_(bold(H)(div))
    + norm(bold(u)_M)_(H^1(Omega))
    <= sqrt(2)/c norm(bold(f))_(L^2(Omega)),
  $
  其中常数 $c$ 来源于 @thm:continuous-stability。
]

#proof[
  由于 $bold(Sigma)_M subset bold(Sigma)$ 且 $bold(U)_M subset bold(U)$，上一节的连续强制性可直接限制到离散子空间上，并且强制性常数不随 $M$ 改变。也就是说，对任意 $(bold(tau)_M, bold(v)_M) in bold(Sigma)_M times bold(U)_M$，都有
  $
    c (
      norm(bold(tau)_M)_(bold(H)(div))^2
      + norm(bold(v)_M)_(H^1(Omega))^2
    )
    <= a ((bold(tau)_M, bold(v)_M), (bold(tau)_M, bold(v)_M)).
  $
  同时由 $a$ 的连续性，$a$ 在离散子空间上也是连续的，且连续性常数同样不依赖于离散维数。

  右端泛函 $ell$ 满足
  $
    abs(ell (bold(tau)_M, bold(v)_M))
    = abs((bold(f), nabla dot bold(tau)_M)_(L^2(Omega)))
    <= norm(bold(f))_(L^2(Omega)) norm(bold(tau)_M)_(bold(H)(div))
    <= norm(bold(f))_(L^2(Omega)) (
      norm(bold(tau)_M)_(bold(H)(div))^2
      + norm(bold(v)_M)_(H^1(Omega))^2
    )^(1/2),
  $
  因此 $ell$ 在离散空间上一致连续。由 Lax-Milgram 定理（参考 @BrennerScott2008 定理 2.7.7），离散解存在唯一。

  取测试函数 $(bold(tau)_M, bold(v)_M) = (bold(sigma)_M, bold(u)_M)$，得到
  $
    a ((bold(sigma)_M, bold(u)_M), (bold(sigma)_M, bold(u)_M))
    = ell (bold(sigma)_M, bold(u)_M)
    <= norm(bold(f))_(L^2(Omega)) norm(bold(sigma)_M)_(bold(H)(div)).
  $
  记
  $
    C_M
    := (
      norm(bold(sigma)_M)_(bold(H)(div))^2
      + norm(bold(u)_M)_(H^1(Omega))^2
    )^(1/2).
  $
  由强制性与上式可得
  $
    c C_M^2 & <= cal(J) ((bold(sigma)_M, bold(u)_M); bold(0)) = a ((bold(sigma)_M, bold(u)_M), (bold(sigma)_M, bold(u)_M)) \
            & <= norm(bold(f))_(L^2(Omega)) norm(bold(sigma)_M)_(bold(H)(div))
              <= norm(bold(f))_(L^2(Omega)) C_M.
  $
  若 $C_M = 0$，稳定性估计显然成立；若 $C_M > 0$，两端同除以 $C_M$，得到
  $
    C_M <= c^(-1) norm(bold(f))_(L^2(Omega)).
  $
  最后利用
  $
    norm(bold(sigma)_M)_(bold(H)(div))
    + norm(bold(u)_M)_(H^1(Omega))
    <= sqrt(2) C_M,
  $
  即得
  $
    norm(bold(sigma)_M)_(bold(H)(div))
    + norm(bold(u)_M)_(H^1(Omega))
    <= sqrt(2)/c norm(bold(f))_(L^2(Omega)),
  $
  证毕。
]

#theorem(title: [二阶线弹性的准最优误差估计])[
  设连续解 $(bold(sigma), bold(u)) in bold(Sigma) times bold(U)$ 存在，离散解 $(bold(sigma)_M, bold(u)_M) in bold(Sigma)_M times bold(U)_M$ 由上式给出。则有
  $
    norm(bold(sigma) - bold(sigma)_M)_(bold(H)(div))
    + norm(bold(u) - bold(u)_M)_(H^1(Omega))
    <= (sqrt(2) C) / c
    inf_((bold(tau)_M, bold(v)_M) in bold(Sigma)_M times bold(U)_M)
    (
      norm(bold(sigma) - bold(tau)_M)_(bold(H)(div))
      + norm(bold(u) - bold(v)_M)_(H^1(Omega))
    ),
  $
  其中常数 $c$ 和 $C$ 来源于 @thm:continuous-stability 。
]

#proof[
  连续解满足连续变分问题，离散解满足离散变分问题。由于 $bold(Sigma)_M subset bold(Sigma)$ 且 $bold(U)_M subset bold(U)$，任意离散测试函数也可作为连续测试函数。因此二者相减可得 Galerkin 正交性：
  $
    a (
      (bold(sigma) - bold(sigma)_M, bold(u) - bold(u)_M),
      (bold(tau)_M, bold(v)_M)
    )
    = 0,
    quad forall (bold(tau)_M, bold(v)_M) in bold(Sigma)_M times bold(U)_M.
  $

  记误差
  $
    (bold(e)_sigma, bold(e)_u)
    := (bold(sigma) - bold(sigma)_M, bold(u) - bold(u)_M).
  $
  对任意 $(bold(tau)_M, bold(v)_M) in bold(Sigma)_M times bold(U)_M$，有
  $
    (bold(e)_sigma, bold(e)_u)
    = (bold(sigma) - bold(tau)_M, bold(u) - bold(v)_M)
    + (bold(tau)_M - bold(sigma)_M, bold(v)_M - bold(u)_M),
  $
  其中第二项属于 $bold(Sigma)_M times bold(U)_M$。由连续强制性与 Galerkin 正交性，
  $
    & c (
        norm(bold(e)_sigma)_(bold(H)(div))^2
        + norm(bold(e)_u)_(H^1(Omega))^2
      ) \
    & <= a ((bold(e)_sigma, bold(e)_u), (bold(e)_sigma, bold(e)_u)) \
    & = a (
        (bold(e)_sigma, bold(e)_u),
        (bold(sigma) - bold(tau)_M, bold(u) - bold(v)_M)
      ).
  $
  由 @cor:a-continuity，
  $
       & a (
           (bold(e)_sigma, bold(e)_u),
           (bold(sigma) - bold(tau)_M, bold(u) - bold(v)_M)
         ) \
    <= & C (
           norm(bold(e)_sigma)_(bold(H)(div))^2
           + norm(bold(e)_u)_(H^1(Omega))^2
         )^(1/2) times (
           norm(bold(sigma) - bold(tau)_M)_(bold(H)(div))^2
           + norm(bold(u) - bold(v)_M)_(H^1(Omega))^2
         )^(1/2).
  $
  若误差范数为零，则结论显然成立；否则两端同除以
  $
    (
      norm(bold(e)_sigma)_(bold(H)(div))^2
      + norm(bold(e)_u)_(H^1(Omega))^2
    )^(1/2),
  $
  得到
  $
    (
      norm(bold(e)_sigma)_(bold(H)(div))^2
      + norm(bold(e)_u)_(H^1(Omega))^2
    )^(1/2)
    <= C/c
    (
      norm(bold(sigma) - bold(tau)_M)_(bold(H)(div))^2
      + norm(bold(u) - bold(v)_M)_(H^1(Omega))^2
    )^(1/2).
  $
  再利用
  $
    norm(bold(e)_sigma)_(bold(H)(div))
    + norm(bold(e)_u)_(H^1(Omega))
    <= sqrt(2) (
      norm(bold(e)_sigma)_(bold(H)(div))^2
      + norm(bold(e)_u)_(H^1(Omega))^2
    )^(1/2)
  $
  以及
  $
    (
      norm(bold(sigma) - bold(tau)_M)_(bold(H)(div))^2
      + norm(bold(u) - bold(v)_M)_(H^1(Omega))^2
    )^(1/2)
    <= norm(bold(sigma) - bold(tau)_M)_(bold(H)(div))
    + norm(bold(u) - bold(v)_M)_(H^1(Omega)),
  $
  并对所有 $(bold(tau)_M, bold(v)_M) in bold(Sigma)_M times bold(U)_M$ 取下确界，即得所述估计。
]

= Kirchhoff-Love 板弯曲模型

板弯曲不再落在 $bold(H)(div) times H_0^1$ 这一二阶框架中，因此需要单独建立与之平行的四阶理论。设 $Omega subset RR^2$ 为板中面区域，未知量为弯矩张量 $bold(cal(M)): Omega -> SS^2$ 与标量挠度 $u: Omega -> RR$。定义曲率张量
$
  bold(cal(K))(u) := - nabla^2 u.
$
于是 Kirchhoff-Love 板的混合强形式写为
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
  D = (E h^3)/(12(1 - nu^2))
$
为弯曲刚度。一般地，假设 $bold(cal(A))$ 在 $SS^2$ 上有界且一致正定，即存在常数 $0 < c_A <= C_A < oo$ 使得
$
  c_A norm(bold(tau))_(L^2(Omega))^2
  <= integral_Omega (bold(cal(A)) : bold(tau)) : bold(tau) dif x
  <= C_A norm(bold(tau))_(L^2(Omega))^2,
  quad forall bold(tau) in (L^2(Omega))^(2 times 2) inter SS^2.
$
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

定义板弯曲的最小二乘泛函
$
  cal(J)_"plate" ((bold(tau), v); f)
  := norm(bold(cal(A)) : bold(tau) - bold(cal(K))(v))_(L^2(Omega))^2
  + norm(nabla dot (nabla dot bold(tau)) + f)_(L^2(Omega))^2.
$
相应双线性形式为
$
  a_"plate" ((bold(cal(M)), u), (bold(tau), v))
  := & (bold(cal(A)) : bold(cal(M)) - bold(cal(K))(u),
         bold(cal(A)) : bold(tau) - bold(cal(K))(v))_(L^2(Omega)) \
     & + (nabla dot (nabla dot bold(cal(M))),
         nabla dot (nabla dot bold(tau)))_(L^2(Omega)),
$
右端项为
$
  ell_"plate" (bold(tau), v)
  := - (f, nabla dot (nabla dot bold(tau)))_(L^2(Omega)).
$

#theorem(title: [板弯曲强形式、最小二乘极小化与变分问题的等价性])[
  若 Kirchhoff-Love 板弯曲的强形式在 $bold(Sigma) times U$ 中解存在，则下列三个问题彼此等价：

  1. Kirchhoff-Love 板弯曲的强形式；
  2. 最小化 $cal(J)_"plate"$；
  3. 求 $(bold(cal(M)), u) in bold(Sigma) times U$，使得
    $
      a_"plate" ((bold(cal(M)), u), (bold(tau), v))
      = ell_"plate" (bold(tau), v),
      quad forall (bold(tau), v) in bold(Sigma) times U.
    $
]

#proof[
  若 $(bold(cal(M)), u)$ 满足混合强形式，则两个残差同时为零，从而
  $
    cal(J)_"plate" ((bold(cal(M)), u); f) = 0.
  $
  由于 $cal(J)_"plate"$ 是两个 $L^2$ 范数平方之和，必有 $cal(J)_"plate" >= 0$，因此该解是全局极小点。

  另一方面，设混合强形式在 $bold(Sigma) times U$ 中存在解 $(bold(cal(M))^*, u^*)$，则
  $
    cal(J)_"plate" ((bold(cal(M))^*, u^*); f) = 0,
  $
  从而 $cal(J)_"plate"$ 的全局最小值为 $0$。因此任意全局极小点 $(bold(cal(M)), u)$ 都满足
  $
    cal(J)_"plate" ((bold(cal(M)), u); f) = 0.
  $
  于是
  $
    bold(cal(A)) : bold(cal(M)) - bold(cal(K))(u) = 0,
    quad
    nabla dot (nabla dot bold(cal(M))) + f = 0
  $
  在 $Omega$ 中几乎处处成立，而 $u in H_0^2(Omega)$ 已自动满足固支边界条件，因此 $(bold(cal(M)), u)$ 满足混合强形式。

  接下来证明最小二乘极小化与变分问题等价。将 $cal(J)_"plate"$ 展开可得
  $
    cal(J)_"plate" ((bold(tau), v); f)
    = a_"plate" ((bold(tau), v), (bold(tau), v))
    - 2 ell_"plate" (bold(tau), v)
    + norm(f)_(L^2(Omega))^2.
  $

  若 $(bold(cal(M)), u)$ 是 $cal(J)_"plate"$ 的全局极小点，则对任意 $(bold(eta), w) in bold(Sigma) times U$，函数
  $
    phi(t) :=& cal(J)_"plate" ((bold(cal(M)) + t bold(eta), u + t w); f) \
    =& a_"plate" ((bold(cal(M)) + t bold(eta), u + t w), (bold(cal(M)) + t bold(eta), u + t w))
    \
    &- 2 ell_"plate" (bold(cal(M)) + t bold(eta), u + t w) + norm(f)_(L^2(Omega))^2 \
    =& a_"plate" ((bold(cal(M)), u), (bold(cal(M)), u)) + 2 t a_"plate" ((bold(cal(M)), u), (bold(eta), w)) + t^2 a_"plate" ((bold(eta), w), (bold(eta), w)) \
    &- 2 ell_"plate" (bold(cal(M)), u) - 2 t ell_"plate" (bold(eta), w) + norm(f)_(L^2(Omega))^2,
  $
  在 $t = 0$ 处取极小值。对上式求导，得到
  $
    phi'(t) = 2 a_"plate" ((bold(cal(M)), u), (bold(eta), w)) + 2t a_"plate" ((bold(eta), w), (bold(eta), w)) - 2 ell_"plate" (bold(eta), w),
  $
  令 $t = 0$，则
  $
    0 = phi'(0)
    = 2 a_"plate" ((bold(cal(M)), u), (bold(eta), w))
    - 2 ell_"plate" (bold(eta), w).
  $
  因此
  $
    a_"plate" ((bold(cal(M)), u), (bold(eta), w))
    = ell_"plate" (bold(eta), w),
    quad forall (bold(eta), w) in bold(Sigma) times U,
  $
  即得到变分问题。

  反之，若 $(bold(cal(M)), u)$ 满足变分问题，则对任意 $(bold(tau), v) in bold(Sigma) times U$，记
  $
    (bold(eta), w)
    := (bold(tau) - bold(cal(M)), v - u).
  $
  利用上面的二次展开与变分等式，有
  $
    & cal(J)_"plate" ((bold(tau), v); f)
      - cal(J)_"plate" ((bold(cal(M)), u); f) \
    & = 2 a_"plate" ((bold(cal(M)), u), (bold(eta), w))
      + a_"plate" ((bold(eta), w), (bold(eta), w))
      - 2 ell_"plate" (bold(eta), w) \
    & = a_"plate" ((bold(eta), w), (bold(eta), w))
      >= 0.
  $
  因此 $(bold(cal(M)), u)$ 是 $cal(J)_"plate"$ 的全局极小点。证毕。
]

== 连续稳定性

#theorem(title: [板弯曲连续稳定性的双边估计])[
  存在常数 $0 < c_"plate" <= C_"plate" < oo$，仅依赖于 $Omega$ 与材料参数，使得对任意 $(bold(tau), v) in bold(Sigma) times U$ 都有
  $
    c_"plate" (
      norm(bold(tau))_(bold(H)(div div))^2
      + norm(v)_(H^2(Omega))^2
    )
    <= cal(J)_"plate" ((bold(tau), v); 0)
    <= C_"plate" (
      norm(bold(tau))_(bold(H)(div div))^2
      + norm(v)_(H^2(Omega))^2
    ).
  $
]<thm:plate-continuous-stability>

#proof[
  上界由 $bold(cal(A))$ 的有界性直接得到：
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
  即得下界。证毕。
]

#corollary(title: [板弯曲双线性形式 $a_"plate"$ 的连续性])[
  对任意 $(bold(cal(M)), u), (bold(tau), v) in bold(Sigma) times U$，
  $
    abs(a_"plate" ((bold(cal(M)), u), (bold(tau), v)))
    <= C_"plate" (
      norm(bold(cal(M)))_(bold(H)(div div))^2
      + norm(u)_(H^2(Omega))^2
    )^(1/2)
    (
      norm(bold(tau))_(bold(H)(div div))^2
      + norm(v)_(H^2(Omega))^2
    )^(1/2),
  $
  其中常数 $C_"plate"$ 来源于 @thm:plate-continuous-stability。
]<cor:a-plate-continuity>

#proof[
  取 $f = 0$。由板弯曲最小二乘泛函的定义，
  $
    cal(J)_"plate" ((bold(tau), v); 0)
    = norm(bold(cal(A)) : bold(tau) - bold(cal(K))(v))_(L^2(Omega))^2
    + norm(nabla dot (nabla dot bold(tau)))_(L^2(Omega))^2.
  $
  因此 $a_"plate"$ 正是该二次型对应的极化双线性形式。对任意
  $(bold(cal(M)), u), (bold(tau), v) in bold(Sigma) times U$，
  由 Cauchy-Schwarz 不等式，
  $
    & abs(a_"plate" ((bold(cal(M)), u), (bold(tau), v))) \
    & <=
      norm(bold(cal(A)) : bold(cal(M)) - bold(cal(K))(u))_(L^2(Omega))
      norm(bold(cal(A)) : bold(tau) - bold(cal(K))(v))_(L^2(Omega)) \
    & quad
      + norm(nabla dot (nabla dot bold(cal(M))))_(L^2(Omega))
      norm(nabla dot (nabla dot bold(tau)))_(L^2(Omega)) \
    & <=
      cal(J)_"plate" ((bold(cal(M)), u); 0)^(1/2)
      cal(J)_"plate" ((bold(tau), v); 0)^(1/2).
  $
  再由 @thm:plate-continuous-stability 的上界，
  $
    cal(J)_"plate" ((bold(cal(M)), u); 0)
    <= C_"plate" (
      norm(bold(cal(M)))_(bold(H)(div div))^2
      + norm(u)_(H^2(Omega))^2
    ),
  $
  且对 $(bold(tau), v)$ 同理。代入上式可得
  $
    abs(a_"plate" ((bold(cal(M)), u), (bold(tau), v)))
    <= C_"plate" (
      norm(bold(cal(M)))_(bold(H)(div div))^2
      + norm(u)_(H^2(Omega))^2
    )^(1/2)
    (
      norm(bold(tau))_(bold(H)(div div))^2
      + norm(v)_(H^2(Omega))^2
    )^(1/2).
  $
  即 $a_"plate"$ 在 $bold(Sigma) times U$ 上连续。证毕。
]

== 离散稳定性与准最优误差估计

设 $bold(Sigma)_M subset bold(Sigma)$、$U_M subset U$ 为任意有限维保形子空间。离散板弯曲问题为：求 $(bold(cal(M))_M, u_M) in bold(Sigma)_M times U_M$，使得
$
  a_"plate" ((bold(cal(M))_M, u_M), (bold(tau)_M, v_M))
  = ell_"plate" (bold(tau)_M, v_M),
  quad forall (bold(tau)_M, v_M) in bold(Sigma)_M times U_M.
$

#theorem(title: [板弯曲离散问题的存在唯一性与稳定性])[
  对任意保形有限维子空间 $bold(Sigma)_M subset bold(Sigma)$ 与 $U_M subset U$，离散问题存在唯一解。记离散问题的解为 $(bold(cal(M))_M, u_M)$，则有
  $
    norm(bold(cal(M))_M)_(bold(H)(div div))
    + norm(u_M)_(H^2(Omega))
    <= sqrt(2)/c_"plate" norm(f)_(L^2(Omega)),
  $
  其中常数 $c_"plate"$ 来源于 @thm:plate-continuous-stability。
]

#proof[
  由于 $bold(Sigma)_M subset bold(Sigma)$ 且 $U_M subset U$，上一节的连续强制性可直接限制到任意保形离散子空间上，并且强制性常数不随 $M$ 改变。也就是说，对任意 $(bold(tau)_M, v_M) in bold(Sigma)_M times U_M$，都有
  $
    c_"plate" (
      norm(bold(tau)_M)_(bold(H)(div div))^2
      + norm(v_M)_(H^2(Omega))^2
    )
    <= a_"plate" ((bold(tau)_M, v_M), (bold(tau)_M, v_M)).
  $
  同时由 $a_"plate"$ 的连续性，$a_"plate"$ 在离散子空间上也是连续的，且连续性常数不依赖于离散维数。

  右端泛函 $ell_"plate"$ 满足
  $
    abs(ell_"plate" (bold(tau)_M, v_M)) & = abs((f, nabla dot (nabla dot bold(tau)_M))_(L^2(Omega))) \
                                        & <= norm(f)_(L^2(Omega)) norm(bold(tau)_M)_(bold(H)(div div)) \
                                        & <= norm(f)_(L^2(Omega)) (
                                            norm(bold(tau)_M)_(bold(H)(div div))^2
                                            + norm(v_M)_(H^2(Omega))^2
                                          )^(1/2),
  $
  因此 $ell_"plate"$ 在离散空间上一致连续。由 Lax-Milgram 定理（参考 @BrennerScott2008 定理 2.7.7），离散解存在唯一。

  取测试函数 $(bold(tau)_M, v_M) = (bold(cal(M))_M, u_M)$，得到
  $
    a_"plate" ((bold(cal(M))_M, u_M), (bold(cal(M))_M, u_M))
    = ell_"plate" (bold(cal(M))_M, u_M)
    <= norm(f)_(L^2(Omega)) norm(bold(cal(M))_M)_(bold(H)(div div)).
  $
  记
  $
    C_M
    := (
      norm(bold(cal(M))_M)_(bold(H)(div div))^2
      + norm(u_M)_(H^2(Omega))^2
    )^(1/2).
  $
  由强制性与上式可得
  $
    c_"plate" C_M^2
    <= norm(f)_(L^2(Omega)) norm(bold(cal(M))_M)_(bold(H)(div div))
    <= norm(f)_(L^2(Omega)) C_M.
  $
  若 $C_M = 0$，稳定性估计显然成立；若 $C_M > 0$，两端同除以 $C_M$，得到
  $
    C_M <= c_"plate"^(-1) norm(f)_(L^2(Omega)).
  $
  最后利用
  $
    norm(bold(cal(M))_M)_(bold(H)(div div))
    + norm(u_M)_(H^2(Omega))
    <= sqrt(2) C_M,
  $
  即得
  $
    norm(bold(cal(M))_M)_(bold(H)(div div))
    + norm(u_M)_(H^2(Omega))
    <= sqrt(2)/c_"plate" norm(f)_(L^2(Omega)),
  $
  证毕。
]

#theorem(title: [板弯曲的准最优误差估计])[
  设连续解 $(bold(cal(M)), u) in bold(Sigma) times U$ 存在，离散解 $(bold(cal(M))_M, u_M) in bold(Sigma)_M times U_M$ 由上式给出。则有
  $
       & norm(bold(cal(M)) - bold(cal(M))_M)_(bold(H)(div div))
         + norm(u - u_M)_(H^2(Omega)) \
    <= & (sqrt(2) C_"plate") / c_"plate"
         inf_((bold(tau)_M, v_M) in bold(Sigma)_M times U_M)
         (
           norm(bold(cal(M)) - bold(tau)_M)_(bold(H)(div div))
           + norm(u - v_M)_(H^2(Omega))
         ),
  $
  其中常数 $c_"plate"$ 和 $C_"plate"$ 来源于 @thm:plate-continuous-stability。
]

#proof[
  连续解满足连续变分问题，离散解满足离散变分问题。由于 $bold(Sigma)_M subset bold(Sigma)$ 且 $U_M subset U$，任意离散测试函数也可作为连续测试函数。因此二者相减可得 Galerkin 正交性：
  $
    a_"plate" (
      (bold(cal(M)) - bold(cal(M))_M, u - u_M),
      (bold(tau)_M, v_M)
    )
    = 0,
    quad forall (bold(tau)_M, v_M) in bold(Sigma)_M times U_M.
  $

  记误差
  $
    (bold(e)_M, e_u)
    := (bold(cal(M)) - bold(cal(M))_M, u - u_M).
  $
  对任意 $(bold(tau)_M, v_M) in bold(Sigma)_M times U_M$，有
  $
    (bold(e)_M, e_u)
    = (bold(cal(M)) - bold(tau)_M, u - v_M)
    + (bold(tau)_M - bold(cal(M))_M, v_M - u_M),
  $
  其中第二项属于 $bold(Sigma)_M times U_M$。由连续强制性与 Galerkin 正交性，
  $
       & c_"plate" (
           norm(bold(e)_M)_(bold(H)(div div))^2
           + norm(e_u)_(H^2(Omega))^2
         ) \
    <= & a_"plate" ((bold(e)_M, e_u), (bold(e)_M, e_u)) \
     = & a_"plate" (
           (bold(e)_M, e_u),
           (bold(cal(M)) - bold(tau)_M, u - v_M)
         ).
  $
  由 @cor:a-plate-continuity，
  $
       & a_"plate" (
           (bold(e)_M, e_u),
           (bold(cal(M)) - bold(tau)_M, u - v_M)
         ) \
    <= & C_"plate" (
           norm(bold(e)_M)_(bold(H)(div div))^2
           + norm(e_u)_(H^2(Omega))^2
         )^(1/2) \
       & quad times (
           norm(bold(cal(M)) - bold(tau)_M)_(bold(H)(div div))^2
           + norm(u - v_M)_(H^2(Omega))^2
         )^(1/2).
  $
  若误差范数为零，则结论显然成立；否则两端同除以
  $
    (
      norm(bold(e)_M)_(bold(H)(div div))^2
      + norm(e_u)_(H^2(Omega))^2
    )^(1/2),
  $
  得到
  $
    (
      norm(bold(e)_M)_(bold(H)(div div))^2
      + norm(e_u)_(H^2(Omega))^2
    )^(1/2)
    <= C_"plate" / c_"plate"
    (
      norm(bold(cal(M)) - bold(tau)_M)_(bold(H)(div div))^2
      + norm(u - v_M)_(H^2(Omega))^2
    )^(1/2).
  $
  再利用
  $
    norm(bold(e)_M)_(bold(H)(div div))
    + norm(e_u)_(H^2(Omega))
    <= sqrt(2) (
      norm(bold(e)_M)_(bold(H)(div div))^2
      + norm(e_u)_(H^2(Omega))^2
    )^(1/2)
  $
  以及
  $
    (
      norm(bold(cal(M)) - bold(tau)_M)_(bold(H)(div div))^2
      + norm(u - v_M)_(H^2(Omega))^2
    )^(1/2)
    <= norm(bold(cal(M)) - bold(tau)_M)_(bold(H)(div div))
    + norm(u - v_M)_(H^2(Omega)),
  $
  并对所有 $(bold(tau)_M, v_M) in bold(Sigma)_M times U_M$ 取下确界，即得所述估计。证毕。
]

上述三条结论表明，板弯曲与二阶线弹性虽然不共享同一个自然能量空间，但二者的最小二乘结构完全平行：都通过“残差平方和控制自然范数”的方式获得连续与离散稳定性，都不需要弱混合方法中的 inf-sup 匹配条件。差别只在于板弯曲要求 $H_0^2$ 保形，因此离散空间必须能够同时强制施加 $u = 0$ 与 $partial_n u = 0$。

= 离散空间逼近

== 随机特征空间

记神经元特征为
$
  xi_m (x) = rho(gamma (bold(a)_m^T x + r_m)), quad m = 1, dots, M,
$
并记常数特征 $xi_0 = 1$。这里 $gamma > 0$ 为全局形状参数，随机方向与随机截距按 @zhang2024transnet 的重参数化方式生成：
$
  bold(w)_m = gamma bold(a)_m, quad
  b_m = gamma r_m,
$
其中 $bold(a)_m$ 表示超平面法向量，$r_m$ 表示截距。采样方式如下：
$
  bold(a)_m = bold(X)_m / norm(bold(X)_m)_2, quad
  r_m = U_m.
$
这里 $bold(X)_m$ 服从标准正态分布，$U_m$ 服从有界区间上的均匀分布。于是可定义神经特征空间如下
$
  Xi_M := span { xi_m : 0 <= m <= M },
$

== 三维线弹性

将上述标量神经特征张量化为应力与位移基函数后，即可构造三维线弹性的离散空间。在立方体区域 $Omega = [0, 1]^3$ 上，考虑如下问题：
$
  cases(
    bold(cal(A)) : bold(sigma) - bold(epsilon)(bold(u)) & = 0 & quad "in" Omega,
    nabla dot bold(sigma) + bold(f) & = 0 & quad "in" Omega,
    bold(u) & = 0 & quad "on" partial Omega.
  )
$
这里 $bold(cal(A))$ 是标准三维各向同性柔度张量，其作用可写为
$
  bold(cal(A)) : bold(tau)
  = 1/(2 mu) (
    bold(tau) - lambda/(2 mu + 3 lambda) tr(bold(tau)) bold(I)
  ).
$
其中
$
  lambda = (E nu)/((1 + nu) (1 - 2 nu)),
  quad
  mu = E/(2(1 + nu)).
$
取函数空间 $bold(Sigma) = bold(H)(div, Omega; SS^3)$，$bold(U) = (H_0^1(Omega))^3$，对应的离散空间取为
$
  bold(Sigma)_M & := span { xi_m^(bold(sigma)) bold(E)_alpha : 0 <= m <= M, 1 <= alpha <= 6 }, \
      bold(U)_M & := span { zeta xi_m^(bold(u)) bold(e)_i : 0 <= m <= M, 1 <= i <= 3 }.
$
其中 ${bold(E)_alpha}_1^6 subset RR^(3 times 3)$ 为 $SS^3$ 中的对称基矩阵，采用 Voigt 顺序 $(11, 22, 33, 12, 23, 13)$。${bold(e)_i}_1^3$ 为 $RR^3$ 的基向量。

取边界包络函数
$
  zeta(x, y, z) := x (1 - x) y (1 - y) z (1 - z),
$
从而 $zeta = 0$ 于 $partial Omega$，位移基函数自动属于 $(H_0^1(Omega))^3$。位移空间自动保形，因此连续稳定性、离散一致稳定性与准最优误差估计都可直接由二阶理论得到。

将离散未知量展开为
$
  bold(Phi)^(bold(sigma)) & = sum_(m=0)^M sum_(alpha=1)^6
                            phi^(bold(sigma))_(m, alpha) xi_m^(bold(sigma)) bold(E)_alpha, \
      bold(Phi)^(bold(u)) & = sum_(m=0)^M sum_(i=1)^3
                            phi^(bold(u))_(m, i) zeta xi_m^(bold(u)) bold(e)_i.
$

离散最小二乘问题等价于：求
$
  (bold(Phi)^(bold(sigma)), bold(Phi)^(bold(u)))
  in bold(Sigma)_M times bold(U)_M,
$
使得
$
  a ((bold(Phi)^(bold(sigma)), bold(Phi)^(bold(u))),
    (bold(Phi)^(bold(tau)), bold(Phi)^(bold(v))))
  = ell (bold(Phi)^(bold(tau)), bold(Phi)^(bold(v))),
  quad forall
  (bold(Phi)^(bold(tau)), bold(Phi)^(bold(v))) in bold(Sigma)_M times bold(U)_M.
$

具体的基函数展开与对称正定线性系统见 @app:3d。

== 平面应力

二维平面应力沿用同一构造，只是应力张量退化为 $SS^2$ 上的三个独立分量。设 $Omega subset RR^2$ 为板中面区域，满足平面应力假设
$
  sigma_(13) = sigma_(23) = sigma_(33) = 0.
$
此时只保留面内位移 $bold(u) = (u_1, u_2)^T : Omega -> RR^2$ 与二维对称应力张量 $bold(sigma): Omega -> SS^2$。二维应变仍记为
$
  bold(epsilon)(bold(u)) = 1/2 (nabla bold(u) + (nabla bold(u))^T).
$
对各向同性材料，平面应力本构写为
$
  sigma_(alpha beta)
  = 2 mu epsilon_(alpha beta)(bold(u))
  + lambda epsilon_(gamma gamma)(bold(u)) delta_(alpha beta),
  quad alpha, beta, gamma = 1, 2,
$
其中
$
  mu = E / (2 (1 + nu)),
  quad
  lambda = (E nu) / (1 - nu^2).
$
对应柔度算子的作用为
$
  bold(cal(A)) : bold(tau)
  = 1/(2 mu) (
    bold(tau) - lambda/(2 mu + 2 lambda) tr(bold(tau)) bold(I)
  ).
$
因此平面应力与三维线弹性的强形式完全同型：
$
  cases(
    bold(cal(A)) : bold(sigma) - bold(epsilon)(bold(u)) & = 0 & quad "in" Omega,
    nabla dot bold(sigma) + bold(f) & = 0 & quad "in" Omega,
    bold(u) & = 0 & quad "on" partial Omega.
  )
$
差别只在于这里的 $bold(cal(A))$ 作用在 $SS^2$ 上，且应力只有三个独立分量。

在随机特征离散层面，取
$
  bold(Sigma)_M
  := span { xi_m^(bold(sigma)) bold(E)_alpha : 0 <= m <= M, 1 <= alpha <= 3 },
$
以及
$
  bold(U)_M
  := span { zeta xi_m^(bold(u)) bold(e)_i : 0 <= m <= M, 1 <= i <= 2 }.
$
这里 ${bold(E)_alpha}_1^3$ 采用 Voigt 顺序 $(11, 22, 12)$。

取边界包络函数

$
  zeta(x, y) := x (1 - x) y (1 - y),
$
从而位移基函数属于 $(H_0^1(Omega))^2$。故位移离散空间属于 $(H_0^1(Omega))^2$。因此平面应力的稳定性结论与三维线弹性完全一致。

离散未知量的展开为
$
  bold(Phi)^(bold(sigma)) & = sum_(m=0)^M sum_(alpha=1)^3
                            phi^(bold(sigma))_(m, alpha) xi_m^(bold(sigma)) bold(E)_alpha, \
      bold(Phi)^(bold(u)) & = sum_(m=0)^M sum_(i=1)^2
                            phi^(bold(u))_(m, i) zeta xi_m^(bold(u)) bold(e)_i.
$

离散最小二乘问题等价于：求
$
  (bold(Phi)^(bold(sigma)), bold(Phi)^(bold(u)))
  in bold(Sigma)_M times bold(U)_M,
$
使得
$
  a ((bold(Phi)^(bold(sigma)), bold(Phi)^(bold(u))),
    (bold(Phi)^(bold(tau)), bold(Phi)^(bold(v))))
  = ell (bold(Phi)^(bold(tau)), bold(Phi)^(bold(v))),
  quad forall
  (bold(Phi)^(bold(tau)), bold(Phi)^(bold(v))) in bold(Sigma)_M times bold(U)_M.
$

具体的基函数展开与对称正定线性系统见 @app:plane。

== 板弯曲

对四阶 Kirchhoff-Love 模型，挠度离散空间必须提升到 $H_0^2(Omega)$ 保形。设 $Omega subset RR^2$ 为板中面区域，未知量为弯矩张量 $bold(cal(M)): Omega -> SS^2$ 与标量挠度 $u: Omega -> RR$。考虑如下方程组
$
  cases(
    bold(cal(A)) : bold(cal(M)) - bold(cal(K))(u) & = 0 & quad "in" Omega,
    nabla dot (nabla dot bold(cal(M))) + f & = 0 & quad "in" Omega,
    u & = 0 & quad "on" partial Omega,
    partial_n u & = 0 & quad "on" partial Omega.
  )
$
这里 $bold(cal(A))$ 为弯矩柔度张量；对各向同性薄板有
$
  bold(cal(A)) : bold(tau)
  = 1/(D(1-nu)) bold(tau)
  - nu/(D(1-nu)(1+nu)) tr(bold(tau)) bold(I),
$
其中 $D = (E h^3)/(12(1 - nu^2))$ 为弯曲刚度，而 $bold(cal(K))(u) := - nabla^2 u$ 为曲率张量。因此板弯曲对应的连续空间为
$
  bold(Sigma) := bold(H)(div div, Omega; SS^2),
  quad
  U := H_0^2(Omega).
$
在随机特征离散层面，取 ${bold(E)_alpha}_1^3$ 为 $SS^2$ 的标准对称基，按 Voigt 顺序 $(11, 22, 12)$ 排列，并定义
$
  bold(Sigma)_M & := span { xi_m^(bold(cal(M))) bold(E)_alpha : 0 <= m <= M, 1 <= alpha <= 3 }, \
            U_M & := span { zeta xi_m^(u) : 0 <= m <= M }.
$
取边界包络函数
$
  zeta(x, y) := [x (1 - x) y (1 - y)]^2,
$
由于 $zeta = 0$ 且 $partial_n zeta = 0$ 于 $partial Omega$，故相应的标量基函数属于 $H_0^2(Omega)$，可以自动满足固支边界条件。再结合特征函数的光滑性，可得 $bold(Sigma)_M subset bold(H)(div div, Omega; SS^2)$，从而该离散空间满足板弯曲最小二乘理论所要求的保形性。

将离散未知量展开为
$
  bold(Phi)^(bold(cal(M))) & = sum_(m=0)^M sum_(alpha=1)^3
                             phi^(bold(cal(M)))_(m, alpha) xi_m^(bold(cal(M))) bold(E)_alpha, \
                   Phi^(u) & = sum_(m=0)^M phi_m^(u) zeta xi_m^(u).
$

离散最小二乘问题等价于：求
$
  (bold(Phi)^(bold(cal(M))), Phi^(u))
  in bold(Sigma)_M times U_M,
$
使得
$
  a ((bold(Phi)^(bold(cal(M))), Phi^(u)),
    (bold(Phi)^(bold(tau)), Phi^(v)))
  = ell (bold(Phi)^(bold(tau)), Phi^(v)),
  quad forall
  (bold(Phi)^(bold(tau)), Phi^(v)) in bold(Sigma)_M times U_M.
$

具体的离散展开与对称正定线性系统见附录 @app:plate。

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
    | 数据类型 | `torch.float64` |
    | 激活函数 | $tanh$ |
    | 采样方式 | Sobol 采样 |
    | 采样种子 | 训练为 $43$，测试为 $45$ |
    | 特征种子 | 应力/弯矩特征为 $42$，位移/挠度特征为 $1042$ |
  ],
  caption: [三类算例的公共实验设置],
)

求解器采用 Lstsq、TSVD 和 Ridge，顺序如下：
- Lstsq 直接调用通用最小二乘求解器。
- TSVD：对对称线性系统采用特征值分解，并按相对截断阈值 $tau_"TSVD" = 10^(-15)$ 截断近零特征值。
- Ridge：在 Gram 矩阵上加入相对正则强度为 $alpha_"Ridge" = 10^(-15)$ 的对角正则项，以提升病态情形下的稳定性。
三种求解器的具体原理见附录 @app:solver。

== 三维线弹性

三维线弹性实验参数设置如下：

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
    | 截距        | $U_m tilde.op cal(U)(0, 1)$ |
    | 训练采样点   | $Q_"train" = (2^5)^3 = 32768$ |
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
  := sqrt(frac(1, Q_"test") sum_(p=1)^(Q_"test") sum_(i = 1)^3 [(bold(Phi)^bold(u) (bold(x)_p))_i - (bold(u)_"ex" (bold(x)_p))_i]^2),
$
而在 Voigt 顺序 $(11, 22, 33, 12, 23, 13)$ 下，对应力采用权重 $bold(w)^"V" = (1, 1, 1, 2, 2, 2)^T$，并定义
$
  norm(bold(Phi)^bold(sigma) - bold(sigma)_"ex")_0
  := sqrt(frac(1, Q_"test") sum_(p=1)^(Q_"test") sum_(alpha=1)^6 w^"V"_alpha [(bold(Phi)^(bold(sigma)) (bold(x)_p))_alpha - (bold(sigma)_"ex" (bold(x)_p))_alpha]^2).
$
散度误差定义为
$
  norm(div(bold(Phi)^bold(sigma) - bold(sigma)_"ex"))_0
  := sqrt(frac(1, Q_"test") sum_(p=1)^(Q_"test") sum_(alpha = 1)^6 w^"V"_alpha [(nabla dot (bold(Phi)^(bold(sigma)) - bold(sigma)_"ex") (bold(x)_p))_alpha]^2).
$
由于精确解满足 $nabla dot bold(sigma)_"ex" + bold(f)_"ex" = 0$，故计算时可等价写为
$
  norm(div(bold(Phi)^bold(sigma) - bold(sigma)_"ex"))_0
  = sqrt(frac(1, Q_"test") sum_(p=1)^(Q_"test") sum_(alpha = 1)^6 w^"V"_alpha [(nabla dot bold(Phi)^(bold(sigma)) (bold(x)_p) + bold(f)_"ex" (bold(x)_p))_alpha]^2).
$

=== 实验结果

实验结果见 @tb:3d-main。

#figure(
  three-line-table(
    columns: 5,
    align: (left, center, center, center, center),
  )[
    | 方法 | $norm(bold(Phi)^bold(u) - bold(u)_"ex")_0$ | $norm(bold(Phi)^bold(sigma) - bold(sigma)_"ex")_0$ | $norm(div(bold(Phi)^bold(sigma) - bold(sigma)_"ex"))_0$ | DOF |
    |:-----------|:----------:|:--------------:|:------------------:|:-------:|
    | Hu--Zhang (1) |   6.13e-02 |   1.99e-01 |       1.47e-00      |  1215   |
    | Hu--Zhang (2) |   7.15e-03 |   8.05e-03 |       9.20e-02      |  8472   |
    | Hu--Zhang (3) |   4.91e-04 |   2.91e-04 |       5.75e-03      |  63666  |
    | LS (Lstsq) | 5.86e-05 | 3.71e-03 |   7.67e-03   |  2709   |
    | LS (TSVD)  | 5.78e-05 | 3.70e-03 |   7.67e-03   |  2709   |
    | LS (Ridge) | 5.65e-05 | 3.69e-03 |   7.67e-03   |  2709   |
  ],
  caption: [三维线弹性主实验结果（$M = 300$）],
)<tb:3d-main>

// #figure(
//   image("/public/images/least-squares/linear-elasticity-3d/l2-error-summary.png"),
//   caption: [三维线弹性主实验结果（$M = 300$）],
// )


=== 特征数消融

特征数消融结果见 @tb:3d-ablation。

#figure(
  three-line-table(
    columns: 6,
    align: (left, center, center, center, center, center),
  )[
    | 方法 | $M$ | $norm(bold(Phi)^bold(u) - bold(u)_"ex")_0$ | $norm(bold(Phi)^bold(sigma) - bold(sigma)_"ex")_0$ | $norm(div(bold(Phi)^bold(sigma) - bold(sigma)_"ex"))_0$ | Time(s) |
    |:-----------|:----:|:---------:|:---------:|:--------------:|:-------:|
    | LS (Lstsq) | 200  | 4.06e-04 | 2.27e-02 |   3.86e-02   |  0.18   |
    | LS (Lstsq) | 400  | 2.15e-05 | 1.47e-03 |   2.50e-03   |  0.44   |
    | LS (Lstsq) | 600  | 3.97e-06 | 3.12e-04 |   6.11e-04   |  1.33   |
    | LS (Lstsq) | 800  | 6.78e-05 | 3.14e-03 |   1.47e-03   |  2.92   |
    | LS (Lstsq) | 1000 | 3.00e-05 | 1.43e-03 |   9.49e-04   |  5.50   |
    | LS (TSVD)  | 200  | 4.05e-04 | 2.27e-02 |   3.86e-02   |  0.31   |
    | LS (TSVD)  | 400  | 2.23e-05 | 1.51e-03 |   2.51e-03   |  1.45   |
    | LS (TSVD)  | 600  | 7.18e-06 | 4.65e-04 |   7.40e-04   |  4.03   |
    | LS (TSVD)  | 800  | 3.78e-06 | 2.41e-04 |   3.09e-04   |  8.73   |
    | LS (TSVD)  | 1000 | 3.00e-06 | 1.70e-04 |   1.99e-04   |  16.44  |
    | LS (Ridge) | 200  | 4.03e-04 | 2.27e-02 |   3.86e-02   |  0.21   |
    | LS (Ridge) | 400  | 2.05e-05 | 1.48e-03 |   2.51e-03   |  1.31   |
    | LS (Ridge) | 600  | 4.75e-06 | 3.96e-04 |   7.11e-04   |  3.53   |
    | LS (Ridge) | 800  | 2.12e-06 | 1.94e-04 |   2.89e-04   |  7.46   |
    | LS (Ridge) | 1000 | 1.47e-06 | 1.40e-04 |   1.74e-04   |  13.76  |
  ],
  caption: [三维线弹性特征数 $M$ 的消融实验],
)<tb:3d-ablation>

#figure(
  image("/public/images/least-squares/linear-elasticity-3d/ablation/M/ablation-M.png"),
  caption: [三维线弹性特征数 $M$ 的消融实验],
)


== 平面应力

平面应力实验参数设置如下：

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
    | 截距        | $U_m tilde.op cal(U)(0, 1)$ |
    | 内部采样点   | $Q_"train" = (2^8)^2 = 65536$ |
    | 测试采样点   | $Q_"test" = (2^7)^2 = 16384$ |
  ],
  caption: [平面应力实验设置],
)

参考 @hu2014triangle 中的二维制造解，取面内位移场
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
与文献中的二维线弹性系数一致。同样地，该制造解在边界上满足 $bold(u)_"ex" = 0$。精确应力 $bold(sigma)_"ex"$ 由平面应力本构计算，体力仍由
$
  bold(f)_"ex" = - nabla dot bold(sigma)(bold(u)_"ex")
$
自动生成，因此可直接用来评估位移与应力误差。

对平面应力，位移误差定义为
$
  norm(bold(Phi)^bold(u) - bold(u)_"ex")_0
  := sqrt(frac(1, Q_"test") sum_(p=1)^(Q_"test") sum_(i = 1)^2 [(bold(Phi)^bold(u) (bold(x)_p))_i - (bold(u)_"ex" (bold(x)_p))_i]^2),
$
而在 Voigt 顺序 $(11, 22, 12)$ 下，对应力采用权重 $bold(w)^"V" = (1, 1, 2)^T$，并定义
$
  norm(bold(Phi)^bold(sigma) - bold(sigma)_"ex")_0
  := sqrt(frac(1, Q_"test") sum_(p=1)^(Q_"test") sum_(alpha=1)^3 w^"V"_alpha [(bold(Phi)^(bold(sigma)) (bold(x)_p))_alpha - (bold(sigma)_"ex" (bold(x)_p))_alpha]^2).
$
散度误差定义为
$
  norm(div(bold(Phi)^bold(sigma) - bold(sigma)_"ex"))_0
  := sqrt(frac(1, Q_"test") sum_(p=1)^(Q_"test") sum_(alpha = 1)^3 w^"V"_alpha [(nabla dot (bold(Phi)^(bold(sigma)) - bold(sigma)_"ex") (bold(x)_p))_alpha]^2).
$
由于精确解满足 $nabla dot bold(sigma)_"ex" + bold(f)_"ex" = 0$，故计算时可等价写为
$
  norm(div(bold(Phi)^bold(sigma) - bold(sigma)_"ex"))_0
  = sqrt(frac(1, Q_"test") sum_(p=1)^(Q_"test") sum_(alpha = 1)^3 w^"V"_alpha [(nabla dot bold(Phi)^(bold(sigma)) (bold(x)_p) + bold(f)_"ex" (bold(x)_p))_alpha]^2).
$

=== 实验结果

实验结果见 @tb:ps-main。

#figure(
  three-line-table(
    columns: 5,
    align: (left, center, center, center, center),
  )[
    | 方法 | $norm(bold(Phi)^bold(u) - bold(u)_"ex")_0$ | $norm(bold(Phi)^bold(sigma) - bold(sigma)_"ex")_0$ | $norm(div(bold(Phi)^bold(sigma) - bold(sigma)_"ex"))_0$ | DOF |
    |:-----------|:----------:|:--------------:|:------------------:|:-------:|
    | Hu--Zhang (1) |   2.46e-03 |   1.07e-02 |       1.32e-01      |   971   |
    | Hu--Zhang (2) |   2.85e-04 |   6.98e-04 |       1.68e-02      |  3763   |
    | Hu--Zhang (3) |   3.50e-05 |   4.42e-05 |       2.12e-03      | 14819   |
    | LS (Lstsq) | 2.72e-06 | 2.53e-04 |   1.87e-04   |  1505   |
    | LS (TSVD)  | 1.84e-07 | 1.48e-05 |   2.77e-05   |  1505   |
    | LS (Ridge) | 8.59e-08 | 1.02e-05 |   2.56e-05   |  1505   |
  ],
  caption: [平面应力主实验结果（$M = 300$）],
)<tb:ps-main>

// #figure(
//   image("/public/images/least-squares/plane-stress/l2-error-summary.png"),
//   caption: [平面应力主实验结果（$M = 300$）],
// )

=== 特征数消融

特征数消融结果见 @tb:ps-ablation。

#figure(
  three-line-table(
    columns: 6,
    align: (left, center, center, center, center, center),
  )[
    | 方法 | $M$ | $norm(bold(Phi)^bold(u) - bold(u)_"ex")_0$ | $norm(bold(Phi)^bold(sigma) - bold(sigma)_"ex")_0$ | $norm(div(bold(Phi)^bold(sigma) - bold(sigma)_"ex"))_0$ | Time(s) |
    |:-----------|:----:|:---------:|:---------:|:--------------:|:-------:|
    | LS (Lstsq) | 200  | 6.21e-07 | 6.54e-05 |   2.07e-04   |  0.08   |
    | LS (Lstsq) | 400  | 9.65e-07 | 7.45e-05 |   6.32e-05   |  0.10   |
    | LS (Lstsq) | 600  | 3.02e-07 | 2.10e-05 |   2.32e-05   |  0.27   |
    | LS (Lstsq) | 800  | 1.96e-07 | 1.06e-05 |   1.19e-05   |  0.58   |
    | LS (Lstsq) | 1000 | 1.69e-07 | 1.05e-05 |   1.07e-05   |  1.07   |
    | LS (TSVD)  | 200  | 1.13e-06 | 9.81e-05 |   2.09e-04   |  0.14   |
    | LS (TSVD)  | 400  | 9.33e-08 | 7.28e-06 |   1.07e-05   |  0.26   |
    | LS (TSVD)  | 600  | 5.73e-08 | 3.75e-06 |   5.38e-06   |  0.80   |
    | LS (TSVD)  | 800  | 5.87e-08 | 3.41e-06 |   4.37e-06   |  1.89   |
    | LS (TSVD)  | 1000 | 4.32e-08 | 2.71e-06 |   3.65e-06   |  3.23   |
    | LS (Ridge) | 200  | 7.36e-07 | 7.54e-05 |   2.12e-04   |  0.06   |
    | LS (Ridge) | 400  | 4.09e-08 | 5.01e-06 |   8.78e-06   |  0.24   |
    | LS (Ridge) | 600  | 2.42e-08 | 2.86e-06 |   4.74e-06   |  0.74   |
    | LS (Ridge) | 800  | 2.13e-08 | 2.45e-06 |   3.85e-06   |  1.68   |
    | LS (Ridge) | 1000 | 1.87e-08 | 2.16e-06 |   3.14e-06   |  2.86   |
  ],
  caption: [平面应力特征数消融实验],
)<tb:ps-ablation>

#figure(
  image("/public/images/least-squares/plane-stress/ablation/M/ablation-M.png"),
  caption: [平面应力特征数消融实验],
)

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
    | 形状参数     | $gamma = 3.0$ |
    | 超平面法向量  | $bold(X) tilde.op cal(N)(0, bold(I)_2)$ |
    | 截距        | $U_m tilde.op cal(U)(0, 1)$ |
    | 内部采样点   | $Q_"train" = (2^8)^2 = 65536$ |
    | 测试采样点   | $Q_"test" = (2^7)^2 = 16384$ |
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
    | 方法 | $e_u$ | $e_bold(cal(M))$ | Time(s) |
    |:-------------------|-----------:|-----------:|---------:|
    | LS (Lstsq)         |   8.04e-04 |   1.09e-01 |     0.08 |
    | LS (TSVD)          |   8.09e-06 |   7.19e-04 |     0.13 |
    | LS (Ridge)         |   4.79e-06 |   5.69e-04 |     0.08 |
  ],
  caption: [板弯曲主实验结果（$M = 300$）],
)<tb:pb-main>

// #figure(
//   image("/public/images/least-squares/plate-bending/l2-error-summary.png"),
//   caption: [板弯曲主实验结果（$M = 300$）],
// )

=== 特征数消融

特征数消融结果见 @tb:pb-ablation。

#figure(
  three-line-table(
    columns: 5,
    align: (left, right, right, right, right),
  )[
    | 方法 | $M$ | $e_u$ | $e_bold(cal(M))$ |  Time(s) |
    |:-------------------|-------:|-----------:|-----------:|---------:|
    | LS (Lstsq)         |    200 |   8.29e-05 |   1.20e-02 |     0.07 |
    | LS (Lstsq)         |    400 |   2.86e-06 |   3.45e-04 |     0.06 |
    | LS (Lstsq)         |    600 |   2.10e-06 |   2.19e-04 |     0.15 |
    | LS (Lstsq)         |    800 |   1.89e-06 |   1.46e-04 |     0.33 |
    | LS (Lstsq)         |   1000 |   9.83e-06 |   7.12e-04 |     0.58 |
    | LS (TSVD)          |    200 |   2.57e-05 |   2.00e-03 |     0.09 |
    | LS (TSVD)          |    400 |   2.41e-06 |   1.85e-04 |     0.15 |
    | LS (TSVD)          |    600 |   2.08e-06 |   1.80e-04 |     0.41 |
    | LS (TSVD)          |    800 |   2.15e-06 |   1.71e-04 |     0.99 |
    | LS (TSVD)          |   1000 |   2.69e-06 |   1.31e-04 |     1.88 |
    | LS (Ridge)         |    200 |   9.53e-06 |   1.57e-03 |     0.04 |
    | LS (Ridge)         |    400 |   9.61e-07 |   1.58e-04 |     0.14 |
    | LS (Ridge)         |    600 |   6.04e-07 |   1.34e-04 |     0.38 |
    | LS (Ridge)         |    800 |   5.50e-07 |   1.22e-04 |     0.90 |
    | LS (Ridge)         |   1000 |   4.38e-07 |   1.02e-04 |     1.69 |
  ],
  caption: [板弯曲特征数消融实验],
)<tb:pb-ablation>

#figure(
  image("/public/images/least-squares/plate-bending/ablation/M/ablation-M.png"),
  caption: [板弯曲特征数消融实验],
)

= 结语

本文提出了一类面向弹性问题的固定随机特征混合最小二乘离散方法。其基本思路是先根据混合模型中的本构关系与平衡关系构造最小二乘泛函，再由对应的变分形式建立离散系统；在随机特征层面，则通过保形空间与包络函数处理齐次边界条件，从而得到统一的对称正定代数结构。

理论结果表明，只要离散空间与相应连续模型的自然空间保持一致，连续稳定性、离散稳定性以及准最优误差估计便可为该方法提供一致的分析基础。数值实验进一步在线弹性、平面应力与 Kirchhoff-Love 板弯曲三个算例上验证了这一框架的可行性；同时，利用对称结构并进行谱截断的 TSVD 求解版本整体更稳健，而通用 Lstsq 对特征数更敏感。

#bibliography("/public/reference/least-squares.bib")

#set heading(numbering: "附录 A.1", supplement: [Appendix])
#counter(heading).update(0)

= 三维线弹性的具体离散推导 <app:3d>

== 问题描述

在 $Omega = [0, 1]^3$ 上，三维线弹性的方程组如下：
$
  cases(
    bold(cal(A)) : bold(sigma) - bold(epsilon)(bold(u)) & = 0 & quad "in" Omega,
    nabla dot bold(sigma) + bold(f) & = 0 & quad "in" Omega,
    bold(u) & = 0 & quad "on" partial Omega.
  )
$

正文已经证明，二阶线弹性的强形式与变分问题彼此等价。因此对于离散问题，实际考虑求解的不是上述强形式的逐点方程，而是变分问题：求 $(bold(sigma), bold(u)) in bold(Sigma) times bold(U)$，使得
$
  a ((bold(sigma), bold(u)), (bold(tau), bold(v)))
  = ell (bold(tau), bold(v)),
  quad forall (bold(tau), bold(v)) in bold(Sigma) times bold(U).
$

== 保形随机特征空间

令 ${bold(E)_alpha}_(alpha = 1)^6$ 为 $SS^3$ 的标准对称基，按 Voigt 顺序 $(11, 22, 33, 12, 23, 13)$ 排列；令 ${bold(e)_i}_(i = 1)^3$ 为 $RR^3$ 的标准基。取包络函数
$
  zeta(x, y, z) = x (1 - x) y (1 - y) z (1 - z).
$
于是三维离散空间定义为
$
  bold(Sigma)_M
  := span { xi_m^(bold(sigma)) bold(E)_alpha : 0 <= m <= M, 1 <= alpha <= 6 },
$
以及
$
  bold(U)_M
  := span { zeta xi_m^(bold(u)) bold(e)_i : 0 <= m <= M, 1 <= i <= 3 }.
$
将离散未知量展开为
$
  bold(Phi)^(bold(sigma))
  = sum_(m=0)^M sum_(alpha=1)^6
  phi^(bold(sigma))_(m, alpha) xi_m^(bold(sigma)) bold(E)_alpha,
$
以及
$
  bold(Phi)^(bold(u))
  = sum_(m=0)^M sum_(i=1)^3
  phi^(bold(u))_(m, i) zeta xi_m^(bold(u)) bold(e)_i.
$

== 对称正定线性系统

离散最小二乘问题等价于：求
$
  (bold(Phi)^(bold(sigma)), bold(Phi)^(bold(u)))
  in bold(Sigma)_M times bold(U)_M,
$
使得
$
  a ((bold(Phi)^(bold(sigma)), bold(Phi)^(bold(u))),
    (bold(Phi)^(bold(tau)), bold(Phi)^(bold(v))))
  = ell (bold(Phi)^(bold(tau)), bold(Phi)^(bold(v))),
  quad forall
  (bold(Phi)^(bold(tau)), bold(Phi)^(bold(v))) in bold(Sigma)_M times bold(U)_M.
$
由于测试空间由上述基函数张成，只需分别取每个基函数作为测试函数，即可得到全部代数方程。现取测试函数
$
  bold(Phi)^(bold(tau)) = xi_n^(bold(sigma)) bold(E)_beta,
  quad
  bold(Phi)^(bold(v)) = zeta xi_n^(bold(u)) bold(e)_j.
$
定义矩阵块
$
  bold(G)^(bold(sigma) bold(sigma))_((n, beta), (m, alpha))
  := & (bold(cal(A)) : (xi_n^(bold(sigma)) bold(E)_beta),
         bold(cal(A)) : (xi_m^(bold(sigma)) bold(E)_alpha))_(L^2(Omega)) \
     & + (nabla dot (xi_n^(bold(sigma)) bold(E)_beta),
         nabla dot (xi_m^(bold(sigma)) bold(E)_alpha))_(L^2(Omega)), \
  bold(G)^(bold(sigma) bold(u))_((n, beta), (m, i))
  := & - (bold(cal(A)) : (xi_n^(bold(sigma)) bold(E)_beta),
         bold(epsilon)(zeta xi_m^(bold(u)) bold(e)_i))_(L^2(Omega)), \
  bold(G)^(bold(u) bold(u))_((n, j), (m, i))
  := & (bold(epsilon)(zeta xi_n^(bold(u)) bold(e)_j),
         bold(epsilon)(zeta xi_m^(bold(u)) bold(e)_i))_(L^2(Omega)),
$
以及载荷向量
$
  bold(F)^(bold(sigma))_((n, beta))
  := - (bold(f), nabla dot (xi_n^(bold(sigma)) bold(E)_beta))_(L^2(Omega)).
$
将离散展开代入离散变分式，并分别以上述两类基函数作为测试函数，可得对任意 $0 <= n <= M$、$1 <= beta <= 6$ 与 $1 <= j <= 3$，
$
  sum_(m=0)^M sum_(alpha=1)^6
  bold(G)^(bold(sigma) bold(sigma))_((n, beta), (m, alpha))
  phi^(bold(sigma))_(m, alpha)
  + sum_(m=0)^M sum_(i=1)^3
  bold(G)^(bold(sigma) bold(u))_((n, beta), (m, i))
  phi^(bold(u))_(m, i)
  = bold(F)^(bold(sigma))_((n, beta)),
$
以及
$
  sum_(m=0)^M sum_(alpha=1)^6
  bold(G)^(bold(sigma) bold(u))_((m, alpha), (n, j))
  phi^(bold(sigma))_(m, alpha)
  + sum_(m=0)^M sum_(i=1)^3
  bold(G)^(bold(u) bold(u))_((n, j), (m, i))
  phi^(bold(u))_(m, i)
  = 0.
$
将这两组方程按未知系数排列，便得到
$
  mat(
    bold(G)^(bold(sigma) bold(sigma)), bold(G)^(bold(sigma) bold(u));
    (bold(G)^(bold(sigma) bold(u)))^T, bold(G)^(bold(u) bold(u))
  )
  mat(
    bold(phi)^(bold(sigma));
    bold(phi)^(bold(u))
  )
  =
  mat(
    bold(F)^(bold(sigma));
    0
  ).
$
其中交叉块互为转置来自双线性形式 $a$ 的对称性，而整体矩阵正定则来自正文已证的强制性在保形离散子空间上的继承。

= 平面应力的具体离散推导 <app:plane>

== 问题描述

设 $Omega subset RR^2$，并满足平面应力条件
$
  sigma_(13) = sigma_(23) = sigma_(33) = 0.
$
于是仅保留面内位移 $bold(u) = (u_1, u_2)^T$ 与二维对称应力张量 $bold(sigma)$。本构关系可写为
$
  sigma_(alpha beta)
  = 2 mu epsilon_(alpha beta)(bold(u))
  + lambda epsilon_(gamma gamma)(bold(u)) delta_(alpha beta),
  quad alpha, beta, gamma = 1, 2,
$
其中
$
  mu = E / (2 (1 + nu)),
  quad
  lambda = (E nu) / (1 - nu^2).
$
因此平面应力的方程如下：
$
  cases(
    bold(cal(A)) : bold(sigma) - bold(epsilon)(bold(u)) & = 0 & quad "in" Omega,
    nabla dot bold(sigma) + bold(f) & = 0 & quad "in" Omega,
    bold(u) & = 0 & quad "on" partial Omega,
  )
$

平面应力仍属于二阶线弹性框架，与上一节类似，后续考虑求解的不是上述强形式的逐点方程，而是变分问题：求 $(bold(sigma), bold(u)) in bold(Sigma) times bold(U)$，使得
$
  a ((bold(sigma), bold(u)), (bold(tau), bold(v)))
  = ell (bold(tau), bold(v)),
  quad forall (bold(tau), bold(v)) in bold(Sigma) times bold(U).
$

== 保形随机特征空间

取 ${bold(E)_alpha}_(alpha = 1)^3$ 为 $SS^2$ 的标准对称基，按 Voigt 顺序 $(11, 22, 12)$ 排列；令 ${bold(e)_i}_(i = 1)^2$ 为 $RR^2$ 的标准基。令
$
  zeta(x, y) = x (1 - x) y (1 - y).
$
于是离散空间写为
$
  bold(Sigma)_M & := span { xi_m^(bold(sigma)) bold(E)_alpha : 0 <= m <= M, 1 <= alpha <= 3 }, \
      bold(U)_M & := span { zeta xi_m^(bold(u)) bold(e)_i : 0 <= m <= M, 1 <= i <= 2 }.
$
离散未知量的展开为
$
  bold(Phi)^(bold(sigma))
  = sum_(m=0)^M sum_(alpha=1)^3
  phi^(bold(sigma))_(m, alpha) xi_m^(bold(sigma)) bold(E)_alpha,
$
以及
$
  bold(Phi)^(bold(u))
  = sum_(m=0)^M sum_(i=1)^2
  phi^(bold(u))_(m, i) zeta xi_m^(bold(u)) bold(e)_i.
$

== 对称正定线性系统

离散最小二乘问题等价于：求
$
  (bold(Phi)^(bold(sigma)), bold(Phi)^(bold(u)))
  in bold(Sigma)_M times bold(U)_M,
$
使得
$
  a ((bold(Phi)^(bold(sigma)), bold(Phi)^(bold(u))),
    (bold(Phi)^(bold(tau)), bold(Phi)^(bold(v))))
  = ell (bold(Phi)^(bold(tau)), bold(Phi)^(bold(v))),
  quad forall
  (bold(Phi)^(bold(tau)), bold(Phi)^(bold(v))) in bold(Sigma)_M times bold(U)_M.
$
由于测试空间由上述基函数张成，只需分别取每个基函数作为测试函数，即可得到全部代数方程。现取测试函数
$
  bold(Phi)^(bold(tau)) = xi_n^(bold(sigma)) bold(E)_beta,
  quad
  bold(Phi)^(bold(v)) = zeta xi_n^(bold(u)) bold(e)_j.
$
定义矩阵块
$
  bold(G)^(bold(sigma) bold(sigma))_((n, beta), (m, alpha))
  := & (bold(cal(A)) : (xi_n^(bold(sigma)) bold(E)_beta),
         bold(cal(A)) : (xi_m^(bold(sigma)) bold(E)_alpha))_(L^2(Omega)) \
     & + (nabla dot (xi_n^(bold(sigma)) bold(E)_beta),
         nabla dot (xi_m^(bold(sigma)) bold(E)_alpha))_(L^2(Omega)), \
  bold(G)^(bold(sigma) bold(u))_((n, beta), (m, i))
  := & - (bold(cal(A)) : (xi_n^(bold(sigma)) bold(E)_beta),
         bold(epsilon)(zeta xi_m^(bold(u)) bold(e)_i))_(L^2(Omega)), \
  bold(G)^(bold(u) bold(u))_((n, j), (m, i))
  := & (bold(epsilon)(zeta xi_n^(bold(u)) bold(e)_j),
         bold(epsilon)(zeta xi_m^(bold(u)) bold(e)_i))_(L^2(Omega)),
$
以及载荷向量
$
  bold(F)^(bold(sigma))_((n, beta))
  := - (bold(f), nabla dot (xi_n^(bold(sigma)) bold(E)_beta))_(L^2(Omega)).
$
将离散展开代入离散变分式，并分别以上述两类基函数作为测试函数，可得对任意 $0 <= n <= M$、$1 <= beta <= 3$ 与 $1 <= j <= 2$，
$
  sum_(m=0)^M sum_(alpha=1)^3
  bold(G)^(bold(sigma) bold(sigma))_((n, beta), (m, alpha))
  phi^(bold(sigma))_(m, alpha)
  + sum_(m=0)^M sum_(i=1)^2
  bold(G)^(bold(sigma) bold(u))_((n, beta), (m, i))
  phi^(bold(u))_(m, i)
  = bold(F)^(bold(sigma))_((n, beta)),
$
以及
$
  sum_(m=0)^M sum_(alpha=1)^3
  bold(G)^(bold(sigma) bold(u))_((m, alpha), (n, j))
  phi^(bold(sigma))_(m, alpha)
  + sum_(m=0)^M sum_(i=1)^2
  bold(G)^(bold(u) bold(u))_((n, j), (m, i))
  phi^(bold(u))_(m, i)
  = 0.
$
将这两组方程按未知系数排列，便得到
$
  mat(
    bold(G)^(bold(sigma) bold(sigma)), bold(G)^(bold(sigma) bold(u));
    (bold(G)^(bold(sigma) bold(u)))^T, bold(G)^(bold(u) bold(u))
  )
  mat(
    bold(phi)^(bold(sigma));
    bold(phi)^(bold(u))
  )
  =
  mat(
    bold(F)^(bold(sigma));
    0
  ).
$
其中交叉块互为转置来自双线性形式 $a$ 的对称性，而整体矩阵正定则来自正文已证的强制性在保形离散子空间上的继承。

= Kirchhoff-Love 板弯曲的具体离散推导 <app:plate>

== 问题描述

Kirchhoff-Love 假设给出中面挠度 $u: Omega -> RR$，以及曲率张量
$
  bold(cal(K))(u) = - nabla^2 u.
$
引入弯矩张量 $bold(cal(M)): Omega -> SS^2$ 后，混合强形式为
$
  cases(
    bold(cal(A)) : bold(cal(M)) - bold(cal(K))(u) & = 0 & quad "in" Omega,
    nabla dot (nabla dot bold(cal(M))) + f & = 0 & quad "in" Omega,
    u & = 0 & quad "on" partial Omega,
    partial_n u & = 0 & quad "on" partial Omega.
  )
$
这里仍取各向同性薄板的弯矩柔度算子
$
  bold(cal(A)) : bold(tau)
  = 1/(D(1-nu)) bold(tau)
  - nu/(D(1-nu)(1+nu)) tr(bold(tau)) bold(I),
$
其中 $D = (E h^3)/(12(1 - nu^2))$ 为弯曲刚度。

正文已经证明，Kirchhoff-Love 板弯曲的强形式与变分问题彼此等价。因此在本附录中，后续实际求解的不是上述强形式的逐点方程，而是变分问题：求 $(bold(cal(M)), u) in bold(Sigma) times U$，使得
$
  a_"plate" ((bold(cal(M)), u), (bold(tau), v))
  = ell_"plate" (bold(tau), v),
  quad forall (bold(tau), v) in bold(Sigma) times U.
$

== 保形随机特征空间

取 ${bold(E)_alpha}_(alpha = 1)^3$ 为 $SS^2$ 的标准对称基，按 Voigt 顺序 $(11, 22, 12)$ 排列。令
$
  zeta(x, y) = [x (1 - x) y (1 - y)]^2.
$
于是板弯曲的保形离散空间取为
$
  bold(Sigma)_M
  := span { xi_m^(bold(cal(M))) bold(E)_alpha : 0 <= m <= M, 1 <= alpha <= 3 },
$
以及
$
  U_M
  := span { zeta xi_m^(u) : 0 <= m <= M }.
$
将离散未知量展开为
$
  bold(Phi)^(bold(cal(M)))
  = sum_(m=0)^M sum_(alpha=1)^3
  phi^(bold(cal(M)))_(m, alpha) xi_m^(bold(cal(M))) bold(E)_alpha,
$
以及
$
  Phi^(u)
  = sum_(m=0)^M phi_m^(u) zeta xi_m^(u).
$

== 对称正定线性系统

离散最小二乘问题等价于：求
$
  (bold(Phi)^(bold(cal(M))), Phi^(u))
  in bold(Sigma)_M times U_M,
$
使得
$
  a ((bold(Phi)^(bold(cal(M))), Phi^(u)),
    (bold(Phi)^(bold(tau)), Phi^(v)))
  = ell (bold(Phi)^(bold(tau)), Phi^(v)),
  quad forall
  (bold(Phi)^(bold(tau)), Phi^(v)) in bold(Sigma)_M times U_M.
$
由于测试空间由上述基函数张成，只需分别取每个基函数作为测试函数，即可得到全部代数方程。现取测试函数
$
  bold(Phi)^(bold(tau)) = xi_n^(bold(cal(M))) bold(E)_beta,
  quad
  Phi^(v) = zeta xi_n^(u).
$
定义矩阵块
$
  bold(G)^(bold(cal(M)) bold(cal(M)))_((n, beta), (m, alpha))
  := & (bold(cal(A)) : (xi_n^(bold(cal(M))) bold(E)_beta),
         bold(cal(A)) : (xi_m^(bold(cal(M))) bold(E)_alpha))_(L^2(Omega)) \
     & + (nabla dot (nabla dot (xi_n^(bold(cal(M))) bold(E)_beta)),
         nabla dot (nabla dot (xi_m^(bold(cal(M))) bold(E)_alpha)))_(L^2(Omega)), \
  bold(G)^(bold(cal(M)) u)_((n, beta), m)
  := & - (bold(cal(A)) : (xi_n^(bold(cal(M))) bold(E)_beta),
         bold(cal(K))(zeta xi_m^(u)))_(L^2(Omega)), \
  bold(G)^(u u)_(n, m)
  := & (bold(cal(K))(zeta xi_n^(u)),
         bold(cal(K))(zeta xi_m^(u)))_(L^2(Omega)),
$
以及载荷向量
$
  bold(F)^(bold(cal(M)))_((n, beta))
  := - (f,
    nabla dot (nabla dot (xi_n^(bold(cal(M))) bold(E)_beta)))_(L^2(Omega)).
$
将离散展开代入离散变分式，并分别以上述两类基函数作为测试函数，可得对任意 $0 <= n <= M$、$1 <= beta <= 3$，
$
  sum_(m=0)^M sum_(alpha=1)^3
  bold(G)^(bold(cal(M)) bold(cal(M)))_((n, beta), (m, alpha))
  phi^(bold(cal(M)))_(m, alpha)
  + sum_(m=0)^M
  bold(G)^(bold(cal(M)) u)_((n, beta), m)
  phi_m^(u)
  = bold(F)^(bold(cal(M)))_((n, beta)),
$
以及
$
  sum_(m=0)^M sum_(alpha=1)^3
  bold(G)^(bold(cal(M)) u)_((m, alpha), n)
  phi^(bold(cal(M)))_(m, alpha)
  + sum_(m=0)^M
  bold(G)^(u u)_(n, m)
  phi_m^(u)
  = 0.
$
将这两组方程按未知系数排列，便得到
$
  mat(
    bold(G)^(bold(cal(M)) bold(cal(M))), bold(G)^(bold(cal(M)) u);
    (bold(G)^(bold(cal(M)) u))^T, bold(G)^(u u)
  )
  mat(
    bold(phi)^(bold(cal(M)));
    phi^(u)
  )
  =
  mat(
    bold(F)^(bold(cal(M)));
    0
  ).
$
其中交叉块互为转置来自双线性形式 $a$ 的对称性；又由于 $bold(Sigma)_M subset bold(H)(div div, Omega; SS^2)$、$U_M subset H_0^2(Omega)$ 保形，正文中的板弯曲连续稳定性与离散稳定性定理保证该系统是对称正定的。

= 三种线性求解器的原理 <app:solver>

三类算例在离散后最终都可写成同一类线性系统
$
  bold(G) bold(phi) = bold(F),
$
其中 $bold(G)$ 表示对称正定的离散 Gram 矩阵，$bold(phi)$ 表示拼接后的未知系数向量，$bold(F)$ 表示右端载荷向量。为统一描述，记
$
  bold(G) = bold(Q) bold(Lambda) bold(Q)^T,
  quad
  bold(Lambda) = diag(lambda_1, dots, lambda_N),
  quad
  lambda_"max" = max_(1 <= i <= N) abs(lambda_i),
$
其中 ${bold(q)_i}_(i=1)^N$ 为 $bold(G)$ 的标准正交特征向量组。

== Lstsq

Lstsq 直接把代数系统视为最小二乘问题，求解
$
  bold(phi)_("Lstsq")
  := arg min_(bold(psi) in RR^N)
  norm(bold(G) bold(psi) - bold(F))_2.
$
当 $bold(G)$ 非奇异时，该解与直接求解 $bold(G) bold(phi) = bold(F)$ 等价；当数值上接近奇异时，它返回相应的最小二乘解。

== TSVD

TSVD 利用 $bold(G)$ 的谱分解，仅保留大于截断阈值的特征模态。记相对截断阈值为 $tau_"TSVD" > 0$，并定义保留指标集
$
  I_"TSVD"
  := { i in {1, dots, N}: lambda_i > tau_"TSVD" lambda_"max" }.
$
于是 TSVD 解写为
$
  bold(phi)_("TSVD")
  = sum_(i in I_"TSVD")
  (bold(q)_i^T bold(F)) / lambda_i bold(q)_i.
$
这等价于先将过小特征值对应的方向截断，再仅在保留下来的稳定子空间中求逆。

== Ridge

Ridge 通过给 $bold(G)$ 加入对角正则项来抬升小特征值方向的谱尺度。记相对正则强度为 $alpha_"Ridge" > 0$，则其线性系统写为
$
  (bold(G) + alpha_"Ridge" lambda_"max" bold(I)) bold(phi)_("Ridge")
  = bold(F).
$
等价地，利用谱分解可写为
$
  bold(phi)_("Ridge")
  = sum_(i=1)^N
  (bold(q)_i^T bold(F)) / (lambda_i + alpha_"Ridge" lambda_"max") bold(q)_i.
$
因此 Ridge 不会直接删除任何模态，而是通过统一的谱移位减弱病态方向对解的放大。
