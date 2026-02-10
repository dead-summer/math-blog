#import "/typ/templates/blog.typ": *
#show: main.with(
  title: "线性弹性力学",
  author: "summer",
  desc: [线性弹性力学的数学建模与变分原理],
  date: "2026-01-20",
  tags: (
    blog-tags.mechanics,
    blog-tags.mathematics,
  ),
  show-outline: true,
)

= 线性弹性问题

== 数学模型

设弹性体占据空间区域 $Omega subset RR^3$，其边界为 $partial Omega$。为简化讨论，假设边界条件为全固支（Clamped），即在边界上位移为零：
$
  bold(u) = bold(0) quad "on" partial Omega.
$

=== 几何关系

设 $bold(r_0(x))$ 为弹性体内点 $M(bold(x))$ 的位置矢量，$bold(r(x))$ 为变形后的位移矢量，则位移矢量为
$ bold(u(x)) = bold(r(x)) - bold(r_0(x)). $
令 $M'(bold(x) + dif bold(x))$ 为无限接近 $M(x)$ 的另一个点，则 $dif bold(r_0)$ 和 $dif bold(r)$ 为变形前后的连接两点的微元向量，
$
  dif bold(r) = dif bold(r_0) + dif bold(u).
$
因此，变形前线元 $M M'$ 的距离平方为：
$
  dif s_0^2 = dif bold(r_0) dot dif bold(r_0) = dif x_i dif x_i,
$
变形后：
$
  dif s^2 & = dif bold(r) dot dif bold(r) \
          & = (dif bold(r_0) + dif bold(u)) dot (dif bold(r_0) + dif bold(u)) \
          & = dif bold(r_0)^2 + 2 dif bold(r_0) dot dif bold(u) + dif bold(u)^2 \
          & = dif s_0^2 + 2 partial_j u_i dif x_i dif x_j + partial_i u_k partial_j u_k dif x_i dif x_j \
          & = dif s_0^2 + 2 D_(i j) dif x_i dif x_j,
$
其中，
$
  D_(i j) = 1/2 (partial_j u_i + partial_i u_j + partial_i u_k partial_j u_k).
$
$D_(i j)$ 称为 Green-Lagrange 应变张量。当变形很小时，位移梯度 $partial_j u_i$ 是一阶小量，我们可以忽略其二次项，从而得到线性应变张量：
$
  epsilon_(i j) = 1/2 (partial_j u_i + partial_i u_j).
$

=== 本构关系

根据 Hooke 定律，对于各向同性线弹性材料，应力张量 $sigma_(i j)$ 与应变张量 $epsilon_(i j)$ 之间存在线性关系：
$
  sigma_(i j) = E/(1 + nu) epsilon_(i j) + (E nu)/((1 + nu)(1 - 2 nu)) epsilon_(k k) delta_(i j),
$
其中 $E$ 为杨氏模量，$nu in (0, 0.5)$ 为泊松比。该关系也可以用四阶弹性张量 $C_(i j k l)$ 表示：
$
  sigma_(i j) = C_(i j k l) epsilon_(k l), quad C_(i j k l) = lambda delta_(i j) delta_(k l) + mu (delta_(i k) delta_(j l) + delta_(i l) delta_(j k)),
$
其中 $lambda$ 和 $mu$ 分别为 Lamé 常数：
$
  lambda = (E nu)/((1 + nu) (1 - 2 nu)), quad mu = E/(2(1 + nu)).
$
以张量形式表示为：
$
  bold(sigma) = lambda tr(bold(epsilon)) bold(I) + 2 mu bold(epsilon).
$

=== 平衡方程

根据静态平衡，作用于弹性体内任意一微元体积 $V subset Omega$ 的合力应为零。这包括体力和面力：
$
  integral_(partial V) bold(sigma) dot bold(n) dif S + integral_V bold(f) dif V = 0,
$
其中 $bold(f)$ 是单位体积的体力，$bold(sigma) dot bold(n)$ 是在边界面 $partial V$ 上的面力密度，$bold(n)$ 为外法向矢量。根据 Gauss 散度定理，可将面积分转化为体积分：
$
  integral_V (nabla dot bold(sigma) + bold(f)) dif V = 0.
$
由于上述等式对任意体积 $V subset Omega$ 均成立，被积函数必须在 $Omega$ 上几乎处处为零，得到平衡方程的微分形式（亦称强形式）：
$
  nabla dot bold(sigma) + bold(f) = 0 quad "in" Omega.
$
或者用分量形式表示：
$
  sigma_(i j, j) + f_i = 0, quad i = 1, 2, 3.
$

== 变分原理

在处理连续介质力学问题时，除了上述微分形式（强形式），我们常采用积分形式（弱形式）或能量最小化形式。定义容许位移场空间 $cal(V)$，对于固支问题，容许函数在边界上为零：
$
  cal(V) = { bold(v) in [H^1(Omega)]^3 : bold(v) = bold(0) "on" partial Omega }.
$

=== 虚功原理

虚功原理指出：对于一个处于平衡状态的变形体，在外力作用下，对于任意满足位移边界条件的虚位移，外力所做的虚功等于物体内部储存的虚应变能。

设 $bold(v) in cal(V)$ 为任意虚位移场（测试函数）。
- *外力虚功*：仅由体力 $bold(f)$ 产生（因边界固支，面力不做功）：
  $ integral_Omega bold(f) dot bold(v) dif V. $
- *内力虚功*：应力在虚应变 $bold(epsilon)(bold(v))$ 上所做的功：
  $
    integral_Omega bold(sigma)(bold(u)) : bold(epsilon)(bold(v)) dif V = integral_Omega sigma_(i j) epsilon_(i j)(bold(v)) dif V.
  $

根据虚功原理，我们得到平衡方程的弱形式：寻找 $bold(u) in cal(V)$ 使得
$
  integral_Omega sigma_(i j)(bold(u)) epsilon_(i j)(bold(v)) dif V = integral_Omega f_i v_i dif V, quad forall bold(v) in cal(V).
$
定义双线性形式 $D(bold(u), bold(v))$ 和线性泛函 $F(bold(v))$：
$
  D(bold(u), bold(v)) & := integral_Omega sigma_(i j)(bold(u)) epsilon_(i j)(bold(v)) dif V, \
           F(bold(v)) & := integral_Omega f_i v_i dif V.
$
则问题表述为：寻找 $bold(u) in cal(V)$ 使得 $D(bold(u), bold(v)) = F(bold(v)), forall bold(v) in cal(V)$。

=== 最小势能原理

最小势能原理指出：在所有满足位移边界条件的容许位移场中，真实位移场使得系统的总势能取最小值。

系统的总势能 $J (bold(u))$ 定义为应变能 $U$ 减去外力势能 $W$：
$
  J (bold(u)) = U(bold(u)) - W(bold(u)).
$
其中：
$
  U(bold(u)) &= 1/2 integral_Omega bold(sigma)(bold(u)) : bold(epsilon)(bold(u)) dif V = 1/2 integral_Omega sigma_(i j)(bold(u)) epsilon_(i j)(bold(u)) dif V, \
  W(bold(u)) &= integral_Omega bold(f) dot bold(u) dif V.
$
因此，总势能泛函为：
$
  J (bold(u)) = 1/2 D(bold(u), bold(u)) - F(bold(u)).
$
最小势能原理表述为变分问题：寻找 $bold(u) in cal(V)$ 使得
$
  J (bold(u)) = min_(bold(v) in cal(V)) J (bold(v)).
$

=== 变分原理与平衡方程

下面证明上述三个问题在数学上是等价的。

#proposition[
  位移场 $bold(u) in cal(V)$ 是总势能泛函 $J (bold(u))$ 的极小值点，当且仅当它满足虚功原理方程 $D(bold(u), bold(v)) = F(bold(v))$。
]

#proof[
  考虑泛函 $J (bold(u))$ 在方向 $bold(v) in cal(V)$ 上的 Gâteaux 导数。令 $theta in RR$，考察函数 $J (bold(u) + theta bold(v))$：
  $
    J (bold(u) + theta bold(v)) &= 1/2 D(bold(u) + theta bold(v), bold(u) + theta bold(v)) - F(bold(u) + theta bold(v)) \
    &= 1/2 [D(bold(u), bold(u)) + theta D(bold(u), bold(v)) + theta D(bold(v), bold(u)) + theta^2 D(bold(v), bold(v))] - F(bold(u)) - theta F(bold(v)).
  $
  利用双线性形式 $D(dot, dot)$ 的对称性，即 $D(bold(u), bold(v)) = D(bold(v), bold(u))$，整理得：
  $
    J (bold(u) + theta bold(v)) = J (bold(u)) + theta [D(bold(u), bold(v)) - F(bold(v))] + 1/2 theta^2 D(bold(v), bold(v)).
  $
  计算 $theta = 0$ 处的导数，即泛函的一阶变分 $delta J (bold(u); bold(v))$：
  $
    delta J (bold(u); bold(v)) = partial/(partial theta) J (bold(u) + theta bold(v))bar_(theta=0) = D(bold(u), bold(v)) - F(bold(v)).
  $
  根据变分法原理，$bold(u)$ 使 $J$ 取驻值的必要条件是对于任意 $bold(v)$，一阶变分为零：
  $
    delta J = 0 arrow.l.r.double D(bold(u), bold(v)) = F(bold(v)), quad forall bold(v) in cal(V).
  $
  这正是虚功原理的表达式。
  此外，由于弹性张量正定，应变能 $1/2 D(bold(v), bold(v)) >= 0$。二阶变分 $delta^2 J = D(bold(v), bold(v))$ 非负，说明 $J (bold(u))$ 是凸泛函。因此，驻值点即为全局极小值点。证毕。
]

#proposition[
  如果足够光滑的位移场 $bold(u) in cal(V)$ 满足虚功原理（或最小势能原理），则它满足微分形式的平衡方程 $sigma_(i j, j) + f_i = 0$。
]

#proof[
  从虚功原理出发：
  $
    integral_Omega sigma_(i j) epsilon_(i j)(bold(v)) dif V = integral_Omega f_i v_i dif V, quad forall bold(v) in cal(V).
  $
  利用应力张量的对称性 $sigma_(i j) = sigma_(j i)$，我们有
  $
    sigma_(i j) epsilon_(i j)(bold(v)) &= sigma_(i j) [1/2 (partial_j v_i + partial_i v_j)] \
    &= sigma_(i j) [1/2 (partial_j v_i + partial_i v_j) + 1/2 (partial_j v_i - partial_i v_j)] \
    &= sigma_(i j) partial_j v_i.
  $
  将上式代入积分，并对第一项进行分部积分：
  $
    integral_Omega sigma_(i j) partial_j v_i dif V
    &= integral_Omega [partial_j (sigma_(i j) v_i) - sigma_(i j, j) v_i] dif V \
    &= integral_(partial Omega) sigma_(i j) n_j v_i dif S - integral_Omega sigma_(i j, j) v_i dif V.
  $
  由于测试函数 $bold(v)$ 在固支边界 $partial Omega$ 上为零，即 $v_i = 0$。因此，边界积分项消失。
  虚功方程变形为：
  $
    - integral_Omega sigma_(i j, j) v_i dif V = integral_Omega f_i v_i dif V.
  $
  移项合并：
  $
    integral_Omega (sigma_(i j, j) + f_i) v_i dif V = 0, quad forall bold(v) in cal(V).
  $
  根据变分法基本引理，由于积分对任意 $v_i$ 均成立，则括号内的项必须在 $Omega$ 内几乎处处为零：
  $
    sigma_(i j, j) + f_i = 0.
  $
  此即微分形式的平衡方程。
]

== 混合形式

=== 鞍点问题

在经典的位移法中，应力是位移的导出量。而在混合形式中，我们将应力 $bold(sigma)$ 与位移 $bold(u)$ 同时视为独立的未知量。
首先引入柔度张量$bold(A) = bold(C)^(-1)$。本构关系 $bold(sigma) = bold(C) : bold(epsilon)$ 可重写为：
$
  bold(A) bold(sigma) = bold(epsilon)(bold(u)).
$
对于各向同性材料，柔度张量的作用形式为：
$
  bold(A) bold(sigma) = 1/(2 mu) bold(sigma) - lambda/(2 mu (2 mu + 3 lambda)) tr(bold(sigma)) bold(I).
$

考虑齐次位移边界条件，线弹性问题的*强形式*方程组为：
$
  cases(
    bold(A) bold(sigma) - bold(epsilon)(bold(u)) & = 0 & quad "in" Omega,
    nabla dot bold(sigma) + bold(f) & = 0 & quad "in" Omega,
    bold(u) & = 0 & quad "on" partial Omega.
  )
$

为了建立变分形式，我们需要为应力和位移选择合适的函数空间。在混合方法中，通常要求应力的散度平方可积，而放宽对位移连续性的要求：
$
  bold(Sigma) &:= bold(H)(div, Omega; SS) := { bold(tau) in (L^2(Omega))^(3 times 3) : bold(tau)=bold(tau)^T, nabla dot bold(tau) in (L^2(Omega))^3 }, \
  bold(Q) &:= (L^2(Omega))^3 .
$
定义如下双线性形式 $a: bold(Sigma) times bold(Sigma) -> RR$ 与 $b: bold(Sigma) times bold(Q) -> RR$：
$
  a(bold(sigma), bold(tau)) & := integral_Omega (bold(A) bold(sigma)) : bold(tau) dif V, \
      b(bold(tau), bold(v)) & := integral_Omega (nabla dot bold(tau)) dot bold(v) dif V.
$
下面推导混合弱形式：
1. 对本构方程点乘测试函数 $bold(tau) in bold(Sigma)$ 并积分：
  $
    integral_Omega (bold(A) bold(sigma)) : bold(tau) dif V - integral_Omega bold(epsilon)(bold(u)) : bold(tau) dif V = 0.
  $
  利用分部积分公式，考虑到 $bold(u)|_(partial Omega) = 0$：
  $
    integral_Omega bold(epsilon)(bold(u)) : bold(tau) dif V = - integral_Omega u_i tau_(i j, j) dif V + integral_(partial Omega) u_i tau_(i j) n_j dif S = - b(bold(tau), bold(u)).
  $
  因此第一式变为：$a(bold(sigma), bold(tau)) + b(bold(tau), bold(u)) = 0$。

2. 对平衡方程点乘测试函数 $bold(v) in bold(Q)$ 并积分：
  $
    integral_Omega (nabla dot bold(sigma)) dot bold(v) dif V + integral_Omega bold(f) dot bold(v) dif V = 0,
  $
  从而有 $b(bold(sigma), bold(v)) = - (bold(f), bold(v))$.

综上，*混合弱形式*为：求 $(bold(sigma), bold(u)) in bold(Sigma) times bold(Q)$ 使得
$
  cases(
    a(bold(sigma), bold(tau)) + & b(bold(tau), bold(u)) = 0 & quad forall bold(tau) in bold(Sigma),
    & b(bold(sigma), bold(v)) = - (bold(f), bold(v)) & quad forall bold(v) in bold(Q).
  )
$
这是一个典型的 KKT 系统（Karush-Kuhn-Tucker system）。

=== Hellinger-Reissner 准则

上述混合弱形式方程组恰好对应于一个拉格朗日泛函的驻值点条件。定义 Hellinger-Reissner 泛函 $Pi: bold(Sigma) times bold(Q) -> RR$ 为：
$
  Pi(bold(tau), bold(v)) := 1/2 a(bold(tau), bold(tau)) + b(bold(tau), bold(v)) + (bold(f), bold(v)).
$
其中，项 $1/2 a(bold(tau), bold(tau))$ 对应于*余能*（Complementary Energy），而 $b(bold(tau), bold(v))$ 充当了通过拉格朗日乘子 $bold(v)$ 施加的约束项。

*Hellinger-Reissner 变分原理*可表述为：混合弱形式的解 $(bold(sigma), bold(u))$ 是泛函 $Pi$ 的鞍点。即：
$
  Pi(bold(sigma), bold(u)) = min_(bold(tau) in bold(Sigma)) max_(bold(v) in bold(Q)) Pi(bold(tau), bold(v)).
$

#proposition[
  若 $(bold(sigma), bold(u)) in bold(Sigma) times bold(Q)$ 是 Hellinger-Reissner 泛函的鞍点（即满足混合弱形式），且解具有足够的正则性，则它们满足强形式的平衡方程和本构方程。
]

#proof[
  鞍点意味着泛函在 $(bold(sigma), bold(u))$ 处关于任意方向 $(delta bold(tau), delta bold(v))$ 的一阶变分为零。

  1. *对应力变量 $bold(tau)$ 的变分：*
    首先计算泛函关于第一个变量 $bold(tau)$ 的变分。对于任意 $bold(tau), bold(v)$，变分结果为：
    $
      delta_(bold(tau)) Pi(bold(tau), bold(v)) &= lr(partial/(partial theta) [1/2 a(bold(tau) + theta delta bold(tau), bold(tau) + theta delta bold(tau)) + b(bold(tau) + theta delta bold(tau), bold(v))] |)_(theta=0) \
      &= a(bold(tau), delta bold(tau)) + b(delta bold(tau), bold(v)).
    $
    令 $(bold(tau), bold(v))$ 取驻值点 $(bold(sigma), bold(u))$，并令变分为零：
    $
      a(bold(sigma), delta bold(tau)) + b(delta bold(tau), bold(u)) = 0, quad forall delta bold(tau) in bold(Sigma).
    $
    代入具体积分表达式：
    $
      integral_Omega (bold(A) bold(sigma)) : delta bold(tau) dif V + integral_Omega (nabla dot delta bold(tau)) dot bold(u) dif V = 0.
    $
    若解 $bold(u)$ 具有足够的正则性，则对第二项进行逆向分部积分（注意 $bold(u)|_(partial Omega)=0$）：
    $
      integral_Omega (nabla dot delta bold(tau)) dot bold(u) dif V = - integral_Omega delta bold(tau) : bold(epsilon)(bold(u)) dif V.
    $
    于是：
    $
      integral_Omega [ bold(A) bold(sigma) - bold(epsilon)(bold(u)) ] : delta bold(tau) dif V = 0.
    $
    由 $delta bold(tau)$ 的任意性，可得本构方程：
    $
      bold(A) bold(sigma) = bold(epsilon)(bold(u)).
    $

  2. *对位移变量 $bold(v)$ 的变分：*
    计算泛函关于第二个变量 $bold(v)$ 的变分。对于任意 $bold(tau), bold(v)$，变分结果为：
    $
      delta_(bold(v)) Pi(bold(tau), bold(v)) &= lr(partial/(partial theta) [b(bold(tau), bold(v) + theta delta bold(v)) + (bold(f), bold(v) + theta delta bold(v))] |)_(theta=0) \
      &= b(bold(tau), delta bold(v)) + integral_Omega bold(f) dot delta bold(v) dif V.
    $
    令 $(bold(tau), bold(v))$ 取驻值点 $(bold(sigma), bold(u))$，并令变分为零：
    $
      b(bold(sigma), delta bold(v)) + integral_Omega bold(f) dot delta bold(v) dif V = 0, quad forall delta bold(v) in bold(Q).
    $
    展开 $b(dot, dot)$：
    $
      integral_Omega (nabla dot bold(sigma)) dot delta bold(v) dif V + integral_Omega bold(f) dot delta bold(v) dif V = 0.
    $
    合并同类项：
    $
      integral_Omega (nabla dot bold(sigma) + bold(f)) dot delta bold(v) dif V = 0.
    $
    由 $delta bold(v)$ 的任意性，可得平衡方程：
    $
      nabla dot bold(sigma) + bold(f) = 0.
    $
]

= 平面应力问题

上一章介绍了三维弹性体的经典线性化理论，一切弹性体都是三维的，因此原则上都可以直接运用上述一般性的三维理论去解决。但是实际中许多典型的弹性结构在几何形体和力学上有其特殊性，例如薄的板壳、细长的梁杆以及种种对称性等。对此要笼统地按一般三维模式去解决往往是很不经济，而且也是不必要的。

平面弹性问题是弹性力学中一个重要的简化模型。在该问题中，位移 $bold(u)$ 只依赖于两个空间坐标 $x_1, x_2$，而与第三个坐标 $x_3$ 无关，即
$
  u_i = u_i (x_1, x_2), quad i = 1, 2, 3.
$

平面应力问题是平面弹性问题的一种，主要针对薄板的板平面内的变形。此时载荷平行于薄板平面并沿薄板厚度均匀分布，两个板面（法向设为 $x_3$）完全自由，在板面上有
$
  sigma_(13) = sigma_(23) = sigma_(33) = 0,
$<eq:plane-stress-asspmtion>
由于板很薄，所以在板的内部也有此关系式。

== 数学模型

用 $Omega$ 表示原弹性体的一个标准断面，$partial Omega$ 为其边界。假设边界条件为全固支（Clamped），即在边界上位移为零：
$
  bold(u) = bold(0) quad "on" partial Omega.
$
约定下标 $alpha, beta, gamma$ 只取值 $1, 2$。

=== 几何关系

将条件 $sigma_(13) = sigma_(23) = 0$ 代入三维弹性体的 Hooke 定律，可得
$
  epsilon_(13) = epsilon_(23) = 0.
$
由应变定义，则
$
  0 = epsilon_(alpha 3)
  = 1/2 ((partial u_3)/(partial x_alpha) + (partial u_alpha)/(partial x_3))
  = 1/2 (partial u_3)/(partial x_alpha),
  quad alpha = 1, 2,
$
所以 $u_3$ 为常数，从而可知在二维平面应力问题中，只有 $u_1, u_2$ 的自由量。因此位移与应变的关系为
$
  epsilon_(alpha beta)(bold(u))
  = 1/2 ((partial u_beta)/(partial x_alpha) + (partial u_alpha)/(partial x_beta)),
  quad alpha, beta = 1, 2.
$
$epsilon_(33)$ 通常不为 $0$，由平面应力条件 $sigma_(33)=0$ 决定。

=== 本构关系

将平面应力假设 $sigma_(33)=0$ 带入 Hooke 定律，可解得厚向应变
$
  epsilon_(33) = - nu/(1 - nu) (epsilon_(11) + epsilon_(22)).
$
再将上式带入三维 Hooke 定律，并令 $i = alpha, j = beta$，得到等效二维本构：
$
  sigma_(alpha beta) & = E/(1 + nu) epsilon_(alpha beta)
                       + (E nu) / (1 - nu^2) epsilon_(gamma gamma) delta_(alpha beta) \
                     & = D'[(1 - nu) epsilon_(alpha beta) + nu epsilon_(gamma gamma) delta_(alpha beta)],
                       quad alpha, beta, gamma=1,2,
$
其中 $D' = (E h)/(1- nu^2)$ 为抗拉刚度。或
$
  sigma_(alpha beta)
  = lambda_"plane" epsilon_(gamma gamma) delta_(alpha beta)
  + 2 mu epsilon_(alpha beta),
  quad alpha, beta, gamma = 1, 2,
$
其中 $lambda_"plane" = (2 lambda mu)/(lambda + 2 mu)$ 为二维等效 Lamé 常数。

=== 平衡方程

由三维平衡方程 $sigma_(i j, j) + f_i = 0$，令 $i = alpha$，并注意到 $sigma_(alpha 3) = 0$，得到二维平面应力问题的平衡方程：
$
  sigma_(alpha beta, beta) + f_alpha = 0, quad alpha, beta = 1, 2.
$

== 变分原理

定义容许位移空间：
$
  cal(V) = { bold(v) in [H^1(Omega)]^2 : bold(v) = bold(0) "on" partial Omega }.
$

=== 虚功原理

令 $bold(v) in cal(V)$ 为任意虚位移场。则
- *外力虚功*：
  $ integral_Omega bold(f) dot bold(v) dif S. $
- *内力虚功*：
  $
    integral_Omega bold(sigma)(bold(u)) : bold(epsilon)(bold(v)) dif S
    = integral_Omega sigma_(alpha beta)(bold(u)) epsilon_(alpha beta)(bold(v)) dif S,
    quad alpha, beta = 1, 2.
  $

虚功原理的弱形式为：求 $bold(u) in cal(V)$ 使得
$
  integral_Omega sigma_(alpha beta)(bold(u)) epsilon_(alpha beta)(bold(v)) dif S
  = integral_Omega f_alpha v_alpha dif S,
  quad forall bold(v) in cal(V).
$
定义双线性形式与线性泛函：
$
  D(bold(u), bold(v)) & := integral_Omega bold(sigma)(bold(u)) : bold(epsilon)(bold(v)) dif S, \
           F(bold(v)) & := integral_Omega bold(f) dot bold(v) dif S,
$
则弱形式为：寻找 $bold(u) in cal(V)$ 使得
$
  D(bold(u), bold(v)) = F(bold(v)), quad forall bold(v) in cal(V).
$

=== 最小势能原理

定义总势能泛函为应变能减去外力势能：
$
  J(bold(u)) = U(bold(u)) - W(bold(u)),
$
其中
$
  U(bold(u)) & = 1/2 integral_Omega bold(sigma)(bold(u)) : bold(epsilon)(bold(u)) dif S
               = 1/2 D(bold(u), bold(u)), \
  W(bold(u)) & = integral_Omega bold(f) dot bold(u) dif S = F(bold(u)).
$
因此
$
  J(bold(u)) = 1/2 D(bold(u), bold(u)) - F(bold(u)).
$
最小势能原理表述为：寻找 $bold(u) in cal(V)$ 使得
$
  J(bold(u)) = min_(bold(v) in cal(V)) J(bold(v)).
$

=== 变分原理与平衡方程

#proposition[
  位移场 $bold(u) in cal(V)$ 是总势能泛函 $J(bold(u))$ 的极小值点，当且仅当它满足虚功原理方程
  $
    D(bold(u), bold(v)) = F(bold(v)), quad forall bold(v) in cal(V).
  $
]

#proof[
  与三维情形完全类似。考虑 $J(bold(u)+theta bold(v))$ 的 Gâteaux 导数，并利用 $D(dot,dot)$ 的对称性：
  $
    delta J(bold(u); bold(v)) = partial/(partial theta) J(bold(u)+theta bold(v))bar_(theta=0)
    = D(bold(u), bold(v)) - F(bold(v)).
  $
  因此 $delta J = 0$ 等价于虚功原理。
  又因平面应力弹性张量仍正定，$D(bold(v),bold(v)) >= 0$，从而 $J$ 为凸泛函，驻值点即为极小值点。证毕。
]

#proposition[
  若足够光滑的位移场 $bold(u) in cal(V)$ 满足虚功原理（或最小势能原理），则它满足强形式的平衡方程
  $
    sigma_(alpha beta, beta) + f_alpha = 0, quad alpha=1,2.
  $
]

#proof[
  从虚功原理出发：
  $
    integral_Omega sigma_(alpha beta) epsilon_(alpha beta)(bold(v)) dif S
    = integral_Omega f_alpha v_alpha dif S, quad forall bold(v) in cal(V).
  $
  由于 $sigma_(alpha beta)=sigma_(beta alpha)$ 且 $alpha,beta=1,2$，有
  $
    sigma_(alpha beta) epsilon_(alpha beta)(bold(v)) = sigma_(alpha beta) partial_beta v_alpha.
  $
  对第一项分部积分：
  $
    integral_Omega sigma_(alpha beta) partial_beta v_alpha dif S
    &= integral_Omega [partial_beta(sigma_(alpha beta) v_alpha) - sigma_(alpha beta, beta) v_alpha] dif S \
    &= integral_(partial Omega) sigma_(alpha beta) n_beta v_alpha dif s - integral_Omega sigma_(alpha beta, beta) v_alpha dif S.
  $
  因为 $bold(v)=0$ 于 $partial Omega$，边界项为零，得
  $
    - integral_Omega sigma_(alpha beta, beta) v_alpha dif S = integral_Omega f_alpha v_alpha dif S.
  $
  即
  $
    integral_Omega (sigma_(alpha beta, beta) + f_alpha) v_alpha dif S = 0, quad forall bold(v) in cal(V).
  $
  由变分法基本引理，得到
  $
    sigma_(alpha beta, beta) + f_alpha = 0, quad alpha=1,2.
  $
  证毕。
]

== 混合形式

=== 鞍点问题

在平面应力的混合方法中，将应力 $bold(sigma)$ 与位移 $bold(u)$ 同时视为未知量。定义平面应力弹性张量 $bold(C)$ 及其逆 $bold(A)=bold(C)^(-1)$，使得
$
  bold(sigma) = bold(C) : bold(epsilon)(bold(u)),
  quad
  bold(A) bold(sigma) = bold(epsilon)(bold(u)).
$
对于各向同性材料，$bold(A)$ 的作用形式可写为：
$
  bold(A) bold(sigma)
  = 1/(2 mu) bold(sigma) - lambda_"plane"/(4 mu (mu + lambda_"plane")) tr(bold(sigma)) bold(I).
$

考虑齐次位移边界条件，平面应力问题的强形式可写为：
$
  cases(
    bold(A) bold(sigma) - bold(epsilon)(bold(u)) & = 0 & quad "in" Omega,
    nabla dot bold(sigma) + bold(f) & = 0 & quad "in" Omega,
    bold(u) & = bold(0) & quad "on" partial Omega.
  )
$

选择应力与位移空间：
$
  bold(Sigma) &:= bold(H)(div, Omega; SS)
  := { bold(tau) in (L^2(Omega))^(2 times 2) : bold(tau)=bold(tau)^T, nabla dot bold(tau) in (L^2(Omega))^2 }, \
  bold(Q) &:= (L^2(Omega))^2 .
$
定义双线性形式 $a: bold(Sigma) times bold(Sigma) -> RR$ 与 $b: bold(Sigma) times bold(Q) -> RR$：
$
  a(bold(sigma), bold(tau)) & := integral_Omega (bold(A) bold(sigma)) : bold(tau) dif S, \
      b(bold(tau), bold(v)) & := integral_Omega (nabla dot bold(tau)) dot bold(v) dif S.
$

推导混合弱形式：

1. 对本构方程与测试函数 $bold(tau) in bold(Sigma)$ 作内积并积分：
  $
    integral_Omega (bold(A) bold(sigma)) : bold(tau) dif S
    - integral_Omega bold(epsilon)(bold(u)) : bold(tau) dif S = 0.
  $
  利用分部积分并注意 $bold(u)|_(partial Omega)=0$：
  $
    integral_Omega bold(epsilon)(bold(u)) : bold(tau) dif S
    = - integral_Omega u_alpha tau_(alpha beta, beta) dif S
    + integral_(partial Omega) u_alpha tau_(alpha beta) n_beta dif s
    = - b(bold(tau), bold(u)).
  $
  因此得到
  $
    a(bold(sigma), bold(tau)) + b(bold(tau), bold(u)) = 0.
  $

2. 对平衡方程与测试函数 $bold(v) in bold(Q)$ 作内积并积分：
  $
    integral_Omega (nabla dot bold(sigma)) dot bold(v) dif S + integral_Omega bold(f) dot bold(v) dif S = 0,
  $
  即
  $
    b(bold(sigma), bold(v)) = - (bold(f), bold(v)).
  $

综上，*混合弱形式*为：求 $(bold(sigma), bold(u)) in bold(Sigma) times bold(Q)$ 使得
$
  cases(
    a(bold(sigma), bold(tau)) + & b(bold(tau), bold(u)) = 0 & quad forall bold(tau) in bold(Sigma),
    & b(bold(sigma), bold(v)) = - (bold(f), bold(v)) & quad forall bold(v) in bold(Q).
  )
$

=== Hellinger-Reissner 准则

定义 Hellinger-Reissner 泛函 $Pi: bold(Sigma) times bold(Q) -> RR$：
$
  Pi(bold(tau), bold(v)) := 1/2 a(bold(tau), bold(tau)) + b(bold(tau), bold(v)) + (bold(f), bold(v)).
$
Hellinger-Reissner 变分原理表述为：混合弱形式的解 $(bold(sigma), bold(u))$ 是 $Pi$ 的鞍点：
$
  Pi(bold(sigma), bold(u)) = min_(bold(tau) in bold(Sigma)) max_(bold(v) in bold(Q)) Pi(bold(tau), bold(v)).
$

#proposition[
  若 $(bold(sigma), bold(u)) in bold(Sigma) times bold(Q)$ 是 Hellinger-Reissner 泛函的鞍点（即满足混合弱形式），且解具有足够的正则性，则它们满足强形式的本构方程与平衡方程：
  $
    bold(A) bold(sigma) = bold(epsilon)(bold(u)), quad nabla dot bold(sigma) + bold(f) = 0.
  $
]

#proof[
  鞍点条件等价于 $Pi$ 在 $(bold(sigma), bold(u))$ 处对任意方向 $(delta bold(tau), delta bold(v))$ 的一阶变分为零。

  1. *对应力变量的变分：*
    $
      delta_(bold(tau)) Pi(bold(tau), bold(v))
      = a(bold(tau), delta bold(tau)) + b(delta bold(tau), bold(v)).
    $
    在驻值点令其为零，得
    $
      a(bold(sigma), delta bold(tau)) + b(delta bold(tau), bold(u)) = 0, quad forall delta bold(tau) in bold(Sigma).
    $
    展开并对第二项逆向分部积分（注意 $bold(u)|_(partial Omega)=0$）：
    $
      0 = integral_Omega (bold(A) bold(sigma)) : delta bold(tau) dif S
      + integral_Omega (nabla dot delta bold(tau)) dot bold(u) dif S
      = integral_Omega [bold(A) bold(sigma) - bold(epsilon)(bold(u))] : delta bold(tau) dif S,
    $
    由 $delta bold(tau)$ 任意性得
    $
      bold(A) bold(sigma) = bold(epsilon)(bold(u)).
    $

  2. *对位移变量的变分：*
    $
      delta_(bold(v)) Pi(bold(tau), bold(v))
      = b(bold(tau), delta bold(v)) + integral_Omega bold(f) dot delta bold(v) dif S.
    $
    在驻值点令其为零，得
    $
      b(bold(sigma), delta bold(v)) + integral_Omega bold(f) dot delta bold(v) dif S = 0, quad forall delta bold(v) in bold(Q),
    $
    即
    $
      integral_Omega (nabla dot bold(sigma) + bold(f)) dot delta bold(v) dif S = 0, quad forall delta bold(v) in bold(Q).
    $
    由任意性得到
    $
      nabla dot bold(sigma) + bold(f) = 0.
    $
  证毕。
]


= 薄板问题

薄板的弹性变形有两种典型模式，一是在纵向（板平面内）载荷作用下的纵向伸缩变形，二是在横向（垂直于板平面）载荷作用下的横向变形，即弯曲。

薄板的纵向伸缩变形就是平面应力问题，已经在上一章讨论过。在薄板的弯曲变形中，由于板很薄，其厚度远小于其他两个方向的几何尺寸，为了得到弯曲变形，只需在板面上加以不大的载荷，至少它们是远小于由此产生的内部的纵向伸缩应力。因此在三维弹性体的边界平衡方程中可以略去载荷 $g_i$，而得到

$ sum_(j=1)^3 sigma_(i j) n_j = g_i approx 0, $

这里 $bold(n) = (n_1, n_2, n_3)^T$ 为边界面的外法向。由于考虑的是小变形，可以认为弯曲后板的外法向 $bold(n)$ 平行于 $x_3$ 轴：$bold(n) approx (0, 0, plus.minus 1)^T$。这样，在板面上应有

$ sum_(j=1)^3 sigma_(i j) n_j approx plus.minus sigma_(i 3) approx 0. $

又由于板很薄，上述关系式在板的内部也成立。在此基础上，我们可以假定

$ sigma_(31) = sigma_(32) = sigma_(33) = 0 $<eq:plate-bending-asspmtion>

在板体内成立，至少 $sigma_(3 j)$ 相对于其他应力分量是小量。这一点与纵向伸缩变形的 @eq:plane-stress-asspmtion 相同。

另一方面，薄板在弯曲变形时，内部的纵向纤维产生拉伸或压缩。在板的向外凸的一面是拉伸，在凹的一面是压缩。从拉伸逐步变到压缩，中间存在一个没有纵向伸缩的中立面，中立面两侧的变形相反号。显然，中立面对称于上下板面，即位于板厚的中间。

== 数学模型

取变形前的中立面 $Omega$ 为 $x_1 - x_2$ 平面，即 $x_3 = 0$。 设 $h$ 为板厚，满足 $h << 1$。假设边界条件为全固支，即在边界上位移为零：
$
  bold(u) = bold(0) quad "on" partial Omega.
$
约定下标 $alpha, beta, gamma$ 只取值 $1, 2$。

=== 几何关系

中立面上的三个方向的位移为
$
  cases(
    u_1^((0)) (x_1, x_2) = u_1 (x_1, x_2, 0) = 0,
    u_2^((0)) (x_1, x_2) = u_2 (x_1, x_2, 0) = 0,
    u_3^((0)) (x_1, x_2) = u_3 (x_1, x_2, 0).
  )
$
由于板很薄，可以认为挠度 $u_3$ 沿板厚是一致的，即

$ u_3 (x_1, x_2, x_3) approx u_3^((0)) (x_1, x_2). $

因此
$
  epsilon_(33) = (partial u_3)/(partial x_3) approx (partial u^((0))_3)/(partial x_3) = 0.
$
由 @eq:plate-bending-asspmtion 式，应变 $epsilon_(3 alpha) = 0, alpha = 1, 2$。由应变定义，有
$
  (partial u_alpha)/(partial x_3) = - (partial u_3)/(partial x_alpha) approx - (partial u^((0))_3)/(partial x_alpha), quad alpha = 1, 2.
$
积分得到
$
  u_alpha (x_1, x_2, x_3) = u_alpha^((0)) (x_1, x_2) - x_3 (partial u_3^((0)))/(partial x_alpha), quad alpha = 1, 2.
$
由于是小变形，$u^((0))_alpha approx 0, alpha = 1, 2$，所以
$
  u_alpha (x_1, x_2, x_3) approx - x_3 (partial u_3^((0)))/(partial x_alpha), alpha = 1, 2.
$
记 $omega = u^((0))_3$。则应变通过横向位移 $omega$ 表示为
$
  epsilon_(alpha beta)(bold(u))
  = - x_3 (partial^2 omega)/(partial x_alpha partial x_beta),
  quad alpha, beta = 1, 2.
$
命
$
  K_(alpha beta) = - (partial^2 omega)/(partial x_alpha partial x_beta),
  quad alpha, beta = 1, 2,
$
其为中立面经过弯曲的曲率张量的一阶近似，于是
$
  epsilon_(alpha beta)(bold(u)) = x_3 K_(alpha beta), quad alpha, beta = 1, 2.
$

=== 本构关系

由于 @eq:plane-stress-asspmtion 和 @eq:plate-bending-asspmtion 相同，薄板弯曲变形的应力应变本构关系与平面应力问题相同：
$
  sigma_(alpha beta)
  = E /(1 + nu) epsilon_(alpha beta)
  + (E nu) / (1 - nu^2) epsilon_(gamma gamma) delta_(alpha beta), quad alpha, beta, gamma = 1, 2.
$
命
$
  M_(alpha beta) = integral_(-h/2)^(h/2) x_3 sigma_(alpha beta) dif x_3,
  quad alpha, beta = 1, 2,
$
将本构关系带入，即得
$
  M_(alpha beta)
  = D[(1 - nu) K_(alpha beta) + nu K_(gamma gamma) delta_(alpha beta)],
  quad alpha, beta, gamma = 1, 2,
$
其中 $D = (E h^3)/(12(1 - nu^2))$ 为弯曲刚度。上式就是薄板弯曲变形模式下的胡克定律，形式上与伸缩变形时相似，但是这里刻画“应变”的是曲率 $K_(i j)$，刻画“应力”的是弯矩 $M_(i j)$。

=== 平衡方程

假设体力 $f_alpha = 0, alpha, beta = 1, 2$，则对三维平衡方程 $sigma_(alpha beta, beta) + sigma_(alpha 3, 3) = 0$ 乘以 $x_3$ 并沿厚度积分：
$
  integral_(-h/2)^(h/2) x_3 sigma_(alpha beta, beta) dif x_3 + integral_(-h/2)^(h/2) x_3 sigma_(alpha 3, 3) dif x_3 = 0, quad alpha, beta = 1, 2.
$
第一项利用弯矩定义得 $M_(alpha beta, beta)$。第二项分部积分为：
$
  [x_3 sigma_(alpha 3)]_(-h/2)^(h/2) - integral_(-h/2)^(h/2) sigma_(alpha 3) dif x_3, quad alpha, beta = 1, 2,
$
由于板面剪应力为零（自由表面），边界项消失。定义横向剪力
$
  Q_alpha = integral_(-h/2)^(h/2) sigma_(alpha 3) dif x_3, quad alpha, beta = 1, 2,
$
则有*力矩平衡*：
$
  M_(alpha beta, beta) - Q_alpha = 0, quad alpha, beta = 1, 2.
$

对三维平衡方程 $sigma_(3 alpha, alpha) + sigma_(33, 3) + f_3 = 0$ 沿厚度积分，
$
  integral_(-h/2)^(h/2) sigma_(3 alpha, alpha) dif x_3 + [sigma_(33)]_(- h/2)^(h/2) + integral_(-h/2)^(h/2) f_3 dif x_3 = 0, quad alpha, beta = 1, 2,
$
定义等效的中面横向载荷
$
  f = [sigma_(33)]_(- h/2)^(h/2) + integral_(-h/2)^(h/2) f_3 dif x_3,
$
则可得*横向力平衡*：
$
  Q_(alpha, alpha) + f = 0, quad alpha, beta = 1, 2.
$
将力矩平衡式代入横向力平衡式消去 $Q_alpha$，得到用弯矩表示的平衡方程：
$
  M_(alpha beta, alpha beta) + f = 0, quad alpha, beta = 1, 2.
$

结合本构关系，对于均匀各向同性板，$nu$ 为常数，从而s可导出关于挠度 $omega$ 的双调和方程：
$
  D Delta^2 omega = f.
$

== 变分原理

在薄板弯曲问题中，应变能包含挠度的二阶导数，因此容许位移场空间需要更高的正则性。对于固支边界条件，取容许空间为：
$
  cal(V) = { v in H^2(Omega) : v = 0, (partial v)/(partial n) = 0 "on" partial Omega } = H_0^2(Omega).
$

=== 虚功原理

令 $v in cal(V)$ 为虚挠度场。
- *外力虚功*：
  $ integral_Omega f v dif S. $
- *内力虚功*：弯矩在虚曲率 $K_(alpha beta)(v)$ 上所做的功：
  $
    integral_Omega M_(alpha beta)(omega) K_(alpha beta)(v) dif S, quad alpha, beta = 1, 2.
  $

虚功原理指出：求 $omega in cal(V)$ 使得
$
  integral_Omega M_(alpha beta)(omega) K_(alpha beta)(v) dif S = integral_Omega f v dif S, quad forall v in cal(V), quad alpha, beta = 1, 2.
$
定义双线性形式 $D(omega, v)$ 和线性泛函 $F(v)$：
$
  D(omega, v) & := integral_Omega M_(alpha beta)(omega) K_(alpha beta)(v) dif S, quad alpha, beta = 1, 2. \
         F(v) & := integral_Omega f v dif S.
$

=== 最小势能原理

定义薄板弯曲应变能
$
  U(omega)
  := 1/2 integral_Omega M_(alpha beta)(omega) K_(alpha beta)(omega) dif S = 1/2 D(omega, omega), quad alpha, beta = 1, 2,
$
外力势能
$
  W(omega) := integral_Omega f omega dif S = F(omega).
$
总势能泛函
$
  J(omega) := U(omega) - W(omega) = 1/2 D(omega, omega) - F(omega).
$
最小势能原理表述为：
$
  J(omega) = min_(v in cal(V)) J(v).
$

=== 变分原理与平衡方程

#proposition[
  位移场 $omega in cal(V)$ 是总势能泛函 $J(omega)$ 的极小值点，当且仅当它满足虚功原理方程 $D(omega, v) = F(v)$。
]
#proof[
  与三维与平面应力情形相同。对任意 $v in cal(V)$，考察 $J(omega + theta v)$：
  $
    J(omega + theta v)
    = 1/2 D(omega + theta v, omega + theta v) - F(omega + theta v).
  $
  利用 $D(dot,dot)$ 的对称双线性，得
  $
    J(omega + theta v)
    = J(omega) + theta [D(omega, v) - F(v)] + 1/2 theta^2 D(v,v).
  $
  因此一阶变分
  $
    delta J(omega; v)
    = partial/(partial theta) J(omega + theta v)bar_(theta=0)
    = D(omega, v) - F(v).
  $
  故驻值条件 $delta J = 0$ 等价于虚功原理。又由弯曲刚度张量正定，可得 $D(v,v) >= 0$，从而 $J$ 为凸泛函，驻值点即为极小值点。证毕。
]

#proposition[
  若足够光滑的位移场 $omega in cal(V)$ 满足虚功原理，则它满足强形式的平衡方程 $M_(alpha beta, alpha beta) + f = 0, alpha, beta = 1, 2$。
]
#proof[
  从虚功原理出发：
  $
    integral_Omega M_(alpha beta) (- partial_(alpha beta) v) dif S = integral_Omega f v dif S, quad forall v in cal(V), quad alpha, beta = 1, 2.
  $
  对左端项进行两次分部积分。第一次分部积分：
  $
    - integral_Omega M_(alpha beta) v_(, alpha beta) dif S
    = integral_Omega M_(alpha beta, beta) v_(, alpha) dif S - integral_(partial Omega) M_(alpha beta) n_beta v_(, alpha) dif s.
  $
  第二次分部积分：
  $
    integral_Omega M_(alpha beta, beta) v_(, alpha) dif S
    = - integral_Omega M_(alpha beta, alpha beta) v dif S + integral_(partial Omega) M_(alpha beta, beta) n_alpha v dif s.
  $
  对于固支板，$v = 0$ 且 $partial_bold(n) v = 0$ 在边界上成立。因此，所有边界积分项均为零。
  方程化为：
  $
    - integral_Omega M_(alpha beta, alpha beta) v dif S = integral_Omega f v dif S.
  $
  即
  $
    integral_Omega (M_(alpha beta, alpha beta) + f) v dif S = 0.
  $
  由 $v$ 的任意性，得到平衡方程 $M_(alpha beta, alpha beta) + f = 0$。
]

== 混合形式

=== 鞍点问题

在薄板弯曲的混合形式中，将弯矩 $bold(M)$ 与挠度 $omega$ 同时视为未知量。由本构方程，可写为
$
  bold(M) = bold(C) : bold(K)(omega),
$
其中 $bold(K)(omega)$ 表示对称曲率张量 $K_(alpha beta)(omega)$，$bold(C)$ 为薄板弯曲刚度算子。引入柔度算子 $bold(A) = bold(C)^(-1)$，使得
$
  bold(A) bold(M) = bold(K)(omega).
$
对各向同性薄板，$bold(A)$ 的作用可显式写为
$
  bold(A) bold(M)
  = 1/(D(1-nu)) bold(M) - nu/(D(1-nu)(1+nu)) tr(bold(M)) bold(I),
$
其中 $tr(bold(M)) = M_(gamma gamma)$，$bold(I)$ 为 $2 times 2$ 单位阵。

考虑齐次固支边界条件，薄板弯曲的*强形式*系统为
$
  cases(
    bold(A) bold(M) - bold(K)(omega) = 0 & quad "in" Omega,
    nabla dot (nabla dot bold(M)) + f = 0 & quad "in" Omega,
    omega = 0 & quad "on" partial Omega,
    partial_n omega = 0 & quad "on" partial Omega.
  )
$
其中 $(nabla dot bold(M))_alpha := M_(alpha beta, beta)$，从而
$
  nabla dot (nabla dot bold(M)) = M_(alpha beta, alpha beta).
$

为建立混合弱形式，选择弯矩空间与挠度空间：
$
  bold(Sigma)
  &:= bold(H)(div div, Omega; SS)
  := { bold(tau) in (L^2(Omega))^(2 times 2) : bold(tau)=bold(tau)^T, tau_(alpha beta, alpha beta) in L^2(Omega) },
  \
  bold(Q) &:= L^2(Omega).
$
定义双线性形式 $a: bold(Sigma) times bold(Sigma) -> RR$ 与 $b: bold(Sigma) times bold(Q) -> RR$：
$
  a(bold(M), bold(tau)) & := integral_Omega (bold(A) bold(M)) : bold(tau) dif S, \
        b(bold(tau), v) & := integral_Omega (nabla dot (nabla dot bold(tau))) v dif S
                          = integral_Omega tau_(alpha beta, alpha beta) v dif S.
$

推导混合弱形式：

1. 对本构方程与测试函数 $bold(tau) in bold(Sigma)$ 作内积并积分：
  $
    integral_Omega (bold(A) bold(M)) : bold(tau) dif S
    - integral_Omega bold(K)(omega) : bold(tau) dif S = 0.
  $
  注意 $bold(K)(omega):bold(tau) = K_(alpha beta)(omega) tau_(alpha beta) = - omega_(,alpha beta) tau_(alpha beta)$。
  对第二项做两次分部积分，并利用固支边界条件，得
  $
    - integral_Omega omega_(,alpha beta) tau_(alpha beta) dif S
    = integral_Omega omega tau_(alpha beta, alpha beta) dif S
    = b(bold(tau), omega).
  $
  因此第一式化为
  $
    a(bold(M), bold(tau)) + b(bold(tau), omega) = 0.
  $

2. 对平衡方程与测试函数 $v in bold(Q)$ 相乘并积分：
  $
    integral_Omega (nabla dot (nabla dot bold(M))) v dif S + integral_Omega f v dif S = 0,
  $
  即
  $
    b(bold(M), v) = - (f, v).
  $

综上，*混合弱形式*为：求 $(bold(M), omega) in bold(Sigma) times bold(Q)$ 使得
$
  cases(
    a(bold(M), bold(tau)) + & b(bold(tau), omega) = 0 & quad forall bold(tau) in bold(Sigma),
    & b(bold(M), v) = - (f, v) & quad forall v in bold(Q).
  )
$

=== Hellinger-Reissner 准则

上述混合弱形式同样对应一个鞍点泛函。定义 Hellinger-Reissner 泛函 $Pi: bold(Sigma) times bold(Q) -> RR$：
$
  Pi(bold(tau), v)
  := 1/2 a(bold(tau), bold(tau)) + b(bold(tau), v) + (f, v).
$
其中 $1/2 a(bold(tau), bold(tau))$ 对应于弯曲*余能*，$b(bold(tau), v)$ 作为约束项引入拉格朗日乘子 $v$。

*Hellinger-Reissner 变分原理*表述为：混合弱形式的解 $(bold(M), omega)$ 是 $Pi$ 的鞍点：
$
  Pi(bold(M), omega)
  = min_(bold(tau) in bold(Sigma)) max_(v in bold(Q)) Pi(bold(tau), v).
$

#proposition[
  若 $(bold(M), omega) in bold(Sigma) times Q$ 是 Hellinger-Reissner 泛函的鞍点，则它们满足强形式的本构方程与平衡方程。
]

#proof[
  鞍点条件等价于 $Pi$ 在 $(bold(M), omega)$ 处对任意方向 $(delta bold(tau), delta v)$ 的一阶变分为零。

  1. *对弯矩变量的变分：*
    $
      delta_(bold(tau)) Pi(bold(tau), v)
      = a(bold(tau), delta bold(tau)) + b(delta bold(tau), v).
    $
    在驻值点令其为零，得
    $
      a(bold(M), delta bold(tau)) + b(delta bold(tau), omega) = 0,
      quad forall delta bold(tau) in bold(Sigma).
    $
    展开积分：
    $
      integral_Omega (bold(A) bold(M)) : delta bold(tau) dif S
      + integral_Omega (nabla dot (nabla dot delta bold(tau))) omega dif S = 0.
    $
    对第二项作逆向两次分部积分，并利用固支边界条件使边界项消失，可得
    $
      integral_Omega (nabla dot (nabla dot delta bold(tau))) omega dif S
      = - integral_Omega bold(K)(omega) : delta bold(tau) dif S.
    $
    于是
    $
      integral_Omega [bold(A) bold(M) - bold(K)(omega)] : delta bold(tau) dif S = 0.
    $
    由 $delta bold(tau)$ 的任意性，得本构方程 $bold(A) bold(M) = bold(K)(omega)$。

  2. *对挠度变量的变分：*
    $
      delta_(v) Pi(bold(tau), v)
      = b(bold(tau), delta v) + (f, delta v).
    $
    在驻值点令其为零，得
    $
      b(bold(M), delta v) + (f, delta v) = 0, quad forall delta v in bold(Q),
    $
    即
    $
      integral_Omega (nabla dot (nabla dot bold(M)) + f) delta v dif S = 0.
    $
    由任意性得平衡方程 $nabla dot (nabla dot bold(M)) + f = 0$，即 $M_(alpha beta, alpha beta) + f = 0$。
  综上得到强形式系统，并由本构消去 $bold(M)$ 可得双调和方程。证毕。
]
