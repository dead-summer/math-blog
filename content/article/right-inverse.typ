#import "/typ/templates/blog.typ": *
#show: main.with(
  title: "混合弱形式稳定离散空间的构造",
  author: "summer",
  desc: [为固定特征与输出层求解的线弹性混合弱形式构造可证明稳定的离散空间],
  date: "2026-04-13",
  tags: (
    blog-tags.numerical-methods,
    blog-tags.pde,
  ),
  show-outline: true,
)

= 问题描述

设 $Omega subset RR^d, quad d in {2, 3}$，是一个有界、连通、拓扑平凡且边界 Lipschitz 的区域。考虑小变形各向同性线弹性问题。位移 $bold(u): Omega -> RR^d$ 与对称应力 $bold(sigma): Omega -> RR^(d times d)$ 满足
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
是线性应变张量，$bold(f)$ 是体力，$bold(cal(A))$ 是柔度张量。

定义函数空间
$
  bold(Sigma) & := { bold(tau) in (L^2(Omega))^(d times d) :
                  bold(tau) = bold(tau)^T,
                  nabla dot bold(tau) in (L^2(Omega))^d }, \
      bold(U) & := (L^2(Omega))^d.
$
并定义双线性形式
$
  a(bold(sigma), bold(tau)) & := integral_Omega (bold(cal(A)) : bold(sigma)) : bold(tau) dif bold(x), \
      b(bold(tau), bold(v)) & := integral_Omega (nabla dot bold(tau)) dot bold(v) dif bold(x).
$

于是 Hellinger-Reissner 混合弱形式为：求
$(bold(sigma), bold(u)) in bold(Sigma) times bold(U)$，使得
$
  cases(
    a(bold(sigma), bold(tau)) + b(bold(tau), bold(u)) & = 0 & quad forall bold(tau) in bold(Sigma),
    b(bold(sigma), bold(v)) & = - (bold(f), bold(v)) & quad forall bold(v) in bold(U).
  )
$

= 神经特征空间

考虑一个单隐层全连接神经网络：
$
  alpha_0 + sum_(m=1)^M alpha_m sigma.alt(bold(w)_m^T bold(x) + b_m),
$
其中 $sigma.alt$ 是激活函数，$M$ 是神经元数量，$alpha_m in RR$ 是输出层权重，$bold(w)_m in RR^3$ 是输入层权重，$b_m in RR$ 是偏置项。

本文采用如下重参数化来生成 $(bold(w)_m, b_m)$：取全局形状参数 $gamma > 0$，并令
$
  bold(w)_m = gamma bold(a)_m, quad b_m = gamma r_m.
$
其中 $bold(a)_m$ 表示超平面法向量，$r_m$ 表示截距。于是
$
  xi_m (bold(x)) = sigma.alt(gamma (bold(a)_m^T bold(x) + r_m)).
$
采样策略为
$
  bold(a)_m = bold(X)_m / norm(bold(X)_m)_2, quad r_m = U_m,
$
其中 $bold(X)_m in RR^3$ 是从标准正态分布采样的随机向量，$U_m in RR$ 是从 $[0, 1]$ 均匀分布采样的随机数。

记隐藏层神经元为
$
  xi_m (bold(x)) = sigma.alt(bold(w)_m^T bold(x) + b_m).
$
将 $xi_m: RR^3 -> RR$ 视为一个特征函数，隐藏层神经元集 ${xi_m}_1^M$ 可视为 $RR^3$ 空间中的一组基。定义神经特征空间
$
  Xi_M := span{ xi_0, xi_1, ..., xi_M },
$
其中 $xi_0 = 1$。因此，神经特征空间 $Xi_M$ 是由单隐层全连接神经网络的隐藏层神经元生成的函数空间。将 $Xi_M$ 视为一个近似空间，用于近似求解线弹性方程的解。

= 离散稳定性

经典 Brezzi 理论指出，要证明离散混合问题适定，关键是构造一个满足离散核上强制性与离散 inf-sup 条件的空间对 @boffi2013mixed。本文的目标是在近似空间 $Xi_M$ 的基础上构造一个空间对 $(bold(Sigma)_M, bold(U)_M)$，使得离散稳定性结构自动成立。

给定有限维子空间
$
  bold(Sigma)_M subset bold(Sigma), quad bold(U)_M subset bold(U),
$
离散混合问题为：求
$(bold(sigma)_M, bold(u)_M) in bold(Sigma)_M times bold(U)_M$，使得
$
  cases(
    a(bold(sigma)_M, bold(tau)_M) + b(bold(tau)_M, bold(u)_M) & = 0 & quad forall bold(tau)_M in bold(Sigma)_M,
    b(bold(sigma)_M, bold(v)_M) & = - (bold(f), bold(v)_M) & quad forall bold(v)_M in bold(U)_M.
  )
$

需要验证两个性质 @boffi2013mixed：

1. $a(dot, dot)$ 在离散核空间
  $
    bold(Z)_M := { bold(tau)_M in bold(Sigma)_M :
      b(bold(tau)_M, bold(v)_M) = 0, quad forall bold(v)_M in bold(U)_M }
  $
  上具有一致强制性；

2. 离散 inf-sup 常数
  $
    beta_M
    := inf_(bold(v)_M in bold(U)_M, bold(v)_M != 0)
    sup_(bold(tau)_M in bold(Sigma)_M, bold(tau)_M != 0)
    b(bold(tau)_M, bold(v)_M)
    / (norm(bold(tau)_M)_(bold(H)(div)) norm(bold(v)_M)_(L^2))
  $
  满足与离散维数无关的正下界。

在三维情形下，一个很自然的想法是分别为应力与位移独立选取一组全局特征，并定义候选空间
$
  bold(Sigma)_M := & span{ xi^(bold(sigma))_m bold(E)_alpha: alpha = 1, 2, ..., 6; 0 <= m <= M } subset bold(Sigma), \
      bold(U)_M := & span{ xi^(bold(u))_m bold(e)_i: 0 <= m <= M, i = 1, 2, 3 } subset bold(U).
$
其中 $bold(E)_alpha$ 是 $RR^(3 times 3)$ 中的对称单位矩阵，排列顺序依次为
$11, 22, 33, 12, 23, 13$，$bold(e)_i$ 是 $RR^3$ 的标准基向量。这个构造的直觉很直接：既然
$Xi_M$ 是一组标量特征，那么将它分别与对称矩阵基和向量基做张量积，似乎就能同时得到应力空间与位移空间。

然而，Brezzi 条件真正敏感的并不是 $a(dot, dot)$ 这一部分，而是
$b(bold(tau)_M, bold(v)_M) = (nabla dot bold(tau)_M, bold(v)_M)$ 所刻画的散度耦合。为方便书写，下文把上述六个对称单位矩阵直接记为
$bold(E)_(11), bold(E)_(22), bold(E)_(33), bold(E)_(12), bold(E)_(23), bold(E)_(13)$。对任意标量特征
$xi$，直接计算可得
$
  nabla dot (xi bold(E)_(11)) & = mat(partial_1 xi; 0; 0), quad
  nabla dot (xi bold(E)_(22)) & = mat(0; partial_2 xi; 0), quad
  nabla dot (xi bold(E)_(33)) & = mat(0; 0; partial_3 xi), \
  nabla dot (xi bold(E)_(12)) & = mat(partial_2 xi; partial_1 xi; 0), quad
  nabla dot (xi bold(E)_(23)) & = mat(0; partial_3 xi; partial_2 xi), quad
  nabla dot (xi bold(E)_(13)) & = mat(partial_3 xi; 0; partial_1 xi).
$
因此
$
  nabla dot bold(Sigma)_M
  subset
  span{ partial_j xi^(bold(sigma))_m bold(e)_i: 0 <= m <= M, 1 <= i, j <= 3 }.
$
可见，散度作用在应力基函数上产生的是特征函数的一阶导数，而不是特征函数本身。于是
$nabla dot bold(Sigma)_M$ 一般落在“导数特征张成的向量空间”中，而不是事先选定的位移空间
$bold(U)_M$。若 ${xi^(bold(sigma))_m}$ 与 ${xi^(bold(u))_m}$ 是两套独立特征，这种不匹配会更加明显；即便强行令两者取成同一组特征，也仍然要求标量特征空间对偏导运算保持封闭，而这对一般全局神经特征而言通常并不成立。

因此，上述朴素构造一般无法从空间定义本身推出与离散维数无关的离散 inf-sup 下界。它至多可能在某些特定特征选择下偶然稳定，但并不具备可证明的统一稳定性。换言之，离散 inf-sup 条件真正要求的是：对任意给定的
$bold(v)_M in bold(U)_M$，都能在应力空间中找到一个
$bold(tau)_M in bold(Sigma)_M$，使得
$
  nabla dot bold(tau)_M = bold(v)_M,
  quad
  norm(bold(tau)_M)_(bold(H)(div))
  <= C norm(bold(v)_M)_(L^2),
$
其中常数 $C > 0$ 不依赖于离散维数 $M$。这就是说，散度算子
$nabla dot: bold(Sigma)_M -> bold(U)_M$ 在离散层面上应当具有一个统一有界的右逆。

一旦从这个角度理解稳定性，构造思路就变得自然：不应先独立猜测一组应力基函数，再去检验它是否恰好与位移空间匹配；更合理的做法是先固定
$bold(U)_M$，再用一个对称散度右逆把它稳定地提升到应力空间中。下面便按照这一思路构造稳定的离散空间对。

= 稳定离散空间的构造

本节给出一个完全结构化的全局固定特征构造。核心思想是：先选定位移空间，再用一个*对称散度右逆算子*把它提升到应力空间；同时加入一个散度自由核空间，以补充应力的逼近能力。

== 位移空间

定义位移离散空间
$
  bold(U)_M := span{ xi_m bold(e)_i: 0 <= m <= M, 1 <= i <= d}.
$

这是一个纯全局固定特征空间。所有隐藏层特征都在离散化之前被固定，离散未知量仅是各分量上的输出层系数。

== 对称散度右逆

在一般区域上，采用一个由 Korn 不等式 @difratta2025korn 诱导的变分右逆。

#proposition[
  设 $Omega subset RR^d, d in {2, 3}$ 是有界、连通、拓扑平凡且边界 Lipschitz 的区域。则存在有界线性算子
  $
    bold(cal(T))_Omega: (L^2(Omega))^d -> bold(H)(div, Omega; SS^d),
  $
  使得对任意 $bold(v) in (L^2(Omega))^d$，
  $
    nabla dot bold(cal(T))_Omega (bold(v)) = bold(v),
  $
  并存在只依赖于 $Omega$ 的常数 $C_Omega > 0$，满足
  $
    norm(bold(cal(T))_Omega (bold(v)))_(bold(H)(div))
    <= C_Omega norm(bold(v))_(L^2).
  $
]

#proof[
  对任意给定的 $bold(v) in (L^2(Omega))^d$，考虑如下变分问题：求
  $bold(w)_v in (H_0^1(Omega))^d$，使得
  $
    integral_Omega bold(epsilon)(bold(w)_v) : bold(epsilon)(bold(z)) dif bold(x)
    = - integral_Omega bold(v) dot bold(z) dif bold(x),
    quad forall bold(z) in (H_0^1(Omega))^d.
  $

  由 Lipschitz 区域上的 Korn 不等式，
  $
    norm(bold(z))_(H^1(Omega))
    <= C_K norm(bold(epsilon)(bold(z)))_(L^2(Omega)),
    quad forall bold(z) in (H_0^1(Omega))^d,
  $
  故左端双线性型在 $(H_0^1(Omega))^d$ 上连续且强制，右端线性泛函连续。于是由 Lax-Milgram 定理，$bold(w)_v$ 唯一存在，且
  $
    norm(bold(epsilon)(bold(w)_v))_(L^2(Omega))
    <= C_K norm(bold(v))_(L^2(Omega)).
  $

  现在定义
  $
    bold(cal(T))_Omega (bold(v)) := bold(epsilon)(bold(w)_v).
  $
  显然它是对称张量场，且
  $
    norm(bold(cal(T))_Omega (bold(v)))_(L^2(Omega))
    <= C_K norm(bold(v))_(L^2(Omega)).
  $

  再证明其散度右逆性质。对任意 $bold(z) in (C_0^oo (Omega))^d$，
  $
    integral_Omega (nabla dot bold(cal(T))_Omega (bold(v))) dot bold(z) dif bold(x)
    &= - integral_Omega bold(cal(T))_Omega (bold(v)) : nabla bold(z) dif bold(x) \
    &= - integral_Omega bold(epsilon)(bold(w)_v) : bold(epsilon)(bold(z)) dif bold(x) \
    &= integral_Omega bold(v) dot bold(z) dif bold(x).
  $
  这里第二步利用了对称张量与反对称梯度正交。于是
  $
    nabla dot bold(cal(T))_Omega (bold(v)) = bold(v)
  $
  以分布意义成立。由于右端属于 $(L^2(Omega))^d$，可知
  $
    bold(cal(T))_Omega (bold(v)) in bold(H)(div, Omega; SS^d).
  $

  最后，结合上面的 $L^2$ 估计与
  $
    norm(nabla dot bold(cal(T))_Omega (bold(v)))_(L^2(Omega))
    = norm(bold(v))_(L^2(Omega)),
  $
  得
  $
    norm(bold(cal(T))_Omega (bold(v)))_(bold(H)(div))
    <= sqrt(C_K^2 + 1) norm(bold(v))_(L^2(Omega)).
  $
  因此可取 $C_Omega := sqrt(C_K^2 + 1)$。证毕。
]


在 $Omega = [0, 1]^3$ 上，可显式构造上述命题的右逆如下。对任意标量函数 $phi: Omega -> RR$，定义三个坐标原函数算子
$
  (cal(I)_1 phi)(bold(x)) & := integral_0^(x_1) phi(s, x_2, x_3) dif s, \
  (cal(I)_2 phi)(bold(x)) & := integral_0^(x_2) phi(x_1, s, x_3) dif s, \
  (cal(I)_3 phi)(bold(x)) & := integral_0^(x_3) phi(x_1, x_2, s) dif s.
$

再定义算子
$
  bold(cal(T))_square (bold(v))
  :=
  mat(
    cal(I)_1 v_1, 0, 0;
    0, cal(I)_2 v_2, 0;
    0, 0, cal(I)_3 v_3
  ).
$

#proposition(title: [三维立方体上的散度右逆])[
  对于立方体区域 $Omega = [0, 1]^3$，算子 $bold(cal(T))_square$ 是从 $(L^2(Omega))^3$ 到 $bold(H)(div, Omega; SS^3)$ 的一个显式有界对称散度右逆。更具体地说，
  $
    nabla dot bold(cal(T))_square (bold(v)) = bold(v),
    quad
    norm(bold(cal(T))_square (bold(v)))_(bold(H)(div))
    <= sqrt(3/2) norm(bold(v))_(L^2).
  $
]

#proof[
  由定义，
  $
    bold(cal(T))_square (bold(v))
    =
    mat(
      cal(I)_1 v_1, 0, 0;
      0, cal(I)_2 v_2, 0;
      0, 0, cal(I)_3 v_3
    ),
  $
  因而
  $
    nabla dot bold(cal(T))_square (bold(v))
    =
    mat(
      partial_1(cal(I)_1 v_1);
      partial_2(cal(I)_2 v_2);
      partial_3(cal(I)_3 v_3)
    )
    =
    mat(v_1; v_2; v_3)
    = bold(v).
  $

  再证明有界性。对任意 $phi in L^2(Omega)$，由 Cauchy-Schwarz 不等式，
  $
    abs((cal(I)_1 phi)(x_1, x_2, x_3))^2
    <= x_1 integral_0^(x_1) abs(phi(s, x_2, x_3))^2 dif s.
  $
  对 $(x_1, x_2, x_3)$ 在 $Omega$ 上积分，并交换积分次序，得到
  $
    norm(cal(I)_1 phi)_(L^2(Omega))^2 & <= integral_0^1 integral_0^1 integral_0^1
                                   x_1 integral_0^(x_1) abs(phi(s, x_2, x_3))^2 dif s dif x_1 dif x_2 dif x_3 \
                                 & = integral_0^1 integral_0^1 integral_0^1
                                   abs(phi(s, x_2, x_3))^2 integral_s^1 x_1 dif x_1 dif s dif x_2 dif x_3 \
                                 & <= 1/2 norm(phi)_(L^2(Omega))^2.
  $
  因此
  $
    norm(cal(I)_1 phi)_(L^2(Omega))
    <= 1/sqrt(2) norm(phi)_(L^2(Omega)).
  $
  对 $cal(I)_2, cal(I)_3$ 同理成立。

  于是对任意 $bold(v) = (v_1, v_2, v_3)^T$，
  $
    norm(bold(cal(T))_square (bold(v)))_(L^2(Omega))^2
    = norm(cal(I)_1 v_1)_(L^2)^2 + norm(cal(I)_2 v_2)_(L^2)^2 + norm(cal(I)_3 v_3)_(L^2)^2
    <= 1/2 norm(bold(v))_(L^2(Omega))^2.
  $
  再结合
  $
    norm(nabla dot bold(cal(T))_square (bold(v)))_(L^2(Omega))
    = norm(bold(v))_(L^2(Omega)),
  $
  得
  $
    norm(bold(cal(T))_square (bold(v)))_(bold(H)(div))^2
    <= 3/2 norm(bold(v))_(L^2(Omega))^2.
  $
  故
  $
    norm(bold(cal(T))_square (bold(v)))_(bold(H)(div))
    <= sqrt(3/2) norm(bold(v))_(L^2(Omega)).
  $
  证毕。
]


== 散度自由核空间

仅靠 $bold(cal(T))_Omega (bold(U)_M)$，虽然已经足以建立稳定性，但它主要负责生成散度部分，难以充分逼近一般对称应力。为此还需要额外构造一个散度自由核空间。

为继续只使用同一组固定特征，按维度分别构造核空间：

当 $d = 2$ 时，取标量势空间
$
  Psi_M := Xi_M,
$
并定义 Airy 算子 @arnold2002mixed
$
  "airy"(psi)
  :=
  mat(
    partial_22 psi, - partial_12 psi;
    - partial_12 psi, partial_11 psi
  ).
$
于是可定义
$
  bold(K)_M := "airy"(Psi_M),
$
并有
$
  nabla dot bold(kappa) = bold(0), quad forall bold(kappa) in bold(K)_M.
$

当 $d = 3$ 时，取对称张量势空间
$
  bold(W)_M
  := span{ xi_m bold(E)_alpha : 0 <= m <= M, 1 <= alpha <= 6 },
$
并定义三维弹性复形中的不相容算子 @arnold2006elasticity_complex
$
  "inc"(bold(Phi)) := "Curl"(("Curl" bold(Phi))^T).
$
若 $bold(Phi)$ 对称，则 $"inc"(bold(Phi))$ 仍为对称张量，且满足
$
  nabla dot "inc"(bold(Phi)) = bold(0).
$
因此可定义
$
  bold(K)_M := "inc"(bold(W)_M).
$

综上，在 $d=2$ 与 $d=3$ 两种情形下，均有
$
  bold(K)_M subset bold(H)(div, Omega; SS^d),
  quad
  nabla dot bold(kappa) = bold(0) quad forall bold(kappa) in bold(K)_M.
$

== 离散空间

现在定义最终的离散空间对
$
  bold(Sigma)_M := bold(cal(T))_Omega (bold(U)_M) + bold(K)_M.
$

该构造满足两个决定性特征：

1. 它仍然是*纯全局固定特征*模型。所有基函数都是显式给定的全局函数，不涉及网格自由度，也不需要求解隐藏层参数。

2. 它把应力空间拆成两部分：$bold(cal(T))_Omega (bold(U)_M)$ 通过散度与位移空间精确耦合，负责稳定性；$bold(K)_M$ 位于散度核中，负责增强应力逼近能力。

= 离散稳定性的证明

下面证明上述构造满足一个与离散维数无关的离散 inf-sup 下界。

== 离散 inf-sup 条件

#theorem[
  设 $Omega subset RR^d, d in {2, 3}$ 是有界、连通、拓扑平凡且边界 Lipschitz 的区域。对任意给定的特征数 $M >= 0$、形状参数 $gamma > 0$ 以及固定特征参数
  ${bold(a)_m}_(m=1)^M$、${r_m}_(m=1)^M$，由
  $
        bold(U)_M & = span{ xi_m bold(e)_i: 0 <= m <= M, 1 <= i <= d}, \
    bold(Sigma)_M & = bold(cal(T))_Omega (bold(U)_M) + bold(K)_M
  $
  定义的离散空间对满足离散 inf-sup 条件，且
  $
    beta_M >= 1 / C_Omega.
  $
]

#proof[
  任取非零 $bold(v)_M in bold(U)_M$。由于
  $
    bold(cal(T))_Omega (bold(U)_M) subset bold(Sigma)_M,
  $
  可选取特定测试函数
  $
    bold(tau)_M := bold(cal(T))_Omega (bold(v)_M) in bold(Sigma)_M.
  $

  由上一节命题，
  $
    nabla dot bold(tau)_M = bold(v)_M.
  $
  因而
  $
    b(bold(tau)_M, bold(v)_M)
    = integral_Omega (nabla dot bold(tau)_M) dot bold(v)_M dif bold(x)
    = norm(bold(v)_M)_(L^2(Omega))^2.
  $

  同时，由 $bold(cal(T))_Omega$ 的有界性，
  $
    norm(bold(tau)_M)_(bold(H)(div))
    <= C_Omega norm(bold(v)_M)_(L^2(Omega)).
  $
  因此
  $
    b(bold(tau)_M, bold(v)_M)
    / (norm(bold(tau)_M)_(bold(H)(div)) norm(bold(v)_M)_(L^2))
    >= 1 / C_Omega.
  $

  由于上式对任意非零 $bold(v)_M in bold(U)_M$ 都成立，再取关于
  $bold(tau)_M$ 的上确界以及关于 $bold(v)_M$ 的下确界，即得
  $
    beta_M >= 1 / C_Omega.
  $
  证毕。
]

== 核上强制性与适定性

接下来验证 Brezzi 理论中的另一项要求。对各向同性材料，若材料参数满足
$
  E > 0, quad -1 < nu < 1/2,
$
柔度张量 $bold(cal(A))$ 在对称张量空间上正定。因此存在常数
$0 < c_A <= C_A < oo$，使得对任意 $bold(tau) in bold(Sigma)$ 都有
$
  c_A norm(bold(tau))_(L^2(Omega))^2
  <= a(bold(tau), bold(tau))
  <= C_A norm(bold(tau))_(L^2(Omega))^2.
$

令
$
  bold(Z)_M
  := { bold(tau)_M in bold(Sigma)_M :
    b(bold(tau)_M, bold(v)_M) = 0, quad forall bold(v)_M in bold(U)_M }.
$
由于 $bold(Z)_M subset bold(Sigma)$，上式对所有 $bold(tau)_M in bold(Z)_M$ 成立，即
$
  a(bold(tau)_M, bold(tau)_M)
  >= c_A norm(bold(tau)_M)_(L^2(Omega))^2.
$
这表明 $a(dot, dot)$ 在离散核空间上具有一致强制性。

#theorem[
  上述离散空间对
  $
    bold(Sigma)_M = bold(cal(T))_Omega (bold(U)_M) + bold(K)_M, quad
    bold(U)_M = (Xi_M)^d
  $
  满足 Brezzi 条件。因此离散混合问题存在唯一解
  $(bold(sigma)_M, bold(u)_M)$，且稳定估计
  $
    norm(bold(sigma)_M)_(bold(H)(div))
    + norm(bold(u)_M)_(L^2)
    <= C norm(bold(f))_(L^2)
  $
  中的常数 $C$ 与特征数 $M$ 无关，但允许依赖于区域 $Omega$ 与材料参数。
]

#proof[
  上一节已证明离散 inf-sup 常数满足
  $
    beta_M >= 1 / C_Omega,
  $
  因而具有与离散维数无关的正下界。本节又证明了
  $
    a(bold(tau)_M, bold(tau)_M)
    >= c_A norm(bold(tau)_M)_(L^2)^2,
    quad forall bold(tau)_M in bold(Z)_M.
  $
  因此，经典 Brezzi 理论的两个核心条件全部成立 @boffi2013mixed。

  由此可知离散混合问题存在唯一解，而且解依赖于右端项连续稳定。由于
  $c_A$ 与 $beta_M$ 的下界都不依赖于特征数 $M$，稳定常数也不依赖于离散维数。证毕。
]

= 逼近性质与误差估计框架

稳定性只说明离散问题可解且不病态，但要获得收敛性，还需要分析应力空间与位移空间的逼近能力。

== 应力的结构分解

任取一个足够光滑的对称应力场 $bold(sigma) in bold(Sigma)$，定义
$
  bold(g) := nabla dot bold(sigma) in bold(U),
$
并令
$
  bold(kappa) := bold(sigma) - bold(cal(T))_Omega (bold(g)).
$
由 $bold(cal(T))_Omega$ 的右逆性质可得
$
  nabla dot bold(kappa)
  = nabla dot bold(sigma) - nabla dot bold(cal(T))_Omega (bold(g))
  = bold(g) - bold(g)
  = bold(0).
$
因此
$
  bold(sigma) = bold(cal(T))_Omega (bold(g)) + bold(kappa),
  quad nabla dot bold(kappa) = bold(0).
$

这表明任意应力可自然分解为：

1. 由散度决定的提升部分 $bold(cal(T))_Omega (bold(g))$；
2. 一个散度自由部分 $bold(kappa)$。

在拓扑平凡假设下，弹性复形具有正合性与正则分解性质 @arnold2006elasticity_complex @arnold2021complexes @pauly2023elasticity_complex @cap2023poincare_bgg，因此散度自由部分可进一步表示为：

1. 当 $d = 2$ 时，$bold(kappa)$ 可由 Airy 势产生；
2. 当 $d = 3$ 时，$bold(kappa)$ 可由 `inc` 势产生。

这正与所构造的离散应力空间
$
  bold(Sigma)_M = bold(cal(T))_Omega (bold(U)_M) + bold(K)_M
$
相对应：第一部分由 $bold(cal(T))_Omega (bold(U)_M)$ 逼近，第二部分由 $bold(K)_M$ 逼近。

需要额外说明的是：若 $Omega$ 只是任意有界 Lipschitz 区域而不再假设拓扑平凡，则散度自由部分一般还会包含一个有限维 harmonic 应力空间。此时需要在 $bold(K)_M$ 之外再补一个有限维修正空间。为保持主线清晰，本文不展开这一更一般但技术性更强的情形。

== 准最优误差估计

由离散问题的稳定性与标准混合有限元抽象理论 @boffi2013mixed，可得如下准最优误差形式：
$
  norm(bold(sigma) - bold(sigma)_M)_(bold(H)(div))
  + norm(bold(u) - bold(u)_M)_(L^2)
  <= C [
    inf_(bold(tau)_M in bold(Sigma)_M)
    norm(bold(sigma) - bold(tau)_M)_(bold(H)(div))
    + inf_(bold(v)_M in bold(U)_M)
    norm(bold(u) - bold(v)_M)_(L^2)
  ].
$

这个估计已经给出了最重要的理论信息：离散误差由空间对
$bold(Sigma)_M times bold(U)_M$ 的最佳逼近误差控制。结合上一小节的结构分解，可以将
$bold(cal(T))_Omega (bold(U)_M)$ 理解为负责应力散度部分的逼近，而将 $bold(K)_M$ 理解为负责散度自由应力部分的逼近。

与多项式谱空间不同，`tanh` 特征空间的具体逼近速度依赖于特征参数
$gamma, bold(a)_m, r_m$ 的选择方式，因此本文不再追求显式的收敛阶公式，而只保留上述抽象的准最优误差估计。换言之，只要所选的固定特征族能够同时逼近位移与应力的相应结构分量，上述稳定构造就会导出收敛。

= 结论

若仅把应力空间与位移空间都看成”各自独立选取的一组全局基函数”的张成空间，则离散稳定条件通常没有保障。与之相对，本文给出的构造
$
      bold(U)_M & = span{ xi_m bold(e)_i: 0 <= m <= M, 1 <= i <= d}, \
  bold(Sigma)_M & = bold(cal(T))_Omega (bold(U)_M) + bold(K)_M
$
通过有界的对称散度右逆 $bold(cal(T))_Omega$ 把神经特征位移空间稳定地嵌入到应力空间中，再通过散度自由核空间 $bold(K)_M$ 补足应力逼近能力，从而同时获得：

1. 一个与离散维数无关的离散 inf-sup 下界；
2. 由 Brezzi 理论给出的离散适定性；
3. 与应力结构分解一致的准最优误差估计框架。

对于三维立方体区域 $[0, 1]^3$，上面的右逆还可以写成显式坐标积分公式，但一般 Lipschitz 区域上的右逆并不依赖这种特殊几何。

#bibliography("/public/reference/right-inverse.bib")
