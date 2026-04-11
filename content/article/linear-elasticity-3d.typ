#import "/typ/templates/blog.typ": *
#show: main.with(
  title: "三维线弹性问题",
  author: "summer",
  desc: [三维线弹性问题的数值算法构造],
  date: "2026-02-14",
  tags: (
    blog-tags.machine-learning,
    blog-tags.pde,
  ),
  show-outline: true,
)

= 问题描述

设弹性体占据空间区域 $Omega subset RR^3$，其边界为 $partial Omega$。三维线弹性问题即为寻找 $(bold(sigma), bold(u)) in bold(Sigma) times bold(U)$，满足如下方程：
$
  cases(
    bold(cal(A)) : bold(sigma) - bold(epsilon)(bold(u)) & = 0 & quad "in" Omega,
    nabla dot bold(sigma) + bold(f) & = 0 & quad "in" Omega,
    bold(u) & = 0 & quad "on" partial Omega.
  )
$
其中 $bold(epsilon)(bold(u)) = 1/2(nabla bold(u) + (nabla bold(u))^T)$ 是应变张量，$bold(f)$ 是体力，$bold(cal(A))$ 为与材料有关的柔度张量。函数空间定义如下：
$
  bold(Sigma) &:= bold(H)(div, Omega; SS) := { bold(tau) in (L^2(Omega))^(3 times 3) : bold(tau)=bold(tau)^T, nabla dot bold(tau) in (L^2(Omega))^3 }, \
  bold(U) &:= (L^2(Omega))^3.
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
我们可将 $xi_m: RR^3 -> RR$ 视为一个特征函数，隐藏层神经元集 ${xi_m}_1^M$ 可视为 $RR^3$ 空间中的一组基。定义神经特征空间
$
  Xi := span{xi_0, xi_1, ..., xi_M },
$
其中 $xi_0 = 1$。因此，神经特征空间 $Xi$ 是由单隐层全连接神经网络的隐藏层神经元生成的函数空间。我们可以将 $Xi$ 视为一个近似空间，用于近似求解线弹性方程的解。将 $bold(Sigma)$ 和 $bold(U)$ 分别近似为神经特征空间 $bold(Xi)$ 的张成空间，即
$
  bold(Sigma)_M := & span{ xi^(bold(sigma))_m bold(E)_alpha: alpha = 1, 2, ..., 6; 0 <= m <= M } subset bold(Sigma), \
      bold(U)_M := & span{xi^(bold(u))_m bold(e)_i: m = 0, ..., M, i = 1, 2, 3 } subset bold(U),
$
其中 $bold(E)_(alpha)$ 是 $RR^(3 times 3)$ 中的对称单位矩阵（排列顺序为 $11, 22, 33, 12, 23, 13$），$bold(e)_i$ 是 $RR^3$ 的标准基向量。

将近似解在上述基上展开：
$
  bold(sigma)_M = sum_(m=0)^M sum_(alpha=1)^6 s_(m, alpha) xi^(bold(sigma))_m bold(E)_alpha, quad
  bold(u)_M = sum_(m=0)^M sum_(i=1)^3 u_(m, i) xi^(bold(u))_m bold(e)_i.
$

#figure(
  neural-net(d: 3, n: 6, y: "sigma"),
  caption: [神经网络结构：蓝色实线表示神经网络参数已随机初始化并固定，红色虚线则为需要求解的系数向量],
)

记系数向量
$
  bold(s) & = (s_(0, 1), ..., s_(0, 6), s_(1, 1), ..., s_(M, 6))^T in RR^(6(M+1)), \
  bold(u) & = (u_(0, 1), ..., u_(0, 3), u_(1, 1), ..., u_(M, 3))^T in RR^(3(M+1)).
$

= 弱形式代数系统

定义如下双线性形式 $a: bold(Sigma) times bold(Sigma) -> RR$ 与 $b: bold(Sigma) times bold(U) -> RR$：
$
  a(bold(sigma), bold(tau)) & := integral_Omega (bold(cal(A)) : bold(sigma)) : bold(tau) dif bold(x), \
      b(bold(tau), bold(v)) & := integral_Omega (nabla dot bold(tau)) dot bold(v) dif bold(x),
$

线弹性方程的近似解 $(bold(sigma)_M, bold(u)_M) in bold(Sigma)_M times bold(U)_M$ 满足如下离散鞍点问题：
$
  cases(
    a(bold(sigma)_M, bold(tau)_M) + & b(bold(tau)_M, bold(u)_M) & = 0 & quad forall bold(tau)_M in bold(Sigma)_M,
    & b(bold(sigma)_M, bold(v)_M) & = -(bold(f), bold(v)_M) & quad forall bold(v)_M in bold(U)_M.
  )
$

需要指出的是，位移边界条件 $bold(u) = 0$ 在离散空间
$bold(U)_M$ 中未必能够被直接强制满足。为此，我们采用罚方法对该边界条件进行弱施加，即在离散方程中加入边界项
$
  c(bold(u), bold(v))
  := lambda_"bc" integral_(partial Omega) bold(u) dot bold(v) dif s,
$
其中 $lambda_"bc" > 0$ 为边界罚参数。

这样得到的离散问题可写为：求
$(bold(sigma)_M, bold(u)_M) in bold(Sigma)_M times bold(U)_M$，使得
$
  cases(
    a(bold(sigma)_M, bold(tau)_M) + b(bold(tau)_M, bold(u)_M) & = 0 & quad forall bold(tau)_M in bold(Sigma)_M,
    b(bold(sigma)_M, bold(v)_M) + c(bold(u)_M, bold(v)_M) & = - (bold(f), bold(v)_M) & quad forall bold(v)_M in bold(U)_M.
  )
$

这里引入罚项并不改变原连续问题的解：对于满足齐次位移边界条件的精确解
$bold(u)$，有 $bold(u)|_(partial Omega) = 0$，因此
$
  c(bold(u), bold(v)) = 0, quad forall bold(v) in bold(U).
$
因此，罚项只是在离散层面上弱化地约束边界条件，使数值解在
$partial Omega$ 上尽可能逼近零；当 $lambda_"bc"$ 取足够大时，这种约束会更强，从而更好地恢复原问题的边界条件。

== 从变分方程到代数方程

取测试函数为同一组基函数：
$
  bold(tau)_M = xi^(bold(sigma))_n bold(E)_beta, quad 0 <= n <= M, 1 <= beta <= 6, \
  bold(v)_M = xi^(bold(u))_n bold(e)_j, quad 0 <= n <= M, 1 <= j <= 3.
$

定义矩阵块与离散载荷向量：
$
  bold(A)_((n, beta), (m, alpha)) & := a(xi^(bold(sigma))_n bold(E)_beta, xi^(bold(sigma))_m bold(E)_alpha), \
      bold(B)_((n, beta), (m, i)) & := b(xi^(bold(sigma))_n bold(E)_beta, xi^(bold(u))_m bold(e)_i), \
         bold(C)_((n, j), (m, i)) & := c(xi^(bold(u))_n bold(e)_j, xi^(bold(u))_m bold(e)_i), \
                 bold(F)_((n, j)) & := (bold(f), xi^(bold(u))_n bold(e)_j).
$

对任意固定的 $(n, beta)$，取测试函数 $bold(tau)_M = xi^(bold(sigma))_n bold(E)_beta$。第一条离散变分方程为
$
  0 = a(bold(sigma)_M, xi^(bold(sigma))_n bold(E)_beta) + b(xi^(bold(sigma))_n bold(E)_beta, bold(u)_M).
$
将系数展开式
$ bold(sigma)_M = sum_(m=0)^M sum_(alpha=1)^6 s_(m, alpha) xi^(bold(sigma))_m bold(E)_alpha $
与
$ bold(u)_M = sum_(m=0)^M sum_(i=1)^3 u_(m, i) xi^(bold(u))_m bold(e)_i $
代入，并利用双线性性；同时注意在线弹性常用对称性假设下 $a(bold(sigma), bold(tau)) = a(bold(tau), bold(sigma))$，得到
$
  0
  = sum_(m=0)^M sum_(alpha=1)^6 s_(m, alpha) a(xi^(bold(sigma))_n bold(E)_beta, xi^(bold(sigma))_m bold(E)_alpha)
  + sum_(m=0)^M sum_(i=1)^3 u_(m, i) b(xi^(bold(sigma))_n bold(E)_beta, xi^(bold(u))_m bold(e)_i).
$
将上述两项分别识别为 $bold(A), bold(B)$ 的矩阵元素，即
$
  sum_(m=0)^M sum_(alpha=1)^6 bold(A)_((n, beta), (m, alpha)) s_(m, alpha)
  + sum_(m=0)^M sum_(i=1)^3 bold(B)_((n, beta), (m, i)) u_(m, i) = 0.
$
这条方程正对应于 $bold(A) bold(s) + bold(B) bold(u) = 0$ 的第 $(n, beta)$ 个分量。

同理，对任意固定的 $(n, j)$，取测试函数 $bold(v)_M = xi^(bold(u))_n bold(e)_j$。第二条离散变分方程为
$
  0 = b(bold(sigma)_M, xi^(bold(u))_n bold(e)_j) + c(bold(u)_M, xi^(bold(u))_n bold(e)_j) + (bold(f), xi^(bold(u))_n bold(e)_j).
$
代入 $bold(sigma)_M$ 与 $bold(u)_M$ 并展开：
$
  0
  = sum_(m=0)^M sum_(alpha=1)^6 s_(m, alpha) b(xi^(bold(sigma))_m bold(E)_alpha, xi^(bold(u))_n bold(e)_j)
  + sum_(m=0)^M sum_(i=1)^3 u_(m, i) c(xi^(bold(u))_m bold(e)_i, xi^(bold(u))_n bold(e)_j)
  + bold(F)_((n, j)).
$
注意
$
  b(xi^(bold(sigma))_m bold(E)_alpha, xi^(bold(u))_n bold(e)_j) = bold(B)_((m, alpha), (n, j)),
$
以及在 $c$ 的对称性下
$
  c(xi^(bold(u))_m bold(e)_i, xi^(bold(u))_n bold(e)_j) = bold(C)_((n, j), (m, i)),
$
因此这条方程正对应于
$
  bold(B)^T bold(s) + bold(C) bold(u) + bold(F) = 0
$
的第 $(n, j)$ 个分量。

将 $bold(A), bold(B), bold(C)$ 视为 $(M+1) times (M+1)$ 块矩阵（块行由 $n$，块列由 $m$ 索引），$bold(F)$ 视为 $(M+1) times 1$ 块行向量，其堆叠形式为
$
  bold(A) =
  mat(
    bold(A)_(0, 0), dots.h, bold(A)_(0, M);
    dots.v, dots.down, dots.v;
    bold(A)_(M, 0), dots.h, bold(A)_(M, M)
  ), quad
  bold(B) =
  mat(
    bold(B)_(0, 0), dots.h, bold(B)_(0, M);
    dots.v, dots.down, dots.v;
    bold(B)_(M, 0), dots.h, bold(B)_(M, M)
  ), quad
  bold(C) =
  mat(
    bold(C)_(0, 0), dots.h, bold(C)_(0, M);
    dots.v, dots.down, dots.v;
    bold(C)_(M, 0), dots.h, bold(C)_(M, M)
  ), quad
  bold(F) =
  mat(
    bold(F)_0;
    dots.v;
    bold(F)_M
  ).
$
其中块矩阵满足（块内行指标为 $beta$，列指标为 $alpha$ 或 $i$）：
$
  bold(A)_(n, m) in RR^(6 times 6), & quad (bold(A)_(n, m))_(beta, alpha) = bold(A)_((n, beta), (m, alpha)), \
  bold(B)_(n, m) in RR^(6 times 3), & quad (bold(B)_(n, m))_(beta, i) = bold(B)_((n, beta), (m, i)), \
  bold(C)_(n, m) in RR^(3 times 3), & quad (bold(C)_(n, m))_(j, i) = bold(C)_((n, j), (m, i)), \
                 bold(F)_n in RR^3, & quad (bold(F)_n)_j = bold(F)_((n, j)).
$
为强调块结构，将系数按特征下标分块：
$
  bold(s)_m & := (s_(m, 1), ..., s_(m, 6))^T in RR^6, \
  bold(u)_m & := (u_(m, 1), ..., u_(m, 3))^T in RR^3, \
$
则 $bold(s) = (bold(s)_0, ..., bold(s)_M)^T$，$bold(u) = (bold(u)_0, ..., bold(u)_M)^T$。

在上述块记号下，两条方程也可按块写为
$
    sum_(m=0)^M bold(A)_(n, m) bold(s)_m + sum_(m=0)^M bold(B)_(n, m) bold(u)_m & = 0, \
  sum_(m=0)^M bold(B)_(m, n)^T bold(s)_m + sum_(m=0)^M bold(C)_(n, m) bold(u)_m & = -bold(F)_n, quad 0 <= n <= M.
$

将系数展开代入两条变分方程，可得线性系统：
$
  mat(bold(A), bold(B); bold(B)^T, bold(C)) mat(bold(s); bold(u)) = mat(0; -bold(F)).
$

其中
$
  bold(A) in RR^(6(M+1) times 6(M+1)), quad
  bold(B) in RR^(6(M+1) times 3(M+1)), quad
  bold(C) in RR^(3(M+1) times 3(M+1)), quad
  bold(F) in RR^(3(M+1)).
$

== $bold(A)$ 块元素

由定义
$
  bold(A)_((n, beta), (m, alpha))
  := a(xi^(bold(sigma))_n bold(E)_beta, xi^(bold(sigma))_m bold(E)_alpha)
  = integral_Omega (bold(cal(A)) : (xi^(bold(sigma))_n bold(E)_beta)) : (xi^(bold(sigma))_m bold(E)_alpha) dif bold(x).
$
由于 $xi^(bold(sigma))_n, xi^(bold(sigma))_m$ 为标量函数且 $bold(E)_alpha, bold(E)_beta$ 为常矩阵，有
$
  bold(A)_((n, beta), (m, alpha))
  = integral_Omega xi^(bold(sigma))_n (bold(x)) xi^(bold(sigma))_m (bold(x)) ((bold(cal(A))(bold(x)) : bold(E)_beta) : bold(E)_alpha) dif bold(x).
$
若将收缩写成坐标分量，则
$
  ((bold(cal(A)) : bold(E)_beta) : bold(E)_alpha)
  = sum_(i=1)^3 sum_(j=1)^3 sum_(k=1)^3 sum_(l=1)^3 bold(cal(A))_(i j k l) (bold(E)_beta)_(k l) (bold(E)_alpha)_(i j),
$
因此
$
  bold(A)_((n, beta), (m, alpha))
  = integral_Omega xi^(bold(sigma))_n xi^(bold(sigma))_m
  sum_(i=1)^3 sum_(j=1)^3 sum_(k=1)^3 sum_(l=1)^3 bold(cal(A))_(i j k l) (bold(E)_beta)_(k l) (bold(E)_alpha)_(i j)
  dif bold(x).
$

== $bold(B)$ 块元素

由定义
$
  bold(B)_((n, beta), (m, i))
  := b(xi^(bold(sigma))_n bold(E)_beta, xi^(bold(u))_m bold(e)_i)
  = integral_Omega (nabla dot (xi^(bold(sigma))_n bold(E)_beta)) dot (xi^(bold(u))_m bold(e)_i) dif bold(x).
$
利用张量散度的分量定义 $(nabla dot bold(tau))_p = tau_(p k, k)$，令 $bold(tau) = xi^(bold(sigma))_n bold(E)_beta$，则
$
  (nabla dot (xi^(bold(sigma))_n bold(E)_beta))_i
  = (xi^(bold(sigma))_n (bold(E)_beta)_(i k))_(,k)
  = (bold(E)_beta)_(i k) partial_k xi^(bold(sigma))_n,
$
即向量形式
$
  nabla dot (xi^(bold(sigma))_n bold(E)_beta) = bold(E)_beta nabla xi^(bold(sigma))_n.
$
因此
$
  bold(B)_((n, beta), (m, i)) = integral_Omega xi^(bold(u))_m (bold(E)_beta nabla xi^(bold(sigma))_n)_i dif bold(x)
  = integral_Omega xi^(bold(u))_m sum_(k=1)^3 (bold(E)_beta)_(i k) partial_k xi^(bold(sigma))_n dif bold(x).
$

对 6 个对称基，$bold(E)_beta nabla xi^(bold(sigma))_n$ 可完全写成梯度分量的线性组合。记
$
  nabla xi^(bold(sigma))_n = (partial_1 xi^(bold(sigma))_n, partial_2 xi^(bold(sigma))_n, partial_3 xi^(bold(sigma))_n)^T,
$
则
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

而对于应力特征函数 $xi^(bold(sigma))_n (bold(x)) = sigma.alt(bold(w)_n^T bold(x) + b_n)$（$n >= 1$），其梯度可直接计算为
$
  nabla xi^(bold(sigma))_n (bold(x)) = sigma.alt'(bold(w)_n^T bold(x) + b_n) bold(w)_n.
$
若采用重参数化 $bold(w)_n = gamma bold(a)_n, b_n = gamma r_n$，则等价地
$
  nabla xi^(bold(sigma))_n (bold(x)) = sigma.alt'(gamma (bold(a)_n^T bold(x) + r_n)) gamma bold(a)_n.
$
特别地 $xi^(bold(sigma))_0 = 1$，故 $nabla xi^(bold(sigma))_0 = 0$，从而所有以 $n=0$ 为测试函数的 $bold(B)$ 行元素均为 $0$。

== $bold(C)$ 块元素

由定义
$
  bold(C)_((n, j), (m, i))
  := c(xi^(bold(u))_n bold(e)_j, xi^(bold(u))_m bold(e)_i)
  = lambda_"bc" integral_(partial Omega) (xi^(bold(u))_n bold(e)_j) dot (xi^(bold(u))_m bold(e)_i) dif s.
$
由于 $bold(e)_j dot bold(e)_i = delta_(j i)$，故
$
  bold(C)_((n, j), (m, i))
  = lambda_"bc" integral_(partial Omega) xi^(bold(u))_n (bold(x)) xi^(bold(u))_m (bold(x)) delta_(j i) dif s.
$
因此 $bold(C)$ 在分量方向上是块对角结构：当 $j != i$ 时矩阵元为 $0$，当 $j = i$ 时三条对角分量共享同一个边界 Gram 积分。

== $bold(F)$ 载荷向量元素

由定义
$
  bold(F)_((n, j))
  := (bold(f), xi^(bold(u))_n bold(e)_j)
  = integral_Omega bold(f) dot (xi^(bold(u))_n bold(e)_j) dif bold(x)
  = integral_Omega f_j (bold(x)) xi^(bold(u))_n (bold(x)) dif bold(x).
$

== 通用数值求积形式

在实现中，通常用一组内部采样点 $ {bold(x)_q}_(q=1)^(Q_"int") subset Omega $ 及权重 ${w_q}_(q=1)^(Q_"int")$ 近似体积分：
$
  integral_Omega g(bold(x)) dif bold(x) approx sum_(q=1)^(Q_"int") w_q g(bold(x)_q).
$
对边界项则采用一组边界采样点 $ {hat(bold(x))_r}_(r=1)^(Q_"bc") subset partial Omega $ 及权重 ${omega_r}_(r=1)^(Q_"bc")$：
$
  integral_(partial Omega) h(bold(x)) dif s approx sum_(r=1)^(Q_"bc") omega_r h(hat(bold(x))_r).
$
于是矩阵块元素可按如下公式逐元素计算：
$
  cases(
    bold(A)_((n, beta), (m, alpha))
    &approx sum_(q=1)^(Q_"int") w_q xi^(bold(sigma))_n (bold(x)_q) xi^(bold(sigma))_m (bold(x)_q) ((bold(cal(A))(bold(x)_q) : bold(E)_beta) : bold(E)_alpha),
    bold(B)_((n, beta), (m, i)) &approx sum_(q=1)^(Q_"int") w_q xi^(bold(u))_m (bold(x)_q) (bold(E)_beta nabla xi^(bold(sigma))_n (bold(x)_q))_i,
    bold(C)_((n, j), (m, i)) &approx lambda_"bc" sum_(r=1)^(Q_"bc") omega_r xi^(bold(u))_n (hat(bold(x))_r) xi^(bold(u))_m (hat(bold(x))_r) delta_(j i),
    bold(F)_((n, j)) & approx sum_(q=1)^(Q_"int") w_q f_j (bold(x)_q) xi^(bold(u))_n (bold(x)_q).
  )
$

= 强形式残差最小二乘

除了从 Hellinger-Reissner 泛函的离散鞍点条件出发，也可以直接将近似解代回线弹性强形式，并通过最小化残差来求解系数。这里仍沿用前文的随机特征空间，但不再通过测试函数构造弱形式，而是直接最小化强形式残差。于是近似解仍写为
$
  bold(sigma)_M = sum_(m=0)^M sum_(alpha=1)^6 s_(m, alpha) xi^(bold(sigma))_m bold(E)_alpha, quad
  bold(u)_M = sum_(m=0)^M sum_(i=1)^3 u_(m, i) xi^(bold(u))_m bold(e)_i.
$

== 系数展开与强形式残差

对任意 $bold(x) in Omega$，位移应变可直接写成
$
  (bold(epsilon)(bold(u)_M)(bold(x)))_(j k)
  = 1/2 sum_(m=0)^M sum_(i=1)^3 u_(m, i)
  (
    delta_(i k) partial_j xi^(bold(u))_m (bold(x))
    + delta_(i j) partial_k xi^(bold(u))_m (bold(x))
  ), quad 1 <= j, k <= 3.
$
同理，由于 $bold(E)_alpha$ 是常矩阵，应力散度满足
$
  nabla dot bold(sigma)_M (bold(x))
  = sum_(m=0)^M sum_(alpha=1)^6 s_(m, alpha) bold(E)_alpha nabla xi^(bold(sigma))_m (bold(x)).
$
特别地，$xi^(bold(sigma))_0 = xi^(bold(u))_0 = 1$，因此 $nabla xi^(bold(sigma))_0 = nabla xi^(bold(u))_0 = 0$，所有涉及导数的项都只由 $m >= 1$ 的特征贡献。

按照强形式方程组，定义三类残差：
$
  bold(r)_"c" (bold(x)) & := bold(cal(A))(bold(x)) : bold(sigma)_M (bold(x)) - bold(epsilon)(bold(u)_M) (bold(x)) in SS, \
  bold(r)_"e" (bold(x)) & := nabla dot bold(sigma)_M (bold(x)) + bold(f)(bold(x)) in RR^3, \
  bold(r)_"b" (bold(x)) & := bold(u)_M (bold(x)) in RR^3.
$
其中 $bold(r)_"c"$ 是本构残差，$bold(r)_"e"$ 是平衡残差，$bold(r)_"b"$ 是边界残差。

== 离散残差与最小二乘系统

取内部采样点及权重
$
  {bold(x)_q}_(q=1)^(Q_"int") subset Omega, quad {w_q}_(q=1)^(Q_"int"),
$
并取边界采样点及权重
$
  {hat(bold(x))_r}_(r=1)^(Q_"bc") subset partial Omega, quad {omega_r}_(r=1)^(Q_"bc").
$
用这些配点近似积分后，得到离散损失
$
  L_"strong" (bold(s), bold(u))
  := 1/2 sum_(q=1)^(Q_"int") w_q
  (
    norm(bold(r)_"c" (bold(x)_q))_"F"^2
    + norm(bold(r)_"e" (bold(x)_q))_2^2
  )
  + lambda_"bc"/2 sum_(r=1)^(Q_"bc") omega_r norm(bold(r)_"b" (hat(bold(x))_r))_2^2.
$

// = PINN

// PINN（Physics-Informed Neural Networks）与上一节使用相同的强形式残差，但不再固定特征后求解线性系数，而是直接训练一个可微神经网络来近似应力场与位移场。这里采用混合形式参数化：
// $
//   cal(N)_theta : Omega -> SS times RR^3, quad
//   cal(N)_theta (bold(x)) = (bold(sigma)_theta (bold(x)), bold(u)_theta (bold(x))).
// $
// 为了保证应力对称性，令网络输出 6 个标量网络 $sigma_(theta, alpha) (bold(x))$，再利用对称基重构：
// $
//   bold(sigma)_theta (bold(x)) = sum_(alpha=1)^6 sigma_(theta, alpha) (bold(x)) bold(E)_alpha.
// $

// == 混合 PINN 的残差与损失函数

// 对任意配点 $bold(x) in Omega$，定义残差
// $
//   bold(r)_"c" (bold(x); theta) &:= bold(cal(A))(bold(x)) : bold(sigma)_theta (bold(x)) - bold(epsilon)(bold(u)_theta) (bold(x)), \
//   bold(r)_"e" (bold(x); theta) &:= nabla dot bold(sigma)_theta (bold(x)) + bold(f)(bold(x)), \
//   bold(r)_"b" (bold(x); theta) &:= bold(u)_theta (bold(x)), quad bold(x) in partial Omega.
// $
// 于是标准 PINN 的离散损失可写为
// $
//   L_"PINN" (theta)
//   := lambda_"c"/Q_"int" sum_(q=1)^(Q_"int") norm(bold(r)_"c" (bold(x)_q; theta))_"F"^2
//   + lambda_"e"/Q_"int" sum_(q=1)^(Q_"int") norm(bold(r)_"e" (bold(x)_q; theta))_2^2
//   + lambda_"bc"/Q_"bc" sum_(r=1)^(Q_"bc") norm(bold(r)_"b" (hat(bold(x))_r; theta))_2^2.
// $
// 若无特别说明，通常可先取 $lambda_"c" = lambda_"e" = 1$，再将 $lambda_"bc" > 0$ 作为边界罚参数调节。

// == 训练算法

// 在第 $k$ 轮迭代中，先在内部采样点 ${bold(x)_q}_(q=1)^(Q_"int")$ 和边界采样点 ${hat(bold(x))_r}_(r=1)^(Q_"bc")$ 上前向计算网络输出，再利用自动微分得到 $bold(epsilon)(bold(u)_theta)$ 与 $nabla dot bold(sigma)_theta$，从而组装损失 $L_"PINN" (theta)$。随后用一阶优化器更新网络参数，例如采用 Adam：
// $
//   theta^(k+1) = theta^(k) - eta_k nabla_theta L_"PINN" (theta^(k)).
// $
// 重复上述过程直到损失或残差收敛，即得到 PINN 近似解 $(bold(sigma)_theta, bold(u)_theta)$。

= 数值实验

== 实验设置

本节采用制造解基准检验前述三类固定特征方法在三维线弹性问题上的精度与效率。精确位移场选为
$
  bold(u)_"ex" (bold(x)) =
  mat(
    sin(pi x_1) sin(pi x_2) sin(pi x_3);
    sin(2 pi x_1) sin(pi x_2) sin(pi x_3);
    sin(pi x_1) sin(2 pi x_2) sin(pi x_3)
  ),
$
对应应力场由各向同性本构关系计算，体力项由 $bold(f)_"ex" = -nabla dot bold(sigma)(bold(u)_"ex")$ 通过自动微分生成。由于上述位移场在 $partial Omega$ 上恒为零，因此齐次位移边界条件能够被精确满足。除非另有说明，各算法共享同一组采样点、测试点和随机种子。主实验的公共配置汇总于 @tb:exp-setup。

#figure(
  three-line-table(
    columns: 2,
    align: (left, left),
  )[
    | 参数                  | 取值                         |
    |:---------------------|:-----------------------------|
    | 计算域                | $Omega = [0, 1]^3$           |
    | 杨氏模量              | $E = 1.0$                    |
    | 泊松比                | $nu = 0.3$                   |
    | 内部采样点数           | $Q_"int" = (2^6)^3 = 262144$ |
    | 边界采样点数           | $Q_"bc" = 6 (2^5)^2 = 6144$  |
    | 测试点数              | $Q_"test" = (2^5)^3 = 32768$ |
    | 特征数                | $M_bold(s) = M_bold(u) = 300$            |
    | 形状参数              | $gamma_s = gamma_u = 2.0$    |
    | 弱式方法罚参数         | $lambda_"bc" = 1.0$          |
    | 强式方法罚参数         | $lambda_"bc" = 10.0$         |
  ],
  caption: [主实验公共设置],
)<tb:exp-setup>

比较的方法包括 Projection、Weak (Eigh)、Weak (Lstsq)、Strong (Eigh) 与 Strong (Lstsq)。其中 Eigh 表示对对称线性系统采用特征值分解并做相对阈值截断，Lstsq 则调用通用最小二乘求解器。
- Projection 是在固定特征空间上分别对精确位移场与精确应力场做最小二乘投影
- Weak 两种算法通过离散 Hellinger-Reissner 系统求解系数
- Strong 两种算法则通过强形式残差最小二乘得到法方程

本文报告位移场和应力场在测试集上的相对 $L^2$ 误差。记测试点为
${bold(x)_q}_(q=1)^(Q_"test")$，则位移误差定义为
$
  e_bold(u) =
  frac(
    sqrt(frac(1, Q_"test") sum_(q=1)^(Q_"test") norm(bold(u)_M (bold(x)_q) - bold(u)_"ex" (bold(x)_q))_2^2),
    sqrt(frac(1, Q_"test") sum_(q=1)^(Q_"test") norm(bold(u)_"ex" (bold(x)_q))_2^2)
  ),
$
应力误差则在 Voigt 记号下采用权重
$bold(w) = (1, 1, 1, 2, 2, 2)^T$：
$
  e_bold(sigma) =
  frac(
    sqrt(frac(1, Q_"test") sum_(q=1)^(Q_"test") bold(w)^T (bold(sigma)_M (bold(x)_q) - bold(sigma)_"ex" (bold(x)_q))^2),
    sqrt(frac(1, Q_"test") sum_(q=1)^(Q_"test") bold(w)^T (bold(sigma)_"ex" (bold(x)_q))^2)
  ).
$

== 实验结果

主实验结果如 @tb:main-results 所示。该组比较固定
$M_bold(s) = M_bold(u) = 300$，只考察不同离散求解策略在同一随机特征空间预算下的表现。

#figure(
  three-line-table(
    columns: 4,
    align: (left, right, right, right),
  )[
    |        算法         |   位移误差  |   应力误差   |  Time(s) |
    |:-------------------|-----------:|-----------:|---------:|
    | Projection         |   1.65e-03 |   2.98e-03 |     6.40 |
    | Weak (Eigh)        |   1.10e-02 |   1.01e-02 |     0.62 |
    | Weak (Lstsq)       |   3.92e-02 |   3.85e-02 |     0.19 |
    | Strong (Eigh)      |   3.23e-03 |   1.40e-02 |     0.59 |
    | Strong (Lstsq)     |   1.15e-01 |   7.61e-01 |     0.19 |
  ],
  caption: [主实验结果（$M = 300$）],
)<tb:main-results>

#figure(
  image("/public/images/linear-elasticity-3d/l2-error-summary.png"),
  caption: [主实验结果（$M = 300$）],
)

从结果看，Projection 在位移和应力两项指标上都给出了最小误差，可视为该特征空间逼近能力的上界。

对于随机特征空间方法，Eigh 版本明显优于对应的 Lstsq 版本：
- Eigh 版本中，两个算法在几乎相同的运行时间下，Strong (Eigh) 给出了更小的位移误差，Weak (Eigh) 则在应力误差上略优。
- Lstsq 版本虽然更快，但精度与稳定性明显变差，尤其 Strong (Lstsq) 在应力预测上出现了显著退化。

这启发我们，对该类由固定特征诱导的线性系统，保留对称结构并进行谱截断更有利于获得稳定解。


== 消融实验

=== 特征数量 $M$ 的影响

为考察特征空间容量对近似质量的影响，分别取
$M_bold(s) = M_bold(u) = M in {200, 400, 600, 800, 1000}$，其余材料参数、采样点和随机种子均保持不变。结果如 @tb:ablation-M 和 @fig:ablation-M 所示。

#figure(
  three-line-table(
    columns: 5,
  )[
    |    $M$ |       算法        |     位移误差  |    应力误差    |  Time(s) |
    |-------:|:-----------------|-------------:|-------------:|---------:|
    |    200 | Projection       |     6.01e-03 |     1.02e-02 |     3.62 |
    |    200 | Weak (Eigh)      |     4.33e-02 |     3.87e-02 |     0.29 |
    |    200 | Weak (Lstsq)     |     1.37e-01 |     1.29e-01 |     0.07 |
    |    200 | Strong (Eigh)    |     1.03e-02 |     3.73e-02 |     0.20 |
    |    200 | Strong (Lstsq)   |     9.71e-03 |     3.38e-02 |     0.07 |
    |    400 | Projection       |     6.62e-04 |     9.53e-04 |     8.85 |
    |    400 | Weak (Eigh)      |     4.76e-03 |     4.36e-03 |     1.31 |
    |    400 | Weak (Lstsq)     |     5.27e-02 |     5.37e-02 |     0.41 |
    |    400 | Strong (Eigh)    |     1.75e-03 |     7.38e-03 |     1.32 |
    |    400 | Strong (Lstsq)   |     4.34e-03 |     2.92e-02 |     0.41 |
    |    600 | Projection       |     1.13e-04 |     1.69e-04 |    20.42 |
    |    600 | Weak (Eigh)      |     1.41e-03 |     1.04e-03 |     3.69 |
    |    600 | Weak (Lstsq)     |     3.96e-02 |     3.91e-02 |     1.20 |
    |    600 | Strong (Eigh)    |     5.76e-04 |     2.76e-03 |     3.70 |
    |    600 | Strong (Lstsq)   |     1.06e-02 |     7.76e-02 |     1.21 |
    |    800 | Projection       |     2.76e-05 |     4.33e-05 |   232.34 |
    |    800 | Weak (Eigh)      |     5.88e-04 |     4.69e-04 |     8.01 |
    |    800 | Weak (Lstsq)     |     3.59e-03 |     3.94e-03 |     2.76 |
    |    800 | Strong (Eigh)    |     2.71e-04 |     1.25e-03 |     7.99 |
    |    800 | Strong (Lstsq)   |     5.53e-03 |     4.44e-02 |     2.68 |
    |   1000 | Projection       |     9.19e-06 |     1.42e-05 |   464.09 |
    |   1000 | Weak (Eigh)      |     4.74e-04 |     3.83e-04 |    14.95 |
    |   1000 | Weak (Lstsq)     |     1.50e-02 |     1.83e-02 |     5.14 |
    |   1000 | Strong (Eigh)    |     2.13e-04 |     9.94e-04 |    14.95 |
    |   1000 | Strong (Lstsq)   |     6.99e-03 |     5.99e-02 |     5.07 |
  ],
  caption: [特征数量 $M$ 消融实验：误差随 $M$ 的变化],
) <tb:ablation-M>

#figure(
  image("/public/images/linear-elasticity-3d/ablation/M/ablation-M.png"),
  caption: [特征数量 $M$ 消融实验：误差随 $M$ 的变化],
) <fig:ablation-M>

总体来看，随着 $M$ 增大，Projection 以及两种 Eigh 方法的误差均得到下降，说明该制造解能够被较大的随机特征空间更充分地表示。其中

- Projection 的精度提升最明显，但时间代价增长也最剧烈。
- Weak (Eigh) 与 Strong (Eigh) 精度虽在不断提升，但是提升速度逐渐放缓，更多的基并未充分体现更强的逼近能力。
- Weak (Lstsq) 与 Strong (Lstsq)的误差随 $M$ 的变化呈现明显非单调性，表明在特征维度升高后，通用最小二乘求解对病态性更加敏感。

