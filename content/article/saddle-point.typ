#import "/typ/templates/blog.typ": *
#show: main.with(
  title: "鞍点问题",
  author: "summer",
  desc: [研究鞍点问题的数值方法],
  date: "2026-02-14",
  tags: (
    blog-tags.machine-learning,
    blog-tags.pde,
  ),
  show-outline: true,
)

= 问题描述

设弹性体占据空间区域 $Omega subset RR^3$，其边界为 $partial Omega$。位移 $bold(u)$ 和应力 $bold(sigma)$ 满足线弹性方程：
$
  cases(
    bold(cal(A)) : bold(sigma) - bold(epsilon)(bold(u)) & = 0 & quad "in" Omega,
    nabla dot bold(sigma) + bold(f) & = 0 & quad "in" Omega,
    bold(u) & = 0 & quad "on" partial Omega.
  )
$
其中 $bold(epsilon)(bold(u)) = 1/2(nabla bold(u) + (nabla bold(u))^T)$ 是应变张量，$bold(f)$ 是体力，$bold(cal(A))$ 为柔度张量。

定义如下双线性形式 $a: bold(Sigma) times bold(Sigma) -> RR$ 与 $b: bold(Sigma) times bold(U) -> RR$：
$
  a(bold(sigma), bold(tau)) & := integral_Omega (bold(cal(A)) : bold(sigma)) : bold(tau) dif bold(x), \
      b(bold(tau), bold(v)) & := integral_Omega (nabla dot bold(tau)) dot bold(v) dif bold(x).
$
其中，函数空间定义如下：
$
  bold(Sigma) &:= bold(H)(div, Omega; SS) := { bold(tau) in (L^2(Omega))^(3 times 3) : bold(tau)=bold(tau)^T, nabla dot bold(tau) in (L^2(Omega))^3 }, \
  bold(U) &:= (L^2(Omega))^3.
$

考虑 Hellinger-Reissner 泛函 $Pi: bold(Sigma) times bold(U) -> RR$ ：
$
  Pi(bold(tau), bold(v)) := 1/2 a(bold(tau), bold(tau)) + b(bold(tau), bold(v)) + (bold(f), bold(v)).
$
则 Hellinger-Reissner 泛函的鞍点 $(bold(sigma), bold(u))$ 是线弹性方程的解，其中鞍点满足如下最小最大关系：
$
  Pi(bold(sigma), bold(u)) = min_(bold(tau) in bold(Sigma)) max_(bold(v) in bold(U)) Pi(bold(tau), bold(v)).
$

= 神经特征空间

考虑一个单隐层全连接神经网络：
$
  phi := alpha_0 + sum_(m=1)^M alpha_m sigma.alt(bold(w)_m^T bold(x) + b_m),
$
其中 $sigma.alt$ 是激活函数，$M$ 是神经元数量，$alpha_m in RR$ 是输出层权重，$bold(w)_m in RR^3$ 是输入层权重，$b_m in RR$ 是偏置项。记隐藏层神经元为
$
  xi_m (bold(x)) = sigma.alt(bold(w)_m^T bold(x) + b_m),
$
我们可将 $xi_m: RR^3 -> RR$ 视为一个特征函数，隐藏层神经元集 ${xi_m}_1^M$ 可视为 $RR^3$ 空间中的一组基。定义神经特征空间
$
  Xi := span{xi_0, xi_1, ..., xi_M },
$
其中 $xi_0 = 1$。因此，神经特征空间 $Xi$ 是由单隐层全连接神经网络的隐藏层神经元生成的函数空间。我们可以将 $Xi$ 视为一个近似空间，用于近似求解线弹性方程的解。

为使位移近似满足齐次 Dirichlet 边界条件 $bold(u) = 0$ on $partial Omega$，引入包络函数 $zeta: overline(Omega) -> RR$，满足 $zeta = 0$ on $partial Omega$ 且 $zeta > 0$ in $Omega$。定义位移特征
$
  xi^(bold(u))_m (bold(x)) := zeta(bold(x)) dot xi_m (bold(x)), quad m = 0, 1, ..., M.
$
对所有 $m$ 均有 $xi^(bold(u))_m = 0$ on $partial Omega$，因此以 ${xi^(bold(u))_m}$ 为基展开的位移在 $partial Omega$ 上自动为零。

本文采用如下重参数化 @zhang2024transnet 来生成 $(bold(w)_m, b_m)$：取全局形状参数 $gamma > 0$，并令
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


= 变分形式离散与线性鞍点系统


将 $bold(Sigma)$ 和 $bold(U)$ 分别近似为神经特征空间 $bold(Xi)$ 的张成空间，即
$
  bold(Sigma)_M := & span{ xi^(bold(sigma))_m bold(E)_alpha: alpha = 1, 2, ..., 6; 0 <= m <= M } subset bold(Sigma), \
      bold(U)_M := & span{ xi^(bold(u))_m bold(e)_i: m = 0, ..., M, i = 1, 2, 3 } subset bold(U),
$
其中 $bold(E)_(alpha)$ 是 $RR^(3 times 3)$ 中的对称单位矩阵（排列顺序为 $11, 22, 33, 12, 23, 13$），$bold(e)_i$ 是 $RR^3$ 的标准基向量。

线弹性方程的近似解 $(bold(sigma)_M, bold(u)_M) in bold(Sigma)_M times bold(U)_M$ 满足如下离散鞍点问题：
$
  Pi(bold(sigma)_M, bold(u)_M) = min_(bold(tau)_M in bold(Sigma)_M) max_(bold(v)_M in bold(U)_M) Pi(bold(tau)_M, bold(v)_M).
$
这意味着泛函在 $(bold(sigma)_M, bold(u)_M)$ 处关于任意方向 $(bold(tau)_M, bold(v)_M)$ 的一阶变分为零：
$
  cases(
    a(bold(sigma)_M, bold(tau)_M) + b(bold(tau)_M, bold(u)_M) & = 0 & quad forall bold(tau)_M in bold(Sigma)_M,
    b(bold(sigma)_M, bold(v)_M) + (bold(f), bold(v)_M) & = 0 & quad forall bold(v)_M in bold(U)_M.
  )
$

== 系数展开

将近似解在上述基上展开：
$
  bold(sigma)_M & = sum_(m=0)^M sum_(alpha=1)^6 s_(m, alpha) xi^(bold(sigma))_m bold(E)_alpha, \
      bold(u)_M & = sum_(m=0)^M sum_(i=1)^3 u_(m, i) xi^(bold(u))_m bold(e)_i.
$
记系数向量
$
  bold(s) & = (s_(0, 1), ..., s_(0, 6), s_(1, 1), ..., s_(M, 6))^T in RR^(6(M+1)), \
  bold(u) & = (u_(0, 1), ..., u_(0, 3), u_(1, 1), ..., u_(M, 3))^T in RR^(3(M+1)).
$

== 选取测试函数并组装矩阵

=== 从变分方程到代数方程

取测试函数为同一组基函数：
$
  bold(tau)_M = xi^(bold(sigma))_n bold(E)_beta, quad 0 <= n <= M, 1 <= beta <= 6, \
  bold(v)_M = xi^(bold(u))_n bold(e)_j, quad 0 <= n <= M, 1 <= j <= 3.
$

定义矩阵块与离散载荷向量：
$
  bold(A)_((n, beta), (m, alpha)) & := a(xi^(bold(sigma))_n bold(E)_beta, xi^(bold(sigma))_m bold(E)_alpha), \
      bold(B)_((n, beta), (m, i)) & := b(xi^(bold(sigma))_n bold(E)_beta, xi^(bold(u))_m bold(e)_i), \
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
  0 = b(bold(sigma)_M, xi^(bold(u))_n bold(e)_j) + (bold(f), xi^(bold(u))_n bold(e)_j).
$
代入 $bold(sigma)_M$ 并展开：
$
  0
  = sum_(m=0)^M sum_(alpha=1)^6 s_(m, alpha) b(xi^(bold(sigma))_m bold(E)_alpha, xi^(bold(u))_n bold(e)_j)
  + bold(F)_((n, j)).
$
注意 $b(xi^(bold(sigma))_m bold(E)_alpha, xi^(bold(u))_n bold(e)_j) = bold(B)_((m, alpha), (n, j))$，因此这条方程正对应于 $bold(B)^T bold(s) + bold(F) = 0$ 的第 $(n, j)$ 个分量。

将 $bold(A), bold(B)$ 视为 $(M+1) times (M+1)$ 块矩阵（块行由 $n$，块列由 $m$ 索引），$bold(F)$ 视为 $(M+1) times 1$ 块行向量，其堆叠形式为
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
  sum_(m=0)^M bold(A)_(n, m) bold(s)_m + sum_(m=0)^M bold(B)_(n, m) bold(u)_m = 0, \
  sum_(m=0)^M bold(B)_(m, n)^T bold(s)_m + bold(F)_n = 0, quad 0 <= n <= M.
$

将系数展开代入两条变分方程，可得线性系统
$
  bold(A) bold(s) + bold(B) bold(u) = 0, \
  bold(B)^T bold(s) + bold(F) = 0.
$

等价地写成块鞍点系统：
$
  mat(bold(A), bold(B); bold(B)^T, 0) mat(bold(s); bold(u)) = mat(0; -bold(F)).
$

其中
$
  bold(A) in RR^(6(M+1) times 6(M+1)), quad bold(B) in RR^(6(M+1) times 3(M+1)), quad bold(F) in RR^(3(M+1)).
$

=== $bold(A)$ 块元素：材料柔度张量与两基函数乘积

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
若将收缩写成坐标分量（Frobenius 内积），则
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

=== $bold(B)$ 块元素：散度只作用在测试基函数上

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

=== $bold(F)$ 载荷向量元素：体力分量在基上的投影

由定义
$
  bold(F)_((n, j))
  := (bold(f), xi^(bold(u))_n bold(e)_j)
  = integral_Omega bold(f) dot (xi^(bold(u))_n bold(e)_j) dif bold(x)
  = integral_Omega f_j (bold(x)) xi^(bold(u))_n (bold(x)) dif bold(x).
$

=== 通用数值求积形式

在实现中，通常用一组采样点 $ {bold(x)_q}_(q=1)^Q subset Omega $ 及权重 ${w_q}_(q=1)^Q$ 近似积分：
$
  integral_Omega g(bold(x)) dif bold(x) approx sum_(q=1)^Q w_q g(bold(x)_q).
$
于是矩阵块元素可按如下公式逐元素计算：
$
  cases(
    bold(A)_((n, beta), (m, alpha))
    &approx sum_(q=1)^Q w_q xi^(bold(sigma))_n (bold(x)_q) xi^(bold(sigma))_m (bold(x)_q) ((bold(cal(A))(bold(x)_q) : bold(E)_beta) : bold(E)_alpha),
    bold(B)_((n, beta), (m, i)) &approx sum_(q=1)^Q w_q xi^(bold(u))_m (bold(x)_q) (bold(E)_beta nabla xi^(bold(sigma))_n (bold(x)_q))_i,
    bold(F)_((n, j)) & approx sum_(q=1)^Q w_q f_j (bold(x)_q) xi^(bold(u))_n (bold(x)_q).
  )
$

= 交替梯度上升/下降法

Hellinger-Reissner 泛函在离散系数 $(bold(s), bold(u))$ 上可写成
$
  Pi(bold(s), bold(u)) := 1/2 bold(s)^T bold(A) bold(s) + bold(s)^T bold(B) bold(u) + bold(F)^T bold(u).
$
其梯度为
$
  nabla_(bold(s)) Pi = bold(A) bold(s) + bold(B) bold(u), \
  nabla_(bold(u)) Pi = bold(B)^T bold(s) + bold(F).
$
由于 $Pi$ 关于 $bold(u)$ 线性，严格的 $max_(bold(u)) Pi$ 在有限维上通常无界；因此在实现中不求精确解，而是将 $(bold(s), bold(u))$ 视为待优化参数，对离散泛函 $Pi$ 做交替的一阶更新。

在第 $k+1$ 轮迭代中，先固定 $bold(s)^k$ 对 $bold(u)$ 做一次梯度上升更新：
$
  bold(u)^(k+1) & = bold(u)^k + eta_bold(u)^"GDA" (bold(B)^T bold(s)^k + bold(F)).
$

再固定 $bold(u)^(k+1)$ 对 $bold(s)$ 做一次梯度下降更新：
$
  bold(s)^(k+1) & = bold(s)^k - eta_bold(s)^"GDA" (bold(A) bold(s)^k + bold(B) bold(u)^(k+1)).
$
这里 $eta_bold(u)^"GDA", eta_bold(s)^"GDA" > 0$ 为步长。上式写成最简单的交替梯度上升/下降形式；在数值实现中，可用 Adam 等自适应优化器替代固定步长。

= Uzawa 算法

Uzawa 算法通过先消去 $bold(s)$ 再更新 $bold(u)$ 来解耦块系统。给定 $bold(u)^(k)$，先令 $bold(s)^(k+1)$ 满足
$
  bold(A) bold(s)^(k+1) = -bold(B) bold(u)^(k),
$
这等价于求解关于 $bold(s)$ 的二次最小化问题
$
  min_(bold(s)) (1/2 bold(s)^T bold(A) bold(s) + bold(s)^T bold(B) bold(u)^(k)),
$
其一阶最优性条件正是 $bold(A) bold(s) + bold(B) bold(u)^(k) = 0$。

随后用第二块方程的残差作梯度上升更新
$
  bold(u)^(k+1) = bold(u)^(k) + eta_bold(u)^"Uzawa" (bold(B)^T bold(s)^(k+1) + bold(F)),
$
其中 $eta_bold(u)^"Uzawa" > 0$ 为步长。等价地，$bold(u)^(k+1)$ 是如下带近端项的最大化问题的解：
$
  max_(bold(u))
  [
    (bold(B)^T bold(s)^(k+1) + bold(F))^T bold(u)
    - 1/(2 eta_bold(u)^"Uzawa") (bold(u) - bold(u)^(k))^T (bold(u) - bold(u)^(k))
  ].
$
为了数值稳健性，在实际更新中用 $bold(A) + rho bold(I)$ 替代 $bold(A)$。具体为：在第 $k$ 轮循环中，更新
$
  (bold(A) + rho bold(I)) bold(s)^(k) & = -bold(B) bold(u)^(k), \
                        bold(u)^(k+1) & = bold(u)^(k) + eta_bold(u)^"Uzawa" (bold(B)^T bold(s)^(k) + bold(F)).
$
其中 $bold(I)$ 为单位阵，$rho >= 0$ 是用于数值稳健性的阻尼参数；当 $bold(A)$ 可能奇异或病态时可取 $rho > 0$。

= Arrow-Hurwicz 算法

Uzawa 算法每步需要解 $bold(A) bold(s) = -bold(B) bold(u)$，可看作先对 $bold(s)$ 做“精确消元”，再用残差更新 $bold(u)$。若不希望每步求解线性系统，可以将 $bold(s)$ 子问题视为二次最小化，并用一次（预条件）梯度下降近似其解，从而得到 Arrow-Hurwicz 型的同时更新。

由上一节梯度表达式
$
  nabla_(bold(s)) Pi = bold(A) bold(s) + bold(B) bold(u), \
  nabla_(bold(u)) Pi = bold(B)^T bold(s) + bold(F),
$
取预条件子 $bold(J), bold(K)$（通常取为对称正定矩阵或其近似），用一阶法对 $bold(s)$ 做下降、对 $bold(u)$ 做上升：
$
  bold(s)^(k+1) = bold(s)^(k) - eta_bold(s)^"AH" bold(J) [(bold(A) + rho bold(I)) bold(s)^(k) + bold(B) bold(u)^(k)], \
  bold(u)^(k+1) = bold(u)^(k) + eta_bold(u)^"AH" bold(K) (bold(B)^T bold(s)^(k+1) + bold(F)).
$
这里 $eta_bold(s)^"AH", eta_bold(u)^"AH" > 0$ 为学习率，$bold(I)$ 为单位阵，$rho >= 0$ 与上一节同义。当 $bold(A)$ 可能奇异或病态时可取 $rho > 0$。

直观上，这相当于对 Uzawa 的 $bold(s)$-消元作非精确求解：当 $bold(J) approx bold(A)^(-1)$ 且 $bold(s)$ 更新迭代到收敛时，可视为逼近 Uzawa 的“先消去 $bold(s)$ 再更新 $bold(u)$”。

= 数值实验

本节给出本文将要进行的 3D 数值实验设置，用于验证前述离散鞍点系统与三种迭代算法（GDA、Uzawa 和 Arrow-Hurwicz）的可实现性与收敛性。所有对比实验均采用相同的 3D 结构（应力 Voigt 6 分量 + 位移 3 分量），并在同一组采样点上组装 $bold(A), bold(B), bold(F)$ 以保证公平比较。
除特别说明外，下文实验均对应上一节的变分形式离散与线性鞍点系统，不采用前文强形式残差最小二乘中通过边界罚项处理齐次 Dirichlet 条件的做法。

== 实验设置

=== 方程与边界条件

考虑 3D 小变形各向同性线弹性模型。应变定义为
$
  bold(epsilon)(bold(u)) = 1/2(nabla bold(u) + (nabla bold(u))^T).
$
给定材料参数 $E, nu$，引入拉梅常数
$
  mu = E/(2(1+nu)), quad
  lambda = E nu/((1+nu)(1-2 nu)).
$
本构关系写为
$
  bold(sigma)(bold(u)) = bold(cal(C)) : bold(epsilon)(bold(u)) = 2 mu bold(epsilon)(bold(u)) + lambda tr(bold(epsilon)(bold(u))) bold(I),
$
其中 $bold(cal(C))$ 为刚度张量，$tr(bold(epsilon)) := bold(epsilon) : bold(I)$。平衡方程为
$
  -nabla dot bold(sigma)(bold(u)) = bold(f) quad "in" Omega,
$
并施加齐次 Dirichlet 边界条件
$
  bold(u) = 0 quad "on" partial Omega.
$

为与前文 Hellinger-Reissner 形式一致，取柔度张量 $bold(cal(A)) = bold(cal(C))^(-1)$ 使得 $bold(cal(A)):bold(sigma) = bold(epsilon)(bold(u))$，并沿用工程 Voigt 记号装配 $bold(A)$ 块。

=== 计算域与制造解

取计算域 $Omega = [0, 1]^3$。设精确位移为
$
  bold(u)_"ex" (bold(x))
  = mat(
    sin(pi x_1) sin(pi x_2) sin(pi x_3);
    sin(2 pi x_1) sin(pi x_2) sin(pi x_3);
    sin(pi x_1) sin(2 pi x_2) sin(pi x_3)
  ).
$
则 $bold(u)_"ex" = 0$ 在 $partial Omega$ 上成立。相应精确应力取
$
  bold(sigma)_"ex" = bold(sigma)(bold(u)_"ex"),
$
体力通过制造解定义为
$
  bold(f)_"ex" (bold(x)) = -nabla dot bold(sigma)_"ex" (bold(x)).
$
实现时将用自动微分或符号计算得到 $bold(sigma)_"ex", bold(f)_"ex"$，不在文中展开其冗长表达式。

=== 材料参数与柔度矩阵

选取常数材料参数
$
  E = 1, quad nu = 0.3.
$
在工程 Voigt 排列顺序 $(11, 22, 33, 12, 23, 13)$ 下，柔度矩阵 $bold(cal(A))$ 满足
$
  epsilon_alpha = Sigma_(beta = 1)^6 cal(A)_(alpha beta) sigma_beta.
$
取
$
  bold(cal(A)) = 1/E mat(
    1, -nu, -nu, 0, 0, 0;
    -nu, 1, -nu, 0, 0, 0;
    -nu, -nu, 1, 0, 0, 0;
    0, 0, 0, 2(1+nu), 0, 0;
    0, 0, 0, 0, 2(1+nu), 0;
    0, 0, 0, 0, 0, 2(1+nu)
  ).
$

=== 神经特征空间与离散未知量

采用前文定义的单隐层全连接随机特征函数，取激活函数 $sigma.alt = tanh$：
$
  xi_0 = 1, quad xi_m (bold(x)) = sigma.alt(bold(w)_m^T bold(x) + b_m), quad m = 1, 2, ..., M.
$
其中 $bold(w)_m in RR^3, b_m in RR$ 在实验开始时按照如下方式随机生成并固定：
$
  bold(w)_m = gamma bold(a)_m, quad b_m = gamma r_m.
$
固定形状参数 $gamma = 2.0$ 以控制特征函数的频率范围；$bold(a)_m = bold(X)_m \/ norm(bold(X)_m)_2$，其中 $bold(X)_m in RR^3$ 是从标准正态分布采样的随机向量，$r_m in RR$ 是从 $[0, 1]$ 均匀分布采样的随机数。

取包络函数为
$
  zeta(bold(x)) = x_1(1-x_1) dot x_2(1-x_2) dot x_3(1-x_3),
$
则 $zeta = 0$ on $partial [0, 1]^3$ 且 $zeta > 0$ in $(0, 1)^3$。注意 $xi^(bold(u))_0 = zeta$（不再是常数函数）。

根据离散应力和位移：
$
  bold(sigma)_M = sum_(m=0)^M sum_(alpha=1)^6 s_(m, alpha) xi^(bold(sigma))_m bold(E)_alpha, \
  bold(u)_M = sum_(m=0)^M sum_(i=1)^3 u_(m, i) xi^(bold(u))_m bold(e)_i.
$
从而得到离散鞍点系统
$
  mat(bold(A), bold(B); bold(B)^T, 0) mat(bold(s); bold(u)) = mat(0; -bold(F)).
$

=== 数值积分与数据划分

用均匀 Monte Carlo 采样近似积分。训练阶段在 $Omega$ 内采样 $Q_"int"$ 个点 ${bold(x)_q}_(q=1)^(Q_"int")$，取等权
$
  w_q = abs(Omega) / Q_"int" = 1 / Q_"int".
$
在该训练点集上一次性组装 $bold(A), bold(B), bold(F)$。另外独立采样 $Q_"test"$ 个测试点用于误差评估。

=== 算法对比设置

为避免离散 $bold(A)$ 病态带来的数值问题，统一采用轻微阻尼 $rho = 10^(-6)$；即在涉及求解 $bold(s)$ 的步骤中以 $bold(A) + rho bold(I)$ 替代 $bold(A)$。

#figure(
  three-line-table(
    columns: 2,
    align: (right, left),
  )[
    | 参数 | 取值 |
    |------|------|
    | 域 | $Omega = [0, 1]^3$ |
    | 边界条件 | 齐次 Dirichlet：$bold(u)=0$ on $partial Omega$ |
    | 材料 | 各向同性常系数：$E=1, nu=0.3$ |
    | 随机特征 | $tanh$ 激活函数，均匀神经元分布 @zhang2024transnet  |
    | 特征采样 | $bold(w)_m = gamma bold(a)_m$，$b_m = gamma r_m$ |
    | 训练点 | $Q_"int" = 64^3$ |
    | 测试点 | $Q_"test" = 32^3$ |
    | 阻尼 | $rho = 10^(-6)$ |
    | 初值 | $bold(s)^0 = 0, bold(u)^0 = 0$ |
    | 迭代 | $20000$ |
  ],
)

算法细节如下：

- *GDA*：采用交替梯度上升/下降更新；实现时使用 Adam 作为自适应一阶优化器，学习率 $eta_bold(u) = eta_bold(s) = 0.02$，动量参数 $bold(beta)^"Adam" = (0.9, 0.98)$，每轮各做 1 次更新。
- *Uzawa*：步长 $eta_bold(u)^"Uzawa"$ 通过 Schur 补矩阵 $bold(S)$ 的谱半径自适应选择。
- *Arrow-Hurwicz*: 步长 $eta_bold(s)^"AH"$、$eta_bold(u)^"AH"$ 分别通过 Jacobi 谱半径和 Schur 补矩阵 $bold(S)$ 的谱半径自适应选择；取预条件子 $bold(J) = [diag(bold(A) + rho bold(I))]^(-1)$，$bold(K) = bold(I)$。

=== 评价指标

- *KKT 残差*: 记
  $
    bold(r)_"c" = bold(A) bold(s) + bold(B) bold(u), \
    bold(r)_"e" = bold(B)^T bold(s) + bold(F),
  $
  记录 $norm(bold(r)_"c")_2$ 与 $norm(bold(r)_"e")_2$ 随迭代的变化。
- *相对 $L^2$ 误差*: 在测试点上用
  $
    norm(bold(u)_M - bold(u)_"ex")_(L^2(Omega))
    approx (abs(Omega)/Q_"test" sum_(q=1)^(Q_"test") abs(bold(u)_M (bold(x)_q) - bold(u)_"ex" (bold(x)_q))^2)^(1/2)
  $
  估计位移误差，并用同样方式估计应力误差（张量按 Frobenius 范数聚合）。报告相对误差
  $
    norm(bold(u)_M - bold(u)_"ex")_(L^2) / norm(bold(u)_"ex")_(L^2), quad
    norm(bold(sigma)_M - bold(sigma)_"ex")_(L^2) / norm(bold(sigma)_"ex")_(L^2).
  $
- *收敛成本*: 记录达到阈值 $norm(bold(r)_"c")_2 + norm(bold(r)_"e")_2 <= 10^(-6)$ 的迭代步数与壁钟时间。

== 实验结果

主实验结果如 @tb:main-results 所示。Projection 为分别在位移特征空间和应力特征空间上对精确解做最小二乘投影。

#figure(
  three-line-table(
    columns: 6,
    align: (left, right, right, right, right, right),
  )[
    | 算法 | $norm(bold(r)_"c")_2$ | $norm(bold(r)_"e")_2$ | 位移误差 | 应力误差 | 时间 (s) |
    |:---------------|-----------:|-----------:|-----------:|-----------:|---------:|
    | Projection     |   3.49e-06 |   1.35e-06 |   9.76e-06 |   2.98e-03 |     6.17 |
    | Eigh           |   2.27e-07 |   1.79e-07 |   1.78e-03 |   8.50e-03 |     0.73 |
    | Lstsq          |   1.37e-07 |   6.78e-09 |   4.55e-01 |   6.05e-02 |     0.21 |
    | GDA            |   7.47e+00 |   7.24e-02 |   1.50e-01 |   2.62e-01 |    11.40 |
    | Uzawa          |   1.97e-04 |   2.85e-04 |   2.06e-02 |   5.94e-02 |    18.06 |
    | Arrow-Hurwicz  |   1.38e-01 |   3.18e-03 |   6.44e-01 |   4.82e-01 |     6.92 |
  ],
  caption: [主实验结果（$M = 300$）],
) <tb:main-results>

#figure(
  image("/public/images/saddle-point/kkt-convergence.png"),
  caption: [KKT 残差收敛曲线],
) <fig:kkt-convergence>

#figure(
  image("/public/images/saddle-point/l2-error-convergence.png"),
  caption: [$L^2$ 相对误差收敛曲线],
) <fig:l2-convergence>

== 消融实验

=== 特征数量 $M$ 的影响

分别取 $M in {200, 400, 600, 800, 1000}$，各算法在相同随机种子下运行 $K = 20000$ 步。结果如 @tb:ablation-M 和 @fig:ablation-M 所示。

#figure(
  three-line-table(
    columns: 6,
    align: (right, left, right, right, right, right),
  )[
    |    $M$ |       算法      | $norm(bold(r)_"c")_2$ | $norm(bold(r)_"e")_2$ |  位移误差 |    应力误差    |
    |-------:|:---------------|-------------:|-------------:|-------------:|-------------:|
    |    200 | Projection     |   1.88e-05 |   1.60e-05 |     5.77e-05 |     1.02e-02 |
    |    200 | Eigh           |   6.57e-07 |   6.08e-07 |     8.48e-03 |     2.57e-02 |
    |    200 | Lstsq          |   5.56e-08 |   4.12e-09 |     2.92e+00 |     1.20e-01 |
    |    200 | GDA            |   4.30e+00 |   2.75e-02 |     1.88e-01 |     2.60e-01 |
    |    200 | Uzawa          |   2.69e-04 |   2.71e-04 |     2.30e-02 |     1.02e-01 |
    |    200 | Arrow-Hurwicz  |   1.12e-01 |   2.64e-03 |     6.59e-01 |     5.26e-01 |
    |    400 | Projection     |   2.17e-06 |   1.21e-07 |     4.42e-06 |     9.53e-04 |
    |    400 | Eigh           |   2.21e-07 |   1.43e-07 |     9.83e-04 |     6.34e-03 |
    |    400 | Lstsq          |   6.54e-07 |   4.11e-08 |     1.28e+00 |     3.74e-01 |
    |    400 | GDA            |   1.02e+01 |   1.37e-01 |     1.94e-01 |     3.03e-01 |
    |    400 | Uzawa          |   1.75e-04 |   3.29e-04 |     2.23e-02 |     5.17e-02 |
    |    400 | Arrow-Hurwicz  |   1.61e-01 |   3.69e-03 |     6.48e-01 |     5.03e-01 |
    |    600 | Projection     |   1.12e-06 |   1.30e-08 |     6.27e-07 |     1.69e-04 |
    |    600 | Eigh           |   1.19e-07 |   7.41e-08 |     3.98e-04 |     2.63e-03 |
    |    600 | Lstsq          |   2.47e-08 |   7.37e-10 |     2.14e-02 |     9.43e-03 |
    |    600 | GDA            |   1.97e+01 |   1.29e-01 |     2.00e-01 |     3.09e-01 |
    |    600 | Uzawa          |   1.37e-04 |   4.22e-04 |     2.49e-02 |     4.49e-02 |
    |    600 | Arrow-Hurwicz  |   1.91e-01 |   4.37e-03 |     6.10e-01 |     4.93e-01 |
    |    800 | Projection     |   1.31e-06 |   8.63e-09 |     1.24e-07 |     4.33e-05 |
    |    800 | Eigh           |   1.18e-07 |   7.42e-08 |     3.49e-04 |     2.15e-03 |
    |    800 | Lstsq          |   1.60e-08 |   7.15e-10 |     7.34e-03 |     5.01e-03 |
    |    800 | GDA            |   3.47e+01 |   2.86e-01 |     3.22e-01 |     4.65e-01 |
    |    800 | Uzawa          |   1.17e-04 |   5.18e-04 |     2.79e-02 |     4.59e-02 |
    |    800 | Arrow-Hurwicz  |   2.14e-01 |   4.94e-03 |     5.93e-01 |     4.85e-01 |
    |   1000 | Projection     |   1.43e-06 |   7.21e-10 |     2.59e-08 |     1.42e-05 |
    |   1000 | Eigh           |   1.21e-07 |   6.18e-08 |     2.88e-04 |     1.91e-03 |
    |   1000 | Lstsq          |   1.06e-08 |   3.72e-10 |     3.85e-03 |     3.74e-03 |
    |   1000 | GDA            |   2.46e+01 |   4.96e-01 |     2.48e-01 |     5.85e-01 |
    |   1000 | Uzawa          |   1.04e-04 |   5.67e-04 |     2.71e-02 |     4.36e-02 |
    |   1000 | Arrow-Hurwicz  |   2.44e-01 |   5.53e-03 |     6.05e-01 |     4.78e-01 |
  ],
  caption: [特征数量 $M$ 消融实验：误差和 KKT 残差随 $M$ 的变化],
) <tb:ablation-M>

#figure(
  image("/public/images/saddle-point/ablation/M/ablation-M.png"),
  caption: [特征数量 $M$ 消融实验：误差和 KKT 残差随 $M$ 的变化],
) <fig:ablation-M>

=== 形状参数 $gamma$ 的影响

取 $gamma in {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0}$，各算法在相同随机种子下运行 $K = 20000$ 步。结果如 @tb:ablation-gamma 和 @fig:ablation-gamma 所示。

#figure(
  three-line-table(
    columns: 6,
    align: (right, left, right, right, right, right),
  )[
    | $gamma$|       算法      | $norm(bold(r)_"c")_2$ | $norm(bold(r)_"e")_2$ |      位移误差 |    应力误差    |
    |-------:|:---------------|-------------:|-------------:|-------------:|-------------:|
    |    1.0 | Projection     |   4.26e-07 |   2.28e-08 |     7.72e-07 |     5.54e-04 |
    |    1.0 | Eigh           |   1.25e-06 |   7.78e-07 |     1.19e-02 |     5.03e-02 |
    |    1.0 | Lstsq          |   3.70e-08 |   2.40e-09 |     4.75e-02 |     2.80e-02 |
    |    1.0 | GDA            |   4.49e+00 |   4.79e-02 |     3.92e-01 |     6.42e-01 |
    |    1.0 | Uzawa          |   5.45e-04 |   3.72e-04 |     1.08e-01 |     2.89e-01 |
    |    1.0 | Arrow-Hurwicz  |   6.82e-02 |   1.98e-03 |     6.61e-01 |     7.26e-01 |
    |    1.5 | Projection     |   1.36e-06 |   9.98e-08 |     3.68e-06 |     1.04e-03 |
    |    1.5 | Eigh           |   3.81e-07 |   1.36e-07 |     1.90e-03 |     1.16e-02 |
    |    1.5 | Lstsq          |   7.96e-08 |   1.20e-08 |     2.29e-01 |     5.91e-02 |
    |    1.5 | GDA            |   5.94e+00 |   4.22e-02 |     1.89e-01 |     3.55e-01 |
    |    1.5 | Uzawa          |   3.05e-04 |   5.10e-04 |     6.15e-02 |     1.56e-01 |
    |    1.5 | Arrow-Hurwicz  |   8.08e-02 |   2.60e-03 |     5.76e-01 |     6.40e-01 |
    |    2.0 | Projection     |   3.49e-06 |   1.35e-06 |     9.76e-06 |     2.98e-03 |
    |    2.0 | Eigh           |   2.27e-07 |   1.79e-07 |     1.78e-03 |     8.50e-03 |
    |    2.0 | Lstsq          |   1.37e-07 |   6.78e-09 |     4.55e-01 |     6.05e-02 |
    |    2.0 | GDA            |   7.47e+00 |   7.24e-02 |     1.50e-01 |     2.62e-01 |
    |    2.0 | Uzawa          |   1.97e-04 |   2.85e-04 |     2.06e-02 |     5.94e-02 |
    |    2.0 | Arrow-Hurwicz  |   1.38e-01 |   3.18e-03 |     6.44e-01 |     4.82e-01 |
    |    2.5 | Projection     |   6.28e-06 |   8.46e-06 |     2.32e-05 |     5.63e-03 |
    |    2.5 | Eigh           |   2.41e-07 |   3.06e-07 |     3.59e-03 |     1.17e-02 |
    |    2.5 | Lstsq          |   2.38e-06 |   1.17e-07 |     1.03e+01 |     1.43e+00 |
    |    2.5 | GDA            |   4.85e+00 |   9.34e-02 |     1.10e-01 |     2.12e-01 |
    |    2.5 | Uzawa          |   1.14e-04 |   9.17e-05 |     8.45e-03 |     3.23e-02 |
    |    2.5 | Arrow-Hurwicz  |   1.90e-01 |   2.55e-03 |     7.67e-01 |     3.20e-01 |
    |    3.0 | Projection     |   6.37e-06 |   2.72e-05 |     6.09e-05 |     8.52e-03 |
    |    3.0 | Eigh           |   3.62e-07 |   4.89e-07 |     6.63e-03 |     1.67e-02 |
    |    3.0 | Lstsq          |   2.53e-07 |   1.64e-08 |     1.42e+00 |     1.34e-01 |
    |    3.0 | GDA            |   8.40e+00 |   4.50e-02 |     1.04e-01 |     1.93e-01 |
    |    3.0 | Uzawa          |   8.10e-05 |   6.65e-05 |     8.07e-03 |     2.94e-02 |
    |    3.0 | Arrow-Hurwicz  |   1.78e-01 |   3.01e-03 |     6.17e-01 |     2.62e-01 |
    |    3.5 | Projection     |   5.29e-05 |   8.09e-05 |     1.49e-04 |     1.29e-02 |
    |    3.5 | Eigh           |   3.71e-07 |   8.47e-07 |     1.15e-02 |     2.19e-02 |
    |    3.5 | Lstsq          |   1.81e-04 |   6.95e-06 |     6.15e+02 |     6.71e+01 |
    |    3.5 | GDA            |   7.87e+00 |   8.00e-02 |     9.61e-02 |     1.89e-01 |
    |    3.5 | Uzawa          |   6.93e-05 |   7.35e-05 |     8.69e-03 |     3.08e-02 |
    |    3.5 | Arrow-Hurwicz  |   1.24e-01 |   3.33e-03 |     3.12e-01 |     2.29e-01 |
    |    4.0 | Projection     |   1.22e-04 |   1.78e-04 |     2.78e-04 |     1.75e-02 |
    |    4.0 | Eigh           |   5.27e-07 |   1.20e-06 |     2.60e-02 |     3.22e-02 |
    |    4.0 | Lstsq          |   5.76e-06 |   2.81e-07 |     2.82e+01 |     2.71e+00 |
    |    4.0 | GDA            |   7.22e+00 |   9.66e-02 |     1.04e-01 |     2.11e-01 |
    |    4.0 | Uzawa          |   6.51e-05 |   8.52e-05 |     9.29e-03 |     3.39e-02 |
    |    4.0 | Arrow-Hurwicz  |   1.14e-01 |   2.44e-03 |     1.91e-01 |     1.85e-01 |
  ],
  caption: [形状参数 $gamma$ 消融实验：误差和 KKT 残差随 $gamma$ 的变化],
) <tb:ablation-gamma>

#figure(
  image("/public/images/saddle-point/ablation/gamma/ablation-gamma.png"),
  caption: [形状参数 $gamma$ 消融实验：误差和 KKT 残差随 $gamma$ 的变化],
) <fig:ablation-gamma>

=== 激活函数 $sigma.alt$ 的影响

取 $sigma.alt in {"tanh", "sigmoid", "relu", "softplus", "elu", "swish"}$，各算法在相同随机种子下运行 $K = 20000$ 步。结果如 @fig:ablation-activation 所示。

#figure(
  three-line-table(
    columns: 6,
    align: (right, left, right, right, right, right),
  )[
    |   激活函数  |       算法      | $norm(bold(r)_"c")_2$ | $norm(bold(r)_"e")_2$ |      位移误差 |    应力误差    |
    |-----------:|:---------------|-------------:|-------------:|-------------:|-------------:|
    | tanh       | Projection     |   3.49e-06 |   1.35e-06 |     9.76e-06 |     2.98e-03 |
    | tanh       | Eigh           |   2.27e-07 |   1.79e-07 |     1.78e-03 |     8.50e-03 |
    | tanh       | Lstsq          |   1.37e-07 |   6.78e-09 |     4.55e-01 |     6.05e-02 |
    | tanh       | GDA            |   7.47e+00 |   7.24e-02 |     1.50e-01 |     2.62e-01 |
    | tanh       | Uzawa          |   1.97e-04 |   2.85e-04 |     2.06e-02 |     5.94e-02 |
    | tanh       | Arrow-Hurwicz  |   1.38e-01 |   3.18e-03 |     6.44e-01 |     4.82e-01 |
    | sigmoid    | Projection     |   3.65e-07 |   1.66e-08 |     7.72e-07 |     5.54e-04 |
    | sigmoid    | Eigh           |   2.07e-06 |   1.21e-06 |     1.69e-02 |     7.64e-02 |
    | sigmoid    | Lstsq          |   9.13e-08 |   6.40e-09 |     1.02e-01 |     6.14e-02 |
    | sigmoid    | GDA            |   6.05e+00 |   2.84e-01 |     6.92e-01 |     8.02e-01 |
    | sigmoid    | Uzawa          |   7.40e-04 |   3.61e-04 |     1.40e-01 |     3.81e-01 |
    | sigmoid    | Arrow-Hurwicz  |   1.05e+00 |   2.84e-02 |     3.35e+00 |     1.02e+00 |
    | relu       | Projection     |   3.73e+00 |   1.57e-02 |     1.34e-02 |     1.40e-01 |
    | relu       | Eigh           |   4.40e-07 |   5.13e-05 |     1.25e+00 |     3.25e-01 |
    | relu       | Lstsq          |   7.88e-02 |   5.73e-03 |     8.65e+08 |     1.22e+02 |
    | relu       | GDA            |   2.47e+01 |   2.30e-01 |     1.55e-01 |     2.78e-01 |
    | relu       | Uzawa          |   6.23e-05 |   7.48e-04 |     4.41e-02 |     1.55e-01 |
    | relu       | Arrow-Hurwicz  |   6.89e-02 |   1.65e-03 |     1.52e-01 |     2.78e-01 |
    | softplus   | Projection     |   1.31e-06 |   2.94e-08 |     9.52e-07 |     6.69e-04 |
    | softplus   | Eigh           |   1.78e-05 |   3.01e-06 |     7.24e-02 |     2.19e-01 |
    | softplus   | Lstsq          |   1.42e-06 |   1.52e-07 |     5.58e-01 |     3.91e-01 |
    | softplus   | GDA            |   2.52e+01 |   7.82e-01 |     5.90e-01 |     8.89e-01 |
    | softplus   | Uzawa          |   7.52e-04 |   7.23e-04 |     1.50e-01 |     4.02e-01 |
    | softplus   | Arrow-Hurwicz  |   4.26e-01 |   9.30e-03 |     5.44e-01 |     8.38e-01 |
    | elu        | Projection     |   3.46e+00 |   4.71e-03 |     3.40e-03 |     7.78e-02 |
    | elu        | Eigh           |   5.91e-07 |   7.68e-06 |     5.44e-02 |     9.77e-02 |
    | elu        | Lstsq          |   1.18e-03 |   8.95e-05 |     7.11e+05 |     5.21e+01 |
    | elu        | GDA            |   2.73e+01 |   2.64e-01 |     1.99e-01 |     3.88e-01 |
    | elu        | Uzawa          |   1.84e-04 |   9.25e-04 |     9.37e-02 |     1.59e-01 |
    | elu        | Arrow-Hurwicz  |   1.21e-01 |   2.54e-03 |     4.86e-01 |     5.98e-01 |
    | swish      | Projection     |   1.09e-06 |   5.57e-08 |     8.68e-07 |     6.09e-04 |
    | swish      | Eigh           |   3.54e-06 |   1.90e-06 |     1.40e-02 |     6.63e-02 |
    | swish      | Lstsq          |   1.55e-07 |   1.65e-08 |     1.26e-01 |     6.34e-02 |
    | swish      | GDA            |   2.05e+01 |   3.87e-01 |     3.45e-01 |     6.98e-01 |
    | swish      | Uzawa          |   4.01e-04 |   5.01e-04 |     1.10e-01 |     2.69e-01 |
    | swish      | Arrow-Hurwicz  |   1.51e-01 |   2.98e-03 |     9.47e-01 |     7.07e-01 |
  ],
  caption: [激活函数 $sigma.alt$ 消融实验：误差和 KKT 残差随 $sigma.alt$ 的变化],
) <tb:ablation-activation>

#figure(
  image("/public/images/saddle-point/ablation/activation/ablation-activation.png"),
  caption: [激活函数 $sigma.alt$ 消融实验：误差和 KKT 残差随 $sigma.alt$ 的变化],
) <fig:ablation-activation>


#bibliography("/public/reference/saddle-point.bib")
