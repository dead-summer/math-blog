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
    bold(S) : bold(sigma) - bold(epsilon)(bold(u)) & = 0 & quad "in" Omega,
    nabla dot bold(sigma) + bold(f) & = 0 & quad "in" Omega,
    bold(u) & = 0 & quad "on" partial Omega.
  )
$
其中 $bold(epsilon)(bold(u)) = 1/2(nabla bold(u) + (nabla bold(u))^T)$ 是应变张量，$bold(f)$ 是体力，$bold(S)$ 为柔性变量。

定义如下双线性形式 $a: bold(Sigma) times bold(Sigma) -> RR$ 与 $b: bold(Sigma) times bold(U) -> RR$：
$
  a(bold(sigma), bold(tau)) & := integral_Omega (bold(S) : bold(sigma)) : bold(tau) dif bold(x), \
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
  bold(Xi) := span{xi_0, xi_1, ..., xi_M },
$
其中 $xi_0 = 1$。因此，神经特征空间 $bold(Xi)$ 是由单隐层全连接神经网络的隐藏层神经元生成的函数空间。我们可以将 $bold(Xi)$ 视为一个近似空间，用于近似求解线弹性方程的解。

为便于在 3D 中稳定采样并计算梯度，本文采用如下重参数化来生成 $(bold(w)_m, b_m)$：取全局形状参数 $gamma > 0$，并令
$
  bold(w)_m = gamma bold(a)_m, quad b_m = gamma r_m.
$
其中 $norm(bold(a)_m)_2 = 1$ 表示超平面法向量，$r_m$ 表示截距。于是
$
  xi_m (bold(x)) = sigma.alt(gamma (bold(a)_m^T bold(x) + r_m)).
$
采样策略为
$
  bold(a)_m = bold(X)_m / norm(bold(X)_m)_2, quad bold(X)_m ~ cal(N)(0, bold(I)_3), quad  r_m ~ cal(U)[0, 1],
$
然后取 $bold(w)_m = gamma bold(a)_m, b_m = gamma r_m$。反过来亦有等价关系
$
  bold(a)_m = bold(w)_m / norm(bold(w)_m)_2, quad r_m = b_m / norm(bold(w)_m)_2, quad gamma = norm(bold(w)_m)_2.
$

= 导出线性系统

将 $bold(Sigma)$ 和 $bold(U)$ 分别近似为神经特征空间 $bold(Xi)$ 的张成空间，即
$
  bold(Xi)_bold(Sigma) := & span{xi_m (bold(E)_(i j) + bold(E)_(j i)): m = 0, ..., M, i, j = 1, 2, 3} subset bold(Sigma), \
      bold(Xi)_bold(U) := & span{xi_m bold(e)_i: m = 0, ..., M, i = 1, 2, 3 } subset bold(U),
$
其中 $bold(E)_(i j)$ 是 $RR^(3 times 3)$ 的标准单位矩阵，$bold(e)_i$ 是 $RR^3$ 的标准基向量。
注意当 $i = j$ 时有 $bold(E)_(i i) + bold(E)_(i i) = 2 bold(E)_(i i)$，为避免对角项的系数重复，下面改用 Voigt 形式的一组对称基 ${upright(bold(E))_alpha}_1^6$：
$
  upright(bold(E))_1 &= bold(E)_(11), & upright(bold(E))_2 &= bold(E)_(22), & upright(bold(E))_3 &= bold(E)_(33), \
  upright(bold(E))_4 &= bold(E)_(12) + bold(E)_(21), & upright(bold(E))_5 &= bold(E)_(23) + bold(E)_(32), & upright(bold(E))_6 &= bold(E)_(13) + bold(E)_(31).
$
于是 $bold(Xi)_bold(Sigma)$ 可等价写为
$ bold(Xi)_bold(Sigma) = span{ xi_m upright(bold(E))_alpha: alpha = 1, 2, ..., 6; 0 <= m <= M }. $

本文采用工程上常用的 Voigt 记号（排列顺序为 $11, 22, 33, 12, 23, 13$）。对称应力张量与对称应变张量的 Voigt 向量分别定义为
$
  upright(bold(sigma)) = (sigma_(11), sigma_(22), sigma_(33), sigma_(12), sigma_(23), sigma_(13))^T, \
  upright(bold(epsilon)) = (epsilon_(11), epsilon_(22), epsilon_(33), 2 epsilon_(12), 2 epsilon_(23), 2 epsilon_(13))^T.
$
在本文对称基 ${upright(bold(E))_alpha}_(alpha=1)^6$ 下，有
$
  bold(sigma) = sum_(alpha=1)^6 upright(bold(sigma))_alpha upright(bold(E))_alpha, quad
  upright(bold(epsilon))_alpha = bold(epsilon) : upright(bold(E))_alpha.
$

线弹性方程的近似解 $(bold(phi)_bold(sigma), bold(phi)_bold(u)) in bold(Xi)_bold(Sigma) times bold(Xi)_bold(U)$ 满足如下离散鞍点问题：
$
  Pi(bold(phi)_bold(sigma), bold(phi)_bold(u)) = min_(bold(phi)_bold(tau) in bold(Xi)_bold(Sigma)) max_(bold(phi)_bold(v) in bold(Xi)_bold(U)) Pi(bold(phi)_bold(tau), bold(phi)_bold(v)).
$
这意味着泛函在 $(bold(phi)_bold(sigma), bold(phi)_bold(u))$ 处关于任意方向 $(bold(phi)_bold(tau), bold(phi)_bold(v))$ 的一阶变分为零：
$
  cases(
    a(bold(phi)_bold(sigma), bold(phi)_bold(tau)) + b(bold(phi)_bold(tau), bold(phi)_bold(u)) & = 0 & quad forall bold(phi)_bold(tau) in bold(Xi)_bold(Sigma),
    b(bold(phi)_bold(sigma), bold(phi)_bold(v)) + (bold(f), bold(phi)_bold(v)) & = 0 & quad forall bold(phi)_bold(v) in bold(Xi)_bold(U).
  )
$

== 系数展开

将近似解在上述基上展开：
$
  bold(phi)_bold(sigma) & = sum_(m=0)^M sum_(alpha=1)^6 s_(m, alpha) xi_m upright(bold(E))_alpha, \
      bold(phi)_bold(u) & = sum_(m=0)^M sum_(i=1)^3 u_(m, i) xi_m bold(e)_i.
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
  bold(phi)_bold(tau) = xi_n upright(bold(E))_beta, quad 0 <= n <= M, 1 <= beta <= 6, \
  bold(phi)_bold(v) = xi_n bold(e)_j, quad 0 <= n <= M, 1 <= j <= 3.
$

定义矩阵块与离散载荷向量：
$
  bold(A)_((n, beta), (m, alpha)) & := a(xi_n upright(bold(E))_beta, xi_m upright(bold(E))_alpha), \
      bold(B)_((n, beta), (m, i)) & := b(xi_n upright(bold(E))_beta, xi_m bold(e)_i), \
                 bold(F)_((n, j)) & := (bold(f), xi_n bold(e)_j).
$

对任意固定的 $(n, beta)$，取测试函数 $bold(phi)_bold(tau) = xi_n upright(bold(E))_beta$。第一条离散变分方程为
$
  0 = a(bold(phi)_bold(sigma), xi_n upright(bold(E))_beta) + b(xi_n upright(bold(E))_beta, bold(phi)_bold(u)).
$
将系数展开式
$ bold(phi)_bold(sigma) = sum_(m=0)^M sum_(alpha=1)^6 s_(m, alpha) xi_m upright(bold(E))_alpha $
与
$ bold(phi)_bold(u) = sum_(m=0)^M sum_(i=1)^3 u_(m, i) xi_m bold(e)_i $
代入，并利用双线性性；同时注意在线弹性常用对称性假设下 $a(bold(sigma), bold(tau)) = a(bold(tau), bold(sigma))$，得到
$
  0
  = sum_(m=0)^M sum_(alpha=1)^6 s_(m, alpha) a(xi_n upright(bold(E))_beta, xi_m upright(bold(E))_alpha)
  + sum_(m=0)^M sum_(i=1)^3 u_(m, i) b(xi_n upright(bold(E))_beta, xi_m bold(e)_i).
$
将上述两项分别识别为 $bold(A), bold(B)$ 的矩阵元素，即
$
  sum_(m=0)^M sum_(alpha=1)^6 bold(A)_((n, beta), (m, alpha)) s_(m, alpha)
  + sum_(m=0)^M sum_(i=1)^3 bold(B)_((n, beta), (m, i)) u_(m, i) = 0.
$
这条方程正对应于 $bold(A) bold(s) + bold(B) bold(u) = 0$ 的第 $(n, beta)$ 个分量。

同理，对任意固定的 $(n, j)$，取测试函数 $bold(phi)_bold(v) = xi_n bold(e)_j$。第二条离散变分方程为
$
  0 = b(bold(phi)_bold(sigma), xi_n bold(e)_j) + (bold(f), xi_n bold(e)_j).
$
代入 $bold(phi)_bold(sigma)$ 并展开：
$
  0
  = sum_(m=0)^M sum_(alpha=1)^6 s_(m, alpha) b(xi_m upright(bold(E))_alpha, xi_n bold(e)_j)
  + bold(F)_((n, j)).
$
注意 $b(xi_m upright(bold(E))_alpha, xi_n bold(e)_j) = bold(B)_((m, alpha), (n, j))$，因此这条方程正对应于 $bold(B)^T bold(s) + bold(F) = 0$ 的第 $(n, j)$ 个分量。

将 $bold(A), bold(B)$ 视为 $(M+1) times (M+1)$ 块矩阵（块行由 $n$，块列由 $m$ 索引），$bold(F)$ 视为 $(M+1) times 1$ 块行向量，其堆叠形式为
$
  bold(A) =
  mat(
    bold(A)_(0, 0), dots.h, bold(A)_(0, M);
    dots.v, dots.down, dots.v;
    bold(A)_(M, 0), dots.h, bold(A)_(M, M)
  ), \
  bold(B) =
  mat(
    bold(B)_(0, 0), dots.h, bold(B)_(0, M);
    dots.v, dots.down, dots.v;
    bold(B)_(M, 0), dots.h, bold(B)_(M, M)
  ), \
  bold(F) =
  mat(
    bold(F)_0;
    dots.v;
    bold(F)_M
  ).
$
其中块矩阵满足（块内行指标为 $beta$，列指标为 $alpha$ 或 $i$）：
$
  bold(A)_(n, m) in RR^(6 times 6), &quad (bold(A)_(n, m))_(beta, alpha) = bold(A)_((n, beta), (m, alpha)), \
  bold(B)_(n, m) in RR^(6 times 3), &quad (bold(B)_(n, m))_(beta, i) = bold(B)_((n, beta), (m, i)), \
  bold(F)_n in RR^3, &quad (bold(F)_n)_j = bold(F)_((n, j)).
$
为强调块结构，将系数按特征下标分块：
$
  bold(s)_m &:= (s_(m, 1), ..., s_(m, 6))^T in RR^6, \
  bold(u)_m &:= (u_(m, 1), ..., u_(m, 3))^T in RR^3, \
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
  := a(xi_n upright(bold(E))_beta, xi_m upright(bold(E))_alpha)
  = integral_Omega (bold(S) : (xi_n upright(bold(E))_beta)) : (xi_m upright(bold(E))_alpha) dif bold(x).
$
由于 $xi_n, xi_m$ 为标量函数且 $upright(bold(E))_alpha, upright(bold(E))_beta$ 为常矩阵，有
$
  bold(A)_((n, beta), (m, alpha))
  = integral_Omega xi_n (bold(x)) xi_m (bold(x)) ((bold(S)(bold(x)) : upright(bold(E))_beta) : upright(bold(E))_alpha) dif bold(x).
$
若将收缩写成坐标分量（Frobenius 内积），则
$
  ((bold(S) : upright(bold(E))_beta) : upright(bold(E))_alpha)
  = sum_(i=1)^3 sum_(j=1)^3 sum_(k=1)^3 sum_(l=1)^3 S_(i j k l) (upright(bold(E))_beta)_(k l) (upright(bold(E))_alpha)_(i j),
$
因此
$
  bold(A)_((n, beta), (m, alpha))
  = integral_Omega xi_n xi_m
  sum_(i=1)^3 sum_(j=1)^3 sum_(k=1)^3 sum_(l=1)^3 S_(i j k l) (upright(bold(E))_beta)_(k l) (upright(bold(E))_alpha)_(i j)
  dif bold(x).
$

为了在实现中更直接地使用工程 Voigt 形式，可定义 $6 times 6$ 的柔度矩阵 $upright(bold(S))$ 使其满足 $upright(bold(epsilon)) = upright(bold(S)) upright(bold(sigma))$。在本文对称基下，可取
$
  upright(bold(S))_(alpha beta) := (bold(S) : upright(bold(E))_beta) : upright(bold(E))_alpha.
$
则对任意 $alpha, beta$，
$
  bold(A)_((n, beta), (m, alpha)) = integral_Omega xi_n xi_m upright(bold(S))_(alpha beta) dif bold(x).
$
注意：工程 Voigt 中剪切应变采用 $2 epsilon_(i j)$；若材料参数以 Kelvin（剪切分量带 $sqrt(2)$）等其他约定给出，需要先做换算再作为 $upright(bold(S))$ 使用。

=== $bold(B)$ 块元素：散度只作用在测试基函数上

由定义
$
  bold(B)_((n, beta), (m, i))
  := b(xi_n upright(bold(E))_beta, xi_m bold(e)_i)
  = integral_Omega (nabla dot (xi_n upright(bold(E))_beta)) dot (xi_m bold(e)_i) dif bold(x).
$
利用张量散度的分量定义 $(nabla dot bold(tau))_p = tau_(p k, k)$，令 $bold(tau) = xi_n upright(bold(E))_beta$，则
$
  (nabla dot (xi_n upright(bold(E))_beta))_i
  = (xi_n (upright(bold(E))_beta)_(i k))_(,k)
  = (upright(bold(E))_beta)_(i k) partial_k xi_n,
$
即向量形式
$
  nabla dot (xi_n upright(bold(E))_beta) = upright(bold(E))_beta nabla xi_n.
$
因此
$
  bold(B)_((n, beta), (m, i)) = integral_Omega xi_m (upright(bold(E))_beta nabla xi_n)_i dif bold(x)
  = integral_Omega xi_m sum_(k=1)^3 (upright(bold(E))_beta)_(i k) partial_k xi_n dif bold(x).
$

对 6 个对称基，$upright(bold(E))_beta nabla xi_n$ 可完全写成梯度分量的线性组合。记
$
  nabla xi_n = (partial_1 xi_n, partial_2 xi_n, partial_3 xi_n)^T,
$
则
$
  cases(
    upright(bold(E))_1 nabla xi_n = (partial_1 xi_n, 0, 0)^T,
    upright(bold(E))_2 nabla xi_n = (0, partial_2 xi_n, 0)^T,
    upright(bold(E))_3 nabla xi_n = (0, 0, partial_3 xi_n)^T,
    upright(bold(E))_4 nabla xi_n = (partial_2 xi_n, partial_1 xi_n, 0)^T,
    upright(bold(E))_5 nabla xi_n = (0, partial_3 xi_n, partial_2 xi_n)^T,
    upright(bold(E))_6 nabla xi_n = (partial_3 xi_n, 0, partial_1 xi_n)^T.
  )
$

而对于神经特征函数 $xi_n (bold(x)) = sigma.alt(bold(w)_n^T bold(x) + b_n)$（$n >= 1$），其梯度可直接计算为
$
  nabla xi_n (bold(x)) = sigma.alt'(bold(w)_n^T bold(x) + b_n) bold(w)_n.
$
若采用重参数化 $bold(w)_n = gamma bold(a)_n, b_n = gamma r_n$，则等价地
$
  nabla xi_n (bold(x)) = sigma.alt'(gamma (bold(a)_n^T bold(x) + r_n)) gamma bold(a)_n.
$
特别地 $xi_0 = 1$，故 $nabla xi_0 = 0$，从而所有以 $n=0$ 为测试函数的 $bold(B)$ 行元素均为 $0$。

=== $bold(F)$ 载荷向量元素：体力分量在基上的投影

由定义
$
  bold(F)_((n, j))
  := (bold(f), xi_n bold(e)_j)
  = integral_Omega bold(f) dot (xi_n bold(e)_j) dif bold(x)
  = integral_Omega f_j (bold(x)) xi_n (bold(x)) dif bold(x).
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
    &approx sum_(q=1)^Q w_q xi_n (bold(x)_q) xi_m (bold(x)_q) ((bold(S)(bold(x)_q) : upright(bold(E))_beta) : upright(bold(E))_alpha),
    bold(B)_((n, beta), (m, i)) &approx sum_(q=1)^Q w_q xi_m (bold(x)_q) (upright(bold(E))_beta nabla xi_n (bold(x)_q))_i,
    bold(F)_((n, j)) &approx sum_(q=1)^Q w_q f_j (bold(x)_q) xi_n (bold(x)_q).
  )
$

= ADMM 算法

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
  bold(u)^(k+1) &= bold(u)^k + eta_bold(u)^"ADMM" (bold(B)^T bold(s)^k + bold(F)).
$

再固定 $bold(u)^(k+1)$ 对 $bold(s)$ 做一次梯度下降更新：
$
  bold(s)^(k+1) &= bold(s)^k - eta_bold(s)^"ADMM" (bold(A) bold(s)^k + bold(B) bold(u)^(k+1)).
$
这里 $eta_bold(u)^"ADMM", eta_bold(s)^"ADMM" > 0$ 为步长。上式写成最简单的梯度上升/下降形式，实际实现中依赖优化器行为（如 Adam 将步长替换为自适应更新）。

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

在第 $k$ 轮循环中，更新
$
  (bold(A) + rho bold(I)) bold(s)^(k) &= -bold(B) bold(u)^(k), \
  bold(u)^(k) &= bold(u)^(k) + eta_bold(u)^"Uzawa" (bold(B)^T bold(s)^(k) + bold(F)).
$
最后设 $bold(u)^(k+1) = bold(u)^(k, T)$。其中 $bold(I)$ 为单位阵，$rho >= 0$ 是用于数值稳健性的阻尼参数；当 $bold(A)$ 可能奇异或病态时可取 $rho > 0$。

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

本节给出本文将要进行的 3D 数值实验设置，用于验证前述离散鞍点系统与四种迭代算法（Direct、ADMM、Uzawa 和 Arrow-Hurwicz）的可实现性与收敛性。所有对比实验均采用相同的 3D 结构（应力 Voigt 6 分量 + 位移 3 分量），并在同一组采样点上组装 $bold(A), bold(B), bold(F)$ 以保证公平比较。

== 方程与边界条件

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
  bold(sigma)(bold(u)) = 2 mu bold(epsilon)(bold(u)) + lambda tr(bold(epsilon)(bold(u))) bold(I),
$
其中 $tr(bold(epsilon)) := bold(epsilon) : bold(I)$。平衡方程为
$
  -nabla dot bold(sigma)(bold(u)) = bold(f) quad "in" Omega,
$
并施加齐次 Dirichlet 边界条件
$
  bold(u) = 0 quad "on" partial Omega.
$

为与前文 Hellinger-Reissner 形式一致，取柔度张量 $bold(S) = bold(C)^(-1)$ 使得 $bold(S):bold(sigma) = bold(epsilon)(bold(u))$，并沿用工程 Voigt 记号装配 $bold(A)$ 块。

== 计算域与制造解

取计算域 $Omega = [0, 1]^3$。为保证齐次 Dirichlet 边界条件，定义包络函数
$
  zeta(bold(x)) = x_1(1-x_1) x_2(1-x_2) x_3(1-x_3).
$
设精确位移为
$
  bold(u)_"ex" (bold(x))
  = zeta(bold(x))
    mat(sin(pi x_1) sin(pi x_2) sin(pi x_3);
     sin(2 pi x_1) sin(pi x_2) sin(pi x_3);
     sin(pi x_1) sin(2 pi x_2) sin(pi x_3)).
$
则 $bold(u)_"ex" = 0$ 在 $partial Omega$ 上成立。相应精确应力取
$
  bold(sigma)_"ex" = bold(sigma)(bold(u)_"ex"),
$
体力通过制造解定义为
$
  bold(f)(bold(x)) = -nabla dot bold(sigma)_"ex" (bold(x)).
$
实现时将用自动微分或符号计算得到 $bold(f)$，不在文中展开其冗长表达式。

== 材料参数与柔度矩阵

选取常数材料参数
$
  E = 1, quad nu = 0.3.
$
在工程 Voigt 排列顺序 $(11, 22, 33, 12, 23, 13)$ 下，柔度矩阵 $upright(bold(S))$ 满足
$
  upright(bold(epsilon)) = upright(bold(S)) upright(bold(sigma)).
$
取
$
  upright(bold(S)) = 1/E mat(
    1, -nu, -nu, 0, 0, 0;
    -nu, 1, -nu, 0, 0, 0;
    -nu, -nu, 1, 0, 0, 0;
    0, 0, 0, 2(1+nu), 0, 0;
    0, 0, 0, 0, 2(1+nu), 0;
    0, 0, 0, 0, 0, 2(1+nu)
  ).
$
该约定与前文工程剪切应变 $2 epsilon_(i j)$ 的定义一致。

== 神经特征空间与离散未知量

采用前文定义的单隐层全连接随机特征函数，取激活函数 $sigma.alt = tanh$：
$
  xi_0 = 1, quad xi_m (bold(x)) = sigma.alt(bold(w)_m^T bold(x) + b_m), quad m = 1, 2, ..., M.
$
其中 $bold(w)_m in RR^3, b_m in RR$ 在实验开始时按照如下方式随机生成并固定：
$
  bold(w)_m = gamma bold(a)_m, quad b_m = gamma r_m.
$
固定 $gamma = 2.0$ 以控制特征函数的频率范围；$bold(a)_m = bold(X)_m \/ norm(bold(X)_m)_2$，其中 $bold(X)_m ~ cal(N)(0, bold(I)_3)$ 是从标准正态分布采样的随机向量；$r_m ~ cal(U)[0, 1]$，是从 $[0, 1]$ 均匀分布采样的随机数。

本文仅迭代更新离散未知量系数 $bold(s), bold(u)$。主实验取 $M = 256$，并在消融实验中考察不同 $M$ 的影响。

在 3D 结构下，应力与位移的近似分别为
$
  bold(phi)_bold(sigma) = sum_(m=0)^M sum_(alpha=1)^6 s_(m, alpha) xi_m upright(bold(E))_alpha, \
  bold(phi)_bold(u) = sum_(m=0)^M sum_(i=1)^3 u_(m, i) xi_m bold(e)_i,
$
从而得到离散鞍点系统
$
  mat(bold(A), bold(B); bold(B)^T, 0) mat(bold(s); bold(u)) = mat(0; -bold(F)).
$

== 数值积分与数据划分

用均匀 Monte Carlo 采样近似积分。训练阶段在 $Omega$ 内均匀采样 $Q_"train" = 20000$ 个点 ${bold(x)_q}_(q=1)^(Q_"train")$，取等权
$
  w_q = abs(Omega) / Q_"train" = 1 / Q_"train".
$
在该训练点集上一次性组装 $bold(A), bold(B), bold(F)$，并在其上迭代三种算法（全量、确定性）。另外独立采样 $Q_"test" = 10000$ 个测试点用于误差评估。

== 算法对比设置

为避免离散 $bold(A)$ 病态带来的数值问题，统一采用轻微阻尼 $rho = 10^(-6)$；即在涉及求解 $bold(s)$ 的步骤中以 $bold(A) + rho bold(I)$ 替代 $bold(A)$。

#figure(
  three-line-table(
    columns: 2,
    align: (right, left)
  )[
    | 参数 | 取值 |
    |------|------|
    | 域 | $Omega = [0, 1]^3$ |
    | 边界条件 | 齐次 Dirichlet：$bold(u)=0$ on $partial Omega$ |
    | 材料 | 各向同性常系数：$E=1, nu=0.3$ |
    | 随机特征 | 激活 $tanh$ |
    | 特征采样 | $bold(w)_m = gamma bold(a)_m$，$b_m = gamma r_m$ |
    | 训练点 | $Q_"train" = 20000$ |
    | 测试点 | $Q_"test" = 10000$ |
    | 阻尼 | $rho = 10^(-6)$ |
    | 初值 | $bold(s)^0 = 0, bold(u)^0 = 0$ |
    | 迭代 | $K = 5000$ 或满足停止准则 |
  ],
)

算法细节如下：

- *ADMM*: 采用 Adam 优化器，学习率 $eta_bold(u) = eta_bold(s) = 0.02$，$bold(beta)^"Adam" = (0.9, 0.98)$，每轮各做 1 次更新。
- *Uzawa*：步长 $eta_bold(u)^"Uzawa"$ 通过 Schur 补谱半径自适应选择。
- *Arrow-Hurwicz*: 步长 $eta_bold(s)^"AH"$、$eta_bold(u)^"AH"$ 分别通过 Jacobi 谱半径和 Schur 补谱半径自适应选择；取预条件子 $bold(J) = [diag(bold(A) + rho bold(I))]^(-1)$，$bold(K) = bold(I)$。


== 评价指标

- *KKT 残差*: 记
  $
    bold(r)_bold(s) = bold(A) bold(s) + bold(B) bold(u), \
    bold(r)_bold(u) = bold(B)^T bold(s) + bold(F),
  $
  记录 $norm(bold(r)_bold(s))_2$ 与 $norm(bold(r)_bold(u))_2$ 随迭代的变化。
- *相对 $L^2$ 误差*: 在测试点上用
  $
    norm(bold(u)_h - bold(u)_"ex")_(L^2(Omega))
    approx (abs(Omega)/Q_"test" sum_(q=1)^(Q_"test") abs(bold(u)_h (bold(x)_q) - bold(u)_"ex" (bold(x)_q))^2)^(1/2)
  $
  估计位移误差，并用同样方式估计应力误差（张量按 Frobenius 范数聚合）。报告相对误差
  $
    norm(bold(u)_h - bold(u)_"ex")_(L^2) / norm(bold(u)_"ex")_(L^2), quad
    norm(bold(sigma)_h - bold(sigma)_"ex")_(L^2) / norm(bold(sigma)_"ex")_(L^2).
  $
- *收敛成本*: 记录达到阈值 $norm(bold(r)_bold(s))_2 + norm(bold(r)_bold(u))_2 <= 10^(-6)$ 的迭代步数与壁钟时间。

== 实验结果

主实验结果（$M = 256$，$Q_"train" = 20000$，$K = 2000$）如 @tb:main-results 所示。Direct 为直接求解鞍点系统的参考解。

#figure(
  three-line-table(
    columns: 6,
    align: (left, right, right, right, right, right),
  )[
    | 算法 | $norm(bold(r)_bold(s))_2$ | $norm(bold(r)_bold(u))_2$ | 位移误差 | 应力误差 | 时间 (s) |
    |------|------|------|------|------|------|
    | Direct | $1.63 times 10^(-8)$ | $7.97 times 10^(-10)$ | $2.31 times 10^(1)$ | $3.21 times 10^(0)$ | $0.06$ |
    | ADMM | $4.38 times 10^(0)$ | $6.81 times 10^(-2)$ | $8.96 times 10^(0)$ | $1.15 times 10^(1)$ | $1.52$ |
    | Uzawa | $5.24 times 10^(-6)$ | $1.40 times 10^(-3)$ | $7.61 times 10^(-1)$ | $8.40 times 10^(-1)$ | $6.40$ |
    | Arrow-Hurwicz | $1.21 times 10^(-3)$ | $1.68 times 10^(-3)$ | $7.57 times 10^(-1)$ | $9.25 times 10^(-1)$ | $1.85$ |
  ],
  caption: [主实验结果（$M = 256$）],
) <tb:main-results>

KKT 残差收敛曲线和 $L^2$ 误差收敛曲线分别见 @fig:kkt-convergence 和 @fig:l2-convergence。从 @tb:main-results 以及 @fig:kkt-convergence、@fig:l2-convergence 可以看到，离散 KKT 残差与对制造解的 $L^2$ 误差并不总是同步变化：Direct 在 $norm(bold(r)_bold(s))_2$ 与 $norm(bold(r)_bold(u))_2$ 上达到 $10^(-8)$ 量级，且耗时最低（$0.06$ s），但位移/应力相对误差分别为 $2.31 times 10^(1)$ 与 $3.21 times 10^(0)$。这说明在 $M=256$ 的特征空间下，“把鞍点系统解得更精确”并不会自动带来更小的物理量误差，误差下限仍受近似空间能力与数值效应等因素制约。

就迭代方法而言，Uzawa 与 Arrow-Hurwicz 在有限迭代预算内给出了更好的误差-成本折中：两者将位移/应力误差稳定降到 $O(10^0)$ 以下。两者的差异主要体现在残差与耗时上：Uzawa 的 $norm(bold(r)_bold(s))_2$ 更小（$5.24 times 10^(-6)$），但耗时更长（$6.40$ s）；Arrow-Hurwicz 用更短时间（$1.85$ s）取得相近误差，但其 $norm(bold(r)_bold(s))_2$ 更大（$1.21 times 10^(-3)$），且从 @fig:kkt-convergence 可见 $norm(bold(r)_bold(s))_2$ 存在较明显的平台与波动。

ADMM 在本实现与超参下未表现出有效收敛：主实验中 $norm(bold(r)_bold(s))_2 = 4.38 times 10^(0)$、$norm(bold(r)_bold(u))_2 = 6.81 times 10^(-2)$，并且位移/应力误差分别为 $8.96 times 10^(0)$ 与 $1.15 times 10^(1)$，整体显著劣于 Uzawa 与 Arrow-Hurwicz；同时 @fig:kkt-convergence 中其残差曲线振荡幅度较大，缺乏稳定下降趋势。因此，在当前设置下 ADMM 难以作为可靠的求解策略，需要进一步的步长/预条件/更新次数设计，或显著增加迭代预算后再做可比性讨论。

#figure(
  image("/public/images/saddle-point/kkt-convergence.png"),
  caption: [KKT 残差收敛曲线],
) <fig:kkt-convergence>

#figure(
  image("/public/images/saddle-point/l2-error-convergence.png"),
  caption: [$L^2$ 相对误差收敛曲线],
) <fig:l2-convergence>

== 消融实验

固定 $Q_"train" = 20000$，分别取 $M in {64, 128, 256, 512, 1024}$，各算法在相同随机种子下运行 $K = 5000$ 步。结果如 @fig:ablation-M 所示。

#figure(
  image("/public/images/saddle-point/ablation-M.png"),
  caption: [特征数量 $M$ 消融实验：误差和 KKT 残差随 $M$ 的变化],
) <fig:ablation-M>

@fig:ablation-M 展示了特征数量 $M$ 对不同求解策略的敏感性差异。Direct 的误差随 $M$ 增大呈现显著的数量级下降：当 $M=64$ 时位移误差可达 $10^4$ 量级，而当 $M=512$ 及以上时迅速降到 $O(1)$ 乃至 $10^(-1)$ 量级；应力误差也从 $10^1$ 量级下降到 $10^(-1)$ 量级。这说明 Direct 的性能主要受近似空间逼近能力支配，$M$ 不足时即便 KKT 残差极小，也可能得到较差的物理量近似。

相比之下，Uzawa 与 Arrow-Hurwicz 在所有 $M$ 下的误差曲线更平坦：位移与应力误差大致维持在 $0.5$ 到 $1.0$ 的量级并随 $M$ 缓慢改善；同时其 KKT 残差也稳定在 $10^(-6)$ 到 $10^(-3)$ 区间。这体现了两者在固定迭代预算 $K=5000$ 下对规模与病态性的鲁棒性，但也意味着它们受残差平台限制，难以像 Direct 那样在大 $M$ 时继续显著降低误差。

ADMM 的规模鲁棒性最差：随着 $M$ 增大，其 KKT 残差与 $L^2$ 误差整体上升，在大 $M$ 下明显劣于其余方法。就本实验配置而言，ADMM 在 $5000$ 步内未能有效收敛，若要公平对比，需要重新调参或引入更合适的更新/预条件机制（或增加迭代步数）。
