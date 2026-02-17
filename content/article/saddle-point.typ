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
  phi := alpha_0 + sum_(m=1)^M alpha_m sigma(bold(w)_m^T bold(x) + b_m),
$
其中 $sigma$ 是激活函数，$M$ 是神经元数量，$alpha_m in RR$ 是输出层权重，$bold(w)_m in RR^3$ 是输入层权重，$b_m in RR$ 是偏置项。记隐藏层神经元为 $xi_m = sigma(bold(w)_m^T bold(x) + b_m)$，我们可将 $xi_m: RR^3 -> RR$ 视为一个特征函数，隐藏层神经元集 ${xi_m}_1^M$ 可视为 $RR^3$ 空间中的一组基。定义神经特征空间
$
  bold(Xi) := span{xi_0, xi_1, ..., xi_M },
$
其中 $xi_0 = 1$。因此，神经特征空间 $bold(Xi)$ 是由单隐层全连接神经网络的隐藏层神经元生成的函数空间。我们可以将 $bold(Xi)$ 视为一个近似空间，用于近似求解线弹性方程的解。

= 导出线性系统

将 $bold(Sigma)$ 和 $bold(U)$ 分别近似为神经特征空间 $bold(Xi)$ 的张成空间，即
$
  bold(Xi)_bold(Sigma) := & span{xi_m (bold(E)_(i j) + bold(E)_(j i)): m = 0, ..., M, i, j = 1, 2, 3} subset bold(Sigma), \
      bold(Xi)_bold(U) := & span{xi_m bold(e)_j: m = 0, ..., M, j = 1, 2, 3 } subset bold(U),
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
      bold(phi)_bold(u) & = sum_(m=0)^M sum_(j=1)^3 u_(m, j) xi_m bold(e)_j.
$
记系数向量
$
  bold(s) & = (s_(0, 1), ..., s_(0, 6), s_(1, 1), ..., s_(M, 6))^T in RR^(6(M+1)), \
  bold(u) & = (u_(0, 1), ..., u_(0, 3), u_(1, 1), ..., u_(M, 3))^T in RR^(3(M+1)).
$

== 选取测试函数并组装矩阵

取测试函数为同一组基函数：
$
  bold(phi)_bold(tau) = xi_n upright(bold(E))_beta, quad 0 <= n <= M, 1 <= beta <= 6, \
  bold(phi)_bold(v) = xi_n bold(e)_j, quad 0 <= n <= M, 1 <= j <= 3.
$

定义矩阵块与离散载荷向量（复合指标按上述堆叠顺序理解）：
$
  bold(A)_((n, beta), (m, alpha)) & := a(xi_n upright(bold(E))_beta, xi_m upright(bold(E))_alpha), \
      bold(B)_((n, beta), (m, j)) & := b(xi_n upright(bold(E))_beta, xi_m bold(e)_j), \
                 bold(F)_((n, j)) & := (bold(f), xi_n bold(e)_j).
$
并将离散载荷按 $(n, j)$ 堆叠为
$
  bold(F) = (bold(F)_((0, 1)), ..., bold(F)_((0, 3)), bold(F)_((1, 1)), ..., bold(F)_((M, 3)))^T.
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

=== 从变分方程到代数方程

对任意固定的 $(n, beta)$，取测试函数 $bold(phi)_bold(tau) = xi_n upright(bold(E))_beta$。第一条离散变分方程为
$
  0 = a(bold(phi)_bold(sigma), xi_n upright(bold(E))_beta) + b(xi_n upright(bold(E))_beta, bold(phi)_bold(u)).
$
将系数展开式
$ bold(phi)_bold(sigma) = sum_(m=0)^M sum_(alpha=1)^6 s_(m, alpha) xi_m upright(bold(E))_alpha $
与
$ bold(phi)_bold(u) = sum_(m=0)^M sum_(j=1)^3 u_(m, j) xi_m bold(e)_j $
代入，并利用双线性性；同时注意在线弹性常用对称性假设下 $a(bold(sigma), bold(tau)) = a(bold(tau), bold(sigma))$，得到
$
  0
  = sum_(m=0)^M sum_(alpha=1)^6 s_(m, alpha) a(xi_n upright(bold(E))_beta, xi_m upright(bold(E))_alpha)
  + sum_(m=0)^M sum_(j=1)^3 u_(m, j) b(xi_n upright(bold(E))_beta, xi_m bold(e)_j).
$
将上述两项分别识别为 $bold(A), bold(B)$ 的矩阵元素，即
$
  sum_(m=0)^M sum_(alpha=1)^6 bold(A)_((n, beta), (m, alpha)) s_(m, alpha)
  + sum_(m=0)^M sum_(j=1)^3 bold(B)_((n, beta), (m, j)) u_(m, j) = 0.
$

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
  = integral_Omega xi_n(bold(x)) xi_m(bold(x)) ((bold(S)(bold(x)) : upright(bold(E))_beta) : upright(bold(E))_alpha) dif bold(x).
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
  bold(B)_((n, beta), (m, j))
  := b(xi_n upright(bold(E))_beta, xi_m bold(e)_j)
  = integral_Omega (nabla dot (xi_n upright(bold(E))_beta)) dot (xi_m bold(e)_j) dif bold(x).
$
利用张量散度的分量定义 $(nabla dot bold(tau))_p = tau_(p k, k)$，令 $bold(tau) = xi_n upright(bold(E))_beta$，则
$
  (nabla dot (xi_n upright(bold(E))_beta))_j
  = (xi_n (upright(bold(E))_beta)_(j k))_(,k)
  = (upright(bold(E))_beta)_(j k) partial_k xi_n,
$
即向量形式
$
  nabla dot (xi_n upright(bold(E))_beta) = upright(bold(E))_beta nabla xi_n.
$
因此
$
  bold(B)_((n, beta), (m, j)) = integral_Omega xi_m (upright(bold(E))_beta nabla xi_n)_j dif bold(x)
  = integral_Omega xi_m sum_(k=1)^3 (upright(bold(E))_beta)_(j k) partial_k xi_n dif bold(x).
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

而对于神经特征函数 $xi_m(bold(x)) = sigma(bold(w)_m^T bold(x) + b_m)$（$m >= 1$），其梯度可直接计算为
$
  nabla xi_m(bold(x)) = sigma'(bold(w)_m^T bold(x) + b_m) bold(w)_m.
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
    bold(B)_((n, beta), (m, j)) &approx sum_(q=1)^Q w_q xi_m (bold(x)_q) (upright(bold(E))_beta nabla xi_n (bold(x)_q))_j,
    bold(F)_((n, j)) &approx sum_(q=1)^Q w_q f_j (bold(x)_q) xi_n (bold(x)_q).
  )
$
