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
  bold(Xi)_bold(Sigma) :=& span{xi_m (E_(i j) + E_(j i))}_(m=0)^M subset bold(Sigma), \
  bold(Xi)_bold(U) :=& span{xi_m e_i}_(m=0)^M subset bold(U),
$
其中 $E_(i j)$ 是 $RR^(3 times 3)$ 的标准单位矩阵，$e_i$ 是 $RR^3$ 的标准基向量。则线弹性方程的近似解 $(bold(phi)_bold(sigma), bold(phi)_bold(u)) in bold(Xi)_bold(Sigma) times bold(Xi)_bold(U)$ 满足如下离散鞍点问题：
$ Pi(bold(phi)_bold(sigma), bold(phi)_bold(u)) = min_(bold(phi)_bold(tau) in bold(Xi)_bold(Sigma)) max_(bold(phi)_bold(v) in bold(Xi)_bold(U)) Pi(bold(phi)_bold(tau), bold(phi)_bold(v)). $
这意味着泛函在 $(bold(phi)_bold(sigma), bold(phi)_bold(u))$ 处关于任意方向 $(bold(phi)_bold(tau), bold(phi)_bold(v))$ 的一阶变分为零：
$
  cases(
    a(bold(phi)_bold(sigma), bold(phi)_bold(tau)) + b(bold(phi)_bold(tau), bold(phi)_bold(u)) & = 0 & quad forall bold(phi)_bold(tau) in bold(Xi)_bold(Sigma),
    b(bold(phi)_bold(sigma), bold(phi)_bold(v)) + (bold(f), bold(phi)_bold(v)) & = 0 & quad forall bold(phi)_bold(v) in bold(Xi)_bold(U).
  )
$
