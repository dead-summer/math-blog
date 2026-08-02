

# A PRIORI ANALYSIS OF STABLE NEURAL NETWORK SOLUTIONS TO NUMERICAL PDEs

Qingguo Hong

Department of Mathematics

Pennsylvania State University

University Park, PA 16802

huq11@psu.edu

Jonathan W. Siegel

Department of Mathematics

Pennsylvania State University

University Park, PA 16802

jus1949@psu.edu

Jinchao Xu

Department of Mathematics

Pennsylvania State University

University Park, PA 16802

xu@math.psu.edu

May 11, 2021

## ABSTRACT

Methods for solving PDEs using neural networks have recently become a very important topic. We provide an a priori error analysis for such methods which is based on the \( {\mathcal{K}}_{1}\left( \mathbb{D}\right) \) -norm of the solution. We show that the resulting constrained optimization problem can be efficiently solved using a greedy algorithm, which replaces stochastic gradient descent. Following this, we show that the error arising from discretizing the energy integrals is bounded both in the deterministic case, i.e. when using numerical quadrature, and also in the stochastic case, i.e. when sampling points to approximate the integrals. In the latter case, we use a Rademacher complexity analysis, and in the former we use standard numerical quadrature bounds. This extends existing results to methods which use a general dictionary of functions to learn solutions to PDEs and importantly gives a consistent analysis which incorporates the optimization, approximation, and generalization aspects of the problem. In addition, the Rademacher complexity analysis is simplified and generalized, which enables application to a wide range of problems.

## 1 Introduction

Recently, due to the dramatic success of deep learning, neural networks have been studied for their potential in solving high-dimensional partial differential equations [16,27,39,41]. Despite remarkable empirical success in using neural networks for solving PDEs [32,49,52], there are many open theoretical questions concerning the numerical analysis of the resulting methods. There has been much recent work on these problems, see for instance [4,10,13,17,19,23,33,35,36,38,42,46,50,51]. In these papers, different aspects of the problem are addressed, specifically the approximation, generalization, and optimization theory behind these methods. However, as far as we know, no rigorous analysis is able to take into account all of these aspects simultaneously. In this work, we provide the first consistent analysis which takes into account all three aspects of the problem: approximation, generalization, and optimization. Our analysis relies heavily upon the notions developed in [44], specifically the \( {\mathcal{K}}_{1}\left( \mathbb{D}\right) \) -space. It depends upon the exact solution lying in \( {\mathcal{K}}_{1}\left( \mathbb{D}\right) \), and also on the solution of a optimization problems of fixed dimension (see equation (31) below), which we argue are more tractable than optimizing the whole network (for instance via gradient descent), and which can provably be solved for shallow neural networks [25].

We consider solving \( {2m} \) -th order elliptic PDEs using finite combinations of dictionary elements for a dictionary \( \mathbb{D} \subset \; {H}^{m}\left( \Omega \right) \). Our approach combines the approximation theory developed in [1,22,43] with a Rademacher complexity analysis [3] to obtain error bounds on the discrete solution when Monte Carlo quadrature is used. We only require the true solution to be bounded in the \( {\mathcal{K}}_{1}\left( \mathbb{D}\right) \) -norm, and when high-order numerical quadrature is used, we require the dictionary elements and the coefficients of the PDE to be sufficiently smooth. We use such a non-standard smoothness assumption because this is the only way to overcome the 'curse of dimensionality.' Indeed, in high dimensions the metric entropy of classical smoothness spaces, i.e. spaces defined by \( {L}^{p} \) -norm bounds on derivatives, decay very slowly [30]. However, in order to obtain methods which scale well with dimension, we must assume our solution is bounded in space whose entropy decays at a rate independent of dimension, which rules out classical smoothness assumptions. That the space \( {\mathcal{K}}_{1}\left( \mathbb{D}\right) \) satisfies this entropy decay for a variety of dictionaries \( \mathbb{D} \) related to neural networks is shown in [44].

In order to incorporate the analysis of the network optimization, we replace stochastic gradient descent by a special type of greedy algorithm closely related to boosting (see [2, 13, 14, 20, 25, 26, 54]), which allows us to guarantee that the numerical solution has bounded \( {\mathcal{K}}_{1}\left( \mathbb{D}\right) \) -norm. The \( {\mathcal{K}}_{1}\left( \mathbb{D}\right) \) -norm was introduced in [7] and studied extensively in [44]. This bound on the \( {\mathcal{K}}_{1}\left( \mathbb{D}\right) \) -norm of the numerical solution is crucial for a consistent Rademacher complexity analysis. We use this Rademacher complexity analysis to control the generalization error, i.e. the error induced by using Monte Carlo quadrature to approximation the energy integrals. The uniform analysis of this quadrature error is the main idea behind our results. Discretizing the integrals appearing in the energy is always necessary for any practical numerical method, and especially so for neural networks, where the corresponding integrals have essentially no hope of being evaluated exactly.

Previously, some methods have proposed applying greedy algorithms [9, 47] to construct dictionary approximations to PDE solutions [6, 12, 24]. In [6, 12, 24], the dictionary is taken to consist of separable functions, rather than neural network functions as in this work. A further novelty of our paper is to consider the discretization of the energy using Monte Carlo sampling. The analysis of the discretization error in this case requires the novel use of tools from statistical learning theory, such as Rademacher complexity, to the numerical analysis of PDEs.

We also believe that we are the first to completely analyze the error when the energy integrals are approximated using deterministic quadrature, without relying on unverified assumptions. In addition, we consider methods which use arbitrary dictionaries of functions to learn the PDE solution. This approach applies both to neural networks with a wide range of activation functions, and also to more general methods which rely on dictionaries.

The paper is organized as follows. In Section 2, we introduce the model problem and basic notation and concepts we will use. In this section, we also refute a Bernstein-type inequality for shallow neural networks. Then, in Section 3 we provide an error analysis when Monte Carlo quadrature is used. The key new result here is a bound on the Rademacher complexity of the class of integrands. Next, in Section 5 we analyze the error when the integral is approximated using high-order numerical quadrature. Finally, we provide concluding remarks and further research directions.

## 2 Basic Setup and Notation

### 2.1 Model Problem

We follow here largely the setting in [53]. Let \( \Omega  \subset  {\mathbb{R}}^{d} \) be a bounded domain with a sufficiently smooth boundary \( \partial \Omega \). For any integer \( m \geq  1 \), we consider the following model \( {2m} \) -th order partial differential equation with certain boundary conditions:

\[\begin{cases} {Lu} &  = f\text{ in }\Omega, \\  {B}^{k}\left( u\right) &  = 0\text{ on }\partial \Omega \;\left( {0 \leq  k \leq  m - 1}\right), \end{cases} \tag{1}\]

where \( {B}^{k}\left( u\right) \) denotes the Dirichlet, Neumann, or mixed boundary conditions which will be discussed in detail in the following. Here \( L \) is the partial differential operator defined as follows

\[{Lu} = \mathop{\sum }\limits_{{\left| \alpha \right|  = m}}{\left( -1\right) }^{m}{\partial }^{\alpha }\left( {{a}_{\alpha }\left( x\right) {\partial }^{\alpha }u}\right)  + {a}_{0}\left( x\right) u, \tag{2}\]

where \( \alpha \) denotes \( n \) -dimensional multi-index \( \alpha  = \left( {{\alpha }_{1},\cdots,{\alpha }_{n}}\right) \) with

\[\left| \alpha \right|  = \mathop{\sum }\limits_{{i = 1}}^{n}{\alpha }_{i},\;{\partial }^{\alpha } = \frac{{\partial }^{\left| \alpha \right| }}{\partial {x}_{1}^{{\alpha }_{1}}\cdots \partial {x}_{n}^{{\alpha }_{n}}}.\]

For simplicity, we assume that \( {a}_{\alpha } \) are strictly positive and bounded on \( \Omega \) for \( \left| \alpha \right|  = m \) and \( \alpha  = 0 \), namely, \( \exists {\alpha }_{0} > 0,{\alpha }_{1} < \; \infty \), such that

\[{\alpha }_{0} \leq  {a}_{\alpha }\left( x\right),{a}_{0}\left( x\right)  \leq  {\alpha }_{1}\forall x \in  \Omega,\left| \alpha \right|  = m. \tag{3}\]

Further, when considering deterministic numerical quadrature in Section 5 we will make the additional assumption that \( {a}_{\alpha } \in  {C}^{\infty }\left( \Omega \right) \).

Given a nonnegative integer \( k \) and a bounded domain \( \Omega  \subset  {\mathbb{R}}^{d} \), let

\[{H}^{k}\left( \Omega \right)  \mathrel{\text{:= }} \left\{  {v \in  {L}^{2}\left( \Omega \right),{\partial }^{\alpha }v \in  {L}^{2}\left( \Omega \right),\left| \alpha \right|  \leq  k}\right\} \tag{4}\]

be standard Sobolev spaces with norm and seminorm given respectively by

\[\parallel v{\parallel }_{k} \mathrel{\text{:= }} {\left( \mathop{\sum }\limits_{{\left| \alpha \right|  \leq  k}}{\begin{Vmatrix}{\partial }^{\alpha }v\end{Vmatrix}}_{0}^{2}\right) }^{1/2},\;{\left| v\right| }_{k} \mathrel{\text{:= }} {\left( \mathop{\sum }\limits_{{\left| \alpha \right|  = k}}{\begin{Vmatrix}{\partial }^{\alpha }v\end{Vmatrix}}_{0}^{2}\right) }^{1/2}.\]

For \( k = 0,{H}^{0}\left( \Omega \right) \) is the standard \( {L}^{2}\left( \Omega \right) \) space with the inner product denoted by \( \left( {\cdot, \cdot  }\right) \). Similarly, for any subset \( K \subset  \Omega \), \( {L}^{2}\left( K\right) \) inner product is denoted by \( {\left( \cdot, \cdot  \right) }_{0, K} \). We note that, by a well-known property of Sobolev spaces, the assumption 3 implies that

\[a\left( {v, v}\right)  \gtrsim  \parallel v{\parallel }_{m,\Omega }^{2},\forall v \in  {H}^{m}\left( \Omega \right). \tag{5}\]

Next, we discuss the boundary conditions in detail. A popular type of boundary conditions are Dirichlet boundary condition when \( {B}^{k} = {B}_{D}^{k} \) are given by the following Dirichlet type trace operators

\[{B}_{D}^{k}\left( u\right)  \mathrel{\text{:= }} {\left. \frac{{\partial }^{k}u}{\partial {v}^{k}}\right| }_{\partial \Omega }\;\left( {0 \leq  k \leq  m - 1}\right), \tag{6}\]

with \( v \) being the outward unit normal vector of \( \partial \Omega \).

For the aforementioned Dirichlet boundary condition, the elliptic boundary value problem (1) is equivalent to

Minimization Problem M: Find \( u \in  {H}_{0}^{m}\left( \Omega \right) \) such that

\[J\left( u\right)  = \mathop{\min }\limits_{{v \in  {H}_{0}^{m}\left( \Omega \right) }}J\left( v\right) \tag{7}\]

with the energy function \( J \) defined by

\[J\left( v\right)  = \frac{1}{2}a\left( {v, v}\right)  - {\int }_{\Omega }{fvdx}, \tag{8}\]

and

\[a\left( {u, v}\right)  \mathrel{\text{:= }} \mathop{\sum }\limits_{{\left| \alpha \right|  = m}}{\left( {a}_{\alpha }{\partial }^{\alpha }u,{\partial }^{\alpha }v\right) }_{0,\Omega } + \left( {{a}_{0}u, v}\right). \tag{9}\]

Enforcing the boundary conditions for \( {H}_{0}^{m}\left( \Omega \right) \) in (7), while relatively easy for finite element methods, is difficult or impossible when using neural networks. As a result, we consider first the pure Neumann boundary conditions, which is equivalent to the minimization problem [7] over the whole space \( {H}^{m}\left( \Omega \right) \):

Minimization Problem N: Find \( u \in  {H}^{m}\left( \Omega \right) \) such that

\[u = \arg \mathop{\min }\limits_{{v \in  {H}^{m}\left( \Omega \right) }}J\left( v\right) \tag{10}\]

with energy function \( J \) defined by (8).

In this sense, the pure Neumann boundary conditions are the most natural and easiest to enforce, especially when optimizing over the class of neural network functions. It remains to be determined which form the pure Neumann boundary conditions, which we denote by \( {B}_{N}^{k} \), take when \( m \geq  2 \). This is given by the following result, which applies to any PDE operator (2).

Lemma 1. [53, Lemma 5.1] For each \( k = 0,1,\ldots, m - 1 \), there exists a bounded linear differential operator of order \( {2m} - k - 1 \):

\[{B}_{N}^{k}: {H}^{2m}\left( \Omega \right)  \mapsto  {L}^{2}\left( {\partial \Omega }\right) \tag{11}\]

such that the following identity holds

\[\left( {{Lu}, v}\right)  = a\left( {u, v}\right)  - \mathop{\sum }\limits_{{k = 0}}^{{m - 1}}{\left\langle  {B}_{N}^{k}\left( u\right),{B}_{D}^{k}\left( v\right) \right\rangle  }_{0,\partial \Omega }.\]

Namely

\[\mathop{\sum }\limits_{{\left| \alpha \right|  = m}}{\left( -1\right) }^{m}{\left( {\partial }^{\alpha }\left( {a}_{\alpha }{\partial }^{\alpha }u\right), v\right) }_{0,\Omega } = \mathop{\sum }\limits_{{\left| \alpha \right|  = m}}{\left( {a}_{\alpha }{\partial }^{\alpha }u,{\partial }^{\alpha }v\right) }_{0,\Omega } - \mathop{\sum }\limits_{{k = 0}}^{{m - 1}}{\left\langle  {B}_{N}^{k}\left( u\right),{B}_{D}^{k}\left( v\right) \right\rangle  }_{0,\partial \Omega } \tag{12}\]

for all \( u \in  {H}^{2m}\left( \Omega \right), v \in  {H}^{m}\left( \Omega \right) \). Furthermore,

\[\mathop{\sum }\limits_{{k = 0}}^{{m - 1}}{\begin{Vmatrix}{B}_{D}^{k}\left( u\right) \end{Vmatrix}}_{{L}^{2}\left( {\partial \Omega }\right) } + \mathop{\sum }\limits_{{k = 0}}^{{m - 1}}{\begin{Vmatrix}{B}_{N}^{k}\left( u\right) \end{Vmatrix}}_{{L}^{2}\left( {\partial \Omega }\right) } \lesssim  \parallel u{\parallel }_{{2m},\Omega }. \tag{13}\]

We are now in a position to state the pure Neumann boundary value problems for the PDE operator (2):

\[\begin{cases} {Lu} &  = f\text{ in }\Omega, \\  {B}_{N}^{k}\left( u\right) &  = 0\text{ on }\partial \Omega \;\left( {0 \leq  k \leq  m - 1}\right). \end{cases} \tag{14}\]

In light of the previous result, we note that problem (14) is equivalent to the optimization (10).

In order to handle Dirichlet boundary conditions, we consider the mixed boundary value problem:

\[\left\{  \begin{matrix} L{u}_{\delta } = f\;\text{ in }\Omega, \\  {B}_{D}^{k}\left( {u}_{\delta }\right)  + \delta {B}_{N}^{k}\left( {u}_{\delta }\right)  = 0,0 \leq  k \leq  m - 1. \end{matrix}\right. \tag{15}\]

It is well-known that (15) is equivalent to the following optimization problem:

\[{u}_{\delta } = \arg \mathop{\min }\limits_{{v \in  {H}^{m}\left( \Omega \right) }}{J}_{\delta }\left( v\right) \tag{16}\]

where

\[{J}_{\delta }\left( v\right)  = \frac{1}{2}{a}_{\delta }\left( {v, v}\right)  - \left( {f, v}\right) \tag{17}\]

and

\[{a}_{\delta }\left( {u, v}\right)  = a\left( {u, v}\right)  + {\delta }^{-1}\mathop{\sum }\limits_{{k = 0}}^{{m - 1}}{\left\langle  {B}_{D}^{k}\left( u\right),{B}_{D}^{k}\left( v\right) \right\rangle  }_{0,\partial \Omega }. \tag{18}\]

In order to handle the Dirichlet conditions, we will let \( \delta  \rightarrow  0 \) and use the theory developed in [53], specifically Lemma 5.4.

The further analysis in this paper will require that the solution \( u \) of the above problem be bounded in the \( {\mathcal{K}}_{1}\left( \mathbb{D}\right) \) -norm (recall the \( {\mathcal{K}}_{1}\left( \mathbb{D}\right) \) space studied in [44]). We note that the results in [44] combined with the standard regularity theory for linear elliptic problems implies this if the coefficients \( {a}_{\alpha } \) and \( {a}_{0} \) are smooth (see Lemma 5.5 and 5.6 in [53]).

### 2.2 Convex Dictionary Spaces

In order to approximate the solutions of (14) and (16), we aim to minimize the objective defined in (8) over a parameterized class of functions with finitely many parameters, i.e. to solve

\[{u}_{n} = \arg \mathop{\min }\limits_{{u \in  {V}_{n}}}J\left( u\right) \tag{19}\]

where \( {V}_{n} \subset  {H}^{m}\left( \Omega \right) \) is the set of functions parameterized by finitely many parameters. For instance, in the traditional conforming finite element method, \( {V}_{n} \) is taken to be a finite element space. We note that the minimizer above may not be unique, but we write equality by abuse of notation.

In our approach, we take the set of functions parameterized by finite expansions with respect to a dictionary \( \mathbb{D} \subset \; {H}^{m}\left( \Omega \right) \). Specifically, we consider the set

\[{\sum }_{n, M}\left( \mathbb{D}\right)  = \left\{  {\mathop{\sum }\limits_{{i = 1}}^{n}{a}_{i}{d}_{i},{d}_{i} \in  \mathbb{D},\mathop{\sum }\limits_{{i = 1}}^{n}\left| {a}_{i}\right|  \leq  M}\right\} . \tag{20}\]

Note that here we restrict the \( {\ell }^{1} \) -norm of the coefficients \( {a}_{i} \) in the expansion. This is necessary when we approximate the integrals in (8) by random sampling in order to control the random errors which are incurred. This means that

\[\overline{\mathop{\bigcup }\limits_{{n = 1}}^{\infty }{\sum }_{n, M}\left( \mathbb{D}\right) } = \left\{  {f \in  {H}^{m}\left( \Omega \right) : \parallel f{\parallel }_{{\mathcal{K}}_{1}\left( \mathbb{D}\right) } \leq  M}\right\} , \tag{21}\]

where \( {\mathcal{K}}_{1}\left( \mathbb{D}\right) \) is the space introduced in [7] and studied further in [44], which for the dictionary of rectified linear ridge functions is equivalent to the Barron space introduced in [11, 34]. For a traditional conforming finite element method, this closure would be \( {H}^{m}\left( \Omega \right) \). In our method, \( M \) is a parameter which must be adjusted depending upon the number of sample points used to approximate the integrals in (8). This is analogous to a regularization parameter in statistical learning.

#### 2.2.1 A counterexample to a Bernstein-type inequality for neural networks

Here we give a counterexample to a Bernstein-type inequality for the spaces \( {\sum }_{n, M}\left( \mathbb{D}\right) \) for \( \mathbb{D} = \{ \operatorname{ReLU}\left( {x + b}\right), b \in \; \left\lbrack  {-2,2}\right\rbrack  \} \). Bernstein’s inequality shown that for polynomials or trigonometric polynomials, higher order derivatives can be bounded in terms of lower order derivatives, with a constant depending upon the degree of the polynomial. For example, if \( {T}_{d} \) is a trigonometric polynomial of degree \( d \), then

\[{\begin{Vmatrix}{T}_{d}^{\prime }\end{Vmatrix}}_{{L}^{2}} \leq  n{\begin{Vmatrix}{T}_{d}\end{Vmatrix}}_{{L}^{2}} \tag{22}\]

Bernstein-type inequalities, i.e. inequalities which bound higher order derivatives in terms of lower order derivatives for certain classes of functions are an important tool in characterizing approximation spaces (see [8], Chapter 7, for instance). Consequently, a natural question is whether or not a Bernstein-type inequality can hold for the class of functions \( {\sum }_{n, M}\left( \mathbb{D}\right) \), with a constant potentially depending upon both \( n \) and \( M \). The following proposition shows that such an inequality cannot exist.

Proposition 1. For each \( \varepsilon  > 0 \), there exists a \( {u}_{\varepsilon } \in  {\sum }_{3,4}\left( \mathbb{D}\right) \) for \( \mathbb{D} = \{ \operatorname{ReLU}\left( {x + b}\right), b \in  \left\lbrack  {-2,2}\right\rbrack  \} \) which satisfies

\[{\begin{Vmatrix}{u}_{\varepsilon }^{\prime }\end{Vmatrix}}_{{L}^{2}\left( \left\lbrack  {-1,1}\right\rbrack  \right) } \geq  \sqrt{3}{\varepsilon }^{-1}{\begin{Vmatrix}{u}_{\varepsilon }\end{Vmatrix}}_{{L}^{2}\left( \left\lbrack  {-1,1}\right\rbrack  \right) }. \tag{23}\]

Proof. Consider the function

\[{u}_{\varepsilon }\left( x\right)  = \operatorname{ReLU}\left( {x + \varepsilon }\right)  - 2\operatorname{ReLU}\left( x\right)  + \operatorname{ReLU}\left( {x - \varepsilon }\right),\; x \in  \left\lbrack  {-1,1}\right\rbrack . \tag{24}\]

Clearly \( {u}_{\varepsilon } \in  {\sum }_{3,4}\left( \mathbb{D}\right) \) for \( \mathbb{D} = \{ \operatorname{ReLU}\left( {x + b}\right), b \in  \left\lbrack  {-2,2}\right\rbrack  \} \) and a direct calculation shows that

\[{\begin{Vmatrix}{u}_{\varepsilon }^{\prime }\end{Vmatrix}}_{{L}^{2}\left( \left\lbrack  {-1,1}\right\rbrack  \right) } = \sqrt{2}{\varepsilon }^{\frac{1}{2}},{\begin{Vmatrix}{u}_{\varepsilon }\end{Vmatrix}}_{{L}^{2}\left( \left\lbrack  {-1,1}\right\rbrack  \right) } = \frac{\sqrt{2}}{\sqrt{3}}{\varepsilon }^{\frac{3}{2}}, \tag{25}\]

which completes the proof.

## 3 Monte Carlo Sampling

In this section, we describe and analyze a procedure for approximating the integrals in (8) and (16) in high dimensions \( d \).

We consider first the case of the pure Neumann boundary conditions (8). In this case, we sample points \( {x}_{1},\ldots,{x}_{N} \in  \Omega \) uniformly at random and approximate the integrals by

\[{J}_{N}\left( u\right)  = \frac{1}{2N}\mathop{\sum }\limits_{{i = 1}}^{N}\mathop{\sum }\limits_{{\left| \alpha \right|  = m}}{a}_{\alpha }\left( {x}_{i}\right) {\left( {\partial }^{\alpha }u\left( {x}_{i}\right) \right) }^{2} + \frac{1}{2N}\mathop{\sum }\limits_{{i = 1}}^{N}{a}_{0}\left( {x}_{i}\right) u{\left( {x}_{i}\right) }^{2} - \frac{1}{N}\mathop{\sum }\limits_{{i = 1}}^{N}f\left( {x}_{i}\right) u\left( {x}_{i}\right). \tag{26}\]

Generating these random points may be difficult for complicated domains \( \Omega \), but for many common choices such as the sphere or cube, these points can be generated easily and efficiently.

In order to approximate solutions to (14), we propose to solve the following optimization problem

\[{u}_{n, M, N} = \arg \mathop{\min }\limits_{{v \in  {B}_{M}\left( \mathbb{D}\right) }}{J}_{N}\left( v\right), \tag{27}\]

where

\[{B}_{M}\left( \mathbb{D}\right)  = \left\{  {u \in  {\mathcal{K}}_{1}\left( \mathbb{D}\right) : \parallel u{\parallel }_{{\mathcal{K}}_{1}\left( \mathbb{D}\right) } \leq  M}\right\} . \tag{28}\]

Some result that says under which circumstances to have an estimate of \( \parallel u{\parallel }_{{\mathcal{K}}_{1}} \) can be found in [53] (see Lemma 5.6) and [31] (See Theorem 2.5 and Theorem 2.6). We note that the minimizer above is achieved as long as the \( \mathop{\sup }\limits_{{d \in  \mathbb{D}}}\parallel d{\parallel }_{{H}^{m}\left( \Omega \right) } < \infty \). For in this case the optimization is over a bounded set and \( {J}_{N} \) only depends upon the values and derivatives at finitely many points. Thus the optimization is effectively over a bounded set in a finite dimensional space, which is compact by the Heine-Borel Theorem.

For the case of mixed boundary conditions (16) we sample \( N \) points \( {x}_{1},\ldots,{x}_{N} \in  \Omega \) uniformly at random and also sample \( {N}_{0} \) points \( {y}_{1},\ldots,{y}_{{N}_{0}} \in  \partial \Omega \) uniformly at random from the boundary. We then approximate the integrals in (16) by

\[{J}_{N,\delta }\left( u\right)  = \frac{1}{2N}\mathop{\sum }\limits_{{i = 1}}^{N}\mathop{\sum }\limits_{{\left| \alpha \right|  = m}}{a}_{\alpha }\left( {x}_{i}\right) {\left( {\partial }^{\alpha }u\left( {x}_{i}\right) \right) }^{2} + \frac{1}{2N}\mathop{\sum }\limits_{{i = 1}}^{N}{a}_{0}\left( {x}_{i}\right) u{\left( {x}_{i}\right) }^{2} - \frac{1}{N}\mathop{\sum }\limits_{{i = 1}}^{N}f\left( {x}_{i}\right) u\left( {x}_{i}\right)  + \frac{{\delta }^{-1}}{{N}_{0}}\mathop{\sum }\limits_{{i = 1}}^{{N}_{0}}\mathop{\sum }\limits_{{k = 0}}^{{m - 1}}{\left| \frac{{\partial }^{k}}{\partial {\nu }^{k}}u\left( {y}_{i}\right) \right| }^{2}. \tag{29}\]

Forming the loss function (29) requires being able to calculate normal derivatives of functions \( u \) and to sample points uniformly from the boundary. This is possible for many important domains, such as the sphere or cube. We note that if the above sampling is not possible, then it suffices to be able to sample from a non-uniform distribution on \( \Omega \) and \( \partial \Omega \) with mass function \( \rho \) if we weight each of the sums above by \( {\rho }^{-1} \). We then propose to solve the following optimization problem to approximate solutions to (16)

\[{u}_{n, M, N,\delta } = \arg \mathop{\min }\limits_{{v \in  {B}_{M}\left( \mathbb{D}\right) }}{J}_{N,\delta }\left( v\right). \tag{30}\]

The optimization problems (27) and (30) can be efficiently approximately solved using greedy algorithms [2, 20, 25, 26, 54]. Specifically, the algorithm we use is the following

\[{u}_{0} = 0,{g}_{k} = \arg \mathop{\max }\limits_{{g \in  \mathbb{D}}}\left\langle  {\nabla {J}_{N}\left( {u}_{k - 1}\right), g}\right\rangle ,{u}_{k} = \left( {1 - {s}_{k}}\right) {u}_{k - 1} - M{s}_{k}{g}_{k}, \tag{31}\]

where \( {s}_{k} = \min \left( {1,\frac{2}{k}}\right) \). Importantly, we have that \( {u}_{k} \in  {\sum }_{k, M}\left( \mathbb{D}\right) \), so that this algorithm produces neural networks with finite width. Note that this algorithm requires the computation of an argmin over the dictionary \( \mathbb{D} \), which may be a complicated step for certain dictionaries. We assume in what follows that this step can be efficiently calculated, which we argue is a reasonable assumption in practice for a wide range of dictionaries. For instance, when using the dictionary \( {\mathbb{P}}_{k}^{d} \) corresponding to \( {\operatorname{ReLU}}^{k} \), this corresponds to an optimization over a compact \( d \) -dimensional set. Solving this problem for shallow neural network training was considered in [25] and numerical experiments demonstrating the application to PDEs can be found in [18].

The algorithm (31) was first introduced in [20] for the quadratic least-squares objective. It was analyzed in the context of neural networks for least squares fitting in [25] and for density estimation [26], and for general activation functions in [54]. Further extensions, such as the orthogonal greedy algorithm, relaxed greedy algorithm, and pure greedy algorithm, which however do not ensure that their iterates have bounded \( {\mathcal{K}}_{1}\left( \mathbb{D}\right) \) -norm, are studied in [29,28,29,45,47]. In what follows, we provide a convergence analysis which is applicable to our problem of interest.

More generally, we assume in our convergence analysis that the argmax in (31) is not solved exactly, but rather is approximated in the following sense

\[\left\langle  {\nabla {J}_{N}\left( {u}_{k - 1}\right),{g}_{k}}\right\rangle   \geq  \frac{1}{R}\mathop{\max }\limits_{{g \in  \mathbb{D}}}\left\langle  {\nabla {J}_{N}\left( {u}_{k - 1}\right), g}\right\rangle \tag{32}\]

for some \( R > 1 \). This is a more tractable problem for most dictionaries. However, we only consider the case of \( R > 1 \) in the convergence analysis given in this section. In later sections, we assume for simplicity that \( R = 1 \) in the analysis of the quadrature error.

For our analysis we will need that the objectives \( {J}_{N} \) and \( {J}_{N,\delta } \) are convex and \( K \) -smooth with respect to a Hilbert space norm \( H \). Recall that a function \( L: H \rightarrow  \mathbb{R} \) is \( K \) -smooth if

\[{J}_{N}\left( g\right)  \leq  {J}_{N}\left( f\right)  + \left\langle  {\nabla {J}_{N}\left( f\right), g - f}\right\rangle   + \frac{K}{2}\parallel g - f{\parallel }_{H}^{2}. \tag{33}\]

In the following analysis, we will let \( H \) be the discrete \( {H}^{k} \) norm corresponding to the sample points \( {x}_{1},\ldots,{x}_{N} \) and \( {y}_{1},\ldots,{y}_{{N}_{0}} \), i.e. \( H \) is given by

\[\langle u, v{\rangle }_{H} = \frac{1}{N}\mathop{\sum }\limits_{{i = 1}}^{N}\mathop{\sum }\limits_{{\left| \alpha \right|  = m}}\left( {{\partial }^{\alpha }u\left( {x}_{i}\right) }\right) \left( {{\partial }^{\alpha }v\left( {x}_{i}\right) }\right)  + \frac{1}{N}\mathop{\sum }\limits_{{i = 1}}^{N}u\left( {x}_{i}\right) v\left( {x}_{i}\right)  + \frac{1}{{N}_{0}}\mathop{\sum }\limits_{{i = 1}}^{{N}_{0}}\mathop{\sum }\limits_{{k = 0}}^{{m - 1}}{\left| \frac{{\partial }^{k}}{\partial {v}^{k}}u\left( {y}_{i}\right) \right| }^{2}. \tag{34}\]

With respect to the norm \( H \) above, the smoothness parameter \( K \) in the pure Neumann case (26) is the maximum of the coefficient functions \( {a}_{\alpha } \) and \( {a}_{0} \). In the case of mixed boundary conditions (29) the smoothness parameter is bounded by \( K \leq  \max \left( {{\begin{Vmatrix}{a}_{\alpha }\end{Vmatrix}}_{{L}^{\infty }},{\begin{Vmatrix}{a}_{0}\end{Vmatrix}}_{{L}^{\infty }}}\right)  + {\delta }^{-1} \).

We have the following convergence result for the algorithm (31).

Theorem 1. Suppose that the dictionary \( \mathbb{D} \) is symmetric and satisfies \( \mathop{\sup }\limits_{{d \in  \mathbb{D}}}\parallel d{\parallel }_{H} \leq  C < \infty \). Let the iterates \( {u}_{n} \) be given by the relaxed greedy algorithm (31) with \( {s}_{k} = \max \left( {1,\frac{2}{k}}\right) \), where the loss function \( L \) is convex and \( K \) -smooth (on the Hilbert space \( H \) ). Suppose that the argmax in (31) is approximated up to a factor \( R \) as in (32). Then we have \( {\begin{Vmatrix}{u}_{n}\end{Vmatrix}}_{{\mathcal{K}}_{1}\left( \mathbb{D}\right) } \leq  M \) and

\[L\left( {u}_{n}\right)  - \mathop{\inf }\limits_{{\parallel v{\parallel }_{{\mathcal{K}}_{1}\left( \mathbb{D}\right) } \leq  {R}^{-1}M}}L\left( v\right)  \leq  \frac{{32}{\left( CM\right) }^{2}K}{n}. \tag{35}\]

In particular, if \( M \geq  \parallel u{\parallel }_{{\mathcal{K}}_{1}\left( \mathbb{D}\right) } \) where \( u = \arg \mathop{\min }\limits_{v}L\left( v\right) \) is the global minimizer, then the objective converges to the optimal value in the above theorem.

We note that applying this theorem to the squared error loss \( L\left( f\right)  = \frac{1}{2}{\begin{Vmatrix}f - {f}^{ * }\end{Vmatrix}}_{H}^{2} \) implies an approximation rate of

\[\mathop{\inf }\limits_{{{f}_{n} \in  {\sum }_{n, M}}}{\begin{Vmatrix}{f}_{n} - {f}^{ * }\end{Vmatrix}}_{H} \lesssim  M{n}^{-\frac{1}{2}} \tag{36}\]

for any \( {f}^{ * } \in  {B}_{M}\left( \mathbb{D}\right) \). This is essentially the method of proof used in [20]. In fact, for special dictionaries \( \mathbb{D} \), the above rate can be improved. For instance, when \( \mathbb{D} = {\mathbb{P}}_{k}^{d} \) the optimal rate was determined in [44] to be \( {n}^{-1 - \frac{2\left( {k - m}\right)  + 1}{d}} \). Further, for more general activation functions it was shown that these rates can also be moderately improved in [43].

Proof. Since \( {u}_{0} = 0 \) and \( {u}_{k} \) is a convex combination of \( {u}_{k - 1} \) and \( - M{g}_{k} \), we see by induction that \( {\begin{Vmatrix}{u}_{k}\end{Vmatrix}}_{{\mathcal{K}}_{1}\left( \mathbb{D}\right) } \leq  M \).

The \( K \) -smoothness of the objective \( L \) implies that

\[L\left( {u}_{k}\right)  \leq  L\left( {u}_{k - 1}\right)  + \left\langle  {\nabla L\left( {u}_{k - 1}\right),{u}_{k} - {u}_{k - 1}}\right\rangle   + \frac{K}{2}{\begin{Vmatrix}{u}_{k} - {u}_{k - 1}\end{Vmatrix}}_{H}^{2}. \tag{37}\]

Using the iteration (31), we see that \( {u}_{k} - {u}_{k - 1} =  - {s}_{k}{u}_{k - 1} - M{s}_{k}{g}_{k} \). Plugging this into the above equation, we get

\[L\left( {u}_{k}\right)  \leq  L\left( {u}_{k - 1}\right)  - {s}_{k}\left\langle  {\nabla L\left( {u}_{k - 1}\right),{u}_{k - 1} + M{g}_{k}}\right\rangle   + \frac{K{s}_{k}^{2}}{2}{\begin{Vmatrix}{u}_{k - 1} + M{g}_{k}\end{Vmatrix}}_{H}^{2}. \tag{38}\]

Since the dictionary elements \( {g}_{k} \) satisfy \( {\begin{Vmatrix}{g}_{k}\end{Vmatrix}}_{H} \leq  C \) and \( {\begin{Vmatrix}{u}_{k - 1}\end{Vmatrix}}_{{\mathcal{K}}_{1}\left( \mathbb{D}\right) } \leq  M \), we see that \( {\begin{Vmatrix}{u}_{k - 1}\end{Vmatrix}}_{H} \leq  {CM} \) as well. Plugging this into the previous equation implies the bound

\[L\left( {u}_{k}\right)  \leq  L\left( {u}_{k - 1}\right)  - {s}_{k}\left\langle  {\nabla L\left( {u}_{k - 1}\right),{u}_{k - 1} + M{g}_{k}}\right\rangle   + 2{\left( CM\right) }^{2}K{s}_{k}^{2}. \tag{39}\]

Now let \( z \) with \( \parallel z{\parallel }_{{\mathcal{K}}_{1}\left( \mathbb{D}\right) } \leq  {R}^{-1}M \) be arbitrary. Then also \( \parallel  - z{\parallel }_{{\mathcal{K}}_{1}\left( \mathbb{D}\right) } \leq  {R}^{-1}M \) and the arg max characterization of \( {g}_{k} \) (32) implies that

\[\left\langle  {\nabla L\left( {u}_{k - 1}\right), - z}\right\rangle   \leq  \left\langle  {\nabla L\left( {u}_{k - 1}\right), M{g}_{k}}\right\rangle . \tag{40}\]

Using this in equation (39) gives

\[L\left( {u}_{k}\right)  \leq  L\left( {u}_{k - 1}\right)  - {s}_{n}\left\langle  {\nabla L\left( {u}_{k - 1}\right),{u}_{k - 1} - z}\right\rangle   + 2{\left( CM\right) }^{2}K{s}_{k}^{2}. \tag{41}\]

The convexity of \( L \) means that \( L\left( {u}_{k - 1}\right)  - L\left( z\right)  \leq  \left\langle  {\nabla L\left( {u}_{k - 1}\right),{u}_{k - 1} - z}\right\rangle \). Using this and subtracting \( L\left( z\right) \) from both sides of the above equation gives

\[L\left( {u}_{k}\right)  - L\left( z\right)  \leq  \left( {1 - {s}_{k}}\right) \left( {L\left( {u}_{k - 1}\right)  - L\left( z\right) }\right)  + 2{\left( CM\right) }^{2}K{s}_{k}^{2}. \tag{42}\]

Expanding the above recursion (using that \( {s}_{k} \leq  1 \) ), we get that

\[L\left( {u}_{n}\right)  - L\left( z\right)  \leq  \left( {\mathop{\prod }\limits_{{k = 1}}^{n}\left( {1 - {s}_{k}}\right) }\right) \left( {L\left( {u}_{0}\right)  - L\left( z\right) }\right)  + 2{\left( CM\right) }^{2}K\mathop{\sum }\limits_{{i = 1}}^{n}\left( {\mathop{\prod }\limits_{{k = i + 1}}^{n}\left( {1 - {s}_{k}}\right) }\right) {s}_{i}^{2}. \tag{43}\]

Using the choice \( {s}_{k} = \max \left( {1,\frac{2}{k}}\right) \), for which \( {s}_{1} = 1 \), we get

\[L\left( {u}_{n}\right)  - L\left( z\right)  \leq  2{\left( CM\right) }^{2}K\mathop{\sum }\limits_{{i = 1}}^{n}\left( {\mathop{\prod }\limits_{{k = i + 1}}^{n}\left( {1 - {s}_{k}}\right) }\right) {s}_{i}^{2}. \tag{44}\]

Finally, we bound the product \( \mathop{\prod }\limits_{{k = i + 1}}^{n}\left( {1 - {s}_{k}}\right) \) using that \( \log \left( {1 + x}\right)  \leq  x \) as

\[\log \left( {\mathop{\prod }\limits_{{k = i + 1}}^{n}\left( {1 - {s}_{k}}\right) }\right)  \leq   - \mathop{\sum }\limits_{{k = i + 1}}^{n}{s}_{k} =  - \mathop{\sum }\limits_{{k = i + 1}}^{n}\frac{2}{k} \leq   - {\int }_{i + 1}^{n + 1}\frac{2}{x}{dx} \leq  2\left( {\log \left( {i + 1}\right)  - \log \left( {n + 1}\right) }\right), \tag{45}\]

for \( i \geq  1 \). Thus, \( \mathop{\prod }\limits_{{k = i + 1}}^{n}\left( {1 - {s}_{k}}\right)  \leq  \frac{{\left( i + 1\right) }^{2}}{{\left( n + 1\right) }^{2}} \). Using this in equation (44), we get

\[L\left( {u}_{n}\right)  - L\left( z\right)  \leq  2{\left( CM\right) }^{2}K\mathop{\sum }\limits_{{i = 1}}^{n}\frac{{\left( i + 1\right) }^{2}}{{\left( n + 1\right) }^{2}}{s}_{i}^{2} \leq  8{\left( CM\right) }^{2}K\frac{1}{{\left( n + 1\right) }^{2}}\mathop{\sum }\limits_{{i = 1}}^{n}\frac{{\left( i + 1\right) }^{2}}{{i}^{2}}. \tag{46}\]

Crudely bounding \( \frac{{\left( i + 1\right) }^{2}}{{i}^{2}} \leq  4 \) for \( i \geq  1 \), we get

\[L\left( {u}_{n}\right)  - L\left( z\right)  \leq  {32}{\left( CM\right) }^{2}K\frac{n}{{\left( n + 1\right) }^{2}} \leq  \frac{{32}{M}^{2}K}{n}, \tag{47}\]

Taking the infimum over \( z \) with \( \parallel z{\parallel }_{{\mathcal{K}}_{1}\left( \mathbb{D}\right) } \leq  {R}^{-1}M \) gives the result.

Our goal in what follows will be to analyze the numerical error in solving (27) and (30).

### 3.1 Relationship with Statistical Learning Theory

The relationship between solving differential equations and statistical learning theory comes from the discretization of the integrals in (8) using Monte Carlo quadrature. Specifically, letting \( \mu \) denote the uniform measure on the set \( \Omega \), the energy can be written as

\[J\left( u\right)  = {\mathbb{E}}_{x \sim  \mu }\left( {\mathop{\sum }\limits_{{\left| \alpha \right|  = m}}{a}_{\alpha }\left( x\right) {\left\lbrack  {\partial }^{\alpha }u\left( x\right) \right\rbrack  }^{2}{dx} + {a}_{0}\left( x\right) u{\left( x\right) }^{2} - f\left( x\right) u\left( x\right) }\right). \tag{48}\]

In this way, the energy function \( J\left( u\right) \) plays the role of the (true) risk function in statistical learning theory [40]. However, an important difference is that here we know the probability distribution \( \mu \) explicitly, while in statistical applications the distribution \( \mu \) is usually not known. Rather, there is a dataset consisting of samples drawn from the distribution \( \mu \). In our case, we know \( \mu \) and generate this ’training data’ artificially. In high-dimensions, this Monte Carlo quadrature is necessary since deterministic quadrature is very inefficient for evaluating high-dimensional integrals.

The minimization problem (27) then corresponds to the empirical risk function in statistical learning theory. In order to ensure that the true risk, or energy, of the minimizer of the empirical risk is small, it is necessary to obtain a law of large number which is uniform in the class of functions we are optimizing over. This is done in the remainder of this section using the well-known tool of Rademacher complexity.

Finally, note that the optimization problem (27) incorporates the a priori bound \( {\begin{Vmatrix}{u}_{n, M, N}\end{Vmatrix}}_{{\mathcal{K}}_{1}\left( \mathbb{D}\right) } \leq  M \). We will see later that our error bounds depend upon the true solution \( u \) also satisfying the bound \( \parallel u{\parallel }_{{\mathcal{X}}_{1}\left( \mathbb{D}\right) } \leq  M \). The parameter \( M \) controls what is called the bias-variance trade-off in statistical learning. If we choose \( M \) too small, we won’t be able to capture the true solution \( u \) and both the empirical (27) and true (8) energy will be large. This represents the high bias regime. On the other hand, if \( M \) is very large, although we capture the true solution, which means that the empirical energy is small, we may overfit the 'data points,' and the true energy will be large. This represents the high variance regime.

In statistical learning, the amount of data is fixed and the hyperparameters of the model, which control the bias-variance trade-off, must be tuned using a validation dataset. Finally, the resulting model is tested once and for all on a test dataset. In the application to PDEs, however, we know the distribution \( \mu \) and can thus generate as much training data as we need. Consequently, once we have determined a suitable value of the parameter \( M \) in (27), we can generate as many datapoints \( N \) as we need to guarantee a good ’generalization accuracy,’ i.e. to guarantee that the true energy will be comparable to the empirical, or discrete, energy. Thus, for PDE applications, there is no need for a validation or test dataset since we can always ensure that there is enough 'training data' to validate the assumptions of our model.

The problem of how to determine a suitable parameter \( M \) remains. In order to solve this problem, we propose to solve the problem (27) for a given value \( {M}_{0} \), and then to adaptively increase \( M \) until the optimal energy determined in (27) stops decreasing. Each time we increase \( M \), we must sample more points \( N \) to ensure that we are not overfitting.

### 3.2 Rademacher Complexity

In order to bound the error incurred by approximating the true energy function (8) by the empirical energy function (26), we use the Rademacher complexity [3]. For more information on the Rademacher complexity and related concepts, we refer the reader to [40].

Given a class of functions \( \mathcal{F}: \Omega  \rightarrow  \mathbb{R} \), and a collection of sample points \( {x}_{1},\ldots,{x}_{N} \in  \Omega \), the empirical Rademacher complexity of \( \mathcal{F} \) is defined by

\[{\widetilde{R}}_{N}\left( \mathcal{F}\right)  = {\mathbb{E}}_{{\xi }_{1},\ldots,{\xi }_{N}}\left\lbrack  {\mathop{\sup }\limits_{{h \in  \mathcal{F}}}\frac{1}{N}\mathop{\sum }\limits_{{i = 1}}^{N}{\xi }_{i}h\left( {x}_{i}\right) }\right\rbrack , \tag{49}\]

where \( {\xi }_{1},\ldots,{\xi }_{n} \) are Rademacher random variables, i.e. uniformly distributed signs. The Rademacher complexity is obtained by averaging over the samples \( {x}_{i} \), which we take to be uniformly distributed over \( \Omega \), i.e. we have

\[{R}_{N}\left( \mathcal{F}\right)  = {\mathbb{E}}_{{x}_{1},\ldots,{x}_{N} \sim  \mu }{\mathbb{E}}_{{\xi }_{1},\ldots,{\xi }_{N}}\left\lbrack  {\mathop{\sup }\limits_{{h \in  \mathcal{F}}}\frac{1}{N}\mathop{\sum }\limits_{{i = 1}}^{N}{\xi }_{i}h\left( {x}_{i}\right) }\right\rbrack , \tag{50}\]

where \( \mu \) is the uniform distribution on \( \Omega \). For the mixed boundary value problem, we will also need the Rademacher complexity with respect to the uniform distribution on the boundary \( \partial \Omega \), which we denote by \( {R}_{\partial, N}\left( \mathcal{F}\right) \).

The utility of the Rademacher complexity is its role in giving a law of large numbers which is uniform over the class \( \mathcal{F} \), detailed by the following theorem.

Theorem 2. [48, Proposition 4.11] Let \( \mathcal{F} \) be a set of functions. Then

\[{\mathbb{E}}_{{x}_{1},\ldots,{x}_{N} \sim  \mu }\mathop{\sup }\limits_{{h \in  \mathcal{F}}}\left| {\frac{1}{N}\mathop{\sum }\limits_{{i = 1}}^{N}h\left( {x}_{i}\right) -\int h\left( x\right) {d\mu }}\right|  \leq  2{R}_{N}\left( \mathcal{F}\right). \tag{51}\]

### 3.3 Bounding the Integral Discretization Error

Our next step will be to apply the Rademacher complexity to bound the Monte Carlo discretization error in equations 26 and 29 uniformly over the class \( {B}_{m}\left( \mathbb{D}\right) \).

To this end, we introduce the class of functions

\[{\mathcal{F}}_{n, M} = \left\{  {\frac{1}{2}\mathop{\sum }\limits_{{\left| \alpha \right|  = m}}{a}_{\alpha }\left( x\right) {\left( {\partial }^{\alpha }u\left( x\right) \right) }^{2} + \frac{1}{2}{a}_{0}\left( x\right) u{\left( x\right) }^{2} - f\left( x\right) u\left( x\right) : u \in  {\sum }_{n, M}\left( \mathbb{D}\right) }\right\} . \tag{52}\]

and proceed to bound the Rademacher complexity \( {R}_{N}\left( {\mathcal{F}}_{n, M}\right) \). For this we will utilize the following fundamental lemma.

Lemma 2. Let \( \mathcal{F},\mathcal{S} \) be classes of functions on \( \Omega \). Then the following bounds hold.

- \( {R}_{N}\left( {\operatorname{conv}\left( \mathcal{F}\right) }\right)  = {R}_{N}\left( \mathcal{F}\right) \).

- Define the set \( \mathcal{F} + \mathcal{S} = \{ h\left( x\right)  + g\left( x\right) : h \in  \mathcal{F}, g \in  \mathcal{S}\} \). We have

\[{R}_{N}\left( {\mathcal{F} + \mathcal{S}}\right)  = {R}_{N}\left( \mathcal{F}\right)  + {R}_{N}\left( \mathcal{S}\right). \tag{53}\]

- Suppose that \( \phi : \mathbb{R} \rightarrow  \mathbb{R} \) is L-Lipschitz. Let \( \phi  \circ  \mathcal{F} = \{ \phi \left( {h\left( x\right) }\right) : h \in  \mathcal{F}\} \). Then

\[{R}_{N}\left( {\phi  \circ  \mathcal{F}}\right)  \leq  L{R}_{N}\left( \mathcal{F}\right). \tag{54}\]

- Suppose that \( f: \Omega  \rightarrow  \mathbb{R} \) is a fixed function. Let \( f \cdot  \mathcal{F} = \{ f\left( x\right) h\left( x\right) : h \in  \mathcal{F}\} \). Then

\[{R}_{N}\left( {f \cdot  \mathcal{F}}\right)  \leq  \parallel f\left( x\right) {\parallel }_{{L}^{\infty }\left( \Omega \right) }{R}_{N}\left( \mathcal{F}\right). \tag{55}\]

Proof. The first, second, and third of these statements are well-known facts, see [40, Lemma 26.7] for the first, [37, Page 56] for the second and [40, Lemma 26.9] for the third, so we only prove the fourth.

Suppose that \( \parallel f\left( x\right) {\parallel }_{{L}^{\infty }\left( \Omega \right) } \leq  1 \), the general results follows by a scaling argument. Let \( {x}_{1},\ldots,{x}_{N} \in  \Omega \) and consider the empirical Rademacher complexity

\[{\widetilde{R}}_{N}\left( {f \cdot  \mathcal{F}}\right)  = {\mathbb{E}}_{{\xi }_{1},\ldots,{\xi }_{N}}\left\lbrack  {\mathop{\sup }\limits_{{h \in  \mathcal{F}}}\frac{1}{N}\mathop{\sum }\limits_{{i = 1}}^{N}{\xi }_{i}f\left( {x}_{i}\right) h\left( {x}_{i}\right) }\right\rbrack . \tag{56}\]

We observe that the right-hand side of the above equation, being an average of a supremum of linear functions, is a convex function of \( \overrightarrow{f} = \left( {f\left( {x}_{1}\right),\ldots, f\left( {x}_{N}\right) }\right) \). Consequently, its maximum must be achieved at the extreme points of the set \( \left\{  {\overrightarrow{y}: \parallel \overrightarrow{y}{\parallel }_{\infty } \leq  1}\right\} \), which correspond to the points where each component is \( \pm  1 \). Thus we only need to consider the case where \( f\left( {x}_{i}\right)  = {\varepsilon }_{i} \in  \{  \pm  1\} \). But then

\[{\mathbb{E}}_{{\xi }_{1},\ldots,{\xi }_{N}}\left\lbrack  {\mathop{\sup }\limits_{{h \in  \mathcal{F}}}\frac{1}{N}\mathop{\sum }\limits_{{i = 1}}^{N}{\xi }_{i}{\varepsilon }_{i}h\left( {x}_{i}\right) }\right\rbrack   = {\mathbb{E}}_{{\xi }_{1},\ldots,{\xi }_{N}}\left\lbrack  {\mathop{\sup }\limits_{{h \in  \mathcal{F}}}\frac{1}{N}\mathop{\sum }\limits_{{i = 1}}^{N}{\xi }_{i}h\left( {x}_{i}\right) }\right\rbrack   = {\widetilde{R}}_{N}\left( \mathcal{F}\right), \tag{57}\]

since the \( {\varepsilon }_{i} \) simply permute the choices of sign \( {\xi }_{i} \) in the expectation. Taking an average over the sample points \( {x}_{1},\ldots,{x}_{N} \) completes the proof.

Utilizing this lemma, we prove the following bound on the Rademacher complexity of the set \( {\mathcal{F}}_{n, M} \) in (52).

Theorem 3. Suppose that \( {\begin{Vmatrix}{a}_{\alpha }\end{Vmatrix}}_{{L}^{\infty }\left( \Omega \right) },{\begin{Vmatrix}{a}_{0}\end{Vmatrix}}_{{L}^{\infty }\left( \Omega \right) } \leq  K \) and \( \mathop{\sup }\limits_{{d \in  \mathbb{D}}}\parallel d{\parallel }_{{W}^{m,\infty }} \leq  C \). Then the Rademacher complexity of the set \( {\mathcal{F}}_{n, M} \) in (52) is bounded by

\[{R}_{N}\left( {\mathcal{F}}_{n, M}\right)  \leq  {CKM}\mathop{\sum }\limits_{{\left| \alpha \right|  = m}}{R}_{N}\left( {{\partial }^{\alpha }\mathbb{D}}\right)  + {CKM}{R}_{N}\left( \mathbb{D}\right)  + \parallel f{\parallel }_{{L}^{\infty }\left( \Omega \right) }M{R}_{N}\left( \mathbb{D}\right), \tag{58}\]

where \( {\partial }^{\alpha }\mathbb{D} = \left\{  {{\partial }^{\alpha }d: d \in  \mathbb{D}}\right\} \).

Theorem 3 implies that to bound the Rademacher complexity of the set of interest, we only need to bound the Rademacher complexity of the derivatives of the dictionary \( \mathbb{D} \), which is a much simpler task. Later, we will detail how to do this for the specific dictionaries corresponding to shallow neural networks.

Proof. The proof is a straightforward application of Lemma 2. We begin by noting that

\[{\mathcal{F}}_{n, M} \subset  \mathop{\sum }\limits_{{\left| \alpha \right|  = m}}{a}_{\alpha } \cdot  \left\lbrack  {\phi  \circ  {\sum }_{n, M}\left( {{\partial }^{\alpha }\mathbb{D}}\right) }\right\rbrack   + {a}_{0} \cdot  \left\lbrack  {\phi  \circ  {\sum }_{n, M}\left( \mathbb{D}\right) }\right\rbrack   + f \cdot  {\sum }_{n, M}\left( \mathbb{D}\right), \tag{59}\]

where \( \phi \left( x\right)  = \frac{1}{2}{x}^{2} \).

Utilizing the first part of Lemma 2, we see that for all \( \alpha \)

\[{R}_{N}\left( {{\sum }_{n, M}\left( {{\partial }^{\alpha }\mathbb{D}}\right) }\right)  \leq  M{R}_{N}\left( {{\partial }^{\alpha }\mathbb{D}}\right). \tag{60}\]

The third part of the Lemma, combined with the bound \( \parallel d{\parallel }_{{W}^{m,\infty }} \leq  C \) and the fact that \( \phi \) is locally Lipschitz, imply that

\[{R}_{N}\left( {\phi  \circ  {\sum }_{n, M}\left( {{\partial }^{\alpha }\mathbb{D}}\right) }\right)  \leq  {CM}{R}_{N}\left( {{\partial }^{\alpha }\mathbb{D}}\right). \tag{61}\]

Finally, the second and fourth parts of the Lemma, combined with the bounds on \( {a}_{\alpha } \) and \( {a}_{0} \) complete the proof.

We are primarily interested in the following corollary of this result, which uniformly bound the Monte Carlo discretization error in both the pure Neumann and Dirichlet cases.

Corollary 1. Under the assumptions of Theorem 3 we have for the pure Neumann boundary conditions that

\[{\mathbb{E}}_{{x}_{1},\ldots,{x}_{N}}\mathop{\sup }\limits_{{{u}_{M} \in  {B}_{M}\left( \mathbb{D}\right) }}\left| {{J}_{N}\left( {u}_{M}\right)  - J\left( {u}_{M}\right) }\right|  \leq  {2CKM}\mathop{\sum }\limits_{{\left| \alpha \right|  = m}}{R}_{N}\left( {{\partial }^{\alpha }\mathbb{D}}\right)  + {2CKM}{R}_{N}\left( \mathbb{D}\right)  + 2\parallel f{\parallel }_{{L}^{\infty }\left( \Omega \right) }M{R}_{N}\left( \mathbb{D}\right). \tag{62}\]

For the mixed boundary conditions, we have

\[{\mathbb{E}}_{{x}_{1},\ldots,{x}_{N},{y}_{1},\ldots,{y}_{{N}_{0}}}\mathop{\sup }\limits_{{{u}_{M} \in  {B}_{M}\left( \mathbb{D}\right) }}\left| {{J}_{N,\delta }\left( {u}_{M}\right)  - {J}_{\delta }\left( {u}_{M}\right) }\right|  \leq  {2CKM}\mathop{\sum }\limits_{{\left| \alpha \right|  = m}}{R}_{N}\left( {{\partial }^{\alpha }\mathbb{D}}\right)  + {2CKM}{R}_{N}\left( \mathbb{D}\right)\]

\[+ 2\parallel f{\parallel }_{{L}^{\infty }\left( \Omega \right) }M{R}_{N}\left( \mathbb{D}\right)  + 2{\delta }^{-1}M\mathop{\sum }\limits_{{\left| \alpha \right|  < m}}{R}_{\partial, N}\left( {{\partial }^{\alpha }\mathbb{D}}\right). \tag{63}\]

Proof. This follows immediately by combining Theorem 3 with Theorem 2 and using Lemma 2 for the boundary terms in the case of mixed boundary conditions.

### 3.4 Rademacher Bounds for Neural Networks

In this section, we show how the Rademacher complexity can be bounded for dictionaries corresponding to shallow neural networks. Specifically, we consider dictionaries of the form

\[{\mathbb{D}}_{\sigma } = \{ \sigma \left( {\omega  \cdot  x + b}\right) : \left( {\omega, b}\right)  \in  \Theta \}  \subset  {H}^{m}\left( \Omega \right), \tag{64}\]

where the parameter set \( \Theta  \subset  {R}^{d + 1} \) is compact. Of particular importance are the dictionaries corresponding to \( {\mathrm{{ReLU}}}^{k} \) activation functions, \( {\mathbb{P}}_{k}^{d} \), which were introduced in [44]. Our main result is the following bound on the Rademacher complexity. This generalizes the results of [11], which calculate the Rademacher complexity of the unit ball in the Barron space for ReLU neural networks (see also [15], Theorem 2 and [21], Theorem 3).

Theorem 4. Suppose that \( \sigma  \in  {W}^{m + 1,\infty } \). Then for any \( \alpha \) with \( \left| \alpha \right|  \leq  m \), we have

\[{R}_{N}\left( {{\partial }^{\alpha }\mathbb{D}}\right)  \lesssim  {N}^{-\frac{1}{2}},{R}_{\partial, N}\left( {{\partial }^{\alpha }\mathbb{D}}\right)  \lesssim  {N}^{-\frac{1}{2}} \tag{65}\]

where the implied constant is independent of \( N \).

Proof. This results follows immediately upon noting that

\[{\partial }^{\alpha }\mathbb{D} = \left\{  {{\omega }^{\alpha }{\sigma }^{\left( \alpha \right) }\left( {\omega  \cdot  x + b}\right) : \left( {\omega, b}\right)  \in  \Theta }\right\} . \tag{66}\]

Since \( \Theta \) is a compact set, \( \left| {\omega }^{\alpha }\right| \) is bounded. Further, since \( \sigma  \in  {W}^{m + 1,\infty } \), we have that \( {\sigma }^{\left( \alpha \right) } \) is Lipschitz. Using the third point in Lemma 2, we obtained

\[{R}_{N}\left( {{\partial }^{\alpha }\mathbb{D}}\right)  \lesssim  {R}_{N}(\{ \omega  \cdot  x + b: \left( {\omega, b}\right)  \in  \Theta \} ), \tag{67}\]

and likewise for \( {R}_{\partial, N}\left( {{\partial }^{\alpha }\mathbb{D}}\right) \).

It is well-known that the Rademacher complexity of the set of linear functions is bounded by [40, Section 26.2]

\[{R}_{N}\left( \{ \omega  \cdot  x + b: \left( {\omega, b}\right)  \in  \Theta \} \right) \lesssim  {N}^{-\frac{1}{2}}. \tag{68}\]

for any distribution on \( x \) which is bounded almost surely. This applies both to the uniform distribution on \( \Omega \) as well as to the uniform distribution on \( \partial \Omega \), which completes the proof.

## 4 Monte Carlo Sampling Error Analysis

Finally, in this section, we collect the previous results to derive a rigorous error analysis for the approximate solution given in (27) and (30). We have the following convergence result in the pure Neumann case.

Theorem 5. Suppose that the conditions of Theorem 3 are satisfied, i.e. that in (8) we have \( {\begin{Vmatrix}{a}_{\alpha }\end{Vmatrix}}_{{L}^{\infty }\left( \Omega \right) },{\begin{Vmatrix}{a}_{0}\end{Vmatrix}}_{{L}^{\infty }\left( \Omega \right) } \leq  K \). In addition, assume that the dictionary \( \mathbb{D} \) satisfies \( \mathop{\sup }\limits_{{d \in  \mathbb{D}}}\parallel d{\parallel }_{{W}^{m,\infty }\left( \Omega \right) } < \infty \) and the Rademacher complexity bound

\[{R}_{N}\left( {{\partial }^{\alpha }\mathbb{D}}\right),{R}_{N}\left( \mathbb{D}\right)  \lesssim  {N}^{-\frac{1}{2}}. \tag{69}\]

Assume that the true solution \( u \in  {\mathcal{K}}_{1}\left( \mathbb{D}\right) \) satisfies \( \parallel u{\parallel }_{{\mathcal{K}}_{1}\left( \mathbb{D}\right) } \leq  M \) and let the numerical solution \( {u}_{n, M, N} \in  {\sum }_{n, M}\left( \mathbb{D}\right) \) be obtained by optimizing (27) using the algorithm (31) for \( n \) steps. Then we have

\[{\mathbb{E}}_{{x}_{1},\ldots,{x}_{N}}\left( {J\left( {u}_{n, M, N}\right)  - J\left( u\right) }\right)  \leq  M\left\lbrack  {{C}_{1}\left( {K + \parallel f{\parallel }_{{L}^{\infty }\left( \Omega \right) }}\right) {N}^{-\frac{1}{2}} + {C}_{2}{KM}{n}^{-1}}\right\rbrack , \tag{70}\]

where the constants \( {C}_{1} \) and \( {C}_{2} \) depend only upon the dictionary \( \mathbb{D} \), the dimension \( d \) and the order \( m \). Using the assumption that the coefficients \( {a}_{\alpha } \) are bounded away from 0, this implies in particular that

\[{\mathbb{E}}_{{x}_{1},\ldots,{x}_{N}}\left( {\begin{Vmatrix}{u}_{n, M, N} - u\end{Vmatrix}}_{{H}^{m}\left( \Omega \right) }^{2}\right)  \leq  M\left\lbrack  {{C}_{1}^{\prime }{N}^{-\frac{1}{2}} + {C}_{2}^{\prime }M{n}^{-1}}\right\rbrack , \tag{71}\]

where \( {C}_{1}^{\prime } \) and \( {C}_{2}^{\prime } \) depend only upon the dictionary and the differential operator.

Note that we have here an a priori bound which depends upon the \( {\mathcal{K}}_{1}\left( \mathbb{D}\right) \) -norm \( M \) of the true solution \( u \).

Proof. We write

\[{\mathbb{E}}_{{x}_{1},\ldots,{x}_{N}}\left( {J\left( {u}_{n, M, N}\right)  - J\left( u\right) }\right)  \leq  {\mathbb{E}}_{{x}_{1},\ldots,{x}_{N}}\left( {{J}_{N}\left( {u}_{n, M, N}\right)  - {J}_{N}\left( u\right) }\right)\]

\[+ {\mathbb{E}}_{{x}_{1},\ldots,{x}_{N}}\left( \left| {{J}_{N}\left( {u}_{n, M, N}\right)  - J\left( {u}_{n, M, N}\right) }\right| \right) \tag{72}\]

\[+ {\mathbb{E}}_{{x}_{1},\ldots,{x}_{N}}\left( \left| {{J}_{N}\left( u\right)  - J\left( u\right) }\right| \right) \text{. }\]

Using Theorem 2, the last two terms on the right are both bounded via the Rademacher complexity by

\[{CM}\left( {K + \parallel f{\parallel }_{{L}^{\infty }\left( \Omega \right) }}\right) {N}^{-\frac{1}{2}}\]

for an appropriate constant \( C \) since \( {u}_{n, M, N}, u \in  {B}_{M}\left( \mathbb{D}\right) \).

The first term on the right hand side of equation (72) is bounded by \( {C}^{\prime }K{M}^{2}{n}^{-1} \) using Theorem 1 for another constant \( {C}^{\prime } \). Specifically, see the remarks about the \( K \) -smoothness of \( {J}_{N} \) preceding Theorem 1 and use the assumption that \( \mathop{\sup }\limits_{{d \in  \mathbb{D}}}\parallel d{\parallel }_{{W}^{m,\infty }\left( \Omega \right) } < \infty \) to conclude that \( \mathbb{D} \) is uniformly bounded in the discrete \( {H}^{m} \) norm (34).

Collecting these bounds completes the proof.

Next, we give the convergence result in the case of mixed boundary conditions (16).

Theorem 6. Suppose that the conditions of Theorem 3 are satisfied, i.e. that in (8) we have \( {\begin{Vmatrix}{a}_{\alpha }\end{Vmatrix}}_{{L}^{\infty }\left( \Omega \right) },{\begin{Vmatrix}{a}_{0}\end{Vmatrix}}_{{L}^{\infty }\left( \Omega \right) } \leq  K \). In addition, assume that the dictionary \( \mathbb{D} \) satisfies \( \mathop{\sup }\limits_{{d \in  \mathbb{D}}}\parallel d{\parallel }_{{W}^{m,\infty }\left( \Omega \right) } < \infty \) and the Rademacher complexity bound

\[{R}_{N}\left( {{\partial }^{\alpha }\mathbb{D}}\right),{R}_{N}\left( \mathbb{D}\right),{R}_{\partial, N}\left( {{\partial }^{\alpha }\mathbb{D}}\right)  \lesssim  {N}^{-\frac{1}{2}}. \tag{73}\]

Assume that the true solution \( {u}_{\delta } \in  {\mathcal{K}}_{1}\left( \mathbb{D}\right) \) of (16) satisfies \( {\begin{Vmatrix}{u}_{\delta }\end{Vmatrix}}_{{\mathcal{K}}_{1}\left( \mathbb{D}\right) } \leq  M \) and let the numerical solution \( {u}_{n, M, N,\delta } \in \; {\sum }_{n, M}\left( \mathbb{D}\right) \) be obtained by optimizing (30) using the algorithm (31) for \( n \) steps. Then we have

\[{\mathbb{E}}_{{x}_{1},\ldots,{x}_{N},{y}_{1},\ldots,{y}_{{N}_{0}}}\left( {{J}_{\delta }\left( {u}_{n, M, N,\delta }\right)  - {J}_{\delta }\left( {u}_{\delta }\right) }\right)  \leq  M\left\lbrack  {{C}_{1}\left( {K + \parallel f{\parallel }_{{L}^{\infty }\left( \Omega \right) } + {\delta }^{-1}}\right) {N}^{-\frac{1}{2}} + {C}_{2}\left( {K + {\delta }^{-1}}\right) M{n}^{-1}}\right\rbrack , \tag{74}\]

where the constants \( {C}_{1} \) and \( {C}_{2} \) depend only upon the dictionary \( \mathbb{D} \), the dimension \( d \) and the order \( m \). Using the assumption that the coefficients \( {a}_{\alpha } \) are bounded away from 0, this implies in particular that

\[{\mathbb{E}}_{{x}_{1},\ldots,{x}_{N}}\left( {\begin{Vmatrix}{u}_{n, M, N,\delta } - {u}_{\delta }\end{Vmatrix}}_{{H}^{m}\left( \Omega \right) }^{2}\right)  \leq  M\left( {1 + {\delta }^{-1}}\right) \left\lbrack  {{C}_{1}^{\prime }{N}^{-\frac{1}{2}} + {C}_{2}^{\prime }M{n}^{-1}}\right\rbrack , \tag{75}\]

where \( {C}_{1}^{\prime } \) and \( {C}_{2}^{\prime } \) depend only upon the dictionary and the differential operator. The proof is completely analogous to the proof of Theorem 5 and we omit it for brevity.

We note that by Theorem 4 both Theorem 5 and 6 apply to the case when \( \mathbb{D} = {\mathbb{D}}_{\sigma } \) corresponds to shallow neural networks. Further, the difference between \( u \) and \( {u}_{\delta } \) is given by the following lemma.

Lemma 3. [53, Lemma 5.4] Let \( u \) be the solution of (7) and \( {u}_{\delta } \) be the solution of (16). Then

\[{\begin{Vmatrix}u - {u}_{\delta }\end{Vmatrix}}_{{H}^{m}} \lesssim  \sqrt{\delta }\parallel u{\parallel }_{{2m},\Omega }. \tag{76}\]

By applying Lemma 3 and choosing \( \delta \) appropriately, Theorem 6 can be used to obtain a result for the pure Dirichlet boundary conditions as well.

Theorem 7. Suppose that the conditions of Theorem 6 are satisfied. Assume that the true solution \( {u}_{\delta } \in  {\mathcal{K}}_{1}\left( \mathbb{D}\right) \) of (16) satisfies \( {\begin{Vmatrix}{u}_{\delta }\end{Vmatrix}}_{{\mathcal{K}}_{1}\left( \mathbb{D}\right) } \leq  M \) for all \( \delta  > 0 \). Let \( u \) be the solution of (7) and the numerical solution \( {u}_{n, M, N,\delta } \in  {\sum }_{n, M}\left( \mathbb{D}\right) \) be obtained by optimizing (30) using the algorithm (31) for \( n \) steps. Then we have

\[{\mathbb{E}}_{{x}_{1},\ldots,{x}_{N}}\left( {\begin{Vmatrix}{u}_{n, M, N,\delta } - u\end{Vmatrix}}_{{H}^{m}\left( \Omega \right) }^{2}\right)  \leq  M\left( {1 + {\delta }^{-1}}\right) \left\lbrack  {{C}_{1}^{\prime }{N}^{-\frac{1}{2}} + {C}_{2}^{\prime }M{n}^{-1}}\right\rbrack   + {C}_{R}\delta \parallel u{\parallel }_{{2m},\Omega }^{2}, \tag{77}\]

where \( {C}_{1}^{\prime } \) and \( {C}_{2}^{\prime } \) depend only upon the dictionary and the differential operator. In particularly, with the choice \( N = \mathcal{O}\left( {n}^{2}\right) \) and \( \delta  = \mathcal{O}\left( {n}^{-\frac{1}{2}}\right) \), we obtain the convergence rate

\[{\mathbb{E}}_{{x}_{1},\ldots,{x}_{N}}\left( {\begin{Vmatrix}{u}_{n, M, N,\delta } - u\end{Vmatrix}}_{{H}^{m}\left( \Omega \right) }^{2}\right)  \lesssim  {n}^{-\frac{1}{2}}. \tag{78}\]

## 5 Numerical Quadrature Analysis

Let \( {\mathcal{T}}_{h} = \{ T\} \) be a uniform partition of \( \Omega \) with mesh size \( h \) and \( {P}_{r}\left( {\mathcal{T}}_{h}\right) \) denote the piecewise degree \( r \) polynomials on \( {\mathcal{T}}_{h} \), i.e.

\[{P}_{r}\left( {\mathcal{T}}_{h}\right)  = \left\{  {p \in  {L}^{2}\left( \Omega \right) : {\left. p\left( x\right) \right| }_{T} \in  {P}_{r}\left( T\right),\forall T \in  {\mathcal{T}}_{h}}\right\} ,\]

where \( {P}_{r}\left( T\right) \) is the set of degree \( r \) polynomials on \( T \).

On each element \( T \in  {\mathcal{T}}_{h} \) and any given function \( g \), we use numerical quadrature to approximate \( {\int }_{T}g\left( x\right) {dx} \), namely

\[{\int }_{T}g\left( x\right) {dx} \approx  \mathop{\sum }\limits_{{\ell  = 1}}^{{n}_{T}}{w}_{T,\ell }g\left( {x}_{T,\ell }\right), \tag{79}\]

where the points \( {x}_{T,\ell } \) and weights \( {w}_{T,\ell } \) are chosen so the above approximation is exact for polynomials of degree \( r \). We collect the quadrature points for each of the elements \( T \) to obtain the deterministic numerical quadrature with quadrature points \( {x}_{i} \) and weights \( {w}_{i} \),

\[{\int }_{\Omega }g\left( x\right) {dx} = \mathop{\sum }\limits_{{T \in  {\mathcal{T}}_{h}}}{\int }_{T}g\left( x\right) {dx} \approx  \mathop{\sum }\limits_{{T \in  {\mathcal{T}}_{h}}}\mathop{\sum }\limits_{{\ell  = 1}}^{{n}_{T}}{w}_{T,\ell }g\left( {x}_{T,\ell }\right)  = \mathop{\sum }\limits_{{i = 1}}^{N}{w}_{i}g\left( {x}_{i}\right). \tag{80}\]

Note that \( {n}_{T} = n \) is a fixed number for each element \( T \) by the quadrature formula given in (79), so that \( N = O\left( {N}_{T}\right) \), where \( {N}_{T} \) is the number of elements of \( {\mathcal{T}}_{h} \). From this we see that \( N = O\left( {h}^{-d}\right) \).

We proceed to apply the deterministic numerical quadrature form (80) to approximate \( J\left( v\right) \) defined in (8), namely

\[{J}_{N}\left( v\right)  = \mathop{\sum }\limits_{{i = 1}}^{N}{w}_{i}\left( {\frac{1}{2}\mathop{\sum }\limits_{{\left| \alpha \right|  = m}}{a}_{\alpha }\left( {x}_{i}\right) {\left( {\partial }^{\alpha }v\left( {x}_{i}\right) \right) }^{2} + \frac{1}{2}{a}_{0}\left( {x}_{i}\right) {\left( v\left( {x}_{i}\right) \right) }^{2} - f\left( {x}_{i}\right) v\left( {x}_{i}\right) }\right). \tag{81}\]

Let \( {u}_{n, M, N} \in  {\sum }_{n, M}\left( \mathbb{D}\right) \) be obtained by applying algorithm (31) to solve

\[{u}_{n, M, N} = \arg \mathop{\min }\limits_{{v \in  {B}_{M}\left( \mathbb{D}\right) }}{J}_{N}\left( v\right) \tag{82}\]

for \( n \) steps, where \( {J}_{N}\left( \cdot \right) \) is defined by (81).

Theorem 8. Let \( u \) be the solution of (10). Suppose that \( u \in  {\mathcal{K}}_{1}\left( \mathbb{D}\right) \) and let \( {u}_{n, M, N} \in  {\sum }_{n, M} \) be obtained by solving (82) using the algorithm (31) with \( M \geq  \parallel u{\parallel }_{{\mathcal{K}}_{1}\left( \mathbb{D}\right) } \). Suppose that

- the coefficients and \( f \) in (8) satisfy \( {\begin{Vmatrix}{a}_{\alpha }\end{Vmatrix}}_{{W}^{r,\infty }\left( \Omega \right) },{\begin{Vmatrix}{a}_{0}\end{Vmatrix}}_{{W}^{r,\infty }\left( \Omega \right) },\parallel f{\parallel }_{{W}^{r,\infty }\left( \Omega \right) } \leq  {K}_{r} \), and

- the dictionary \( \mathbb{D} \) satisfies \( \mathop{\sup }\limits_{{d \in  \mathbb{D}}}\parallel d{\parallel }_{{W}^{m + r + 1,\infty }\left( \Omega \right) } = C < \infty \).

Then, if we choose the number of quadrature points as \( N = \mathcal{O}\left( {n}^{\frac{d}{r + 1}}\right) \), we get the estimate

\[{\begin{Vmatrix}{u}_{n, M, N} - u\end{Vmatrix}}_{a} \lesssim  {n}^{-\frac{1}{2}}. \tag{83}\]

Proof. By the assumption the quadrature points \( {x}_{i} \) and weights \( {w}_{i} \) satisfy

\[{\int }_{\Omega }g\left( x\right) {dx} - \mathop{\sum }\limits_{{i = 1}}^{N}{w}_{i}g\left( {x}_{i}\right)  = 0,\;\forall g\left( x\right)  \in  {P}_{r}\left( {\mathcal{T}}_{h}\right). \tag{84}\]

Using the Bramble-Hilbert Lemma [5], we get

\[\left| {{\int }_{\Omega }h\left( x\right) {dx} - \mathop{\sum }\limits_{{i = 1}}^{N}{w}_{i}h\left( {x}_{i}\right) }\right|  \leq  {C}_{r}{N}^{-\frac{r + 1}{d}}\parallel h{\parallel }_{r + 1,\infty }\;\forall h \in  {W}^{r + 1,\infty }\left( \Omega \right). \tag{85}\]

For any \( v \in  {W}^{m + r + 1,\infty }\left( \Omega \right) \), letting \( h\left( x\right)  = \frac{1}{2}\mathop{\sum }\limits_{{\left| \alpha \right|  = m}}{a}_{\alpha }\left( x\right) {\left( {\partial }^{\alpha }v\left( x\right) \right) }^{2} + \frac{1}{2}{a}_{0}\left( x\right) {\left( v\left( x\right) \right) }^{2} - f\left( x\right) v\left( x\right) \), we have

\[\left| {{J}_{N}\left( v\right)  - J\left( v\right) }\right|  \leq  {C}_{m}{C}_{r}{K}_{r}{N}^{-\frac{r + 1}{d}}\parallel v{\parallel }_{m + r + 1,\infty }. \tag{86}\]

We now calculate

\[\frac{1}{2}{\begin{Vmatrix}{u}_{n, M, N} - u\end{Vmatrix}}_{a}^{2} = J\left( {u}_{n, M, N}\right)  - J\left( u\right)  \leq  {J}_{N}\left( {u}_{n, M, N}\right)  - {J}_{N}\left( u\right)  + \left| {J\left( {u}_{n, M, N}\right)  - {J}_{N}\left( {u}_{n, M, N}\right) }\right|  + \left| {J\left( u\right)  - {J}_{N}\left( u\right) }\right|. \tag{87}\]

The last two terms above are bounded by equation (86) since we have \( {u}_{n, M, N}, u \in  {B}_{M}\left( \mathbb{D}\right) \) and thus \( {\begin{Vmatrix}{u}_{n, M, N}\end{Vmatrix}}_{{W}^{m + r + 1,\infty }\left( \Omega \right) },\parallel u{\parallel }_{{W}^{m + r + 1,\infty }\left( \Omega \right) } \leq  {CM} \). The first term is bounded by \( {C}^{\prime }K{M}^{2}{n}^{-1} \) for some constant \( {C}^{\prime } \). Putting this together, we get

\[\frac{1}{2}{\begin{Vmatrix}{u}_{n, M, N} - u\end{Vmatrix}}_{a}^{2} \lesssim  {N}^{-\frac{r + 1}{d}} + {n}^{-1}. \tag{88}\]

Choosing \( N = O\left( {n}^{\frac{d}{r + 1}}\right) \) completes the proof.

Now we proceed to apply the deterministic numerical quadrature to approximate \( {J}_{\delta }\left( v\right) \) defined in (17), namely

\[{J}_{N,\delta }\left( v\right)  = \mathop{\sum }\limits_{{i = 1}}^{N}{w}_{i}\left( {\frac{1}{2}\mathop{\sum }\limits_{{\left| \alpha \right|  = m}}{a}_{\alpha }\left( {x}_{i}\right) {\left( {\partial }^{\alpha }v\left( {x}_{i}\right) \right) }^{2} + \frac{1}{2}{a}_{0}\left( {x}_{i}\right) {\left( v\left( {x}_{i}\right) \right) }^{2} - f\left( {x}_{i}\right) v\left( {x}_{i}\right) }\right)  + {\delta }^{-1}\mathop{\sum }\limits_{{i = 1}}^{{N}_{0}}{w}_{i,0}\mathop{\sum }\limits_{{k = 0}}^{{m - 1}}{\left( \frac{{\partial }^{k}}{\partial {\nu }^{k}}v\left( {y}_{i}\right) \right) }^{2}, \tag{89}\]

here \( {w}_{i,0},{y}_{i},1 \leq  i \leq  {N}_{0} \) are the quadrature wights and points for approximating the integral \( \mathop{\sum }\limits_{{k = 0}}^{{m - 1}}{\left\langle  {B}_{D}^{k}\left( u\right),{B}_{D}^{k}\left( v\right) \right\rangle  }_{0,\partial \Omega } \).

Let \( {u}_{n, M, N,\delta } \in  {\sum }_{n, M}\left( \mathbb{D}\right) \) be obtained by solving

\[{u}_{n, M, N,\delta } = \arg \mathop{\min }\limits_{{v \in  {\sum }_{n, M}\left( \mathbb{D}\right) }}{J}_{N,\delta }\left( v\right) \tag{90}\]

using the greedy algorithm (31), where \( {J}_{N,\delta }\left( \cdot \right) \) is defined by (89).

Theorem 9. Let \( {u}_{\delta } \) be the solution of (16). Suppose that \( {u}_{\delta } \in  {\mathcal{K}}_{1}\left( \mathbb{D}\right) \) and let \( {u}_{n, M, N,\delta } \in  {\sum }_{n, M} \) be obtained by solving (90) using the algorithm (31) with \( M \geq  {\begin{Vmatrix}{u}_{\delta }\end{Vmatrix}}_{{\mathcal{K}}_{1}\left( \mathbb{D}\right) } \). Suppose that

- the coefficients and \( f \) in (17) satisfy \( {\begin{Vmatrix}{a}_{\alpha }\end{Vmatrix}}_{{W}^{r,\infty }\left( \Omega \right) },{\begin{Vmatrix}{a}_{0}\end{Vmatrix}}_{{W}^{r,\infty }\left( \Omega \right) },\parallel f{\parallel }_{{W}^{r,\infty }\left( \Omega \right) } \leq  {K}_{r} \), and

- the dictionary \( \mathbb{D} \) satisfies \( \mathop{\sup }\limits_{{d \in  \mathbb{D}}}\parallel d{\parallel }_{{W}^{m + r + 1,\infty }\left( \Omega \right) } = C < \infty \).

Then, we have

\[{\begin{Vmatrix}{u}_{n, M, N,\delta } - {u}_{\delta }\end{Vmatrix}}_{a} \lesssim  {N}^{-\frac{r + 1}{2d}} + \left( {1 + {\delta }^{-\frac{1}{2}}}\right) \left( {{N}_{0}^{-\frac{r + 1}{{2d} - 2}} + {n}^{-\frac{1}{2}}}\right). \tag{91}\]

The proof of Theorem 9 is similar to the proof of (8) and we omit it for brevity.

Further, by applying Lemma 3 and choosing \( \delta \) appropriately, Theorem 9 can be used to obtain a result for the pure Dirichlet boundary conditions as well.

Theorem 10. Suppose that the conditions of Theorem 9 are satisfied. Assume that the true solution \( {u}_{\delta } \in  {\mathcal{K}}_{1}\left( \mathbb{D}\right) \) of (16) satisfies \( {\begin{Vmatrix}{u}_{\delta }\end{Vmatrix}}_{{\mathcal{K}}_{1}\left( \mathbb{D}\right) } \leq  M \) for all \( \delta  > 0 \). Let \( u \) be the solution of (7) and the numerical solution \( {u}_{n, M, N,\delta } \in  {\sum }_{n, M}\left( \mathbb{D}\right) \) be obtained by solving \( \left( {90}\right) \) using the algorithm (31) with \( M \geq  {\begin{Vmatrix}{u}_{\delta }\end{Vmatrix}}_{{\mathcal{K}}_{1}\left( \mathbb{D}\right) } \). Then, if we choose \( \delta  = \mathcal{O}\left( {n}^{-\frac{1}{2}}\right) \) and the number of quadrature points as \( N = \mathcal{O}\left( {n}^{\frac{d}{{2r} + 2}}\right) \) and \( {N}_{0} = \mathcal{O}\left( {n}^{\frac{d - 1}{r + 1}}\right) \), we get the estimate

\[{\begin{Vmatrix}{u}_{n, M, N,\delta } - u\end{Vmatrix}}_{a} \lesssim  {n}^{-\frac{1}{4}}. \tag{92}\]

## 6 Conclusion

We have provided an analysis of the finite neuron method for both Monte Carlo and numerical quadrature. The key is to constrain the \( {\mathcal{K}}_{1}\left( \mathbb{D}\right) \) -norm of the numerical solution, which enables an a priori Rademacher complexity analysis. This is made possible by using a greedy algorithm instead of the commonly used stochastic gradient descent algorithm.

Similar to the analysis of many numerical methods in scientific computing, our analysis is based on an a priori assumption, namely on the \( {\mathcal{K}}_{1}\left( \mathbb{D}\right) \) -norm of the solution, which determines the choice of \( M \). For example the convergence analysis of Newton's method requires the initial iterate to be close enough to the minimizer. In addition, a priori error analysis for finite element methods relies on an a priori bound on the solution in an appropriate norm. Similar to the adaptive finite element method, developing a fully adaptive method with an a posteriori error analysis is a topic of ongoing work.

Another interesting further research direction is the study of more efficient algorithms for solving the argmax subproblem which occurs in equation (31). In addition, it would also be interesting to extend this analysis to more general PDEs, such as parabolic and hyperbolic equations.

## 7 Acknowledgements

We would like to thank Professors George Karniadakis, Jason Klusowski, Yulong Lu, Yeonjong Shin and Haizhao Yang for helpful discussions which inspired this work. This work was supported by the Verne M. Willaman Chair Fund at the Pennsylvania State University, and the National Science Foundation (Grant No. DMS-1819157).

## References

[1] Andrew R Barron. Universal approximation bounds for superpositions of a sigmoidal function. IEEE Transactions on Information theory, 39(3):930-945, 1993.

[2] Andrew R Barron, Albert Cohen, Wolfgang Dahmen, Ronald A DeVore, et al. Approximation and learning by greedy algorithms. The annals of statistics, 36(1):64-94, 2008.

[3] Peter L Bartlett and Shahar Mendelson. Rademacher and gaussian complexities: Risk bounds and structural results. Journal of Machine Learning Research, 3(Nov):463-482, 2002.

[4] Julius Berner, Philipp Grohs, and Arnulf Jentzen. Analysis of the generalization error: Empirical risk minimization over deep artificial neural networks overcomes the curse of dimensionality in the numerical approximation of black-scholes partial differential equations. SIAM Journal on Mathematics of Data Science, 2(3):631-657, 2020.

[5] James H Bramble and SR Hilbert. Estimation of linear functionals on sobolev spaces with application to fourier transforms and spline interpolation. SIAM Journal on Numerical Analysis, 7(1):112-124, 1970.

[6] Eric Cances, Virginie Ehrlacher, and Tony Lelievre. Greedy algorithms for high-dimensional non-symmetric linear problems. In ESAIM: Proceedings, volume 41, pages 95-131. EDP Sciences, 2013.

[7] Ronald A DeVore. Nonlinear approximation. Acta numerica, 7:51-150, 1998.

[8] Ronald A DeVore and George G Lorentz. Constructive approximation, volume 303. Springer Science & Business Media, 1993.

[9] Ronald A DeVore and Vladimir N Temlyakov. Some remarks on greedy algorithms. Advances in computational Mathematics, 5(1):173-187, 1996.

[10] Chenguang Duan, Yuling Jiao, Yanming Lai, Xiliang Lu, and Zhijian Yang. Convergence rate analysis for deep ritz method. arXiv preprint arXiv:2103.13330, 2021.

[11] Weinan E, Chao Ma, and Lei Wu. Barron spaces and the compositional function spaces for neural network models. arXiv preprint arXiv:1906.08039, 2019.

[12] Leonardo E Figueroa and Endre Süli. Greedy approximation of high-dimensional ornstein-uhlenbeck operators. Foundations of Computational Mathematics, 12(5):573-623, 2012.

[13] Yoav Freund and Robert E Schapire. A decision-theoretic generalization of on-line learning and an application to boosting. Journal of computer and system sciences, 55(1):119-139, 1997.

[14] Jerome H Friedman. Greedy function approximation: a gradient boosting machine. Annals of statistics, pages 1189-1232, 2001.

[15] Wei Gao and Zhi-Hua Zhou. Dropout rademacher complexity of deep neural networks. Science China Information Sciences, 59(7):1-12, 2016.

[16] Jiequn Han, Arnulf Jentzen, and Weinan E. Solving high-dimensional partial differential equations using deep learning. Proceedings of the National Academy of Sciences, 115(34):8505-8510, 2018.

[17] Jiequn Han and Jihao Long. Convergence of the deep bsde method for coupled fbsdes. Probability, Uncertainty and Quantitative Risk, 5(1):1-33, 2020.

[18] Wenrui Hao, Xianlin Jin, Jonathan W Siegel, and Jinchao Xu. An efficient training algorithm for neural networks and applications in PDEs. In preparation, 2021.

[19] Martin Hutzenthaler, Arnulf Jentzen, Thomas Kruse, Tuan Anh Nguyen, and Philippe von Wurstemberger. Overcoming the curse of dimensionality in the numerical approximation of semilinear parabolic partial differential equations. Proceedings of the Royal Society A, 476(2244):20190630, 2020.

[20] Lee K Jones. A simple lemma on greedy approximation in hilbert space and convergence rates for projection pursuit regression and neural network training. The annals of Statistics, 20(1):608-613, 1992.

[21] Sham M Kakade, Karthik Sridharan, and Ambuj Tewari. On the complexity of linear prediction: Risk bounds, margin bounds, and regularization. 2008.

[22] Jason M Klusowski and Andrew R Barron. Approximation by combinations of relu and squared relu ridge functions with 10 and 11 controls. IEEE Transactions on Information Theory, 64(12):7649-7656, 2018.

[23] Samuel Lanthaler, Siddhartha Mishra, and George Em Karniadakis. Error estimates for deeponets: A deep learning framework in infinite dimensions. arXiv preprint arXiv:2102.09618, 2021.

[24] Claude Le Bris, Tony Lelievre, and Yvon Maday. Results and questions on a nonlinear approximation approach for solving high-dimensional partial differential equations. Constructive Approximation, 30(3):621, 2009.

[25] Wee Sun Lee, Peter L Bartlett, and Robert C Williamson. Efficient agnostic learning of neural networks with bounded fan-in. IEEE Transactions on Information Theory, 42(6):2118-2132, 1996.

[26] Jonathan Q Li and Andrew R Barron. Mixture density estimation. In NIPS, volume 12, pages 279-285, 1999.

[27] Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Multipole graph neural operator for parametric partial differential equations. arXiv preprint arXiv:2006.09535, 2020.

[28] Evgenii Davidovich Livshits. Lower bounds for the rate of convergence of greedy algorithms. Izvestiya: Mathematics, 73(6):1197, 2009.

[29] ED Livshitz and VN Temlyakov. Two lower estimates in greedy approximation. Constructive approximation, 19(4):509-523, 2003.

[30] George G Lorentz, Manfred v Golitschek, and Yuly Makovoz. Constructive approximation: advanced problems, volume 304. Springer, 1996.

[31] Jianfeng Lu, Yulong Lu, and Min Wang. A priori generalization analysis of the deep ritz method for solving high dimensional elliptic equations. arXiv preprint arXiv:2101.01708, 2021.

[32] Lu Lu, Xuhui Meng, Zhiping Mao, and George Em Karniadakis. Deepxde: A deep learning library for solving differential equations. SIAM Review, 63(1):208-228, 2021.

[33] Tao Luo and Haizhao Yang. Two-layer neural networks for partial differential equations: Optimization and generalization theory. arXiv preprint arXiv:2006.15733, 2020.

[34] Chao Ma, Lei Wu, et al. A priori estimates of the population risk for two-layer neural networks. arXiv preprint arXiv:1810.06397, 2018.

[35] Siddhartha Mishra and Roberto Molinaro. Estimates on the generalization error of physics informed neural networks (pinns) for approximating pdes ii: A class of inverse problems. arXiv preprint arXiv:2007.01138, 2020.

[36] Siddhartha Mishra and T Konstantin Rusch. Enhancing accuracy of deep learning algorithms by training with low-discrepancy sequences. arXiv preprint arXiv:2005.12564, 2020.

[37] Mehryar Mohri, Afshin Rostamizadeh, and Ameet Talwalkar. Foundations of machine learning. MIT press, 2018.

[38] Johannes Müller and Marius Zeinhofer. Error estimates for the variational training of neural networks with boundary penalty. arXiv preprint arXiv:2103.01007, 2021.

[39] Maziar Raissi and George Em Karniadakis. Hidden physics models: Machine learning of nonlinear partial differential equations. Journal of Computational Physics, 357:125-141, 2018.

[40] Shai Shalev-Shwartz and Shai Ben-David. Understanding machine learning: From theory to algorithms. Cambridge university press, 2014.

[41] Yeonjong Shin, Jerome Darbon, and George Em Karniadakis. On the convergence of physics informed neural networks for linear second-order elliptic and parabolic type pdes. arXiv preprint arXiv:2004.01806, 2020.

[42] Yeonjong Shin, Zhongqiang Zhang, and George Em Karniadakis. Error estimates of residual minimization using neural networks for linear pdes. arXiv preprint arXiv:2010.08019, 2020.

[43] Jonathan W Siegel and Jinchao Xu. Approximation rates for neural networks with general activation functions. Neural Networks, 2020.

[44] Jonathan W. Siegel and Jinchao Xu. Optimal approximation rates and metric entropy of \( {\mathrm{{ReLU}}}^{k} \) and cosine networks. arXiv preprint arXiv:2101.12365, 2021.

[45] AV Sil’nichenko. Rate of convergence of greedy algorithms. Mathematical Notes, 76(3):582-586, 2004.

[46] Hwijae Son, Jin Woo Jang, Woo Jin Han, and Hyung Ju Hwang. Sobolev training for the neural network solutions of pdes. arXiv preprint arXiv:2101.08932, 2021.

[47] Vladimir N Temlyakov. Greedy approximation. Acta Numerica, 17(235):409, 2008.

[48] Martin J Wainwright. High-dimensional statistics: A non-asymptotic viewpoint, volume 48. Cambridge University Press, 2019.

[49] Sifan Wang, Hanwen Wang, and Paris Perdikaris. On the eigenvector bias of fourier feature networks: From regression to solving multi-scale pdes with physics-informed neural networks. arXiv preprint arXiv:2012.10047, 2020.

[50] Sifan Wang, Xinling Yu, and Paris Perdikaris. When and why pinns fail to train: A neural tangent kernel perspective. arXiv preprint arXiv:2007.14527, 2020.

[51] E Weinan, Chao Ma, and Lei Wu. A comparative analysis of optimization and generalization properties of two-layer neural network and random feature models under gradient descent dynamics. Science China Mathematics, pages 1-24, 2020.

[52] E Weinan and Bing Yu. The deep ritz method: a deep learning-based numerical algorithm for solving variational problems. Communications in Mathematics and Statistics, 6(1):1-12, 2018.

[53] Jinchao Xu. Finite neuron method and convergence analysis. Communications in Computational Physics, 28(5):1707-1745, 2020.

[54] Tong Zhang. Sequential greedy approximation for certain convex optimization problems. IEEE Transactions on Information Theory, 49(3):682-691, 2003.
