

# Finite element approximations of symmetric tensors on simplicial grids in \( {\mathbb{R}}^{n} \): The lower order case

Jun Hu

LMAM, School of Mathematical Sciences, and

Beijing International Center for Mathematical Research,

Peking University, Beijing 100871, P. R. China

hujun@math.pku.edu.cn

Shangyou Zhang

Department of Mathematical Sciences, University of Delaware,

Newark, DE 19716, USA

szhang@udel.edu

Received 6 May 2015

Revised 11 March 2016

Accepted 27 April 2016

Published 18 July 2016

Communicated by D. Arnold

In this paper, we construct, in a unified fashion, lower order finite element subspaces of spaces of symmetric tensors with square-integrable divergence on a domain in any dimension. These subspaces are essentially the symmetric tensor finite element spaces of order \( k \) from [Finite element approximations of symmetric tensors on simplicial grids in \( {\mathbb{R}}^{n} \): The higher order case, J. Comput. Math. 33 (2015) 283-296], enriched, for each \( \left( {n - 1}\right) \) -dimensional simplex, by \( \frac{\left( {n + 1}\right) n}{2} \) face bubble functions in the symmetric tensor finite element space of order \( n + 1 \) from [Finite element approximations of symmetric tensors on simplicial grids in \( {\mathbb{R}}^{n} \): The higher order case, J. Comput. Math. 33 (2015) 283- 296] when \( 1 \leq  k \leq  n - 1 \), and by \( \frac{\left( {n - 1}\right) n}{2} \) face bubble functions in the symmetric tensor finite element space of order \( n + 1 \) from [Finite element approximations of symmetric tensors on simplicial grids in \( {\mathbb{R}}^{n} \): The higher order case, J. Comput. Math. 33 (2015) 283- 296] when \( k = n \). These spaces can be used to approximate the symmetric matrix field in a mixed formulation problem where the other variable is approximated by discontinuous piecewise \( {P}_{k - 1} \) polynomials. This in particular leads to first-order mixed elements on simplicial grids with total degrees of freedom per element 18 plus 3 in 2D, 48 plus 6 in 3D. The previous record of the degrees of freedom of first-order mixed elements is, 21 plus 3 in 2D, and 156 plus 6 in 3D, on simplicial grids. We also derive, in a unified way which is completely different from those used in [D. Arnold, G. Awanou and R. Winther, Finite elements for symmetric tensors in three dimensions, Math. Comput. 77 (2008) 1229-1251; D. N. Arnold and R. Winther, Mixed finite element for elasticity, Number Math. 92 (2002) 401-419], a family of Arnold-Winther mixed finite elements in any space dimension. One example in this family is the Raviart-Thomas elements in one dimension, the second example is the mixed finite elements for linear elasticity in two

dimensions due to Arnold and Winther, the third example is the mixed finite elements for linear elasticity in three dimensions due to Arnold, Awanou and Winther.

Keywords: Mixed finite element; symmetric finite element; first-order system; simplicial grid; inf-sup condition.

AMS Subject Classification: 65N30, 73C02

## 1. Introduction

The constructions, using polynomial-shape functions, of stable pairs of finite element spaces for approximating the pair of spaces \( H\left( {\operatorname{div},\Omega;\mathbb{S}}\right)  \times  {L}^{2}\left( {\Omega;{\mathbb{R}}^{n}}\right) \) in first-order systems are a long-standing, challenging and open problem, see for instance, Refs. 4 and 6. For mixed finite elements of linear elasticity, many mathematicians have been working on this problem and compromised to weakly symmetric or composite elements, cf. Refs. 3, 7, 8, 34, 36, 37, 39-42 and 45. It is not until 2002 that Arnold and Winther were able to propose the first family of mixed finite element spaces with polynomial-shape functions in two dimensions. \( {}^{10} \) Such a two-dimensional family was extended to a three-dimensional family of mixed elements, \( {}^{6} \) while the lowest order element with \( k = 2 \) was first proposed in Ref. 2. We refer interested readers to Refs. 2, 5, 6, 10, 12, 17-19, 11, 24, 31, 35, 43, 44, 9, 13, 20, 25, 26, 27, 30 and 29, for recent progress on mixed finite elements for linear elasticity. See Ref. 14 for the DPG method for the linear elasticity problem.

In Refs. 32 and 33, Hu and Zhang proposed new ideas to design discrete stress spaces and analyze the discrete inf-sup condition. In particular, they were able to construct suitable \( H\left( {\operatorname{div},\Omega;\mathbb{S}}\right)  - {P}_{k} \) space, namely, \( {\Sigma }_{k, h} \) defined in (4.1) below, with \( k \geq  3 \) for \( 2\mathrm{D} \), and \( k \geq  4 \) for \( 3\mathrm{D} \), finite element spaces for the stress discretization in both two and three dimensions. In Ref 28, Hu constructed, in a unified fashion, suitable \( H\left( {\operatorname{div},\Omega;\mathbb{S}}\right)  - {P}_{k} \) space with \( k \geq  n + 1 \), and proposed a set of degrees of freedom for the shape function space, in any dimension.

The purpose of this paper is to extend those elements in Ref. 28 to lower order cases where \( 1 \leq  k \leq  n \). Since it is, at moment, very difficult to prove that the pair of the space \( {\Sigma }_{k, h} \) and the \( {L}^{2} - {P}_{k - 1} \) space, namely, \( {V}_{k, h} \) defined in (4.2), is stable, the \( {\Sigma }_{k, h} \) space has to be enriched by some higher order polynomials whose divergence are in \( {V}_{k, h} \). Thanks to Ref. 28, it suffices to control the piecewise rigid motion space. Hence, we only need to add, for each \( \left( {n - 1}\right) \) -dimensional simplex, \( \frac{\left( {n + 1}\right) n}{2} \) simplex bubble functions in \( {\Sigma }_{n + 1, h} \) when \( 2 \leq  k \leq  n - 1 \), and \( \frac{\left( {n - 1}\right) n}{2} \) simplex bubble functions in \( {\Sigma }_{n + 1, h} \) when \( k = n \). This in particular leads to first-order mixed elements on simplicial grids with total degrees of freedom per element 18 plus 3 in 2D, 48 plus 6 in 3D. The previous record of the degrees of freedom of first-order mixed elements is 21 plus 3 in 2D, and 156 plus 6 in 3D, on simplicial grids. These enriched bubble functions belong to the lowest order space from a family of Arnold-Winther mixed finite elements in any space dimension which, together with the \( {V}_{k, h} \) space, form a stable pair of spaces for first-order systems. Note that these spaces in this family of Arnold-Winther mixed finite elements are constructed in a unified and direct way which is completely different from those used in Refs. 6 and 10. One example in this family is the Raviart-Thomas elements in one dimension, the second example is the mixed finite elements for linear elasticity in two dimensions due to Arnold and Winther, \( {}^{10} \) the third example is the mixed finite elements for linear elasticity in three dimensions due to Arnold, Awanou and Winther. \( {}^{6} \)

We end this section by introducing first-order systems and related notations. We consider mixed finite element methods \( {}^{22} \) of first-order systems with symmetric tensors: find \( \left( {\sigma, u}\right)  \in  \Sigma  \times  V \mathrel{\text{:= }} H\left( {\operatorname{div},\Omega;\mathbb{S}}\right)  \times  {L}^{2}\left( {\Omega;{\mathbb{R}}^{n}}\right) \), such that

\[\left\{  \begin{array}{ll} \left( {{A\sigma },\tau }\right)  + \left( {\operatorname{div}\tau, u}\right)  = 0 & \text{ for all }\tau  \in  \Sigma, \\  \left( {\operatorname{div}\sigma, v}\right)  = \left( {f, v}\right) & \text{ for all }v \in  V. \end{array}\right. \tag{1.1}\]

Here the symmetric tensor space for the stress \( \Sigma \) is defined by

\[H\left( {\operatorname{div},\Omega;\mathbb{S}}\right)  \mathrel{\text{:= }} \left\{  {\left. {\tau  = \left( \begin{matrix} {\tau }_{11} & \cdots & {\tau }_{1n} \\  \vdots & \vdots & \vdots \\  {\tau }_{n1} & \cdots & {\tau }_{nn} \end{matrix}\right)  \in  H\left( {\operatorname{div},\Omega;{\mathbb{R}}^{n \times  n}}\right) }\right| \;{\tau }^{T} = \tau }\right\} , \tag{1.2}\]

and the space for the vector displacement \( V \) is

\[{L}^{2}\left( {\Omega;{\mathbb{R}}^{n}}\right)  \mathrel{\text{:= }} \left\{  {{\left( {u}_{1},\ldots,{u}_{n}\right) }^{T} \mid  {u}_{i} \in  {L}^{2}\left( \Omega \right), i = 1,\ldots, n}\right\} . \tag{1.3}\]

This paper denotes by \( {H}^{k}\left( {T; X}\right) \) the Sobolev space \( {}^{1} \) consisting of functions with domain \( T \subset  {\mathbb{R}}^{n} \), taking values in the finite-dimensional vector space \( X \), and with all derivatives of order at most \( k \) square-integrable. For our purposes, the range space \( X \) will be either \( \mathbb{S},{\mathbb{R}}^{n} \), or \( \mathbb{R} \). Let \( \parallel  \cdot  {\parallel }_{k, T} \) be the norm of \( {H}^{k}\left( T\right) \), and \( \mathbb{S} \) denote the space of symmetric tensors, and \( H\left( {\operatorname{div}, T;\mathbb{S}}\right) \) consist of square-integrable symmetric matrix fields with square-integrable divergence. The \( H \) (div)-norm is defined by

\[\parallel \tau {\parallel }_{H\left( {\operatorname{div}, T}\right) }^{2} \mathrel{\text{:= }} \parallel \tau {\parallel }_{0, T}^{2} + \parallel \operatorname{div}\tau {\parallel }_{0, T}^{2}.\]

Let \( {L}^{2}\left( {T;{\mathbb{R}}^{n}}\right) \) be the space of vector-valued functions which are square-integrable. Here, the compliance tensor \( A = A\left( x\right) : \mathbb{S} \rightarrow  \mathbb{S} \), characterizing the properties of the material, is bounded and symmetric positive definite uniformly for \( x \in  \Omega \).

The rest of the paper is organized as follows. In the next section, we present some preliminary results from Ref. 28; see also Refs. 32 and 33, for the cases \( n = 2 \) and \( n = 3 \), respectively. In Sec. 3, based on these preliminary results, we propose a family of Arnold-Winther mixed finite elements in any space dimension. In Sec. 4, we present lower order mixed finite elements and analyze the well-posedness of the discrete problem and error estimates of the approximation solution. In Sec. 5, we present the first-order mixed elements. The paper ends with Sec. 6 which lists some numerics.

## 2. Preliminary Results

Suppose that the domain \( \Omega \) is subdivided by a family of shape regular simplicial grids \( {\mathcal{T}}_{h} \) (with the grid size \( h \) ). For any edge \( {\mathbf{x}}_{i}{\mathbf{x}}_{j} \) of element \( K, i \neq  j \), let \( {\mathbf{t}}_{i, j} \) denote associated tangent vectors, which allow for us to introduce the following symmetric matrices of rank one

\[{T}_{i, j} \mathrel{\text{:= }} {\mathbf{t}}_{i, j}{\mathbf{t}}_{i, j}^{T},\;0 \leq  i < j \leq  n. \tag{2.1}\]

For these matrices of rank one, we have the following result from Ref. 28; see also Refs. 32 and 33, for the cases \( n = 2 \) and \( n = 3 \), respectively.

Lemma 2.1. The \( \frac{\left( {n + 1}\right) n}{2} \) symmetric tensors \( {T}_{i, j} \) in (2.1) are linearly independent, and form a basis of \( \mathbb{S} \).

With these symmetric matrices \( {T}_{i, j} \) of rank one, we define an \( H\left( {\operatorname{div}, K;\mathbb{S}}\right) \) bubble function space

\[{\Sigma }_{K, k, b} \mathrel{\text{:= }} \mathop{\sum }\limits_{{0 \leq  i < j \leq  n}}{\lambda }_{i}{\lambda }_{j}{P}_{k - 2}\left( {K;\mathbb{R}}\right) {T}_{i, j}, \tag{2.2}\]

where \( {\lambda }_{i}, i = 0,\ldots, n \), are the barycenter coordinates of element \( K \). Define the full \( H\left( {\operatorname{div}, K;\mathbb{S}}\right) \) bubble function space consisting of polynomials of degree \( \leq  k \):

\[{\Sigma }_{\partial K, k,0} \mathrel{\text{:= }} \left\{  {{\left. \tau  \in  {P}_{k}\left( K;\mathbb{S}\right),\tau \nu \right| }_{\partial K} = 0}\right\} . \tag{2.3}\]

Here \( \nu \) is the normal vector of \( \partial K \). We have the following result due to Ref. 28.

Lemma 2.2. It holds that

\[{\Sigma }_{K, k, b} = {\Sigma }_{\partial K, k,0}. \tag{2.4}\]

Let \( {\Sigma }_{K, b, h} \) denote the sum of these \( H\left( {\operatorname{div}, K;\mathbb{S}}\right) \) bubble function spaces, namely,

\[{\Sigma }_{k, b, h} \mathrel{\text{:= }} \mathop{\sum }\limits_{{K \in  {\mathcal{T}}_{h}}}{\Sigma }_{K, k, b}. \tag{2.5}\]

We need an important result concerning the divergence space of the bubble function space. To this end, we introduce the following rigid motion space on each element \( K \):

\[R\left( K\right)  \mathrel{\text{:= }} \left\{  {v \in  {H}^{1}\left( {K;{\mathbb{R}}^{n}}\right),\epsilon \left( v\right)  \mathrel{\text{:= }} \left( {\nabla v + \nabla {v}^{T}}\right) /2 = 0}\right\} . \tag{2.6}\]

It follows from the definition that \( R\left( K\right) \) is a subspace of \( {P}_{1}\left( {K;{\mathbb{R}}^{n}}\right) \). For \( n = 1 \), \( R\left( K\right) \) is the constant function space over \( K \). The dimension of \( R\left( K\right) \) is \( \frac{n\left( {n + 1}\right) }{2} \). For two dimensions, the rigid motion space \( R\left( K\right) \) is

\[R\left( K\right)  \mathrel{\text{:= }} \left\{  {\left( \begin{array}{l} {a}_{1} \\  {a}_{2} \end{array}\right)  + b\left( \begin{array}{r}  - {x}_{2} \\  {x}_{1} \end{array}\right),{a}_{1},{a}_{2}, b \in  \mathbb{R}}\right\} ; \tag{2.7}\]

for three dimensions, the rigid motion space \( R\left( K\right) \) reads

\[R\left( K\right)  \mathrel{\text{:= }} \left\{  {\left( \begin{array}{l} {a}_{1} \\  {a}_{2} \\  {a}_{3} \end{array}\right)  + {b}_{1}\left( \begin{matrix}  - {x}_{2} \\  {x}_{1} \\  0 \end{matrix}\right)  + {b}_{2}\left( \begin{matrix}  - {x}_{3} \\  0 \\  {x}_{1} \end{matrix}\right)  + {b}_{3}\left( \begin{array}{r} 0 \\   - {x}_{3} \\  {x}_{2} \end{array}\right),{a}_{i},{b}_{i} \in  \mathbb{R}, i = 1,2,3}\right\} . \tag{2.8}\]

This allows for defining the orthogonal complement space of \( R\left( K\right) \) with respect to \( {P}_{k - 1}\left( {K;{\mathbb{R}}^{n}}\right) \) by

\[{R}^{ \bot  }\left( K\right)  \mathrel{\text{:= }} \left\{  {v \in  {P}_{k - 1}\left( {K;{\mathbb{R}}^{n}}\right),{\left( v, w\right) }_{K} = 0\text{ for any }w \in  R\left( K\right) }\right\} , \tag{2.9}\]

where the inner product \( {\left( v, w\right) }_{K} \) over \( K \) reads \( {\left( v, w\right) }_{K} = {\int }_{K}v \cdot  {wd}\mathbf{x} \). When \( k = 1 \) we have \( {R}^{ \bot  }\left( K\right)  = \{ 0\} \).

Lemma 2.3. For any \( K \in  {\mathcal{T}}_{h} \), it holds that

\[\operatorname{div}{\Sigma }_{K, k, b} = {R}^{ \bot  }\left( K\right). \tag{2.10}\]

Proof. The proof can be found in Ref. 28; see also Refs. 32 and 33, for the cases \( n = 2 \) and \( n = 3 \), respectively.

We need a classical result and its variant.

Lemma 2.4. It holds the following Chu-Vandermonde combinatorial identity and its variant

\[\mathop{\sum }\limits_{{\ell  = 0}}^{n}{C}_{n + 1}^{\ell  + 1}{C}_{k - 1}^{\ell } = \mathop{\sum }\limits_{{\ell  = 0}}^{n}{C}_{n + 1}^{n - \ell }{C}_{k - 1}^{\ell } = {C}_{n + k}^{n}, \tag{2.11}\]

and

\[\mathop{\sum }\limits_{{\ell  = 0}}^{n}{C}_{n + 1}^{\ell  + 1}{C}_{k - 1}^{\ell }{C}_{\ell  + 1}^{2} = \frac{\left( {n + 1}\right) n}{2}{C}_{n + k - 2}^{n}, \tag{2.12}\]

where the combinatorial number \( {C}_{n}^{m} = \frac{n\cdots \left( {n - m + 1}\right) }{m\cdots 1} \) for \( n \geq  m \) and \( {C}_{n}^{m} = 0 \) for \( n < m \).

## 3. A Family of Arnold-Winther Mixed Elements in Any Space Dimension

### 3.1. The lowest order Arnold-Winther mixed elements in any space dimension

To define lower order mixed finite elements with \( k \leq  n \), we extend the lowest order Arnold-Winther mixed elements in \( 2{\mathrm{D}}^{10} \) and \( 3{\mathrm{D}}^{2,6} \) to any space dimension. To this end, we introduce the following divergence-free space for element \( K \in  {\mathcal{T}}_{h} \),

\[{\Sigma }_{3 \rightarrow  n + 1,{DF}}\left( {K;\mathbb{S}}\right)  \mathrel{\text{:= }} \left\{  {\tau  \in  {P}_{n + 1}\left( {K;\mathbb{S}}\right)  \smallsetminus  {P}_{2}\left( {K;\mathbb{S}}\right),\operatorname{div}\tau  = 0}\right\} , \tag{3.1}\]

where

\[{P}_{n + 1}\left( {K;\mathbb{S}}\right)  \smallsetminus  {P}_{2}\left( {K;\mathbb{S}}\right)  \mathrel{\text{:= }} \left\{  {\tau  \in  {P}_{n + 1}\left( {K;\mathbb{S}}\right) \text{ and }\tau  \notin  {P}_{2}\left( {K;\mathbb{S}}\right) }\right\} . \tag{3.2}\]

For any \( \tau  \in  {P}_{n + 1}\left( {K;\mathbb{S}}\right)  \smallsetminus  {P}_{2}\left( {K;\mathbb{S}}\right) \), its divergence div \( \tau \) is an \( n \) -dimensional vector-valued polynomial of degree \( \leq  n \). This implies that the dimension of div \( {P}_{n + 1}\left( {K;\mathbb{S}}\right)  \smallsetminus \; {P}_{2}\left( {K;\mathbb{S}}\right) \) is

\[n\left( {\frac{\left( {2n}\right)!}{n!n!} - \left( {n + 1}\right) }\right)\]

which is equal to the number of the divergence-free constraints in (3.1). Since the dimension of the space \( {P}_{n + 1}\left( {K,\mathbb{S}}\right)  \smallsetminus  {P}_{2}\left( {K,\mathbb{S}}\right) \) is

\[\left( {\frac{\left( {{2n} + 1}\right)!}{n!\left( {n + 1}\right)!} - \frac{\left( {n + 2}\right)!}{2!n!}}\right) \frac{n\left( {n + 1}\right) }{2},\]

it follows that the dimension of the space \( {\Sigma }_{3 \rightarrow  n + 1,{DF}}\left( {K;\mathbb{S}}\right) \) is

\[\left( {\frac{\left( {{2n} + 1}\right)!}{n!\left( {n + 1}\right)!} - \frac{\left( {n + 2}\right)!}{2!n!}}\right) \frac{n\left( {n + 1}\right) }{2} - n\frac{\left( {2n}\right)!}{n!n!} + n\left( {n + 1}\right). \tag{3.3}\]

Then we can define the following enriched \( {P}_{2}\left( {K;\mathbb{S}}\right) \) space

\[{P}_{2}^{ * }\left( {K;\mathbb{S}}\right)  \mathrel{\text{:= }} {P}_{2}\left( {K;\mathbb{S}}\right)  + {\Sigma }_{3 \rightarrow  n + 1,{DF}}\left( {K;\mathbb{S}}\right). \tag{3.4}\]

The dimension of \( {P}_{2}^{ * }\left( {K;\mathbb{S}}\right) \) is equal to the dimension of \( {P}_{2}\left( {K;\mathbb{S}}\right) \) plus the dimension of \( {\Sigma }_{3 \rightarrow  n + 1,{DF}}\left( {K;\mathbb{S}}\right) \). Thanks to (3.3),

\[\text{the dimension of } {P}_{2}^{ * }\left( {K;\mathbb{S}}\right) = \frac{\left( {{2n} + 1}\right)!}{n!\left( {n + 1}\right)!}\frac{n\left( {n + 1}\right) }{2} - n\frac{\left( {2n}\right)!}{n!n!} + n\left( {n + 1}\right) . \tag{3.5}\]

To present the degrees of freedom of \( {P}_{2}^{ * }\left( {K;\mathbb{S}}\right) \), we define

\[{M}_{2}\left( K\right)  \mathrel{\text{:= }} \left\{  {\tau  \in  {P}_{2}^{ * }\left( {K;\mathbb{S}}\right),\operatorname{div}\tau  = 0\text{ and }{\left. \tau \nu \right| }_{\partial K} = 0}\right\} , \tag{3.6}\]

where \( \nu \) is the normal vector of \( \partial K \). For the space \( {M}_{2}\left( K\right) \), we have the following important result.

Lemma 3.1. The dimension of \( {M}_{2}\left( K\right) \) is

\[\frac{\left( {{2n} - 1}\right)!}{n!\left( {n - 1}\right)!}\frac{n\left( {n + 1}\right) }{2} + \frac{n\left( {n + 1}\right) }{2} - n\frac{\left( {2n}\right)!}{n!n!}. \tag{3.7}\]

Proof. By Lemma 2.1, \( {T}_{i, j},0 \leq  i < j \leq  n \), are linearly independent, which implies that the dimension of the bubble function space \( {\Sigma }_{K, n + 1, b} \) defined in (2.2) is

\[\frac{\left( {{2n} - 1}\right)!}{n!\left( {n - 1}\right)!}\frac{n\left( {n + 1}\right) }{2} \tag{3.8}\]

Since the dimension of \( R\left( K\right) \) is \( \frac{n\left( {n + 1}\right) }{2} \), the dimension of \( {R}^{ \bot  }\left( K\right) \) (with respect to \( \left. {{P}_{n}\left( {K;{\mathbb{R}}^{n}}\right) }\right) \) is

\[n\frac{\left( {2n}\right)!}{n!n!} - \frac{n\left( {n + 1}\right) }{2}. \tag{3.9}\]

Thanks to Lemma 2.2, the \( H \) (div) bubble function space \( {\Sigma }_{K, n + 1, b} \) is the full \( H \) (div) bubble function space of symmetric tensor-valued polynomials of degree \( \leq  n + 1 \) over \( K \). Then, the definitions of \( {P}_{2}^{ * }\left( {K;\mathbb{S}}\right) \) and \( {M}_{2}\left( K\right) \) imply that \( {M}_{2}\left( K\right) \) is identical to the space of all the divergence-free functions in \( {\Sigma }_{K, n + 1, b} \). Therefore, its dimension of \( {M}_{2}\left( K\right) \) is equal to

\[\text{ the dimension of }{\Sigma }_{K, n + 1, b}\text{ - the dimension of }{R}^{ \bot  }\left( K\right)\]

which completes the proof.

Before presenting the degrees of freedom of \( {P}_{2}^{ * }\left( {K;\mathbb{S}}\right) \), we introduce a more notation \( {\bigtriangleup }_{\ell } \) with \( 0 \leq  \ell  \leq  n \) which denotes an \( \ell \) -dimensional simplex of \( K \). For \( \ell  = 0,{\bigtriangleup }_{0} \) is a vertex of \( K \); for \( \ell  = 1,{\bigtriangleup }_{1} \) is an edge of \( K \).

Theorem 3.1. A matrix field \( \tau  \in  {P}_{2}^{ * }\left( {K;\mathbb{S}}\right) \) can be uniquely determined by the following degrees of freedom:

(1) the mean moments of degree at most \( n - \ell \) over \( {\bigtriangleup }_{\ell } \), of \( {\mathbf{t}}_{l}^{T}\tau {\nu }_{i},{\nu }_{i}^{T}\tau {\nu }_{j}, l = \; 1,\ldots,\ell, i, j = 1,\ldots, n - \ell,\left( {{C}_{n + 1 - \ell }^{2} + \ell \left( {n - \ell }\right) }\right) {C}_{n}^{\ell } = \frac{\left( {n - \ell }\right) \left( {n + \ell  + 1}\right) }{2}{C}_{n}^{\ell } \) degrees of freedom, for each \( \ell \) -dimensional simplex \( {\bigtriangleup }_{\ell } \) of \( K,0 \leq  \ell  \leq  n - 1 \), with \( \ell \) linearly independent tangential vectors \( {\mathbf{t}}_{1},\ldots,{\mathbf{t}}_{\ell } \), and \( n - \ell \) linearly independent normal vectors \( {\nu }_{1},\ldots,{\nu }_{n - \ell } \);

(2) the average of \( \tau \) over \( K,\frac{n\left( {n + 1}\right) }{2} \) degrees of freedom;

(3) the values of moments \( {\int }_{K}\tau : {\theta d}\mathbf{x},\theta  \in  {M}_{2}\left( K\right),\frac{\left( {{2n} - 1}\right)!}{n!\left( {n - 1}\right)!}\frac{n\left( {n + 1}\right) }{2} + \frac{n\left( {n + 1}\right) }{2} - n\frac{\left( {2n}\right)!}{n!n!} \) degrees of freedom.

Proof. We assume that all degrees of freedom vanish and show that \( \tau  = 0 \). Note that the mean moment becomes the value of \( \tau \) for a zero-dimensional simplex \( {\bigtriangleup }_{0} \), namely, a vertex, of \( K \). The first set of degrees of freedom implies that \( {\tau \nu } = 0 \) on \( \partial K \) while the second set of degrees of freedom shows div \( \tau  = 0 \). Then the third set of degrees of freedom proves that \( \tau  = 0 \). Next we shall prove that the sum of these degrees of freedom is equal to the dimension of the space \( {P}_{2}^{ * }\left( {K,\mathbb{S}}\right) \). In fact the sum of the first set of degrees of freedom is

\[\mathop{\sum }\limits_{{\ell  = 0}}^{{n - 1}}{C}_{n + 1}^{\ell  + 1}\frac{\left( {n - \ell }\right) \left( {n + \ell  + 1}\right) }{2}{C}_{n}^{\ell },\]

we refer interested readers to Ref. 28 for a detailed proof of the numbers of degrees of freedom in the first set. By the Chu-Vandermonde combinatorial identity (2.11) and its variant (2.12), see more details from Ref. 28,

\[\mathop{\sum }\limits_{{\ell  = 0}}^{{n - 1}}{C}_{n + 1}^{\ell  + 1}\frac{\left( {n - \ell }\right) \left( {n + \ell  + 1}\right) }{2}{C}_{n}^{\ell } = \frac{\left( {{2n} + 1}\right)!}{n!\left( {n + 1}\right)!}\frac{n\left( {n + 1}\right) }{2} - \frac{\left( {{2n} - 1}\right)!}{n!\left( {n - 1}\right)!}\frac{n\left( {n + 1}\right) }{2}.\]

Hence the desired result follows from (3.5), and the sum of the second and third sets of degrees of freedom.

We denote by \( {\Sigma }_{2, h}^{ * } \) the space of symmetric tensor fields that belong piecewise to \( {P}_{2}^{ * }\left( {K;\mathbb{S}}\right) \), and with the continuity conditions induced by the degrees of freedom defined in Theorem 3.1. In particular, for \( \tau  \in  {\Sigma }_{2, h}^{ * } \), the normal components \( {\tau \nu } \) are continuous across all the internal \( \left( {n - 1}\right) \) -dimensional simplices \( {\bigtriangleup }_{n - 1} \) and, hence \( {\Sigma }_{2, h}^{ * } \subset  H\left( {\operatorname{div},\Omega;\mathbb{S}}\right). \)

Remark 3.1. For \( n = 2,{\Sigma }_{2, h}^{ * } \) is the lowest order Arnold-Winther element stress space proposed in Ref. 10; for \( n = 3,{\Sigma }_{2, h}^{ * } \) is the discrete stress space defined in Ref. 2, see also Ref. 6.

To define a family of first-order mixed elements, we need a family of simplified lowest order mixed elements, which is defined by

\[{\widehat{P}}_{2}^{ * }\left( {K;\mathbb{S}}\right)  \mathrel{\text{:= }} \left\{  {\tau  \in  {P}_{2}^{ * }\left( {K;\mathbb{S}}\right),\operatorname{div}\tau  \in  R\left( K\right) }\right\} . \tag{3.10}\]

Since the dimension of \( R\left( K\right) \) is \( \frac{n\left( {n + 1}\right) }{2} \), the equation gives \( \frac{n\left( {n + 1}\right) }{2} \) constraints on \( {P}_{2}^{ * }\left( {K;\mathbb{S}}\right) \). Hence the dimension of \( {\widehat{P}}_{2}^{ * }\left( {K;\mathbb{S}}\right) \) is

\[\frac{\left( {{2n} + 1}\right)!}{n!\left( {n + 1}\right)!}\frac{n\left( {n + 1}\right) }{2} - n\frac{\left( {2n}\right)!}{n!n!} + \frac{n\left( {n + 1}\right) }{2}.\]

A complete set of degrees of freedom for \( {\widehat{P}}_{2}^{ * }\left( {K;\mathbb{S}}\right) \) is obtained by removing the \( \frac{n\left( {n + 1}\right) }{2} \) average values over \( K \) for \( {P}_{2}^{ * }\left( {K;\mathbb{S}}\right) \). The global space \( {\widehat{\Sigma }}_{2, h}^{ * } \) is defined in a similar way as \( {\Sigma }_{2, h}^{ * } \).

Remark 3.2. For \( n = 2,3,{\widehat{\Sigma }}_{2, h}^{ * } \) are the simplified lowest order element stress spaces in Refs. 10 and 6, respectively.

### 3.2. Higher order Arnold-Winther mixed elements

To define Arnold-Winther mixed elements of order \( k > 2 \), we introduce the following divergence-free space for element \( K \in  {\mathcal{T}}_{h} \),

\[{\Sigma }_{k + 1 \rightarrow  k + n - 1,{DF}}\left( {K;\mathbb{S}}\right)  \mathrel{\text{:= }} \left\{  {\tau  \in  {P}_{k + n - 1}\left( {K;\mathbb{S}}\right)  \smallsetminus  {P}_{k}\left( {K;\mathbb{S}}\right),\operatorname{div}\tau  = 0}\right\} . \tag{3.11}\]

Here \( {P}_{k + n - 1}\left( {K;\mathbb{S}}\right)  \smallsetminus  {P}_{k}\left( {K;\mathbb{S}}\right) \) and \( {P}_{k + n - 2}\left( {K;\mathbb{R}}\right)  \smallsetminus  {P}_{k - 1}\left( {K;\mathbb{R}}\right) \) are defined in a similar way as \( {P}_{n + 1}\left( {K;\mathbb{S}}\right)  \smallsetminus  {P}_{2}\left( {K;\mathbb{S}}\right) \) defined in (3.2). The dimension of \( {\Sigma }_{k + 1 \rightarrow  k + n - 1,{DF}}\left( {K;\mathbb{S}}\right) \) can be analyzed by a similar argument as that for \( {\Sigma }_{3 \rightarrow  n + 1,{DF}}\left( {K;\mathbb{S}}\right) \) in the previous subsection. In fact, since the dimension of the space \( {P}_{k + n - 2}\left( {K;\mathbb{R}}\right)  \smallsetminus  {P}_{k - 1}\left( {K;\mathbb{R}}\right) \) is

\[\frac{\left( \left( {k + {2n} - 2}\right) \right)!}{n!\left( {k + n - 2}\right)!} - \frac{\left( {n + k - 1}\right)!}{n!\left( {k - 1}\right)!},\]

the number of the divergence-free constraints imposed in (3.11) is

\[n\left( {\frac{\left( \left( {k + {2n} - 2}\right) \right)!}{n!\left( {k + n - 2}\right)!} - \frac{\left( {n + k - 1}\right)!}{n!\left( {k - 1}\right)!}}\right).\]

In addition, the dimension of the space \( {P}_{k + n - 1}\left( {K;\mathbb{S}}\right)  \smallsetminus  {P}_{k}\left( {K;\mathbb{S}}\right) \) is

\[\left( {\frac{\left( {k + {2n} - 1}\right)!}{n!\left( {k + n - 1}\right)!} - \frac{\left( {n + k}\right)!}{k!n!}}\right) \frac{n\left( {n + 1}\right) }{2}.\]

It follows that the dimension of the space \( {\Sigma }_{k + 1 \rightarrow  k + n - 1,{DF}}\left( {K;\mathbb{S}}\right) \) is

\[\left( {\frac{\left( {k + {2n} - 1}\right)!}{n!\left( {k + n - 1}\right)!} - \frac{\left( {n + k}\right)!}{k!n!}}\right) \frac{n\left( {n + 1}\right) }{2} - n\left( {\frac{\left( {k + {2n} - 2}\right)!}{n!\left( {k + n - 2}\right)!} - \frac{\left( {n + k - 1}\right)!}{n!\left( {k - 1}\right)!}}\right). \tag{3.12}\]

Then, we define the following enriched \( {P}_{k}\left( {K;\mathbb{S}}\right) \) space

\[{P}_{k}^{ * }\left( {K;\mathbb{S}}\right)  \mathrel{\text{:= }} {P}_{k}\left( {K;\mathbb{S}}\right)  + {\Sigma }_{k + 1 \rightarrow  k + n - 1,{DF}}\left( {K;\mathbb{S}}\right). \tag{3.13}\]

It follows that the dimension of \( {P}_{k}^{ * }\left( {K;\mathbb{S}}\right) \) is equal to the sum of the dimension of \( {P}_{k}\left( {K;\mathbb{S}}\right) \) and the dimension of \( {\Sigma }_{k + 1 \rightarrow  k + n - 1,{DF}}\left( {K;\mathbb{S}}\right) \), namely,

\[\frac{\left( {k + {2n} - 1}\right)!}{n!\left( {k + n - 1}\right)!}\frac{n\left( {n + 1}\right) }{2} - n\left( {\frac{\left( {k + {2n} - 2}\right)!}{n!\left( {k + n - 2}\right)!} - \frac{\left( {n + k - 1}\right)!}{n!\left( {k - 1}\right)!}}\right). \tag{3.14}\]

To present the degrees of freedom of \( {P}_{k}^{ * }\left( {K;\mathbb{S}}\right) \), we define

\[{M}_{k}\left( K\right)  \mathrel{\text{:= }} \left\{  {\tau  \in  {P}_{k}^{ * }\left( {K;\mathbb{S}}\right),\operatorname{div}\tau  = 0\text{ and }{\left. \tau \nu \right| }_{\partial K} = 0}\right\} , \tag{3.15}\]

where \( \nu \) is the normal vector of \( \partial K \). For the space \( {M}_{k}\left( K\right) \), we have the following important result.

Lemma 3.2. The dimension of \( {M}_{k}\left( K\right) \) is

\[\frac{\left( {k + {2n} - 3}\right)!}{n!\left( {k + n - 3}\right)!}\frac{n\left( {n + 1}\right) }{2} + \frac{n\left( {n + 1}\right) }{2} - n\frac{\left( {k + {2n} - 2}\right)!}{n!\left( {k + n - 2}\right)!}. \tag{3.16}\]

Proof. The dimension of the space \( {\Sigma }_{K, k + n - 1, b} \) reads

\[\frac{\left( {k + {2n} - 3}\right)!}{n!\left( {k + n - 3}\right)!}\frac{n\left( {n + 1}\right) }{2} \tag{3.17}\]

Since the dimension of \( R\left( K\right) \) is \( \frac{n\left( {n + 1}\right) }{2} \), the dimension of \( {R}^{ \bot  }\left( K\right) \) (with respect to \( {P}_{k + n - 2}\left( {K;{\mathbb{R}}^{n}}\right) ) \) is

\[n\frac{\left( {k + {2n} - 2}\right)!}{n!\left( {k + n - 2}\right)!} - \frac{n\left( {n + 1}\right) }{2}. \tag{3.18}\]

It follows from the definition of \( {P}_{k}^{ * }\left( {K;\mathbb{S}}\right) \) and Lemma 2.2 that \( {M}_{k}\left( K\right) \) contains all divergence-free tensor-value functions of \( {\Sigma }_{K, k + n - 1, b} \). Then the desired result follows from Lemma 2.3.

Theorem 3.2. A matrix field \( \tau  \in  {P}_{k}^{ * }\left( {K;\mathbb{S}}\right) \) can be uniquely determined by the following degrees of freedom:

(1) the mean moments of degree at most \( k + n - \ell  - 2 \) over \( {\bigtriangleup }_{\ell } \), of \( {\mathbf{t}}_{l}^{T}\tau {\nu }_{i},{\nu }_{i}^{T}\tau {\nu }_{j} \), \( l = 1,\ldots,\ell, i, j = 1,\ldots, n - \ell,\left( {{C}_{n + 1 - \ell }^{2} + \ell \left( {n - \ell }\right) }\right) {C}_{k + n - 2}^{\ell } = \frac{\left( {n - \ell }\right) \left( {n + \ell  + 1}\right) }{2}{C}_{k + n - 2}^{\ell } \) degrees of freedom, for each \( \ell \) -dimensional simplex \( {\bigtriangleup }_{\ell } \) of \( K,0 \leq  \ell  \leq  n - 1 \), with \( \ell \) linearly independent tangential vectors \( {\mathbf{t}}_{1},\ldots,{\mathbf{t}}_{\ell } \), and \( n - \ell \) linearly independent normal vectors \( {\nu }_{1},\ldots,{\nu }_{n - \ell } \);

(2) the values \( {\int }_{K}\tau : {\theta d}\mathbf{x} \) for any \( \theta  \in  \epsilon \left( {{P}_{k - 1}\left( {K;{\mathbb{R}}^{n}}\right) }\right), n{C}_{n + k - 1}^{n} - \frac{n\left( {n + 1}\right) }{2} \) degrees of freedom;

(3) the values \( {\int }_{K}\tau : {\theta d}\mathbf{x} \) for any \( \theta  \in  {M}_{k}\left( K\right),\frac{\left( {k + {2n} - 3}\right)!}{n!\left( {k + n - 3}\right)!}\frac{n\left( {n + 1}\right) }{2} + \frac{n\left( {n + 1}\right) }{2} - n\frac{\left( {k + {2n} - 2}\right)!}{n!\left( {k + n - 2}\right)!} \) degrees of freedom.

Proof. We assume that all degrees of freedom vanish and show that \( \tau  = 0 \). Note that the mean moment becomes the value of \( \tau \) for a zero-dimensional simplex \( {\bigtriangleup }_{0} \), namely, a vertex, of \( K \). The first set of degrees of freedom implies that \( {\tau \nu } = 0 \) on

\( \partial K \) while the second set of degrees of freedom shows div \( \tau  = 0 \). Then the third set of degrees of freedom proves that \( \tau  = 0 \).

Next we shall prove that the sum of these degrees of freedom is equal to the dimension of the space \( {P}_{k}^{ * }\left( {K;\mathbb{S}}\right) \). In fact, it follows from the Chu-Vandermonde combinatorial identity (2.11) and its variant (2.12) that the number of degrees in the first set is

\[\mathop{\sum }\limits_{{\ell  = 0}}^{{n - 1}}{C}_{n + 1}^{\ell  + 1}\frac{\left( {n - \ell }\right) \left( {n + \ell  + 1}\right) }{2}{C}_{n + k - 2}^{\ell } = \frac{n\left( {n + 1}\right) }{2}\left( {{C}_{k + {2n} - 1}^{n} - {C}_{k + {2n} - 3}^{n}}\right); \tag{3.19}\]

we refer interested readers to Ref. 28 for a detailed proof of the numbers of degrees of freedom in the first set. The desired result follows from (3.14) and (3.16).

We denote by \( {\Sigma }_{k, h}^{ * } \) the space of symmetric tensor fields that belong piecewise to \( {P}_{k}^{ * }\left( {K;\mathbb{S}}\right) \), and with the continuity conditions induced by the degrees of freedom defined in Theorem 3.2. In particular, for \( \tau  \in  {\Sigma }_{k, h}^{ * } \), the normal components \( {\tau \nu } \) are continuous across all the internal \( \left( {n - 1}\right) \) -dimensional simplices \( {\bigtriangleup }_{n - 1} \) and, hence \( {\Sigma }_{k, h}^{ * } \subset  H\left( {\operatorname{div},\Omega;\mathbb{S}}\right). \)

Remark 3.3. For \( n = 2,{\Sigma }_{k, h}^{ * } \) is the higher order mixed element stress spaces in Ref. 10; for \( n = 3,{\Sigma }_{k, h}^{ * } \) is the higher order mixed element stress spaces in Ref. 6.

## 4. A Family of Lower Order Mixed Elements

### 4.1. Mixed methods

For \( 2 \leq  k \leq  n \), we follow the idea of Refs. 28,32 and 33 to define the following discrete stress space:

\[{\Sigma }_{k, h} \mathrel{\text{:= }} \left\{  {\sigma  \in  H\left( {\operatorname{div},\Omega;\mathbb{S}}\right),\sigma  = {\sigma }_{c} + {\sigma }_{b},{\sigma }_{c} \in  {H}^{1}\left( {\Omega;\mathbb{S}}\right),}\right.\]

\[{\left. {\sigma }_{c}\right| }_{K} \in  {P}_{k}\left( {K;\mathbb{S}}\right),{\left. {\sigma }_{b}\right| }_{K} \in  {\Sigma }_{K, k, b},\forall K \in  {\mathcal{T}}_{h}\} \tag{4.1}\]

which is an \( H \) (div) bubble enrichment of the \( {H}^{1} \) space

\[{\widetilde{\Sigma }}_{k, h} \mathrel{\text{:= }} \left\{  {\tau  \in  {H}^{1}\left( {\Omega;\mathbb{S}}\right),{\left. \tau \right| }_{K} \in  {P}_{k}\left( {K;\mathbb{S}}\right),\forall K \in  {\mathcal{T}}_{h}}\right\} .\]

For \( \tau  \in  {\widetilde{\Sigma }}_{k, h} \), the degrees of freedom on any element \( K \) are: the mean moments of degree at most \( k - \ell  - 1 \) over each \( \ell \) -dimensional simplex \( {\bigtriangleup }_{\ell } \) of \( K,0 \leq  \ell  \leq  n \) of \( \tau \). A standard argument is able to prove that these degrees of freedom are unisolvent.

In order to get a stable pair of spaces, we take the discrete displacement space as the space of discontinuous piecewise vector-valued polynomials of degree \( \leq  k - 1 \), namely,

\[{V}_{k, h} \mathrel{\text{:= }} \left\{  {v \in  {L}^{2}\left( {\Omega;{\mathbb{R}}^{n}}\right),{\left. v\right| }_{K} \in  {P}_{k - 1}\left( {K;{\mathbb{R}}^{n}}\right) \text{ for all }K \in  {\mathcal{T}}_{h}}\right\} . \tag{4.2}\]

Unfortunately, we cannot establish the stability of the pair of spaces \( {\Sigma }_{k, h} \) and \( {V}_{k, h} \). We have to enrich the discrete stress space by some bubble functions. More precisely, the displacement space \( {V}_{k, h} \) can be decomposed as

\[{V}_{k, h} = {V}_{k, h}^{R} \oplus  {V}_{k, h}^{{R}^{ \bot  }},\]

where

\[{V}_{k, h}^{R} \mathrel{\text{:= }} \left\{  {v \in  {L}^{2}\left( {\Omega;{\mathbb{R}}^{n}}\right),{\left. v\right| }_{K} \in  R\left( K\right) \text{ for all }K \in  {\mathcal{T}}_{h}}\right\}\]

and

\[{V}_{k, h}^{{R}^{ \bot  }} \mathrel{\text{:= }} \left\{  {v \in  {L}^{2}\left( {\Omega;{\mathbb{R}}^{n}}\right),{\left. v\right| }_{K} \in  {R}^{ \bot  }\left( K\right) \text{ for all }K \in  {\mathcal{T}}_{h}}\right\} .\]

By Lemma 2.3,

\[\operatorname{div}{\Sigma }_{k, b, h} = {V}_{k, h}^{{R}^{ \bot  }},\]

where \( {\Sigma }_{k, b, h} \) is defined in (2.5). That is to say, to get a stable pair of spaces, we only need to enrich the discrete stress space \( {\Sigma }_{k, h} \) by some higher order bubble functions which are able to control the piecewise rigid motions in \( {V}_{k, h}^{R} \). We shall select these bubble functions from the lowest order Arnold-Winther element stress space \( {\Sigma }_{2, h}^{ * } \) defined in the previous section. To this end, given an \( \left( {n - 1}\right) \) -dimensional simplex \( F \) of \( {\mathcal{T}}_{h} \), let \( {\omega }_{F} \mathrel{\text{:= }} {K}^{ - } \cup  {K}^{ + } \) denote the union of two elements that share \( F \). We recall that \( R\left( {\omega }_{F}\right) \) is the rigid motion space over \( {\omega }_{F} \) while \( {\left. R\left( {\omega }_{F}\right) \right| }_{F} \) is the restriction on \( F \). Further we let \( {\left( {\left. R\left( {\omega }_{F}\right) \right| }_{F}\right) }^{ \bot  } \) denote the orthogonal complement space of \( {\left. R\left( {\omega }_{F}\right) \right| }_{F} \) with respect to \( {P}_{1}\left( {F;{\mathbb{R}}^{n}}\right) \) which allows to define the following \( \left( {n - 1}\right) \) -dimensional simplex \( H \) (div) bubble function space:

\[{\mathbb{B}}_{F}^{1} \mathrel{\text{:= }} \left\{  {\tau  \in  {\Sigma }_{2, h}^{ * },\tau  = 0\text{ on }\Omega  \smallsetminus  {\omega }_{F},{\int }_{F}{\tau \nu } \cdot  {pds} = 0}\right. \text{ for any }p \in  {\left( {\left. R\left( {\omega }_{F}\right) \right| }_{F}\right) }^{ \bot  }\text{, }\]

the averages of \( \tau \) over both \( {K}^{ - } \) and \( {K}^{ + } \) vanish,

\[\text{ the values of }{\int }_{K}\tau : {\theta d}\mathbf{x}\text{ vanish for any }\left. {\theta  \in  {M}_{2}\left( K\right), K = {K}^{ - }\text{ and }{K}^{ + }}\right\}  \text{. } \tag{4.3}\]

Here \( \nu \) is the normal vector of \( F \). We also need a subspace of \( {\mathbb{B}}_{F}^{1} \) defined by

\[{\mathbb{B}}_{F}^{2} \mathrel{\text{:= }} \left\{  {\tau  \in  {\mathbb{B}}_{F}^{1},{\int }_{F}{\tau \nu } \cdot  {pds} = 0\text{ for any }p \in  {P}_{0}\left( {F;{\mathbb{R}}^{n}}\right) }\right\} . \tag{4.4}\]

Hence we define the following enriched stress space

\[{\Sigma }_{k, h}^{ + } = {\Sigma }_{k, h} + \mathop{\sum }\limits_{F}{\mathbb{B}}_{F}^{1}\;\text{ for }2 \leq  k \leq  n - 1; \tag{4.5}\]

and

\[{\Sigma }_{k, h}^{ + } = {\Sigma }_{k, h} + \mathop{\sum }\limits_{F}{\mathbb{B}}_{F}^{2}\;\text{ for }k = n. \tag{4.6}\]

Lemma 4.1. The space \( {\Sigma }_{k, h}^{ + } \) is a direct sum of the spaces \( {\Sigma }_{k, h} \) and \( {\Sigma }_{F}{\mathbb{B}}_{F}^{1} \) for \( 2 \leq  k \leq  n - 1 \), and it is a direct sum of the spaces \( {\Sigma }_{k, h} \) and \( {\Sigma }_{F}{\mathbb{B}}_{F}^{2} \) for \( k = n \).

Proof. We only prove the first part of the theorem since the proof of the second part is similar. In fact, on the one hand, given \( F \), it follows from the definition of \( {\mathbb{B}}_{F}^{1} \) that for any \( \tau  \in  {\mathbb{B}}_{F}^{1} \) it vanishes on the following degrees of freedom:

- the mean moments of degree at most \( n - \ell \) over \( {\bigtriangleup }_{\ell } \), of \( {\mathbf{t}}_{l}^{T}\tau {\nu }_{i},{\nu }_{i}^{T}\tau {\nu }_{j}, l = 1,\ldots,\ell \), \( i, j = 1,\ldots, n - \ell,\left( {{C}_{n + 1 - \ell }^{2} + \ell \left( {n - \ell }\right) }\right) {C}_{n}^{\ell } = \frac{\left( {n - \ell }\right) \left( {n + \ell  + 1}\right) }{2}{C}_{n}^{\ell } \) degrees of freedom, for each \( \ell \) -dimensional simplex \( {\bigtriangleup }_{\ell } \) of \( K,0 \leq  \ell  \leq  n - 2 \), with \( \ell \) linearly independent tangential vectors \( {\mathbf{t}}_{1},\ldots,{\mathbf{t}}_{\ell } \), and \( n - \ell \) linearly independent normal vectors \( {\nu }_{1},\ldots,{\nu }_{n - \ell }, \)

for any \( K \in  {\mathcal{T}}_{h} \). On the other hand, for any \( \tau  \in  {P}_{k}\left( {K;\mathbb{S}}\right) \) with \( 2 \leq  k \leq  n - 1 \), if it vanishes on the above degrees of freedom, \( {\tau \nu } = 0 \) on \( \partial K \) where \( \nu \) is the normal vector of \( \partial K \); see Ref. 28 for more details. This indicates that \( {\tau \nu } = 0 \) on \( F \) which implies the first part of the theorem.

It follows from the definitions of \( {V}_{k, h}\left( {{P}_{k - 1}\text{ polynomials })}\right. \) and \( {\Sigma }_{k, h}^{ + } \) (enriched \( {P}_{k} \) polynomials) that

\[\operatorname{div}{\Sigma }_{k, h}^{ + } \subset  {V}_{k, h}\text{. }\]

This, in turn, leads to a strong divergence-free space:

\[{Z}_{h} \mathrel{\text{:= }} \left\{  {{\tau }_{h} \in  {\Sigma }_{k, h}^{ + } \mid  \left( {\operatorname{div}{\tau }_{h}, v}\right)  = 0\text{ for all }v \in  {V}_{k, h}}\right\}\]

\[= \left\{  {{\tau }_{h} \in  {\Sigma }_{k, h}^{ + } \mid  \operatorname{div}{\tau }_{h} = 0}\right. \text{ pointwise }\} \text{. } \tag{4.7}\]

The mixed finite element approximation of problem (1.1) reads: find \( \left( {{\sigma }_{h},{u}_{h}}\right)  \in \; {\Sigma }_{k, h}^{ + } \times  {V}_{k, h} \) such that

\[\left\{  \begin{array}{ll} \left( {A{\sigma }_{h},\tau }\right)  + \left( {\operatorname{div}\tau,{u}_{h}}\right)  = 0 & \text{ for all }\tau  \in  {\Sigma }_{k, h}^{ + }, \\  \left( {\operatorname{div}{\sigma }_{h}, v}\right)  = \left( {f, v}\right) & \text{ for all }v \in  {V}_{k, h}. \end{array}\right. \tag{4.8}\]

### 4.2. Stability analysis and error estimates

The convergence of the finite element solution follows the stability and the standard approximation property. So we consider first the well-posedness of the discrete problem (4.8). By the standard theory, we only need to prove the following two conditions, based on their counterpart at the continuous level.

(1) \( K \) -ellipticity. There exists a constant \( C > 0 \), independent of the meshsize \( h \) such that

\[\left( {{A\tau },\tau }\right)  \geq  C\parallel \tau {\parallel }_{H\left( \operatorname{div}\right) }^{2}\text{ for all }\tau  \in  {Z}_{h}, \tag{4.9}\]

where \( {Z}_{h} \) is the divergence-free space defined in (4.7).

(2) Discrete B-B condition. There exists a positive constant \( C > 0 \) independent of the meshsize \( h \), such that

\[\mathop{\inf }\limits_{{0 \neq  v \in  {V}_{k, h}}}\mathop{\sup }\limits_{{0 \neq  \tau  \in  {\Sigma }_{k, h}^{ + }}}\frac{\left( \operatorname{div}\tau, v\right) }{\parallel \tau {\parallel }_{H\left( \operatorname{div}\right) }\parallel v{\parallel }_{0}} \geq  C. \tag{4.10}\]

Theorem 4.1. For the discrete problem (4.8), the K-ellipticity (4.9) and the discrete B-B condition (4.10) hold uniformly. Consequently, the discrete mixed problem (4.8) has a unique solution \( \left( {{\sigma }_{h},{u}_{h}}\right)  \in  {\Sigma }_{k, h}^{ + } \times  {V}_{k, h} \).

Proof. The K-ellipticity immediately follows from the fact that \( \operatorname{div}{\Sigma }_{k, h}^{ + } \subset  {V}_{k, h} \). To prove the discrete B-B condition (4.10), for any \( {v}_{h} \in  {V}_{k, h} \), we will construct an interpolation operator \( {\Pi }_{h}: {H}^{1}\left( {\Omega;\mathbb{S}}\right)  \rightarrow  {\Sigma }_{k, h}^{ + } \) such that, for any \( \tau  \in  {H}^{1}\left( {\Omega;\mathbb{S}}\right) \),

\[{\int }_{K}\operatorname{div}\left( {\tau  - {\Pi }_{h}\tau }\right)  \cdot  {vd}\mathbf{x} = 0\;\text{ for any }v \in  {V}_{k, h}, \tag{4.11}\]

for any \( K \in  {\mathcal{T}}_{h} \). Further, if \( \tau  \in  {H}^{k + 1}\left( {\Omega;\mathbb{S}}\right) \), it holds

\[{\begin{Vmatrix}\tau  - {\Pi }_{h}\tau \end{Vmatrix}}_{0} + h{\begin{Vmatrix}\operatorname{div}\left( \tau  - {\Pi }_{h}\tau \right) \end{Vmatrix}}_{0} \leq  C{h}^{k + 1}{\left| \tau \right| }_{k + 1}. \tag{4.12}\]

We only show the above result for the cases \( 2 \leq  k \leq  n - 1 \) since the proof for the case \( k = n \) is similar. First let \( {I}_{h}: {H}^{1}\left( {\Omega;\mathbb{S}}\right)  \rightarrow  {\widetilde{\Sigma }}_{k, h} \) be a Scott-Zhang \( {}^{38} \) interpolation operator such that

\[{\begin{Vmatrix}\tau  - {I}_{h}\tau \end{Vmatrix}}_{0} + h{\begin{Vmatrix}\nabla {I}_{h}\tau \end{Vmatrix}}_{0} \leq  {Ch}\parallel \nabla \tau {\parallel }_{0}. \tag{4.13}\]

Since \( {I}_{h} \) preserves symmetric \( {P}_{k} \) functions locally,

\[{\begin{Vmatrix}\tau  - {I}_{h}\tau \end{Vmatrix}}_{0} + h{\begin{Vmatrix}\nabla \left( \tau  - {I}_{h}\tau \right) \end{Vmatrix}}_{0} \leq  C{h}^{k + 1}{\left| \tau \right| }_{k + 1}, \tag{4.14}\]

provided that \( \tau  \in  {H}^{k + 1}\left( {\Omega;\mathbb{S}}\right) \). See Ref. 28 for more details.

Second, these enriched bubble functions in \( \mathop{\sum }\limits_{F}{\mathbb{B}}_{F}^{1} \) on the \( \left( {n - 1}\right) \) -dimensional simplices \( F \) allow for defining a correction \( {\delta }_{h}^{F} \in  {\mathbb{B}}_{F}^{1} \) such that

\[{\int }_{F}{\delta }_{h}^{F}\nu  \cdot  {pd}\mathbf{s} = {\int }_{F}\left( {\tau  - {I}_{h}\tau }\right) \nu  \cdot  {pd}\mathbf{s}\;\text{ for any }p \in  {\left. R\left( K\right) \right| }_{F}. \tag{4.15}\]

For these corrections \( {\delta }_{h}^{F} \), we have

\[{\begin{Vmatrix}{\delta }_{h}^{F}\end{Vmatrix}}_{0,{\omega }_{F}} + h{\begin{Vmatrix}\operatorname{div}{\delta }_{h}^{F}\end{Vmatrix}}_{0,{\omega }_{F}} \leq  C\left( {{\begin{Vmatrix}\tau  - {I}_{h}\tau \end{Vmatrix}}_{0,{\omega }_{F}} + h{\begin{Vmatrix}\nabla \left( \tau  - {I}_{h}\tau \right) \end{Vmatrix}}_{0,{\omega }_{F}}}\right). \tag{4.16}\]

Finally, we take

\[{\Pi }_{h}^{1}\tau  = {I}_{h}\tau  + \mathop{\sum }\limits_{F}{\delta }_{h}^{F}. \tag{4.17}\]

We get a partial-divergence matching property of \( {\Pi }_{h}^{1}\tau \): for any \( p \in  R\left( K\right) \), as the symmetric gradient \( \epsilon \left( p\right)  = 0 \),

\[{\int }_{K}\left( {\operatorname{div}{\Pi }_{h}^{1}\tau  - \operatorname{div}\tau }\right)  \cdot  {pd}\mathbf{x} = {\int }_{\partial K}\left( {{\Pi }_{h}^{1}\tau  - \tau }\right) \nu  \cdot  {pd}\mathbf{s} = 0. \tag{4.18}\]

Next we make a correction \( {\delta }_{h}^{K} \in  {\Sigma }_{K, k, b} \) on each element \( K \) such that

\[{\int }_{K}\operatorname{div}{\delta }_{h}^{K} \cdot  {pd}\mathbf{x} = {\int }_{K}\operatorname{div}\left( {\tau  - {\Pi }_{h}^{1}\tau }\right)  \cdot  {pd}\mathbf{x}\;\text{ for any }p \in  {R}^{ \bot  }\left( K\right). \tag{4.19}\]

The existence of \( {\delta }_{h}^{K} \) follows from Lemma 2.3, which also implies that

\[{\begin{Vmatrix}\operatorname{div}{\delta }_{h}^{K}\end{Vmatrix}}_{0, K} \leq  {\begin{Vmatrix}\operatorname{div}\left( \tau  - {\Pi }_{h}^{1}\tau \right) \end{Vmatrix}}_{0, K}. \tag{4.20}\]

In addition, the \( {\delta }_{h}^{K} \) can be selected such that

\[{\begin{Vmatrix}{\delta }_{h}^{K}\end{Vmatrix}}_{0, K} = \min \left\{  {{\begin{Vmatrix}{\delta }_{K}\end{Vmatrix}}_{0, K},\operatorname{div}{\delta }_{K} = {\Pi }_{K}^{ \bot  }\operatorname{div}\left( {\tau  - {\Pi }_{h}^{1}\tau }\right),{\delta }_{K} \in  {\Sigma }_{K, k, b}}\right\} , \tag{4.21}\]

where \( {\Pi }_{K}^{ \bot  }: {L}^{2}\left( {K,{\mathbb{R}}^{n}}\right)  \rightarrow  {R}^{ \bot  }\left( K\right) \) denotes the \( {L}^{2} \) projection operator. It follows that \( {\begin{Vmatrix}\operatorname{div}{\delta }_{h}^{K}\end{Vmatrix}}_{0, K} \) defines a norm for it. Then, a scaling argument proves

\[{\begin{Vmatrix}{\delta }_{h}^{K}\end{Vmatrix}}_{0, K} \leq  {Ch}{\begin{Vmatrix}\operatorname{div}{\delta }_{h}^{K}\end{Vmatrix}}_{0, K} \leq  {Ch}{\begin{Vmatrix}\operatorname{div}\left( \tau  - {\Pi }_{h}^{1}\tau \right) \end{Vmatrix}}_{0, K}. \tag{4.22}\]

Finally, we define

\[{\Pi }_{h}\tau  = {\Pi }_{h}^{1}\tau  + \mathop{\sum }\limits_{K}{\delta }_{h}^{K}. \tag{4.23}\]

Hence Eq. (4.11) follows from (4.18) and (4.19). The estimate (4.12) follows from (4.14), (4.16), (4.20) and (4.22). In addition, by (4.13), (4.20) and (4.22),

\[{\begin{Vmatrix}\tau  - {\Pi }_{h}\tau \end{Vmatrix}}_{0} + h{\begin{Vmatrix}\nabla {\Pi }_{h}\tau \end{Vmatrix}}_{0} \leq  {Ch}\parallel \nabla \tau {\parallel }_{0}. \tag{4.24}\]

In the sequel, we use the interpolation operator \( {\Pi }_{h} \) to prove the discrete inf-sup condition (4.10). Indeed, by the stability of the continuous formulation, there is a \( \tau  \in  {H}^{1}\left( {\Omega;\mathbb{S}}\right) \) such that,

\[\operatorname{div}\tau  = {v}_{h}\;\text{ and }\;\parallel \tau {\parallel }_{1} \leq  C{\begin{Vmatrix}{v}_{h}\end{Vmatrix}}_{0}.\]

In this paper, we only consider the domain such that the above stability holds. We refer interested readers to Ref. 23 for the classical result which states it is true for Lipschitz domains in \( {\mathbb{R}}^{n} \); see Ref. 21 for more refined results.

It follows from (4.11) and (4.24) that

\[\operatorname{div}{\Pi }_{h}\tau  = {v}_{h}\;\text{ and }\;{\begin{Vmatrix}\tau  - {\Pi }_{h}\tau \end{Vmatrix}}_{0} + h{\begin{Vmatrix}\nabla {\Pi }_{h}\tau \end{Vmatrix}}_{0} \leq  {Ch}{\begin{Vmatrix}{v}_{h}\end{Vmatrix}}_{0}, \tag{4.25}\]

which shows (4.10) and completes the proof.

Theorem 4.2. Let \( \left( {\sigma, u}\right)  \in  \Sigma  \times  V \) be the exact solution of problem (1.1) and \( \left( {{\tau }_{h},{u}_{h}}\right)  \in  {\Sigma }_{k, h}^{ + } \times  {V}_{k, h} \) the finite element solution of (4.8). Then, for \( 2 \leq  k \leq  n \),

\[{\begin{Vmatrix}\sigma  - {\sigma }_{h}\end{Vmatrix}}_{H\left( \operatorname{div}\right) } + {\begin{Vmatrix}u - {u}_{h}\end{Vmatrix}}_{0} \leq  C{h}^{k}\left( {\parallel \sigma {\parallel }_{k + 1} + \parallel u{\parallel }_{k}}\right), \tag{4.26}\]

and

\[{\begin{Vmatrix}\sigma  - {\sigma }_{h}\end{Vmatrix}}_{0} \leq  C{h}^{k + 1}{\left| \sigma \right| }_{k + 1}. \tag{4.27}\]

Proof. The estimate (4.27) follows from the stability of the elements in Theorem 4.1 and the standard theory of mixed finite element methods. \( {}^{{15},{16}} \) Let the interpolation operator \( {\Pi }_{h} \) be defined in (4.11). It follows

\[\operatorname{div}{\Pi }_{h}\sigma  = \operatorname{div}{\sigma }_{h}. \tag{4.28}\]

This leads to

\[\left( {A\left( {{\sigma }_{h} - {\Pi }_{h}\sigma }\right),{\sigma }_{h} - {\Pi }_{h}\sigma }\right)  = \left( {A\left( {{\sigma }_{h} - \sigma }\right),{\sigma }_{h} - {\Pi }_{h}\sigma }\right)\]

\[+ \left( {A\left( {\sigma  - {\Pi }_{h}\sigma }\right),{\sigma }_{h} - {\Pi }_{h}\sigma }\right)\]

\[= \left( {A\left( {\sigma  - {\Pi }_{h}\sigma }\right),{\sigma }_{h} - {\Pi }_{h}\sigma }\right).\]

Hence

\[{\begin{Vmatrix}{\sigma }_{h} - {\Pi }_{h}\sigma \end{Vmatrix}}_{0} \leq  C{\begin{Vmatrix}\sigma  - {\Pi }_{h}\sigma \end{Vmatrix}}_{0}.\]

Then the estimate (5.8) follows from the triangle inequality and (4.12).

## 5. First-Order Mixed Elements

In order to get first-order mixed elements, we propose to take the following discrete displacement space

\[{V}_{1, h} \mathrel{\text{:= }} \left\{  {v \in  {L}^{2}\left( {\Omega;{\mathbb{R}}^{n}}\right),{\left. v\right| }_{K} \in  R\left( K\right) \text{ for any }K \in  {\mathcal{T}}_{h}}\right\} . \tag{5.1}\]

To design the space for the stress, we define

\[{\Sigma }_{1, h} \mathrel{\text{:= }} \left\{  {\tau  \in  {H}^{1}\left( {\Omega;\mathbb{S}}\right),{\left. \tau \right| }_{K} \in  {P}_{1}\left( {K,\mathbb{S}}\right) \text{ for any }K \in  {\mathcal{T}}_{h}}\right\} . \tag{5.2}\]

Since the pair \( \left( {{\Sigma }_{1, h},{V}_{1, h}}\right) \) is unstable, we propose to enrich \( {\Sigma }_{1, h} \) by some \( \left( {n - 1}\right) \) - dimensional simplex bubble function spaces. Given an \( \left( {n - 1}\right) \) -dimensional simplex \( F \) of \( {\mathcal{T}}_{h} \), we define

\[{\widehat{\mathbb{B}}}_{F} \mathrel{\text{:= }} \left\{  {\tau  \in  {\widehat{\Sigma }}_{2, h}^{ * },\tau  = 0}\right. \text{ on }\Omega  \smallsetminus  {\omega }_{F},{\int }_{F}{\tau \nu } \cdot  {pds} = 0\text{ for any }p \in  {\left( {\left. R\left( {\omega }_{F}\right) \right| }_{F}\right) }^{ \bot  }\text{, }\]

\[\text{ the values of }{\int }_{K}\tau : {\theta d}\mathbf{x}\text{ vanish for any }\left. {\theta  \in  {M}_{2}\left( K\right), K = {K}^{ - }\text{ and }{K}^{ + }}\right\}  \text{. } \tag{5.3}\]

This allows for defining the following enriched stress space

\[{\widehat{\Sigma }}_{1, h}^{ + } = {\Sigma }_{1, h} + \mathop{\sum }\limits_{F}{\widehat{\mathbb{B}}}_{F}. \tag{5.4}\]

For this enriched space \( {\widehat{\Sigma }}_{1, h}^{ + } \), the number of degrees of freedom on each simplex is 18 and 48 for \( n = 2,3 \), respectively, which are the simplest conforming mixed elements so far. A similar argument of Lemma 4.1 shows that \( {\widehat{\Sigma }}_{1, h}^{ + } \) is a direct sum of \( {\Sigma }_{1, h} \) and \( {\Sigma }_{F}{\widehat{\mathbb{B}}}_{F} \).

The mixed finite element approximation of problem (1.1) reads: find \( \left( {{\sigma }_{h},{u}_{h}}\right)  \in \; {\widehat{\Sigma }}_{1, h}^{ + } \times  {V}_{1, h} \) such that

\[\left\{  \begin{array}{ll} \left( {A{\sigma }_{h},\tau }\right)  + \left( {\operatorname{div}\tau,{u}_{h}}\right)  = 0 & \text{ for all }\tau  \in  {\widehat{\Sigma }}_{1, h}^{ + }, \\  \left( {\operatorname{div}{\sigma }_{h}, v}\right)  = \left( {f, v}\right) & \text{ for all }v \in  {V}_{1, h}. \end{array}\right. \tag{5.5}\]

It follows from div \( {\widehat{\Sigma }}_{1, h}^{ + } \subset  {V}_{1, h} \) that div \( \tau  = 0 \) for any \( \tau  \in  {Z}_{h} \), which implies the above K-ellipticity condition (4.9). A similar proof of Theorem 4.1 shows the discrete inf-sup condition (4.10). In particular, there exists an interpolation operator \( {\Pi }_{h}: {H}^{1}\left( {\Omega,\mathbb{S}}\right)  \rightarrow  {\widehat{\Sigma }}_{1, h}^{ + } \) such that

\[{\begin{Vmatrix}\tau  - {\Pi }_{h}\tau \end{Vmatrix}}_{0} + h\parallel \operatorname{div}\left( {\tau  - {\Pi }_{h}\tau }\right) \parallel  \leq  {h}^{k}\parallel \tau {\parallel }_{k},\; k = 1,2, \tag{5.6}\]

and

\[{\int }_{K}\operatorname{div}\left( {\tau  - {\Pi }_{h}\tau }\right) : {pd}\mathbf{x} = {\int }_{\partial K}\left( {\tau  - {\Pi }_{h}\tau }\right) \nu  \cdot  {pds} = 0\;\text{ for any }p \in  R\left( K\right), \tag{5.7}\]

for any \( K \in  {\mathcal{T}}_{h} \). A summary of these results leads to the error estimates in the following theorem.

Theorem 5.1. Let \( \left( {\sigma, u}\right)  \in  \Sigma  \times  V \) be the exact solution of problem (1.1) and \( \left( {{\tau }_{h},{u}_{h}}\right)  \in  {\widehat{\Sigma }}_{1, h}^{ + } \times  {V}_{1, h} \) the finite element solution of (5.5). Then,

\[{\begin{Vmatrix}\sigma  - {\sigma }_{h}\end{Vmatrix}}_{H\left( \operatorname{div}\right) } + {\begin{Vmatrix}u - {u}_{h}\end{Vmatrix}}_{0} \leq  {Ch}\left( {\parallel \operatorname{div}\sigma {\parallel }_{1} + \parallel u{\parallel }_{1}}\right) \tag{5.8}\]

and

\[{\begin{Vmatrix}\sigma  - {\sigma }_{h}\end{Vmatrix}}_{0} \leq  C{h}^{2}\parallel \sigma {\parallel }_{2}. \tag{5.9}\]

## 6. Numerical Tests

We compute a 2D pure displacement problem on the unit square \( \Omega  = {\left\lbrack  0,1\right\rbrack  }^{2} \) with a homogeneous boundary condition that \( u \equiv  0 \) on \( \partial \Omega \). In the computation, we let the compliance tensor in (1.1):

\[{A\sigma } = \frac{1}{2\mu }\left( {\sigma  - \frac{\lambda }{{2\mu } + {n\lambda }}\operatorname{tr}\left( \sigma \right) \delta }\right),\; n = 2,\]

where \( \delta  = \left( \begin{array}{ll} 1 & 0 \\  0 & 1 \end{array}\right) \), and \( \mu  = 1/2 \) and \( \lambda  = 1 \) are the Lamé constants. Let the exact solution be

\[u = \left( \begin{matrix} {e}^{x - y}x\left( {1 - x}\right) y\left( {1 - y}\right) \\  \sin \left( {\pi x}\right) \sin \left( {\pi y}\right)  \end{matrix}\right). \tag{6.1}\]

The true stress function \( \sigma \) and the load function \( f \) are defined by the equations in (1.1), for the given solution \( u \).

In the computation, the level one grid consists of two right triangles, obtained by cutting the unit square with a north-east line. Each grid is refined into a half-sized grid uniformly, to get a higher level grid. In all the computation, the discrete systems of equations are solved by Matlab backslash solver.

We use the bubble enriched \( {P}_{2} \) symmetric stress finite element with \( {P}_{1} \) discontinuous displacement finite element, \( k = 2 \) in (4.2) and in (4.6), and \( k = 2 \) in (4.1). That is, three \( {P}_{3} \) bubbles are enriched each edge. In Table 1, the errors and the convergence order in various norms are listed for the true solution (6.1). The optimal order of convergence is observed for both displacement and stress, see Table 1, as shown in the theorem.

Table 1. The errors, \( {e}_{h} = \sigma  - {\sigma }_{h} \), and the order of convergence, by the \( 2\mathrm{D}k = 2 \) element in (4.6) and (4.2), for (6.1).

<table><tr><td></td><td>\( {\begin{Vmatrix}u - {u}_{h}\end{Vmatrix}}_{0} \)</td><td>rate</td><td>\( {\begin{Vmatrix}{e}_{h}\end{Vmatrix}}_{0} \)</td><td>rate</td><td>\( {\begin{Vmatrix}\operatorname{div}{e}_{h}\end{Vmatrix}}_{0} \)</td><td>\( {h}^{n} \)</td></tr><tr><td>1</td><td>0.27452</td><td>0.0</td><td>1.24637</td><td>0.0</td><td>6.97007772</td><td>0.0</td></tr><tr><td>2</td><td>0.07432</td><td>1.9</td><td>0.18054</td><td>2.8</td><td>2.13781130</td><td>1.7</td></tr><tr><td>3</td><td>0.01959</td><td>1.9</td><td>0.02429</td><td>2.9</td><td>0.57734125</td><td>1.9</td></tr><tr><td>4</td><td>0.00497</td><td>2.0</td><td>0.00314</td><td>2.9</td><td>0.14709450</td><td>2.0</td></tr><tr><td>5</td><td>0.00125</td><td>2.0</td><td>0.00040</td><td>3.0</td><td>0.03694721</td><td>2.0</td></tr></table>

Table 2. The errors, \( {e}_{h} = \sigma  - {\sigma }_{h} \), and the order of convergence, by the Arnold-Winther 21/3 element, for (6.1).

<table><tr><td></td><td>\( {\begin{Vmatrix}u - {u}_{h}\end{Vmatrix}}_{0} \)</td><td>rate</td><td>\( {\begin{Vmatrix}{e}_{h}\end{Vmatrix}}_{0} \)</td><td>rate</td><td>\( {\begin{Vmatrix}\operatorname{div}{e}_{h}\end{Vmatrix}}_{0} \)</td><td>\( {h}^{n} \)</td></tr><tr><td>1</td><td>0.30554</td><td>0.0</td><td>1.58058</td><td>0.0</td><td>10.31991249</td><td>0.0</td></tr><tr><td>2</td><td>0.22589</td><td>0.4</td><td>0.89927</td><td>0.8</td><td>6.81340378</td><td>0.6</td></tr><tr><td>3</td><td>0.10922</td><td>1.0</td><td>0.25584</td><td>1.8</td><td>3.61633797</td><td>0.9</td></tr><tr><td>4</td><td>0.05354</td><td>1.0</td><td>0.06633</td><td>1.9</td><td>1.83690959</td><td>1.0</td></tr><tr><td>5</td><td>0.02661</td><td>1.0</td><td>0.01674</td><td>2.0</td><td>0.92212628</td><td>1.0</td></tr></table>

As a comparison, we also test the Arnold-Winther element from Ref. 10, which has a same degree of freedom as ours, 21, on each element. But the displacement in that element is approximated by the rigid-motion space only, instead of the full \( {P}_{1} \) space, i.e. 3 dof versus 6 dof on each triangle. The total degrees of freedom for the stress for the new element are \( 3\left| \mathbb{V}\right|  + 3\left| \mathbb{E}\right|  + 3\left| \mathbb{K}\right| \), where \( \left| \mathbb{V}\right|,\left| \mathbb{E}\right| \) and \( \left| \mathbb{K}\right| \) are the numbers of vertices, edges and elements of \( {\mathcal{T}}_{h} \), respectively, while those for the Arnold-Winther element are \( 3\left| \mathbb{V}\right|  + 4\left| \mathbb{E}\right| \). Since the three bubble functions on each element can be easily condensed, these two elements almost have the same complexity for solving. The errors and the orders of convergence are listed in Table 2. Because the new element uses the full \( {P}_{1} \) displacement space, the order of convergence is one higher than that of the Arnold-Winther element. Also as the new element includes the full \( {P}_{2} \) stress space, the order of convergence of stress is one order higher, see the data in Tables 1 and 2.

## Appendix A. The Basis Functions of \( {\Sigma }_{k, h} \) in Two Dimensions

Let \( \bigtriangleup {\mathbf{x}}_{0}{\mathbf{x}}_{1}{\mathbf{x}}_{2} = : K \in  {\mathcal{T}}_{h} \) with three edges \( {E}_{i} \) and corresponding three barycentric variables \( {\lambda }_{i} \). Here \( {\lambda }_{i} \) is a linear function which vanishes on edge \( {E}_{i} \) and assumes a nodal value 1 at the opposite vertex \( {\mathbf{x}}_{i} \). Given \( {E}_{i} = \overrightarrow{{\mathbf{x}}_{i - 1}{\mathbf{x}}_{i + 1}} \), its two endpoints are \( {\mathbf{x}}_{i - 1} \) and \( {\mathbf{x}}_{i + 1} \), which allows for defining its \( k - 1 \) interior nodal points by

\[{\mathbf{x}}_{{E}_{i}, j} = \frac{j}{k}{\mathbf{x}}_{i - 1} + \frac{k - j}{k}{\mathbf{x}}_{i + 1},\; j = 1,\ldots, k - 1. \tag{A.1}\]

We also define \( \frac{\left( {k - 1}\right) \left( {k - 2}\right) }{2} \) nodal points inside \( K \) by

\[{\mathbf{x}}_{K, l, m} = \frac{l}{k}{\mathbf{x}}_{0} + \frac{m}{k}{\mathbf{x}}_{1} + \frac{k - l - m}{k}{\mathbf{x}}_{2},\;1 \leq  l, m\text{ and }l + m \leq  k - 1. \tag{A.2}\]

Then the nodes for the Lagrange element of order \( k \) is

\[{X}_{K} = \left\{  {{\mathbf{x}}_{i}, i = 0,1,2}\right\}   \cup  \left\{  {{\mathbf{x}}_{{E}_{i}, j}, i = 0,1,2, j = 1,\ldots, k - 1}\right\}\]

\[\cup  \left\{  {{\mathbf{x}}_{K, l, m},1 \leq  l, m\text{ and }l + m \leq  k - 1}\right\} .\]

Given node \( {\mathbf{x}}_{{E}_{i}, j} \) on edge \( {E}_{i}, j = 1,\ldots, k - 1 \), let \( {\phi }_{{E}_{i}, j} \in  {P}_{k}\left( {K;\mathbb{R}}\right) \) be its associated nodal basis function of the Lagrange element of order \( k \) such that

\[{\phi }_{{E}_{i}, j}\left( {\mathbf{x}}_{{E}_{i}, j}\right)  = 1\text{ and }{\phi }_{{E}_{i}, j}\left( {\mathbf{x}}^{\prime }\right)  = 0\text{ for any }{\mathbf{x}}^{\prime } \in  {X}_{K}\text{ other than }{\mathbf{x}}_{{E}_{i}, j}\text{. } \tag{A.3}\]

Let \( {\mathbf{n}}_{i} = {\left\langle  {n}_{i,1},{n}_{i,2}\right\rangle  }^{T} \) and \( {\mathbf{n}}_{i}^{ \bot  } = {\left\langle  -{n}_{i,2},{n}_{i,1}\right\rangle  }^{T} \) be normal and tangent vectors on edge \( {E}_{i} \), respectively. We define a matrix of rank one by

\[{\mathbb{T}}_{{E}_{i}} = {\mathbf{n}}_{i}^{ \bot  }{\mathbf{n}}_{i}^{ \bot  }{}^{T}. \tag{A.4}\]

With these \( \left( {k - 1}\right) \) -edge bubble functions \( {\phi }_{{E}_{i}, j} \) on each edge and the matrix \( {\mathbb{T}}_{{E}_{i}} \) of rank one, we can define exactly \( \left( {k - 1}\right) \) stress functions \( {\tau }_{{\mathbb{E}}_{i}, j} \) by

\[{\tau }_{{E}_{i}, j} = {\phi }_{{E}_{i}, j}{\mathbb{T}}_{{E}_{i}},\; j = 1,2,\ldots, k - 1, i = 0,1,2. \tag{A.5}\]

By the definition, we have

\[{\left. {\tau }_{{E}_{i}, j} \cdot  {\mathbf{n}}_{l}\right| }_{{E}_{l}} = 0,\; i, l = 0,1,2, j = 1,\ldots, k - 1, \tag{A.6}\]

which implies that they are \( H \) (div) bubble functions on element \( K \).

Given \( E \) we need a basis which takes \( {\mathbb{T}}_{E} \) for \( \mathbb{S} \). To this end, we let

\[{\mathbb{T}}_{E,1} = {\mathbf{n}}_{E}{\mathbf{n}}_{E}^{T}\;\text{ and }\;{\mathbb{T}}_{E,2} = \frac{1}{2}\left( {{\mathbf{n}}_{E}^{ \bot  }{\mathbf{n}}_{E}^{T} + {\mathbf{n}}_{E}{\left( {\mathbf{n}}_{E}^{ \bot  }\right) }^{T}}\right).\]

It is straightforward to see that \( {\mathbb{T}}_{E},{\mathbb{T}}_{E,1} \) and \( {\mathbb{T}}_{E,2} \) are linearly independent and therefore form a basis of \( \mathbb{S} \). The canonical basis of \( \mathbb{S} \) reads

\[{\mathbb{T}}_{1} = \left( \begin{array}{ll} 1 & 0 \\  0 & 0 \end{array}\right),\;{\mathbb{T}}_{2} = \left( \begin{array}{ll} 0 & 1 \\  1 & 0 \end{array}\right),\;\text{ and }\;{\mathbb{T}}_{3} = \left( \begin{array}{ll} 0 & 0 \\  0 & 1 \end{array}\right). \tag{A.7}\]

Let \( {\mathcal{X}}_{\mathbb{E}} \) denote all interior nodes, defined in (A.1), of all the edges, \( {\mathcal{X}}_{\mathbb{K}} \) denote all interior nodes, defined in (A.2), of all the elements, and \( {\mathcal{X}}_{\mathbb{V}} \) denote all the vertices of \( {\mathcal{T}}_{h} \). Define the Lagrange element space of order \( k \) by

\[{\mathbb{P}}_{h} \mathrel{\text{:= }} {H}^{1}\left( {\Omega;\mathbb{R}}\right)  \cap  \left\{  {v \in  {L}^{2}\left( {\Omega;\mathbb{R}}\right),{\left. v\right| }_{K} \in  {P}_{k}\left( {K;\mathbb{R}}\right),\forall K \in  {\mathcal{T}}_{h}}\right\} .\]

Given node \( \mathbf{x} \in  {\mathcal{X}}_{\mathbb{V}} \cup  {\mathcal{X}}_{\mathbb{E}} \cup  {\mathcal{X}}_{\mathbb{K}} \), let \( {\phi }_{\mathbf{x}} \in  {\mathbb{P}}_{h} \) be its associated nodal basis function, which is similarly defined as \( {\phi }_{{E}_{i}, j} \) in (A.3).

The basis functions of \( {\Sigma }_{k, h} \) can be classified into four classes:

(1) Vertex-based basis functions: Given vertex \( \mathbf{x} \in  {\mathcal{X}}_{\mathbb{V}} \), its three associated basis functions of \( {\Sigma }_{k, h} \) read

\[{\tau }_{V,\mathbf{x}, i} = {\phi }_{\mathbf{x}}{\mathbb{T}}_{i},\; i = 1,2,3.\]

(2) Volume-based basis functions: Given node \( \mathbf{x} \in  {\mathcal{X}}_{\mathbb{K}} \) inside \( K \), its three associated basis functions of \( {\Sigma }_{k, h} \) read

\[{\tau }_{K,\mathbf{x}, i} = {\phi }_{\mathbf{x}}{\mathbb{T}}_{i},\; i = 1,2,3.\]

(3) Edge-based basis functions with nonzero fluxes: Given node \( \mathbf{x} \in  {\mathcal{X}}_{\mathbb{E}} \) on edge \( E \), its two associated basis functions with nonzero fluxes of \( {\Sigma }_{k, h} \) read

\[{\tau }_{E,\mathbf{x}, i}^{\left( nb\right) } = {\phi }_{\mathbf{x}}{\mathbb{T}}_{E, i},\; i = 1,2.\]

(4) Edge-based bubble functions: Given node \( \mathbf{x} \in  {\mathcal{X}}_{\mathbb{E}} \) on edge \( E \) which is shared by elements \( {K}_{1} \) and \( {K}_{2} \), its bubble functions in \( {\Sigma }_{k, h} \) read

\[{\tau }_{E,\mathbf{x}, i}^{\left( b\right) } = {\left. {\phi }_{\mathbf{x}}\right| }_{{K}_{i}}{\mathbb{T}}_{E},\; i = 1,2.\]

It is straightforward to see that these functions defined in the above four terms form a basis of \( {\Sigma }_{k, h} \), which are very easy to construct.

## Acknowledgment

The first author was supported by the NSFC Projects 11271035, 91430213 and 11421101.

## References

1. R. A. Adams, Sobolev Spaces (Academic Press, 1975).

2. S. Adams and B. Cockburn, A mixed finite element method for elasticity in three dimensions, J. Sci. Comput. 25 (2005) 515-521.

3. M. Amara and J. M. Thomas, Equilibrium finite elements for the linear elastic problem, Numer. Math. 33 (1979) 367-383.

4. D. N. Arnold, Differential complexes and numerical stability, in Proc. of the International Congress of Mathematicians, Vol. I: Plenary Lectures and Ceremonies (Higher Ed. Press, 2002), pp. 137-157.

5. D. N. Arnold and G. Awanou, Rectangular mixed finite elements for elasticity, Math. Models Methods Appl. Sci. 15 (2005) 1417-1429.

6. D. Arnold, G. Awanou and R. Winther, Finite elements for symmetric tensors in three dimensions, Math. Comput. 77 (2008) 1229-1251.

7. D. N. Arnold, F. Brezzi and J. Douglas Jr., PEERS: A new mixed finite element for plane elasticity, Jpn. J. Appl. Math. 1 (1984) 347-367.

8. D. N. Arnold, J. Douglas Jr. and C. P. Gupta, A family of higher order mixed finite element methods for plane elasticity, Numer. Math. 45 (1984), 1-22.

9. D. N. Arnold, R. Falk and R. Winther, Mixed finite element methods for linear elasticity with weakly imposed symmetry, Math. Comput. 76 (2007) 1699-1723.

10. D. N. Arnold and R. Winther, Mixed finite element for elasticity, Numer. Math. 92 (2002), 401-419.

11. D. N. Arnold and R. Winther, Nonconforming mixed elements for elasticity, Math. Models Methods Appl. Sci. 13 (2003) 295-307.

12. G. Awanou, Two remarks on rectangular mixed finite elements for elasticity, J. Sci. Comput. 50 (2012) 91-102.

13. D. Boffi, F. Brezzi and M. Fortin, Reduced symmetry elements in linear elasticity, Commun. Pure Appl. Anal. 8 (2009) 95-121.

14. J. Bramwell, L. Demkowicz, J. Gopalakrishnan and W. F. Qiu, A locking free hp DPG method for linear elasticity with symmetric stresses, Numer. Math. 122 (2012) 671-707.

15. F. Brezzi, On the existence, uniqueness and approximation of saddle-point problems arising from Lagrangian multipliers, Rev. Francaise Automat. Informat. Recherche Operationnelle Ser. Rouge 8 (1974) 129-151.

16. F. Brezzi and M. Fortin, Mixed and Hybrid Finite Element Methods (Springer, 1991).

17. C. Carstensen, M. Eigel and J. Gedicke, Computational competition of symmetric mixed FEM in linear elasticity, Comput. Methods Appl. Mech. Engrg. 200 (2011) 2903-2915.

18. C. Carstensen, D. Günther, J. Reininghaus and J. Thiele, The Arnold-Winther mixed FEM in linear elasticity. Part I: Implementation and numerical verification, Comput. Methods Appl. Mech. Engrg. 197 (2008) 3014-3023.

19. S. C. Chen and Y. N. Wang, Conforming rectangular mixed finite elements for elasticity, J. Sci. Comput. 47 (2011) 93-108.

20. B. Cockburn, J. Gopalakrishnan and J. Guzmán, A new elasticity element made for enforcing weak stress symmetry, Math. Comput. 79 (2010) 1331-1349.

21. R. G. Durán and M. A. Muschietti, An explicit right inverse of the divergence operator which is continuous in weighted norms, Studia Math. 148 (2001) 207-219.

22. B. M. Fraejis de Veubeke, Displacement and equilibrium models in the finite element method, in Stress Analysis, eds. O. C. Zienkiewics and G. S. Holister (Wiley, 1965), pp. 145-197.

23. V. Girault and P. A. Raviart, Finite Element Methods for Navier-Stokes Equations (Springer, 1986).

24. J. Gopalakrishnan and J. Guzmán, Symmetric nonconforming mixed finite elements for linear elasticity, SIAM J. Numer. Anal. 49 (2011) 1504-1520.

25. J. Gopalakrishnan and J. Guzmán, A second elasticity element using the matrix bubble, IMA J. Numer. Anal. 32 (2012) 352-372.

26. J. Guzmán, A unified analysis of several mixed methods for elasticity with weak stress symmetry, J. Sci. Comput. 44 (2010) 156-169.

27. J. Hu, A new family of efficient conforming mixed finite elements on both rectangular and cuboid meshes for linear elasticity in the symmetric formulation, SIAM J. Numer. Anal. 53 (2015) 1438-1463.

28. J. Hu, Finite element approximations of symmetric tensors on simplicial grids in \( {\mathbb{R}}^{n} \): The higher order case, J. Comput. Math. 33 (2015) 283-296.

29. J. Hu, H. Man, J. Wang and S. Zhang, The simplest nonconforming mixed finite element method for linear elasticity in the symmetric formulation on \( n \) -rectangular grids, Comput. Math. Appl. 71 (2016) 1317-1336.

30. J. Hu, H. Y. Man and S. Zhang, A simple conforming mixed finite element for linear elasticity on rectangular grids in any space dimension, J. Sci. Comput. 58 (2014) 367-379.

31. J. Hu and Z. C. Shi, Lower order rectangular nonconforming mixed elements for plane elasticity, SIAM J. Numer. Anal. 46 (2007) 88-102.

32. J. Hu and S. Zhang, A family of conforming mixed finite elements for linear elasticity on triangle grids, preprint (2014), arXiv: 1406.7457v2.

33. J. Hu and S. Zhang, A family of conforming mixed finite elements for linear elasticity on tetrahedral grids, Sci. China Math. 58 (2015) 297-307.

34. C. Johnson and B. Mercier, Some equilibrium finite element methods for two-dimensional elasticity problems, Numer. Math. 30 (1978) 103-116.

35. H.-Y. Man, J. Hu and Z.-C. Shi, Lower order rectangular nonconforming mixed finite element for the three-dimensional elasticity problem, Math. Models Methods Appl. Sci. 19 (2009) 51-65.

36. M. Morley, A family of mixed finite elements for linear elasticity, Numer. Math. 55 (1989) 633-666.

37. W. F. Qiu and L. Demkowicz, Mixed hp-finite element method for linear elasticity with weakly imposed symmetry, Comput. Methods Appl. Mech. Engrg. 198 (2009) 3682-3701.

38. L. R. Scott and S. Zhang, Finite-element interpolation of non-smooth functions satisfying boundary conditions, Math. Comput. 54 (1990) 483-493.

39. R. Stenberg, On the construction of optimal mixed finite element methods for the linear elasticity problem, Numer. Math. 48 (1986) 447-462.

40. R. Stenberg, Two low-order mixed methods for the elasticity problem, in The Mathematics of Finite Elements and Applications, Vol. 6, ed. J. R. Whiteman (Academic Press, 1988), pp. 271-280.

41. R. Stenberg, A family of mixed finite elements for the elasticity problem, Numer. Math. 53 (1988) 513-538.

42. V. B. Watwood Jr. and B. J. Hartz, An equilibrium stress field model for finite element solution of two-dimensional elastostatic problems, Int. J. Solids Struct. 4 (1968) 857- 873.

43. S. Y. Yi, Nonconforming mixed finite element methods for linear elasticity using rectangular elements in two and three dimensions, Calcolo 42 (2005) 115-133.

44. S. Y. Yi, A new nonconforming mixed finite element method for linear elasticity, Math. Models Methods Appl. Sci. 16 (2006) 979-999.

45. O. C. Zienkiewicz, R. L. Taylor and J. Z. Zhu, The Finite Element Method: Its Basis and Fundamentals, 6th edn., Vol. 1 (Elsevier, 2005).
