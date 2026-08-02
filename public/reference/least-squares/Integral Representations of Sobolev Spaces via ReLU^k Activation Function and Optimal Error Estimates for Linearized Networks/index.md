

# Integral Representations of Sobolev Spaces via ReLU \( {}^{k} \) Activation Function and Optimal Error Estimates for Linearized Networks

Xinliang Liu \( {}^{1} \) Tong Mao \( {}^{1} \) Jinchao \( {\mathrm{{Xu}}}^{1} \)

## Abstract

This paper presents two main theoretical results concerning shallow neural networks with \( {\mathrm{{ReLU}}}^{k} \) activation functions. We establish a novel integral representation for Sobolev spaces, showing that every function in \( {\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) \) can be expressed as an \( {\mathcal{L}}^{2} \) -weighted integral of ReLU \( {}^{k} \) ridge functions over the unit sphere. This result mirrors the known representation of Barron spaces and highlights a fundamental connection between Sobolev regularity and neural network representations. Moreover, we prove that linearized shallow networks—constructed by fixed inner parameters and optimizing only the linear coefficients—achieve optimal approximation rates \( \mathcal{O}\left( {n}^{-\frac{1}{2} - \frac{{2k} + 1}{2d}}\right) \) in Sobolev spaces.

## 1 Introduction

Deep neural networks (DNNs) are at the core of numerous breakthroughs in artificial intelligence [1], providing a powerful framework for solving complex problems. The efficiency of deep learning depends on several critical factors, including expressive power, generalization ability, training efficiency, and robustness.

Mathematically, a deep neural network is a specialized class of functions constructed through an iterative composition of shallow neural networks. Thus, the following shallow neural networks function class serves as a fundamental building block of artificial intelligence technology:

\[{\sum }_{n}^{\sigma } = \left\{  {\mathop{\sum }\limits_{{j = 1}}^{n}{a}_{j}\sigma \left( {{w}_{j} \cdot  x + {b}_{j}}\right) : {w}_{j} \in  {\mathbb{R}}^{d},{a}_{j},{b}_{j} \in  \mathbb{R}}\right\} , \tag{1.1}\]

Among various activation functions, the Rectified Linear Unit (ReLU) has emerged as the dominant choice in modern deep learning. In this paper, we focus on ReLU activation function, \( \operatorname{ReLU}\left( x\right)  = \; \max \left( {0, x}\right) \), and its variants, \( {\sigma }_{k}\left( x\right)  = {\operatorname{ReLU}}^{k}\left( x\right) \) ( \( k \) is a nonnegative integer). Due to the homogeneity of \( {\sigma }_{k} \), we assume the parameters lie on the unit sphere \( {\mathbb{S}}^{d} \). The corresponding function class of shallow \( {\operatorname{ReLU}}^{k} \) neural networks, denoted by \( {\sum }_{n}^{k} = {\sum }_{n}^{{\sigma }_{k}} \), is then

\[{\sum }_{n}^{k} = \left\{  {\mathop{\sum }\limits_{{j = 1}}^{n}{a}_{j}{\sigma }_{k}\left( {{\theta }_{j} \cdot  \widetilde{x}}\right) : {\theta }_{j} \in  {\mathbb{S}}^{d},{a}_{j} \in  \mathbb{R}}\right\} , \tag{1.2}\]

where \( \widetilde{x} = \left( \begin{array}{l} x \\  1 \end{array}\right) \) and \( {\theta }_{j} = \left( \begin{matrix} {w}_{j} \\  {b}_{j} \end{matrix}\right) \). To ensure encodability and numerical stability, it is essential to impose restrictions on the size of the parameters in a neural network. Consequently, given a parameter \( M \), which essentially determines the complexity of the function class, the stable shallow neural network function class is defined as

\[{\sum }_{n, M}^{k} = \left\{  {\mathop{\sum }\limits_{{j = 1}}^{n}{a}_{j}{\sigma }_{k}\left( {{\theta }_{j} \cdot  \widetilde{x}}\right) : {\theta }_{j} \in  {\mathbb{S}}^{d},\mathop{\sum }\limits_{{j = 1}}^{n}\left| {a}_{j}\right|  \leq  M}\right\} . \tag{1.3}\]

---

\( {}^{1} \) King Abdullah University of Science and Technology, Thuwal 23955, Saudi Arabia

---

In the early works, qualitative convergence has been extensively investigated since the 1990s (see, e.g., [2, 3]). A fundamental result states that the universal approximation property holds, namely the functions in \( {\sum }_{n}^{k} \) can approximate any continuous function, as long as the activation function \( \sigma \) is not a polynomial; see, e.g., [4].

The approximation rate of the shallow neural networks has also been widely studied in the literature. Roughly speaking, these results pertain to three types of function spaces: Barron spaces \( {\mathcal{B}}^{\sigma }\left( \Omega \right) \), also known as variation spaces \( {\mathcal{K}}_{1}\left( {\mathbb{D}}_{\sigma }\right) \) (see (2.2) below for definition); spectral Barron space [5, 6]; and Sobolev space (see (2.9) below for definition). In most existing works, the following error estimate has been established for the aforementioned function spaces:

\[\mathop{\inf }\limits_{{{f}_{n} \in  {\sum }_{n}^{\sigma }}}{\begin{Vmatrix}f - {f}_{n}\end{Vmatrix}}_{{\mathcal{L}}^{2}\left( \Omega \right) } = \mathcal{O}\left( {n}^{-\frac{1}{2}}\right). \tag{1.4}\]

A key technique used to establish (1.4) can be traced back to [7], which applied Maurey's sampling method from [8]; see also [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]. An overview of aforementioned and other related results can be found in [22, 23, 24].

Several studies have also explored improved approximation rates compared to (1.4), namely:

\[\mathop{\inf }\limits_{{{f}_{n} \in  {\sum }_{n}^{\sigma }}}{\begin{Vmatrix}f - {f}_{n}\end{Vmatrix}}_{{\mathcal{L}}^{2}\left( \Omega \right) } = \mathcal{O}\left( {n}^{-\frac{1}{2} - \frac{\alpha }{d}}\right) \tag{1.5}\]

for some \( \alpha  > 0 \). For Barron spaces, we refer to [25, 26, 5, 27, 28, 29, 30]. For Sobolev spaces, we refer to [31, 32, 33].

Notably, [5] and [27] established an approximation rate of \( \mathcal{O}\left( {n}^{-\frac{1}{2} - \frac{k}{d}}\right) \) for spectral Barron spaces \( {\widetilde{\mathcal{B}}}^{k}\left( \Omega \right) \), with further generalizations in [34]. For closer study of \( {\widetilde{\mathcal{B}}}^{k}\left( \Omega \right) \), see also [35].

In the context of shallow \( {\mathrm{{ReLU}}}^{k} \) networks, a strong connection exists between Barron spaces \( {\mathcal{B}}^{k}\left( \Omega \right)  = {\mathcal{B}}^{{\sigma }_{k}}\left( \Omega \right) \) (see (2.4)) and the class of shallow networks (1.3) (for further details, see Theorem 2.1):

1. A function \( f \) can be approximated by neural network classes \( {\left\{  {\sum }_{n, M}^{k}\right\}  }_{n = 1}^{\infty } \) if and only if \( f \in \; {\mathcal{B}}^{k}\left( \Omega \right) \) [36].

2. The Barron space \( {\mathcal{B}}^{k}\left( \Omega \right) \) has an integration form [36]

\[{\mathcal{B}}^{k}\left( \Omega \right)  = \left\{  {{\int }_{{\mathbb{S}}^{d}}{\sigma }_{k}\left( {\theta  \cdot  \widetilde{x}}\right) {d\mu }\left( \theta \right) : \mu  \in  \mathcal{M}\left( {\mathbb{S}}^{d}\right) }\right\} . \tag{1.6}\]

Moreover,

\[\parallel f{\parallel }_{{\mathcal{B}}^{k}\left( \Omega \right) } \simeq  \mathop{\inf }\limits_{{\mu  \in  \mathcal{M}\left( {\mathbb{S}}^{d}\right) }}\left\{  {\left| \mu \right| \left( {\mathbb{S}}^{d}\right) : f\left( x\right)  = {\int }_{{\mathbb{S}}^{d}}{\sigma }_{k}\left( {\theta  \cdot  \widetilde{x}}\right) {d\mu }\left( \theta \right) }\right\} . \tag{1.7}\]

3. Shallow ReLU \( {}^{k} \) networks achieve the optimal approximation rate in \( {\mathcal{B}}^{k}\left( \Omega \right) \) [28]:

\[\mathop{\inf }\limits_{{{f}_{n} \in  {\sum }_{n, M}^{k}}}{\begin{Vmatrix}f - {f}_{n}\end{Vmatrix}}_{{\mathcal{L}}^{2}\left( \Omega \right) } = \mathcal{O}\left( {n}^{-\frac{1}{2} - \frac{{2k} + 1}{2d}}\right) \tag{1.8}\]

where \( M \simeq  \parallel f{\parallel }_{{\mathcal{B}}^{k}\left( \Omega \right) } \).

The constraint on the parameter bound \( M \) ensures both encodability and numerical stability, which is crucial in the generalization analysis of machine learning models.

In this paper, we further investigate the approximation properties of shallow \( {\mathrm{{ReLU}}}^{k} \) networks. Specifically, we focus on a linear subset of \( {\sum }_{n}^{k} \), which we refer to as the finite neuron space (FNS). Given a predetermined set of parameters \( {\left\{  {\theta }_{j}^{ * }\right\}  }_{j = 1}^{n} \subset  {\mathbb{S}}^{d} \), we define the corresponding basis functions

\[{\phi }_{j}\left( x\right)  = {\sigma }_{k}\left( {{\theta }_{j}^{ * } \cdot  \widetilde{x}}\right)  = {\sigma }_{k}\left( {{w}_{j}^{ * } \cdot  x + {b}_{j}^{ * }}\right),\; j = 1,\ldots, n,\]

and the finite neuron space as

\[{L}_{n}^{k} = {L}_{n}^{k}\left( {\left\{  {\theta }_{j}^{ * }\right\}  }_{j = 1}^{n}\right)  = \operatorname{span}\left\{  {{\phi }_{1},\ldots,{\phi }_{n}}\right\} . \tag{1.9}\]

Analogous to \( {\sum }_{n, M}^{k} \) in (1.3), we define the constrained version of \( {L}_{n}^{k} \) as

\[{L}_{n, M}^{k} = \left\{  {\mathop{\sum }\limits_{{j = 1}}^{n}{a}_{j}{\phi }_{j}: {\left( n\mathop{\sum }\limits_{{j = 1}}^{n}{a}_{j}^{2}\right) }^{\frac{1}{2}} \leq  M}\right\} . \tag{1.10}\]

By applying the Cauchy-Schwarz inequality, it follows that

\[{L}_{n, M}^{k} \subset  {\sum }_{n, M}^{k} \tag{1.11}\]

A key advantage of the linear spaces \( {L}_{n}^{k} \) and \( {L}_{n, M}^{k} \) lies in their structural simplicity, which enables more efficient analysis and computation. Notably, optimization over \( {L}_{n, M}^{k} \) reduces to a convex least squares problem, whereas training in \( {\sum }_{n, M}^{k} \) generally involves solving a highly nonconvex optimization problem.

Similar to the results on \( {\operatorname{ReLU}}^{k} \) neural networks \( {\sum }_{n, M}^{k} \), this paper presents a novel perspective on characterizing the finite neuron space \( {L}_{n}^{k} \) and its constrained counterpart \( {L}_{n, M}^{k} \). Under the assumption that the parameters \( {\left\{  {\theta }_{j}^{ * }\right\}  }_{j = 1}^{n} \) form a well-distributed mesh over \( {\mathbb{S}}^{d} \) (see the precise definition in (2.10)), we establish the following results:

1. A function \( f \) can be approximated by FNS \( {\left\{  {L}_{n, M}^{k}\right\}  }_{n = 1}^{\infty } \) if and only if \( f \in  {\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) \);

2. The Sobolev space \( {\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) \) has an integration form

\[{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right)  = \left\{  {{\oint }_{{\mathbb{S}}^{d}}{\sigma }_{k}\left( {\theta  \cdot  \widetilde{x}}\right) \psi \left( \theta \right) {d\theta }: \psi  \in  {\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) }\right\} . \tag{1.12}\]

Moreover,

\[\parallel f{\parallel }_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) } \simeq  \mathop{\inf }\limits_{{\psi  \in  {\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) }}\left\{  {\parallel \psi {\parallel }_{{\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) }: f\left( x\right)  = {\oint }_{{\mathbb{S}}^{d}}{\sigma }_{k}\left( {\theta  \cdot  \widetilde{x}}\right) \psi \left( \theta \right) {d\theta }}\right\} , \tag{1.13}\]

3. The FNS achieves the optimal approximation rate in \( {\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) \):

\[\mathop{\inf }\limits_{{{f}_{n} \in  {L}_{n, M}^{k}}}{\begin{Vmatrix}f - {f}_{n}\end{Vmatrix}}_{{\mathcal{L}}^{2}\left( \Omega \right) } = \mathcal{O}\left( {n}^{-\frac{1}{2} - \frac{{2k} + 1}{2d}}\right) \tag{1.14}\]

where \( M \simeq  \parallel f{\parallel }_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) } \).

We would like to note that the approximation estimate in (1.14) is related to an earlier result by Petrushev [32]. For the \( {\mathrm{{ReLU}}}^{k} \) activation function, the result in [32] yields to an approximation rate estimate similar to (1.14) by using a different construction of the weights. Specifically, it uses the tensor product points \( {\left\{  \left( \begin{matrix} {w}_{j} \\  {b}_{i} \end{matrix}\right) \right\}  }_{\begin{matrix} {1 \leq  j \leq  {n}_{1}} \\  {1 \leq  i \leq  {n}_{2}} \end{matrix}} \), where \( {\left\{  {w}_{j}\right\}  }_{j = 1}^{{n}_{1}} \) are suitable quadrature points on \( {\mathbb{S}}^{d - 1} \) and \( {\left\{  {b}_{i}\right\}  }_{i = 1}^{{n}_{2}} \) are well-distributed points on \( \left\lbrack  {-1,1}\right\rbrack \). As demonstrated in Corollary 2.1 below, our result can be used to directly derive the result in [32] for the \( {\mathrm{{ReLU}}}^{k} \) case. However, it is important to note that our result cannot be derived from those in [32], nor can they be obtained using the proof techniques in [32]. In fact, the proof in our paper differs significantly from that used in [32].

Most importantly, our analysis yields the following key coefficient estimate

\[\sqrt{n}\parallel a{\parallel }_{2} \lesssim  \parallel f{\parallel }_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) } \tag{1.15}\]

As studied in [37], the bound on the coefficients such as (1.15) is essential in approximation theory in general. For example, it plays a central role in the generalization analysis (see Section 7). More importantly, without a bound like (1.15), approximation error estimates may lose their practical relevance [37]. In fact, [38] (see also [37]) showed that deep neural networks with a fixed number of parameters can achieve universal approximation when the weights are unbounded. Therefore, a bound estimate for coefficients is indispensable for both theoretical and practical applications.

Given the continuous embedding \( {\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right)  \hookrightarrow  {\mathcal{B}}^{k}\left( \Omega \right) \) [39], the metric entropy of \( {\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) \) is asymptotically same as that of \( {\mathcal{B}}^{k}\left( \Omega \right) \) (see (2.18)). Thus, linearized shallow networks achieve approximation rates comparable to the nonlinear class \( {\sum }_{n, M}^{k} \). Comparing this characterization with \( {\mathcal{B}}^{k}\left( \Omega \right) \) clarifies which functions are approximable by \( {\sum }_{n, M}^{k} \) but not by \( {L}_{n, M}^{k} \).

A consequence of (1.13) is that for \( f \in  {\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) \), there exists \( \psi  \in  {\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) \) (not necessarily unique) such that

\[f\left( x\right)  = \mathbb{E}\left\lbrack  {\psi \left( \theta \right) {\sigma }_{k}\left( {\theta  \cdot  \widetilde{x}}\right) }\right\rbrack , \tag{1.16}\]

where the expectation is over the uniform distribution on \( {\mathbb{S}}^{d} \). This identity provides a theoretical foundation for estimating approximation error in neural networks with random parameters—also known as random feature methods (see Section 6). In particular, if \( {\left\{  {\theta }_{j}\right\}  }_{j = 1}^{n} \) are i.i.d. uniform samples on \( {\mathbb{S}}^{d} \), then

\[{\mathbb{E}}_{n}\left\lbrack  {\mathop{\inf }\limits_{{a \in  {\mathbb{R}}^{n}}}{\begin{Vmatrix}f\left( x\right)  - \mathop{\sum }\limits_{{j = 1}}^{n}{a}_{j}{\sigma }_{k}\left( {\theta }_{j} \cdot  \widetilde{x}\right) \end{Vmatrix}}_{{\mathcal{L}}^{2}\left( \Omega \right) }}\right\rbrack   \lesssim  {n}^{-\frac{1}{2}}\parallel f{\parallel }_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) }. \tag{1.17}\]

Our refined rate (1.14) improves this bound to \( \mathcal{O}\left( {\left( \frac{n}{\log n}\right) }^{-\frac{1}{2} - \frac{{2k} + 1}{2d}}\right) \) via intricate analysis.

These insights also help explain the success of randomized approaches, which fix the inner weights \( {\theta }_{j} \) independently of the data. This framework includes stochastic basis selection [40,41], extreme learning machines [42, 43], random feature methods [44, 45, 46], and randomized neural networks [47, 48]. Their theoretical underpinnings have been further studied in [49, 50, 51, 52, 53]. Recent applications to numerical PDEs appear in [54, 55, 56, 57, 58, 59, 60, 61]. Our results indicate that the effectiveness of these methods is primarily due to the randomly generated weights are nearly well-distributed (see (2.7)), rather than any intrinsic randomness.

As another application of (1.14), we estimate the generalization error for learning \( f \in  {\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) \). Let \( {\left\{  \left( {x}_{i}, f\left( {x}_{i}\right) \right) \right\}  }_{i = 1}^{m} \) be i.i.d. uniform samples from \( \Omega \). Under the setting of Theorem 7.1, the bound holds:

\[{\mathbb{E}}_{{x}_{1},\ldots,{x}_{m}}\left\lbrack  {\begin{Vmatrix}{f}_{n, m} - f\end{Vmatrix}}_{{\mathcal{H}}^{1}\left( \Omega \right) }^{2}\right\rbrack   \lesssim  \left( {\parallel h{\parallel }_{{\mathcal{L}}^{\infty }\left( \Omega \right) } + M}\right) M{m}^{-\frac{1}{2}}. \tag{1.18}\]

Here \( {f}_{n, m} \) is the empirical risk minimizer:

\[{f}_{n, m} \mathrel{\text{:= }} \arg \mathop{\min }\limits_{{{f}_{n} \in  {L}_{n, M}^{k}}}\frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}{\left( f\left( {x}_{i}\right)  - {f}_{n}\left( {x}_{i}\right) \right) }^{2}. \tag{1.19}\]

Comparing with classical finite element space \( {V}_{n}^{k} \) (piecewise polynomials of degree \( \leq  k \) on \( n \) elements [62]), our results show an advantage. Given the lower bound \( \mathcal{O}\left( {n}^{-\frac{k + 1}{d}}\right) \) for finite elements [63], there exists \( f \in  {\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) \) such that

\[\mathop{\inf }\limits_{{{f}_{n} \in  {L}_{n}^{k}}}{\begin{Vmatrix}f - {f}_{n}\end{Vmatrix}}_{{\mathcal{L}}^{2}\left( \Omega \right) } = \mathcal{O}\left( {n}^{-\frac{1}{2}\left( {1 - \frac{1}{d}}\right) }\right) \mathop{\inf }\limits_{{{f}_{n} \in  {V}_{n}^{k}}}{\begin{Vmatrix}f - {f}_{n}\end{Vmatrix}}_{{\mathcal{L}}^{2}\left( \Omega \right) }. \tag{1.20}\]

This highlights that for smooth functions, \( {V}_{n}^{k} \) suffers from the curse of dimensionality, while \( {L}_{n}^{k} \) does not, despite both using piecewise polynomials of degree \( k \).

We also prove error estimates similar to (1.14) for lower-order Sobolev spaces:

\[\mathop{\inf }\limits_{{{f}_{n} \in  {L}_{n}^{k}}}{\begin{Vmatrix}f - {f}_{n}\end{Vmatrix}}_{{\mathcal{L}}^{2}\left( \Omega \right) } = \mathcal{O}\left( {n}^{-\frac{r}{d}}\right) \parallel f{\parallel }_{{\mathcal{H}}^{r}\left( \Omega \right) },\; r \leq  \frac{d + {2k} + 1}{2}. \tag{1.21}\]

The rate \( \mathcal{O}\left( {n}^{-\frac{r}{d}}\right) \) matches optimal rates for Sobolev spaces by classic methods like polynomials, Fourier series [64], and finite elements [62]. Early works [33] showed this rate for shallow networks with smooth sigmoidal activations \( \sigma \) (see also [31, 32]).

Comparing with the \( {\mathcal{L}}^{2} \) -rate for shallow \( {\operatorname{ReLU}}^{k} \) networks [39] (see also [65,66] for \( {\mathcal{L}}^{\infty } \) -norm)

\[\mathop{\inf }\limits_{{{f}_{n} \in  {\sum }_{n}^{k}}}{\begin{Vmatrix}f - {f}_{n}\end{Vmatrix}}_{{\mathcal{L}}^{2}\left( \Omega \right) } = \mathcal{O}\left( {n}^{-\frac{r}{d}}\right) \parallel f{\parallel }_{{\mathcal{H}}^{r}\left( \Omega \right) },\; r \leq  \frac{d + {2k} + 1}{2}, \tag{1.22}\]

we conclude that optimal nonlinear estimates (1.22) are fully realized by linear approximation in Sobolev spaces. Rate (1.22) is also optimal up to log factors (Theorem 6.2).

This paper is organized as follows. Section 2 introduces notation and main results. Section 3 reviews spherical harmonics and Legendre polynomials essential for the proofs. Section 4 establishes approximation results on the sphere using linear \( {\operatorname{ReLU}}^{k} \) spaces with well-distributed weights. Section 5 extends these results to general domains \( \Omega  \subset  {\mathbb{R}}^{d} \), completing the main proofs. Section 6 connects linearized shallow networks and random feature methods, showing the latter's approximation rate follows from the former. Section 7 applies our approximation results and coefficient estimates to compute the generalization error for an elliptic PDE.

## 2 Background and main results

Before presenting our main results, we introduce the notation used in this paper and provide brief discussions of main results in this paper. Some of these notations were previously mentioned in Section 1, where they were assumed to be familiar to the reader. Here, we provide explicit definitions for clarity.

First of all, following [67], we adapt the notation \( \gtrsim , \lesssim \), and \( \simeq \), which express upper and lower bounds up to constant factors. When we write

\[f\left( x\right)  \gtrsim  g\left( x\right),\; g\left( x\right)  \lesssim  h\left( x\right),\; h\left( x\right)  \simeq  k\left( x\right)\]

it means that there exist constants \( {c}_{1},{c}_{2},{c}_{3},{c}_{4} \) independent of \( x \) such that

\[f\left( x\right)  \geq  {c}_{1}g\left( x\right),\; g\left( x\right)  \leq  {c}_{2}h\left( x\right),\;{c}_{3}h\left( x\right)  \leq  k\left( x\right)  \leq  {c}_{4}h\left( x\right).\]

### 2.1 Shallow neural networks, Barron spaces and Sobolev spaces

Given an activation function \( \sigma : \mathbb{R} \rightarrow  \mathbb{R} \) and \( G \subset  {\mathbb{R}}^{d + 1} \), we denote the dictionary

\[{\mathbb{D}}_{\sigma } \mathrel{\text{:= }} \{  \pm  \sigma \left( {\theta  \cdot  \widetilde{x}}\right) : \theta  \in  G\},\]

and the \( {\mathcal{L}}^{2} \) -closure of its convex hull \( {B}_{1} \mathrel{\text{:= }} \overline{\operatorname{conv}\left( {\mathbb{D}}_{\sigma }\right) } \). Following [28] (see also [9, 12, 14, 15, 16, 17, 18]), we denote the Barron space \( {\mathcal{B}}^{\sigma }\left( \Omega \right) \) as follows:

\[{\mathcal{B}}^{\sigma }\left( \Omega \right)  \mathrel{\text{:= }} \left\{  {f \in  {\mathcal{L}}^{2}\left( \Omega \right) : \parallel f{\parallel }_{{\mathcal{B}}^{\sigma }\left( \Omega \right) } < \infty }\right\} , \tag{2.1}\]

where

\[\parallel f{\parallel }_{{\mathcal{B}}^{\sigma }\left( \Omega \right) } = \inf \left\{  {t > 0: f \in  t{B}_{1}}\right\} .\]

In particular, when \( \sigma  = {\sigma }_{k} \), we take \( G = {\mathbb{S}}^{d} \) as \( {\sigma }_{k} \) is homogeneous. In this case, we denote

\[{\mathcal{B}}^{k}\left( \Omega \right)  = {\mathcal{B}}^{{\sigma }_{k}}\left( \Omega \right). \tag{2.2}\]

The Barron spaces can be characterized by some qualitative approximation property of the following compact subset of the \( {\mathrm{{ReLU}}}^{k} \) neural networks (1.3). We state the relation between the \( {\mathrm{{ReLU}}}^{k} \) neural networks and Barron spaces as the following theorem.

Theorem 2.1 ([28,36]). Let \( d \in  \mathbb{N},\Omega  \subset  {\mathbb{R}}^{d} \) be a bounded domain.

1. \( f \in  {\mathcal{B}}^{k}\left( \Omega \right) \) if and only if, with some \( M \simeq  \parallel f{\parallel }_{{\mathcal{B}}^{k}\left( \Omega \right) } \),

\[\mathop{\lim }\limits_{{n \rightarrow  \infty }}\mathop{\inf }\limits_{{{f}_{n} \in  {\sum }_{n, M}^{k}}}{\begin{Vmatrix}f - {f}_{n}\end{Vmatrix}}_{{\mathcal{L}}^{2}\left( \Omega \right) } = 0. \tag{2.3}\]

2. The Barron space \( {\mathcal{B}}^{k}\left( \Omega \right) \) can be alternatively written as

\[{\mathcal{B}}^{k}\left( \Omega \right)  = \left\{  {{\int }_{{\mathbb{S}}^{d}}{\sigma }_{k}\left( {\theta  \cdot  \widetilde{x}}\right) {d\mu }\left( \theta \right) : \mu  \in  \mathcal{M}\left( {\mathbb{S}}^{d}\right) }\right\} . \tag{2.4}\]

Furthermore,

\[\parallel f{\parallel }_{{\mathcal{B}}^{k}\left( \Omega \right) } \simeq  \mathop{\inf }\limits_{{\mu  \in  \mathcal{M}\left( {\mathbb{S}}^{d}\right) }}\left\{  {\left| \mu \right| \left( {\mathbb{S}}^{d}\right) : f\left( x\right)  = {\int }_{{\mathbb{S}}^{d}}{\sigma }_{k}\left( {\theta  \cdot  \widetilde{x}}\right) {d\mu }\left( \theta \right) }\right\} . \tag{2.5}\]

3. Let \( f \in  {\mathcal{B}}^{k}\left( \Omega \right) \), with some \( M \simeq  \parallel f{\parallel }_{{\mathcal{B}}^{k}\left( \Omega \right) } \)

\[\mathop{\inf }\limits_{{{f}_{n} \in  {\sum }_{n, M}^{k}}}{\begin{Vmatrix}f - {f}_{n}\end{Vmatrix}}_{{\mathcal{L}}^{2}\left( \Omega \right) } \lesssim  {n}^{-\frac{1}{2} - \frac{{2k} + 1}{2d}}\parallel f{\parallel }_{{\mathcal{B}}^{k}\left( \Omega \right) }, \tag{2.6}\]

All the corresponding constants are independent of \( n,{\left\{  {\theta }_{j}^{ * }\right\}  }_{j = 1}^{n} \), and \( f \).

Even if we remove the restriction on the coefficients, the rate (2.6) is optimal up to logarithmic factors [28]:

\[\mathop{\sup }\limits_{{\parallel f{\parallel }_{{\mathcal{B}}^{k}\left( \Omega \right) } \leq  1}}\mathop{\inf }\limits_{{{f}_{n} \in  {\sum }_{n}^{k}}}{\begin{Vmatrix}f - {f}_{n}\end{Vmatrix}}_{{\mathcal{L}}^{2}\left( \Omega \right) } \geq  c{\left( n\log n\right) }^{-\frac{1}{2} - \frac{{2k} + 1}{2d}} \tag{2.7}\]

We also refer to [68, 24, 69, 70] for relevant results.

We now introduce the Sobolev spaces \( {\mathcal{H}}^{r}\left( \Omega \right) \). Let \( \Omega  \subset  {\mathbb{R}}^{d} \) be bounded, for \( r \in  \mathbb{N} \), the Sobolev spaces are defined as

\[{\mathcal{H}}^{r}\left( \Omega \right)  = \left\{  {f \in  {\mathcal{L}}^{2}\left( \Omega \right) : \parallel f{\parallel }_{{\mathcal{H}}^{r}\left( \Omega \right) } < \infty }\right\} . \tag{2.8}\]

where norm is given by

\[\parallel f{\parallel }_{{\mathcal{H}}^{r}\left( \Omega \right) } = {\left( \parallel f{\parallel }_{{\mathcal{L}}^{2}\left( \Omega \right) }^{2} + \mathop{\sum }\limits_{{\alpha  \in  {\mathbb{N}}_{0}^{d},\parallel \alpha {\parallel }_{1} = r}}{\begin{Vmatrix}{D}^{\alpha }f\end{Vmatrix}}_{{\mathcal{L}}^{2}\left( \Omega \right) }^{2}\right) }^{\frac{1}{2}}. \tag{2.9}\]

For \( r \notin  \mathbb{N}, {\mathcal{H}}^{r}\left( \Omega \right) \) is defined through a standard interpolation (see, e.g., [71]) of \( {\mathcal{H}}^{\lfloor r\rfloor }\left( \Omega \right) \) and \( {\mathcal{H}}^{\lceil r\rceil }\left( \Omega \right) \).

In this paper, we establish an approximation rate of \( \mathcal{O}\left( {n}^{-\frac{r}{d}}\right) \) for functions in the Sobolev space \( {\mathcal{H}}^{r}\left( \Omega \right) \) under the condition that these parameters are well-distributed on \( {\mathbb{S}}^{d} \).

Definition 2.1 (Well-distributed and quasi-uniform). Let \( d \in  \mathbb{N} \), a set of points \( {\left\{  {\theta }_{j}^{ * }\right\}  }_{j = 1}^{n} \subset  {\mathbb{S}}^{d} \) is said to be well-distributed if

\[\mathop{\max }\limits_{{\theta  \in  {\mathbb{S}}^{d}}}\mathop{\min }\limits_{{1 \leq  j \leq  n}}\rho \left( {\theta,{\theta }_{j}^{ * }}\right)  \lesssim  {n}^{-\frac{1}{d}} \tag{2.10}\]

Moreover, a well-distributed collection \( {\left\{  {\theta }_{j}^{ * }\right\}  }_{j = 1}^{n} \) is said to be quasi-uniform if

\[\mathop{\max }\limits_{{\theta  \in  {\mathbb{S}}^{d}}}\mathop{\min }\limits_{{1 \leq  j \leq  n}}\rho \left( {\theta,{\theta }_{j}^{ * }}\right)  \lesssim  \mathop{\min }\limits_{{i \neq  j}}\rho \left( {{\theta }_{i}^{ * },{\theta }_{j}^{ * }}\right). \tag{2.11}\]

The corresponding constants are independent of \( n \).

### 2.2 Main results

We now present the main results of this paper. The first theorem focuses on analyzing the approximation properties of the \( {\operatorname{ReLU}}^{k} \) activation function with predetermined parameters \( {\left\{  {\theta }_{j}^{ * }\right\}  }_{j = 1}^{n} \).

Theorem 2.2. Let \( d, n \in  \mathbb{N}, k \in  {\mathbb{N}}_{0},\Omega  \subset  {\mathbb{R}}^{d} \) be a bounded domain with Lipschitz boundary, and \( {\left\{  {\theta }_{j}^{ * }\right\}  }_{j = 1}^{n} \subset  {\mathbb{S}}^{d} \). Then for any \( f \in  {\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) \), with some

\[M \simeq  \parallel f{\parallel }_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) }\]

\[\mathop{\inf }\limits_{{{f}_{n} \in  {L}_{n, M}^{k}}}{\begin{Vmatrix}f - {f}_{n}\end{Vmatrix}}_{{\mathcal{L}}^{2}\left( \Omega \right) } \lesssim  {h}^{\frac{d + {2k} + 1}{2}}\parallel f{\parallel }_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) }. \tag{2.12}\]

where

\[h = \mathop{\max }\limits_{{\theta  \in  {\mathbb{S}}^{d}}}\mathop{\min }\limits_{{1 \leq  j \leq  n}}\rho \left( {\theta,{\theta }_{j}^{ * }}\right).\]

More generally, let \( r \leq  \frac{d + {2k} + 1}{2} \) and \( s \leq  \min \{ k, r\} \). Then for any \( f \in  {\mathcal{H}}^{r}\left( \Omega \right) \), with some \( M \simeq \; {h}^{-\frac{d + {2k} + 1 - {2r}}{2}}\parallel f{\parallel }_{{\mathcal{H}}^{r}\left( \Omega \right) }, \)

\[\mathop{\inf }\limits_{{{f}_{n} \in  {L}_{n, M}^{k}}}{\begin{Vmatrix}f - {f}_{n}\end{Vmatrix}}_{{\mathcal{H}}^{s}\left( \Omega \right) } \lesssim  {h}^{r - s}\parallel f{\parallel }_{{\mathcal{H}}^{r}\left( \Omega \right) }. \tag{2.13}\]

If the collection \( {\left\{  {\theta }_{j}^{ * }\right\}  }_{j = 1}^{n} \subset  {\mathbb{S}}^{d} \) is well-distributed, then for any \( f \in  {\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) \), with some \( M \simeq  \parallel f{\parallel }_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) } \), we have

\[\mathop{\inf }\limits_{{{f}_{n} \in  {L}_{n, M}^{k}}}{\begin{Vmatrix}f - {f}_{n}\end{Vmatrix}}_{{\mathcal{L}}^{2}\left( \Omega \right) } \lesssim  {n}^{-\frac{1}{2} - \frac{{2k} + 1}{2d}}\parallel f{\parallel }_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) }. \tag{2.14}\]

All the corresponding constants are independent of \( n,{\left\{  {\theta }_{j}^{ * }\right\}  }_{j = 1}^{n} \), and \( f \).

Remark 2.1. The points \( {\left\{  {\theta }_{j}^{ * }\right\}  }_{j = 1}^{n} \) do not necessarily need to be distributed over the entire domain \( {\mathbb{S}}^{d} \). By applying a shift, we may assume that \( 0 \in  \Omega \). Let \( {\lambda }_{\Omega } \mathrel{\text{:= }} \operatorname{diam}\left( \Omega \right) \). Then, for any \( x \in  \Omega \), we have

\[{\sigma }_{k}\left( {\theta  \cdot  \widetilde{x}}\right)  = {\left( \theta  \cdot  \widetilde{x}\right) }^{k},\;\theta  \in  {G}_{ + } \mathrel{\text{:= }} \left\{  {\theta  \in  {\mathbb{S}}^{d}: {\theta }_{d + 1} \geq  \frac{{\lambda }_{\Omega }}{\sqrt{1 + {\lambda }_{\Omega }^{2}}}}\right\} ,\]

\[{\sigma }_{k}\left( {\theta  \cdot  \widetilde{x}}\right)  = 0,\;\theta  \in  {G}_{ - } \mathrel{\text{:= }} \left\{  {\theta  \in  {\mathbb{S}}^{d}: {\theta }_{d + 1} \leq   - \frac{{\lambda }_{\Omega }}{\sqrt{1 + {\lambda }_{\Omega }^{2}}}}\right\} .\]

We take points \( {\left\{  {\vartheta }_{j}^{ * }\right\}  }_{j = 1}^{\left( \begin{matrix} k + d \\  d \end{matrix}\right) } \) in \( {G}_{ + } \) such that the collection \( {\left\{  {\left( {\vartheta }_{j}^{ * } \cdot  \widetilde{x}\right) }^{k}\right\}  }_{j = 1}^{\left( \begin{matrix} k + d \\  d \end{matrix}\right) } \) are linearly independent.

To get the linear approximation rates in Theorem 2.2, it suffices to combine the points \( {\left\{  {\vartheta }_{j}^{ * }\right\}  }_{j = 1}^{\left( \begin{matrix} k + d \\  d \end{matrix}\right) } \) with a collection of well-distributed points in

\[{\mathbb{S}}^{d} \smallsetminus  \left( {{G}_{ + } \cup  {G}_{ - }}\right)  = \left\{  {\theta  \in  {\mathbb{S}}^{d}: \left| {\theta }_{d + 1}\right|  \leq  \frac{{\lambda }_{\Omega }}{\sqrt{1 + {\lambda }_{\Omega }^{2}}}}\right\} .\]

It is worth noting that \( {\left\{  {\vartheta }_{j}^{ * }\right\}  }_{j = 1}^{\left( \begin{matrix} k + d \\  d \end{matrix}\right) } \) can also be replaced by a collection of points \( {\left\{  {\theta }_{j}^{ * }\right\}  }_{j = 1}^{\left( \begin{matrix} k + d \\  d \end{matrix}\right) } \cup  {\left\{  -{\theta }_{j}^{ * }\right\}  }_{j = 1}^{\left( \begin{matrix} k + d \\  d \end{matrix}\right) } \) in \( {\mathbb{S}}^{d} \smallsetminus  \left( {{G}_{ + } \cup  {G}_{ - }}\right) \), since there is the relation \( {t}^{k} = {\sigma }_{k}\left( t\right)  + {\left( -1\right) }^{k}{\sigma }_{k}\left( {-t}\right) \).

Corollary 2.1 (Theorem 8.2. [32]). Let \( d, n, k,\Omega \) be as in Theorem 2.2, \( {n}_{1} \simeq  {n}^{\frac{d - 1}{d}} \) and \( {n}_{2} \simeq  {n}^{\frac{1}{d}} \), there exists quadrature points \( {\left\{  {w}_{j}^{ * }\right\}  }_{j = 1}^{{n}_{1}} \subset  {\mathbb{S}}^{d - 1} \) and uniform distributed points \( {\left\{  {b}_{i}^{ * }\right\}  }_{i = 1}^{{n}_{2}} \subset  \left\lbrack  {-1,1}\right\rbrack \), such that for any \( f \in  {\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( {\mathbb{B}}^{d}\right) \),

\[\mathop{\inf }\limits_{{{a}_{i, j} \in  \mathbb{R},\forall i, j}}{\begin{Vmatrix}f\left( x\right)  - \mathop{\sum }\limits_{{j = 1}}^{{n}_{2}}\mathop{\sum }\limits_{{i = 1}}^{{n}_{1}}{a}_{i, j}{\sigma }_{k}\left( {w}_{j}^{ * } \cdot  x + {b}_{i}^{ * }\right) \end{Vmatrix}}_{{\mathcal{L}}^{2}\left( {\mathbb{B}}^{d}\right) } \lesssim  {n}^{-\frac{1}{2} - \frac{{2k} + 1}{2d}}\parallel f{\parallel }_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( {\mathbb{B}}^{d}\right) }. \tag{2.15}\]

Proof. The points constructed in [32] satisfies

\[\mathop{\max }\limits_{{\left( \begin{matrix} w \\  b \end{matrix}\right)  \in  {\mathbb{S}}^{d - 1} \times  \left\lbrack  {-1,1}\right\rbrack  }}\mathop{\min }\limits_{\substack{{1 \leq  j \leq  {n}_{1}} \\  {1 \leq  i \leq  {n}_{2}} }}\left| {\left( \begin{matrix} w \\  b \end{matrix}\right)  - \left( \begin{matrix} {w}_{j} \\  {b}_{i} \end{matrix}\right) }\right|  \lesssim  {n}^{-\frac{1}{d}}.\]

Then it suffices to notice the normalization \( \mathfrak{P}: {\mathbb{S}}^{d - 1} \times  \left\lbrack  {-1,1}\right\rbrack   \rightarrow  \left\{  {\theta  \in  {\mathbb{S}}^{d}: \left| {\theta }_{d + 1}\right|  \leq  \frac{1}{\sqrt{2}}}\right\} \)

\[\mathfrak{P}: \left( \begin{array}{l} w \\  b \end{array}\right)  \mapsto  \frac{1}{\sqrt{1 + {b}^{2}}}\left( \begin{array}{l} w \\  b \end{array}\right),\;\left( \begin{array}{l} w \\  b \end{array}\right)  \in  {\mathbb{S}}^{d - 1} \times  \left\lbrack  {1,1}\right\rbrack \tag{2.16}\]

is a uniform homeomorphism. Together with Remark 2.1 completes the proof.

As we mentioned in (2.7), Theorem 2.2 is optimal up to logarithmic factors. Interestingly, it leads to an important function space embedding result, previously established in [39]:

\[{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right)  \hookrightarrow  {\mathcal{B}}^{k}\left( \Omega \right) \tag{2.17}\]

and the gap between the above two spaces is small in the sense that the metric entropies of their unit balls are of the same order, namely

\[{\epsilon }_{n}{\left( \mathbb{B}\left( {\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) \right) \right) }_{{\mathcal{L}}^{2}\left( \Omega \right) } \simeq  {\epsilon }_{n}{\left( \mathbb{B}\left( {\mathcal{B}}^{k}\left( \Omega \right) \right) \right) }_{{\mathcal{L}}^{2}\left( \Omega \right) } \simeq  {n}^{-\frac{d + {2k} + 1}{2d}}. \tag{2.18}\]

Here the metric entropies mean the infimum of \( \lambda \) such that \( {2}^{n}{\mathcal{L}}^{2} \) -balls of radius \( \lambda \) can cover the unit balls in \( {\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) \) and \( {\mathcal{B}}^{k}\left( \Omega \right) \) respectively [72]. The first equivalence in (2.18) follows from classical results on Sobolev space (see, e.g., [64]), and the second equivalence was recently established in [28].

Corollary 2.2 (Theorem 1. [39]). Let \( k, d,\Omega \) be as in (2.10), then there is the continuous embedding

\[{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right)  \hookrightarrow  {\mathcal{B}}^{k}\left( \Omega \right)\]

Proof. Let \( f \in  {\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) \) and \( {f}_{n} \in  {L}_{n, M}^{k} \) be the approximants of \( f \) in Theorem 2.2. Then

\[{f}_{n} \in  {L}_{n, M}^{k} \subset  {\sum }_{n, M}^{k}.\]

Notice that \( \mathop{\lim }\limits_{{n \rightarrow  \infty }}{\begin{Vmatrix}{f}_{n} - f\end{Vmatrix}}_{{\mathcal{L}}^{2}\left( \Omega \right) } = 0 \) implies

\[f = \mathop{\lim }\limits_{{n \rightarrow  \infty }}{f}_{n} \subset  \overline{\mathop{\bigcup }\limits_{{n = 1}}^{\infty }{\sum }_{n, M}^{k}} = M\overline{\operatorname{conv}\left( {\mathbb{D}}_{{\sigma }_{k}}\right) }. \tag{2.19}\]

It means

\[\parallel f{\parallel }_{{\mathcal{B}}^{k}\left( \Omega \right) } \leq  M \lesssim  \parallel f{\parallel }_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) }.\]

The embedding theorem provides some aspect of characterizing the smoothness of Barron spaces in terms of smoothness. In particular, the metric entropy argument confirms that \( {\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) \) is the maximal Sobolev space for the embedding. We mentioned the integral representation for Barron spaces in (2.4), which coincides with the definitions used in [19, 20, 21] (see also [73, 13] for similar spaces). To further clarify the difference between \( {\mathcal{B}}^{k}\left( \Omega \right) \) and \( {\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) \), we present a new integral representation result for Sobolev spaces in comparison.

Theorem 2.3. Let \( d \in  \mathbb{N}, k \in  {\mathbb{N}}_{0},\Omega  \subset  {\mathbb{R}}^{d} \) be a bounded domain, then the Sobolev space \( {\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) \) given in (2.9) can be alternatively written as

\[{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right)  = \left\{  {{\oint }_{{\mathbb{S}}^{d}}{\sigma }_{k}\left( {\theta  \cdot  \widetilde{x}}\right) \psi \left( \theta \right) {d\theta }: \psi  \in  {\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) }\right\} . \tag{2.20}\]

In particular, for any \( f \in  {\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) \), there exists a \( \psi  \in  {\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) \) (that may not be unique) such that

\[f\left( x\right)  = {\oint }_{{\mathbb{S}}^{d}}{\sigma }_{k}\left( {\theta  \cdot  \widetilde{x}}\right) \psi \left( \theta \right) {d\theta }. \tag{2.21}\]

Furthermore,

\[\parallel f{\parallel }_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) } \simeq  \mathop{\inf }\limits_{{\psi  \in  {\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) }}\left\{  {\parallel \psi {\parallel }_{{\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) }: \psi \text{ satisfies (2.21) }}\right\} , \tag{2.22}\]

This theorem provides a new characterization of Sobolev spaces and highlights the structural differences between Barron spaces and Sobolev spaces. To specify, (2.4) and Theorem 2.3 establish a space isomorphism (as vector spaces, not Banach spaces)

\[{\mathcal{B}}^{k}\left( \Omega \right) /{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right)  \cong  \left( {\mathcal{M}\left( {\mathbb{S}}^{d}\right) /{\mathcal{N}}_{k}}\right) /\left( {{\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) /\left( {{\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right)  \cap  {\mathcal{N}}_{k}}\right) }\right)  \cong  \mathcal{M}\left( {\mathbb{S}}^{d}\right) /\left( {{\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right)  + {\mathcal{N}}_{k}}\right),\]

where \( {\mathcal{N}}_{k} \) is the kernel of the homomorphism \( \mu  \mapsto  {\int }_{{\mathbb{S}}^{d}}{\sigma }_{k}\left( {\theta  \cdot  \widetilde{x}}\right) {d\mu }\left( \theta \right) \):

\[{\mathcal{N}}_{k} = \left\{  {\mu  \in  \mathcal{M}\left( {\mathbb{S}}^{d}\right) : {\int }_{{\mathbb{S}}^{d}}{\sigma }_{k}\left( {\theta  \cdot  \widetilde{x}}\right) {d\mu }\left( \theta \right)  = 0,\;\text{ a.e. }}\right\} , \tag{2.23}\]

and \( \mathcal{M}\left( {\mathbb{S}}^{d}\right) \) is the space of signed Borel measures. Notably, we slightly abuse notation by treating \( {\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) \) as a subspace of \( \mathcal{M}\left( {\mathbb{S}}^{d}\right) \). Specifically, any function \( \psi  \in  {\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) \) can be naturally identified with the measure \( \mu  \in  \mathcal{M}\left( {\mathbb{S}}^{d}\right) \) whose density function is given by \( \psi \).

An interesting special case arises when considering even/odd Barron spaces and Sobolev spaces defined on the sphere \( {\mathbb{S}}^{d} \). In this setting, \( {\mathcal{N}}_{k} = \{ 0\} \), which implies that the quotient space \( {\mathcal{B}}^{k}\left( {\mathbb{S}}^{d}\right) /{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( {\mathbb{S}}^{d}\right) \) is almost equivalent to the quotient \( \mathcal{M}\left( {\mathbb{S}}^{d}\right) /{\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) \), up to even/odd function spaces (see Theorem 4.2).

Moreover, this characterization has direct implications for function approximation. Specifically, it implies that functions that can be approximated as in (2.14) necessarily belong to \( {\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) \). We summarize all these insights above and provide an analogue of Theorem 2.1.

Theorem 2.4. Let \( d \in  \mathbb{N}, k \in  {\mathbb{N}}_{0},\Omega  \subset  {\mathbb{R}}^{d} \) be a bounded domain, then

1. \( f \in  {\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) \) if and only if, with some \( M \simeq  \parallel f{\parallel }_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) } \), there exists a sequence of spaces \( {\left\{  {L}_{n, M}^{k}\right\}  }_{n = 1}^{\infty } \) induced by quasi-uniform weights \( {\left\{  {\theta }_{j, n}^{ * }\right\}  }_{j = 1}^{n} \),

\[\mathop{\lim }\limits_{{n \rightarrow  \infty }}\mathop{\inf }\limits_{{{f}_{n} \in  {L}_{n, M}^{k}}}{\begin{Vmatrix}f - {f}_{n}\end{Vmatrix}}_{{\mathcal{L}}^{2}\left( \Omega \right) } = 0. \tag{2.24}\]

2. The Sobolev space \( {\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) \) can be alternatively written as

\[{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right)  = \left\{  {{\int }_{{\mathbb{S}}^{d}}{\sigma }_{k}\left( {\theta  \cdot  \widetilde{x}}\right) \psi \left( \theta \right) {d\theta }: \psi  \in  {\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) }\right\} . \tag{2.25}\]

Furthermore,

\[\parallel f{\parallel }_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) } \simeq  \mathop{\inf }\limits_{{\psi  \in  {\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) }}\left\{  {\parallel \psi {\parallel }_{{\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) }: f\left( x\right)  = {\int }_{{\mathbb{S}}^{d}}{\sigma }_{k}\left( {\theta  \cdot  \widetilde{x}}\right) \psi \left( \theta \right) {d\theta }}\right\} . \tag{2.26}\]

3. Let \( f \in  {\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) \), with some \( M \simeq  \parallel f{\parallel }_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) } \), we have

\[\mathop{\inf }\limits_{{{f}_{n} \in  {L}_{n, M}^{k}}}{\begin{Vmatrix}f - {f}_{n}\end{Vmatrix}}_{{\mathcal{L}}^{2}\left( \Omega \right) } \lesssim  {n}^{-\frac{1}{2} - \frac{{2k} + 1}{2d}}\parallel f{\parallel }_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) }. \tag{2.27}\]

All the corresponding constants are independent of \( n,{\left\{  {\theta }_{j}^{ * }\right\}  }_{j = 1}^{n} \), and \( f \).

Proof. Given Theorem 2.2 and 2.3, we only need to prove (2.24) implies \( f \in  {\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) \). Let each \( {f}_{n} \) in (2.24) have the form

\[{f}_{n}\left( x\right)  = \mathop{\sum }\limits_{{j = 1}}^{n}a{\left( n\right) }_{j}{\sigma }_{k}\left( {{\theta }_{j, n}^{ * } \cdot  \widetilde{x}}\right).\]

We can take a disjoint quasi-uniform partition \( {\mathbb{S}}^{d} = \mathop{\bigcup }\limits_{{j = 1}}^{n}{A}_{j, n} \) with

\[{\theta }_{j, n}^{ * } \in  {A}_{j, n},\;{\oint }_{{A}_{j, n}}{1d\eta } \simeq  {n}^{-1},\;\operatorname{diam}\left( {A}_{j, n}\right)  \lesssim  {n}^{-\frac{1}{d}},\; j = 1,\ldots, n. \tag{2.28}\]

Then the piecewise constant function \( {\psi }_{n} \) given by \( {\psi }_{n} = \mathop{\sum }\limits_{{j = 1}}^{n}\frac{a{\left( n\right) }_{j}}{{\int }_{{A}_{j, n}}{d\eta }}{\mathbf{1}}_{{A}_{j, n}} \) has a convergence subsequence. One could verify the limit of this subsequence is the desired function \( \psi \).

To understand the difference between nonlinear and linear ReLU \( {}^{k} \) networks, we can compare Theorem 2.4 with Theorem 2.1. We remark that the differences between the conditions here and those in Theorem 2.1 are (1) the parameters \( {\left\{  {\theta }_{j}\right\}  }_{j = 1}^{n} \) here are well-distributed whereas they are arbitrary in Theorem 2.1, and (2) the condition \( \parallel a{\parallel }_{2} \leq  \frac{M}{\sqrt{n}} \) here is stronger than \( \parallel a{\parallel }_{1} \leq  M \) in Theorem 2.1, which follows from Schwartz's inequality.

## 3 Spherical harmonics and Legendre polynomials

This subsection briefly reviews harmonic analysis on the unit sphere \( {\mathbb{S}}^{d} \mathrel{\text{:= }} \left\{  {\eta  \in  {\mathbb{R}}^{d + 1}: \left| \eta \right|  = 1}\right\} \), following [74, 75]. The average integral uses the normalized surface measure \( {d\eta } \),

\[{\oint }_{{\mathbb{S}}^{d}}f\left( \eta \right) {d\eta } = \frac{1}{{\omega }_{d}}{\int }_{{\mathbb{S}}^{d}}f\left( \eta \right) {d\eta },\; f \in  {\mathcal{L}}^{1}\left( {\mathbb{S}}^{d}\right),\]

where \( {\omega }_{d} = {\int }_{{\mathbb{S}}^{d}}{1d\eta } \). The geodesic distance is \( \rho \left( {\eta,\theta }\right)  = \arccos \left( {\eta  \cdot  \theta }\right) \).

Let \( {\mathbb{P}}_{m}\left( {\mathbb{S}}^{d}\right) \) be the space of polynomials of degree at most \( m \) restricted to \( {\mathbb{S}}^{d} \), with inner product

\[\langle p, q{\rangle }_{{\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) } \mathrel{\text{:= }} {\oint }_{{\mathbb{S}}^{d}}p\left( \eta \right) q\left( \eta \right) {d\eta }.\]

The dimension of \( {\mathbb{P}}_{m}\left( {\mathbb{S}}^{d}\right) \) is \( \left( \begin{matrix} d + 1 + m \\  m \end{matrix}\right) \) for \( m = 0,1 \) and \( \left( \begin{matrix} d + 1 + m \\  m \end{matrix}\right)  - \left( \begin{matrix} d - 1 + m \\  m - 2 \end{matrix}\right) \) for \( m \geq  2 \).

The space of spherical harmonics \( {\mathbb{Y}}_{m} \) is the orthogonal complement of \( {\mathbb{P}}_{m - 1}\left( {\mathbb{S}}^{d}\right) \) in \( {\mathbb{P}}_{m}\left( {\mathbb{S}}^{d}\right) \), which is known as the space of spherical harmonics of degree \( m \). Let \( {\left\{  {Y}_{m,\ell }\right\}  }_{\ell  = 1}^{N\left( m\right) } \) be an orthonormal basis for \( {\mathbb{Y}}_{m} \), then its dimension is \( N\left( 0\right)  = \dim \left( {\mathbb{Y}}_{0}\right)  = 1 \) and

\[N\left( m\right)  = \dim \left( {\mathbb{Y}}_{m}\right)  = \frac{{2m} + d - 1}{m}\left( \begin{matrix} m + d - 2 \\  d - 1 \end{matrix}\right),\; m \geq  0.\]

By Weierstrass’ theorem, any \( f \in  {\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) \) has the harmonic expansion

\[f\left( \eta \right)  = \mathop{\sum }\limits_{{m = 0}}^{\infty }\mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}\widehat{f}\left( {m,\ell }\right) {Y}_{m,\ell }\left( \eta \right),\;\text{ a.e. }\eta  \in  {\mathbb{S}}^{d},\]

where \( \widehat{f}\left( {m,\ell }\right)  = {\left\langle  f,{Y}_{m,\ell }\right\rangle  }_{{\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) } \). The \( {\mathcal{L}}^{2} \) -projection onto \( {\mathbb{Y}}_{m} \) is \( {\Pi }_{m}f = \mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}\widehat{f}\left( {m,\ell }\right) {Y}_{m,\ell } \).

Given the definition of projections, we are ready to define the Sobolev spaces.

Definition 3.1 (Sobolev spaces on the sphere). For \( r > 0 \), the Sobolev space \( {\mathcal{H}}^{r}\left( {\mathbb{S}}^{d}\right) \) is defined as \( {\mathcal{H}}^{r}\left( {\mathbb{S}}^{d}\right)  = \left\{  {f \in  {\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) : \parallel f{\parallel }_{{\mathcal{H}}^{r}\left( {\mathbb{S}}^{d}\right) } < \infty }\right\} \), with norm squared

\[\parallel f{\parallel }_{{\mathcal{H}}^{r}\left( {\mathbb{S}}^{d}\right) }^{2} = \parallel f{\parallel }_{{\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) }^{2} + \mathop{\sum }\limits_{{m = 1}}^{\infty }{m}^{2r}{\begin{Vmatrix}{\Pi }_{m}f\end{Vmatrix}}_{{\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) }^{2} = \mathop{\sum }\limits_{{m = 0}}^{\infty }\mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}\left( {{m}^{2r} + 1}\right) {\left| \widehat{f}\left( m,\ell \right) \right| }^{2}. \tag{3.1}\]

### 3.1 Legendre polynomial and Legendre expansion of \( {\sigma }_{k} \)

Define the space \( {\mathcal{L}}_{{w}_{d}}^{2}\left( \left\lbrack  {-1,1}\right\rbrack  \right) \) by

\[\langle f, g{\rangle }_{{w}_{d}} = {\int }_{-1}^{1}f\left( t\right) g\left( t\right) {\left( 1 - {t}^{2}\right) }^{\frac{d - 2}{2}}{dt},\;\parallel f{\parallel }_{{\mathcal{L}}_{{w}_{d}}^{2}\left( \left\lbrack  {-1,1}\right\rbrack  \right) } = \langle f, f{\rangle }_{{w}_{d}}^{\frac{1}{2}}. \tag{3.2}\]

The space \( {\mathcal{L}}_{{w}_{d}}^{2}\left( \left\lbrack  {-1,1}\right\rbrack  \right) \) has an orthogonal polynomial basis \( {\left\{  {p}_{m}\right\}  }_{m = 0}^{\infty } \) with \( \deg \left( {p}_{m}\right)  = m \) satisfying

\[{\left\langle  {p}_{m},{p}_{l}\right\rangle  }_{{w}_{d}} = 0,\; l \neq  m,\; m \in  \mathbb{N}. \tag{3.3}\]

Such polynomials are called Legendre polynomials, which are known to necessarily have the form (see, e.g., [76])

\[{p}_{m}\left( t\right)  = {\lambda }_{m}{\left( 1 - {t}^{2}\right) }^{-\frac{d - 2}{2}}{\left( \frac{d}{dt}\right) }^{m}\left\lbrack  {\left( 1 - {t}^{2}\right) }^{m + \frac{d - 2}{2}}\right\rbrack ,\; t \in  \left\lbrack  {-1,1}\right\rbrack . \tag{3.4}\]

In this paper, we choose the normalization factors \( {\left\{  {\lambda }_{m}\right\}  }_{m = 0}^{\infty } \) properly such that (3.6) below holds.

The integration formula (see, e.g., [74, Lemma A.5.2.]) shows a univariate function \( f \in \; {\mathcal{L}}_{{w}_{d}}^{2}\left( \left\lbrack  {-1,1}\right\rbrack  \right) \) has the property

\[{\omega }_{d - 1}{\int }_{-1}^{1}f\left( t\right) {\left( 1 - {t}^{2}\right) }^{\frac{d - 2}{2}}{dt} = {\omega }_{d}{\oint }_{{\mathbb{S}}^{d}}f\left( {\theta  \cdot  \eta }\right) {d\eta },\;\theta  \in  {\mathbb{S}}^{d}. \tag{3.5}\]

By [74, Theorem 1.2.6], there exist Legendre polynomials \( {\left\{  {p}_{m}\right\}  }_{m = 0}^{\infty } \) with \( \deg \left( {p}_{m}\right)  = m \) such that

\[{p}_{m}\left( {\eta  \cdot  \theta }\right)  = \mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}{Y}_{m,\ell }\left( \eta \right) {Y}_{m,\ell }\left( \theta \right). \tag{3.6}\]

With (3.5) and (3.6), the polynomials \( {\left\{  {p}_{m}\right\}  }_{m = 0}^{\infty } \) form an orthogonal basis with respect to the weights \( {\left( 1 - {t}^{2}\right) }^{\frac{d - 2}{2}} \). Together with (3.3), we call \( {\left\{  {p}_{m}\right\}  }_{m = 0}^{\infty } \) in (3.6) to be the Legendre polynomials throughout this paper.

We remark that (3.6) are not standard Legendre polynomials as they are not normalized to have norm equal to 1. The norms of \( {\left\{  {p}_{m}\right\}  }_{m = 0}^{\infty } \) can be determined by using the integration formula (3.5):

\[{\begin{Vmatrix}{p}_{m}\end{Vmatrix}}_{{\mathcal{L}}_{{w}_{d}}^{2}\left( \left\lbrack  {-1,1}\right\rbrack  \right) } = {\int }_{-1}^{1}{p}_{m}{\left( t\right) }^{2}{\left( 1 - {t}^{2}\right) }^{\frac{d - 2}{2}}{dt} = \frac{{\omega }_{d}}{{\omega }_{d - 1}}{\int }_{{\mathbb{S}}^{d}}{p}_{m}{\left( {e}_{1} \cdot  \eta \right) }^{2}{d\eta }\]

\[= \frac{{\omega }_{d}}{{\omega }_{d - 1}}{\int }_{{\mathbb{S}}^{d}}{\left( \mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}{Y}_{m,\ell }\left( {e}_{1}\right) {Y}_{m,\ell }\left( \eta \right) \right) }^{2}{d\eta } = \frac{{\omega }_{d}}{{\omega }_{d - 1}}\mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}{Y}_{m,\ell }{\left( {e}_{1}\right) }^{2} = \frac{{\omega }_{d}}{{\omega }_{d - 1}}{p}_{m}\left( 1\right)\]

\[= \frac{{\omega }_{d}}{{\omega }_{d - 1}}{\int }_{{\mathbb{S}}^{d}}{p}_{m}\left( 1\right) {d\eta } = \frac{{\omega }_{d}}{{\omega }_{d - 1}}{\int }_{{\mathbb{S}}^{d}}\mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}{Y}_{m,\ell }{\left( \eta \right) }^{2}{d\eta } = \frac{{\omega }_{d}}{{\omega }_{d - 1}}N\left( m\right). \tag{3.7}\]

Although we will not need them, the normalization factors \( {\left\{  {\lambda }_{m}\right\}  }_{m = 0}^{\infty } \) in (3.4) can be computed by comparing (3.7) with the norm of the standard Legendre polynomials (see, e.g., [76, Chapter 4.3]),

\[{\lambda }_{m} = \frac{{\omega }_{d}}{{\omega }_{d - 1}}\frac{N\left( m\right) }{\Gamma \left( {m + d/2}\right) }\sqrt{\frac{\left( {{2m} + d - 1}\right) \Gamma \left( {m + d - 1}\right) }{{2}^{{2m} + d - 1}\Gamma \left( {m + 1}\right) }}.\]

The function \( {\sigma }_{k} \in  {\mathcal{L}}_{{w}_{d}}^{2}\left( \left\lbrack  {-1,1}\right\rbrack  \right) \) has the Legendre expansion in terms of the orthogonal basis \( {\left\{  {p}_{m}\right\}  }_{m = 0}^{\infty } \) as

\[{\sigma }_{k} = \mathop{\sum }\limits_{{m = 0}}^{\infty }\widehat{{\sigma }_{k}}\left( m\right) {p}_{m} \tag{3.8}\]

where the Legendre coefficients are given as

\[\widehat{{\sigma }_{k}}\left( m\right)  = \frac{{\left\langle  {p}_{m},{\sigma }_{k}\right\rangle  }_{{w}_{d}}}{{\begin{Vmatrix}{p}_{m}\end{Vmatrix}}_{{\mathcal{L}}_{{w}_{d}}^{2}\left( \left\lbrack  {-1,1}\right\rbrack  \right) }^{2}}.\]

The coefficients \( {\left\{  \widehat{{\sigma }_{k}}\left( n\right) \right\}  }_{n = 0}^{\infty } \) are studied in [77,78,79,26]. Denote the set

\[{E}_{{\sigma }_{k}} \mathrel{\text{:= }} \left\{  {m \in  \mathbb{N}: \widehat{{\sigma }_{k}}\left( m\right)  \neq  0}\right\} , \tag{3.9}\]

then by [26, Appendix D.2],

\[{E}_{{\sigma }_{k}} = \{ m \geq  k + 1: m - k\text{ is odd }\}  \cup  \{ 0,\ldots, k\}\]

\[\widehat{{\sigma }_{k}}\left( m\right)  = \frac{{\omega }_{d - 1}k!\Gamma \left( {d/2}\right) }{{\omega }_{d}}\frac{{\left( -1\right) }^{\left( {m - k - 1}\right) /2}\Gamma \left( {m - k}\right) }{{2}^{m}\Gamma \left( \frac{m - k + 1}{2}\right) \Gamma \left( \frac{m + d + k + 1}{2}\right) },\; m \in  {E}_{{\sigma }_{k}}. \tag{3.10}\]

To proceed, we introduce the standard notation of forward and backward difference. Given any \( K \in  \mathbb{N} \) and a sequence \( \{ \mathfrak{a}\left( m\right) {\} }_{m = K}^{\infty } \), we denote the forward difference \( \{ \left( {\Delta \mathfrak{a}}\right) \left( m\right) {\} }_{m = K}^{\infty } \) by

\[\left( {\Delta \mathfrak{a}}\right) \left( m\right)  = \mathfrak{a}\left( {m + 1}\right)  - \mathfrak{a}\left( m\right),\; m \geq  K \tag{3.11}\]

and the \( \beta \) -th forward difference \( {\left\{  \left( {\Delta }^{\beta }\mathfrak{a}\right) \left( m\right) \right\}  }_{m = K}^{\infty } \) by

\[\left( {{\Delta }^{\beta }\mathfrak{a}}\right) \left( m\right)  = \underset{\beta }{\underbrace{\Delta  \circ  \cdots  \circ  \Delta }}\mathfrak{a}\left( m\right)  = \mathop{\sum }\limits_{{j = 0}}^{\beta }\left( \begin{array}{l} \beta \\  j \end{array}\right) {\left( -1\right) }^{\beta  - j}\mathfrak{a}\left( {m + j}\right),\; m \in  \mathbb{N}. \tag{3.12}\]

Similarly, writing \( \mathfrak{a}\left( {K - 1}\right)  \mathrel{\text{:= }} 0 \), we denote the backward difference \( \{ \left( {\nabla \mathfrak{a}}\right) \left( m\right) {\} }_{m = K}^{\infty } \) by

\[\left( {\nabla \mathfrak{a}}\right) \left( m\right)  = \mathfrak{a}\left( m\right)  - \mathfrak{a}\left( {m - 1}\right),\; m \geq  K \tag{3.13}\]

and \( {\left\{  \left( {\nabla }^{\beta }\mathfrak{a}\right) \left( m\right) \right\}  }_{m = K}^{\infty } \) by

\[\left( {{\nabla }^{\beta }\mathfrak{a}}\right) \left( m\right)  = \underset{\beta }{\underbrace{\nabla  \circ  \cdots  \circ  \nabla }} \circ  \mathfrak{a}\left( m\right),\; m \geq  K. \tag{3.14}\]

It is easy to see the inverses of \( \nabla \) and \( {\nabla }^{\beta } \) are

\[\left( {{\nabla }^{-1}\mathfrak{a}}\right) \left( m\right)  = \mathop{\sum }\limits_{{\nu  = 0}}^{m}\mathfrak{a}\left( \nu \right)\]

\[\left( {{\nabla }^{-\beta }\mathfrak{a}}\right) \left( m\right)  = \underset{\beta }{\underbrace{{\nabla }^{-1} \circ  \cdots  \circ  {\nabla }^{-1}}} \circ  \mathfrak{a}\left( m\right)  = \mathop{\sum }\limits_{{\nu  = 0}}^{m}\left( \begin{matrix} m + \beta  - \nu \\  \beta  \end{matrix}\right) \mathfrak{a}\left( \nu \right) \tag{3.15}\]

We have the following lemma for the coefficients of \( {\sigma }_{k} \).

Lemma 3.1. Let \( r \leq  \frac{d + {2k} + 1}{2} \), there exists a function \( \xi : \lbrack k + 1,\infty ) \rightarrow  \lbrack 0,\infty ) \) such that

\[
\begin{aligned}
\xi \left( m\right)  &= {\widehat{\sigma }}_{k}{\left( m\right) }^{2}{m}^{2r},\; m \geq  k + 1, m \in  {E}_{{\sigma }_{k}},\\
0 &\leq  {\left( -1\right) }^{\beta }\left( {{\Delta }^{\beta }\xi }\right) \left( m\right)  \lesssim  {m}^{{2r} - \left( {d + {2k} + 1}\right)  - \beta },\;\beta  = 0,1,\ldots .
\end{aligned}
\tag{3.16}\]

In particular, if \( r < \frac{d + {2k} + 1}{2} \),

\[{\left( -1\right) }^{\beta }\left( {{\Delta }^{\beta }\xi }\right) \left( m\right)  \simeq  {m}^{{2r} - \left( {d + {2k} + 1}\right)  - \beta },\;\beta  = 0,1,\ldots. \tag{3.17}\]

Proof. Applying the Legendre duplication formula,

\[\frac{\Gamma \left( {m - k}\right) }{{2}^{m}\Gamma \left( \frac{m - k}{2}\right) \Gamma \left( \frac{m - k + 1}{2}\right) } = \frac{1}{{2}^{k + 1}\sqrt{\pi }},\; m - k \notin   - \frac{\mathbb{N}}{2},\]

we have

\[\frac{\Gamma \left( {m - k}\right) }{{2}^{m}\Gamma \left( \frac{m - k + 1}{2}\right) \Gamma \left( \frac{m + d + k + 1}{2}\right) } = \frac{1}{{2}^{k + 1}\sqrt{\pi }}\frac{\Gamma \left( \frac{m - k}{2}\right) }{\Gamma \left( \frac{m + d + k + 1}{2}\right) },\; m \geq  k + 1.\]

Denote the function

\[\xi \left( t\right)  = {\left( \frac{{\omega }_{d - 1}}{{\omega }_{d}}\frac{k!\Gamma \left( {d/2}\right) }{{2}^{k + 1}\sqrt{\pi }}\right) }^{2}{t}^{2r}{\left( \frac{\Gamma \left( \frac{t - k}{2}\right) }{\Gamma \left( \frac{t + d + k + 1}{2}\right) }\right) }^{2}, \tag{3.18}\]

then with (3.10), we have

\[\xi \left( m\right)  = {\widehat{\sigma }}_{k}{\left( m\right) }^{2}{m}^{2r},\; m \geq  k + 1, m \in  {E}_{{\sigma }_{k}}.\]

Now for \( t > k \),

\[\frac{{\xi }^{\prime }\left( t\right) }{\xi \left( t\right) } = \frac{d}{dt}\log \xi \left( t\right)  = \frac{2r}{t} + \psi \left( \frac{t - k}{2}\right)  - \psi \left( \frac{t + d + k + 1}{2}\right) \tag{3.19}\]

where \( \psi \) is the digamma function \( \psi \left( t\right)  = \frac{d}{dt}\left( {\log \Gamma \left( t\right) }\right) \). The derivatives of \( \psi \) are called polygamma functions and have series representations

\[{\psi }^{\left( j\right) }\left( t\right)  = {\left( -1\right) }^{j + 1}j!\mathop{\sum }\limits_{{\nu  = 0}}^{\infty }{\left( t + \nu \right) }^{-\left( {j + 1}\right) } \approx  {\left( -1\right) }^{j + 1}\left( {j - 1}\right)!{t}^{-j},\]

where the notation \( \approx \) here signifies \( \mathop{\lim }\limits_{{t \rightarrow  \infty }}\frac{{\psi }^{\left( j\right) }\left( t\right) }{{\left( -1\right) }^{j + 1}\left( {j - 1}\right)!{t}^{-j}} = 1 \).

We prove (3.16) by induction. Suppose for each \( j = 0,\ldots, r \) we have

\[{\left( d + 2k + 1 - 2r\right) }^{j} \leq  \mathop{\lim }\limits_{{t \rightarrow  \infty }}\frac{{\left( -1\right) }^{j}{\xi }^{\left( j\right) }\left( t\right) }{{t}^{{2r} - \left( {d + {2k} + 1}\right)  - j}} \leq  {\left( d + 2k + j + 1 - 2r\right) }^{j}, \tag{3.20}\]

\[{\left( -1\right) }^{j}{\xi }^{\left( j\right) }\left( t\right)  \geq  0,\; t \geq  k + 1,\]

then

\[{\xi }^{\left( \beta  + 1\right) }\left( t\right)  = {\left( \frac{d}{dt}\right) }^{\beta }\left\lbrack  {\xi \left( t\right) \frac{{\xi }^{\prime }\left( t\right) }{\xi \left( t\right) }}\right\rbrack   = \mathop{\sum }\limits_{{j = 0}}^{\beta }\left( \begin{array}{l} \beta \\  j \end{array}\right) {\xi }^{\left( \beta  - j\right) }\left( t\right) {\left( \frac{d}{dt}\right) }^{j}\left( \frac{{\xi }^{\prime }\left( t\right) }{\xi \left( t\right) }\right)\]

\[= \mathop{\sum }\limits_{{j = 0}}^{\beta }\left( \begin{array}{l} \beta \\  j \end{array}\right) {\xi }^{\left( \beta  - j\right) }\left( t\right) \left( {{2r}{\left( -1\right) }^{j}j!{t}^{-\left( {j + 1}\right) } + {2}^{-j}{\psi }^{\left( j\right) }\left( \frac{t - k}{2}\right)  - {2}^{-j}{\psi }^{\left( j\right) }\left( \frac{t + d + k + 1}{2}\right) }\right)\]

\[\approx  \mathop{\sum }\limits_{{j = 0}}^{\beta }\left( \begin{array}{l} \beta \\  j \end{array}\right) {\xi }^{\left( \beta  - j\right) }\left( t\right) \left\lbrack  {{2r}{\left( -1\right) }^{j}j!{t}^{-\left( {j + 1}\right) } + {2}^{-j}{\left( -1\right) }^{j + 1}\left( {j - 1}\right)!\left( {{\left( \frac{t - k}{2}\right) }^{-j} - {\left( \frac{t + d + k + 1}{2}\right) }^{-j}}\right) }\right\rbrack\]

\[\approx  \mathop{\sum }\limits_{{j = 0}}^{\beta }\left( \begin{array}{l} \beta \\  j \end{array}\right) {\xi }^{\left( \beta  - j\right) }\left( t\right) \left\lbrack  {{2r}{\left( -1\right) }^{j}j!{t}^{-\left( {j + 1}\right) } + {\left( -1\right) }^{j + 1}j!\left( {d + {2k} + 1}\right) {t}^{-\left( {j + 1}\right) }}\right\rbrack\]

\[= \mathop{\sum }\limits_{{j = 0}}^{\beta }\left( \begin{array}{l} \beta \\  j \end{array}\right) \left( {d + {2k} + 1 - {2r}}\right) \frac{{\left( -1\right) }^{\beta  - j}{\xi }^{\left( \beta  - j\right) }\left( t\right) }{{t}^{{2r} - \left( {d + {2k} + 1}\right)  - \beta  + j}}{\left( -1\right) }^{\beta  + 1}{t}^{{2r} - \left( {d + {2k} + 1}\right)  - \beta  - 1}.\]

Then (3.20) yields

\[\left( {d + {2k} + 1 - {2r}}\right) \mathop{\sum }\limits_{{j = 0}}^{\beta }\left( \begin{array}{l} \beta \\  j \end{array}\right) {\left( d + 2k + 1 - 2r\right) }^{\beta  - j} \leq  \mathop{\lim }\limits_{{t \rightarrow  \infty }}\frac{{\left( -1\right) }^{\beta  + 1}{\xi }^{\left( \beta  + 1\right) }\left( t\right) }{{t}^{{2r} - \left( {d + {2k} + 1}\right)  - \beta  - 1}}\]

\[\leq  \left( {d + {2k} + 1 - {2r}}\right) \mathop{\sum }\limits_{{j = 0}}^{\beta }\left( \begin{array}{l} \beta \\  j \end{array}\right) {\left( d + 2k + \beta  - j + 1 - 2r\right) }^{\beta  - j},\]

which proves

\[{\left( d + 2k + 1 - 2r\right) }^{\beta  + 1} \leq  \mathop{\lim }\limits_{{t \rightarrow  \infty }}\frac{{\left( -1\right) }^{\beta  + 1}{\xi }^{\left( \beta  + 1\right) }\left( t\right) }{{t}^{{2r} - \left( {d + {2k} + 1}\right)  - \beta  - 1}} \leq  {\left( d + 2k + \beta  + 2 - 2r\right) }^{\beta  + 1}. \tag{3.21}\]

On the other hand,

\[{\left( \frac{d}{dt}\right) }^{j}\left( \frac{{\xi }^{\prime }\left( t\right) }{\xi \left( t\right) }\right)  = {2r}{\left( -1\right) }^{j}j!{t}^{-\left( {j + 1}\right) } + {2}^{-j}{\psi }^{\left( j\right) }\left( \frac{t - k}{2}\right)  - {2}^{-j}{\psi }^{\left( j\right) }\left( \frac{t + d + k + 1}{2}\right)\]

\[= {2r}{\left( -1\right) }^{j}j!{t}^{-\left( {j + 1}\right) } + \frac{{\left( -1\right) }^{j + 1}}{{2}^{j}}j!\mathop{\sum }\limits_{{\nu  = 0}}^{\infty }\left( {{\left( \frac{t - k}{2} + \nu \right) }^{-\left( {j + 1}\right) } - {\left( \frac{t + d + k + 1}{2} + \nu \right) }^{-\left( {j + 1}\right) }}\right)\]

Some calculus estimation yields

\[{\left( -1\right) }^{j}{\left( \frac{d}{dt}\right) }^{j}\left( \frac{{\xi }^{\prime }\left( t\right) }{\xi \left( t\right) }\right)  > 0,\]

together with the induction hypothesis (3.20), it implies

\[{\left( -1\right) }^{\beta  + 1}{\xi }^{\left( \beta  + 1\right) }\left( t\right)  = \mathop{\sum }\limits_{{j = 0}}^{\beta }\left( \begin{array}{l} \beta \\  j \end{array}\right) {\left( -1\right) }^{\beta  - j}{\xi }^{\left( \beta  - j\right) }\left( t\right) {\left( -1\right) }^{j + 1}{\left( \frac{d}{dt}\right) }^{j}\left( \frac{{\xi }^{\prime }\left( t\right) }{\xi \left( t\right) }\right)  \geq  0.\]

This completes the induction and gives

\[0 \leq  {\left( -1\right) }^{\beta }{\xi }^{\left( \beta \right) }\left( t\right)  \lesssim  {t}^{{2r} - \left( {d + {2k} + 1}\right)  - \beta },\;\beta  = 0,1,\ldots\]

and for \( r < \frac{d + {2k} + 1}{2} \),

\[{\left( -1\right) }^{\beta }{\xi }^{\left( \beta \right) }\left( t\right)  \simeq  {t}^{{2r} - \left( {d + {2k} + 1}\right)  - \beta },\;\beta  = 0,1,\ldots\]

Consequently the formula

\[{\left( -1\right) }^{\beta }\left( {{\Delta }^{\beta }\xi }\right) \left( m\right)  = {\int }_{m}^{m + 1}\underset{\beta  - 1}{\underbrace{{\int }_{0}^{1}\ldots {\int }_{0}^{1}}}{\left( -1\right) }^{\beta }{\xi }^{\left( \beta \right) }\left( {{t}_{1} + \cdots  + {t}_{\beta }}\right) d{t}_{\beta }\ldots d{t}_{1}.\]

proves (3.16) and (3.17).

### 3.2 Properties of the polynomials with scattered points

In this subsection, we consider a finite subset \( {\left\{  {\theta }_{j}^{ * }\right\}  }_{j = 1}^{n} \subset  {\mathbb{S}}^{d} \) comprising distinct, scattered points. The mesh size for \( {\left\{  {\theta }_{j}^{ * }\right\}  }_{j = 1}^{n} \subset  {\mathbb{S}}^{d} \) is defined by

\[h = \mathop{\max }\limits_{{\eta  \in  {\mathbb{S}}^{d}}}\mathop{\min }\limits_{{1 \leq  j \leq  n}}\rho \left( {\eta,{\theta }_{j}^{ * }}\right). \tag{3.22}\]

The existence of positive quadrature rules on \( {\mathbb{S}}^{d} \) based on scattered points is a known result in approximation theory (see, e.g., [80, 81, 82]). For our case, one could construct a \( {\left\{  {\theta }_{j}^{ * }\right\}  }_{j = 1}^{n} \) - compatible decomposition as described in [80] based on the spherical shells \( {\mathbb{B}}_{\rho }\left( {{\theta }_{j}^{ * }, h}\right)  \smallsetminus  {\mathbb{B}}_{\rho }\left( {{\theta }_{j}^{ * }, h/2}\right) \). Then the following lemma follows straightforwardly by [80, Theorem 4.1].

Lemma 3.2 (Theorem 4.1. [80]). Given scattered points \( {\left\{  {\theta }_{j}^{ * }\right\}  }_{j = 1}^{n} \subset  {\mathbb{S}}^{d} \) with mesh norm \( h \), there exist nonnegative weights \( {\tau }_{1},\ldots,{\tau }_{n} \) with \( {\tau }_{j} \lesssim  {h}^{d} \) and a constant \( {C}_{1} \) (independent of \( n, h \) ) such that

\[{\oint }_{{\mathbb{S}}^{d}}p\left( \eta \right) q\left( \eta \right) {d\eta } = \mathop{\sum }\limits_{{j = 1}}^{n}{\tau }_{j}p\left( {\theta }_{j}^{ * }\right) q\left( {\theta }_{j}^{ * }\right),\;\forall p, q \in  {\mathbb{P}}_{J}\left( {\mathbb{S}}^{d}\right), \tag{3.23}\]

where \( J = \left\lfloor  {{C}_{1}{h}^{-1}}\right\rfloor \).

In this paper, the matrices induced by Legendre polynomials, defined as

\[P\left( m\right)  = {\left( {p}_{m}\left( {\theta }_{i}^{ * } \cdot  {\theta }_{j}^{ * }\right) \right) }_{i, j = 1}^{n}, \tag{3.24}\]

plays a crucial role in our analysis. In this subsection, we summarize the key properties of these matrices, which are essential for the proof of our main result and will be utilized in the next section.

For \( \beta  \in  \mathbb{N} \), we denote sequences of matrices \( {\left\{  {P}_{\beta }\left( m\right) \right\}  }_{m = 0}^{\infty } \) inductively by

\[{P}_{\beta  + 1}\left( m\right)  = \mathop{\sum }\limits_{{\nu  = 0}}^{m}{P}_{\beta }\left( \nu \right)  = \left( {{\nabla }^{-\beta }P}\right) \left( m\right)  = \mathop{\sum }\limits_{{\nu  = 0}}^{m}\left( \begin{matrix} m + \beta  - \nu \\  \beta  \end{matrix}\right) P\left( \nu \right),\; m \in  \mathbb{N}. \tag{3.25}\]

We also allow the summation to begin at indices other than 0. Given the sequence \( {\left\{  {P}_{\beta }\left( m\right) \right\}  }_{m = K}^{\infty } \), we denote the sequences \( {\left\{  {P}_{K,\beta  + 1}\left( m\right) \right\}  }_{m = K}^{\infty } \) by

\[{P}_{K,\beta  + 1}\left( m\right)  = \left( {{\nabla }^{-1}{P}_{\beta }}\right) \left( m\right)  = \mathop{\sum }\limits_{{\nu  = K}}^{m}{P}_{\beta }\left( \nu \right),\; m \geq  K. \tag{3.26}\]

An important property of the Legendre polynomials is that the Cesàro summations, \( {\left\{  {P}_{\beta }\left( m\right) \right\}  }_{m = 0}^{\infty } \), become increasingly "diagonally dominated" as \( \beta \) grows (see, e.g., [83]). Specifically, the ratio \( \frac{\left| {\left( {P}_{\beta }\left( m\right) \right) }_{i, i}\right| }{\mathop{\sum }\limits_{{i \neq  j}}\left| {\left( {P}_{\beta }\left( m\right) \right) }_{i, j}\right| } \) increases with \( \beta \). This behavior indicates that the Cesàro summations of \( {\left\{  {P}_{\beta }\left( m\right) \right\}  }_{m = 0}^{\infty } \) exhibit a highly localized property, which plays a crucial role in studying approximation properties (see, e.g., [74, Chapter 2.6, Chapter 11.4]). A similar phenomenon arises in Fourier analysis, where the Cesàro summations of the Dirichlet kernels, known as Fejér kernels, exhibit localization properties (see, e.g., [64, Chapter 7]). This highly-localized property is also significant in analyzing kernel approximation properties (see [84, 85]).

Lemma 3.4 establishes several key properties of \( {P}_{\beta }\left( m\right) \), enabling sharp estimations in the proof presented in Section 4. Before proving Lemma 3.4, we restate a related result from [83, Corollary 14.7] using our notation.

Lemma 3.3. For \( \beta  \geq  2 \) and all \( i \neq  j \),

\[\left| {\left( {P}_{\beta }\left( m\right) \right) }_{i, j}\right|  \lesssim  \frac{{m}^{\frac{d - 1}{2}}}{\rho {\left( {\theta }_{i}^{ * },{\theta }_{j}^{ * }\right) }^{\frac{d - 1}{2} + \beta }{\left( \max \left\{  \rho \left( {\theta }_{i}^{ * }, - {\theta }_{j}^{ * }\right),\frac{1}{m}\right\}  \right) }^{\frac{d - 1}{2}}}. \tag{3.27}\]

where the corresponding constant is only dependent of \( d \).

Proof. Let \( {\left\{  {p}_{n}^{\left( \frac{d - 2}{2},\frac{d - 2}{2}\right) }\right\}  }_{n = 0}^{\infty } \) be the Legendre polynomials defined in [83], then each \( {p}_{n}^{\left( \frac{d - 2}{2},\frac{d - 2}{2}\right) } \) must equal to \( {p}_{n} \) multiplied by a real number. The function \( {L}_{n}^{\left( {\frac{d - 2}{2},\frac{d - 2}{2}}\right),\beta } \) is denoted in [83, (2.27)] by,

for \( t \in  \left\lbrack  {-1,1}\right\rbrack \),

\[\left( \begin{matrix} n + \beta  - 1 \\  \beta  - 1 \end{matrix}\right) {L}_{n}^{\left( {\frac{d - 2}{2},\frac{d - 2}{2}}\right),\beta }\left( {t,1}\right)  = \mathop{\sum }\limits_{{\nu  = 0}}^{n}\left( \begin{matrix} n + \beta  - 1 - \nu \\  \beta  - 1 \end{matrix}\right) \frac{{p}_{\nu }^{\left( \frac{d - 2}{2},\frac{d - 2}{2}\right) }\left( t\right) {p}_{\nu }^{\left( \frac{d - 2}{2},\frac{d - 2}{2}\right) }\left( 1\right) }{{\begin{Vmatrix}{p}_{\nu }^{\left( \frac{d - 2}{2},\frac{d - 2}{2}\right) }\end{Vmatrix}}_{{L}_{{w}_{d}}^{2}\left( \left\lbrack  {-1,1}\right\rbrack  \right) }^{2}}\]

\[= \mathop{\sum }\limits_{{\nu  = 0}}^{n}\left( \begin{matrix} n + \beta  - 1 - \nu \\  \beta  - 1 \end{matrix}\right) \frac{{p}_{\nu }\left( t\right) {p}_{\nu }\left( 1\right) }{{\begin{Vmatrix}{p}_{\nu }\end{Vmatrix}}_{{L}_{{w}_{d}}^{2}\left( \left\lbrack  {-1,1}\right\rbrack  \right) }^{2}} \tag{3.28}\]

\[= \frac{{\omega }_{d - 1}}{{\omega }_{d}}\mathop{\sum }\limits_{{\nu  = 0}}^{n}\left( \begin{matrix} n + \beta  - 1 - \nu \\  \beta  - 1 \end{matrix}\right) {p}_{\nu }\left( t\right)  = \frac{{\omega }_{d - 1}}{{\omega }_{d}}{p}_{n,\beta }\left( t\right).\]

where the last equality follows by (3.7). By [83, Corollary 14.7],

\[\left| {{L}_{n}^{\left( \frac{d - 2}{2},\frac{d - 2}{2}\right) }\left( {t,1}\right) }\right|\]

\[\lesssim  {\left( n{\left( \sqrt{\frac{1 - t}{2}} + \frac{1}{n}\right) }^{d - 1}{\left( \sqrt{1 - t} + \frac{1}{n}\right) }^{2}\right) }^{-1} \tag{3.29}\]

\[+ {\left( {n}^{\beta  - 1}{\left( \sqrt{1 - t} + \frac{1}{n}\right) }^{\frac{d - 1}{2}}{\left( \sqrt{1 + t} + \frac{1}{n}\right) }^{\frac{d - 1}{2}}{\left( \frac{1}{n}\right) }^{\frac{d - 1}{2}}{\left( \sqrt{1 - t} + \frac{1}{n}\right) }^{\beta }\right) }^{-1}\]

By noticing the formula

\[\sqrt{1 - {\theta }_{i} \cdot  {\theta }_{j}} = \sqrt{1 - \cos \left( {\rho \left( {{\theta }_{i},{\theta }_{j}}\right) }\right) } = \sqrt{2}\sin \left( \frac{\rho \left( {{\theta }_{i},{\theta }_{j}}\right) }{2}\right),\;{\theta }_{i} \cdot  {\theta }_{j} \geq  0,\]

we have

\[\sqrt{1 - {\theta }_{i} \cdot  {\theta }_{j}} \simeq  \rho \left( {{\theta }_{i},{\theta }_{j}}\right).\]

As \( n \geq  M \gtrsim  \underline{h} \), we have

\[\rho \left( {{\theta }_{i},{\theta }_{j}}\right)  \geq  \underline{h} \gtrsim  \frac{1}{n},\]

then

\[\left| {{p}_{n,\beta }\left( {{\theta }_{i} \cdot  {\theta }_{j}}\right) }\right|  \lesssim  {n}^{\beta  - 1}\left( {{\left( n\rho {\left( {\theta }_{i},{\theta }_{j}\right) }^{d + 1}\right) }^{-1} + {\left( {n}^{\beta  - 1}\rho {\left( {\theta }_{i},{\theta }_{j}\right) }^{\frac{d - 1}{2}}\rho {\left( {\theta }_{i}, - {\theta }_{j}\right) }^{\frac{d - 1}{2}}{\left( \frac{1}{n}\right) }^{\frac{d - 1}{2}}\rho {\left( {\theta }_{i},{\theta }_{j}\right) }^{\beta }\right) }^{-1}}\right)\]

\[\lesssim  \frac{{n}^{\frac{d - 1}{2}}}{\rho {\left( {\theta }_{i},{\theta }_{j}\right) }^{\frac{d - 1}{2} + \beta }{\left( \max \left\{  \rho \left( {\theta }_{i}, - {\theta }_{j}\right),\frac{1}{n}\right\}  \right) }^{\frac{d - 1}{2}}}.\]

Lemma 3.4. Let \( r \in  \mathbb{N} \), the matrices \( \left\{  {{P}_{\beta }\left( m\right) }\right\} \) have the following properties:

(a) Each \( {P}_{\beta }\left( m\right) \) is semi-positive definite.

(b) The diagonal elements of \( {P}_{\beta }\left( m\right) \) are equal and satisfy

\[{\left( {P}_{\beta }\left( m\right) \right) }_{i, i} = \mathop{\max }\limits_{{1 \leq  i, j \leq  n}}\left| {\left( {P}_{\beta }\left( m\right) \right) }_{i, j}\right|  \simeq  {m}^{\beta  + d - 1}, \tag{3.30}\]

(c) Let \( \alpha  = \left\lceil  \frac{d + 2}{2}\right\rceil \) and \( J \) be the integer in Lemma 3.2

\[{\begin{Vmatrix}{P}_{\alpha }\left( m\right) \end{Vmatrix}}_{2} \lesssim  {m}^{\alpha  + d - 1}{\left( \frac{h}{\underline{h}}\right) }^{\frac{d - 1}{2} + \alpha },\; m \geq  J + 1, \tag{3.31}\]

where \( \underline{h} \mathrel{\text{:= }} \mathop{\min }\limits_{{i \neq  j}}\rho \left( {{\theta }_{i}^{ * },{\theta }_{j}^{ * }}\right) \).

Proof. To prove (a), it suffices to show each \( P\left( m\right) \) is semi-positive definite. For any vector \( a \in  {\mathbb{R}}^{n} \), by (3.6),

\[{a}^{\top }P\left( m\right) a = \mathop{\sum }\limits_{{1 \leq  i, j \leq  n}}{a}_{i}{a}_{j}p\left( {{\theta }_{i}^{ * } \cdot  {\theta }_{j}^{ * }}\right)  = \mathop{\sum }\limits_{{1 \leq  i, j \leq  n}}\mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}{a}_{i}{a}_{j}{Y}_{m,\ell }\left( {\theta }_{i}^{ * }\right) {Y}_{m,\ell }\left( {\theta }_{j}^{ * }\right)\]

\[= \mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}{\left( \mathop{\sum }\limits_{{j = 1}}^{n}{a}_{j}{Y}_{m,\ell }\left( {\theta }_{j}^{ * }\right) \right) }^{2} \geq  0. \tag{3.32}\]

This proves \( P\left( m\right) \) is semi-positive definite and consequently (a).

To prove (b), we apply (3.6), (3.7) and write

\[\left| {{p}_{m}\left( {\eta  \cdot  \theta }\right) }\right|  = \left| {\mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}{Y}_{m,\ell }\left( \eta \right) {Y}_{m,\ell }\left( \theta \right) }\right|\]

\[\leq  {\left( \mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}{Y}_{m,\ell }{\left( \eta \right) }^{2}\right) }^{1/2}{\left( \mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}{Y}_{m,\ell }{\left( \theta \right) }^{2}\right) }^{1/2}\]

\[= {p}_{m}\left( 1\right)  = N\left( m\right) \text{. } \tag{3.33}\]

Thus \( {\left( P\left( m\right) \right) }_{i, j} = {p}_{m}\left( {{\theta }_{i}^{ * } \cdot  {\theta }_{j}^{ * }}\right) \) attains its maximum value \( {p}_{m}\left( 1\right)  = N\left( m\right) \) when \( i = j \). Therefore,

\[\mathop{\max }\limits_{{1 \leq  i, j \leq  n}}\left| {\left( {P}_{\beta }\left( m\right) \right) }_{i, j}\right|  = {\left( {P}_{\beta }\left( m\right) \right) }_{i, i} = \left| {\mathop{\sum }\limits_{{\nu  = 0}}^{m}\left( \begin{matrix} m + \beta  - 1 - \nu \\  \beta  - 1 \end{matrix}\right) {\left( {P}_{\nu,0}\right) }_{i, i}}\right|\]

\[= \mathop{\sum }\limits_{{\nu  = 0}}^{m}\left( \begin{matrix} m + \beta  - 1 - \nu \\  \beta  - 1 \end{matrix}\right) N\left( \nu \right)  \simeq  {m}^{\beta  + d - 1}. \tag{3.34}\]

To prove (c), we apply Lemma 3.3. In our case, \( \alpha  = \left\lceil  \frac{d + 2}{2}\right\rceil   \geq  2 \), so Lemma 3.3 holds true. Given this lemma, we divide the set \( \left\{  {{\theta }_{i}^{ * }: 1 \leq  i \leq  n, i \neq  j}\right\} \) in terms of the distance to \( {\theta }_{j}^{ * } \) and \( - {\theta }_{j}^{ * } \) as

\[\left\{  {{\theta }_{i}: 1 \leq  i \leq  n, i \neq  j}\right\}   = {\mathcal{I}}_{-1, j} \cup  \mathop{\bigcup }\limits_{{p = 0}}^{\left\lceil  {\log }_{2}\left( \frac{\pi }{2h}\right) \right\rceil  }\left( {{\mathcal{I}}_{p, j, + } \cup  {\mathcal{I}}_{p, j, - }}\right),\]

where \( {\mathcal{I}}_{-1, j} \mathrel{\text{:= }} \left\{  {i: \rho \left( {{\theta }_{i}^{ * }, - {\theta }_{j}^{ * }}\right)  < \underline{h}}\right\} \) and for \( p = 0,1,\ldots \),

\[{\mathcal{I}}_{p, j, + } \mathrel{\text{:= }} \left\{  {i: {2}^{p}\underline{h} \leq  \rho \left( {{\theta }_{i}^{ * },{\theta }_{j}^{ * }}\right)  < {2}^{p + 1}\underline{h}}\right\} ,\;{\mathcal{I}}_{p, j, - } \mathrel{\text{:= }} \left\{  {i: {2}^{p}\underline{h} \leq  \rho \left( {{\theta }_{i}^{ * }, - {\theta }_{j}^{ * }}\right)  < {2}^{p + 1}\underline{h}}\right\} .\]

By a measure argument, it is easy to verify

\[\# {\mathcal{I}}_{-1, j} \lesssim  1,\;\# {\mathcal{I}}_{p, j, + } \lesssim  {2}^{pd},\;\# {\mathcal{I}}_{p, j, - } \lesssim  {2}^{pd}\]

where the corresponding constants are only dependent of \( d \).

By Lemma 3.3,

\[\left| {\left( {P}_{\alpha }\left( m\right) \right) }_{i, j}\right|  \lesssim  \frac{{m}^{\frac{d - 1}{2}}}{\rho {\left( {\theta }_{i}^{ * },{\theta }_{j}^{ * }\right) }^{\frac{d - 1}{2} + \alpha }{\left( \max \left\{  \rho \left( {\theta }_{i}^{ * }, - {\theta }_{j}^{ * }\right),\frac{1}{m}\right\}  \right) }^{\frac{d - 1}{2}}},\]

then we can write

\[\mathop{\sum }\limits_{{j \neq  i}}\left| {\left( {P}_{\alpha }\left( m\right) \right) }_{i, j}\right|  \lesssim  \mathop{\sum }\limits_{{p = 0}}^{\left\lceil  {\log }_{2}\left( \frac{\pi }{2h}\right) \right\rceil  }\mathop{\sum }\limits_{{i \in  {\mathcal{I}}_{p, j, + } \cup  {\mathcal{I}}_{p, j, - }}}\frac{{m}^{\frac{d - 1}{2}}}{\rho {\left( {\theta }_{i}^{ * },{\theta }_{j}^{ * }\right) }^{\frac{d - 1}{2} + \alpha }\rho {\left( {\theta }_{i}^{ * }, - {\theta }_{j}^{ * }\right) }^{\frac{d - 1}{2}}} + \mathop{\sum }\limits_{{i \in  {\mathcal{I}}_{-1, j}}}\frac{{m}^{d - 1}}{\rho {\left( {\theta }_{i}^{ * },{\theta }_{j}^{ * }\right) }^{\frac{d - 1}{2} + \alpha }}\]

\[\lesssim  \mathop{\sum }\limits_{{p = 0}}^{\left\lceil  {\log }_{2}\left( \frac{\pi }{2h}\right) \right\rceil  }\mathop{\sum }\limits_{{i \in  {\mathcal{I}}_{p, j, + } \cup  {\mathcal{I}}_{p, j, - }}}{\left( {2}^{p}\underline{h}\right) }^{-\left( {\frac{d - 1}{2} + \alpha }\right) }{m}^{\frac{d - 1}{2}} + \mathop{\sum }\limits_{{i \in  {\mathcal{I}}_{-1, j}}}{m}^{d - 1}\]

\[\lesssim  \mathop{\sum }\limits_{{p = 0}}^{\left\lceil  {\log }_{2}\left( \frac{\pi }{2h}\right) \right\rceil  }{2}^{pd}{\left( {2}^{p}\underline{h}\right) }^{-\left( {\frac{d - 1}{2} + \alpha }\right) }{m}^{\frac{d - 1}{2}} + {m}^{d - 1}\]

\[\lesssim  {m}^{\frac{d - 1}{2}}{\underline{h}}^{-\frac{d - 1}{2} + \alpha } + {m}^{d - 1}. \tag{3.35}\]

where the last inequality follows by recalling \( \alpha  = \left\lceil  \frac{d + 2}{2}\right\rceil \).

Together with (3.30), we get

\[\mathop{\sum }\limits_{{j = 1}}^{n}\left| {\left( {P}_{\alpha }\left( m\right) \right) }_{i, j}\right|  \lesssim  {m}^{d - 1} + {m}^{\frac{d - 1}{2}}{\underline{h}}^{-\frac{d - 1}{2} + \alpha } + {m}^{\alpha  + d - 1} \lesssim  {m}^{\alpha  + d - 1}{\left( \frac{h}{\underline{h}}\right) }^{\frac{d - 1}{2} + \alpha },\; m \geq  J + 1. \tag{3.36}\]

Thus we can estimate the matrix norm of \( {P}_{\alpha }\left( m\right) \) as

\[{\begin{Vmatrix}{P}_{\alpha }\left( m\right) \end{Vmatrix}}_{2} \leq  \sqrt{{\begin{Vmatrix}{P}_{\alpha }\left( m\right) \end{Vmatrix}}_{1}{\begin{Vmatrix}{P}_{\alpha }\left( m\right) \end{Vmatrix}}_{\infty }} \lesssim  {m}^{\alpha  + d - 1}{\left( \frac{h}{\underline{h}}\right) }^{\frac{d - 1}{2} + \alpha },\; m \geq  J + 1. \tag{3.37}\]

## 4 Approximation and characterization theorems for Sobolev spaces on spheres

In this section, we establish a spherical counterpart of Theorem 2.2 and 2.3. We begin by addressing the spherical case because the function \( {\sigma }_{k}\left( {\theta  \cdot  \eta }\right)  \in  {\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) \) admits a spherical harmonic expansion, whereas obtaining such an expansion for functions of the form \( {\sigma }_{k}\left( {w \cdot  x + b}\right) \) is challenging (as they are not periodic and prevents an application of Fourier series). This approach is partially inspired by [26] and [66].

### 4.1 Approximating functions on spheres by ReLU \( {}^{k} \) linear vector spaces

Theorem 4.1. Let \( d, n \in  \mathbb{N}, k \in  {\mathbb{N}}_{0}, r \in  \left( {0,\frac{d + {2k} + 1}{2}}\right\rbrack , s \leq  \min \{ k, r\} \), and \( {\left\{  {\theta }_{j}^{ * }\right\}  }_{j = 1}^{n} \subset  {\mathbb{S}}^{d} \). Then for any \( f \in  {\mathcal{H}}^{r}\left( {\mathbb{S}}^{d}\right) \) satisfying

\[f\left( \eta \right)  = {\left( -1\right) }^{k + 1}f\left( {-\eta }\right),\;\eta  \in  {\mathbb{S}}^{d}, \tag{4.1}\]

there exists \( a \in  {\mathbb{R}}^{n} \) with \( \parallel a{\parallel }_{2} \lesssim  {h}^{-\frac{{2k} + 1 - {2r}}{2}}\parallel f{\parallel }_{{\mathcal{H}}^{r}\left( {\mathbb{S}}^{d}\right) } \) such that

\[{\begin{Vmatrix}f - \mathop{\sum }\limits_{{j = 1}}^{n}{a}_{j}{\sigma }_{k}\left( {\theta }_{j}^{ * } \cdot   \circ  \right) \end{Vmatrix}}_{{\mathcal{H}}^{s}\left( {\mathbb{S}}^{d}\right) } \lesssim  {h}^{r - s}\parallel f{\parallel }_{{\mathcal{H}}^{r}\left( {\mathbb{S}}^{d}\right) }. \tag{4.2}\]

where

\[h = \mathop{\max }\limits_{{\eta  \in  {\mathbb{S}}^{d}}}\mathop{\min }\limits_{{1 \leq  j \leq  n}}\rho \left( {\eta,{\theta }_{j}^{ * }}\right).\]

If the collection \( {\left\{  {\theta }_{j}^{ * }\right\}  }_{j = 1}^{n} \subset  {\mathbb{S}}^{d} \) is well-distributed, for any \( f \in  {\mathcal{H}}^{r}\left( {\mathbb{S}}^{d}\right) \), there exists a \( \in  {\mathbb{R}}^{n} \) with \( \parallel a{\parallel }_{2} \lesssim  {n}^{\frac{{2k} + 1 - {2r}}{2d}}\parallel f{\parallel }_{{\mathcal{H}}^{r}\left( {\mathbb{S}}^{d}\right) } \) such that

\[{\begin{Vmatrix}f - \mathop{\sum }\limits_{{j = 1}}^{n}{a}_{j}{\sigma }_{k}\left( {\theta }_{j}^{ * } \cdot   \circ  \right) \end{Vmatrix}}_{{\mathcal{H}}^{s}\left( {\mathbb{S}}^{d}\right) } \lesssim  {h}^{r - s}\parallel f{\parallel }_{{\mathcal{H}}^{r}\left( {\mathbb{S}}^{d}\right) } \simeq  {n}^{-\frac{r - s}{d}}\parallel f{\parallel }_{{\mathcal{H}}^{r}\left( {\mathbb{S}}^{d}\right) }. \tag{4.3}\]

With \( r = \frac{d + {2k} + 1}{2} \) and \( s = 0 \), there exists \( a \in  {\mathbb{R}}^{n} \) with \( \parallel a{\parallel }_{2} \lesssim  {n}^{-\frac{1}{2}}\parallel f{\parallel }_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( {\mathbb{S}}^{d}\right) } \) such that

\[{\begin{Vmatrix}f - \mathop{\sum }\limits_{{j = 1}}^{n}{a}_{j}{\sigma }_{k}\left( {\theta }_{j}^{ * } \cdot   \circ  \right) \end{Vmatrix}}_{{\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) } \lesssim  {n}^{-\frac{1}{2} - \frac{{2k} + 1}{2d}}\parallel f{\parallel }_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( {\mathbb{S}}^{d}\right) }. \tag{4.4}\]

All the corresponding constants are independent of \( n,{\left\{  {\theta }_{j}^{ * }\right\}  }_{j = 1}^{n} \), and \( f \).

Proof. To start up, there exists a subset of \( {\left\{  {\vartheta }_{j}^{ * }\right\}  }_{j = 1}^{\widetilde{n}} \subset  {\left\{  {\theta }_{j}^{ * }\right\}  }_{j = 1}^{n} \) with \( \widetilde{n} \simeq  n \) such that

\[\mathop{\max }\limits_{{\eta  \in  {\mathbb{S}}^{d}}}\mathop{\min }\limits_{{1 \leq  j \leq  \widetilde{n}}}\rho \left( {\eta,{\vartheta }_{j}^{ * }}\right)  \lesssim  \mathop{\min }\limits_{{i \neq  j}}\rho \left( {{\vartheta }_{i}^{ * },{\vartheta }_{j}^{ * }}\right). \tag{4.5}\]

Thus it suffices to prove the theorem for \( {\left\{  {\theta }_{j}^{ * }\right\}  }_{j = 1}^{n} \) satisfying \( h \lesssim  \underline{h} \).

For each \( j = 1,\ldots, n \), we can write the expansion (3.8) as

\[{\sigma }_{k}\left( {{\theta }_{j}^{ * } \cdot  \eta }\right)  = \mathop{\sum }\limits_{{m \in  {E}_{{\sigma }_{k}}}}\widehat{{\sigma }_{k}}\left( m\right) {p}_{m}\left( {{\theta }_{j}^{ * } \cdot  \eta }\right)  = \mathop{\sum }\limits_{{m \in  {E}_{{\sigma }_{k}}}}\widehat{{\sigma }_{k}}\left( m\right) \mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}{Y}_{m,\ell }\left( {\theta }_{j}^{ * }\right) {Y}_{m,\ell }\left( \eta \right) \tag{4.6}\]

Then with \( a = {\left( {a}_{1},\ldots,{a}_{n}\right) }^{\top } \) which is to be determined later,

\[{f}_{n}\left( \eta \right)  = \mathop{\sum }\limits_{{j = 1}}^{n}{a}_{j}{\sigma }_{k}\left( {{\theta }_{j}^{ * } \cdot  \eta }\right)  = \mathop{\sum }\limits_{{m \in  {E}_{{\sigma }_{k}}}}\mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}\widehat{{f}_{n}}\left( {m,\ell }\right) {Y}_{m,\ell }\left( \eta \right),\;\text{ a.e. }\;\eta  \in  {\mathbb{S}}^{d}. \tag{4.7}\]

where

\[\widehat{{f}_{n}}\left( {m,\ell }\right)  = \widehat{{\sigma }_{k}}\left( m\right) \mathop{\sum }\limits_{{j = 1}}^{n}{a}_{j}{Y}_{m,\ell }\left( {\theta }_{j}^{ * }\right). \tag{4.8}\]

On the other hand, since the Legendre polynomials satisfy

\[{p}_{m}\left( {-t}\right)  = {\left( -1\right) }^{m}{p}_{m}\left( t\right),\; t \in  \left\lbrack  {-1,1}\right\rbrack , m \in  \mathbb{N}, \tag{4.9}\]

if \( m - k \) is even,

\[\left( {{\Pi }_{m}f}\right) \left( \theta \right)  = {\int }_{{\mathbb{S}}^{d}}f\left( {-\eta }\right) {p}_{m}\left( {-\eta  \cdot  \theta }\right) {d\eta } = {\int }_{{\mathbb{S}}^{d}}{\left( -1\right) }^{k + 1}f\left( \eta \right) {\left( -1\right) }^{m}{p}_{m}\left( {\eta  \cdot  \theta }\right) {d\eta }\]

\[=  - {\int }_{{\mathbb{S}}^{d}}f\left( \eta \right) {p}_{m}\left( {\eta  \cdot  \theta }\right) {d\eta } =  - \left( {{\Pi }_{m}f}\right) \left( \theta \right),\;\theta  \in  {\mathbb{S}}^{d},\]

which implies \( {\Pi }_{m}f \equiv  0 \) and \( f \in  {\bigoplus }_{m \in  {E}_{{\sigma }_{k}}}{\mathbb{Y}}_{m} \).

So we can write the expansion

\[f\left( \eta \right)  = \mathop{\sum }\limits_{{m \in  {E}_{{\sigma }_{k}}}}\mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}\widehat{f}\left( {m,\ell }\right) {Y}_{m,\ell }\left( \eta \right),\;\text{ a.e. }\;\eta  \in  {\mathbb{S}}^{d}.\]

By Lemma 3.2, there exists nonnegative numbers

\[{\tau }_{1},\ldots,{\tau }_{n} \lesssim  {h}^{d} \tag{4.10}\]

such that

\[\widehat{f}\left( {m,\ell }\right)  = {\int }_{{\mathbb{S}}^{d}}f\left( \eta \right) {Y}_{m,\ell }\left( \eta \right) {d\eta } = {\int }_{{\mathbb{S}}^{d}}\left( {{\Pi }_{m}f}\right) \left( \eta \right) {Y}_{m,\ell }\left( \eta \right) {d\eta }\]

\[= {\int }_{{\mathbb{S}}^{d}}\left( {\mathop{\sum }\limits_{{{m}^{\prime } \in  {E}_{{\sigma }_{k}}, J}}\widehat{{\sigma }_{k}}{\left( {m}^{\prime }\right) }^{-1}\left( {{\Pi }_{{m}^{\prime }}f}\right) \left( \eta \right) }\right) \widehat{{\sigma }_{k}}\left( m\right) {Y}_{m,\ell }\left( \eta \right) {d\eta } \tag{4.11}\]

\[= \mathop{\sum }\limits_{{j = 1}}^{n}{\tau }_{j}\left( {\mathop{\sum }\limits_{{{m}^{\prime } \in  {E}_{{\sigma }_{k}, J}}}\widehat{{\sigma }_{k}}{\left( {m}^{\prime }\right) }^{-1}\left( {{\Pi }_{{m}^{\prime }}f}\right) \left( {\theta }_{j}^{ * }\right) }\right) \widehat{{\sigma }_{k}}\left( m\right) {Y}_{m,\ell }\left( {\theta }_{j}^{ * }\right),\; m \leq  J.\]

where \( J \) is the integer in Lemma 3.2, and

\[{E}_{{\sigma }_{k}, J} \mathrel{\text{:= }} \left\{  {m \in  {E}_{{\sigma }_{k}}: m \leq  J}\right\} .\]

Comparing (4.8) and (4.11), by taking

\[{a}_{j} = {\tau }_{j}\left( {\mathop{\sum }\limits_{{m \in  {E}_{{\sigma }_{k}, J}}}\widehat{{\sigma }_{k}}{\left( m\right) }^{-1}\left( {{\Pi }_{m}f}\right) \left( {\theta }_{j}^{ * }\right) }\right),\; j = 1,\ldots, n, \tag{4.12}\]

we have

\[\widehat{{f}_{n}}\left( {m,\ell }\right)  = \widehat{f}\left( {m,\ell }\right),\; m \leq  J. \tag{4.13}\]

\[{\begin{Vmatrix}f - {f}_{n}\end{Vmatrix}}_{{\mathcal{H}}^{s}\left( {\mathbb{S}}^{d}\right) }^{2} = \mathop{\sum }\limits_{{m = 0}}^{\infty }\mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}{\left( \widehat{f}\left( m,\ell \right)  - \widehat{{f}_{n}}\left( m,\ell \right) \right) }^{2}\left( {{m}^{2s} + 1}\right)\]

\[= \mathop{\sum }\limits_{{m = J + 1}}^{\infty }\mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}{\left( \widehat{f}\left( m,\ell \right)  - \widehat{{f}_{n}}\left( m,\ell \right) \right) }^{2}\left( {{m}^{2s} + 1}\right) \tag{4.14}\]

\[\leq  4\mathop{\sum }\limits_{{m = J + 1}}^{\infty }\mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}\widehat{f}{\left( m,\ell \right) }^{2}{m}^{2s} + 4\mathop{\sum }\limits_{{m = J + 1}}^{\infty }\mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}\widehat{{f}_{n}}{\left( m,\ell \right) }^{2}{m}^{2s}.\]

Write

\[{I}_{1} = \mathop{\sum }\limits_{{m = J + 1}}^{\infty }\mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}\widehat{f}{\left( m,\ell \right) }^{2}{m}^{2s},\;{I}_{2} = \mathop{\sum }\limits_{{m = J + 1}}^{\infty }\mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}\widehat{{f}_{n}}{\left( m,\ell \right) }^{2}{m}^{2s},\]

then

\[{I}_{1} \leq  \mathop{\sum }\limits_{{m = J + 1}}^{\infty }\mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}\widehat{f}{\left( m,\ell \right) }^{2}{m}^{2r}{J}^{{2s} - {2r}} \leq  {J}^{{2s} - {2r}}\parallel f{\parallel }_{{\mathcal{H}}^{r}\left( {\mathbb{S}}^{d}\right) }^{2} \simeq  {h}^{{2r} - {2s}}\parallel f{\parallel }_{{\mathcal{H}}^{r}\left( {\mathbb{S}}^{d}\right) }^{2}. \tag{4.15}\]

Let \( P\left( m\right) \) be as in (3.32),

\[{I}_{2} = \mathop{\sum }\limits_{{m = J + 1}}^{\infty }\mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}\widehat{{\sigma }_{k}}{\left( m\right) }^{2}{\left( \mathop{\sum }\limits_{{j = 1}}^{n}{a}_{j}{Y}_{m,\ell }\left( {\theta }_{j}^{ * }\right) \right) }^{2}{m}^{2s} = \mathop{\sum }\limits_{{m = J + 1}}^{\infty }\widehat{{\sigma }_{k}}{\left( m\right) }^{2}{m}^{2s}{a}^{\top }P\left( m\right) a \tag{4.16}\]

\[\leq  \mathop{\sum }\limits_{{m = J + 1}}^{\infty }\xi \left( m\right) {a}^{\top }P\left( m\right) a,\]

where \( \xi \) is the nonnegative function determined in Lemma 3.1, which satisfies

\[\xi \left( m\right)  = \widehat{{\sigma }_{k}}{\left( m\right) }^{2}{m}^{2s},\; m \in  {E}_{{\sigma }_{k}}, m \geq  J + 1.\]

Summation by parts, we have

\[\mathop{\sum }\limits_{{m = J + 1}}^{\infty }\xi \left( m\right) P\left( m\right)  = \mathop{\sum }\limits_{{m = J + 1}}^{\infty }\xi \left( m\right) \left( {\nabla {P}_{J + 1,1}}\right) \left( m\right)\]

\[= \mathop{\lim }\limits_{{m \rightarrow  \infty }}\xi \left( {m + 1}\right) {P}_{J + 1,1}\left( m\right)  - \mathop{\sum }\limits_{{m = J + 1}}^{\infty }\left( {\nabla \xi }\right) \left( {m + 1}\right) {P}_{J + 1,1}\left( m\right)\]

\[= \mathop{\lim }\limits_{{m \rightarrow  \infty }}\xi \left( {m + 1}\right) {P}_{J + 1,1}\left( m\right)  - \mathop{\sum }\limits_{{m = J + 1}}^{\infty }\left( {\Delta \xi }\right) \left( m\right) {P}_{J + 1,1}\left( m\right).\]

where we abused the notation that \( {P}_{J + 1,1}\left( J\right)  = \mathop{\sum }\limits_{{\nu  = J + 1}}^{J}{P}_{\nu,0} = 0 \). Similarly, for \( \beta  = 1,\ldots,\alpha  - 1 \),

\[\mathop{\sum }\limits_{{m = J + 1}}^{\infty }{\left( -1\right) }^{\beta }\left( {{\Delta }^{\beta }\xi }\right) \left( m\right) {P}_{\beta }\left( m\right)  = \mathop{\lim }\limits_{{m \rightarrow  \infty }}{\left( -1\right) }^{\beta }\left( {{\Delta }^{\beta }\xi }\right) \left( {m + 1}\right) {P}_{J + 1,\beta  + 1}\left( m\right)\]

\[+ \mathop{\sum }\limits_{{m = J + 1}}^{\infty }{\left( -1\right) }^{\beta  + 1}\left( {{\Delta }^{\beta  + 1}\xi }\right) \left( m\right) {P}_{J + 1,\beta  + 1}\left( m\right).\]

By Lemma 3.1,

\[\mathop{\sum }\limits_{{m = J + 1}}^{\infty }{\left( -1\right) }^{\beta }\left( {{\Delta }^{\beta }\xi }\right) \left( m\right) {P}_{\beta }\left( m\right)  = \mathop{\sum }\limits_{{m = J + 1}}^{\infty }{\left( -1\right) }^{\beta  + 1}\left( {{\Delta }^{\beta  + 1}\xi }\right) \left( m\right) {P}_{J + 1,\beta  + 1}\left( m\right)\]

\[\leq  \mathop{\sum }\limits_{{m = J + 1}}^{\infty }{\left( -1\right) }^{\beta  + 1}\left( {{\Delta }^{\beta  + 1}\xi }\right) \left( m\right) {P}_{\beta  + 1}\left( m\right).\]

Therefore,

\[{I}_{2} \leq  \mathop{\sum }\limits_{{m = J + 1}}^{\infty }\xi \left( m\right) {a}^{\top }P\left( m\right) a \leq  \mathop{\sum }\limits_{{m = J + 1}}^{\infty }{\left( -1\right) }^{\alpha }\left( {{\Delta }^{\alpha }\xi }\right) \left( m\right) {a}^{\top }{P}_{\alpha }\left( m\right) a. \tag{4.17}\]

Together with (3.16) and (3.31),

\[{I}_{2} \lesssim  {\left( \frac{h}{\underline{h}}\right) }^{\frac{d - 1}{2} + \alpha }\parallel a{\parallel }_{2}^{2}\mathop{\sum }\limits_{{m = J + 1}}^{\infty }{m}^{{2s} - {2k} - 2} \simeq  {\left( \frac{h}{\underline{h}}\right) }^{\frac{d - 1}{2} + \alpha }\parallel a{\parallel }_{2}^{2}{h}^{{2k} + 1 - {2s}}, \tag{4.18}\]

Thus

\[{\begin{Vmatrix}f - {f}_{n}\end{Vmatrix}}_{{\mathcal{H}}^{s}\left( {\mathbb{S}}^{d}\right) }^{2} \lesssim  {h}^{{2r} - {2s}}\parallel f{\parallel }_{{\mathcal{H}}^{r}\left( {\mathbb{S}}^{d}\right) }^{2} + {\left( \frac{h}{\underline{h}}\right) }^{\frac{d - 1}{2} + \alpha }\parallel a{\parallel }_{2}^{2}{h}^{{2k} + 1 - {2s}}. \tag{4.19}\]

For the norm \( \parallel a{\parallel }_{2} \), we apply (4.10) and the quadrature formula (3.23) to conclude

\[\parallel a{\parallel }_{2}^{2} = \mathop{\sum }\limits_{{j = 1}}^{n}{\tau }_{j}^{2}{\left( \mathop{\sum }\limits_{{m \in  {E}_{{\sigma }_{k}, J}}}\widehat{{\sigma }_{k}}{\left( m\right) }^{-1}\left( {\Pi }_{m}f\right) \left( {\theta }_{j}^{ * }\right) \right) }^{2} \lesssim  {h}^{d}\mathop{\sum }\limits_{{j = 1}}^{n}{\tau }_{j}{\left( \mathop{\sum }\limits_{{m \in  {E}_{{\sigma }_{k}, J}}}\widehat{{\sigma }_{k}}{\left( m\right) }^{-1}\left( {\Pi }_{m}f\right) \left( {\theta }_{j}^{ * }\right) \right) }^{2}\]

\[= {h}^{d}{\oint }_{{\mathbb{S}}^{d}}{\left( \mathop{\sum }\limits_{{m \in  {E}_{{\sigma }_{k}, J}}}\widehat{{\sigma }_{k}}{\left( m\right) }^{-1}\mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}\widehat{f}\left( m,\ell \right) {Y}_{m,\ell }\left( \eta \right) \right) }^{2}{d\eta } = {h}^{d}\mathop{\sum }\limits_{{m \in  {E}_{{\sigma }_{k}, J}}}\widehat{{\sigma }_{k}}{\left( m\right) }^{-2}\mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}\widehat{f}{\left( m,\ell \right) }^{2}\]

\[\leq  {h}^{d}\mathop{\max }\limits_{{m \in  {E}_{{\sigma }_{k}, J}}}\left( \frac{{m}^{d + {2k} + 1}}{{m}^{2r} + 1}\right) \mathop{\sum }\limits_{{m \in  {E}_{{\sigma }_{k}, J}}}\mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}\left( {{m}^{2r} + 1}\right) \widehat{f}{\left( m,\ell \right) }^{2}\]

\[\lesssim  {h}^{{2r} - {2k} - 1}\parallel f{\parallel }_{{\mathcal{H}}^{r}\left( {\mathbb{S}}^{d}\right) }^{2}. \tag{4.20}\]

Substituting this into (4.18),

\[{\left( \frac{h}{\underline{h}}\right) }^{\frac{d - 1}{2} + \alpha }\parallel a{\parallel }_{2}^{2}{h}^{{2k} + 1 - {2s}} \lesssim  {\left( \frac{h}{\underline{h}}\right) }^{\frac{d - 1}{2} + \alpha }{h}^{2\left( {r - s}\right) }\parallel f{\parallel }_{{\mathcal{H}}^{r}\left( {\mathbb{S}}^{d}\right) }^{2} \lesssim  {h}^{2\left( {r - s}\right) }\parallel f{\parallel }_{{\mathcal{H}}^{r}\left( {\mathbb{S}}^{d}\right) }^{2},\]

where the second inequality follows by (4.5). Together with (4.19), we get (4.2) and complete the proof.

Remark 4.1. We remark here that the condition (4.1) is necessary. This is because any \( {\sigma }_{k}\left( {{\theta }^{ * } \cdot  \eta }\right) \) is essentially an even (odd) function when \( k \) is odd (even) up to a polynomial. It suffices to notice

\[{\sigma }_{k}\left( {{\theta }^{ * } \cdot  \eta }\right)  = {\left( -1\right) }^{k + 1}{\sigma }_{k}\left( {-{\theta }^{ * } \cdot  \eta }\right)  + {\left( {\theta }^{ * } \cdot  \eta \right) }^{k}.\]

### 4.2 Integral representation for Sobolev and Barron functions on spheres

In this subsection, we provide a new characterization of \( {\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( {\mathbb{S}}^{d}\right) \). While the operator

\[\mathcal{G}: \mu  \mapsto  {\int }_{{\mathbb{S}}^{d}}{\sigma }_{k}\left( {\theta  \cdot  \eta }\right) {d\mu }\left( \theta \right),\;\mu  \in  \mathcal{M}\left( {\mathbb{S}}^{d}\right) \tag{4.21}\]

characterize the Barron space \( {\mathcal{B}}^{k}\left( {\mathbb{S}}^{d}\right) \) given by the norm (in comparison with 2.5)

\[\parallel f{\parallel }_{{\mathcal{B}}^{k}\left( {\mathbb{S}}^{d}\right) } = \mathop{\inf }\limits_{{\mu  \in  \mathcal{M}\left( {\mathbb{S}}^{d}\right) }}\left\{  {\left| \mu \right| \left( {\mathbb{S}}^{d}\right) : f = \mathcal{G}\left( \mu \right) }\right\} \tag{4.22}\]

From Remark 4.1, up to a polynomial, a neuron \( {\sigma }_{k}\left( {\theta  \cdot  \eta }\right) \) is essentially an even/odd function on the sphere if \( k \) is odd/even. So for all the spaces on the sphere, it suffices to consider their even/odd subspace.

For notation simplicity, in this section, we consider \( k \) to be fixed. We write

\[{\mathcal{M}}_{ * }\left( {\mathbb{S}}^{d}\right)  \mathrel{\text{:= }} \left\{  {\mu  \in  \mathcal{M}\left( {\mathbb{S}}^{d}\right) : \mu \left( A\right)  = {\left( -1\right) }^{k + 1}\mu \left( {-A}\right),\; A \subset  {\mathbb{S}}^{d}}\right\} ,\]

and for any function space \( \mathcal{S}\left( {\mathbb{S}}^{d}\right) \),

\[{\mathcal{S}}_{ * }\left( {\mathbb{S}}^{d}\right)  \mathrel{\text{:= }} \left\{  {f \in  \mathcal{S}\left( {\mathbb{S}}^{d}\right) : f\left( \eta \right)  = {\left( -1\right) }^{k + 1}f\left( {-\eta }\right),\;\eta  \in  {\mathbb{S}}^{d}}\right\} .\]

Namely, \( {\mathcal{M}}_{ * }\left( {\mathbb{S}}^{d}\right) \) (or \( {\mathcal{S}}_{ * }\left( {\mathbb{S}}^{d}\right) \) ) consists of even measures (or functions) if \( k \) is odd and odd (even) measures (or functions) if \( k \) is even.

Theorem 4.2. For \( d \in  \mathbb{N}, k \in  {\mathbb{N}}_{0} \), the map \( \mathcal{G}: {\mathcal{M}}_{ * }\left( {\mathbb{S}}^{d}\right)  \rightarrow  {\mathcal{B}}_{ * }^{k}\left( {\mathbb{S}}^{d}\right) \) is an isomorphism, and satisfies

\[\left| \mu \right| \left( {\mathbb{S}}^{d}\right)  = \parallel \mathcal{G}\left( \mu \right) {\parallel }_{{\mathcal{B}}^{k}\left( {\mathbb{S}}^{d}\right) }. \tag{4.23}\]

Furthermore, \( \mathcal{G}: {\mathcal{L}}_{ * }^{2}\left( {\mathbb{S}}^{d}\right)  \rightarrow  {\mathcal{H}}_{ * }^{\frac{d + {2k} + 1}{2}}\left( {\mathbb{S}}^{d}\right) \) is an isomorphism on subspaces with

\[\parallel \psi {\parallel }_{{\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) } \simeq  \parallel \mathcal{G}\left( \psi \right) {\parallel }_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( {\mathbb{S}}^{d}\right) }. \tag{4.24}\]

Proof. Consider the space \( {\mathcal{M}}_{ * }\left( {\mathbb{S}}^{d}\right) \). By the earlier work (see, e.g., [36, Lemma 3]),

\[\mathcal{G}: {\mathcal{M}}_{ * }\left( {\mathbb{S}}^{d}\right)  \rightarrow  {\mathcal{B}}_{ * }^{k}\left( {\mathbb{S}}^{d}\right)\]

is surjective. To see it is an injection, let \( {\mu }_{1},{\mu }_{2} \in  {\mathcal{M}}_{ * }\left( {\mathbb{S}}^{d}\right) \) be different. For the harmonic basis, we denote

\[{\widehat{\mu }}_{i}\left( {m,\ell }\right)  = {\int }_{{\mathbb{S}}^{d}}{Y}_{m,\ell }\left( \theta \right) d{\mu }_{i}\left( \theta \right),\; i = 1,2,\]

then there exists some \( \left( {{m}^{\prime },{\ell }^{\prime }}\right) \) such that

\[\widehat{{\mu }_{1}}\left( {{m}^{\prime },{\ell }^{\prime }}\right)  \neq  \widehat{{\mu }_{2}}\left( {{m}^{\prime },{\ell }^{\prime }}\right), \tag{4.25}\]

On the other hand, by Fubini’s theorem, the \( {\mathcal{L}}^{2} \) -projections satisfy

\[{\Pi }_{n}\left( {\mathcal{G}\left( {\mu }_{i}\right) }\right) \left( \eta \right)  = {\int }_{{\mathbb{S}}^{d}}{\Pi }_{n}\left( {{\sigma }_{k}\left( {\theta  \cdot  \eta }\right) }\right) d{\mu }_{i}\left( \theta \right)  = \mathop{\sum }\limits_{{m = 0}}^{n}\widehat{{\sigma }_{k}}\left( m\right) \mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}\widehat{{\mu }_{i}}\left( {m,\ell }\right) {Y}_{m,\ell }\left( \eta \right),\; i = 1,2.\]

Therefore,

\[{\begin{Vmatrix}\mathcal{G}\left( {\mu }_{1}\right)  - \mathcal{G}\left( {\mu }_{2}\right) \end{Vmatrix}}_{{\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) }^{2} = \mathop{\lim }\limits_{{n \rightarrow  \infty }}{\begin{Vmatrix}{\Pi }_{n}\left( \mathcal{G}\left( {\mu }_{1}\right) \right)  - {\Pi }_{n}\left( \mathcal{G}\left( {\mu }_{2}\right) \right) \end{Vmatrix}}_{{\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) }^{2}\]

\[\geq  \widehat{{\sigma }_{k}}{\left( {m}^{\prime }\right) }^{2}{\left( \widehat{{\mu }_{1}}\left( {m}^{\prime },{\ell }^{\prime }\right)  - \widehat{{\mu }_{2}}\left( {m}^{\prime },{\ell }^{\prime }\right) \right) }^{2} > 0\]

As \( {\mathcal{B}}^{k}\left( {\mathbb{S}}^{d}\right) \) is a subspace of \( {\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) \),

\[{\begin{Vmatrix}\mathcal{G}\left( {\mu }_{1}\right)  - \mathcal{G}\left( {\mu }_{2}\right) \end{Vmatrix}}_{{\mathcal{B}}^{k}\left( {\mathbb{S}}^{d}\right) } \gtrsim  {\begin{Vmatrix}\mathcal{G}\left( {\mu }_{1}\right)  - \mathcal{G}\left( {\mu }_{2}\right) \end{Vmatrix}}_{{\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) } > 0. \tag{4.26}\]

Hence \( \mathcal{G}: {\mathcal{M}}_{ * }\left( {\mathbb{S}}^{d}\right)  \rightarrow  {\mathcal{B}}_{ * }^{k}\left( {\mathbb{S}}^{d}\right) \) is an isomorphism. Then the definition of \( \parallel  \cdot  {\parallel }_{{\mathcal{B}}^{k}\left( {\mathbb{S}}^{d}\right) } \) gives

\[\parallel f{\parallel }_{{\mathcal{B}}^{k}\left( {\mathbb{S}}^{d}\right) } = \mathop{\inf }\limits_{{\mu  \in  \mathcal{M}\left( {\mathbb{S}}^{d}\right) }}\left\{  {\left| \mu \right| \left( {\mathbb{S}}^{d}\right) : f\left( x\right)  = {\int }_{{\mathbb{S}}^{d}}{\sigma }_{k}\left( {\theta  \cdot  \eta }\right) {d\mu }\left( \theta \right) }\right\}   = \left| {{\mathcal{G}}^{-1}\left( \mu \right) }\right| \left( {\mathbb{S}}^{d}\right),\; f \in  {\mathcal{B}}_{ * }^{k}\left( {\mathbb{S}}^{d}\right), \tag{4.27}\]

which is exactly (4.23).

Now we prove the result on \( {\mathcal{L}}_{ * }^{2}\left( {\mathbb{S}}^{d}\right) \). It suffices to apply the orthogonality and write

\[\mathcal{G}\left( \psi \right) \left( \eta \right)  = {\int }_{{\mathbb{S}}^{d}}\left( {\mathop{\sum }\limits_{{m \equiv  k + 1{\;\operatorname{mod}\;2}}}\mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}\widehat{\psi }\left( {m,\ell }\right) {Y}_{m,\ell }\left( \theta \right) }\right) \left( {\mathop{\sum }\limits_{{{m}^{\prime } \equiv  k + 1{\;\operatorname{mod}\;2}}}\widehat{{\sigma }_{k}\left( {m}^{\prime }\right) }\mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( {m}^{\prime }\right) }}{Y}_{{m}^{\prime },\ell }\left( \theta \right) {Y}_{{m}^{\prime },\ell }\left( \eta \right) }\right) {d\theta }\]

\[= \mathop{\sum }\limits_{{m \equiv  k + 1{\;\operatorname{mod}\;2}}}\widehat{{\sigma }_{k}}\left( m\right) \mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}\widehat{\psi }\left( {m,\ell }\right) {Y}_{m,\ell }\left( \eta \right).\]

By Lemma 3.1,

\[\left( {{m}^{d + {2k} + 1} + 1}\right) \widehat{{\sigma }_{k}}{\left( m\right) }^{2} \simeq  1,\; m \equiv  k + 1{\;\operatorname{mod}\;2}\]

then the norm \( \parallel \mathcal{G}\left( \psi \right) {\parallel }_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( {\mathbb{S}}^{d}\right) } \) can be estimated by (3.1) as

\[\parallel \mathcal{G}\left( \psi \right) {\parallel }_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( {\mathbb{S}}^{d}\right) }^{2} = \mathop{\sum }\limits_{{m \equiv  k + 1{\;\operatorname{mod}\;2}}}\left( {{m}^{d + {2k} + 1} + 1}\right) \widehat{{\sigma }_{k}}{\left( m\right) }^{2}\mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}\widehat{\psi }{\left( m,\ell \right) }^{2}\]

\[\simeq  \mathop{\sum }\limits_{{m \equiv  k + 1{\;\operatorname{mod}\;2}}}\mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}\widehat{\psi }{\left( m,\ell \right) }^{2} = \parallel \psi {\parallel }_{{\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) }^{2}.\]

Conversely, denote \( {\mathcal{G}}^{-1} \) by

\[{\mathcal{G}}^{-1}\left( \psi \right)  = \mathop{\sum }\limits_{{m \equiv  k + 1{\;\operatorname{mod}\;2}}}\widehat{{\sigma }_{k}}{\left( m\right) }^{-1}\mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}\widehat{\psi }\left( {m,\ell }\right) {Y}_{m,\ell }\left( \eta \right),\;\psi  \in  {\mathcal{H}}_{ * }^{\frac{d + {2k} + 1}{2}}\left( {\mathbb{S}}^{d}\right),\]

then

\[{\begin{Vmatrix}{\mathcal{G}}^{-1}\left( \psi \right) \end{Vmatrix}}_{{\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) }^{2} = \mathop{\sum }\limits_{{m \equiv  k + 1{\;\operatorname{mod}\;2}}}\widehat{{\sigma }_{k}}{\left( m\right) }^{-2}\mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}\widehat{\psi }{\left( m,\ell \right) }^{2}\]

\[\simeq  \mathop{\sum }\limits_{{m \equiv  k + 1{\;\operatorname{mod}\;2}}}\left( {{m}^{d + {2k} + 1} + 1}\right) \mathop{\sum }\limits_{{\ell  = 1}}^{{N\left( m\right) }}\widehat{\psi }{\left( m,\ell \right) }^{2} = \parallel \psi {\parallel }_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( {\mathbb{S}}^{d}\right) }^{2}.\]

Finally, it is straightforward to check \( {\mathcal{G}}^{-1} \) is the inverse of \( \mathcal{G} \).

With this theorem, the difference of the Barron spaces and Sobolev spaces can be further characterized as

\[{\mathcal{B}}_{ * }^{k}\left( {\mathbb{S}}^{d}\right) /{\mathcal{H}}_{ * }^{\frac{d + {2k} + 1}{2}}\left( {\mathbb{S}}^{d}\right)  \cong  {\mathcal{M}}_{ * }\left( {\mathbb{S}}^{d}\right) /{\mathcal{L}}_{ * }^{2}\left( {\mathbb{S}}^{d}\right) \tag{4.28}\]

\[\cong  {\mathcal{M}}_{ * }^{ \bot  }\left( {\mathbb{S}}^{d}\right)  \oplus  \left( {{\mathcal{L}}_{ * }^{1}\left( {\mathbb{S}}^{d}\right) /{\mathcal{L}}_{ * }^{2}\left( {\mathbb{S}}^{d}\right) }\right),\]

where \( {\mathcal{M}}_{ * }^{ \bot  }\left( {\mathbb{S}}^{d}\right) \) is the space of discrete measures, and \( \left( {{\mathcal{L}}_{ * }^{1}\left( {\mathbb{S}}^{d}\right) /{\mathcal{L}}_{ * }^{2}\left( {\mathbb{S}}^{d}\right) }\right) \) can be characterized by the norm

\[\parallel \left\lbrack  f\right\rbrack  {\parallel }_{\left( {\mathcal{L}}_{ * }^{1}\left( {\mathbb{S}}^{d}\right) /{\mathcal{L}}_{ * }^{2}\left( {\mathbb{S}}^{d}\right) \right) } = \mathop{\inf }\limits_{{g \in  {\mathcal{L}}_{ * }^{2}\left( {\mathbb{S}}^{d}\right) }}\parallel f - g{\parallel }_{{\mathcal{L}}^{1}\left( {\mathbb{S}}^{d}\right) }\]

for any coset \( \left\lbrack  f\right\rbrack \) represented by some \( f \in  {\mathcal{L}}_{ * }^{1}\left( {\mathbb{S}}^{d}\right) \).

## 5 Proof of main results on general domains

Building on Theorems 4.1–4.2, we now establish their analogues on the ball. The proof relies on an operator, originally introduced in [26], that maps \( {\operatorname{ReLU}}^{k} \) functions defined on \( {\mathbb{S}}^{d} \) to \( {\operatorname{ReLU}}^{k} \) functions on \( {\mathbb{B}}^{d} \). We note that the existence of such an operator relies on the homogeneity of the activation function.

While there exists a more natural homeomorphism from a cap of \( {\mathbb{S}}^{d} \) to the ball \( {\mathbb{B}}^{d} \), the operator defined below has the distinct advantage of preserving the structure of \( {\mathrm{{ReLU}}}^{k} \) functions. This property is crucial for extending the spherical result to the ball, enabling us to rigorously prove Theorems 2.2–2.3 as direct consequences of their spherical counterparts.

In the remainder of this section, we define this operator, explore its key properties, and demonstrate how it facilitates the desired approximation result on the ball.

Definition 5.1. Given \( d \in  \mathbb{N}, k \in  {\mathbb{N}}_{0} \), let

\[G \mathrel{\text{:= }} \left\{  {\eta  \in  {\mathbb{S}}^{d}: {\eta }_{d + 1} \geq  \frac{1}{\sqrt{2}}}\right\} .\]

Define operator \( {S}_{k}: {\mathcal{L}}^{2}\left( G\right)  \rightarrow  {\mathcal{L}}^{2}\left( {\mathbb{B}}^{d}\right) \) by

\[\left( {{S}_{k}g}\right) \left( x\right)  \mathrel{\text{:= }} {\left| \widetilde{x}\right| }^{k}g\left( \frac{\widetilde{x}}{\left| \widetilde{x}\right| }\right),\; x \in  {\mathbb{B}}^{d}\]

and operator \( {T}_{k}: {\mathcal{L}}^{2}\left( {\mathbb{B}}^{d}\right)  \rightarrow  {\mathcal{L}}^{2}\left( G\right) \) by

\[{T}_{k}f\left( \eta \right)  = {\eta }_{d + 1}^{k}f\left( \frac{\overline{\eta }}{{\eta }_{d + 1}}\right),\;\eta  \in  G.\]

where \( \widetilde{x} \) has been defined earlier as \( \widetilde{x} = \left( \begin{matrix} x \\  1 \end{matrix}\right) \), and \( \overline{\eta } = {\left( {\eta }_{1},\ldots,{\eta }_{d}\right) }^{\top } \).

Following the idea in [26] and [66], it is straightforward to verify that

\[{S}_{k}{T}_{k} = {\operatorname{id}}_{{\mathcal{L}}^{2}\left( {\mathbb{B}}^{d}\right) },\;{T}_{k}{S}_{k} = {\operatorname{id}}_{{\mathcal{L}}^{2}\left( G\right) } \tag{5.1}\]

and

\[\left( {{S}_{k}{\sigma }_{k}\left( {\theta  \cdot   \circ  }\right) }\right) \left( x\right)  = {\sigma }_{k}\left( {\theta  \cdot  \widetilde{x}}\right),\; x \in  {\mathbb{B}}^{d},\]

\[\left( {{T}_{k}{\sigma }_{k}\left( {\theta  \cdot  \widetilde{ \circ  }}\right) }\right) \left( \eta \right)  = {\sigma }_{k}\left( {\theta  \cdot  \eta }\right),\;\eta  \in  G. \tag{5.2}\]

Lemma 5.1. For any \( r \geq  0 \),

\[\parallel g{\parallel }_{{\mathcal{H}}^{r}\left( G\right) } \simeq  {\begin{Vmatrix}{S}_{k}g\end{Vmatrix}}_{{\mathcal{H}}^{r}\left( {\mathbb{B}}^{d}\right) } \tag{5.3}\]

Proof. To start up, we assume \( r \in  \mathbb{N} \). Write \( {S}_{k}g \) as

\[{S}_{k}g\left( x\right)  = {\zeta }_{1}\left( x\right) g\left( {{\lambda }_{1}\left( x\right) }\right),\; x \in  {\mathbb{B}}^{d}, \tag{5.4}\]

where \( {\zeta }_{1}: {\mathbb{B}}^{d} \rightarrow  \mathbb{R} \) and \( {\lambda }_{1}: {\mathbb{B}}^{d} \rightarrow  G \) are given as

\[{\lambda }_{1}\left( x\right)  = \frac{\widetilde{x}}{\left| \widetilde{x}\right| } = \frac{1}{\sqrt{{\left| x\right| }^{2} + 1}}\left( \begin{array}{l} x \\  1 \end{array}\right),\;{\zeta }_{1}\left( x\right)  = {\left| \widetilde{x}\right| }^{k} = {\left( {\left| x\right| }^{2} + 1\right) }^{k/2},\; x \in  {\mathbb{B}}^{d}.\]

with \( {\lambda }_{1} \) has its inverse

\[{\lambda }_{2}\left( \eta \right)  = \frac{\overline{\eta }}{{\eta }_{d + 1}} = {\left( \frac{{\eta }_{1}}{{\eta }_{d + 1}},\ldots,\frac{{\eta }_{d}}{{\eta }_{d + 1}}\right) }^{\top },\;\eta  \in  G. \tag{5.5}\]

Suppose \( g \) be smooth enough, then the multivariate chain rule gives

\[\left| {\frac{{\partial }^{\left| \alpha \right| }\left( {{S}_{k}g}\right) }{\partial {x}_{1}^{{\alpha }_{1}}\ldots \partial {x}_{d}^{{\alpha }_{d}}}\left( x\right) }\right|  \lesssim  \mathop{\max }\limits_{{\left| \gamma \right|  \leq  \left| \alpha \right| }}\left| {\frac{{\partial }^{\left| \gamma \right| }g}{\partial {x}_{1}^{{\gamma }_{1}}\ldots \partial {x}_{d + 1}^{{\gamma }_{d + 1}}}\left( {{\lambda }_{1}\left( x\right) }\right) }\right| {\begin{Vmatrix}{\lambda }_{1}\end{Vmatrix}}_{{\mathcal{W}}^{\left| \alpha \right|,\infty }\left( {\mathbb{B}}^{d}\right) }{\begin{Vmatrix}{\zeta }_{1}\end{Vmatrix}}_{{\mathcal{W}}^{\left| \alpha \right|,\infty }\left( {\mathbb{B}}^{d}\right) }\]

\[\lesssim  \mathop{\max }\limits_{{\left| \gamma \right|  \leq  \left| \alpha \right| }}\left| {\frac{{\partial }^{\left| \gamma \right| }g}{\partial {x}_{1}^{{\gamma }_{1}}\ldots \partial {x}_{d + 1}^{{\gamma }_{d + 1}}}\left( {{\lambda }_{1}\left( x\right) }\right) }\right|.\]

As the trivial projection

\[\eta  \mapsto  \overline{\eta } = {\left( {\eta }_{1},\ldots,{\eta }_{d}\right) }^{\top }\]

is a homeomorphism from \( G \rightarrow  \left\{  {x \in  {\mathbb{B}}^{d}: \left| x\right|  \leq  1/\sqrt{2}}\right\} ,{\lambda }_{2} \) is then also a homeomorphism from \( G \) to its image \( {\mathbb{B}}^{d} \). Consequently,

\[{\begin{Vmatrix}{S}_{k}g\end{Vmatrix}}_{{\mathcal{H}}^{r}\left( {\mathbb{B}}^{d}\right) } \lesssim  \parallel g{\parallel }_{{\mathcal{H}}^{r}\left( G\right) }. \tag{5.6}\]

As (5.6) holds for all sufficiently smooth functions \( g \), it holds for all \( g \in  {\mathcal{H}}^{r}\left( G\right) \).

To prove \( \parallel g{\parallel }_{{\mathcal{H}}^{r}\left( \Omega \right) } \lesssim  {\begin{Vmatrix}{S}_{k}g\end{Vmatrix}}_{{\mathcal{H}}^{r}\left( {\mathbb{B}}^{d}\right) } \), it suffices to take \( f = {S}_{k}g \) and prove

\[{\begin{Vmatrix}{T}_{k}f\end{Vmatrix}}_{{\mathcal{H}}^{r}\left( \Omega \right) } \lesssim  \parallel f{\parallel }_{{\mathcal{H}}^{r}\left( {\mathbb{B}}^{d}\right) }. \tag{5.7}\]

We write

\[{T}_{k}f\left( \eta \right)  = {\zeta }_{2}\left( \eta \right) f\left( {{\lambda }_{2}\left( \eta \right) }\right),\;\eta  \in  G, \tag{5.8}\]

where \( {\lambda }_{2} \) as in (5.5) and \( {\zeta }_{2}\left( \eta \right)  = {\eta }_{d + 1}^{k} \). Then (5.7) can be proved using the same argument as above. Thus for \( r \in  \mathbb{N} \),

\[\parallel g{\parallel }_{{\mathcal{H}}^{r}\left( G\right) } \simeq  {\begin{Vmatrix}{S}_{k}g\end{Vmatrix}}_{{\mathcal{H}}^{r}\left( {\mathbb{B}}^{d}\right) }\]

For \( r \notin  \mathbb{N} \), it suffices to apply

\[\parallel g{\parallel }_{{\mathcal{H}}^{\left\lbrack  r\right\rbrack  }\left( G\right) } \simeq  {\begin{Vmatrix}{S}_{k}g\end{Vmatrix}}_{{\mathcal{H}}^{\left\lbrack  r\right\rbrack  }\left( {\mathbb{B}}^{d}\right) },\;\parallel g{\parallel }_{{\mathcal{H}}^{\left\lceil  r\right\rceil  }\left( G\right) } \simeq  {\begin{Vmatrix}{S}_{k}g\end{Vmatrix}}_{{\mathcal{H}}^{\left\lceil  r\right\rceil  }\left( {\mathbb{B}}^{d}\right) },\]

then the interpolation theory of linear operators yields

\[{\begin{Vmatrix}{S}_{k}g\end{Vmatrix}}_{{\mathcal{H}}^{r}\left( {\mathbb{B}}^{d}\right) } \lesssim  \parallel g{\parallel }_{{\mathcal{H}}^{r}\left( G\right) },\;{\begin{Vmatrix}{T}_{k}f\end{Vmatrix}}_{{\mathcal{H}}^{r}\left( G\right) } \lesssim  \parallel f{\parallel }_{{\mathcal{H}}^{r}\left( {\mathbb{B}}^{d}\right) }\]

and completes the proof of this lemma.

### 5.1 Proof of Theorem 2.2

We are now in the position to provide the detailed proof for Theorem 2.2.

Proof. Without loss of generality, assume \( \Omega  \subset  {\mathbb{B}}^{d} \). By the classical extension theory [86], there exists an extension \( {f}_{E} \) such that

\[{\begin{Vmatrix}{f}_{E}\end{Vmatrix}}_{{\mathcal{H}}^{r}\left( {\mathbb{B}}^{d}\right) } \leq  {\begin{Vmatrix}{f}_{E}\end{Vmatrix}}_{{\mathcal{H}}^{r}\left( {\mathbb{R}}^{d}\right) } \lesssim  \parallel f{\parallel }_{{\mathcal{H}}^{r}\left( \Omega \right) }.\]

By Lemma 5.1, we have

\[{\begin{Vmatrix}{T}_{k}{f}_{E}\end{Vmatrix}}_{{\mathcal{H}}^{r}\left( G\right) } \lesssim  {\begin{Vmatrix}{f}_{E}\end{Vmatrix}}_{{\mathcal{H}}^{r}\left( {\mathbb{B}}^{d}\right) } \lesssim  \parallel f{\parallel }_{{\mathcal{H}}^{r}\left( \Omega \right) }.\]

By the extension theorems again, there exists a function \( g \) on \( {\mathbb{S}}^{d} \), which is an extension of \( {T}_{k}{f}_{E} \), such that

\[\parallel g{\parallel }_{{\mathcal{H}}^{r}\left( {\mathbb{S}}^{d}\right) } \lesssim  {\begin{Vmatrix}{T}_{k}{f}_{E}\end{Vmatrix}}_{{\mathcal{H}}^{r}\left( G\right) } \lesssim  \parallel f{\parallel }_{{\mathcal{H}}^{r}\left( \Omega \right) },\]

\[g\left( \eta \right)  = {\left( -1\right) }^{k + 1}g\left( {-\eta }\right),\;\eta  \in  {\mathbb{S}}^{d}.\]

Now by Theorem 4.1, there exists a vector \( a \in  {\mathbb{R}}^{n} \) with \( \parallel a{\parallel }_{2} \lesssim  {h}^{-\frac{{2k} + 1 - {2r}}{2}}\parallel f{\parallel }_{{\mathcal{H}}^{r}\left( {\mathbb{B}}^{d}\right) } \) such that

\[{\begin{Vmatrix}{T}_{k}{f}_{E} - \mathop{\sum }\limits_{{j = 1}}^{n}{a}_{j}{\sigma }_{k}\left( {\theta }_{j}^{ * } \cdot   \circ  \right) \end{Vmatrix}}_{{\mathcal{H}}^{s}\left( G\right) } \leq  {\begin{Vmatrix}g - \mathop{\sum }\limits_{{j = 1}}^{n}{a}_{j}{\sigma }_{k}\left( {\theta }_{j}^{ * } \cdot   \circ  \right) \end{Vmatrix}}_{{\mathcal{H}}^{s}\left( {\mathbb{S}}^{d}\right) } \tag{5.9}\]

\[\lesssim  {\left( \frac{h}{\underline{h}}\right) }^{d + 1}{h}^{r - s}\parallel g{\parallel }_{{\mathcal{H}}^{r}\left( {\mathbb{S}}^{d}\right) } \lesssim  {\left( \frac{h}{\underline{h}}\right) }^{d + 1}{h}^{r - s}\parallel f{\parallel }_{{\mathcal{H}}^{r}\left( \Omega \right) }.\]

Applying Lemma 5.1 again, we obtain

\[{\begin{Vmatrix}{f}_{E} - \mathop{\sum }\limits_{{j = 1}}^{m}{a}_{j}{\sigma }_{k}\left( \circ  \cdot  {w}_{j}^{ * } + {b}_{j}^{ * }\right) \end{Vmatrix}}_{{\mathcal{H}}^{s}\left( {\mathbb{B}}^{d}\right) } = {\begin{Vmatrix}{S}_{k}\left( {T}_{k}{f}_{E} - \mathop{\sum }\limits_{{j = 1}}^{m}{a}_{j}{\sigma }_{k}\left( {\theta }_{j}^{ * } \cdot   \circ  \right) \right) \end{Vmatrix}}_{{\mathcal{H}}^{s}\left( {\mathbb{B}}^{d}\right) }\]

\[\lesssim  {\begin{Vmatrix}{T}_{k}{f}_{E} - \mathop{\sum }\limits_{{j = 1}}^{m}{a}_{j}{\sigma }_{k}\left( {\theta }_{j}^{ * } \cdot   \circ  \right) \end{Vmatrix}}_{{\mathcal{H}}^{s}\left( G\right) } \lesssim  {\left( \frac{h}{\underline{h}}\right) }^{d + 1}{h}^{r - s}\parallel f{\parallel }_{{\mathcal{H}}^{r}\left( \Omega \right) }. \tag{5.10}\]

This proves

\[{\begin{Vmatrix}f - \mathop{\sum }\limits_{{j = 1}}^{n}{a}_{j}{\phi }_{j}\end{Vmatrix}}_{{\mathcal{H}}^{s}\left( \Omega \right) } \lesssim  {h}^{r - s}\parallel f{\parallel }_{{\mathcal{H}}^{r}\left( \Omega \right) }. \tag{5.11}\]

### 5.2 Proof of Theorem 2.3

Proof. Without loss of generality, we assume \( \Omega  \subset  {\mathbb{B}}^{d} \). By the extension theorem again, for \( f \in \; {\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) \), there exists \( g \) on \( {\mathbb{S}}^{d} \), which is an extension of \( {T}_{k}f \), such that

\[\parallel g{\parallel }_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( {\mathbb{S}}^{d}\right) } \lesssim  {\begin{Vmatrix}{T}_{k}f\end{Vmatrix}}_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( G\right) } \lesssim  \parallel f{\parallel }_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) },\]

\[g\left( \eta \right)  = {\left( -1\right) }^{k + 1}g\left( {-\eta }\right),\;\eta  \in  {\mathbb{S}}^{d}.\]

By Theorem 4.2, there exists \( {\psi }_{0} \in  {\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) \) such that

\[g\left( \eta \right)  = {\oint }_{{\mathbb{S}}^{d}}{\psi }_{0}\left( \theta \right) {\sigma }_{k}\left( {\theta  \cdot  \eta }\right) {d\theta }.\]

Then

\[{\int }_{{\mathbb{S}}^{d}}{\psi }_{0}\left( \theta \right) {\sigma }_{k}\left( {\theta  \cdot  \widetilde{x}}\right) {d\theta } = {\int }_{{\mathbb{S}}^{d}}{\psi }_{0}\left( \theta \right) {\left| \widetilde{x}\right| }^{k}{\sigma }_{k}\left( {\theta  \cdot  \frac{\widetilde{x}}{\left| \widetilde{x}\right| }}\right) {d\theta }\]

\[= {\left| \widetilde{x}\right| }^{k}g\left( \frac{\widetilde{x}}{\left| \widetilde{x}\right| }\right)  = {S}_{k}g\left( x\right)  = f\left( x\right),\; x \in  {\mathbb{B}}^{d},\]

hence Theorem 4.2 implies

\[\parallel f{\parallel }_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) } \gtrsim  \parallel g{\parallel }_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( {\mathbb{S}}^{d}\right) } \simeq  {\begin{Vmatrix}{\psi }_{0}\end{Vmatrix}}_{{\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) }\]

\[\geq  \mathop{\inf }\limits_{{\psi  \in  {\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) }}\left\{  {\parallel \psi {\parallel }_{{\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) }: {\int }_{{\mathbb{S}}^{d}}{\sigma }_{k}\left( {\theta  \cdot  \widetilde{x}}\right) \psi \left( \theta \right) {d\theta } = f\left( x\right) }\right\} .\]

Now suppose there is some \( {\psi }_{1} \) such that

\[{\int }_{{\mathbb{S}}^{d}}{\sigma }_{k}\left( {\theta  \cdot  \widetilde{x}}\right) {\psi }_{1}\left( \theta \right) {d\theta } = f\left( x\right),\]

then

\[{\int }_{{\mathbb{S}}^{d}}{\psi }_{1}\left( \theta \right) {\sigma }_{k}\left( {\theta  \cdot  \eta }\right) {d\theta } = {\int }_{{\mathbb{S}}^{d}}{\psi }_{1}\left( \theta \right) {\eta }_{d + 1}^{k}{\sigma }_{k}\left( {\theta  \cdot  \frac{\eta }{{\eta }_{d + 1}}}\right) {d\theta }\]

\[= {\eta }_{d + 1}^{k}f\left( \frac{\overline{\eta }}{{\eta }_{d + 1}}\right)  = {T}_{k}f\left( \eta \right).\]

Again, Theorem 4.2 implies

\[\parallel f{\parallel }_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) } \lesssim  \parallel f{\parallel }_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( {\mathbb{B}}^{d}\right) } \simeq  {\begin{Vmatrix}{T}_{k}f\end{Vmatrix}}_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( {\mathbb{S}}^{d}\right) } \lesssim  {\begin{Vmatrix}{\psi }_{1}\end{Vmatrix}}_{{\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) },\]

which yields

\[\parallel f{\parallel }_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) } \lesssim  \mathop{\inf }\limits_{{\psi  \in  {\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) }}\left\{  {\parallel \psi {\parallel }_{{\mathcal{L}}^{2}\left( {\mathbb{S}}^{d}\right) }: {\int }_{{\mathbb{S}}^{d}}{\sigma }_{k}\left( {\theta  \cdot  \widetilde{x}}\right) \psi \left( \theta \right) {d\theta } = f\left( x\right) }\right\} .\]

## 6 Deterministic analysis of randomized neural networks: questioning the necessity of randomization

Neural network training involves non-convex optimization, which is computationally challenging. While greedy methods [87, 88] offer convergence guarantees, they can be costly. Randomized approaches, such as stochastic basis selection [41], Extreme Learning Machines (ELM) [42, 43, 89], and random features [44, 45, 50, 49, 56, 53], provide efficient alternatives by fixing hidden layer parameters randomly (often uniformly or Gaussian [90]) and only optimizing linear weights. These methods have shown empirical success in various tasks, including function approximation, classification, and solving PDEs [91, 92, 51, 52, 54, 55, 57, 61, 58, 60]. However, the theoretical justification for specific sampling strategies often lags behind empirical performance.

Let \( {\mathbb{E}}_{n,\mu } \) denote expectation over \( n \) i.i.d. samples \( {\left\{  {\theta }_{j}\right\}  }_{j = 1}^{n} \) from a probability distribution \( \mu \) on \( {\mathbb{S}}^{d} \). If \( \mu \) is uniform, we write \( {\mathbb{E}}_{n} \). Standard concentration inequalities applied to the integral representation (1.16) yield the follwoing theorem.

Theorem 6.1. Let \( d, k, n,\Omega \) be as in Theorem 2.2, then for \( f \in  {\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) \),

\[{\mathbb{E}}_{n}\left\lbrack  {\mathop{\inf }\limits_{{a \in  {\mathbb{R}}^{n}}}{\begin{Vmatrix}f - \mathop{\sum }\limits_{{j = 1}}^{n}{a}_{j}{\phi }_{j}\end{Vmatrix}}_{{\mathcal{L}}^{2}\left( \Omega \right) }^{2}}\right\rbrack   \lesssim  {n}^{-1}\parallel f{\parallel }_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) }^{2}. \tag{6.1}\]

This \( \mathcal{O}\left( {n}^{-1/2}\right) {\mathcal{L}}^{2} \) -rate provides theoretical backing for random feature methods. (Note: Theorem 2.3 allows deterministic points + uniform sampling, improving on [28] where the measure was unknown. We focus here on the stronger results derived from Theorem 2.2.)

Leveraging our deterministic approximation result (Theorem 2.2), we can refine the analysis of random sampling:

Theorem 6.2. Let \( {\left\{  {\theta }_{j}\right\}  }_{j = 1}^{n} \) be i.i.d. uniform samples from \( {\mathbb{S}}^{d} \). With probability at least \( 1 - \delta \), there exists \( a \in  {\mathbb{R}}^{n} \) with \( \parallel a{\parallel }_{2} \lesssim  {\left( \frac{n}{\log \left( {n/\delta }\right) }\right) }^{\frac{{2k} + 1 - {2r}}{2d}}\parallel f{\parallel }_{{\mathcal{H}}^{r}\left( \Omega \right) } \) such that

\[{\begin{Vmatrix}f - \mathop{\sum }\limits_{{j = 1}}^{n}{a}_{j}{\phi }_{j}\end{Vmatrix}}_{{\mathcal{H}}^{s}\left( \Omega \right) } \lesssim  {\left( \frac{n}{\log \left( {n/\delta }\right) }\right) }^{-\frac{r - s}{d}}\parallel f{\parallel }_{{\mathcal{H}}^{r}\left( \Omega \right) }. \tag{6.2}\]

Consequently, the expected error satisfies

\[{\mathbb{E}}_{n}\left\lbrack  {\mathop{\inf }\limits_{{a \in  {\mathbb{R}}^{n}}}{\begin{Vmatrix}f - \mathop{\sum }\limits_{{j = 1}}^{n}{a}_{j}{\phi }_{j}\end{Vmatrix}}_{{\mathcal{H}}^{s}\left( \Omega \right) }}\right\rbrack   \lesssim  {\left( \frac{n}{\log n}\right) }^{-\frac{r - s}{d}}\parallel f{\parallel }_{{\mathcal{H}}^{r}\left( \Omega \right) }. \tag{6.3}\]

In particular, for \( {\mathcal{L}}^{2} \) approximation \( \left( {s = 0, r = \frac{d + {2k} + 1}{2}}\right) \),

\[{\mathbb{E}}_{n}\left\lbrack  {\mathop{\inf }\limits_{{a \in  {\mathbb{R}}^{n}}}{\begin{Vmatrix}f - \mathop{\sum }\limits_{{j = 1}}^{n}{a}_{j}{\phi }_{j}\end{Vmatrix}}_{{\mathcal{L}}^{2}\left( \Omega \right) }}\right\rbrack   \lesssim  {\left( \frac{n}{\log n}\right) }^{-\frac{1}{2} - \frac{{2k} + 1}{2d}}\parallel f{\parallel }_{{\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) }. \tag{6.4}\]

Proof. The proof relies on Theorem 2.2, which depends on \( h = \mathop{\max }\limits_{{\theta  \in  {\mathbb{S}}^{d}}}\mathop{\min }\limits_{{1 \leq  j \leq  n}}\rho \left( {\theta,{\theta }_{j}}\right) \). Standard covering arguments and concentration inequalities show that for i.i.d. uniform points \( \left\{  {\theta }_{j}\right\} , h \lesssim \; {\left( n/\log \left( n/\delta \right) \right) }^{-1/d} \) with probability at least \( 1 - \delta \). Substituting this into (2.12) yields the result.

This analysis clarifies why randomization works: it efficiently generates parameter distributions \( \left\{  {\theta }_{j}\right\} \) that are nearly well-distributed (i.e., \( h \) is small) with high probability. However, comparing Theorem 6.2 with Theorem 2.2 reveals that deterministically chosen well-distributed points achieve a *better* approximation rate (lacking the \( \log n \) factor) and eliminate the small probability \( \delta \) of failure associated with random sampling. Since well-distributed point sets on spheres can be constructed deterministically, we conclude that they are sufficient and theoretically preferable to random sampling for achieving optimal approximation rates in this context. Randomization is thus a practical means to approximate a desirable deterministic configuration.

## 7 Generalization Analysis for FNS using Rademacher Complexity

We analyze the generalization error when approximating a target function \( f \) using a dataset \( {D}_{m} \). Given the approximation rate (2.13) and coefficient bound \( M \) from Theorem 2.2, we can estimate the expected error between \( f \) and an approximant constructed from \( m \) i.i.d. samples.

Standard generalization analysis uses concentration inequalities [93] and complexity measures like Rademacher complexity, covering numbers, or spectral complexity [94, 95]. These bounds depend critically on the coefficient norms of the approximants.

We illustrate using a second-order linear elliptic PDE with Neumann boundary conditions on a bounded \( {C}^{\infty } \) domain \( \Omega  \subset  {\mathbb{R}}^{d} \) with \( \left| \Omega \right|  = 1 \):

\[\left\{  \begin{array}{ll}  - {\Delta g}\left( x\right)  + g\left( x\right)  = h\left( x\right), & x \in  \Omega \\  \frac{\partial g}{\partial n}\left( x\right)  = 0, & x \in  \partial \Omega  \end{array}\right. \tag{7.1}\]

where \( h \in  {\mathcal{L}}^{\infty }\left( \Omega \right) \) is given. The variational formulation seeks \( g \in  {\mathcal{H}}^{1}\left( \Omega \right) \) minimizing the energy functional:

\[\mathcal{E}\left( g\right)  \mathrel{\text{:= }} {\int }_{\Omega }\Psi \left( g\right) \left( x\right) {dx},\;\text{ where }\Psi \left( g\right) \left( x\right)  = \frac{1}{2}{\left| \nabla g\left( x\right) \right| }^{2} + \frac{1}{2}g{\left( x\right) }^{2} - h\left( x\right) g\left( x\right). \tag{7.2}\]

The target function \( f \) is the unique minimizer of \( \mathcal{E}\left( g\right) \) due to strict convexity:

\[f \mathrel{\text{:= }} \arg \mathop{\min }\limits_{{g \in  {\mathcal{H}}^{1}\left( \Omega \right) }}\mathcal{E}\left( g\right) \tag{7.3}\]

Using Monte Carlo, we draw \( m \) i.i.d. uniform samples \( {x}_{1},\ldots,{x}_{m} \) from \( \Omega \) and approximate \( \mathcal{E}\left( g\right) \) with the empirical risk:

\[{\mathcal{E}}_{m}\left( g\right)  \mathrel{\text{:= }} \frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}\Psi \left( g\right) \left( {x}_{i}\right). \tag{7.4}\]

Assuming \( f \in  {\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) \), we use the hypothesis space \( {L}_{n, M}^{k} \) (from Theorem 2.2). The empirical risk minimizer (ERM) is:

\[{f}_{n, m} \mathrel{\text{:= }} \arg \mathop{\min }\limits_{{g \in  {L}_{n, M}^{k}}}{\mathcal{E}}_{m}\left( g\right). \tag{7.5}\]

Since \( {L}_{n, M}^{k} \) is linear and \( {\mathcal{E}}_{m}\left( g\right) \) is quadratic in coefficients, \( {f}_{n, m} \) is unique and efficiently computable via convex optimization.

The generalization error is the expected excess risk:

\[{\mathbb{E}}_{{x}_{1},\ldots,{x}_{m}}\left\lbrack  {\mathcal{E}\left( {f}_{n, m}\right)  - \mathcal{E}\left( f\right) }\right\rbrack . \tag{7.6}\]

Due to strong convexity of \( \mathcal{E}\left( g\right) \), bounding this also bounds the expected squared \( {\mathcal{H}}^{1}\left( \Omega \right) \) error \( {\mathbb{E}}_{{x}_{1},\ldots,{x}_{m}}\left\lbrack  {\begin{Vmatrix}{f}_{n, m} - f\end{Vmatrix}}_{{\mathcal{H}}^{1}\left( \Omega \right) }^{2}\right\rbrack . \)

### 7.1 Rademacher Complexity

We use Rademacher complexity [96] to bound (7.6). For a function class \( \mathcal{F}: \Omega  \rightarrow  \mathbb{R} \), its Rademacher complexity is:

\[{R}_{m}\left( \mathcal{F}\right)  = {\mathbb{E}}_{{x}_{i},{\xi }_{i}}\left\lbrack  {\mathop{\sup }\limits_{{g \in  \mathcal{F}}}\frac{1}{m}\mathop{\sum }\limits_{{i = 1}}^{m}{\xi }_{i}g\left( {x}_{i}\right) }\right\rbrack , \tag{7.7}\]

where \( {x}_{i} \) are i.i.d. uniform samples and \( {\xi }_{i} \) are i.i.d. Rademacher variables \( \left( {\pm 1}\right) \).

We bound \( {R}_{m}\left( {\mathcal{F}}_{n, M}\right) \) for \( {\mathcal{F}}_{n, M} \mathrel{\text{:= }} \left\{  {\Psi \left( g\right) : g \in  {L}_{n, M}^{k}}\right\} \). Since \( {L}_{n, M}^{k} \subset  {\sum }_{n, M}^{k} \), monotonicity gives \( {R}_{m}\left( {\mathcal{F}}_{n, M}\right)  \leq  {R}_{m}\left( \left\{  {\Psi \left( g\right) : g \in  {\sum }_{n, M}^{k}}\right\}  \right) \). Using standard Rademacher complexity properties (sum rule, Lipschitz composition, product rule [97,96,98,99] and known bounds \( {R}_{m}\left( {\sum }_{n, M}^{k}\right)  \lesssim  M{m}^{-1/2} \) and \( {R}_{m}\left( \left\{  {\partial g/\partial {x}_{j}: g \in  {\sum }_{n, M}^{k}}\right\}  \right)  \lesssim  M{m}^{-1/2} \) [97, Theorem 6], we get:

\[{R}_{m}\left( {\mathcal{F}}_{n, M}\right)  \leq  {R}_{m}\left( \left\{  {\Psi \left( g\right) : g \in  {\sum }_{n, M}^{k}}\right\}  \right)  \lesssim  \left( {M + \parallel h{\parallel }_{{\mathcal{L}}^{\infty }\left( \Omega \right) }}\right) M{m}^{-\frac{1}{2}}. \tag{7.8}\]

### 7.2 Generalization analysis

Combining Theorem 2.2 and the complexity bound (7.8) yields the generalization error estimate.

Theorem 7.1. Let \( \Omega  \subset  {\mathbb{R}}^{d} \) be a bounded \( {C}^{\infty } \) domain. Let \( f,{f}_{n, m} \) be defined by (7.3) and (7.5). If \( f \in  {\mathcal{H}}^{\frac{d + {2k} + 1}{2}}\left( \Omega \right) \), the expected excess risk satisfies:

\[{\mathbb{E}}_{{x}_{1},\ldots,{x}_{m}}\left\lbrack  {\mathcal{E}\left( {f}_{n, m}\right)  - \mathcal{E}\left( f\right) }\right\rbrack   \lesssim  \left( {M + \parallel h{\parallel }_{{\mathcal{L}}^{\infty }\left( \Omega \right) }}\right) M{m}^{-\frac{1}{2}} + {M}^{2}{n}^{-1 - \frac{{2k} - 1}{d}}. \tag{7.9}\]

Choosing \( n = \left\lceil  {m}^{\frac{d}{2\left( {d + {2k} - 1}\right) }}\right\rceil \) balances the terms, giving:

\[{\mathbb{E}}_{{x}_{1},\ldots,{x}_{m}}\left\lbrack  {\mathcal{E}\left( {f}_{n, m}\right)  - \mathcal{E}\left( f\right) }\right\rbrack   \lesssim  \left( {M + \parallel h{\parallel }_{{\mathcal{L}}^{\infty }\left( \Omega \right) }}\right) M{m}^{-\frac{1}{2}}. \tag{7.10}\]

By strong convexity of \( \mathcal{E}\left( g\right) \), the expected squared \( {\mathcal{H}}^{1}\left( \Omega \right) \) error is bounded:

\[{\mathbb{E}}_{{x}_{1},\ldots,{x}_{m}}\left\lbrack  {\begin{Vmatrix}{f}_{n, m} - f\end{Vmatrix}}_{{\mathcal{H}}^{1}\left( \Omega \right) }^{2}\right\rbrack   \lesssim  \left( {M + \parallel h{\parallel }_{{\mathcal{L}}^{\infty }\left( \Omega \right) }}\right) M{m}^{-\frac{1}{2}}. \tag{7.11}\]

The proof follows standard arguments [96, 98]. The first term in (7.9) arises from the Rademacher complexity bound (7.8) via generalization bounds. The second term is the approximation error from Theorem 2.2. Strong convexity yields (7.11). See [97] for similar detailed proofs for PDE approximation.

## 8 Concluding remarks

In this paper, we developed a new integral representation of Sobolev space using \( {\operatorname{ReLU}}^{k} \) function, and compared it with the corresponding result of Barron spaces. We also showed in Sobolev spaces, the approximation properties of nonlinear shallow neural networks can be fully realized through simple linearization, and provided an upper bound of the corresponding coefficients. Such an upper bound is essential in approximation theory, and allows the corresponding generalization analysis. It is also worth noting that the techniques in this paper also inspires a Bernstein inequality for ReLU \( k \) neural networks, for which we are completing a new paper entitled "Bernstein Inequalities for Linearized ReLU \( k \) Neural Networks and Applications".

Since a DNN is essentially a composition of shallow neural networks, we hope that the findings in this paper provide valuable insights into the role of nonlinearity within deep neural networks. Furthermore, we aim for this work to inspire further mathematical research into the interplay between nonlinearity and expressive power (approximation properties) in deep learning.

## References

[1] Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. Nature, 521(7553):436-444, 2015.

[2] George Cybenko. Approximation by superpositions of a sigmoidal function. Mathematics of control, signals and systems, 2(4):303-314, 1989.

[3] Kurt Hornik, Maxwell Stinchcombe, and Halbert White. Multilayer feedforward networks are universal approximators. Neural networks, 2(5):359-366, 1989.

[4] Moshe Leshno, Vladimir Ya Lin, Allan Pinkus, and Shimon Schocken. Multilayer feedforward networks with a nonpolynomial activation function can approximate any function. Neural networks, 6(6):861-867, 1993.

[5] Jason M Klusowski and Andrew R Barron. Approximation by combinations of relu and squared relu ridge functions with 11 and 10 controls. IEEE Transactions on Information Theory, 64(12):7649-7656, 2018.

[6] Jonathan W Siegel and Jinchao Xu. High-order approximation rates for shallow neural networks with cosine and ReLUk activation functions. Applied and Computational Harmonic Analysis, 58:1-26, 2022.

[7] Gilles Pisier. Remarques sur un résultat non publié de B. Maurey. Séminaire d'Analyse fonctionnelle (dit" Maurey-Schwartz"), pages 1-12, 1981.

[8] B Maurey. Type et cotype dans les espaces munis de structures locales inconditionnelles. Seminaire Maurey-Schwartz, pages 1-25, 1973.

[9] Lee K Jones. A simple lemma on greedy approximation in hilbert space and convergence rates for projection pursuit regression and neural network training. The Annals of Statistics, 20(1):608-613, 1992.

[10] Andrew R Barron. Universal approximation bounds for superpositions of a sigmoidal function. IEEE Transactions on Information theory, 39(3):930-945, 1993.

[11] Andrew R Barron. Approximation and estimation bounds for artificial neural networks. Machine learning, 14:115-133, 1994.

[12] Ronald A DeVore and Vladimir N Temlyakov. Some remarks on greedy algorithms. Advances in computational Mathematics, 5(1):173-187, 1996.

[13] Yuly Makovoz. Uniform approximation by neural networks. Journal of Approximation Theory, 95(2):215-228, 1998.

[14] V. Kürková and M. Sanguineti. Bounds on rates of variable basis and neural network approximation. IEEE Transactions on Information Theory, 47(6):2659-2665, 2001.

[15] V. Kürková and M. Sanguineti. Comparison of worst case errors in linear and neural network approximation. IEEE Transactions on Information Theory, 48(1):264-275, 2002.

[16] Grzegorz Lewicki and Giuseppe Marino. Approximation of functions of finite variation by superpositions of a sigmoidal function. Applied Mathematics Letters, 17(10):1147-1152, 2004.

[17] Vladimir N Temlyakov. Greedy approximation. Acta Numerica, 17:235-409, 2008.

[18] Andrew R Barron, Albert Cohen, Wolfgang Dahmen, and Ronald A DeVore. Approximation and learning by greedy algorithms. Annals of Statistics, 36(1):64-94, 2008.

[19] Weinan E, Chao Ma, and Lei Wu. A priori estimates of the population risk for two-layer neural networks. Communications in Mathematical Sciences, 17(5):1407-1425, 2019.

[20] Weinan E, Chao Ma, and Lei Wu. The barron space and the flow-induced function spaces for neural network models. Constructive Approximation, 55:259-292, 2022.

[21] Weinan E and Stephan Wojtowytsch. Representation formulas and pointwise properties for barron functions. Calculus of Variations and Partial Differential Equations, 61(2):46, 2022.

[22] Ronald A DeVore. Nonlinear approximation. Acta Numerica, 7:51-150, 1998.

[23] Ronald DeVore, Boris Hanin, and Guergana Petrova. Neural network approximation. Acta Numerica, 30:327-444, 2021.

[24] Sergei Vladimirovich Konyagin, Aleksandr Andreevich Kuleshov, and Vitalii Evgen'evich Maiorov. Some problems in the theory of ridge functions. Proceedings of the Steklov Institute of Mathematics, 301:144-169, 2018.

[25] Yuly Makovoz. Random approximants and neural networks. Journal of Approximation Theory, 85(1):98-109, 1996.

[26] Francis Bach. Breaking the curse of dimensionality with convex neural networks. The Journal of Machine Learning Research, 18(1):629-681, 2017.

[27] Jinchao Xu. Finite neuron method and convergence analysis. Communications in Computational Physics, 28(5):1707-1745, 2020.

[28] Jonathan W Siegel and Jinchao Xu. Sharp bounds on the approximation rates, metric entropy, and n-widths of shallow neural networks. Foundations of Computational Mathematics, pages 1-57, 2022.

[29] Jonathan W Siegel and Jinchao Xu. Optimal convergence rates for the orthogonal greedy algorithm. IEEE Transactions on Information Theory, 68(5):3354-3361, 2022.

[30] Hrushikesh Mhaskar and Tong Mao. Tractability of approximation by general shallow networks. arXiv preprint arXiv:2308.03230, 2023.

[31] Hrushikesh Narhar Mhaskar. Approximation properties of a multilayered feedforward artificial neural network. Advances in Computational Mathematics, 1:61-80, 1993.

[32] Pencho P Petrushev. Approximation by ridge functions and neural networks. SIAM Journal on Mathematical Analysis, 30(1):155-189, 1998.

[33] Allan Pinkus. Approximation theory of the mlp model in neural networks. Acta numerica, 8:143-195, 1999.

[34] Limin Ma, Jonathan W Siegel, and Jinchao Xu. Uniform approximation rates and metric entropy of shallow neural networks. Research in the Mathematical Sciences, 9(3):46, 2022.

[35] Yan Meng and Pingbing Ming. A new function space from barron class and application to neural network approximation. Communications in Computational Physics, 32(5):1361-1400, 2022.

[36] Jonathan W Siegel and Jinchao Xu. Characterization of the variation spaces corresponding to shallow neural networks. Constructive Approximation, 57(3):1109-1132, 2023.

[37] Tong Mao and Jinchao Xu. Do neural networks have better approximation properties than polynomials or finite elements for high-dimensional problems? preprint, 2025.

[38] Zuowei Lu, Zhiqiang Shen, Haizhao Yang, and Shijun Zhang. Deep neural networks with fixed width can be universal approximators. Neural Networks, 141:202-211, 2021.

[39] Tong Mao, Jonathan W Siegel, and Jinchao Xu. Approximation rates for shallow reluk neural networks on sobolev spaces via the radon transform. arXiv preprint arXiv:2408.10996, 2024.

[40] Yoh-Han Pao, Gwang-Hoon Park, and Dejan J Sobajic. Learning and generalization characteristics of the random vector functional-link net. Neurocomputing, 6(2):163-180, 1994.

[41] Boris Igelnik and Yoh-Han Pao. Stochastic choice of basis functions in adaptive function approximation and the functional-link net. IEEE Transactions on Neural Networks, 6(6):1320- 1329, 1995.

[42] Guang-Bin Huang, Qin-Yu Zhu, and Chee-Kheong Siew. Extreme learning machine: theory and applications. Neurocomputing, 70:489-501, 2006.

[43] Guang-Bin Huang, Lei Chen, and Chee-Kheong Siew. Universal approximation using incremental constructive feedforward networks with random hidden nodes. IEEE transactions on neural networks, 17(4):879-892, 2006.

[44] Ali Rahimi and Benjamin Recht. Random features for large-scale kernel machines. In Advances in Neural Information Processing Systems, pages 1177-1184, 2007.

[45] Ali Rahimi and Benjamin Recht. Weighted sums of random kitchen sinks: Replacing minimization with randomization in learning. Advances in neural information processing systems, 21, 2008.

[46] Ali Rahimi and Benjamin Recht. Uniform approximation of functions with random bases. In 2008 46th annual allerton conference on communication, control, and computing, pages 555-561. IEEE, 2008.

[47] Andrew M Saxe, Pang Wei Koh, Zhenghao Chen, Maneesh Bhand, Bipin Suresh, and Andrew Y Ng. On random weights and unsupervised feature learning. In Icml, volume 2, page 6, 2011.

[48] Raja Giryes, Guillermo Sapiro, and Alex M Bronstein. Deep neural networks with random gaussian weights: A universal classification strategy? IEEE Transactions on Signal Processing, 64(13):3444-3457, 2016.

[49] Francis Bach. On the equivalence between kernel quadrature rules and random feature expansions. Journal of machine learning research, 18(21):1-38, 2017.

[50] Zhu Li, Jean-Francois Ton, Dino Oglic, and Dino Sejdinovic. Towards a unified analysis of random fourier features. In International conference on machine learning, pages 3905-3914. PMLR, 2019.

[51] Federica Gerace, Bruno Loureiro, Florent Krzakala, Marc Mézard, and Lenka Zdeborová. Generalisation error in learning with random features and the hidden manifold model. In International Conference on Machine Learning, pages 3452-3462. PMLR, 2020.

[52] Hong Hu and Yue M Lu. Universality laws for high-dimensional learning with random features. IEEE Transactions on Information Theory, 69(3):1932-1964, 2022.

[53] Song Mei and Andrea Montanari. The generalization error of random features regression: Precise asymptotics and the double descent curve. Communications on Pure and Applied Mathematics, 75(4):667-766, 2022.

[54] Vikas Dwivedi and Balaji Srinivasan. Physics informed extreme learning machine (pielm)-a rapid method for the numerical solution of partial differential equations. Neurocomputing, 391:96-118, 2020.

[55] Suchuan Dong and Zongwei Li. Local extreme learning machines and domain decomposition for solving linear and nonlinear partial differential equations. Computer Methods in Applied Mechanics and Engineering, 387:114129, 2021.

[56] Nicholas H Nelsen and Andrew M Stuart. The random feature model for input-output maps between banach spaces. SIAM Journal on Scientific Computing, 43(5): A3212-A3243, 2021.

[57] Jingrun Chen, Xurong Chi, Weinan E, and Zhouwang Yang. Bridging traditional and machine learning-based algorithms for solving pdes: the random feature method. J Mach Learn, 1:268- 98, 2022.

[58] Haoning Dang and Fei Wang. Local randomized neural networks with hybridized discontinuous petrov-galerkin methods for stokes-darcy flows. Physics of Fluids, 36(8), 2024.

[59] Yiran Wang and Suchuan Dong. An extreme learning machine-based method for computational pdes in higher dimensions. Computer Methods in Applied Mechanics and Engineering, 418:116578, 2024.

[60] Xurong Chi, Jingrun Chen, and Zhouwang Yang. The random feature method for solving interface problems. Computer Methods in Applied Mechanics and Engineering, 420:116719, 2024.

[61] Zezhong Zhang, Feng Bao, Lili Ju, and Guannan Zhang. Transferable neural networks for partial differential equations. Journal of Scientific Computing, 99(1):2, 2024.

[62] Philippe G. Ciarlet. The Finite Element Method for Elliptic Problems, volume 4 of Studies in Mathematics and its Applications. North-Holland, 1978.

[63] Qun Lin, Hehu Xie, and Jinchao Xu. Lower bounds of the discretization error for piecewise polynomials. Mathematics of Computation, 83(285):1-13, 2014.

[64] Ronald A DeVore and George G Lorentz. Constructive approximation, volume 303. Springer Science & Business Media, 1993.

[65] Tong Mao and Ding-Xuan Zhou. Rates of approximation by relu shallow neural networks. Journal of Complexity, 79:101784, 2023.

[66] Yunfei Yang and Ding-Xuan Zhou. Optimal rates of approximation by shallow relu k neural networks and applications to nonparametric regression. Constructive Approximation, pages 1-32, 2024.

[67] Jinchao Xu. Iterative methods by space decomposition and subspace correction. SIAM review, 34(4):581-613, 1992.

[68] Vitaly E Maiorov and Ron Meir. On the near optimality of the stochastic approximation of smooth functions by neural networks. Advances in Computational Mathematics, 13:79-103, 2000.

[69] Peter L Bartlett, Nick Harvey, Christopher Liaw, and Abbas Mehrabian. Nearly-tight vc-dimension and pseudodimension bounds for piecewise linear neural networks. The Journal of Machine Learning Research, 20(1):2285-2301, 2019.

[70] Jonathan W Siegel. Optimal approximation rates for deep relu neural networks on sobolev and besov spaces. Journal of Machine Learning Research, 24(1):1-52, 2023.

[71] Jöran Bergh and Jörgen Löfström. Interpolation spaces: an introduction, volume 223. Springer Science & Business Media, 2012.

[72] Andrei Nikolaevich Kolmogorov. On linear dimensionality of topological vector spaces. In Doklady Akademii Nauk, volume 120(2), pages 239-241. Russian Academy of Sciences, 1958.

[73] Joseph E Yukich, Maxwell B Stinchcombe, and Halbert White. Sup-norm approximation bounds for networks through probabilistic methods. IEEE Transactions on Information Theory, 41(4):1021-1027, 1995.

[74] Feng Dai. Approximation theory and harmonic analysis on spheres and balls. Springer, 2013.

[75] Elias M Stein and Guido Weiss. Introduction to Fourier analysis on Euclidean spaces, volume 1. Princeton university press, 1971.

[76] G. Szegö. Orthogonal polynomials, volume 23 of Amer. Math. Soc. Colloq. Publ. Amer. Math. Soc., Providence, 1975.

[77] Rolf Schneider. Zu einem problem von shephard über die projektionen konvexer körper. Mathematische Zeitschrift, 101:71-82, 1967.

[78] Jean Bourgain and Joram Lindenstrauss. Projection bodies. In Geometric Aspects of Functional Analysis: Israel Seminar (GAFA) 1986-87, pages 250-270. Springer, 2006.

[79] Hrushikesh Narhar Mhaskar. Weighted quadrature formulas and approximation by zonal function networks on the sphere. Journal of Complexity, 22(3):348-370, 2006.

[80] H Mhaskar, F Narcowich, and J Ward. Spherical marcinkiewicz-zygmund inequalities and positive quadrature. Mathematics of computation, 70(235):1113-1130, 2001.

[81] Andriy Bondarenko, Danylo Radchenko, and Maryna Viazovska. Optimal asymptotic bounds for spherical designs. Annals of mathematics, pages 443-452, 2013.

[82] K Jetter, J Stöckler, and JD Ward. Norming sets and spherical cubature formulas. In Advances in computational mathematics, pages 237-244. CRC Press, 2023.

[83] Sagun Chanillo and Benjamin Muckenhoupt. Weak type estimates for Cesaro sums of Jacobi polynomial series, volume 487. American Mathematical Soc., 1993.

[84] Hrushikesh N Mhaskar. Eignets for function approximation on manifolds. Applied and Computational Harmonic Analysis, 29(1):63-87, 2010.

[85] Hrushikesh N Mhaskar. Kernel-based analysis of massive data. Frontiers in Applied Mathematics and Statistics, 6:30, 2020.

[86] Elias M Stein. Singular integrals and differentiability properties of functions. Princeton university press, 1970.

[87] Jonathan W Siegel, Qingguo Hong, Xianlin Jin, Wenrui Hao, and Jinchao Xu. Greedy training algorithms for neural networks and applications to pdes. Journal of Computational Physics, 484:112084, 2023.

[88] Jinchao Xu and Xiaofeng Xu. Efficient and provably convergent randomized greedy algorithms for neural network optimization. arXiv preprint arXiv:2407.17763, 2024.

[89] Xiang Liu, Shouyi Lin, Jun Fang, and Zongben Xu. Is extreme learning machine feasible? a theoretical assessment (part 1). IEEE Transactions on Neural Networks and Learning Systems, 26(1):7-20, 2014.

[90] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision, pages 1026-1034, 2015.

[91] David Lopez-Paz, Suvrit Sra, Alex Smola, Zoubin Ghahramani, and Bernhard Schölkopf. Randomized nonlinear component analysis. In International conference on machine learning, pages 1359-1367. PMLR, 2014.

[92] Mikhail Belkin, Daniel Hsu, and Ji Xu. Two models of double descent for weak features. SIAM Journal on Mathematics of Data Science, 2(4):1167-1180, 2020.

[93] Stéphane Boucheron, Gábor Lugosi, and Olivier Bousquet. Concentration inequalities. In Summer school on machine learning, pages 208-240. Springer, 2003.

[94] Felipe Cucker and Ding Xuan Zhou. Learning theory: an approximation theory viewpoint, volume 24. Cambridge University Press, 2007.

[95] Shai Shalev-Shwartz and Shai Ben-David. Understanding machine learning: From theory to algorithms. Cambridge university press, 2014.

[96] Peter L. Bartlett and Shahar Mendelson. Rademacher and gaussian complexities: Risk bounds and structural results. Journal of Machine Learning Research, 3:463-482, 2002.

[97] Jonathan W Siegel, Qingguo Hong, Xianlin Jin, Wenrui Hao, and Jinchao Xu. Greedy training algorithms for neural networks and applications to pdes. Journal of Computational Physics, 484:112084, 2023.

[98] Mehryar Mohri, Afshin Rostamizadeh, and Ameet Talwalkar. Foundations of Machine Learning. MIT Press, 2018.

[99] Martin J. Wainwright. High-dimensional Statistics: A Non-asymptotic Viewpoint. Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, 2019.
