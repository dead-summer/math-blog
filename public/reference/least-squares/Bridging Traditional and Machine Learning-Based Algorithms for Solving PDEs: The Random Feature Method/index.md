# Bridging Traditional and Machine Learning-Based Algorithms for Solving PDEs: The Random Feature Method

Jingrun Chen \( {}^{*1,2} \), Xurong Chi \( {}^{\dagger 1} \), Weinan E \( {}^{\ddagger 3,4} \), and Zhouwang Yang \( {}^{\ddagger 1,4} \)

\( {}^{1} \) School of Mathematical Sciences, University of Science and Technology of China, P.R. China. \( {}^{2} \) Suzhou Institute for Advanced Research, University of Science and Technology of China, P.R. China.

\( {}^{3} \) AI for Science Institute, Beijing and Center for Machine Learning Research and School of Mathematical, Sciences, Peking University, P.R. China.

\( {}^{4} \) School of Data Science, University of Science and Technology of China, P.R. China.

Abstract. One of the oldest and most studied subject in scientific computing is algorithms for solving partial differential equations (PDEs). A long list of numerical methods have been proposed and successfully used for various applications. In recent years, deep learning methods have shown their superiority for high-dimensional PDEs where traditional methods fail. However, for low dimensional problems, it remains unclear whether these methods have a real advantage over traditional algorithms as a direct solver. In this work, we propose the random feature method (RFM) for solving PDEs, a natural bridge between traditional and machine learning-based algorithms. RFM is based on a combination of well-known ideas: 1. representation of the approximate solution using random feature functions; 2. collocation method to take care of the PDE; 3. penalty method to treat the boundary conditions, which allows us to treat the boundary condition and the PDE in the same footing. We find it crucial to add several additional components including multi-scale representation and adaptive weight rescaling in the loss function. We demonstrate that the method exhibits spectral accuracy and can compete with traditional solvers in terms of both accuracy and efficiency. In addition, we find that RFM is particularly suited for problems with complex geometry, where both traditional and machine learning-based algorithms encounter difficulties.

Keywords: Article Info.: Article History:

Partial differential equation, Volume: 1 Received: 26/7/2022

Machine learning, Number: 3 Accepted: 31/8/2022

Random feature method, Pages: 268- 298

Rescaling. Date: September/2022 Communicated by:

doi.org/10.4208/jml.220726 Zhi-Qin John Xu

## 1 Introduction

One of the oldest and most studied subject in scientific computing is algorithms for solving partial differential equations (PDEs). Finite difference [1], finite element [2], spectral methods [3] and a host of other methodologies have been proposed and studied, with great success. At the same time, a variety of scientific softwares based on these methodologies have been developed and widely used by the academia as well as the industry. They have become standard resources in most, if not all, engineering applications.

---

*jingrunchen@ustc.edu.cn.

†cxr123@mail.ustc.edu.cn.

\( {}^{ \ddagger  } \) weinan@math.pku.edu.cn.

§yangzw@ustc.edu.cn.

---

In recent years, as neural network models have achieved great success in a variety of artificial intelligence (AI) tasks, the idea of using these models to solve PDEs has gained a lot of popularity [4-9]. Though back in the 90's, it was already proposed to use neural networks as test or trial functions in PDE solvers [10], the recent proposals often have some non-trivial new twist. The most notable success is to solve PDEs and control problems in high dimensions [4,5,7,11], a class of problems that traditional algorithms are not able to handle. Indeed deep learning-based algorithms have now made it fairly routine to solve a large class of PDEs in hundreds of or even higher dimensions [12], something impossible to do just a few years ago. In another direction, neural networks can also be used to parametrize the solution operator of PDEs [13-15], which is also beyond the capability of traditional algorithms.

Despite the great deal of efforts and the great deal of success, the situation with solving PDEs is not entirely satisfactory even for some of the most popular engineering problems. Here is an incomplete list of some of the difficulties we still encounter.

1. Problems with complex geometry. A typical problem is Stokes flow in porous media [16]. In principle the finite element method (FEM) is ideally suited for problems with complex geometry. In practice, coming up with a suitable mesh is often a highly non-trivial task both in terms of the human efforts and the actual computational cost required. Machine learning-based algorithms, while easy to code, have not proven to be reliable and competitive in practical situations against traditional algorithms.

2. Kinetic equations. Although its dimensionality is much lower than the high dimensional ones mentioned above, kinetic equations such as the Boltzmann equation are traditionally regarded as high dimensional problems for which classical methods do encounter difficulties. Ideas based on sparse grids should help [17,18], but at the moment the most popular approach for solving kinetic equations is still the direct simulation Monte Carlo algorithm (DSMC) [19]. One problem with DSMC is that the solutions produced contain too much noise.

3. Multi-scale problems. Examples include problems involving chemical kinetics that typically span a large range of time scales; fully developed turbulent flows that contain a large range of spatial and temporal scales; and the modeling of composite materials; see [20] for example.

Our objective in this paper is to propose a methodology for solving general PDEs that shares the merits of both traditional and machine learning-based algorithms. This new class of algorithms can be made spectrally accurate. At the same time, they are also mesh-free, making them easy to use even in settings with complex geometry. Our starting point is based on a combination of rather simple and well-known ideas: We use random feature functions to represent the approximate solution, the collocation method to take care of the PDE as well as the boundary conditions in the least-squares sense, and a rescaling procedure to balance the contributions from the PDE and the boundary conditions in the loss function. In actual implementations, we take several inspirations from the machine learning literature.

1. Our preferred choices of the basis functions are random feature functions. As a result, the method bears close similarity with the random feature model in machine learning. Though deterministic basis functions can also work, we have found that random feature functions are generally more reliable and more efficient. If necessary, the actual choice of the basis functions can be tuned beforehand in a precomputing stage in order to make them more adapted to the nature of the problem. For this reason, we will name the class of algorithms proposed here the "random feature method (RFM)".

The feature (or basis) functions adopted here are the ones coming from neural networks (for details, see below). This means that we are using a special class of random feature models: models that come from two-layer neural networks with the inner parameters fixed. As a comparison, if we use radial basis functions as the feature functions, we are uncertain whether the performance of the RFM would be equally good. We also find that the additional gain in training the inner parameters does not offset at all the increased complexity in the training.

Another important component is a multi-scale representation of the solutions. We use a partition of unity (PoU) to piece together different local representations as well as a global representation for the large scale components of the solution. This strategy has proven to be vital in practice in order to achieve good accuracy.

2. A rescaling procedure is needed to balance the contributions from the PDE and the boundary/initial conditions in the loss function, by tuning the weight parameters ( \( \left\{  {\lambda }_{Ii}^{k}\right\} \) and \( \left\{  {\lambda }_{Bj}^{\ell }\right\} \) in (2.2)). Although the situation is similar to the one in training neural network models, the reduced complexity from using the random feature model instead of the neural network model seems to make the task of parameter tuning much simpler. In fact, this rescaling procedure is quite simple and fully automatic.

A closely related method is the local extreme learning machine (locELM) proposed in [21], using the combination of extreme learning machines [22] and domain decomposition. The idea of domain decomposition can be viewed as a special choice of PoU. In that case, some smoothness conditions have to be enforced explicitly across the boundary of the different domains. For a series of one-dimensional and two-dimensional problems with simple geometries and explicit solutions, this method shows spectral accuracy. However, it does not seem to work well for more practical problems such as the linear elasticity problem, even with simple geometry.

This paper is organized as follows. In Section 2, we present the RFM: the construction of approximate solution, the loss function, and the optimization procedure. This is followed by results of two sets of numerical experiments: one with explicit solutions and the other with complex geometries. We use the former to demonstrate that RFM has spectral accuracy, and the latter to demonstrate its feasibility for solving complex problems. These results are shown in Section 3. In Section 4, we present some discussions.

## 2 The random feature method

Consider the following problem

\[\left\{  \begin{array}{ll} \mathcal{L}\mathbf{u}\left( \mathbf{x}\right)  = \mathbf{f}\left( \mathbf{x}\right), & \mathbf{x} \in  \Omega, \\  \mathcal{B}\mathbf{u}\left( \mathbf{x}\right)  = \mathbf{g}\left( \mathbf{x}\right), & \mathbf{x} \in  \partial \Omega, \end{array}\right. \tag{2.1}\]

where \( \mathbf{x} = {\left( {x}_{1},\cdots,{x}_{d}\right) }^{T} \), and \( \Omega \) is bounded and connected domain in \( {\mathbb{R}}^{d} \). Examples include the elliptic problem, the linear elasticity problem, and the Stokes flow problem.

Roughly speaking, much like the random feature model in machine learning, RFM relies on three key components: 1. The loss function is built on the least-squares (strong) formulation of the PDEs on collocation points; 2. The approximate solution is constructed using a set of random feature functions; 3. The training is very much like neural network training, with the additional step of rescaling the penalty parameters to balance the contributions from different terms. In what follows, we will discuss each component in some detail.

### 2.1 Loss function

There are three standard approaches for solving (2.1): the weak form, the strong form and the variational form in cases when (2.1) is the Euler-Lagrange equation of some variational problem. Each of these approaches gives rise to some particular choices of loss function. For neural network-based algorithms, examples of these different loss functions can be found in [6, 8, 9]. In this paper, we will focus on the strong form at collocation points to construct the loss function. Corresponding to (2.1), we have two sets of collocation points: \( {C}_{I} \), the set of interior points in \( \Omega \) and \( {C}_{B} \), the set of boundary points on \( \partial \Omega \). Let \( C = {C}_{I} \cup  {C}_{B} \) be the set of all collocation points. At each collocation point, we will enforce either the PDE or the boundary condition. Let \( {K}_{I} \) and \( {K}_{B} \) be the number of conditions at each interior point and boundary point, respectively. The total number of conditions is \( N = {K}_{I}\# {C}_{I} + {K}_{B}\# {C}_{B} \). See Fig. 2.1 for an illustration. Detailed selection algorithm for \( C \) will be specified later.

A simple choice of the loss function for (2.1) is as follows:

\[\text{ Loss } = \mathop{\sum }\limits_{{{x}_{i} \in  {C}_{I}}}\mathop{\sum }\limits_{{k = 1}}^{{K}_{I}}{\lambda }_{Ii}^{k}{\begin{Vmatrix}{\mathcal{L}}^{k}\mathbf{u}\left( {\mathbf{x}}_{i}\right)  - {\mathbf{f}}^{k}\left( {\mathbf{x}}_{i}\right) \end{Vmatrix}}_{{l}^{2}}^{2} + \mathop{\sum }\limits_{{{\mathbf{x}}_{j} \in  {C}_{B}}}\mathop{\sum }\limits_{{\ell  = 1}}^{{K}_{B}}{\lambda }_{Bj}^{\ell }{\begin{Vmatrix}{\mathcal{B}}^{\ell }\mathbf{u}\left( {\mathbf{x}}_{j}\right)  - {\mathbf{g}}^{\ell }\left( {\mathbf{x}}_{j}\right) \end{Vmatrix}}_{{l}^{2}}^{2}. \tag{2.2}\]

Here \( \left\{  {\lambda }_{Ii}^{k}\right\} \) and \( \left\{  {\lambda }_{Bj}^{\ell }\right\} \) are the penalty parameters. In this form, we allow different choices of the penalty parameters at different collocation points. By treating the boundary conditions and the PDE in the same footing, we do not need to impose boundary conditions for the feature function. This gives us much needed flexibility for treating problems with complex geometry.

![Figure 2.1](images/figure-2-1.png)

Figure 2.1: Collocation points for a square domain: \( {C}_{I} \), interior points in orange and blue; \( {C}_{B} \), boundary points in green.

### 2.2 Random feature functions

Following the random feature model in machine learning, we construct the approximate solution \( {u}_{M} \) of \( u \) by a linear combination of \( M \) network basis functions \( \left\{  {\phi }_{m}\right\} \) over \( \Omega \) as follows

\[{u}_{M}\left( \mathbf{x}\right)  = \mathop{\sum }\limits_{{m = 1}}^{M}{u}_{m}{\phi }_{m}\left( \mathbf{x}\right). \tag{2.3}\]

For vectorial solutions, we approximate each component of the solution using (2.3), i.e.

\[{\mathbf{u}}_{M}\left( \mathbf{x}\right)  = {\left( \mathop{\sum }\limits_{{m = 1}}^{M}{u}_{m}^{1}{\phi }_{m}^{1}\left( \mathbf{x}\right),\cdots,\mathop{\sum }\limits_{{m = 1}}^{M}{u}_{m}^{{K}_{I}}{\phi }_{m}^{{K}_{I}}\left( \mathbf{x}\right) \right) }^{T}.\]

Generally speaking the basis functions will be chosen as the ones that occur naturally in neural networks, for example:

\[{\phi }_{m}\left( \mathbf{x}\right)  = \sigma \left( {{\mathbf{k}}_{m} \cdot  \mathbf{x} + {b}_{m}}\right),\]

where \( \sigma \) is some scalar nonlinear function, \( {\mathbf{k}}_{m},{b}_{m} \) are some random but fixed parameters. For solving PDE problems, activation functions such as tanh, sin, and cos can all be used.

In practice, additional ideas are needed to achieve good performance.

#### 2.2.1 Partition of unity and local random feature models

Random feature functions are globally defined, while the solution of the PDE typically has local variations, possibly at small scales. To accommodate this, we construct many local solutions, each of which corresponds to a random feature model, and piece them together using partition of unity (PoU).

![Figure 2.2](images/figure-2-2.png)

Figure 2.2: Visualization of \( {\psi }^{a}\left( x\right) \) in 2.6 and \( {\psi }^{b}\left( x\right) \) in 2.7

To construct the PoU, we start with a set of points \( {\left\{  {\mathbf{x}}_{n}\right\}  }_{n = 1}^{{M}_{p}} \subset  \Omega \), each of which serves as the center for a component in the partition. For each \( n \), construct the normalized coordinate:

\[\widetilde{\mathbf{x}} = \frac{1}{{\mathbf{r}}_{n}}\left( {\mathbf{x} - {\mathbf{x}}_{n}}\right),\; n = 1,\cdots,{M}_{p}, \tag{2.4}\]

where \( {\mathbf{r}}_{n} = \left( {{r}_{n1},{r}_{n2},\cdots,{r}_{nd}}\right) \) and \( \left\{  {\mathbf{r}}_{n}\right\} \) is preselected. This linear transformation maps \( \left\lbrack  {{x}_{n1} - {r}_{n1},{x}_{n1} + {r}_{n1}}\right\rbrack   \times  \cdots  \times  \left\lbrack  {{x}_{nd} - {r}_{nd},{x}_{nd} + {r}_{nd}}\right\rbrack \) onto \( {\left\lbrack  -1,1\right\rbrack  }^{d} \).

Next, we construct \( {J}_{n} \) random feature functions by

\[{\phi }_{nj}\left( \mathbf{x}\right)  = \sigma \left( {{\mathbf{k}}_{nj} \cdot  \widetilde{\mathbf{x}} + {b}_{nj}}\right),\; j = 1,\cdots,{J}_{n}, \tag{2.5}\]

where the feature vectors \( \left\{  \left( {{\mathbf{k}}_{nj},{b}_{nj}}\right) \right\} \) often chosen randomly. A common choice is the uniform distribution \( {\mathbf{k}}_{nj} \sim  \mathbb{U}\left( {\left\lbrack  -{R}_{nj},{R}_{nj}\right\rbrack  }^{d}\right) \) and \( {b}_{nj} \sim  \mathbb{U}\left( \left\lbrack  {-{R}_{nj},{R}_{nj}}\right\rbrack  \right) \), though different distributions can be used. In this way the locally space-dependent information is incorporated into \( M = \mathop{\sum }\limits_{{n = 1}}^{{M}_{p}}{J}_{n} \) random feature functions.

We now discuss the construction of the PoU. When \( d = 1 \), let

\[{\psi }_{n}^{a}\left( x\right)  = {\mathbb{I}}_{-1 \leq  \widetilde{x} < 1} \tag{2.6}\]

and

\[{\psi }_{n}^{b}\left( x\right)  = \left\{  \begin{array}{ll} \frac{1 + \sin \left( {{2\pi }\widetilde{x}}\right) }{2}, &  - \frac{5}{4} \leq  \widetilde{x} <  - \frac{3}{4}, \\  1, &  - \frac{3}{4} \leq  \widetilde{x} < \frac{3}{4}, \\  \frac{1 - \sin \left( {{2\pi }\widetilde{x}}\right) }{2}, & \frac{3}{4} \leq  \widetilde{x} < \frac{5}{4}, \\  0, & \text{ otherwise. } \end{array}\right. \tag{2.7}\]

See Fig. 2.2 for the visualization of \( {\psi }^{a} \) and \( {\psi }^{b} \).

High-dimensional PoU can be constructed using the tensor product of one-dimensional PoU functions \( {\psi }_{n}\left( \mathbf{x}\right)  = \mathop{\prod }\limits_{{k = 1}}^{d}{\psi }_{n}\left( {x}_{k}\right) \).

Putting together, the approximate solution \( {u}_{M} \) in (2.3) is given by

\[{u}_{M}\left( \mathbf{x}\right)  = \mathop{\sum }\limits_{{n = 1}}^{{M}_{p}}{\psi }_{n}\left( \mathbf{x}\right) \mathop{\sum }\limits_{{j = 1}}^{{J}_{n}}{u}_{nj}{\phi }_{nj}\left( \mathbf{x}\right). \tag{2.8}\]

#### 2.2.2 Multi-scale basis

In some situations, (2.8) alone is less efficient in capturing the large scale features in the solution. Therefore, on top of the PoU-based local basis functions, we can add another global component:

\[{u}_{M}\left( \mathbf{x}\right)  = {u}_{g}\left( \mathbf{x}\right)  + \mathop{\sum }\limits_{{n = 1}}^{{M}_{p}}{\psi }_{n}\left( \mathbf{x}\right) \mathop{\sum }\limits_{{j = 1}}^{{J}_{n}}{u}_{nj}{\phi }_{nj}\left( \mathbf{x}\right), \tag{2.9}\]

where \( {u}_{g} \) is a global random feature function; see (2.3).

#### 2.2.3 Adaptive basis

The ideal choice of the distribution for the feature vectors is one that reflects the spectral distribution of the solution, which is not available to us beforehand. In some situations, we can obtain some incomplete information about the spectral distribution of the solution in the precomputing stage. For example, if the PDE has an inhomogeneous forcing term, we can perform a spectral analysis of the forcing term. The result can be used to guide us in the selection of the spectral distribution of the feature vectors. We have seen that this is particularly useful when sin/cos is used as the activation function.

Such a procedure can be seen as a compromise between the random feature method and the two-layer neural network model. In the two-layer neural network model, the inner parameters, i.e. the feature vectors, are part of the parameter set to be optimized. This in principle allows us to obtain an optimal choice of the feature vectors. The price we pay is that the optimization problem is much more complicated than simply fixing the feature vectors. In the random feature model, the feature vectors are fixed. If we choose a wrong set of feature vectors, the accuracy will deteriorate. With an adaptive procedure, one might be able to avoid this and at the same time still retain the simplicity of a linear model.

### 2.3 Optimization

Recall the loss function:

\[\text{ Loss } = \mathop{\sum }\limits_{{{\mathbf{x}}_{i} \in  {C}_{I}}}\mathop{\sum }\limits_{{k = 1}}^{{K}_{I}}{\lambda }_{Ii}^{k}{\begin{Vmatrix}{\mathcal{L}}^{k}{\mathbf{u}}_{M}\left( {\mathbf{x}}_{i}\right)  - {\mathbf{f}}^{k}\left( {\mathbf{x}}_{i}\right) \end{Vmatrix}}_{{l}^{2}}^{2} + \mathop{\sum }\limits_{{{\mathbf{x}}_{j} \in  {C}_{B}}}\mathop{\sum }\limits_{{\ell  = 1}}^{{K}_{B}}{\lambda }_{Bj}^{\ell }{\begin{Vmatrix}{\mathcal{B}}^{\ell }{\mathbf{u}}_{M}\left( {\mathbf{x}}_{j}\right)  - {\mathbf{g}}^{\ell }\left( {\mathbf{x}}_{j}\right) \end{Vmatrix}}_{{l}^{2}}^{2}, \tag{2.10}\]

where

\[{\mathbf{u}}_{M}\left( \mathbf{x}\right)  = {\left( \mathop{\sum }\limits_{{n = 1}}^{{M}_{p}}{\psi }_{n}\left( \mathbf{x}\right) \mathop{\sum }\limits_{{{j}^{\prime } = 1}}^{{J}_{n}}{u}_{n{j}^{\prime }}^{1}{\phi }_{n{j}^{\prime }}^{1}\left( \mathbf{x}\right),\cdots,\mathop{\sum }\limits_{{n = 1}}^{{M}_{p}}{\psi }_{n}\left( \mathbf{x}\right) \mathop{\sum }\limits_{{{j}^{\prime } = 1}}^{{J}_{n}}{u}_{n{j}^{\prime }}^{{K}_{I}}{\phi }_{n{j}^{\prime }}^{{K}_{I}}\left( \mathbf{x}\right) \right) }^{T}.\]

This optimization problem can be solved using standard algorithms for least-squares approximation. One important new twist is the tuning of the penalty parameters. One simple yet effective way is to rescale the penalty parameters so that each term in the loss function is of the same order of magnitude. This can be done using the largest term in the loss function as the reference. The detailed formula will be shown in the next section. This is particularly important in situations where the physical constants in the PDEs are of disparate size.

### 2.4 Collocation points

There are many existing collocation point sampling methods for general geometric representations. If the boundary has an explicit parametric representation, the collocation points can be chosen as uniform grid points in the parameter space. In the case when the boundary has an implicit geometric representation, we can easily identify the interior points and define an energy function for finding a point on the boundary. In the implementations presented below, we uniformly sample \( Q \) points over a rectangle \( R \) containing \( \Omega \) and delete points in \( R \cap  {\Omega }^{c} \). The construction of collocation points over a two-dimensional rectangular domain \( \Omega \) is illustrated in Fig. 2.1.

We summarize the main steps of RFM in Algorithm 1.

Algorithm 1 The random feature method.

---

Input: Number of basis functions \( M \); number of collocation points \( Q \); rule for generating collocation points.

Output: The approximate solution \( {u}_{M} \).

1: Construct \( M \) random feature functions \( \left\{  {\phi }_{m}\right\} \) and the PoU \( \left\{  {\psi }_{n}\right\} \).

2: Sample points \( C = {C}_{I} \cup  {C}_{B} \) according to some predetermined rule.

3: Evaluate equations at \( {C}_{I} \) and boundary conditions at \( {C}_{B} \); construct the loss function (2.10) (\( M \) is not necessarily equal to \( N \)).

4: Solve the optimization problem.

5: Return \( {u}_{M} \).

---

## 3 Numerical results

We report results for two kinds of situations: We use problems with explicit solutions to study how the performance of RFM depends on the different components in the algorithms such as the choice of the basis functions. We also present results for problems that do not have explicit solutions to demonstrate the power of RFM in complicated situations.

The examples discussed below are all second-order PDEs. We need \( {C}^{1} \) smoothness for the approximate solution. If we use \( \left\{  {\psi }^{a}\right\} \), we impose this smoothness condition explicitly on the collocation points at the interfaces of the elements in the partition. If we use \( \left\{  {\psi }^{b}\right\} \), no additional smoothness conditions need to be imposed.

In what follows, unless indicated otherwise, we use the default setup where the collocation points are uniformly distributed on \( \Omega \), the weights \( \left\{  {k}_{m}\right\} \) and \( \left\{  {b}_{m}\right\} \) are assumed to follow the distribution \( \mathbb{U}\left\lbrack  {-1,1}\right\rbrack \), the activation function is chosen to be tanh, the PoU is \( \left\{  {\psi }^{a}\right\} \). To achieve good accuracy, we find that the distribution of \( \left\{  {k}_{m}\right\} \) and \( \left\{  {b}_{m}\right\} \) should be weakly problem-dependent when sin and/or cos is used as the activation function. \( {R}_{m} = 1 \) works well for the examples discussed if tanh is used as the activation function. We first select a set of points \( {\left\{  {x}_{n}\right\}  }_{n = 1}^{{M}_{p}} \) and construct the PoU as follows. For each \( {x}_{n} \), we construct \( {J}_{n} \) random feature functions with radius \( {\mathbf{r}}_{n} \). Then we sample \( Q \) equally spaced collocation points. The ones that are outside \( \Omega \) are deleted.

When evaluating the error, we take a refined grid with grid size being half of the size for the collocation points. The errors are evaluated on this refined grid.

We start with some simple examples and use them to study how the performance of RFM, particularly the accuracy, depends on the details of algorithm.

### 3.1 Choice of random feature functions

Example 3.1 (Helmholtz equation). Consider the one-dimensional Helmholtz equation with Dirichlet boundary condition over \( \Omega  = \left\lbrack  {0,8}\right\rbrack \)

\[\left\{  \begin{array}{l} \frac{{\mathrm{d}}^{2}u\left( x\right) }{\mathrm{d}{x}^{2}} - {\lambda u}\left( x\right)  = f\left( x\right),\; x \in  \Omega, \\  u\left( 0\right)  = {c}_{1},\; u\left( 8\right)  = {c}_{2}. \end{array}\right. \tag{3.1}\]

Once an explicit form of \( u \) is given, \( {c}_{1},{c}_{2} \), and \( f \) can be computed.

Example 3.2 (Poisson equation). Consider the Poisson equation with Dirichlet boundary condition over \( \Omega  = \left\lbrack  {0,1}\right\rbrack   \times  \left\lbrack  {0,1}\right\rbrack \)

\[\left\{  \begin{array}{l} {\Delta u}\left( {x, y}\right)  = f\left( {x, y}\right),\;\left( {x, y}\right)  \in  \Omega, \\  u\left( {x,0}\right)  = {g}_{1}\left( x\right),\; u\left( {x,1}\right)  = {g}_{2}\left( x\right), \\  u\left( {0, y}\right)  = {h}_{1}\left( y\right),\; u\left( {1, y}\right)  = {h}_{2}\left( y\right). \end{array}\right. \tag{3.2}\]

Again once an explicit form of \( u \) is given, \( {g}_{1},{g}_{2},{h}_{1},{h}_{2} \), and \( f \) can be computed.

Detailed numerical results for these problems can be found in Appendix A. Here we briefly summarize the main findings.

We find that as long as the support of the distribution for the weights approximately covers the frequency domain of the true solution, RFM produces stable results using sin/cos as the activation functions. In addition, we observe that in most cases, random sampling performs better than deterministic choices of the feature vectors.

In addition, the introduction of multi-scale basis functions in (2.9) improves the accuracy. A Fourier analysis of the error confirms that the use of multi-scale basis functions reduce the low-frequency error more effectively.

We see also that with \( {\psi }^{a} \), the error is more concentrated around the interface between the macro-elements in the partition. With \( {\psi }^{b} \), the error tends to be concentrated at the boundary.

One point of interest is the comparison between RFM and PINN. We observe that the accuracy of PINN is around \( {1E} - 3 \). Increasing the network size does not seem to improve the accuracy. In contrast, we observe exponential rate of convergence for RFM in terms of the number of random feature functions.

### 3.2 Rescaling

An important consideration is the balance between contributions from the PDE terms and the boundary conditions in the loss function. This requires tuning the weights of different terms. In this subsection, we consider the elasticity problem when \( d = 2 \) and demonstrate how the rescaling strategy works. The idea is to rescale each term in the loss function to the same order of magnitude according to the largest term in the sum. Specifically, we choose the penalty parameters in (2.10) as follows:

\[{\lambda }_{Ii}^{k} = \frac{c}{\mathop{\max }\limits_{{1 \leq  n \leq  {M}_{p}}}\mathop{\max }\limits_{{1 \leq  {j}^{\prime } \leq  {J}_{n}}}\mathop{\max }\limits_{{1 \leq  {k}^{\prime } \leq  {K}_{I}}}\left| {{\mathcal{L}}^{k}\left( {{\phi }_{n{j}^{\prime }}^{{k}^{\prime }}\left( {\mathbf{x}}_{i}\right) {\psi }_{n}\left( {\mathbf{x}}_{i}\right) }\right) }\right| },\;{\mathbf{x}}_{i} \in  {C}_{I},\; k = 1,\cdots,{K}_{I}, \tag{3.3}\]

\[{\lambda }_{Bj}^{\ell } = \frac{c}{\mathop{\max }\limits_{{1 \leq  n \leq  {M}_{p}}}\mathop{\max }\limits_{{1 \leq  {j}^{\prime } \leq  {J}_{n}}}\mathop{\max }\limits_{{1 \leq  {\ell }^{\prime } \leq  {K}_{I}}}\left| {{\mathcal{B}}^{\ell }\left( {{\phi }_{n{j}^{\prime }}^{{\ell }^{\prime }}\left( {\mathbf{x}}_{j}\right) {\psi }_{n}\left( {\mathbf{x}}_{j}\right) }\right) }\right| },\;{\mathbf{x}}_{j} \in  {C}_{B},\ell  = 1,\cdots,{K}_{B}, \tag{3.4}\]

where \( c \) is a universal constant and we set \( c = {100} \).

We will see that this simple strategy significantly improves the accuracy, particularly in situations when the physical constants in the PDE are of disparate size.

The two-dimensional elasticity problem we consider here is of the following form

\[\left\{  \begin{array}{ll}  - \operatorname{div}\left( {\sigma \left( {\mathbf{u}\left( \mathbf{x}\right) }\right) }\right)  = \mathbf{B}\left( \mathbf{x}\right), & \mathbf{x} \in  \Omega, \\  \sigma \left( {\mathbf{u}\left( \mathbf{x}\right) }\right)  \cdot  \mathbf{n} = \mathbf{N}\left( \mathbf{x}\right), & \mathbf{x} \in  {\Gamma }_{N}, \\  \mathbf{u}\left( \mathbf{x}\right)  \cdot  \mathbf{n} = \mathbf{U}\left( \mathbf{x}\right), & \mathbf{x} \in  {\Gamma }_{D}, \end{array}\right. \tag{3.5}\]

where \( \sigma : {\mathbb{R}}^{2} \rightarrow  {\mathbb{R}}^{2} \) is the stress tensor induced by the displacement field \( \mathbf{u}: \Omega  \rightarrow  {\mathbb{R}}^{2},\mathbf{B} \) is the body force over \( \Omega, N \) is the surface force on \( {\Gamma }_{N},\mathbf{U} \) is the displacement on \( {\Gamma }_{D} \), and \( \partial \Omega  = {\Gamma }_{N} \cup  {\Gamma }_{D} \)

Following [23], we consider the Timoshenko beam problem with size \( L \times  D \), subject to a parabolic traction at the free end as shown in Fig. 3.1. The exact solution and the experimental setup are presented in Appendix B.1.

Let us first look at the accuracy of RFM and locELM. To this end, we presented in Table 3.1 the relative error. Since \( {\sigma }_{y} \) is zero everywhere, we omit this term. From Table 3.1, we observe that the relative error of locELM is around \( {1E} - 3 \), while RFM still has spectral accuracy. We attribute the improved performance of RFM to the rescaling strategy.

Next, we study a two-dimensional elasticity problem with a complex geometry; see Fig. 3.2. Here \( \Omega \) is defined as a square \( \left( {0,8}\right)  \times  \left( {0,8}\right) \) with 40 holes of radius between 0.3 and 0.6] inside. Note that there is a cluster of holes that are nearly touching, as shown in the inset. More details can be found in Appendix B.1.

In this example, the error produced by locELM is around \( {10}^{-3} \sim  {10}^{-2} \), while RFM still maintains spectral accuracy, as is shown in Table B.1, Appendix B.1.

![Figure 3.1](images/figure-3-1.png)

Figure 3.1: The Timoshenko beam problem.

Table 3.1: Comparison of RFM and locELM for the Timoshenko beam problem.

<table><tr><td>Method</td><td>\( M \)</td><td>\( N \)</td><td>\( u \) error</td><td>\( v \) error</td><td>\( {\sigma }_{x} \) error</td><td>\( {\tau }_{xy} \) error</td></tr><tr><td rowspan="8">RFM</td><td rowspan="4">800</td><td>400</td><td>1.36E-2</td><td>3.43E-3</td><td>1.40E-2</td><td>1.63E-2</td></tr><tr><td>1200</td><td>7.14E-6</td><td>7.98E-7</td><td>8.93E-6</td><td>7.45E-6</td></tr><tr><td>4000</td><td>6.41E-11</td><td>4.34E-11</td><td>6.41E-11</td><td>6.58E-11</td></tr><tr><td>14400</td><td>8.16E-12</td><td>1.01E-12</td><td>1.07E-11</td><td>1.03E-11</td></tr><tr><td rowspan="4">3200</td><td>1680</td><td>1.02E-2</td><td>1.42E-3</td><td>1.13E-2</td><td>7.65E-3</td></tr><tr><td>4960</td><td>4.51E-6</td><td>7.89E-7</td><td>4.98E-6</td><td>4.36E-6</td></tr><tr><td>16320</td><td>1.22E-11</td><td>7.23E-12</td><td>1.56E-11</td><td>1.40E-11</td></tr><tr><td>58240</td><td>5.17E-13</td><td>1.49E-13</td><td>1.47E-12</td><td>1.99E-11</td></tr><tr><td rowspan="8">locELM</td><td rowspan="4">800</td><td>400</td><td>5.22E-3</td><td>4.90E-3</td><td>1.33E-2</td><td>2.39E-2</td></tr><tr><td>1200</td><td>1.55E-4</td><td>5.25E-5</td><td>1.44E-4</td><td>1.02E-4</td></tr><tr><td>4000</td><td>6.36E-4</td><td>3.47E-4</td><td>6.55E-4</td><td>7.26E-4</td></tr><tr><td>14400</td><td>1.76E-3</td><td>1.64E-3</td><td>1.93E-3</td><td>2.57E-3</td></tr><tr><td rowspan="4">3200</td><td>1680</td><td>8.50E-2</td><td>4.04E-2</td><td>7.72E-2</td><td>4.19E-2</td></tr><tr><td>4960</td><td>1.32E-5</td><td>6.19E-6</td><td>3.25E-5</td><td>4.22E-5</td></tr><tr><td>16320</td><td>1.33E-3</td><td>1.12E-3</td><td>1.31E-3</td><td>1.04E-3</td></tr><tr><td>58240</td><td>6.42E-4</td><td>1.91E-4</td><td>1.18E-3</td><td>1.38E-3</td></tr></table>

### 3.3 Comparison with FEM

In this subsection, we compare RFM with the classical adaptive FEM for two elasticity problems.

The domain for the first example is given by a square \( \left( {-1,1}\right)  \times  \left( {-{0.5},{0.5}}\right) \) jointed by a semi-disk centered at \( \left( {{1.0},{0.0}}\right) \) with radius 0.5, with two disks centered at \( \left( {{1.2},{0.0}}\right),\left( {-{0.5},{0.0}}\right) \) with radius 0.2 removed (see Fig. 3.3). The material parameters are the same as those in Section 3.2. The left boundary \( x =  - {0.5} \) is fixed and a load \( P = {10}^{7} \) Pa is applied on the upper half of the semicircle. Dirichlet boundary condition is applied on the left boundary \( x =  - {0.5} \) and Neumann boundary condition is applied on the other boundaries.

![Figure 3.2](images/figure-3-2.png)

Figure 3.2: A two-dimensional complex domain.

![Figure 3.3](images/figure-3-3.png)

Figure 3.3: Numerical solution by the random feature method for the elasticity problem.

More details of the experimental setup can be found in Appendix B.2. Fig. 3.3 visualizes the displacement fields \( u, v \), and the stress fields \( {\sigma }_{x},{\tau }_{xy},{\sigma }_{y} \). For this example, it is quite straightforward to obtain a FEM solution. From Table B.2 in Appendix B.2, we see that the difference between the RFM and FEM solutions is about 1%.

For the second example, we use the same domain as in Fig. 3.2, Section 3.2 and the same materials constants. The lower boundary \( y = 0 \) is fixed, and a load of \( {10}^{5}\sin \left( {x + y}\right) {e}^{y}\mathrm{\;{Pa}} \) along the positive \( x \) direction is applied on both the left and right boundaries. More details of the experimental setup can be found in Appendix B.2.

Fig. 3.4 visualizes the displacement fields \( u, v \), and the stress fields \( {\sigma }_{x},{\tau }_{xy},{\sigma }_{y} \). For this example, it is quite difficult to generate a mesh for the FEM. If we simply remove the cluster in the lower right corner, we incur an \( {L}^{\infty } \) error of about \( {50}\% \) for \( {\sigma }_{x} \); see Fig. 3.4(c) and 3.4(d). In contrast, it is quite straightforward to use RFM to solve such a problem. As recorded in Table B.3 in Appendix B.2. RFM shows a clear trend of numerical convergence. The error against the reference solution for displacements and stresses is reduced to about 5%.

### 3.4 Homogenization

In this subsection, we take a preliminary look at how RFM performs for problems with multi-scale solutions. We consider the elliptic equation over the unit disk

\[\left\{  \begin{array}{ll}  - \operatorname{div}\left( {a\left( \mathbf{x}\right) \nabla u\left( \mathbf{x}\right) }\right)  = f\left( \mathbf{x}\right), & \mathbf{x} \in  \Omega, \\  u\left( \mathbf{x}\right)  = 0, & \mathbf{x} \in  \partial \Omega, \end{array}\right. \tag{3.6}\]

where \( a\left( \mathbf{x}\right)  = {e}^{h\left( \mathbf{x}\right) }, h\left( \mathbf{x}\right)  = \mathop{\sum }\limits_{{\left| \mathbf{k}\right|  \leq  R}}\left( {{a}_{\mathbf{k}}\sin \left( {{2\pi }\mathbf{k} \cdot  \mathbf{x}}\right)  + {b}_{\mathbf{k}}\cos \left( {{2\pi }\mathbf{k} \cdot  \mathbf{x}}\right) }\right), R = 6 \), and \( \left\{  {a}_{k}\right\} \) and \( \left\{  {b}_{k}\right\} \) are independent, identically distributed random variables with the distribution \( \mathbb{U}\left\lbrack  {-{0.3},{0.3}}\right\rbrack \). This is chosen so that there is no clear scale separation in the coefficient [24].

More details of the experimental setup can be found in Appendix B.3 Fig. 3.5 visualizes the coefficient functions \( h \) and \( a \), the numerical solution and its first-order derivatives obtained by RFM. Table B.4 in Appendix B.3 records the convergence behavior of RFM when the solution with \( N = {86219} \) is taken as the reference.

### 3.5 Stokes flow

Consider Stokes flow defined by

\[\left\{  \begin{array}{ll}  - \Delta \mathbf{u}\left( \mathbf{x}\right)  + \nabla p\left( \mathbf{x}\right)  = \mathbf{f}\left( \mathbf{x}\right), & \mathbf{x} \in  \Omega, \\  \nabla  \cdot  \mathbf{u}\left( \mathbf{x}\right)  = 0, & \mathbf{x} \in  \Omega, \\  \mathbf{u}\left( \mathbf{x}\right)  = \mathbf{U}\left( \mathbf{x}\right), & \mathbf{x} \in  \partial \Omega. \end{array}\right. \tag{3.7}\]

In this problem, \( p \) is only determined up to a constant. To avoid difficulties, we fix the value of \( p \) at the left-bottom corner.

One problem with spectral methods is that spurious pressure mode arises due to the rank deficiency of the discrete systems [25]. One interesting feature of RFM is that it always looks for an optimal solution with minimal norm. This allows us to automatically bypass the issue of rank deficiency, as we see in the following examples.

First, we consider (3.7) with an explicit solution and inhomogeneous boundary condition, where \( \Omega \) is the square \( \left( {0,1}\right)  \times  \left( {0,1}\right) \) with three holes centered at \( \left( {{0.5},{0.2}}\right),\left( {{0.2},{0.8}}\right) \), \( \left( {{0.8},{0.8}}\right) \) of radius 0.1. The exact displacement fields and the experimental setup are detailed in Appendix B.4. Table B.5 in Appendix B.4 records the convergence behavior of RFM and spectral accuracy is observed for \( u, v \) as well as \( p \).

![Figure 3.4](images/figure-3-4.png)

Figure 3.4: Numerical solution by the random feature method for the two-dimensional elasticity problem over a complex geometry.

![Figure 3.5](images/figure-3-5.png)

Figure 3.5: Coefficient functions \( h \) and \( a \), the numerical solution and its first-order derivatives obtained by the random feature method for the homogenization problem.

Next, we consider two-dimensional channel flows for four sets of complex obstacles with the inhomogeneous boundary condition

\[{\left. \left( u, v\right) \right| }_{\partial \Omega } = \left\{  \begin{array}{ll} \left( {y\left( {1 - y}\right),0}\right), & \text{ if }x = 0, \\  \left( {y\left( {1 - y}\right),0}\right), & \text{ if }x = 1, \\  \left( {0,0}\right), & \text{ otherwise. } \end{array}\right. \tag{3.8}\]

The pressure diagram is plotted in Fig. 3.6, and the velocity diagram for a complex domain is plotted in Fig. 3.7. Numerical convergence is observed for all four examples and the difference with respect to the reference solution is about \( 1\% \) for \( u, v, p \). The difference is about 0.01% for uniformly distributed holes.

![Figure 3.6](images/figure-3-6.png)

Figure 3.6: Pressure diagram generated by the random feature method for four sets of complex obstacles.

![Figure 3.7](images/figure-3-7.png)

Figure 3.7: Velocity field \( \left( {u, v}\right) \) generated by the random feature method.

## 4 Discussions

One main disadvantage of the traditional algorithms is that they are not flexible enough. This lack of flexibility is reflected in several aspects. For example, most traditional algorithms require that the number of free parameters be the same as the number of conditions. Spectral methods often require that the basis functions satisfy particular boundary conditions, and that the basis functions are constructed in a tensor-product form.

Neural network-based methods such as Deep BSDE [4], Deep Ritz [6] and PINN [8] do not have these problems. However, they typically lack robustness. With these methods, it is often quite easy to get a roughly accurate solution, but very hard to systematically improve the accuracy. As a result, one is only willing to use them when traditional algorithms fail to work.

The random feature method introduced here seems to have both the flexibility and robustness needed. There are two major differences of RFM compared with neural network-based methods such as Deep BSDE, Deep Ritz or PINN. The first is that we use only the random feature model instead of the neural network model to represent the solution. We find that for these low dimensional problems, there is not much gain in terms of the representative power by switching to neural networks, yet the non-convexity introduced in the loss function makes the task of training much harder. The second is that we use a multi-scale representation instead of a single global representation.

Compared with traditional algorithms, an important difference is that RFM typically works in the situation where the number of unknown parameters (say \( M \) ) is different from the number of conditions (say \( N \) ). This forces us to use a least square framework, which increases the complexity of the training process since the condition number is now much bigger. In return, it allows us to obtain reasonable solutions with much lower human and computer cost. For example, in the complex geometry problems treated earlier, we need a large value of \( N \) to resolve the geometry. If we used the same value of \( M \), the cost would be too big in our current implementation due to the usage of scipy in Python.

Another important difference compared with traditional algorithms is that the boundary condition is treated in the same fashion as the PDE. In particular, we do not force the basis functions to satisfy particular boundary conditions. This increased flexibility is vital for the success on problems with complex geometry.

A third important difference is the adoption of neural network-like basis functions instead of traditional tensor-product based basis functions. This means that the number of terms needed does not necessarily go up like \( {n}^{d} \) with \( n \) being the number of unknown parameters in each dimension and \( d \) being the dimensionality. Instead it depends entirely on the complexity of the solution. This is also part of the reason why random choices of the feature vectors is generally preferred.

RFM differs from the local extreme learning machines in the following aspects. The first is that a partition of unity is used to construct the local random feature functions instead of domain decomposition. This allows us to bypass the smoothness conditions and thus simplifies the loss function and the subsequent training. The second is the rescaling procedure. Though the difference for model problems is small, we find it crucial for practical problems of interest such as the elasticity problem or the Stokes flow problem.

Let us now turn to a discussion of the crucial components in RFM. The first is the choice of the basis functions. Here a crucial issue is the probability distribution for the feature vector. We can use pre-computing to give us a rough idea about this distribution. In practice, we find that as long as the support of the distribution approximately covers the frequency range of the true solution, RFM produces stable results with sin/cos activation functions. In addition, we observe that in most cases, random sampling performs better than deterministic choices of the feature vectors. For example, results for the Helmholtz equation in Section A.3 show that even when \( d = 1 \), in most cases, random sampling of \( k \) and \( b \) performs better than choosing the values of \( k \) and \( b \) from a uniform grid, and is at least as good in the remaining cases. This might be counterintuitive to what we have learned in classical numerical analysis that quadrature schemes are superior to the Monte-Carlo method for low-dimensional integrals (say \( d \leq  3 \) ). We are in the process of trying to quantify this finding.

The second is the choice of collocation points. Ideally we would like the collocation points to be equally distributed, both in the interior and at the boundary. This becomes non-trivial for three dimensional situations when the boundary is a surface. We are in the process of developing techniques that can help us to accomplish this.

The third technical aspect is the training. By turning to a least square formulation we may have increased the size of the condition number. There are a number of preconditioning and reformulation techniques that might be useful to alleviate this problem. In any case, it would be useful to carry out a precise numerical analysis of carefully chosen model problems to gain some insight about the convergence behavior of the training process.

## Acknowledgments

We thank Prof. Suchuan Dong for providing the one-dimensional code of local extreme learning machines and for helpful discussion. The work is supported by NSFC Major Research Plan - Interpretable and General-purpose Next-generation Artificial Intelligence, Anhui Center for Applied Mathematics, and the Major Project of Science & Technology of Anhui Province (No. 202203a05020050). J. Chen is also supported by NSFC 11971021.

## A Numerical results for different choices of random feature functions

### A.1 Partition of unity and local random feature models

The introduction of PoU generates local random feature functions and provides a more general strategy than domain decomposition and mesh generation.

Consider the following explicit solution to (3.1)

\[u\left( x\right)  = \sin \left( {{3\pi x} + \frac{3\pi }{20}}\right) \cos \left( {{2\pi x} + \frac{\pi }{10}}\right)  + 2. \tag{A.1}\]

In a series of tests, we set the hyper-parameters as follows:

Table A.1: Comparison of the RFM and PINN for the one-dimensional Helmholtz equation.

<table><tr><td rowspan="2">M</td><td colspan="2">\( {\psi }^{a} \)</td><td colspan="2">\( {\psi }^{b} \)</td><td colspan="2">PINN</td></tr><tr><td>N</td><td>\( {L}^{\infty } \) error</td><td>N</td><td>\( {L}^{\infty } \) error</td><td>N</td><td>\( {L}^{\infty } \) error</td></tr><tr><td>200</td><td>208</td><td>8.76E-2</td><td>202</td><td>2.51E-2</td><td>202</td><td>2.59E-2</td></tr><tr><td>400</td><td>416</td><td>5.89E-7</td><td>402</td><td>5.18E-7</td><td>402</td><td>6.77E-3</td></tr><tr><td>800</td><td>832</td><td>4.44E-10</td><td>802</td><td>6.61E-10</td><td>802</td><td>1.35E-2</td></tr><tr><td>1600</td><td>1664</td><td>8.84E-12</td><td>1602</td><td>1.18E-11</td><td>1602</td><td>8.94E-3</td></tr></table>

- \( M = {200},{400},{800},{1600} \);

- \( {J}_{n} = {50} \);

- \( {M}_{p} = \frac{M}{50} \)

- \( {x}_{n} = {x}_{n} = 8\frac{{2n} - 1}{2{M}_{p}}, n = 1,\cdots,{M}_{p} \);

- \( {r}_{n} = {r}_{n} = \frac{8}{2{M}_{p}}, n = 1,\cdots,{M}_{p} \);

- \( Q = {200},{400},{800},{1600} \).

For \( {\psi }^{a} \), we need additional \( 2\left( {{M}_{p} - 1}\right) \) smoothness conditions for the solution and its first derivative at \( x = \frac{1}{{M}_{p}},\cdots,\frac{{M}_{p} - 1}{{M}_{p}} \). No additional conditions are needed for \( {\psi }^{b} \). The total number of conditions is \( N = {208},{416},{832},{1664} \) for \( {\psi }^{a} \), and \( N = {202},{402},{802},{1602} \) for \( {\psi }^{b} \), respectively.

Table A.1 compares RFM and PINN [8] in terms of accuracy. The network used in PINN has the same structure as that in RFM, one hidden layer with \( M \) neurons and the tanh activation function is used. Collocation points are also chosen to be the same as those in RFM with \( {\psi }^{b} \). Since the inner parameters are trainable in PINN, we use the Adam optimizer with learning rate 0.001 to train the network. The training process ends after 100000 epochs when \( M = {200},{400} \) and 200000 epochs when \( M = {800},{1600} \).

From Table A.1, we observe that error in PINN is around 1E - 3 without notable further improvement, while RFM for different PoU functions has exponential convergence. Fig. A.1 plots the error of RFM and PINN as a function terms of \( M \) in the semi-log scale and the exponential convergence of RFM is observed. This suggests that fixing the inner parameters greatly simplifies the optimization problem and allows us to obtain accurate and robust solutions.

Next, we report the results for Poisson equation (3.2) with the following explicit solution

\[u\left( {x, y}\right)  =  - \left\lbrack  {\frac{3}{2}\cos \left( {{\pi x} + \frac{2\pi }{5}}\right)  + 2\cos \left( {{2\pi x} - \frac{\pi }{5}}\right) }\right\rbrack  \left\lbrack  {\frac{3}{2}\cos \left( {{\pi y} + \frac{2\pi }{5}}\right)  + 2\cos \left( {{2\pi y} - \frac{\pi }{5}}\right) }\right\rbrack . \tag{A.2}\]

In this problem, we set the hyper-parameters as follows:

![Figure A.1](images/figure-A-1.png)

Figure A.1: Convergence of RFM and PINN for Helmholtz equation in the semi-log scale.

- \( M = {200},{400},{800},{1600} \);

- \( {J}_{n} = {400} \);

- \( {M}_{p} = \frac{M}{400} \)

- \( {x}_{n} = \left( {{x}_{i},{y}_{j}}\right)  = \left( {\frac{{2i} - 1}{2\sqrt{{M}_{p}}},\frac{{2j} - 1}{2\sqrt{{M}_{p}}}}\right), i, j = 1,\cdots,\sqrt{{M}_{p}} \);

- \( {r}_{n} = {r}_{i, j} = \left( {\frac{1}{2\sqrt{{M}_{p}}},\frac{1}{2\sqrt{{M}_{p}}}}\right), i, j = 1,\cdots,\sqrt{{M}_{p}} \);

\[\text{ - }Q = {400}{M}_{p},{625}{M}_{p},{900}{M}_{p},{1225}{M}_{p},{1600}{M}_{p}\text{. }\]

For \( {\psi }^{a} \), we impose the smoothness conditions at \( \left( {\frac{i}{\sqrt{{M}_{p}}},\frac{j}{\sqrt{{M}_{p}}}}\right), i, j = 1,2,\cdots,\sqrt{{M}_{p}} \). Table A.2 shows the error of RFM for different PoU functions. Both show exponential convergence.

Fig. A.2 shows the error distribution of RFM for different choices of the PoU functions when \( M = {1600} \) and \( Q = {2500} \). For \( {\psi }^{a} \), the error is more concentrated near the intersection of different sub-domains where smoothness conditions are imposed. For \( {\psi }^{b} \), however, the error is concentrated near the boundary. Similar results are observed for the one-dimensional Helmholtz equation.

### A.2 Multi-scale basis

In this subsection, we show that the combination of local and global random feature functions works better in cases when the solution has both significant low and high frequency components.

Table A.2: Results for the two-dimensional Poisson equation with explicit solution (A.2)

<table><tr><td rowspan="2">M</td><td colspan="2">\( {\psi }^{a} \)</td><td colspan="2">\( {\psi }^{b} \)</td></tr><tr><td>N</td><td>\( {L}^{\infty } \) error</td><td>N</td><td>\( {L}^{\infty } \) error</td></tr><tr><td rowspan="5">1600</td><td>1920</td><td>1.74E-8</td><td>1760</td><td>1.90E-7</td></tr><tr><td>2900</td><td>1.55E-9</td><td>2700</td><td>1.22E-10</td></tr><tr><td>4080</td><td>2.31E-10</td><td>3840</td><td>3.89E-10</td></tr><tr><td>5460</td><td>6.29E-11</td><td>5180</td><td>2.67E-10</td></tr><tr><td>7040</td><td>5.04E-11</td><td>6720</td><td>4.68E-10</td></tr><tr><td rowspan="5">6400</td><td>7680</td><td>7.77E-10</td><td>6720</td><td>1.61E-8</td></tr><tr><td>11600</td><td>5.74E-11</td><td>10400</td><td>1.91E-11</td></tr><tr><td>16320</td><td>7.04E-12</td><td>14880</td><td>5.64E-11</td></tr><tr><td>21840</td><td>9.93E-12</td><td>20160</td><td>5.21E-11</td></tr><tr><td>28160</td><td>1.66E-11</td><td>26240</td><td>4.97E-11</td></tr></table>

![Figure A.2](images/figure-A-2.png)

Figure A.2: Error distribution of the RFM with different choices of PoU for Poisson equation with solution (A.2)

Consider the Poisson equation with the following explicit solution

\[u\left( {x, y}\right)\]

\[=  - A\left\lbrack  {\frac{3}{2}\cos \left( {{\pi x} + \frac{2\pi }{5}}\right)  + 2\cos \left( {{2\pi x} - \frac{\pi }{5}}\right) }\right\rbrack  \left\lbrack  {\frac{3}{2}\cos \left( {{\pi y} + \frac{2\pi }{5}}\right)  + 2\cos \left( {{2\pi y} - \frac{\pi }{5}}\right) }\right\rbrack\]

\[- B\left\lbrack  {\frac{3}{2}\cos \left( {{2\pi x} + \frac{4\pi }{5}}\right)  + 2\cos \left( {{4\pi x} - \frac{2\pi }{5}}\right) }\right\rbrack  \left\lbrack  {\frac{3}{2}\cos \left( {2\pi y + \frac{4\pi }{5}}\right)  + 2\cos \left( {{4\pi y} - \frac{2\pi }{5}}\right) }\right\rbrack .\]

(A.3)

Three cases are considered: 1. a low-frequency problem when \( A = {1.0}, B = {0.0};2 \). a high-frequency problem when \( A = {0.0}, B = {1.0};3 \). mixed-frequency problem when \( A = {0.5} \),

Table A.3: Comparison of PoU-based local basis and multi-scale basis functions for Poisson equation with the explicit solution (A.3).

<table><tr><td>Solution frequency</td><td>M</td><td>N</td><td>PoU-based basis</td><td>Multi-scale basis</td></tr><tr><td rowspan="3">Low</td><td>1200</td><td>1920</td><td>1.93E-8</td><td>3.28E-9</td></tr><tr><td>2700</td><td>4320</td><td>3.62E-9</td><td>6.42E-10</td></tr><tr><td>4800</td><td>7680</td><td>8.61E-10</td><td>3.05E-10</td></tr><tr><td rowspan="3">High</td><td>1200</td><td>1920</td><td>6.42E-6</td><td>9.36E-7</td></tr><tr><td>2700</td><td>4320</td><td>1.34E-7</td><td>3.58E-8</td></tr><tr><td>4800</td><td>7680</td><td>4.16E-8</td><td>1.75E-8</td></tr><tr><td rowspan="3">Mixed</td><td>1200</td><td>1920</td><td>3.22E-6</td><td>4.68E-7</td></tr><tr><td>2700</td><td>4320</td><td>6.54E-8</td><td>1.80E-8</td></tr><tr><td>4800</td><td>7680</td><td>2.06E-8</td><td>8.92E-9</td></tr></table>

\( B = {0.5} \).

The other hyper-parameters are set as follows:

- \( M = {1200},{2700},{4800} \);

- \( {J}_{n} = {300} \);

- \( {M}_{p} = \frac{M}{300} \)

- \( {x}_{n} = \left( {{x}_{i},{y}_{j}}\right)  = \left( {\frac{{2i} - 1}{2\sqrt{{M}_{p}}},\frac{{2j} - 1}{2\sqrt{{M}_{p}}}}\right), i, j = 1,\cdots,\sqrt{{M}_{p}} \);

- \( {r}_{n} = \left( {{r}_{i},{r}_{j}}\right)  = \left( {\frac{1}{2\sqrt{{M}_{p}}},\frac{1}{2\sqrt{{M}_{p}}}}\right), i, j = 1,\cdots,\sqrt{{M}_{p}} \);

- \( Q = {1600},{3600},{6400} \);

\[\text{ - }N = {1920},{4320},{7680}\text{. }\]

Multi-scale basis functions are constructed using \( \frac{{300}{M}_{p}}{{M}_{p} + 1} \) basis functions for the values of \( \left\{  \left( {{x}_{i},{y}_{j}}\right) \right\} \) and \( \left\{  \left( {{r}_{i},{r}_{j}}\right) \right\} \) given above, together with \( \frac{{300}{M}_{p}}{{M}_{p} + 1} \) global basis functions with the parameters \( \left( {x, y}\right)  = \left( {\frac{1}{2},\frac{1}{2}}\right),\left( {{r}_{x},{r}_{y}}\right)  = \left( {\frac{1}{2},\frac{1}{2}}\right) \). 300 local basis functions are constructed for each \( \left( {{x}_{i},{y}_{j}}\right) \) with the values of \( \left( {{r}_{i},{r}_{j}}\right) \) given above.

Table A.3 shows the error for RFM with multi-scale basis and with only PoU-based local basis functions. It is clear that the inclusion of global basis functions improves the accuracy when the solution has a significant low-frequency component. A Fourier analysis of the errors confirms that the inclusion of global basis functions reduces the low-frequency error more effectively.

### A.3 Adaptive basis

Here we demonstrate how the a prior information helps us to select better random feature functions.

Consider the one-dimensional Helmholtz equation with the following explicit solution

\[u\left( x\right)  = 4\cos \left( {4\left( {x + \frac{3}{20}}\right) }\right)  + 5\sin \left( {\sqrt{5}\left( {x + \frac{7}{20}}\right) }\right)\]

\[+ 2\sin \left( {\sqrt{3}\left( {x + \frac{1}{20}}\right) }\right)  + 3\sin \left( {x + \frac{17}{20}}\right)  + 2 \tag{A.4}\]

The other hyper-parameters are as follows:

- \( M = {400},{800},{1600} \);

- \( {J}_{n} = {100} \);

\[\text{ - }{M}_{p} = \frac{M}{100}\]

\[\text{ - }{x}_{n} = {x}_{n} = 8\frac{{2n} - 1}{2{M}_{p}}, n = 1,\cdots,{M}_{p}\text{; }\]

\[\text{ - }{r}_{n} = {r}_{n} = \frac{8}{2{M}_{p}}, n = 1,\cdots,{M}_{p}\text{; }\]

\[\text{ - }Q = {200},{400},{800}\text{; }\]

\[\text{ - }N = {208},{416},{832}\text{. }\]

We use tanh and sin as the activation function. We test two different initialization methods: random initialization with the distribution \( \mathbb{U}\left\lbrack  {-{R}_{m},{R}_{m}}\right\rbrack \) and equally spaced grids over \( \left\lbrack  {-{R}_{m},{R}_{m}}\right\rbrack \) with \( \left\{  {k}_{m}\right\} \) and \( \left\{  {b}_{m}\right\} \) being \( - {R}_{m} + 2{R}_{m}\frac{i}{10}, i = 1,\cdots,{10} \). Eight values of \( {R}_{m} \). By the spectral analysis of the source term, we find that the highest frequency is 4. Since a normalization is applied from \( x \) to \( \widetilde{x} \), the highest-frequency term in \( u \) corresponds to \( k = \frac{4}{1/{r}_{m}} = \frac{16}{{M}_{p}} \). Therefore, the best results are obtained when the sin activation function is used over \( \left\lbrack  {-{R}_{m},{R}_{m}}\right\rbrack \) with \( {R}_{m} \geq  k \).

Results of using adaptive random feature functions for this problem are shown in Table A.4. Another interesting observation is that while choosing equally spaced feature vectors works for some cases, the random initialization is found to be generally more reliable.

We now turn to the Poisson equation with the solution (A.2). We set the hyper-parameters as follows:

\[\text{ - }M = {4000}\text{; }\]

\[\text{ - }{J}_{n} = {1000}\text{; }\]

\[\text{ - }{M}_{p} = \frac{M}{1000}\]

\[\text{ - }{x}_{n} \in  \left\{  {\left( {\frac{1}{4},\frac{1}{4}}\right),\left( {\frac{1}{4},\frac{3}{4}}\right),\left( {\frac{3}{4},\frac{1}{4}}\right),\left( {\frac{3}{4},\frac{3}{4}}\right) }\right\}  \text{; }\]

\[\text{ - }{r}_{n} = \left( {{r}_{x},{r}_{y}}\right)  = \left( {\frac{1}{4},\frac{1}{4}}\right) \text{; }\]

\[\text{ - }Q = {1600}\text{; }\]

\[\text{ - }N = {1920}\text{. }\]

Table A.4: Results of the adaptive RFM for one-dimensional Helmholtz equation with solution (A.4).

<table><tr><td rowspan="2">\( M \)</td><td rowspan="2">\( {R}_{m} \)</td><td colspan="2">tanh</td><td colspan="2">sin</td></tr><tr><td>\( \mathbb{U}\left\lbrack  {-{R}_{m},{R}_{m}}\right\rbrack \)</td><td>Equally spaced</td><td>\( \mathbb{U}\left\lbrack  {-{R}_{m},{R}_{m}}\right\rbrack \)</td><td>Equally spaced</td></tr><tr><td rowspan="8">400</td><td>1</td><td>3.03E-10</td><td>1.01E-10</td><td>6.21E-2</td><td>2.70E-2</td></tr><tr><td>2</td><td>3.41E-11</td><td>1.75E-10</td><td>9.34E-5</td><td>1.23E-2</td></tr><tr><td>3</td><td>5.12E-9</td><td>6.55E-10</td><td>4.68E-8</td><td>2.29E-3</td></tr><tr><td>4</td><td>1.45E-7</td><td>2.93E-7</td><td>7.55E-13</td><td>8.02E-6</td></tr><tr><td>5</td><td>1.77E-5</td><td>5.13E-4</td><td>1.14E-13</td><td>1.67E-4</td></tr><tr><td>6</td><td>1.44E-4</td><td>3.02E-3</td><td>1.64E-13</td><td>9.60E-3</td></tr><tr><td>7</td><td>8.20E-4</td><td>1.54E-2</td><td>2.31E-13</td><td>7.21E-2</td></tr><tr><td>8</td><td>1.97E-2</td><td>8.58E-1</td><td>7.02E-14</td><td>5.62E-1</td></tr><tr><td rowspan="8">800</td><td>1</td><td>9.78E-11</td><td>1.25E-11</td><td>2.44E-7</td><td>9.01E-6</td></tr><tr><td>2</td><td>2.35E-11</td><td>1.66E-11</td><td>4.39E-13</td><td>3.77E-8</td></tr><tr><td>3</td><td>2.11E-9</td><td>5.13E-11</td><td>2.52E-13</td><td>3.46E-6</td></tr><tr><td>4</td><td>1.16E-7</td><td>1.01E-8</td><td>8.92E-13</td><td>1.04E-5</td></tr><tr><td>5</td><td>3.03E-6</td><td>2.71E-5</td><td>1.02E-12</td><td>2.45E-4</td></tr><tr><td>6</td><td>8.75E-5</td><td>2.22E-4</td><td>1.60E-12</td><td>3.46E-3</td></tr><tr><td>7</td><td>5.94E-4</td><td>1.57E-3</td><td>2.17E-13</td><td>6.17E-2</td></tr><tr><td>8</td><td>1.91E-3</td><td>9.51E-2</td><td>1.36E-12</td><td>5.12E-1</td></tr><tr><td rowspan="8">1600</td><td>1</td><td>5.76E-12</td><td>7.29E-13</td><td>1.12E-12</td><td>1.68E-11</td></tr><tr><td>2</td><td>6.64E-12</td><td>1.26E-11</td><td>6.23E-13</td><td>8.88E-7</td></tr><tr><td>3</td><td>1.82E-9</td><td>1.50E-10</td><td>2.09E-13</td><td>3.48E-5</td></tr><tr><td>4</td><td>2.63E-7</td><td>7.59E-9</td><td>2.04E-13</td><td>2.18E-4</td></tr><tr><td>5</td><td>6.00E-6</td><td>6.25E-6</td><td>1.29E-12</td><td>3.98E-3</td></tr><tr><td>6</td><td>1.22E-4</td><td>8.59E-5</td><td>3.96E-12</td><td>1.07E-2</td></tr><tr><td>7</td><td>4.55E-3</td><td>7.24E-4</td><td>1.16E-12</td><td>2.39E-1</td></tr><tr><td>8</td><td>4.65E-3</td><td>5.16E-2</td><td>9.79E-13</td><td>2.07E+0</td></tr></table>

We use tanh and sin as the activation function. We test two different initialization methods: random initialization with distribution \( \mathbb{U}\left\lbrack  {-{R}_{m},{R}_{m}}\right\rbrack \) and equally spaced grids over \( \left\lbrack  {-{R}_{m},{R}_{m}}\right\rbrack \) with \( \left\{  {{k}_{m}^{1},{k}_{m}^{2}}\right\} \) and \( \left\{  {b}_{m}\right\} \) being \( - {R}_{m} + 2{R}_{m}\frac{i}{10}, i = 1,\cdots,{10} \). 10 values of \( {R}_{m} \) are tested. Results of using adaptive random feature functions are shown in Table A.5. Again, the best results are observed when the sin activation function is used over \( \left\lbrack  {-{R}_{m},{R}_{m}}\right\rbrack \) with \( {R}_{m} \geq  k \) and random initialization is found to be generally more reliable.

## B Experimental setup and numerical results for problems with complex geometry

### B.1 Rescaling

The exact displacement solution for the Timoshenko beam problem is

\[u =  - \frac{Py}{6EI}\left\lbrack  {\left( {{6L} - {3x}}\right) x + \left( {2 + v}\right) \left( {{y}^{2} - \frac{{D}^{2}}{4}}\right) }\right\rbrack , \tag{B.1}\]

\[v = \frac{P}{6EI}\left\lbrack  {{3\nu }{y}^{2}\left( {L - x}\right)  + \left( {4 + {5\nu }}\right) \frac{{D}^{2}x}{4} + \left( {{3L} - x}\right) {x}^{2}}\right\rbrack ,\]

Table A.5: Results of using adaptive random feature functions for the two-dimensional Poisson equation with solution (A.2).

<table><tr><td rowspan="2">\( {R}_{m} \)</td><td colspan="2">tanh</td><td colspan="2">sin</td></tr><tr><td>\( \mathbb{U}\left\lbrack  {-{R}_{m},{R}_{m}}\right\rbrack \)</td><td>Equally spaced</td><td>\( \mathbb{U}\left\lbrack  {-{R}_{m},{R}_{m}}\right\rbrack \)</td><td>Equally spaced</td></tr><tr><td>0.5</td><td>4.92E-9</td><td>1.01E-9</td><td>2.55E-3</td><td>6.05E-4</td></tr><tr><td>1.0</td><td>2.91E-8</td><td>9.36E-9</td><td>8.96E-7</td><td>2.58E-5</td></tr><tr><td>1.5</td><td>1.33E-6</td><td>5.95E-7</td><td>1.79E-9</td><td>1.47E-6</td></tr><tr><td>2.0</td><td>8.75E-5</td><td>7.85E-5</td><td>3.30E-12</td><td>4.29E-7</td></tr><tr><td>2.5</td><td>8.16E-4</td><td>4.70E-5</td><td>2.86E-12</td><td>7.66E-6</td></tr><tr><td>3.0</td><td>2.06E-2</td><td>5.27E-4</td><td>7.32E-12</td><td>2.17E-5</td></tr><tr><td>3.5</td><td>1.53E-3</td><td>3.95E-3</td><td>6.10E-12</td><td>7.45E-5</td></tr><tr><td>4.0</td><td>2.66E-3</td><td>1.27E-3</td><td>6.10E-12</td><td>5.59E-5</td></tr><tr><td>4.5</td><td>5.39E-3</td><td>1.76E-2</td><td>2.29E-11</td><td>1.24E-3</td></tr><tr><td>5.0</td><td>1.29E-2</td><td>5.16E-2</td><td>2.17E-11</td><td>6.72E-3</td></tr></table>

where \( I = \frac{{D}^{3}}{12} \). Homogeneous Dirichlet boundary condition is applied on the left boundary \( x = 0 \) and Homogeneous Neumann boundary condition is applied on the other boundaries. The material parameters are as follows: the Young’s modulus \( E = 3 \times  {10}^{7} \) Pa, Poisson ratio \( v = {0.3} \). We choose \( D = {10}, L = {10} \), and the shear force is \( P = {1000}\mathrm{\;{Pa}} \).

The other hyper-parameters for the Timoshenko beam problem are as follows:

- \( M = {800},{3200} \); - \( {J}_{n} = {200} \) - \( {M}_{p} = \frac{M}{200} \) - \( {\mathbf{x}}_{n} = \left( {{x}_{i},{y}_{j}}\right)  = \left( {{10}\frac{{2i} - 1}{2\sqrt{{M}_{p}}},{10}\frac{{2j} - 1}{2\sqrt{{M}_{p}}}}\right), i, j = 1,\cdots,\sqrt{{M}_{p}} \); - \( {r}_{n} = \left( {{r}_{x},{r}_{y}}\right)  = \left( {5,5}\right) \); - \( Q = {25}{M}_{p},{100}{M}_{p},{400}{M}_{p},{1600}{M}_{p} \).

We construct \( \frac{{200}{M}_{p}}{{M}_{p} + 1} \) basis functions associated with the choice of \( \left\{  \left( {{x}_{i},{y}_{j}}\right) \right\} \) with \( \left\{  \left( {{r}_{x},{r}_{y}}\right) \right\} \) given above, and add \( \frac{{200}{M}_{p}}{{M}_{p} + 1} \) basis functions associated with the point \( \left( {5,5}\right) \) with \( \left( {{r}_{x},{r}_{y}}\right)  = \; \left( {5,5}\right) \). To count the total number of basis functions and the number of conditions, we convert \( {N}_{x},{N}_{y},{Q}_{x},{Q}_{y} \), and \( {M}^{\prime } \) in locELM to \( M \) and \( N \) in RFM according to \( M = 2{N}_{x}{N}_{y}{M}^{\prime } \) and \( N = 2{N}_{x}{N}_{y}{Q}_{x}{Q}_{y} + 4{N}_{x}{Q}_{x} + 4{N}_{y}{Q}_{y} + 6{N}_{x}{Q}_{x}\left( {{N}_{y} - 1}\right)  + 6{N}_{y}{Q}_{y}\left( {{N}_{x} - 1}\right) \).

Table B.1: Results of RFM for the elasticity problem with complex geometry.

<table><tr><td>Method</td><td>\( M \)</td><td>\( N \)</td><td>\( u \) error</td><td>\( v \) error</td><td>\( {\sigma }_{x} \) error</td><td>\( {\sigma }_{y} \) error</td><td>\( {\tau }_{xy} \) error</td></tr><tr><td rowspan="8">RFM</td><td rowspan="4">3200</td><td>1784</td><td>4.96E-1</td><td>8.37E-1</td><td>1.09E+0</td><td>3.52E+0</td><td>5.24E-1</td></tr><tr><td>4658</td><td>5.82E-3</td><td>7.12E-3</td><td>1.04E-2</td><td>5.47E-2</td><td>3.85E-3</td></tr><tr><td>13338</td><td>1.69E-5</td><td>1.19E-5</td><td>2.89E-5</td><td>6.40E-5</td><td>8.18E-6</td></tr><tr><td>42820</td><td>1.39E-5</td><td>1.55E-5</td><td>4.92E-5</td><td>6.16E-5</td><td>1.29E-5</td></tr><tr><td rowspan="4">12800</td><td>6578</td><td>9.11E-2</td><td>6.41E-2</td><td>1.03E-1</td><td>2.46E-1</td><td>2.95E-2</td></tr><tr><td>17178</td><td>2.35E-4</td><td>2.10E-4</td><td>3.02E-4</td><td>7.56E-4</td><td>8.93E-5</td></tr><tr><td>50500</td><td>5.46E-7</td><td>4.98E-7</td><td>8.45E-7</td><td>2.03E-6</td><td>2.67E-7</td></tr><tr><td>165184</td><td>2.32E-7</td><td>1.89E-7</td><td>9.28E-8</td><td>2.32E-7</td><td>2.43E-8</td></tr></table>

The exact displacement field for the two-dimensional elasticity problem with complex geometry shown in Fig. 3.2 is

\[u = \frac{1}{10}y\left( {\left( {x + {10}}\right) \sin y + \left( {y + 5}\right) \cos x}\right), \tag{B.2}\]

\[v = \frac{1}{60}y\left( {\left( {{30} + {5x}\sin \left( {5x}\right) }\right) \left( {4 + {e}^{-{5y}}}\right)  - {100}}\right).\]

Dirichlet boundary condition is applied on the lower boundary \( y = 0 \) and Neumann boundary condition is applied on the other boundaries and the holes inside. The material constants are: the Young’s modulus \( E = 3 \times  {10}^{7}\mathrm{\;{Pa}} \) and Poisson ratio \( v = {0.3} \).

The other hyper-parameters are as follows:

- \( M = {3200} \),12800; - \( {J}_{n} = {200} \); - \( {M}_{p} = \frac{M}{200} \) - \( {x}_{n} = \left( {{x}_{i},{y}_{j}}\right)  = \left( {8\frac{{2i} - 1}{2\sqrt{{M}_{p}}},8\frac{{2j} - 1}{2\sqrt{{M}_{p}}}}\right), i, j = 1,\cdots,\sqrt{{M}_{p}} \); - \( {r}_{n} = \left( {{r}_{x},{r}_{y}}\right)  = \left( {\frac{8}{2\sqrt{{M}_{p}}},\frac{8}{2\sqrt{{M}_{p}}}}\right) \); - \( Q = {25}{M}_{p},{100}{M}_{p},{400}{M}_{p},{1600}{M}_{p} \).

We construct \( \frac{{200}{M}_{p}}{{M}_{p} + 1} \) basis functions for each point \( \left( {{x}_{i},{y}_{j}}\right) \) with \( \left( {{r}_{x},{r}_{y}}\right) \) given above, and adds \( \frac{{200}{M}_{p}}{{M}_{p} + 1} \) basis functions for the point \( \left( {4,4}\right) \) with \( \left( {{r}_{x},{r}_{y}}\right)  = \left( {4,4}\right) \).

Results of the RFM for the elasticity problem with this explicit solution are shown in Table B.1.

### B.2 Comparison with FEM

For the two-dimensional elasticity problem in Section 3.3, we set hyper-parameters as follows:

Table B.2: Comparison of the numerical solutions of RFM and FEM for the elasticity problem.

<table><tr><td>Method</td><td>Reference</td><td>\( M \)</td><td>\( N \)</td><td>\( u \) error</td><td>\( v \) error</td><td>\( {\sigma }_{x} \) error</td><td>\( {\sigma }_{y} \) error</td><td>\( {\tau }_{xy} \) error</td></tr><tr><td rowspan="3">RFM</td><td rowspan="3">RFM \( N = {490176} \)</td><td rowspan="3">16000</td><td>40326</td><td>1.28E+0</td><td>1.12E+0</td><td>1.29E+0</td><td>9.37E-1</td><td>1.03E+0</td></tr><tr><td>135442</td><td>1.12E-1</td><td>1.16E-1</td><td>1.13E-1</td><td>1.03E-2</td><td>1.20E-1</td></tr><tr><td>285472</td><td>6.52E-4</td><td>6.98E-4</td><td>1.03E-3</td><td>3.01E-5</td><td>1.88E-3</td></tr><tr><td rowspan="4">RFM</td><td rowspan="4">FEM \( M = {153562} \)</td><td rowspan="4">16000</td><td>40326</td><td>1.30E+0</td><td>1.12E+0</td><td>1.28E+0</td><td>9.37E-1</td><td>1.03E+0</td></tr><tr><td>135442</td><td>7.65E-2</td><td>8.55E-2</td><td>1.16E-1</td><td>1.31E-1</td><td>1.25E-1</td></tr><tr><td>285472</td><td>3.94E-2</td><td>3.36E-2</td><td>6.59E-3</td><td>5.95E-2</td><td>2.31E-2</td></tr><tr><td>490176</td><td>4.00E-2</td><td>3.43E-2</td><td>6.20E-3</td><td>5.92E-2</td><td>2.30E-2</td></tr><tr><td rowspan="3">FEM</td><td rowspan="3">FEM \( M = {153562} \)</td><td>3716</td><td>3716</td><td>3.15E-4</td><td>4.54E-4</td><td>1.41E-2</td><td>5.81E-2</td><td>3.35E-2</td></tr><tr><td>10438</td><td>10438</td><td>1.20E-4</td><td>1.81E-4</td><td>9.39E-3</td><td>3.61E-2</td><td>2.13E-2</td></tr><tr><td>40054</td><td>40054</td><td>2.88E-5</td><td>3.93E-5</td><td>4.65E-3</td><td>1.62E-2</td><td>9.40E-3</td></tr><tr><td rowspan="4">FEM</td><td rowspan="4">RFM \( N = {490176} \)</td><td>3716</td><td>3716</td><td>3.87E-2</td><td>3.36E-2</td><td>1.43E-2</td><td>8.93E-2</td><td>3.86E-2</td></tr><tr><td>10438</td><td>10438</td><td>3.86E-2</td><td>3.34E-2</td><td>1.05E-2</td><td>7.29E-2</td><td>2.99E-2</td></tr><tr><td>40054</td><td>40054</td><td>3.85E-2</td><td>3.32E-2</td><td>7.19E-3</td><td>6.33E-2</td><td>2.44E-2</td></tr><tr><td>153562</td><td>153562</td><td>3.85E-2</td><td>3.32E-2</td><td>6.22E-3</td><td>6.01E-2</td><td>2.31E-2</td></tr></table>

- \( M = {16000} \)

- \( {J}_{n} = {400} \)

- \( {M}_{p} = \frac{M}{400} \);

- \( {x}_{n} = \left( {{x}_{i},{y}_{j}}\right)  = \left( {\frac{{2i} - 1}{8} - 1,\frac{{2j} - 1}{8} - \frac{1}{2}}\right), i = 1,\cdots,{10}, j = 1,\cdots,4 \);

- \( {r}_{n} = \left( {{r}_{x},{r}_{y}}\right)  = \left( {\frac{1}{8},\frac{1}{8}}\right) \);

- \( Q = {16000},{64000},{144000},{256000} \);

- \( N = {40326},{135442},{285472},{490176} \).

For comparison, we implement the standard adaptive FEM with total degrees of freedom \( M = {3716},{10438},{40054},{153562}. \)

Table B.2 shows the error between RFM and FEM for the elasticity problem in Section 3.3

We set the hyper-parameters for the elasticity problem over a complex geometry in Section 3.3 as follows:

- \( M = {14400} \);

- \( {J}_{n} = {400} \);

- \( {M}_{p} = \frac{M}{400} \)

- \( {x}_{n} = \left( {{x}_{i},{y}_{j}}\right)  = \left( {8\frac{{2i} - 1}{12},8\frac{{2j} - 1}{12}}\right), i, j = 1,\cdots,6 \);

Table B.3: Numerical results of the RFM for the elasticity problem over a complex geometry. The result with \( N = {332606} \) is taken as the reference solution.

<table><tr><td>\( M \)</td><td>\( N \)</td><td>\( u \) error</td><td>\( v \) error</td><td>\( {\sigma }_{x} \) error</td><td>\( {\sigma }_{y} \) error</td><td>\( {\tau }_{xy} \) error</td></tr><tr><td rowspan="4">14400</td><td>195146</td><td>2.30E-1</td><td>1.30E-1</td><td>6.64E-2</td><td>1.72E-1</td><td>1.71E-1</td></tr><tr><td>226132</td><td>8.97E-2</td><td>1.23E-1</td><td>5.60E-2</td><td>1.41E-1</td><td>1.32E-1</td></tr><tr><td>259400</td><td>6.47E-2</td><td>6.94E-2</td><td>3.66E-2</td><td>9.04E-2</td><td>8.15E-2</td></tr><tr><td>294878</td><td>7.30E-2</td><td>6.68E-2</td><td>3.46E-2</td><td>7.13E-2</td><td>7.05E-2</td></tr></table>

- \( {r}_{n} = \left( {{r}_{x},{r}_{y}}\right)  = \left( {\frac{8}{12},\frac{8}{12}}\right) \);

- \( Q = {129600},{152100},{176400},{202500},{230400} \);

- \( N = {195146},{226132},{259400},{294878},{332606} \).

Table B.3 records the results of RFM for the elasticity problem over a complex geometry in Section 3.3.

### B.3 The elliptic homogenization problem

For the homogenization problem, we set the hyper-parameters as follows:

- \( M = {25600} \);

- \( {J}_{n} = {400} \)

- \( {M}_{p} = \frac{M}{400} \)

- \( {x}_{n} = \left( {{x}_{i},{y}_{j}}\right)  = \left( {\frac{{2i} - 1}{8} - 1,\frac{{2j} - 1}{8} - 1}\right), i, j = 1,\cdots,8 \);

- \( {r}_{n} = \left( {{r}_{x},{r}_{y}}\right)  = \left( {\frac{1}{8},\frac{1}{8}}\right) \);

- \( Q = {25600},{102400},{230400},{409600} \);

- \( N = {25554},{91339},{197360},{343586} \).

Results of RFM for the homogenization problem in Section 3.4 are shown in Table B.4

Table B.4: Numerical convergence of the random feature method for the homogenization problem.

<table><tr><td>\( M \)</td><td>\( N \)</td><td>\( u \) error</td><td>\( {u}_{x} \) error</td><td>\( {u}_{y} \) error</td></tr><tr><td rowspan="4">25600</td><td>25554</td><td>1.42E+0</td><td>8.68E+0</td><td>8.73E+0</td></tr><tr><td>91339</td><td>3.13E-2</td><td>3.54E-2</td><td>3.62E-2</td></tr><tr><td>197360</td><td>3.48E-3</td><td>6.45E-3</td><td>7.18E-3</td></tr><tr><td>343586</td><td colspan="3">Reference</td></tr></table>

### B.4 Stokes flow

The exact displacement field for the Stokes flow is given by

\[u = x + {x}^{2} - {2xy} + {x}^{3} - {3x}{y}^{2} + {x}^{2}y\]

\[v =  - y - {2xy} + {y}^{2} - 3{x}^{2}y + {y}^{3} - x{y}^{2}, \tag{B.3}\]

\[p = {xy} + x + y + {x}^{3}{y}^{2} - \frac{4}{3}.\]

We set the hyper-parameters for the Stokes flow as follows:

- \( M = {400},{800},{1600} \);

- \( {J}_{n} = {100},{200},{400} \);

- \( {M}_{p} = 4 \);

- \( {x}_{n} \in  \left\{  {\left( {\frac{1}{4},\frac{1}{4}}\right),\left( {\frac{1}{4},\frac{3}{4}}\right),\left( {\frac{3}{4},\frac{1}{4}}\right),\left( {\frac{3}{4},\frac{3}{4}}\right) }\right\} \);

- \( {r}_{n} = \left( {{r}_{x},{r}_{y}}\right)  = \left( {\frac{1}{4},\frac{1}{4}}\right) \);

- \( Q = {100},{400},{1600},{6400} \);

- \( N = {512},{1596},{5390},{19488} \).

Results of the RFM are shown in Table B.5.

Table B.5: Numerical results of the RFM for the Stokes flow with an explicit solution.

<table><tr><td>\( M \)</td><td>\( N \)</td><td>\( u \) error</td><td>\( v \) error</td><td>\( p \) error</td></tr><tr><td rowspan="4">400</td><td>512</td><td>3.22E-4</td><td>2.28E-4</td><td>3.21E-2</td></tr><tr><td>1596</td><td>6.13E-7</td><td>3.44E-7</td><td>9.72E-5</td></tr><tr><td>5390</td><td>4.22E-7</td><td>2.54E-7</td><td>1.64E-4</td></tr><tr><td>19488</td><td>1.44E-7</td><td>1.03E-7</td><td>1.31E-5</td></tr><tr><td rowspan="4">800</td><td>512</td><td>5.25E-4</td><td>3.49E-4</td><td>4.39E-2</td></tr><tr><td>1596</td><td>4.95E-7</td><td>3.03E-7</td><td>2.77E-5</td></tr><tr><td>5390</td><td>1.60E-10</td><td>9.48E-11</td><td>1.73E-7</td></tr><tr><td>19488</td><td>1.15E-10</td><td>6.01E-11</td><td>1.06E-7</td></tr><tr><td rowspan="4">1600</td><td>512</td><td>3.33E-4</td><td>3.61E-4</td><td>4.64E-2</td></tr><tr><td>1596</td><td>1.11E-6</td><td>6.23E-7</td><td>5.67E-5</td></tr><tr><td>5390</td><td>3.02E-12</td><td>1.56E-12</td><td>1.06E-9</td></tr><tr><td>19488</td><td>2.45E-13</td><td>1.63E-13</td><td>1.37E-9</td></tr></table>

## References

[1] Randall J. LeVeque. Finite Difference Methods for Ordinary and Partial Differential Equations: Steady-State and Time-Dependent Problems. SIAM, 2007.

[2] Olek C. Zienkiewicz, Robert Leroy Taylor, and Jian Z. Zhu. The Finite Element Method: Its Basis and Fundamentals. Elsevier, 2005.

[3] Jie Shen, Tao Tang, and Li-Lian Wang. Spectral Methods: Algorithms, Analysis and Applications, volume 41. Springer Science & Business Media, 2011.

[4] Weinan E, Jiequn Han, and Arnulf Jentzen. Deep learning-based numerical methods for high-dimensional parabolic partial differential equations and backward stochastic differential equations. Communications in Mathematics and Statistics, 5(4):349-380, 2017.

[5] Jiequn Han, Arnulf Jentzen, and Weinan E. Solving high-dimensional partial differential equations using deep learning. Proceedings of the National Academy of Sciences of the United States of America, 115(34):8505- 8510, 2018.

[6] Weinan E and Bing Yu. The deep Ritz method: A deep learning-based numerical algorithm for solving variational problems. Communications in Mathematics and Statistics, 6(1):1-12, Mar 2018.

[7] Justin A. Sirignano and Konstantinos Spiliopoulos. DGM: A deep learning algorithm for solving partial differential equations. Journal of Computational Physics, 375:1339-1364, 2018.

[8] Maziar Raissi, Paris Perdikaris, and George Em Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378:686-707, 2019.

[9] Yaohua Zang, Gang Bao, Xiaojing Ye, and Haomin Zhou. Weak adversarial networks for high-dimensional partial differential equations. Journal of Computational Physics, 411:109409, 2020.

[10] Hyuk Lee and In Seok Kang. Neural algorithm for solving differential equations. Journal of Computational Physics, 91:110-131, 1990.

[11] Jiequn Han and Weinan E. Deep learning approximation for stochastic control problems. In Deep Reinforcement Learning Workshop, NIPS, 2016.

[12] Weinan E, Jiequn Han, and Arnulf Jentzen. Algorithms for solving high dimensional PDEs: From nonlinear Monte Carlo to machine learning. arXiv preprint arXiv:2008.13333, 2020.

[13] Yuehaw Khoo, Jianfeng Lu, and Lexing Ying. Solving parametric pde problems with artificial neural networks. European Journal of Applied Mathematics, 32(3):421-435, 2021.

[14] Lu Lu, Pengzhan Jin, Guofei Pang, Zhongqiang Zhang, and George Em Karniadakis. Learning nonlinear operators via deeponet based on the universal approximation theorem of operators. Nature Machine Intelligence, 3(3):218-229, 2021.

[15] Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial differential equations. arXiv preprint arXiv:2010.08895, 2020.

[16] Myron B. Allen III. The Mathematics of Fluid Flow Through Porous Media. John Wiley & Sons, 2021.

[17] Hans-Joachim Bungartz and Michael Griebel. Sparse grids. Acta Numerica, 13:147-269, 2004.

[18] Jie Shen and Haijun Yu. Efficient spectral sparse grid methods and applications to high-dimensional elliptic problems. SIAM Journal on Scientific Computing, 32(6):3228-3250, 2010.

[19] S.K. Stefanov. On the basic concepts of the direct simulation Monte Carlo method. Physics of Fluids, 31(6):067104, 2019.

[20] Weinan E. Principles of Multiscale Modeling. Cambridge University Press, 2011.

[21] Suchuan Dong and Zongwei Li. Local extreme learning machines and domain decomposition for solving linear and nonlinear partial differential equations. Computer Methods in Applied Mechanics and Engineering, 387:114129, 2021.

[22] Guang-Bin Huang, Qin-Yu Zhu, and Chee-Kheong Siew. Extreme learning machine: Theory and applications. Neurocomputing, 70(1-3):489-501, 2006.

[23] Vinh Phu Nguyen, Timon Rabczuk, Stéphane Bordas, and Marc Duflot. Meshless methods: A review and computer implementation aspects. Mathematics and Computers in Simulation, 79(3):763-813, 2008.

[24] Houman Owhadi and Lei Zhang. Metric-based upscaling. Communications on Pure and Applied Mathematics, 60(5):675-723, 2007.

[25] Mark R. Schumack, William W. Schultz, and John P. Boyd. Spectral method solution of the Stokes equations on nonstaggered grids. Journal of Computational Physics, 94(1):30-58, 1991.
