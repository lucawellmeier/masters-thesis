---
layout: post
title: Consistency of Interpolation with Laplace Kernels is a High-Dimensional Phenomenon
year: 2018
authors: Rakhlin and Zhai

---

If one keeps the dimension of data constant but increases the sample count, the risk of interpolating Laplace kernel regression is always lower-bounded by a constant. Proven by harmonic analysis using a representation of the RKHS norm as Sobolev norm.

<!--more-->

- available at: [paper on arxiv](https://arxiv.org/abs/1812.11167)
- Question of out-of-sample performance of interpolated estimators asked in the regimes of overparametrized NNs, kernel methods and local nonparametric rules (I haven't looked into the latter one yet, see the two Belkin papers from 2018)
- KRR performs unreasonably well for $\lambda = 0$ even though the solution (generally) interpolates the data
  - conditions why that is are poorly understood. [2019 \| Liang, Rakhlin] has shown that in the high-dimensional regime this is due to a sort-of implicit regularization due to the curvature of the kernel function, high-dimensionalty and favorable geometric properties quantiefied by the spectral decay of the kernel and covariance matrices
  - But is it really only high-dimensionalty that makes this possible? Maybe there is another mechanism...
- experimentally it seems that this is not the case: minimum-norm interpolants are not performing well in low dimensions
- present paper gives a theoretical justification for that
- Setup: KRR with Laplacian $K_c(x,x') = c^d e^{-c \| x-x' \|}$ for several reasons:
  - as noted for instance in [2018 \| Belkin, Ma, Mandal] the Laplacian kernel is very similar to ReLU neural nets and have a "large computational reach" 
  - RKHS norm corresponding to that kernel can be related to a Sobolev norm
- $K$ non-differentiable as required in said paper, but possible to extend the upper bound, because differentiability is only needed locally (see other summary)
- $c$ is the width parameter; important role (to be proven here): no choice of $c$ can make the interpolation method consistent if $d$ is a constant
  - compare with [2018 \| Liang, Rakhlin]: upper bounds there were only shown there with $c \asymp \sqrt d$
- Main theorem informally summarized: If $Y_i$ are noisy observations of $f_0(X_i)$, then the minimum-norm interpolant $\hat{f}_c$ with any data-dependent choice of the width $c$ is **inconsistent**: with probability close to 1
  $$\mathbb{E}(\hat{f(X)}_c - f_0(X))^2 \geq \Omega_d(1),$$
  where the $\Omega_d$ notation stresses that the $d$ is constant
- More precisely:
  - we take $f_0$ unknown but smooth over $\Omega = \overline{B_d(0,1)}$ and not identically zero
  - $P$ unknown distribution over $\Omega$ with probability density $\rho$ bounded by $c_\rho$ and $C_\rho$
  - $X_1, \dots, X_n$ sampled IID from $P$ and $Y_i = f_0(X_i) + \xi_i$ where $\xi_i$ is IID Rademacher noise; together $S = \{ (X_i, Y_i) \}$
  - $\hat{f}_c$ minimum-norm function interpolating $(X_i, Y_i)$ wrt the Laplace kernel
  - THEOREM: for fixed $n$ and odd dimension $d$, with probability at least $1 - O(1 / \sqrt{n})$, for all $c$
    $$\mathbb{E}(\hat{f(X)}_c - f_0(X))^2 \geq \Omega_d(1)$$
    - lower bound holds for *any* data-dependent choice of $c$
    - $d$ is odd for technical simplicity, will probably be generalizable by using more complicated tools of harmonic analysis
    - binary noise for brevity; magnitude can be rescaled quite easily
- parameter $\lambda > 0$ leads to a control of the norm of $\hat f$ in regularized least squares; here: ABSENT so complexity control is more difficult
  - intuition: norm of the solution should be related to distances between datapoints instead of derivatives
    - interpolation on noisy data (separated by a constant) implies large derivatives if datapoints are close
- hence we define
  $$ r_i = \min( \min_{i \neq j} \| X_i - X_j \|, \operatorname{dist}(X_i, \Omega))$$
  - analyzing these RVs is the core of the proof
  - known: $\mathbb{E}[r_i] \lesssim n^{-1/2}$ (see [this book](https://link.springer.com/book/10.1007/b97848)) but we need stronger result
- Overview of the proof (which, in general, is very technical):
  - in odd dimension $d$ the RKHS norm has an explicit form equal to a Sobolev one, which is basically done by observing that the eigenfunctions of the integral operator related to the Laplace kernel are Fourier transforms of the kernel basis function; the odd $d$ is needed to simplify the appearing Gamma functions
  - using this expression, one can control "smoothness"; this together with the constant difference between estimator and unknown function makes it possible to lower bound the squared loss in small regions around the datapoints, BUT only for $c$ small enough
  - for big $c$, the RKHS norm approximates $L^2$ norm in $\R^d$, from which one can derive that the $L^2$ norm of the estimator is within a constant fraction of $f_0$, which implies lower bound of total squared loss
  - without being obvious these two bounds cover all possible $c$
- Some keywords of inequalities used within the proof (TO REVIEW): Gigliardo-Nirenberg interpolation, Morrey, local Hölder continuity around samples
