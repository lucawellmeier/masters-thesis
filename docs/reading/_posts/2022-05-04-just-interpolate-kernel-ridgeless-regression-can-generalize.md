---
layout: post
title: "Just Interpolate: Kernel \"Ridgeless\" Regression Can Generalize"
year: 2019
authors: Liang and Rakhlin

---

Data-dependent risk bound for high-dimensional kernel ridge regression, where the regularization parameter is set to 0 - which usually means interpolation. The assumptions are quite strong, but the idea shows that MNIST is eligible, which is proven experimentally.

<!--more-->

- available at: [paper on arxiv](https://arxiv.org/abs/1808.00387)
- Background
  - explicit regularization is by convention added to least-squares objective when in a high- (or infinite-) dimensional space:
    $$\min_{f\in\mathcal H} \frac 1n \sum (f(x_i) - y_i)^2 + \lambda \lVert f \rVert^2$$
  - Recent empirical observations show that best results are achieved without any regularization at all (i.e. $\lambda = 0$)
- Setup: **Kernel Ridge Regression** 
  - observe IID pairs $(x_i, y_i) \in \Omega \times \R$ with $\Omega \subset \R^d$ compact, drawn from unknown distribution $\mu(x,y)$. Matrix $X$ with rows $x_i$ and $Y$ the response vector
  - estimate conditional expectation function $\mathbb{E}[\bold y \mid \bold x = x]$, which is assumed to lie in a RKHS $\mathcal H$ with kernel $K(\cdot,\cdot)$, with the **minimal norm interpolation estimator**
  - Notation: $K$ will be applied entry-wise on matrix and vector arguments.
  - We assume that $K(X,X)$ has full-rank, in which case the estimator can be written explicitly as: 
    $$\hat f(x) = K(x,X) K(X,X)^{-1} Y$$
  - by reproducing property let $K_x \colon \R \to \mathcal H$ such that $f(x) = \langle K_x, f \rangle_{\mathcal H} = K_x^\ast f$ from which follows that $K(x,z) = \langle K_x, K_z \rangle = K_x^\ast K_z$
  - let $\mathcal{T}_\mu \colon L^2 \to L^2$ wrt the [marginal measure](https://en.wikipedia.org/wiki/Marginal_distribution) $\mu(x)$ be defined by
    $$\mathcal{T}_\mu f(z) = \int K(z,x) f(x) \, d\mu(x)$$
    with eigenfunctions $e(x) = \{e_1(x), \dots, e_p(x)\}$ (possibly infinite)
  - we have $\mathcal{T} e_i = t_i e_i$ and $\int e_i(x) e_j(x) \, d\mu(x) = \delta_{ij}$, so that we can write a spectral decomposition with $T = \operatorname{diag}(t_1, \dots, t_p)$ as
    $$K(x,z) = e(x)^\ast \, T \, e(z),$$
    - also yields a new expression for the interpolation estimator plugging this expression in there
  - by extending $K_x$ to $K_X \colon \R^n \to \mathcal H$ we define the sample version of the kernel operator by 
    $$ \hat{\mathcal T} = \frac 1n K_X K_X^\ast = T^{1/2}\left(\frac 1n e(X) e(X)^\ast\right)T^{1/2} $$
    - eigenvalues $\lambda_j(\hat{\mathcal T})$ same as those of $\frac 1n K(X,X)$
- Main result holds under some assumptions
  - **High-dimensionality**: there are constants such that $c \leq d/n \leq C$. Let $\Sigma_d = \mathbb{E}_\mu [x_i x_i^\ast]$ covariance matrix with operator norm $\leq 1$
  - **$(8+m)$-moments**: IID random vectors $z_i = \Sigma_d^{-1/2} x_i$ such that their entries $z_i(k)$ are drawn IID with mean zero and unit variance and $\lvert z_i(k) \lvert \leq C d^{2\, / \, (8 + m)}$
    - the IID assumption is strong and should be relaxed in future work
    - existence of $(8+m)$-moments is enough to ensure the last condition but it is relatively weak: in bounded or subgaussian case, $m = \infty$
  - **Noise condition**: $\mathbb{E}[(f^\ast(\bold x) - \bold y)^2 \mid \bold x = x] \leq \sigma^2$ for all $x\in\Omega$
  - **Non-linear kernel**: $K(x,x) \leq M$ for all $x$ and $K(x,x') = h(\langle x, x' \rangle / d)$ is an inner-product kernel with $h$ non-linear smooth in a neighborhood of 0
    - can be extended to RBF kernels using results from [El Karoui, 2010](https://arxiv.org/abs/1001.0492)
    - some curvature related quantities:
      $$\alpha = h(0) + h''(0) \frac{\operatorname{tr}(\Sigma_d^2)}{d^2}, \quad \beta = h'(0),$$
      $$\gamma = h\left(\frac{\operatorname{tr}(\Sigma_d)}{d}\right) - h(0) - h'(0) \frac{\operatorname{tr}(\Sigma_d)}{d}$$
  - Under these assumptions and for $d$ large enough, with probability at least $1-2\delta-d^{-2}$ given $X$ we have the **data-dependent bound**:
    $$\mathbb{E}_{Y\mid X} \lVert \hat f - f_\ast \rVert^2_{L_\mu^2} \leq \phi_{n,d}(X,f_\ast) + \epsilon(n,d)$$
    where
    $$\begin{aligned} &\phi_{n,d}(X,f_\ast) = \bold V + \bold B \\
    &\bold V = \frac{8\sigma^2\lVert\Sigma_d\rVert_{\textrm op}}{d} \sum_j \frac{\lambda_j \left(\frac{XX^\ast}{d} + \frac{\alpha}\beta \mathbb{1} \mathbb{1}^\ast\right)}{\left(\frac{\gamma}{\beta} + \lambda_j \left(\frac{XX^\ast}{d} + \frac{\alpha}\beta \mathbb{1} \mathbb{1}^\ast \right)\right)^2} \\
    &\bold B = \lVert f_\ast \rVert_{\mathcal H}^2 \inf_k \left\{\frac 1n \sum_{j>k} \lambda_j (K_X K_X^\ast) + 2M \sqrt{\frac kn} \right\} \\
    &\epsilon(n,d) = O(d^{-\frac{m}{8+m}} \log^{4.1} d) + O(n^{-1/2} \log^{0.5}(n/\delta)) \end{aligned}$$
- What's used in the proof:
  - straightforward variance-bias decomposition
  - variance:
    - using noise assumption: $\bold V \leq \sigma^2 \mathbb{E} \lVert K(X,X)^{-1} K(X,\bold x) \rVert^2$ with expectation over the marginal $\bold x \sim \mu$
    - for specific linearizations $K^{\textrm{lin}}(X,X)$ and $K^{\textrm{lin}}(X,x)$ defined using the curvature quantities, the difference in norm between them and their counterparts is nicely bounded with high probability (using concentration inequalities and kernel matrix approximations from the paper above)

  - bias
    - better to work in frequency domain using the spectral decomposition
    - with the top $k$ columns $\hat{U}_k$ of the orthogonal matrix of the SD of the empirical kernel operator, one finds:
      $$\bold B \leq \lVert f_\ast \rVert_{\mathcal H}^2 \int g_{\hat{U}_k}(x) \, d\mu(x) \\ g_{U_k}(x) = \left\lVert P^{\textrm{ortho}}_{U_k}\left( T^{1/2}e(x) \right) \right\rVert^2 = \operatorname{tr}(e^\ast(x) T^{1/2}U_k U_k^T T^{1/2} e(x)$$
    - authors having been inspired by [Kernel Methods for Pattern Analysis](https://www.cambridge.org/core/books/kernel-methods-for-pattern-analysis/811462F4D6CD6A536A05127319A8935A#:~:text=Kernel%20methods%20provide%20a%20powerful,classifications%2C%20regressions%2C%20clusters), they bound the bias by employing empirical process theory bounding the Rademacher complexity of the set of all such functiont $g_{U_k}$ with matrices that satisfy the orthogonality condition (non-square though)
- Further investigations and conclusions:
  - they kept the bound data-dependent as a sort-of sanity check if your given problem could work well with interpolation
  - variance low when the data matrix has a certain decay of eigenvalues ("favorable geometric properties of the data")
  - bias low when the eigenvalues of the kernel matrix decay very fast or the kernel matrix is "effectively low-rank"
  - the last two points were illustrated by applying the bounds on simple settings
    - specializing to the cases $n>d$ / $n<d$, $\bold x$ gaussian, low rank by fixing diagonal covariance with only a portion non-zero, approximative low-rank by putting many small values $\epsilon$ in the diagonal covariance matrix, fixing specific slow-decaying eigenvalue structure, fixing the trace
    - results are asymptotic here, partially found using Random Matrix Theory
    - numerical experiment with synthetic data

  - numerical experiment indicating that MNIST is eligible for the findings... Indeed, no-regularization performs best almost always. I have reproduced this experiment: [see here](https://github.com/lucawellmeier/krrc-mnist-smart-cache)
