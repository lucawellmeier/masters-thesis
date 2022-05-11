---
layout: post
title: Benign Overfitting in Linear Regression
year: 2018
authors: Bartlett, Long, Lugosi and Tsigler

---

Non-asymptotic, tight lower and upper concentration bounds for the risk in overparametrized, linear regression on high- (or infinite-) dimensional subgaussian data.

<!--more-->

- available at: [arxiv](https://arxiv.org/abs/1906.11300), [PNAS](https://www.pnas.org/doi/10.1073/pnas.1907378117), [lecture at Alan Turing Institute on YouTube](https://www.youtube.com/watch?v=HnWdKzgfVTQ)
- Original version published in 2018 only by Bartlett and Long; improved version with all the authors in 2019; I'm referring to the most recent 2020 update
- Background: experimental evidence shows that neural networks perform better than expected when you achieve zero training error (which happens often when using stochastic gradient descent) - even if the labels are noisy
  - Contradicts classical statistical belief, so there must be some sort of "benign" overfitting of the training data
- Simplest setting with hope for finding the phenomenon: **linear regression**
  - Data $x$ in a separable Hilbert space and scalar response $y$
  - $(x,y)$ jointly gaussian with mean-zero and the covariance matrix $\Sigma = \mathbb{E}[x^Tx]$ has rank greater than sample size
    - the rank assumption ensures that we can find many **interpolating estimators** (i.e. those perfectly fitting the training data)
    - presented with weaker subgaussian assumptions
  - Use the unique **least squares minimum norm estimator** $\hat{\theta}$ which is given by
    $$\hat\theta = X^T (X X^T)^\dagger Y,$$
    where $X$  is the training data vector, $Y$ the responses vector and the dagger denotes the [pseudoinverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse)
- Main result: find a characterization by bounding the excess risk $R(\hat \theta)$ in terms of eigenvalue structure of the covariance matrix and the training sample size 
  - using two very related notions of **effective ranks**
    $$r_k(\Sigma) = \frac{\sum_{i > k} \lambda_i}{\lambda_{k+1}}, R_k(\Sigma) = \frac{\left( \sum_{i > k} \lambda_i \right)^2}{\sum_{i > k} \lambda_i^2}$$
    - Relation to norms: for $k = 0$ lower is $\frac{\ell_1}{\ell_\infty}$, upper is $\left(\frac{\ell_1}{\ell_2}\right)^2$
  - Define the critical rank $k^\ast = \min\{k\geq 0 \mid r_k(\Sigma) \geq bn \}$
    - If the critical rank is bigger than $n$, then the expected excess risk is larger than the data variance, so assume the opposite from now on
  - Upper bound: with high probability
    $$R(\hat \theta) \leq c \left( \lVert \theta^\ast \rVert^2 \, \lVert\Sigma\rVert \max\left\{ \sqrt{\frac{r_0(\Sigma)}{n}}, \frac{r_0(\Sigma)}{n}, \sqrt{\frac{\log(1/\delta)}{n}} \right\} \right) \\ + \, c \, \log(1/\delta) \left( \frac{k^\ast}{n} + \frac{n}{R_{k^\ast}(\Sigma)} \right)$$
    - first term coming from 
  - Lower bound:
    $$ \mathbb{E}R(\hat\theta) \geq \frac{\sigma^2}{c} \left( \frac{k^\ast}{n} + \frac{n}{R_{k^\ast}(\Sigma)} \right) $$
    - As a consequence, for large enough $n$ there is an optimal estimator $\theta^\ast$ with  fixed desired norm such that with probability at least 1/4
      $$R(\hat\theta) \geq \frac{1}{c_1} \lVert \theta^\ast \rVert^2 \, \lVert\Sigma\rVert \mathbb{1}\left[ \frac{r_0(\Sigma)}{n\,\log(1+r_0(\Sigma))} \geq c_2 \right]$$
- Proof ideas:
  - Upper bound base: $R(\hat \theta) \leq 2 {\theta^\ast}^T B \theta^\ast + c \sigma^2 \log(1/\delta)\operatorname{tr}(C)$ 
  - Lower bound base: $\mathbb{E}_\epsilon R(\hat \theta) \geq {\theta^\ast}^T B \theta^\ast + \sigma^2 \operatorname{tr}(C)$, which holds with probability at least $1-\delta$ over the noise $\epsilon = Y - X\theta^\ast$
  - The two appearing matrices $B$ and $C$ are products of $\Sigma$ with very symmetric combinations of $X$'s on both sides, so that $\Sigma$ can be seen as a sort of modulator distributing weight between the parameter space directions
  - Both bounds are not difficult to prove; the trace comes into play as described [here](https://math.stackexchange.com/questions/2228398/trace-trick-for-expectations-of-quadratic-forms)
  - The consequential "probability at least 1/4" statement only from the lower bound on the expectation seems to have been given a very interesting proof using algorithmic formulation and packing numbers right in the spirit of [Vershynin](https://www.math.uci.edu/~rvershyn/papers/HDP-book/HDP-book.html)... look it up later and write a synthesis of the arguments used
  - Core of the argument is to control the trace term
    - Can be written as a function of many independent subgaussians defined via the spectral decomposition of $\Sigma$ in a lengthy computation using tools like [Sherman-Morrison formula](https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula)
    - Different sums of subsets of symmetric tensors (matrices of the form $v^T v$) of these subgaussian vectors weighted by their respective eigenvalues are defined; they compose the beforementioned trace formulation and with the help of which it is given a meaning to the lower effective rank by proving concentration of their eigenvalues
      - the concentration is proved by standard arguments for subgaussian and subexponential random variables: [Bernstein](https://en.wikipedia.org/wiki/Bernstein_inequalities_(probability_theory)), [$\varepsilon$-nets](https://terrytao.wordpress.com/tag/epsilon-net-argument/)
      - After some more work one gets then to the upper bound on the trace term in the same spirit
    - The lower bound is not too interesting as it involves more or less straightforward concentration arguments
- Observations and conclusions
  - The two bounds are reasonably tight
  - If $\Sigma_n$ may change in the sample size we call such a sequence **benign** if 
    $$\lim \frac{r_0(\Sigma_n)}{n} = \lim \frac{k_n^\ast}{n} = \lim \frac{n}{R_{k^\ast}(\Sigma_n)} = 0$$
    - $\lambda_k(\Sigma_n) = k^{-\alpha} \ln^{-\beta}(k+1)$ is benign iff $\alpha = 1$ and $\beta > 1$
      - In infinite dimensions with fixed $\Sigma$ overfitting is benign only if the eigenvalue decay is such that they are just barely summable, which is a weird rare phenomenon
    - Consider $\lambda_k(\Sigma_n) = \gamma_k + \epsilon_k$ if $k \leq p_n$ or 0 otherwise, with $\lambda_k = \Theta(\exp(-k/\tau))$. This is benign iff $p_n = \omega(n)$ and $n\exp(-o(n)) = \epsilon_n p_n = o(n)$
      - In very high- but finite-dimensional spaces eigenvalues can decay arbitrarily slow or not at all because the summability is automatic
  - **Overparametrization** (many low-variance - hence unimportant - directions) is essential for benign overfitting
- Future directions
  - What happens in the misspecified case, i.e. $\mathbb{E}[y \mid x]$ not linear in $x$ (the data is not suitable for linear regression)?
  - Can we relax the assumption that the covariates need to be a linear function of a vector of independent RVs? Would help for infinite-dimensional RKHS, for instance
  - Other loss functions besides squared error? Other overfitting estimators beyond the minimum norm interpolating one?
  - Neural networks have uncovered this... Can we say something there?
    - Neural Tangent Kernel: very wide neural networks, trained with SGD from a suitable random initialization can be accurately approximated by linear functions in an appropriate Hilbert space, and SGD finds an interpolating solution quickly here
    - Many similarities but again a problem with the linearity of the covarites
    - All aside, slowly decaying or constant in high but finite-dimensional space might important also here
    - Also, truncating to a finite-dimensional space might be important for the statistical performance in the overfitting regime
