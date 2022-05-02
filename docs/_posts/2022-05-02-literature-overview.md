---
layout: post
title: "Literature overview: Benign Overfitting"
published: 2022-04-30

---

This is an overview over my personal progress in working through the literature related to the concept of Benign Overfitting. Papers that I consider interesting will be put in the "Potential next reads" section. As soon as I have done my review I put them at the end of the "My notes" section with a concise summary of the main ideas, the framework and hypotheses.

<!--more-->

**Contents**
* table of contents
{:toc}
## Potential next reads (unsorted)

- [Rakhlin, Zhai 2018] Consistency of Interpolation with Laplace Kernels is a High-Dimensional Phenomenon
- [Hastie, Montanari, Rosset, Tibshirani 2019] Surprises in High-Dimensional Ridgeless Least Squares Interpolation
- [Belkin 2018] Approximation beats Concentration? An approximation view on inference with smooth radial kernels
- [Belkin, Rakhlin, Tsybakov 2018] Does data interpolation contradict statistical optimality?
- [Liang, Rakhlin 2019] Just Interpolate: Kernel "Ridgeless" Regression Can Generalize
- [Belkin, Hsu, Xu 2019] Two models of double descent for weak features
- [Jacot, Gabriel, Hongler 2018] Neural Tangent Kernel: Convergence and Generalization in Neural Networks
- [Du, Zhai, Poczós, Singh 2019] Gradient Descent Provably Optimizes Over-parametrized Neural Networks
- [Du, Lee, Li, Wang, Zhai 2019] Gradient Descent Finds Global Minima of Deep Neural Networks
- [Allen-Zhu, Li, Song 2019] A Convergence Theory for Deep Learning via Over-Parametrization
- [Chizat, Bach 2018] A Note on Lazy Training in Supervised Differentiable Programming
- [Cao, Chen, Belkin, Gu 2022] Benign Overfitting in Two-layer Convolutional Neural Networks
- [Adlam, Pennington 2020] The neural tangent kernel in high-dimensions: Triple descent and a multi-scale theory of generalization
- [Li, Zhou, Getton 2021] Towards an understanding of benign overfitting in neural networks
- [Wu, Xu 2020] On the optimal weighted $\ell_2$ regularization in overparametrized linear regression
- [Chatterji, Long 2020] Finite-sample analysis of interpolating linear classifiers in the overparametrized regime
- [Zou, Wu, Braverman, Gu, Kakade 2021] Benign overfitting of constant-stepsize sgd for linear regression
- [Cao, Gu, Belkin 2021] Risk bounds for over-parametrized maximum margin classification on sub-gaussian mixtures
- [Montanari, Zhong 2020] The interpolation phase transition in neural networks: Memorization and generalization under lazy training
- [Hofmann, Schölkopf, Smola 2008] Kernel Methods in Machine Learning

## Other (remotely related but) interesting material

- [Scalable Kernel Methods via Doubly Stochastic Gradients](https://www.youtube.com/watch?v=1eA9XN7gk20&list=WL&index=1) (YouTube). Seminar at Microsoft Research presented by Le Song in 2016. Presents a new method on how to speed up SGD in kernel methods by adding a second component of randomness, which is made efficient by memorizing specific seeds of a random generator. Comparison with large-scale image recognization experiment with a NN, and the new method performs very reasonably but still very much not perfectly

## My notes

### `[2020 | Bartlett, Long, Lugosi, Tsigler]` Benign Overfitting in Linear Regression

Links: [paper on arxiv](https://arxiv.org/abs/1906.11300), [paper in PNAS](https://www.pnas.org/doi/10.1073/pnas.1907378117), [lecture from Alan Turing Institute on YouTube](https://www.youtube.com/watch?v=HnWdKzgfVTQ)

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
    - Relation to norms: lower is sort of $\frac{\ell_1}{\ell_\infty}$, upper is $\frac{\ell_1^2}{\ell_2^2}$
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

### `[2018 | Belkin, Ma, Mandal]` To understand deep learning we need to understand kernel learning

Links: [paper on arxiv](https://arxiv.org/abs/1802.01396#)

- How to measure **generalization performance of a classifier**?
  - Assume standard setting of Empirical Risk Minimization wrt some training data, a class of functions and a loss function
  - Most approaches: control/analyze capacity/complexity $c$ of function space ([VC dimension](https://en.wikipedia.org/wiki/Vapnik%E2%80%93Chervonenkis_dimension), [fat shattering dimension](https://www.cambridge.org/core/books/abs/neural-network-learning/pseudodimension-and-fatshattering-dimension/68B7608D896234212964C3267A3B5181), [Rademacher complexity](https://en.wikipedia.org/wiki/Rademacher_complexity), covering numbers)
    - often yield a bound for the generalization gap, i.e. the difference between expected total loss of the minimizer $f^\ast$ of the total loss and the empirical one on the training data; often of the form $O\left(\sqrt{c/n} \right)$
- Deep learning employs parameter counts that exceed the size of the training data by orders of magnitude; allows for zero training error through over-parametrization and generalizes well for some reason
  - here: **overfitting** = zero classification error in training, **interpolation** = zero regression error
- **Kernel regression/machines** = linear regression in [RKHS](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space) = two-layer NN with a fixed first layer
  - relevant kernels in the paper: [smooth Gaussian, non-smooth Laplacian](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space#Common_examples)
  - the near-optimality of overfitting or interpolating kernel classifiers has not been recognized yet in the literature 
  - Very nice and short summary of kernel methods presented in Section 2; explains also where SGD can be used; try to write a synthesis of kernel methods, as linear regression in RKHS and as two-layer NNs based on this intro... also with computational aspects
- Numerical experiments on **SGD vs interpolation and Gaussian vs Laplacian**:
  - they've run Gaussian kernel and Laplacian kernel regression classification with either direct inversion of the kernel matrix, i.e. interpolating, or with [EigenPro-SGD](https://scikit-learn-extra.readthedocs.io/en/stable/modules/eigenpro.html) with up to 20 on 6 datasets
    - [MNIST](http://yann.lecun.com/exdb/mnist/), [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), [SVHN](http://ufldl.stanford.edu/housenumbers/), [TIMIT](https://catalog.ldc.upenn.edu/LDC93s1), [HINTS](https://hints.cancer.gov/data/default.aspx), [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/) (maybe interesting for future experiments)
    - all achieve nearly zero classification error; shown quantities are regression error in training and both classification and regression error in testing
    - Laplace is more precise in generalization everywhere except in MNIST (a dataset that is almost deterministic since the digits are usually clearly separable)
    - Gauss+EigenPro actually outperforms Gauss+Interpolation after a certain epoch; this never happens with the Laplacian 
    - Key Observation: even though we are interpolating, the generalization performance is remarkable and actually this is very comparable to deep networks; hence this phenomenon itself cannot be only *deep*

- Existing **generalization bounds for kernel methods** do not capture this
  - Setup: $(x_i, y_i) \in \Omega \times \{-1,1\}$, $\Omega$ bounded in $\R^d$ with loss of [Bayes optimal classifier](https://en.wikipedia.org/wiki/Bayes_classifier) not zero (i.e. there is some label noise); $y$ is not a deterministic function of $x$ in some set of measure non-zero; $\mathcal H$ RKHS with Gaussian kernel (or other smooth ones)
    - we say that $h \in \mathcal H$ $t$-overfits the data, if zero classification loss and $y_i h(x_i) > t > 0$ for at least a fixed portion of the training; necessary because a zero classification loss classifier with arbitrarily small norm can be obtained by scaling any interpolating solution

  - generalization bounds are generally of the form
    $$\left| \frac 1n \sum_i l(f(x_i), y_i) - \mathbb{E}_P[l(f(x),y)] \right| \leq C_1 + C_2 \frac{\lVert f \rVert_{\mathcal{H}}^\alpha}{n^\beta}$$
  - the authors prove in the above setup that, with high-probability,
    $$\lVert h \rVert_{\mathcal H} > A \exp(B n^{1/d})$$
    - Proof idea: take open ball $B_R$ in $\mathcal H$ and prove that it cannot contain $t$-overfitting functions with high probability unless $R$ is very big
      - use a generalization bound coming from the fat shattering dimension $\operatorname{fat}_\gamma(B_R)$ (see Section 13.2 [in this book](https://www.cambridge.org/core/books/abs/neural-network-learning/sample-complexity-of-classification-learning/71470A997FA6F19971DD52B417381703)), use the overfitting property and "not deterministic implies non-zero expectation" to bound the expected loss from below by a small enough $\gamma$ to derive that $B_R$ contains no $t$-overfitting function unless
        $$\operatorname{fat}_\gamma(B_R) > \frac{n}{C_2} \left( \mathbb{E}\left[l(f(x),y)\right] - C_1 \gamma \right)^2$$
      -  Conclude with a specific logarithmic bound $\operatorname{fat}_\gamma(B_R) < O\left(\log^d \, \frac R\gamma \right)$
    - despite the fact that the existing bounds to not apply to interpolated classifiers, this seems to show that such bounds do not exist, since their norms grow exponentially so that the RHS of such a bound would diverge
  - Numerical experiments on **norm magnitudes in interpolation vs overfitting**
    - (easily proved) proposition: adding a label noise fraction $\epsilon$ in $k$-class problems does not change the Bayes optimal classifier; moreover, for the error rates:
      $$P_\epsilon(c^\ast(x) \neq y) = \epsilon \frac{k-1}{k} + (1-\epsilon) P(c^\ast(x) \neq y)$$
    - experiment on separable classes + noise: $x \in \R^{50}$, $x_1 \sim N(0,1)$ if $y = 1$ and $x_1 \sim N(10,1)$ if $y=-1$, all other coordinates uniformily between -1 and 1
      - this is separable with Bayes optimal classifier $c^\ast(x) = \operatorname{sign}(x_1 - 5)$
      - for different noise levels (0%, 1% and 10%) both overfitting and interpolating classifiers approach the Bayes optimal error rate easily
      - norms of overfitting solutions are generally way smaller than those of interpolating ones while maintaining the same performance
        - norm and performance have only little connection
    - experiment on non-separable classes as above but now the two possibilities for $x_1$ are $N(0,1)$ and $N(2,1)$; Bayes optimal error rate: 15.9%
      - already noisy, adding extra noise has little effect on the norm
      - accuracy of interpolated classifier is consistently within 5% of the Bayes optimal one
    - similar experiments on real data: noise in the (clean) MNIST has huge effect on the norm, only little in the (messy) TIMIT but test performance
- ReLU NNs and Laplacian/Gaussian kernels
  - NNs can fit label noise easily with only three times the SGD epochs as in the noise-less case
    - very similar behavior for the Laplacian kernel
    - note that they are not smooth with a discontinuity like the ReLU function

  - Gaussian kernels need many more iterations
  - generalization performance of classifiers appear to be related to properties of the kernels rather than their properties wrt to the optimization methods (e.g. SGD)
  - Authors believe that Laplacian kernels "hold significant promise for future work on scaling to very large data"

