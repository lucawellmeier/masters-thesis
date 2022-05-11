---
layout: post
title: To understand Deep Learning we need to understand Kernel Learning
year: 2018
authors: Belkin, Ma and Mandal

---

Similarities between NNs and kernel regression are exposed. Several experiments on kernel regression are conducted, revealing some favorable behavior of the Laplacian kernel. Moreover, the authors argue that existing generalization bounds don't work for interpolating estimators since their norms (i.e. the complexity) can grow exponentially while performing still well.

<!--more-->

- available at [paper on arxiv](https://arxiv.org/abs/1802.01396#)
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
  - they've run Gaussian kernel and Laplacian kernel regression classification with either direct inversion of the kernel matrix, i.e. interpolating, or with [EigenPro-SGD](https://scikit-learn-extra.readthedocs.io/en/stable/modules/eigenpro.html) with up to 20 epochs on 6 datasets
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
- **ReLU NNs and Laplacian/Gaussian kernels**
  - NNs can fit label noise easily with only three times the SGD epochs as in the noise-less case
    - very similar behavior for the Laplacian kernel
    - note that they are not smooth with a discontinuity like the ReLU function

  - Gaussian kernels need many more iterations
  - generalization performance of classifiers appear to be related to properties of the kernels rather than their properties wrt to the optimization methods (e.g. SGD)
  - Authors believe that Laplacian kernels "hold significant promise for future work on scaling to very large data"
