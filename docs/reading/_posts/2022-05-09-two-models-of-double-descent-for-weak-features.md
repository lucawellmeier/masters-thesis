---
layout: post
title: Two Models of Double Descent for Weak Features
year: 2020
authors: Belkin, Hsu and Xu

---

Present and do the computations for a Gaussian and Fourier model that exemplify well the previously proposed double-descent curve two understand the modern regime (way more parameters than samples) better.

<!--more-->

- available at: [paper on arxiv](https://arxiv.org/abs/1903.07571)
- "Double descent" risk curve proposed to describe out-of-sample performance of variably parametrized machine learning models by ADD & READ REFERENCE
  - model with $p$ parameters is fit to a sample of size $n$; risk as a function of $p$
  - often we have a descent toward both sides away from $p=n$ with an "explosion" when they coincide (often to $\infty$)
  - classically: sweet spot of $p \in [0,n]$ for B-V trade-off
  -  modern: training risk 0 but test risk decreases as $p$ grows beyond $n$ (even when fitting noisy data) when using a **suitable inductive bias** like a **least-norm solution**
  - not always the case: but seems like in the modern regime with more parameters and noise it is
- **Gaussian** model: **linear regression** problem
  - $y = \bold{x}^T \bold{\beta} + \sigma \epsilon = \sum_{j=1}^D x_j \beta_j + \sigma \epsilon$
  - here $\bold x$ is $D$-dimensional data from a standard Gaussian, $\epsilon$ independent from $\bold x$ is standard Gaussian noise scaled by some positive $\sigma$, and take $n$ IID copies of $(\bold x, y)$ with indexed by superscripts. $\bold X$ be the $n \times D$ design matrix and $\bold y$ the sampled response vector
  - for $T \subset \{1,\dots,D\}$ with cardinality $p$ let 
    $$\hat{\beta}_T = \bold{X}_T^\dagger \bold{y}$$
    be the usual minimum-norm least-squares solution with all-zeros for the rest of the indices: $\hat{\beta}_{T^c} = 0$. These two compose $\hat{\beta}$
  - under these assumptions, we can find explicit expressions for the risk $\mathbb{E}[(y - \bold{x}^T \hat \beta)^2]$
    - if $p \leq n - 2$, then the risk is
      $$(\| \beta_{T^c} \|^2 + \sigma^2) \cdot \left( 1 + \frac{p}{n-p-1} \right)$$
    - if $n-1 \leq p \leq n+1$, then the risk is $\infty$
    - and finally if $p \geq n+2$, then the risk is 
      $$\| \beta_T \|^2 \cdot \left(1 - \frac np\right)+  (\| \beta_{T^c} \|^2 + \sigma^2) \cdot \left( 1 + \frac{p}{n-p-1} \right)$$
  - the risk explodes at $n=p$ descents when going into the underparametrized domain, but rises again as $p$ approaches 0. But it descents normally in the overparametrized domain
  - a straightforward corollary is obtained by sampling $T$ uniformly: in this case, however, the rise of risk for $p$ close to 0 does not happen
  - Proof: the "classical regime" is well-known. The modern regime might be very interesting for me to know. TODO Synthesis. Some keywords: decompose by algebraic property of **pseudo-inverse**, find a orthogonal projection, rotational symmetry of standard normal distribution, **trace trick**, **$\chi^2$ distribution**, **Inverse-Wishart distribution**
  - In this case the authors also prove concentration. For $t > 0$ with probability at least $1-\exp(-t)$:
    $$\left\lvert \lVert\beta_T\rVert^2 - \frac pD \lVert\beta\rVert^2 \right\lvert \leq \lVert\beta\rVert^2 \left(\sqrt{2\left( \mu^2 - \frac 1D \right) \min\left\{ \frac pD, 1 - \frac pD \right\} t} + \frac{2\mu^2t}{3} \right)$$
    - $\mu = \max_{i \in [D]} \lvert \beta_i \rvert / \lVert \beta \rVert$ has range $[1/\sqrt D, 1]$. Crucial: small when there are many relevant "weak" features (each with small coefficient in $\beta$). Large when $\beta$ is concentrated on a sparse subset of features
- The analyze a Fourier model afterwards. The regression coefficients are given by by random choice of DFT matrix rows and columns (similar to before).  I will skip this for now. Could be a nice computational exercise.
- Conclusion and discussion: 
  - when features are chosen uniformly, it is often good to chose as many as possible (even though way more than data).
  - Done in NNs: in speech, image and signal features are weak individually but come in abundance
  - contrast to applications where features are chosen with care: medical context e.g. Also in the prescient Gaussian that selects the best number of features t balance bias and variance can be better than the cost of using all features
  - classical regime deeply explored. Sharp divide, however, between that and the new one; the latter is emerging only right now
  - Goal: develop best practices in model and feature selection in this regime; depending on exactly these kinds of analyses
