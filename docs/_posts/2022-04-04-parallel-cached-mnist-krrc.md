---
layout: post
title: "Parallel and cached computations for MNIST kernel ridge regression experiment"

---

This is a reproduction of a numerical experiment presented in the 2019 paper [Just Interpolate: Kernel "Ridgeless" Regression Can Generalize](https://arxiv.org/abs/1808.00387) by Tengyuan Liang and Alexander Rakhlin. They argue that interpolating, i.e. having a very small to zero training error, can perform well under certain circumstances, and they analyze asymptotically when this happens in the case of kernel ridge regression. The usage of kernels gives enough freedom to allow interpolation, which is expected to happen in kernel ridge regression if we don't regularize by setting $\alpha = 0$ ("ridgeless"). I will go into the details of how to see this phenomenon using `scikit-learn`, exploring parallel computations and smart caching of intermediate products in long computations. 

<!--more-->

The complete code is available for download from [this GitHub repo](https://github.com/lucawellmeier/krrc-mnist-smart-cache). To run it yourself, you'll need `Python 3.10` and `scikit-learn 1.0.2` or compatible versions. My machine is a ThinkPad T450s with Intel Core i7 5600U with 2 cores (4 threads), 2.6 GHz base frequency and 11 GB of RAM running ArchLinux. The timings refer to this.

This is not supposed to be a guide and I do not claim to employ best practices in the field (which I don't do neither know yet). I am still learning. Please feel free to contact me <lucwellm@gmail.com> for any suggestions, questions, ...

**Contents**
* table of contents
{:toc}

## The Problem and Computation using `scikit-learn`

The challenge is to classify handwritten digits which can be part of either of two classes from the MNIST dataset. This shall be done using kernel ridge regression: we label one digit by 1, the other one by -1, perform the regression on a new unknown drawing and receive a number between -1 and 1, which gives an estimate which class would be closer. The used kernel will be standard Gaussian (radial basis function) and we measure the error with the following metric (called `error_metric` in the code below):

$$ \frac{\sum_{i=1}^N  (\hat{f}(x_i) - y_i)^2}{\sum_{i=1}^N  (\mathbb{E}[y_i] - y_i)^2} $$

The following code describes how that would generally look like in `scikit-learn`:

```python
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge

def my_error_metric(Ytrue, Ypred):
    ybar = np.average(y_true);
    return np.sum( (y_pred - y_true)**2 ) / np.sum( (ybar - y_true)**2 )

digitA = 2
digitB = 5
Ntrain = 2000
Ntest = 500
alpha = 0.3

# step1: fetch MNIST dataset (this takes a few minutes) and prepare two-class sample
mnist = fetch_openml('mnist_784')
all_digits = mnist.data.to_numpy() / 255.0  # normalize; otherwise kernel produces values too close to zero
two_class_data = all_digits[mnist.target == str(digitA) or mnist.target == str(digitB)]
labels = np.zeros(all_digits.shape[0])
labels[mnist.target == str(digitA)] = 1
labels[mnist.target == str(digitB)] = -1
labels = labels[labels != 0]

# shuffle and take only what we need
N = Ntrain + Ntest
perm = np.random.RandomState().permutation(two_class_data.shape[0])
Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    two_class_data[perm][:N], labels[perm][:N], test_size=Ntest)

# train model and predict
model = KernelRidge(alpha=alpha, kernel='rbf_kernel')
model.fit(Xtrain, Ytrain)
train_error = my_error_metric(Ytr, model.predict(Xtr))
test_error = my_error_metric(Yte, model.predict(Xte))
```

Since we will also consider the spectrum of the kernel matrix, it makes sense to take its computation in our own hands:

```python
from sklearn.metrics.pairwise import rbf_kernel

train_kernel = rbf_kernel(Xtrain, Xtrain)
test_kernel = rbf_kernel(Xtest, Xtrain)

# step spectrum
spec,_ = np.linalg.eig(train_kernel)

model = KernelRidge(alpha=alpha, kernel='precomputed')
model.fit(train_kernel)
train_error = my_error_metric(Ytr, model.predict(train_kernel))
test_error = my_error_metric(Yte, model.predict(test_kernel))
```

Here's what we want to compute with `Ntrain = 6000` and `Ntest = 2000`:

- kernel spectrum for all 45 pairs of digits
- covariance spectrum of the image data of all 45 pairs of digits
- both train and test error for all 45 pairs of digits and `alpha = 0, 0.01, 0.02, 0.04, 0.08, 0.1, 0.16, 0.32, 0.64, 1, 1.28`

For reference: each image has size 28 x 28 = 784.

## Setup of Parallelism and Smart Cache for the Computations

As I was starting to implement and compute, I immediately ran into some quite nasty problems: long computation times, huge amounts of memory and disk space needed. Hence I decided to use this as an opportunity to experiment with parallelism and to develop a sort of "smart cache" that keeps byproducts stored on the disk only as long as they are needed and chooses the next computational steps in an educated way. The final code to compute the above targets would look like this:

```python
from dep_cache import ParallelDependencyCache

cache = ParallelDependencyCache('/home/luca/krrc_mnist_cache/')

Ntr = 6000
Nte = 2000
alphas = [0, 0.01, 0.02, 0.04, 0.08, 0.1, 0.16, 0.32, 0.64, 1, 1.28]

for i in range(10):
    for j in range(i+1,10):
        cache.request(CovSpec(Ntr,Nte,i,j))
        cache.request(KernelSpec(Ntr,Nte,i,j))
        
        for alpha in alphas:
            cache.request(Errors(Ntr,Nte,i,j,alpha))

cache.fetch(n_threads=4)
```

The three classes `CovSpec`, `KernelSpec` and `Errors` are subclasses of the class `CachedFile` (see `dep_cache.py` in the [repo](https://github.com/lucawellmeier/krrc-mnist-smart-cache)), which needs them to define parameters, dependencies, computation and save/load. Each `request` call extends an internal dependency polytree. For example, the following simplified preparation

```python
cache.request(CovSpec(Ntr, Nte, 2, 5))
cache.request(KernelSpec(Ntr, Nte, 2, 5))
cache.request(CovSpec(Ntr, Nte, 4, 5))
cache.request(KernelSpec(Ntr, Nte, 4, 5))

cache.request(Errors(Ntr, Nte, 2, 5, 0))
cache.request(Errors(Ntr, Nte, 2, 5, 0.1))
cache.request(Errors(Ntr, Nte, 2, 5, 1))

cache.request(Errors(Ntr, Nte, 4, 5, 0))
cache.request(Errors(Ntr, Nte, 4, 5, 0.1))
cache.request(Errors(Ntr, Nte, 4, 5, 1))
```

produces the following messy polytree (dropping the repetitive declaration of `Ntr` and `Nte`):

<div class="mermaid">flowchart LR
    MNIST --> Digit_2
    MNIST --> Digit_4
    MNIST --> Digit_5
    Digit_2 & Digit_5 --> TrainTestSplit_2_5
    Digit_4 & Digit_5 --> TrainTestSplit_4_5
    TrainTestSplit_2_5 --> CovSpec_2_5
    TrainTestSplit_4_5 --> CovSpec_4_5
    TrainTestSplit_2_5 --> Kernels_2_5
    TrainTestSplit_4_5 --> Kernels_4_5
    Kernels_2_5 --> KernelSpec_2_5
    Kernels_4_5 --> KernelSpec_4_5
    Kernels_2_5 & TrainTestSplit_2_5 --> Errors_2_5_00
    Kernels_2_5 & TrainTestSplit_2_5 --> Errors_2_5_01
    Kernels_2_5 & TrainTestSplit_2_5 --> Errors_2_5_10
    Kernels_4_5 & TrainTestSplit_4_5 --> Errors_4_5_00
    Kernels_4_5 & TrainTestSplit_4_5 --> Errors_4_5_01
    Kernels_4_5 & TrainTestSplit_4_5 --> Errors_4_5_10
</div>

Each node here represents a file that can be cached for future computations. After the declaration, one calls `fetch` to actually start producing the data. This happens in parallel (in the above example in 4 threads). Each of the workers will continuously be assigned new tasks, i.e. new files to produce, according to the following rule: 

> Among all the tasks that haven't been touched yet look for those which have all dependencies satisfied. Sort them after their tree depth and choose one of those with the highest depth.

In this way, the algorithm will go deep as fast as possible which also means that the intermediate dependencies can be deleted very timely. In fact, the cached data used was constantly low. In the beginning I have underestimated the size of some objects - even when compressed - which resulted in an "out of disk space" error after the cache contained more than 100 GB of data (not even close to the finish). The same experiment with the new algorithm kept the usage below 2 GB all the time. The clear drawback of this method is the need to write, read, compress and decompress quite frequently. But since RAM usage was constantly at about 4 GB, there is certainly the possibility to keep objects longer (might be added in the future). It is also worth noting, that the computation can be interrupted and restarted without problems since the current cache contains always exactly what is needed for the next steps.

The final code above took about 15 hours to compute on my machine. However, I switched from 3 to 4 cores in the middle of the computation and at that time I also had another intermediate object for a dump of the model, which weighed more than 300 MB when compressed, and whose (de)compression took a considerable amount of computing time. Therefore, I expect the real time to be way less. It will be added here as soon as I have repeated the experiment.

## Reproduction of Figures of the Paper

That's about the original part of this post. Now, only for reference, we're recreating the figures presented in the paper. Again, the Jupyter notebook can be found in the [repo](https://github.com/lucawellmeier/krrc-mnist-smart-cache).

### Training and Test Errors

![MNIST KRRC training error](/assets/img/train_errors.png)

As expected, the ridgeless regression completely learns the training data ...

![MNIST KRRC test errors](/assets/img/test_errors.png)

... but surprisingly the test error curve behaves similarly. In all except one of the digit pairs in the graph above ridgeless regression performs the best. In fact, among all the 45 pairs the following have the best test performance for $\alpha$ very small but not zero: `(1,2)`, `(1,4)`, `(1,7)`, `(4,9)`. I suppose this is due to different writing styles of the digits 1, 4 and 7, while the last case is simply the closeness that they might have sometimes.

### Covariance and Kernel Spectra

![MNIST covariance spectrum](/assets/img/spectrum_covariance.png)

![MNIST kernel spectrum](/assets/img/spectrum_kernel.png)

Both graphs are reasonably similar to the ones found by the authors, only that the maximal eigenvalues were a little smaller.
