import sys
import os
import bz2
import _pickle as cPickle
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.kernel_ridge import KernelRidge
from dep_cache import CachedFile

class MNIST(CachedFile):
    def __init__(self):
        super().__init__('mnist', (), 'npz', never_delete=True)
    def recover(self):
        d = np.load(self.filename)
        return d['data'], d['labels']
    def create_and_save(self):
        mnist = fetch_openml('mnist_784')
        data = mnist.data.to_numpy() / 255.0
        labels = np.zeros(data.shape[0])
        for i in range(len(labels)):
            labels[i] = int(mnist.target[i])
        np.savez_compressed(self.filename, data=data, labels=labels)

class DigitClass(CachedFile):
    def __init__(self, digit):
        assert(0 <= digit and digit <= 9)
        super().__init__('digit', (digit,), 'npz')
        self.digit = digit
        self.mnist = self.require(MNIST())
    def recover(self):
        return np.load(self.filename)['digit_data']
    def create_and_save(self):
        data, labels = self.mnist.get()
        indices = (labels == self.digit)
        digit_data = data[indices]
        np.savez_compressed(self.filename, digit_data=digit_data)

class TrainTestSplit(CachedFile):
    def __init__(self, Ntr, Nte, digitA, digitB):
        assert(digitA < digitB)
        assert(Ntr + Nte <= 70000)
        super().__init__('traintestsplit', (Ntr,Nte,digitA,digitB,), 'npz')      
        self.Ntr = Ntr
        self.Nte = Nte
        self.digitA = self.require(DigitClass(digitA))
        self.digitB = self.require(DigitClass(digitB))
    def recover(self):
        d = np.load(self.filename)
        return (d['Xtr'], d['Xte'], d['Ytr'], d['Yte'])
    def create_and_save(self):
        dataA = self.digitA.get()
        dataB = self.digitB.get()
        data = np.concatenate((dataA, dataB), axis=0)
        labelsA = np.ones(dataA.shape[0])
        labelsB = - np.ones(dataB.shape[0])
        labels = np.concatenate((labelsA, labelsB))
        perm = np.random.RandomState().permutation(data.shape[0])
        Xtr, Xte, Ytr, Yte = train_test_split(
            data[perm][:self.Ntr + self.Nte], 
            labels[perm][:self.Ntr + self.Nte], 
            test_size=self.Nte)
        np.savez_compressed(self.filename, 
                            Xtr=Xtr,Xte=Xte,Ytr=Ytr,Yte=Yte)

class Kernels(CachedFile):
    def __init__(self, Ntr, Nte, digitA, digitB):
        super().__init__('kernels', (Ntr,Nte,digitA,digitB,), 'npz')      
        self.data = self.require(TrainTestSplit(Ntr,Nte,digitA,digitB))
    def recover(self):
        d = np.load(self.filename)
        return (d['Ktr'], d['Kte'])
    def create_and_save(self):
        Xtr,Xte,_,_ = self.data.get()
        Ktr = rbf_kernel(Xtr, Xtr)
        Kte = rbf_kernel(Xte, Xtr)
        np.savez_compressed(self.filename, Ktr=Ktr,Kte=Kte)    

class CovSpec(CachedFile):
    def __init__(self, Ntr, Nte, digitA, digitB):
        super().__init__('covspec', (Ntr,Nte,digitA,digitB,), 'npz')
        self.data = self.require(TrainTestSplit(Ntr,Nte,digitA,digitB))
    def recover(self):
        d = np.load(self.filename)
        return d['spec']
    def create_and_save(self):
        Xte = self.data.get()[1]
        cov = np.cov(Xte)
        spec = np.linalg.eig(cov)[0]
        np.savez_compressed(self.filename, spec=spec)

class KernelSpec(CachedFile):
    def __init__(self, Ntr, Nte, digitA, digitB):
        super().__init__('kernelspec', (Ntr,Nte,digitA,digitB,), 'npz')
        self.kernels = self.require(Kernels(Ntr,Nte,digitA,digitB))
    def recover(self):
        return np.load(self.filename)['spec']
    def create_and_save(self):
        Ktr = self.kernels.get()[0]
        cov = np.cov(Ktr)
        spec = np.linalg.eig(cov)[0]
        np.savez_compressed(self.filename, spec=spec)

def error_metric(y_true, y_pred):
    ybar = np.average(y_true);
    return np.sum( (y_pred - y_true)**2 ) / np.sum( (ybar - y_true)**2 )
class Errors(CachedFile):
    def __init__(self, Ntr, Nte, digitA, digitB, alpha100):
        super().__init__('errors', (Ntr,Nte,digitA,digitB,alpha100), 'npy')
        self.data = self.require(TrainTestSplit(Ntr,Nte,digitA,digitB))
        self.kernels = self.require(Kernels(Ntr,Nte,digitA,digitB))
        self.alpha = alpha100 / 100
    def recover(self):
        d = np.load(self.filename)
        return d[0], d[1]
    def create_and_save(self):
        _,_,Ytr,Yte = self.data.get()
        Ktr,Kte = self.kernels.get()
        model = KernelRidge(alpha=self.alpha, kernel='precomputed')
        model.fit(Ktr, Ytr)
        Ypr = model.predict(Ktr)
        errors = [error_metric(Ytr,Ypr)]
        Ypr = model.predict(Kte)
        errors.append(error_metric(Yte,Ypr))
        np.save(self.filename, np.array(errors))