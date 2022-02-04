from typing import List, Tuple, Any, Callable, TypeVar, Optional, Generic, Dict
import scipy as scp
import scipy.special as scpspec
import math
from utils import *

A = TypeVar('A')


class Support(Generic[A]):
    values: List[A]
    logits: List[float]
    probs: List[float]

    def __init__(self, values, logits, probs):
        self.values = values
        self.logits = logits
        self.probs = probs


class Distrib(Generic[A]):
    _sample: Callable[[], A]
    _logpdf: Callable[[A], float]
    _mean: Optional[Callable[[], float]]
    _var: Optional[Callable[[], float]]
    _samples: List[A]
    _support: Optional[Support[A]]

    def __init__(self, sample, logpdf, mean=None, var=None, support=None,
                 n=10000):
        samples = [sample() for i in range(n)]

        self._sample = sample
        self._logpdf = logpdf
        self._mean = mean
        self._var = var
        self._samples = samples
        self._support = support

    def draw(self):
        return self._sample()

    def get_samples(self):
        return self._samples

    def get_support(self, shrink=False):
        if not shrink:
            return self._support
        else:
            values = self._support.values
            probs = self._support.probs
            values, probs = shrink(values, probs)
            return Support(values, [math.log(x) for x in probs], probs)

    def logpdf(self, x):
        return self._logpdf(x)

    def mean_generic(self, transform) -> float:
        if self._mean is not None:
            return self.mean()
        elif self._support is not None:
            values = scpspec.logsumexp 
        else:
            pass

import scipy.stats as sp

def bernoulli(p):
    assert(0 <= p <= 1)
    sp.bernoulli.

def binomial(p, n):
    pass

def dirac():
    pass

def support():
    pass

def beta():
    pass

def gaussian():
    pass

def uniform():
    pass