from typing import List, Tuple, Any, Callable, TypeVar, Optional, Generic, Dict
from typing_extensions import Protocol
import scipy as scp
import scipy.stats as sp
import scipy.special as scpspec
import math
from math import log, comb, inf
import utils
import matplotlib.pyplot as plt
from collections.abc import MutableSequence

A = TypeVar('A')
B = TypeVar('B', covariant=True)


class Sample(Generic[B], Protocol):
    def __call__(self) -> B:
        pass


class LogPdf(Generic[B], Protocol):
    def __call__(self, x):
        # type : B -> float
        pass


class Support(Generic[A]):
    values: List[A]
    logits: List[float]
    probs: List[float]

    def __init__(self, values, logits, probs):
        self.values = values
        self.logits = logits
        self.probs = probs
    
    def __iter__(self):
        return zip(self.values, self.logits, self.probs)


class LazyList(MutableSequence):
    def __init__(self, f, n):
        self._eval = [False for i in range(n)]
        self._list = [None for i in range(n)]
        self._f = f

    def __len__(self):
        return len(self._list)

    def __getitem__(self, i):
        if self._eval[i]:
            return self._list[i]
        else:
            x = self._f(i)
            self._list[i] = x
            return x

    def __setitem__(self, i, v):
        self._list[i] = v
        self._eval[i] = True

    def __delitem__(self, i):
        del self._list[i]
        del self._eval[i]

    def insert(self, i, v):
        self._list.insert(i, v)
        self._eval.insert(i, True)


class Distrib(Generic[A]):
    _sample: Sample[A]
    _logpdf: LogPdf[A]
    _mean: Optional[Callable[[], float]]
    _var: Optional[Callable[[], float]]
    _samples: LazyList
    _support: Optional[Support[A]]

    def __init__(self, sample, logpdf, mean=None, var=None, support=None,
                 n=10000):
        #samples = [sample() for i in range(n)]
        #samples = sample(size=n)
        samples = LazyList(lambda _i: sample(), n)

        self._sample = sample
        self._logpdf = logpdf
        self._mean = mean
        self._var = var
        self._samples = samples
        self._support = support

    def draw(self) -> A:
        return self._sample()

    def get_samples(self) -> LazyList:
        return self._samples

    def get_support(self, shrink=False) -> Optional[Support[A]]:
        if not shrink:
            return self._support
        else:
            assert (self._support is not None)
            values = self._support.values
            probs = self._support.probs
            values, probs = utils.shrink(values, probs)
            return Support(values, [math.log(x) if x != 0. else -float('inf') for x in probs], probs)

    def logpdf(self, x: A) -> float:
        return self._logpdf(x)

    # def mean_generic(self, transform) -> float:
    #     if self._mean is not None:
    #         return self.mean()
    #     elif self._support is not None:
    #         values = scpspec.logsumexp
    #     else:
    #         pass

    def plot(self, plot_with_support=False, plot_style='scatter'):
        if plot_with_support:
            if self._support is None:
                print("Pas de support à plot")
                return
            supp = self.get_support(shrink=True)
            if plot_style == 'bar':
                plt.bar(supp.values, supp.probs, width=0.05)
            elif plot_style == 'scatter':
                plt.scatter(supp.values, supp.probs)
                plot_y_size = max(supp.probs)
                plt.ylim((-plot_y_size*1/20, plot_y_size*21/20))
            elif plot_style == 'line':
                plt.plot(*zip(*sorted(zip(supp.values, supp.probs))))
                plot_y_size = max(supp.probs)
                plt.ylim((-plot_y_size*1/20, plot_y_size*21/20))
            else:
                print("L'argument plot_style est invalide. Il doit être "\
                    "'bar', 'scatter', ou 'line'")
                return
        else:
            plt.hist(self.get_samples(), 100)
        plt.title('Distribution')
        plt.grid(True)
        plt.show()


def bernoulli(p):
    assert(0 <= p <= 1)
    sample  = lambda: sp.bernoulli.rvs(p)
    logpdf  = lambda x: sp.bernoulli.logpmf(x, p)
    mean    = lambda: sp.bernoulli.mean(p)
    var     = lambda: sp.bernoulli.var(p)
    if p == 0:
        logits = [-float('inf'), 0]
    elif p == 1:
        logits = [0, -float('inf')]
    else:
        logits = [log(1.-p), log(p)]
    support = Support([0, 1], logits, [1.-p, p])
    return Distrib(sample, logpdf, mean, var, support)

def binomial(p, n):
    assert(0 <= p <= 1 and 0 <= n)
    sample  = lambda: sp.binom.rvs(n, p)
    logpdf  = lambda x: sp.binom.logpmf(x, n, p)
    mean    = lambda: sp.binom.mean(n, p)
    var     = lambda: sp.binom.var(n, p)
    #If n is too big, it takes too much time to compute all comb(n,k)
    if n < 500:
        support_values = list(range(n+1))
        support_probs = [comb(n, k) for k in support_values]
        support_logits = [log(x) for x in support_probs]
        support = Support(support_values, support_logits, support_probs)
    else:
        support = None
    return Distrib(sample, logpdf, mean, var, support)

def dirac(v):
    sample = lambda: v
    logpdf = lambda x: 0. if x == v else -inf
    mean   = lambda: v
    var    = lambda: 0.
    return Distrib(sample, logpdf, mean, var)

def support(values, logits):
    assert(len(values) == len(logits))
    probs = utils.normalize(logits)
    sp_distrib = sp.rv_discrete(values=(range(len(values)), probs))
    sample  = lambda: values[sp_distrib.rvs()]
    logpdf  = lambda x: utils.findprob(values, probs, x)
    _mean = sum(values[i]*probs[i] for i in range(len(values)))
    _var = sum((values[i] - _mean)**2 for i in range(len(values))) / len(probs)
    mean    = lambda: _mean
    var     = lambda: _var
    support = Support(values, logits, probs)
    return Distrib(sample, logpdf, mean, var, support)

def uniform_support(values):
    logits = [0]*len(values)
    return support(values, logits)

def beta(a, b):
    assert(a > 0. and b > 0.)
    sample  = lambda: sp.beta.rvs(a, b)
    logpdf  = lambda x: sp.beta.logpdf(x, a, b)
    mean    = lambda: sp.beta.mean(a, b)
    var     = lambda: sp.beta.var(a, b)
    return Distrib(sample, logpdf, mean, var)

def gaussian(mu, sigma):
    assert(0. < sigma)
    sample  = lambda: sp.norm.rvs(loc=mu, scale=sigma)
    logpdf  = lambda x: sp.norm.logpdf(x, loc=mu, scale=sigma)
    mean    = lambda: sp.norm.mean(loc=mu, scale=sigma)
    var     = lambda: sp.norm.var(loc=mu, scale=sigma)
    return Distrib(sample, logpdf, mean, var)

def uniform(a, b):
    assert(a <= b)
    #scipy.stats.uniform(loc=0, scale=1) tire selon une loi uniforme
    #dans l'intervalle [loc, loc+scale]
    loc = a
    scale = b-a
    sample  = lambda: sp.uniform.rvs(loc=loc, scale=scale)
    logpdf  = lambda x: sp.uniform.logpdf(x, loc=loc, scale=scale)
    mean    = lambda: sp.uniform.mean(loc=loc, scale=scale)
    var     = lambda: sp.uniform.var(loc=loc, scale=scale)
    return Distrib(sample, logpdf, mean, var)


if __name__ == '__main__':
    '''p = 0.5
    x1 = bernoulli(p)
    x2 = binomial(p, 3)
    x3 = dirac(3)
    x4 = beta(2, 5)
    x5 = gaussian(0, 1)
    x6 = uniform(0, 2)'''
    logits = [log(2), log(3), log(5)]
    values = [0, 1, 3]
    y = support(values, logits)
    y.plot()
    print('main')
