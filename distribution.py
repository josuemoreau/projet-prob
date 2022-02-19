from typing import List, Tuple, Callable, TypeVar, Optional, Generic, Any, Protocol
import scipy.stats as sp
import math
from math import log, comb, inf
import utils
import matplotlib.pyplot as plt

_A = TypeVar('_A')
_C = TypeVar('_C')


class CallableProtocol(Protocol[_C]):
    __call__: _C


class Support(Generic[_A]):
    values: List[_A]
    logits: List[float]
    probs: List[float]

    def __init__(self, values: List[_A], logits: List[float],
                 probs: List[float]):
        self.values = values
        self.logits = logits
        self.probs = probs

    def __iter__(self) -> List[Tuple[_A, float, float]]:
        return list(zip(self.values, self.logits, self.probs))


class Distrib(Generic[_A]):
    _sample: CallableProtocol[Callable[[], _A]]
    _sample2: Optional[CallableProtocol[Callable[[float], float]]]
    _logpdf: CallableProtocol[Callable[[_A], float]]
    mean: Optional[Callable[[], _A]]
    var: Optional[Callable[[], float]]
    _samples: Optional[List[_A]]
    _support: Optional[Support[_A]]
    _n: int

    def __init__(self, sample: Callable[[], _A], logpdf: Callable[[_A], float],
                 mean: Optional[Callable[[], _A]] = None,
                 var: Optional[Callable[[], float]] = None,
                 support: Optional[Support[_A]] = None,
                 sample2: Optional[Callable[[float], float]] = None,
                 n: int = 10000):
        self._n = n
        self._sample = sample
        self._logpdf = logpdf
        self.mean = mean
        self.var = var
        self._samples = None
        self._support = support
        self._sample2 = sample2

    def draw(self) -> _A:
        return self._sample()

    def draw2(self, x: float) -> float:
        assert (self._sample2 is not None)
        return self._sample2(x)

    def get_samples(self) -> List[_A]:
        if self._samples is not None:
            return self._samples
        else:
            samples = [self._sample() for i in range(self._n)]
            self._samples = samples
            return samples

    def get_support(self, shrink: bool = False) -> Optional[Support[_A]]:
        if not shrink:
            return self._support
        else:
            assert (self._support is not None)
            values = self._support.values
            probs = self._support.probs
            values, probs = utils.shrink(values, probs)
            return Support(values, [math.log(x) if x != 0. else -float('inf')
                                    for x in probs], probs)

    def logpdf(self, x: _A) -> float:
        return self._logpdf(x)

    def shrink_support(self) -> None:
        if self._support is not None:
            values = self._support.values
            probs = self._support.probs
            values, probs = utils.shrink(values, probs)
            self._support = Support(values,
                                    [math.log(x) if x != 0. else -float('inf')
                                     for x in probs], probs)

    def split_list(self) -> List[Distrib[Any]]:
        supp = self.get_support()
        assert (supp is not None)
        assert (len(supp.values) > 0)
        assert (isinstance(supp.values[0], List))
        assert (all(isinstance(v, List)  # vérification pour le typage
                    and len(v) == len(supp.values[0])
                    for v in supp.values))
        if all(len(v) == [] for v in supp.values):  # type: ignore
            return []
        res: List[Distrib[Any]] = []
        for i in range(len(supp.values[0])):
            values = [v[i] for v in supp.values]  # type: ignore
            res.append(support(values, supp.logits))
        return res

    def plot(self, plot_with_support: bool = False,
             plot_style: str = 'scatter',
             model_name: str = "", method_name: str = "") -> None:
        if plot_with_support:
            if self._support is None:
                print("Pas de support à plot")
                return
            supp = self.get_support()
            assert(isinstance(supp, Support))
            if plot_style == 'stem':
                plt.stem(supp.values, supp.probs)
            elif plot_style == 'scatter':
                plt.scatter(supp.values, supp.probs)
                plot_y_size = max(supp.probs)
                plt.ylim((-plot_y_size*1/20, plot_y_size*21/20))
            elif plot_style == 'line':
                plt.plot(*zip(*sorted(zip(supp.values, supp.probs))))
                plot_y_size = max(supp.probs)
                plt.ylim((-plot_y_size*1/20, plot_y_size*21/20))
            elif plot_style in ['scatter+line', 'line+scatter'] :
                plt.scatter(supp.values, supp.probs)
                plt.plot(*zip(*sorted(zip(supp.values, supp.probs))))
                plot_y_size = max(supp.probs)
                plt.ylim((-plot_y_size*1/20, plot_y_size*21/20))
            else:
                print("L'argument plot_style est invalide. Il doit être "\
                      "'stem', 'scatter', 'line', ou 'line+scatter'")
                return
        else:
            plt.hist(self.get_samples(), 100)
        plt.title(f"{model_name} - {method_name}")
        plt.grid(True)
        plt.show()


def bernoulli(p: float, size: Optional[int] = None) -> Distrib[int]:
    assert(0 <= p <= 1)
    sample  = lambda: sp.bernoulli.rvs(p, size=size)
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


def binomial(p: float, n: int, size: Optional[int] = None) -> Distrib[int]:
    assert(0 <= p <= 1 and 0 <= n)
    sample  = lambda: sp.binom.rvs(n, p, size=size)
    logpdf  = lambda x: sp.binom.logpmf(x, n, p)
    mean    = lambda: sp.binom.mean(n, p)
    var     = lambda: sp.binom.var(n, p)
    #If n is too big, it takes too much time to compute all comb(n,k)
    if n < 500:
        support_values = list(range(n+1))
        all_combs = [comb(n, k) for k in support_values]
        sum_combs = sum(all_combs)
        support_probs = [elem / sum_combs for elem in all_combs]
        support_logits = [log(x) for x in support_probs]
        support = Support(support_values, support_logits, support_probs)
    else:
        support = None
    return Distrib(sample, logpdf, mean, var, support)


def dirac(v: _A, size: Optional[int] = None) -> Distrib[_A]:
    sample = lambda: v if size is not None else [v] * size  # type: ignore
    logpdf = lambda x: 0. if x == v else -inf
    mean   = lambda: v
    var    = lambda: 0.
    return Distrib(sample, logpdf, mean, var)  # type: ignore


def support(values: List[_A], logits: List[float],
            size: Optional[int] = None) \
        -> Distrib[_A]:
    assert(len(values) == len(logits))
    probs = utils.normalize(logits)
    sp_distrib = sp.rv_discrete(values=(range(len(values)), probs))
    sample  = lambda: values[sp_distrib.rvs(size=size)]
    logpdf  = lambda x: utils.findprob(values, probs, x)
    try:
        # if values support product with a floatting number
        _mean = sum(values[i] * probs[i] for i in range(len(values)))  # type: ignore
        _var = sum((values[i] - _mean)**2 for i in range(len(values))) / len(probs)  # type: ignore
    except TypeError:
        # otherwise, we cannot compute mean and variance
        _mean = None
        _var = None
    mean    = lambda: _mean
    var     = lambda: _var
    support = Support(values, logits, probs)
    # return Distrib(sample, logpdf, support)  # type: ignore
    return Distrib(sample, logpdf, mean, var, support)  # type: ignore


def uniform_support(values: List[_A], size: Optional[int] = None) \
        -> Distrib[_A]:
    logits = [0.]*len(values)
    return support(values, logits, size=size)


def beta(a: float, b: float, size: Optional[int] = None) \
        -> Distrib[float]:
    assert(a > 0. and b > 0.)
    sample  = lambda: sp.beta.rvs(a, b, size=size)
    logpdf  = lambda x: sp.beta.logpdf(x, a, b)
    mean    = lambda: sp.beta.mean(a, b)
    var     = lambda: sp.beta.var(a, b)
    return Distrib(sample, logpdf, mean, var)


def gaussian(mu: float, sigma: float, size: Optional[int] = None) \
        -> Distrib[float]:
    assert(0. < sigma)
    sample  = lambda: sp.norm.rvs(loc=mu, scale=sigma, size=size)
    logpdf  = lambda x: sp.norm.logpdf(x, loc=mu, scale=sigma)
    mean    = lambda: sp.norm.mean(loc=mu, scale=sigma)
    var     = lambda: sp.norm.var(loc=mu, scale=sigma)
    sample2 = lambda x: x
    return Distrib(sample, logpdf, mean, var, sample2=sample2)


def uniform(a: float, b: float, size: Optional[int] = None) -> Distrib[float]:
    assert(a <= b)
    #scipy.stats.uniform(loc=0, scale=1) tire selon une loi uniforme
    #dans l'intervalle [loc, loc+scale]
    loc = a
    scale = b-a
    sample  = lambda: sp.uniform.rvs(loc=loc, scale=scale, size=size)
    logpdf  = lambda x: sp.uniform.logpdf(x, loc=loc, scale=scale)
    mean    = lambda: sp.uniform.mean(loc=loc, scale=scale)
    var     = lambda: sp.uniform.var(loc=loc, scale=scale)
    sample2 = lambda x: a + (b - a) * (1 / (1 + math.exp(-x)))
    return Distrib(sample, logpdf, mean, var, sample2=sample2)


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
