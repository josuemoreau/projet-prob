from typing import Callable, TypeVar, Generic, List
from typing_extensions import Protocol
from distribution import Distrib, support, uniform_support
import utils
from math import log

# Rejection Sampling
#

A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C', covariant=True)


class Reject(Exception):
    pass

class UndefinedSupport(Exception):
    pass


class RejectionSampling(Generic[A, B]):

    class Prob(object):
        def assume(self, p: bool):
            if not p:
                raise Reject

        def observe(self, d: Distrib[A], x: A):
            y = d.draw()
            self.assume(x == y)

        def sample(self, d: Distrib[A]):
            return d.draw()

    class Model(Generic[C], Protocol):
        def __call__(self, prob, data):
            # type : (C,  A) -> B
            pass

    _model: Model[Prob]
    _data: A

    def __init__(self, model: Callable[[Prob, A], B], data: A):
        # self._prob = self.Prob()
        self._model = model
        self._data = data

    def infer(self, n=1000):
        prob = self.Prob()

        def exec(i: int): #Faire ça avec while pour éviter la limite de récurrence ?
            try:
                return self._model(prob, self._data)
            except Reject:
                return exec(i)

        values = []
        for i in range(n):
            values.append(exec(i))
        return uniform_support(values)


class ImportanceSampling(Generic[A, B]):

    class Prob(object):
        _id: int
        _scores: List[float]

        def __init__(self, id: int, scores: List[float]):
            self._id = id
            self._scores = scores

        def factor(self, s: float):
            self._scores[self._id] += s

        def assume(self, p: bool):
            self.factor(0. if p else -float('inf'))

        def observe(self, d: Distrib[A], x: A):
            self.factor(d.logpdf(x))

        def sample(self, d: Distrib[A]):
            return d.draw()

    class Model(Generic[C], Protocol):
        def __call__(self, prob, data):
            # type : (C,  A) -> B
            pass

    _model: Model[Prob]
    _data: A

    def __init__(self, model: Callable[[Prob, A], B], data: A):
        # self._prob = self.Prob()
        self._model = model
        self._data = data

    def infer(self, n=1000):
        scores = [0.] * n
        values = [self._model(self.Prob(i, scores), self._data)
                  for i in range(n)]
        return support(values, scores)


class EnumerationSampling():

    def __init__(self):
        self.sample_idx = -1
        self.curr_logit = 0
        self.path = []
        self.has_ended = False

    def init_next_path(self):
        #Reinitialisation des valeurs.
        self.sample_idx = -1
        self.curr_logit = 0
        #Calcul de la prochaine trajectoire.
        n = len(self.path) - 1
        for i in range(n, -1, -1):
            idx, choices, _ = self.path[i]
            #On effectue le prochain choix
            if idx + 1 < len(choices):
                self.path[i][0] = idx + 1
                return
            #Si on a fait tous les choix possibles, on revient en arrière.
            else:
                self.path.pop(i)
        #Si la prochaine trajectoire est nulle, on a fini.
        if len(self.path) == 0:
            self.has_ended = True

    def get_logit(self):
        return self.curr_logit

    def sample(self, distr):
        self.sample_idx += 1
        if self.sample_idx >= len(self.path):
            #On est sûr dans ce cas que sample_idx == len(path)
            if distr._support is None:
                raise UndefinedSupport("""The support is undefined or infinite.
                Exhaustive Sampling can't be used in that case.""")
            support = distr.get_support(shrink=True)
            self.path.append([0, support.values, support.logits])

        idx, choices, logits = self.path[self.sample_idx]
        self.curr_logit += logits[idx] #Mult petits probs ou Add grands logits ?
        return choices[idx]
    
    def assume(self, p: bool):
        if not p:
            raise Reject
            
    def observe(self, distr, x):
        pass

    @classmethod
    def infer(cls, model, data):
        sampler = cls()
        values = []
        logits = []
        while not sampler.has_ended:
            try:
                value = model(sampler, data)
                values.append(value)
                logits.append(sampler.get_logit())
            except Reject:
                pass
            sampler.init_next_path()
        probs = utils.normalize(logits)
        values, probs = utils.shrink(values, probs)
        return support(values, [log(p) for p in probs])

if __name__ == "__main__":
    from funny_bernoulli import funny_bernoulli
    d = EnumerationSampling.infer(funny_bernoulli, None)
    d.plot()
