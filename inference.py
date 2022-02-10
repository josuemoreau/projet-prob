from typing import Callable, TypeVar, Generic, List
from typing_extensions import Protocol
from distribution import Distrib, uniform_support, support

# Rejection Sampling
#

A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C', covariant=True)


class Reject(Exception):
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

        def exec(i: int):
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
