from typing import Callable, TypeVar, Generic
from typing_extensions import Protocol
from distribution import Distrib

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


if __name__ == "__main__":
    def model(prob, data):
        pass
    rs = RejectionSampling(model, 0)
    rs.infer()
