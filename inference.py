from typing import Callable, TypeVar, Generic, List, Any, NamedTuple
from typing_extensions import Protocol
from distribution import Distrib, support, uniform_support
import utils
from math import log, exp
from random import randint
from random import uniform as randuniform
from abc import ABC, abstractmethod

A = TypeVar('A')
B = TypeVar('B')
_C = TypeVar('_C')


class Prob(ABC):

    @abstractmethod
    def assume(self, p: bool) -> None:
        pass

    @abstractmethod
    def observe(self, d: Distrib[A], x: A) -> None:
        pass

    @abstractmethod
    def sample(self, d: Distrib[A]) -> A:
        pass


class InferenceMethod(ABC, Generic[A, B]):

    @abstractmethod
    def __init__(self, model: Callable[[Prob, A], B], data: A):
        pass

    @abstractmethod
    def infer(self, n: int = 1000) -> Distrib[B]:
        pass

    @staticmethod
    @abstractmethod
    def name() -> str:
        pass


class Reject(Exception):
    pass


class UndefinedSupport(Exception):
    pass


class CallableProtocol(Protocol[_C]):
    __call__: _C


class RejectionSampling(InferenceMethod[A, B]):

    class RejSampProb(Prob):
        def assume(self, p: bool) -> None:
            if not p:
                raise Reject

        def observe(self, d: Distrib[A], x: A) -> None:
            y = d.draw()
            self.assume(x == y)

        def sample(self, d: Distrib[A]) -> A:
            return d.draw()

    _model: CallableProtocol[Callable[[Prob, A], B]]
    _data: A

    def __init__(self, model: Callable[[Prob, A], B], data: A):
        self._model = model
        self._data = data

    @staticmethod
    def name() -> str:
        return "Rejection Sampling"

    def infer(self, n: int = 1000) -> Distrib[B]:
        prob = self.RejSampProb()
        values: List[B] = []
        while len(values) < n:
            try:
                value = self._model(prob, self._data)
            except Reject:
                continue
            values.append(value)
        return uniform_support(values)


class ImportanceSampling(InferenceMethod[A, B]):

    class ImpSampProb(Prob):
        _id: int
        _scores: List[float]

        def __init__(self, idx: int, scores: List[float]):
            self._id = idx
            self._scores = scores

        def factor(self, s: float) -> None:
            self._scores[self._id] += s

        def assume(self, p: bool) -> None:
            self.factor(0. if p else -float('inf'))

        def observe(self, d: Distrib[A], x: A) -> None:
            self.factor(d.logpdf(x))

        def sample(self, d: Distrib[A]) -> A:
            return d.draw()

    _model: CallableProtocol[Callable[[Prob, A], B]]
    _data: A

    def __init__(self, model: Callable[[Prob, A], B], data: A):
        self._model = model
        self._data = data

    @staticmethod
    def name() -> str:
        return "Importance Sampling"

    def infer(self, n: int = 1000) -> Distrib[B]:
        scores = [0.] * n
        values = [self._model(self.ImpSampProb(i, scores), self._data)
                  for i in range(n)]
        return support(values, scores)


_SampleState = NamedTuple('_SampleState', [('idx', int),
    ('choices', List[A]), ('logits', List[float])])

class EnumerationSampling(InferenceMethod[A, B]):

    class EnumSampProb(Prob):
        _state_idx: int
        _score: float
        _path: List[_SampleState]
        _has_ended: bool

        def __init__(self) -> None:
            self._state_idx = -1
            self._score = 0
            self._path = []
            self._has_ended = False

        def init_next_path(self) -> None:
            # Reinitialisation des valeurs.
            self._state_idx = -1
            self._score = 0
            # Calcul de la prochaine trajectoire.
            n = len(self._path) - 1
            for i in range(n, -1, -1):
                idx, choices, logits = self._path[i]
                # On effectue le prochain choix
                if idx + 1 < len(choices):
                    self._path[i] = _SampleState(idx=idx + 1,
                    choices=choices, logits=logits)
                    return
                # Si on a fait tous les choix possibles, on revient en arrière.
                else:
                    self._path.pop() #i enlevé !!!!
            # Si la prochaine trajectoire est nulle, on a fini.
            if len(self._path) == 0:
                self._has_ended = True

        def assume(self, p: bool) -> None:
            if not p:
                raise Reject

        def observe(self, d: Distrib[A], x:A) -> None:
            y = self.sample(d)
            self.assume(x == y)

        def sample(self, d: Distrib[A]) -> A:
            self._state_idx += 1
            if self._state_idx >= len(self._path):
                # On est sûr dans ce cas que sample_idx == len(path)
                if d._support is None:
                    raise UndefinedSupport("The support is undefined or infinite. "\
                            "Exhaustive Sampling can't be used in that case.")
                support = d.get_support(shrink=True)
                assert(support is not None)
                self._path.append(_SampleState(idx=0, choices=support.values,
                logits=support.logits))

            idx, choices, logits = self._path[self._state_idx]
            self._score += logits[idx]  # Mult petits probs ou Add grands logits ?
            return choices[idx]


    _model: CallableProtocol[Callable[[Prob, A], B]]
    _data: A

    def __init__(self, model: Callable[[Prob, A], B], data: A):
        self._model = model
        self._data = data

    @staticmethod
    def name() -> str:
        return "Enumeration Sampling"

    def infer(self, _:int = 0) -> Distrib[B]:
        prob = self.EnumSampProb()
        values = []
        logits = []
        while not prob._has_ended:
            try:
                value = self._model(prob, self._data)
                values.append(value)
                logits.append(prob._score)
            except Reject:
                pass
            prob.init_next_path()
        probs = utils.normalize(logits)
        values, probs = utils.shrink(values, probs)
        return support(values, [log(p) if p != 0 else -float('inf') for p in probs])


class MetropolisHastings(InferenceMethod[A, B]):

    class MHProb(Prob):
        _id: int
        _scores: List[float]
        _weights: List[float]
        _sampleResults: List[Any]
        _i: int
        _len: int
        _reuseI: int

        def __init__(self, idx: int, scores: List[float]):
            self._id = idx
            self._scores = scores
            self._i = 0
            self._reuseI = 0
            self._len = 0
            self._sampleResults = []
            self._weights = []

        def factor(self, s: float) -> None:
            self._scores[self._id] += s

        def assume(self, p: bool) -> None:
            self.factor(0. if p else -float('inf'))

        def observe(self, d: Distrib[A], x: A) -> None:
            self.factor(d.logpdf(x))

        def sample(self, d: Distrib[A]) -> A:
            if self._i < self._len:
                v = self._sampleResults[self._i]
                self._i += 1
            else:
                v = d.draw()
                self._sampleResults.append(v)
                # print('logpdf :', d.logpdf(v))
                self._weights.append(d.logpdf(v))
                self._i += 1
                self._len += 1
            return v  # type: ignore

        def go_back_to_step(self, i: int, new_id: int) -> None:
            self._sampleResults = self._sampleResults[:i]
            self._weights = self._weights[:i]
            self._reuseI = i
            self._i = 0
            self._len = i
            self._id = new_id

        def pick_random_step(self) -> int:
            assert(self._len >= 1)
            return randint(0, self._len - 1)

        def computeScore(self) -> float:
            s = 0.
            for i in range(self._reuseI):
                s += self._weights[i]
            # print(s)
            # on a maintenant :
            # s : commme des log des poids des variables sur lesquelles on
            #     a appelé sample
            # self._scores[self._id] : somme des log des poids des variables
            #     observées (= score dans importance sampling)
            # self._len : le nombre de variables de sample
            return exp(s + self._scores[self._id]) / self._len

    _model: CallableProtocol[Callable[[Prob, A], B]]
    _data: A

    def __init__(self, model: Callable[[Prob, A], B], data: A):
        self._model = model
        self._data = data

    @staticmethod
    def name() -> str:
        return "Metropolis Hastings"

    def infer(self, n: int = 1000) -> Distrib[B]:
        scores = [0.] * n
        values = []
        probs = []
        prob = self.MHProb(0, scores)
        lastValue = self._model(prob, self._data)
        lastScore = prob.computeScore()
        values.append(lastValue)
        probs.append(log(lastScore) if lastScore > 0 else -float('inf'))
        for i in range(1, n):
            # on choisit aléatoirement un sample qui a été effectué par le
            # modèle
            step = prob.pick_random_step()
            # print(f"Last model has {prob._len} samples and {prob._observed} observed values")
            # on exécute à nouveau le modèle depuis ce point
            prob.go_back_to_step(step, i)
            # print(f"Going back to step {step}")
            v = self._model(prob, self._data)
            score = prob.computeScore()
            # if score >= lastScore:
            #     # si le score de la nouvelle exécution est meilleur, on la
            #     # garde
            #     values.append(v)
            #     lastScore = score
            # else:
            # sinon, on la garde avec une probabilité
            # score actuel / score de l'exécution précédente
            x = randuniform(0, 1)
            # print(f"Current value : {v}")
            # print(f"Current score : {score}")
            # print(f"Last score : {lastScore}")
            if lastScore == 0 or x < min(1, score / lastScore):
                lastValue = v
                lastScore = score
            values.append(lastValue)
            probs.append(log(lastScore) if lastScore > 0 else -float('inf'))
        assert(len(values) == len(scores))
        # print("remove :", remove_first_iterations)
        # print("values :", len(values[remove_first_iterations:]))
        # print("scores :", len(scores[remove_first_iterations:]))
        return support(values, probs)
