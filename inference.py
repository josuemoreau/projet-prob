from typing import Callable, TypeVar, Generic, List, Any
from typing_extensions import Protocol
from distribution import Distrib, support, uniform_support
import utils
from math import log, exp
from random import randint, uniform
from numpy import log as nplog

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
        self._model = model
        self._data = data

    def infer(self, n=1000):
        prob = self.Prob()
        values = []
        while len(values) < n:
            try:
                value = self._model(prob, self._data)
            except Reject:
                continue
            values.append(value)
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
        # Reinitialisation des valeurs.
        self.sample_idx = -1
        self.curr_logit = 0
        # Calcul de la prochaine trajectoire.
        n = len(self.path) - 1
        for i in range(n, -1, -1):
            idx, choices, _ = self.path[i]
            # On effectue le prochain choix
            if idx + 1 < len(choices):
                self.path[i][0] = idx + 1
                return
            # Si on a fait tous les choix possibles, on revient en arrière.
            else:
                self.path.pop(i)
        # Si la prochaine trajectoire est nulle, on a fini.
        if len(self.path) == 0:
            self.has_ended = True

    def get_logit(self):
        return self.curr_logit

    def sample(self, distr):
        self.sample_idx += 1
        if self.sample_idx >= len(self.path):
            # On est sûr dans ce cas que sample_idx == len(path)
            if distr._support is None:
                raise UndefinedSupport("The support is undefined or infinite. "
                        + "Exhaustive Sampling can't be used in that case.")
            support = distr.get_support(shrink=True)
            self.path.append([0, support.values, support.logits])

        idx, choices, logits = self.path[self.sample_idx]
        self.curr_logit += logits[idx]  # Mult petits probs ou Add grands logits ?
        return choices[idx]

    def assume(self, p: bool):
        if not p:
            raise Reject

    def observe(self, distr, x):
        y = self.sample(distr)
        self.assume(x == y)

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
        return support(values, [log(p) if p != 0 else -float('inf') for p in probs])


class MetropolisHastings(Generic[A, B]):

    class RandomVariable():
        def __init__(self, name, val):
            self.name = name
            self.val = val

    class Prob(object):
        _id: int
        _scores: List[float]
        _weights: List[float]
        _sampleResults: List[Any]
        _i: int
        _len: int
        _varId: int
        _reuseI: int

        def __init__(self, id: int, scores: List[float]):
            self._id = id
            self._scores = scores
            self._i = 0
            self._reuseI = 0
            self._len = 0
            self._sampleResults = []
            self._weights = []

        def factor(self, s: float):
            self._scores[self._id] += s

        def assume(self, p: bool):
            self.factor(0. if p else -float('inf'))

        def observe(self, d: Distrib[A], x: A):
            self.factor(d.logpdf(x))

        def sample(self, d: Distrib[A]):
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
            return v

        def go_back_to_step(self, i, new_id):
            self._sampleResults = self._sampleResults[:i]
            self._weights = self._weights[:i]
            self._reuseI = i
            self._i = 0
            self._len = i
            self._id = new_id

        def pick_random_step(self):
            assert(self._len >= 1)
            d = uniform_support(list(range(self._len)))
            return d.draw()
            # return randint(0, self._len - 1)

        def computeScore(self):
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

    class Model(Generic[C], Protocol):
        def __call__(self, prob, data):
            # type : (C,  A) -> B
            pass

    _model: Model[Prob]
    _data: A

    def __init__(self, model: Callable[[Prob, A], B], data: A):
        self._model = model
        self._data = data

    def infer(self, n=1000, remove_first_iterations=0):
        scores = [0.] * n
        values = []
        probs = []
        prob = self.Prob(0, scores)
        lastValue = self._model(prob, self._data)
        lastScore = prob.computeScore()
        values.append(lastValue)
        probs.append(nplog(lastScore))
        for i in range(1, n):
            # on choisit aléatoirement un sample qui a été effectué par le
            # modèle
            step = prob.pick_random_step()
            # on exécute à nouveau le modèle depuis ce point
            prob.go_back_to_step(step, i)
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
            a = uniform(0, 1)
            if lastScore == 0 or a < min(1, score / lastScore):
                lastValue = v
                lastScore = score
            values.append(lastValue)
            probs.append(nplog(lastScore))
        assert(len(values) == len(scores))
        print("remove :", remove_first_iterations)
        print("values :", len(values[remove_first_iterations:]))
        print("scores :", len(scores[remove_first_iterations:]))
        return support(values[remove_first_iterations:],
                       probs[remove_first_iterations:])


if __name__ == "__main__":
    from funny_bernoulli import funny_bernoulli
    d = EnumerationSampling.infer(funny_bernoulli, None)
    d.plot()
    x = MetropolisHastings(funny_bernoulli, None)
    d = x.infer()
    d.plot()
