from typing import Callable, TypeVar, Generic, List, Any, NamedTuple, Protocol
from distribution import Distrib, support, uniform_support
import utils
from math import log, exp
from random import randint
from random import uniform as randuniform
from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import multivariate_normal, norm
import numdifftools as nd

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

    @abstractmethod
    def factor(self, s: float) -> None:
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

        def factor(self, _: float) -> None:
            pass

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

    def infer(self, n:int = 0) -> Distrib[B]:
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
        logits = []
        prob = self.MHProb(0, scores)
        lastValue = self._model(prob, self._data)
        lastScore = prob.computeScore()
        values.append(lastValue)
        logits.append(log(lastScore) if lastScore > 0 else -float('inf'))
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
            logits.append(log(lastScore) if lastScore > 0 else -float('inf'))
        assert(len(values) == len(scores))
        # print("remove :", remove_first_iterations)
        # print("values :", len(values[remove_first_iterations:]))
        # print("scores :", len(scores[remove_first_iterations:]))
        return support(values, logits)


class DomainError(Exception):
    pass


class HamiltonianMonteCarlo(InferenceMethod[A, B]):

    class HMCProb(Prob):
        _id: int
        _scores: List[float]
        _sampleResults: List[Any]
        _i: int
        _len: int
        _trueSample: bool

        def __init__(self, idx: int, scores: List[float], samples: List[float]):
            # print("INIT WITH SAMPLES :", samples)
            self._id = idx
            self._scores = scores
            self._i = 0
            self._len = len(samples)
            self._sampleResults = samples
            self._trueSample = len(samples) == 0
            # print("TRUESAMPLE : ", self._trueSample)
            # print("SAMPLES :", samples)

        def factor(self, s: float) -> None:
            self._scores[self._id] += s

        def assume(self, p: bool) -> None:
            self.factor(0. if p else -float('inf'))

        def observe(self, d: Distrib[A], x: A) -> None:
            self.factor(d.logpdf(x))

        def get_samples_results(self):
            return self._sampleResults.copy()

        def sample(self, d: Distrib[A]) -> A:
            # if self._trueSample:
            #     v = d.draw()
            #     # print("SAMPLE1 :", v, flush=True)
            #     self._sampleResults.append(v)
            #     self._i += 1
            #     self._len += 1
            # elif self._i < self._len:
            if self._i < self._len:
                v = self._sampleResults[self._i]
                v = d.draw2(v)
                self.observe(d, v)
                self._i += 1
            else:
                # print('SAMPLE3')
                raise DomainError
            return v  # type: ignore

    _model: CallableProtocol[Callable[[Prob, A], B]]
    _data: A

    def __init__(self, model: Callable[[Prob, A], B], data: A):
        self._model = model
        self._data = data

    @staticmethod
    def name() -> str:
        return "Hamiltonian Monte Carlo"

    def model_logpdf(self, state: List[float]) -> float:
        scores = [0.]
        prob = self.HMCProb(0, scores, state)
        self._model(prob, self._data)
        return scores[0]

    def model_nb_samples(self, state: List[float]) -> int:
        scores = [0.]
        prob = self.HMCProb(0, scores, state)
        self._model(prob, self._data)
        return prob._i

    def propose_new_state(self, q0: np.ndarray, p0: np.ndarray,
                          U: Callable[[int], Callable[[np.ndarray], float]],
                          eps: float, L: int):
        # print("Computing new state ...")
        q, p = q0.copy(), p0.copy()
        for i in range(L):
            # p = p - (eps / 2) * nd.Gradient(U(len(q)))(q)
            p = p - (eps / 2) * np.array(utils.gradient(U(len(q)), q, 0.0001))
            q = q + eps * p
            # print(f"q = {q}, p = {p}")
            (q, p), (q0, p0) = self.extend(q, p, q0, p0, i * eps, U)
            # p = p - (eps / 2) * nd.Gradient(U(len(q)))(q)
            p = p - (eps / 2) * np.array(utils.gradient(U(len(q)), q, 0.0001))
        return (q, p), (q0, p0)

    def extend(self, q, p, q0, p0, t, U):
        # print("Extend ...")
        # q, p = q.copy(), p.copy()
        # q0, p0 = q0.copy(), p0.copy()
        q, p = list(q), list(p)
        q0 = list(q0)
        p0 = list(p0)
        while True:
            # print(q)
            try:
                # on ne récupère pas la valeur, on veut juste vérifier que
                # q (c'est à dire un ensemble de résultats de sample) est dans
                # le domaine du modèle, c'est-à-dire qu'il y a assez de valeurs
                # pour tous les sample que va appeler le modèle
                U(len(q))(q)
                break
            except DomainError:
                x0 = norm.rvs()
                y0 = norm.rvs()
                x, y = x0 + t * y0, y0
                q0.append(x0)
                p0.append(y0)
                q.append(x)
                p.append(y)
        return (q, p), (q0, p0)

    def infer(self, n: int = 1000, eps: float = 0.01, L: int = 20) \
            -> Distrib[B]:
        scores = [0.] * n
        values = []
        logits = []

        (state, _), (_, _) = self.extend([], [], [], [], 0, lambda n: lambda q: -self.model_logpdf(q[:n]))

        prob = self.HMCProb(0, scores, state)

        # state0 = [1.093226789944025, 0.46179919341633946, -0.42287521801174344, -0.10236153707493756, 0.8475671404937191, -0.9504729483923995, -0.6597561483599956, -0.0005515655039916023, 0.9765556570346068, 0.9717289503506064, -0.12864929115445656, 0.40441063228386986, 0.7737931044480051, 0.6437629478826674, -0.5929933682584381, -0.6656567766373178, 0.07789285242845745, -0.42395810651760346, 0.5926198869567285, -0.06676937681845674, -0.7230708249049229]
        # state1 = [1.0927526339222369, 0.4601490122556122, -0.4280947097342658, -0.10482786215887524, 0.8470821650621851, -0.9489371748050817, -0.6622424897224177, 0.0014252970864457992, 0.97702015530997, 0.9776528669199678, -0.12742638729673772, 0.40541335699141823, 0.7702357078774391, 0.6433083682040504, -0.5952978732491891, -0.6635979406465697, 0.07770143077765324, -0.42566373939359026, 0.5929726741999204, -0.06583805015249483, -0.7208094027616725]

        # prob = self.HMCProb(0, scores, state0)
        # self._model(prob, self._data)
        # print(scores[0])


        # return 

        lastValue = self._model(prob, self._data)
        lastScore = scores[0]
        state = np.array(prob.get_samples_results())

        values.append(lastValue)
        logits.append(lastScore)

        phi = lambda n: lambda q: \
            multivariate_normal.pdf(q, mean=np.zeros(n),
                                    cov=np.identity(n)) \
            if len(q) > 1 else \
            np.array([multivariate_normal.pdf(q, mean=np.zeros(n),
                                              cov=np.identity(n))])

        for i in range(1, n):
            print(f"Iteration {i}", flush=True)

            # if i % 50 == 0:
            #     print('Restart ...')
            #     scores = [0.]
            #     prob = self.HMCProb(0, scores, [])
            #     lastValue = self._model(prob, self._data)
            #     lastScore = scores[0]
            #     state = np.array(prob.get_samples_results())
            #     values.append(lastValue)
            #     logits.append(lastScore)
            #     continue


            # state = np.array([ 0.22335286,  0.94171858,  0.51335159,  0.99492119, -0.11005573, -0.10505592,
            #                 -0.76476933, -0.38981185, -0.47405379,  0.24983975,  0.48748061, -0.16994817,
            #                  -0.70619371,  0.98891105, -0.79866958, -0.49817204,  0.6379942,   0.61240333,
            #                 0.26287404,  0.81438024])

            # print(list(state))

            print("STATE SIZE :", len(state))
            # if len(state) > 1:
            #     p0 = multivariate_normal.rvs(mean=np.zeros(len(state)),
            #                                  cov=np.identity(len(state)))
            # else:
            #     p0 = np.array([
            #         multivariate_normal.rvs(mean=np.zeros(len(state)),
            #                                 cov=np.identity(len(state)))])

            # p0 = np.array([norm.rvs(loc=0, scale=1) for j in range(len(state))])
            p0 = norm.rvs(loc=0, scale=1, size=len(state))
            U = lambda n: lambda q: -self.model_logpdf(q[:n])
            (q, p), (q0, p0) = self.propose_new_state(state, p0, U, eps, L)

            # print("OLD STATE = ", q0)
            # print("NEW STATE = ", q)



            # print("q0 = ", len(q0))
            # print("q = ", len(q))
            # print("p0 = ", len(p0))
            # print("p = ", len(p))

            scores = [0., 0.]
            # print("1q = ", len(q))
            prob0 = self.HMCProb(0, scores, q0)
            # print("2q = ", len(q))
            prob = self.HMCProb(1, scores, q)
            # print("3q = ", len(q))
            v0 = self._model(prob0, self._data)
            # print("4q = ", len(q))
            v = self._model(prob, self._data)
            # print("5q = ", len(q))

            l0 = np.concatenate((q0, p0))
            l = np.concatenate((q, p))

            # print("q0 = ", len(q0))
            # print("q = ", len(q))
            # print("p0 = ", len(p0))
            # print("p = ", len(p))

            # print(f"phi : {phi(2 * len(q))(l)}")
            # print(f"phi0 : {phi(2 * len(q0))(l0)}")

            # print(2 * len(q))
            # print(len(l))

            a = min(1, (exp(scores[1]) * phi(2 * len(q))(l)) / (exp(scores[0]) * phi(2 * len(q0))(l0)))
            x = randuniform(0, 1)
            vf = v if x < a else v0
            pf = scores[1] if x < a else scores[0]
            # print(pf, scores[0], scores[1])
            values.append(vf)
            logits.append(pf)

            state = list(q)[:prob._i]
        # print(logits)

        # print(values)
        return support(values, logits)
