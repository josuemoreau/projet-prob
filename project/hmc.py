from inference import *
import numpy as np
from scipy.stats import norm, multivariate_normal


class DomainError(Exception):
    pass


class HamiltonianMonteCarlo(InferenceMethod[A, B]):

    class HMCProb(Prob):
        _id: int
        _scores: List[float]
        _sampleResults: List[Any]
        _i: int
        _len: int

        def __init__(self, idx: int, scores: List[float], samples: List[float]):
            self._id = idx
            self._scores = scores
            self._i = 0
            self._len = len(samples)
            self._sampleResults = samples
            self._trueSample = len(samples) == 0

        def factor(self, s: float) -> None:
            self._scores[self._id] += s

        def assume(self, p: bool) -> None:
            self.factor(0. if p else -float('inf'))

        def observe(self, d: Distrib[A], x: A) -> None:
            self.factor(d.logpdf(x))

        def get_samples_results(self):
            return self._sampleResults.copy()

        def sample(self, d: Distrib[A]) -> A:
            if self._i < self._len:
                v = self._sampleResults[self._i]
                v = d.draw2(v)
                self.observe(d, v)
                self._i += 1
            else:
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
        q, p = q0.copy(), p0.copy()
        for i in range(L):
            p = p - (eps / 2) * np.array(utils.gradient(U(len(q)), q, 0.00001))
            q = q + eps * p
            (q, p), (q0, p0) = self.extend(q, p, q0, p0, i * eps, U)
            p = p - (eps / 2) * np.array(utils.gradient(U(len(q)), q, 0.00001))
        return (q, p), (q0, p0)

    def extend(self, q, p, q0, p0, t, U):
        q, p = list(q), list(p)
        q0 = list(q0)
        p0 = list(p0)
        while True:
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

            print("STATE SIZE :", len(state))

            p0 = norm.rvs(loc=0, scale=1, size=len(state))
            U = lambda n: lambda q: -self.model_logpdf(q[:n])
            (q, p), (q0, p0) = self.propose_new_state(state, p0, U, eps, L)

            scores = [0., 0.]
            prob0 = self.HMCProb(0, scores, q0)
            prob = self.HMCProb(1, scores, q)
            v0 = self._model(prob0, self._data)
            v = self._model(prob, self._data)

            l0 = np.concatenate((q0, p0))
            l = np.concatenate((q, p))

            a = min(1, (exp(scores[1]) * phi(2 * len(q))(l)) / (exp(scores[0]) * phi(2 * len(q0))(l0)))
            x = randuniform(0, 1)
            vf = v if x < a else v0
            pf = scores[1] if x < a else scores[0]
            values.append(vf)
            logits.append(pf)

            state = list(q)[:prob._i]
        return support(values, logits)
