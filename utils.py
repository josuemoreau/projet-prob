from typing import List, Tuple, Any, Callable, TypeVar, Optional, Generic, Dict
import scipy.special as spspec
from math import log, exp


def shrink(values: List[Any], probs: List[float]) \
        -> Tuple[List[Any], List[float]]:
    d: Dict[Any, float] = {}
    for value, prob in zip(values, probs):
        if value in d:
            d[value] += prob
        else:
            d[value] = prob
    return (list(d.keys()), list(d.values()))


def normalize(scores: List[float]) -> List[float]:
    norm = exp(spspec.logsumexp(scores))
    if norm < 1e-10:
        print("Scores are too small to give a valid distribution.")
    return [exp(elem) / norm for elem in scores]


def normalize_probs(probs: List[float]) -> List[float]:
    norm = sum(probs)
    return [elem/norm for elem in probs]


def findprob(values: List[Any], probs: List[float], v: Any) -> float:
    try:
        return probs[values.index(v)]
    except ValueError:
        return 0


if __name__ == '__main__':
    print(normalize([log(2), log(3), log(5)]))
