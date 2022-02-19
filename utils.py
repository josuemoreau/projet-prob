from typing import List, Tuple, Any, Callable, TypeVar, Optional, Generic, Dict
import scipy.special as spspec
from math import log, exp
import numpy as np

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
    '''if norm < 1e-10:
        print("Scores are too small to give a valid distribution.")'''
    return [exp(elem) / norm for elem in scores]


def normalize_probs(probs: List[float]) -> List[float]:
    norm = sum(probs)
    return [elem/norm for elem in probs]


def findprob(values: List[Any], probs: List[float], v: Any) -> float:
    try:
        return probs[values.index(v)]
    except ValueError:
        return 0


def gradient(f: Callable[[List[float]], float], x: List[float], eps: float) \
        -> List[float]:
    # print("GRADIENT POINT :", x)
    # print("VALUE AT x : ", f(x))
    L = []
    y = x.copy()
    for i in range(len(x)):
        y[i] += eps / 2
        vr = f(x)
        y[i] -= eps
        vl = f(x)
        # print(f"vr = {vr}, vl = {vl}")
        y[i] += eps / 2
        L.append((vr - vl) / eps)
    return L


if __name__ == '__main__':
    print(normalize([log(2), log(3), log(5)]))
