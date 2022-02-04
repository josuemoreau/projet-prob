from typing import List, Tuple, Any, Callable, TypeVar, Optional, Generic, Dict

def shrink(values: List[Any], probs: List[float]) \
        -> Tuple[List[Any], List[float]]:
    d: Dict[Any, float] = {}
    for value, prob in zip(values, probs):
        if value in d:
            d[value] += prob
        else:
            d[value] = prob
    return (list(d.keys()), list(d.values()))