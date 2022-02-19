from inference import ImportanceSampling, MetropolisHastings, Prob, InferenceMethod
from typing import List, Type
from distribution import Distrib, gaussian, support
from numpy import linspace


def hmm(prob: Prob, data: List[float]) -> List[float]:
    states: List[float] = []
    for y in data:
        if len(states) == 0:
            states.append(y)
        else:
            pre_x = states[0]
            x = prob.sample(gaussian(pre_x, 1.0))
            prob.observe(gaussian(x, 1.0), y)
            states = [x] + states
    return states


def test_hmm(data: List[float], method: Type[InferenceMethod[List[float], List[float]]]) -> None:
    print("-- Hidden Markov Model, {} --".format(method.name()))
    m = method(hmm, data)
    dist = m.infer()
    dists = dist.split_list()
    m_x = [d.mean() for d in reversed(dists)]  # type: ignore
    for i in range(len(data)):
        print(f"{data[i]:.5f} >> {m_x[i]:.5f}")


if __name__ == '__main__':
    data = list(linspace(0., 20., 20))
    test_hmm(data, ImportanceSampling)
    test_hmm(data, MetropolisHastings)
