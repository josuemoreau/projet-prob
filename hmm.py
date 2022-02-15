from inference import ImportanceSampling, MetropolisHastings
from distribution import Distrib, gaussian, support
# from test_inference import *
from numpy import linspace


def hmm(prob, data):
    states = []
    for y in data:
        if len(states) == 0:
            states.append(y)
        else:
            pre_x = states[0]
            x = prob.sample(gaussian(pre_x, 1.0))
            prob.observe(gaussian(x, 1.0), y)
            states = [x] + states
    return states


def test_hmm(data, method):
    print("-- Hidden Markov Model, {} --".format(method.name()))
    m = method(hmm, data)
    dist = m.infer()
    dists = dist.split_list()
    m_x = [d.mean() for d in reversed(dists)]
    for i in range(len(data)):
        print(f"{data[i]:.5f} >> {m_x[i]:.5f}")


if __name__ == '__main__':
    data = list(linspace(0., 20., 20))
    test_hmm(data, ImportanceSampling)
    test_hmm(data, MetropolisHastings)
