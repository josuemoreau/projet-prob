from inference import ImportanceSampling, MetropolisHastings, Prob, InferenceMethod
from typing import Callable, TypeVar, Type
from hmc import HamiltonianMonteCarlo
from distribution import gaussian, uniform
from test_inference import test


A = TypeVar('A')
B = TypeVar('B')

def test(model: Callable[[Prob, A], B], data: A, name: str,
         method: Type[InferenceMethod[A, B]] = ImportanceSampling,
         n: int = 1000,
         print_support: bool = False,
         shrink: bool = False,
         plot_with_support: bool = False,
         plot_style: str = 'scatter',
         eps: float = 0.01, L: int = 20) \
         -> None:
    print("-- {}, {} --".format(name, method.name()))
    mh = method(model, data)
    if method == HamiltonianMonteCarlo:
        dist = mh.infer(n=n, eps=eps, L=L)  # type: ignore
    else:
        dist = mh.infer(n=n)
    if shrink:
        dist.shrink_support()
    if print_support:
        supp = dist.get_support()
        assert (supp is not None)
        for i in range(len(supp.values)):
            print(f"{supp.values[i]} {supp.probs[i]}")
    dist.plot(plot_with_support=plot_with_support, plot_style=plot_style,
              model_name=name, method_name=method.name())



def model(prob, data):
    q = prob.sample(gaussian(0, 1))
    s = 0
    for i in range(40):
        if s >= q:
            break
        s += prob.sample(gaussian(0, 1))
    else:
        prob.assume(False)
        return 0
    prob.observe(gaussian(q, 1), s)
    return q

def model(prob, data):
    start = prob.sample(uniform(0, 3))
    position = start
    distance = 0
    while position > 0 and distance < 10:
        step = prob.sample(uniform(-1, 1))
        position += step
        distance += abs(step)
    prob.observe(gaussian(1.1, 0.1), distance)
    return start

def model(prob, data):
    q = prob.sample(gaussian(0, 1))
    r = prob.sample(gaussian(0, 1))
    prob.observe(gaussian(0, 1), q)
    return q

def model(prob, data):
    if prob.sample(uniform(0, 1)) < 0.2: return 1
    else: return 1 + model(prob, data)

if __name__ == '__main__':
    model = model
    data = None
    name = "Test HMC"
    options = {
        'n': 100,
        'shrink': True,
        'plot_with_support': True,
        'plot_style': 'line+scatter',
        'eps': 0.1,
        'L': 5
    }

    import cProfile
    import pstats

    with cProfile.Profile() as pr:
        test(model, data, name, method=HamiltonianMonteCarlo, **options)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename='profiling.prof')
