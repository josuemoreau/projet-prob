from inference import ImportanceSampling, MetropolisHastings
from hmc import HamiltonianMonteCarlo
from distribution import gaussian, uniform
from test_inference import test


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
