from inference import ImportanceSampling, HamiltonianMonteCarlo, MetropolisHastings
from distribution import gaussian, uniform
from test_inference import test
import math

def model(prob, data):
    # print('HELLO ???????', flush=True)
    q = prob.sample(gaussian(0, 1))
    # r = prob.sample(gaussian(0, 1))
    s = 0
    for i in range(40):
        # print(f"{q} : {s}")
        if s >= q:
            break
        s += prob.sample(gaussian(0, 1))
    else:
        prob.assume(False)
        return 0
    prob.observe(gaussian(q, 1), s)
    return q
    # prob.observe(gaussian(0, 1), q + r)
    # prob.observe(gaussian(0, 1), q)
    # return q

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
    # print('HELLO ???????', flush=True)
    q = prob.sample(gaussian(0, 1))
    r = prob.sample(gaussian(0, 1))
    # prob.observe(gaussian(0, 1), q + r)
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
        'n': 500,
        'shrink': True,
        'plot_with_support': True,
        'plot_style': 'line+scatter',
        'eps': 0.1,
        'L': 20
    }
    #Tourne beaucoup trop longtemps
    #test(model, data, name, method=RejectionSampling, **options)

    #Fonctionne
    # test(model, data, name, method=ImportanceSampling, **options)
    # test(model, data, name, method=MetropolisHastings, **options)

    import cProfile
    import pstats

    with cProfile.Profile() as pr:
        test(model, data, name, method=HamiltonianMonteCarlo, **options)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    #stats.print_stats()
    stats.dump_stats(filename='profiling.prof')

    #N'est pas applicable car uniforme n'a pas de support fini.
    #test(model, data, name, method=EnumerationSampling, **options)
