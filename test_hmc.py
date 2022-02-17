from inference import ImportanceSampling, HamiltonianMonteCarlo, MetropolisHastings
from distribution import gaussian, uniform
from test_inference import test

def model(prob, data):
    # print('HELLO ???????', flush=True)
    q = prob.sample(gaussian(0, 1))
    r = prob.sample(gaussian(0, 1))
    # s = 0
    # for i in range(20):
    #     s += prob.sample(gaussian(0, 1))
    # prob.observe(gaussian(0, 1), s)
    # return s
    # prob.observe(gaussian(0, 1), q + r)
    prob.observe(gaussian(0, 1), q)
    return q

if __name__ == '__main__':
    model = model
    data = None
    name = "Test HMC"
    options = {
        'n': 200,
        'shrink': False,
        'plot_with_support': True,
        'plot_style': 'line',
        'eps': 0.05,
        'L': 10
    }
    #Tourne beaucoup trop longtemps
    #test(model, data, name, method=RejectionSampling, **options)

    #Fonctionne
    test(model, data, name, method=ImportanceSampling, **options)
    # test(model, data, name, method=MetropolisHastings, **options)
    test(model, data, name, method=HamiltonianMonteCarlo, **options)

    #N'est pas applicable car uniforme n'a pas de support fini.
    #test(model, data, name, method=EnumerationSampling, **options)
