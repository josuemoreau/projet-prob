from inference import ImportanceSampling, MetropolisHastings, EnumerationSampling
from distribution import bernoulli, uniform, uniform_support
from test_inference import test, enumsamp_test
from numpy import linspace

def coin(prob, data):
    z = prob.sample(uniform(0., 1.))
    for elem in data:
        prob.observe(bernoulli(z), elem)
    return z

def discrete_coin(prob, data):
    z = prob.sample(uniform_support(linspace(0, 1, 100)))
    for elem in data:
        prob.observe(bernoulli(z), elem)
    return z

if __name__ == '__main__':
    model = coin
    data = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    name = "Coin"
    options = {
        'shrink': False,
        'plot_with_support': True,
        'plot_style': 'line'
    }

    test(model, data, name, method=ImportanceSampling, **options)
    test(model, data, name, method=MetropolisHastings, **options)

    ## Version continue

    #Tourne indéfiniement
    #rejsamp_test(model, data, name, plot_with_support, plot_style)

    #Fonctionne
    #impsamp_test(model, data, name, plot_with_support, plot_style)

    #N'est pas applicable car uniforme n'a pas de support fini.
    #enumsamp_test(model, data, name, plot_with_support, plot_style)

    #Tourne indéfiniement
    #rejsamp_test(model, data, name, plot_with_support, plot_style)


    ## Version discrète
    name = "Discrete coin"
    model = discrete_coin
    options['plot_style'] = 'stem'
    #rejsamp_test(model, data, name, plot_with_support, plot_style)

    test(model, data, name, method=ImportanceSampling, **options)
    test(model, data, name, method=EnumerationSampling, **options)
