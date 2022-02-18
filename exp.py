from inference import ImportanceSampling, MetropolisHastings, EnumerationSampling
from distribution import bernoulli, uniform, uniform_support, gaussian
from test_inference import test, enumsamp_test
from numpy import linspace
import math

def exp(prob, data):
    # z = prob.sample(uniform_support(data))
    z = prob.sample(uniform(0, 1))
    prob.observe(gaussian(0, 1), z)
    return -math.log(z)

def exp_discrete(prob, data):
    z = prob.sample(uniform_support(data))
    return -math.log(z)

if __name__ == '__main__':
    model = exp_discrete
    data = list(linspace(0.00000001, 1, 20))
    name = "Exp"
    options = {
        'n': 1000,
        'shrink': False,
        'plot_with_support': True,
        'plot_style': 'scatter',
        'print_support': True
    }

    test(model, data, name, method=EnumerationSampling, **options)

    model = exp
    
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