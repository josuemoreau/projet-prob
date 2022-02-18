from inference import ImportanceSampling, MetropolisHastings, EnumerationSampling
from distribution import bernoulli, uniform, uniform_support, gaussian
from test_inference import test, enumsamp_test
from numpy import linspace
import math

def gaussian_model(prob, data):
    # x = prob.sample(uniform(0, 1))
    # y = prob.sample(uniform(0, 1))
    # r = math.sqrt(-2 * math.log(x)) * math.cos(2 * math.pi * y)
    r = prob.sample(gaussian(0, 1))
    prob.observe(gaussian(0, 1), r)
    return r

# def exp_discrete(prob, data):
#     z = prob.sample(uniform_support(data))
#     return -math.log(z)

if __name__ == '__main__':
    model = gaussian_model
    data = None  # list(linspace(0.00000001, 1, 20))
    name = "Gaussian"
    options = {
        'n': 1000,
        'shrink': False,
        'plot_with_support': True,
        'plot_style': 'line',
        'print_support': True
    }

    # test(model, data, name, method=EnumerationSampling, **options)

    # model = exp
    
    test(model, data, name, method=ImportanceSampling, **options)
    test(model, data, name, method=MetropolisHastings, **options)