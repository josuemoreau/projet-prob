from inference import ImportanceSampling, MetropolisHastings, EnumerationSampling
from distribution import bernoulli, uniform, uniform_support, gaussian
from test_inference import test
from numpy import linspace
import math

def gaussian_model(prob, data):
    x = prob.sample(uniform(0, 1))
    y = prob.sample(uniform(0, 1))
    r = math.sqrt(-2 * math.log(x)) * math.cos(2 * math.pi * y)
    prob.observe(gaussian(0, 1), r)
    return r

if __name__ == '__main__':
    model = gaussian_model
    data = None
    name = "Gaussian"
    options = {
        'n': 1000,
        'shrink': False,
        'plot_with_support': True,
        'plot_style': 'line',
        'print_support': True
    }
    
    test(model, data, name, method=ImportanceSampling, **options)
    test(model, data, name, method=MetropolisHastings, **options)