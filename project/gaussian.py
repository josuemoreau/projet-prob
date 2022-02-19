from inference import ImportanceSampling, MetropolisHastings, EnumerationSampling, Prob
from distribution import bernoulli, uniform, uniform_support, gaussian
from test_inference import test
from numpy import linspace
import math

def gaussian_model(prob: Prob, data: None) -> float:
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
        'print_support': False
    }
    
    test(model, data, name, method=ImportanceSampling, **options)  # type: ignore
    test(model, data, name, method=MetropolisHastings, **options)  # type: ignore