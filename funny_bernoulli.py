from inference import RejectionSampling, ImportanceSampling, MetropolisHastings, EnumerationSampling
from distribution import bernoulli
from test_inference import test, enumsamp_test

def funny_bernoulli(prob, _x):
    a = prob.sample(bernoulli(0.5))
    b = prob.sample(bernoulli(0.5))
    c = prob.sample(bernoulli(0.5))
    prob.assume(a == 1 or b == 1)
    return a + b + c

if __name__ == '__main__':
    foo = funny_bernoulli
    data = None
    name = "Funny Bernoulli"
    options = {
        'shrink': True,
        'plot_with_support': True,
        'plot_style': 'stem'
    }
    test(foo, data, name, method=RejectionSampling, **options)
    test(foo, data, name, method=ImportanceSampling, **options)
    test(foo, data, name, method=EnumerationSampling, **options)
    test(foo, data, name, method=MetropolisHastings, **options)
