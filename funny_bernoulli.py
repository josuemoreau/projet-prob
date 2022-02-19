from inference import RejectionSampling, ImportanceSampling, MetropolisHastings, \
    EnumerationSampling, Prob
from distribution import bernoulli
from test_inference import test

def funny_bernoulli(prob: Prob, _x: None) -> int:
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
    test(foo, data, name, method=RejectionSampling, **options)  # type: ignore
    test(foo, data, name, method=ImportanceSampling, **options)  # type: ignore
    test(foo, data, name, method=EnumerationSampling, **options)  # type: ignore
    test(foo, data, name, method=MetropolisHastings, **options)  # type: ignore
