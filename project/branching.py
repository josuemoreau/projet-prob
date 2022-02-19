from types import NoneType
from inference import RejectionSampling, EnumerationSampling, ImportanceSampling, MetropolisHastings, Prob
from distribution import bernoulli, binomial
from test_inference import test

def branching(prob: Prob, data: NoneType) -> int:
    n = prob.sample(binomial(0.5, 10))
    acc = 0
    for _ in range(n):
        coin = prob.sample(bernoulli(0.5))
        if coin == 0:
            coin = prob.sample(bernoulli(0.25))
        acc += coin
    return acc

if __name__ == '__main__':
    model = branching
    data = None
    name = "Branching"
    options = {
        'shrink': True,
        'plot_with_support': True,
        'plot_style': 'stem'
    }
    test(model, data, name, method=RejectionSampling, **options)  # type: ignore
    test(model, data, name, method=ImportanceSampling, **options)  # type: ignore
    test(model, data, name, method=EnumerationSampling, **options)  # type: ignore
    test(model, data, name, method=MetropolisHastings, **options)  # type: ignore