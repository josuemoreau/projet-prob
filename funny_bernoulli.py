from distribution import bernoulli
from test_inference import *

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
    plot_with_support = True
    plot_style = 'bar'
    rejsamp_test(foo, data, name, plot_with_support, plot_style)
    impsamp_test(foo, data, name, plot_with_support, plot_style)
    enumsamp_test(foo, data, name, plot_with_support, plot_style)
