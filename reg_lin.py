from inference import HamiltonianMonteCarlo, ImportanceSampling, MetropolisHastings, EnumerationSampling
from distribution import bernoulli, uniform, uniform_support, gaussian
from test_inference import test
import numpy as np
import math
from matplotlib import pyplot as plt

def linear_regression(prob, data):
    a = prob.sample(gaussian(data['a_lower'], data['a_upper']))
    b = prob.sample(uniform(data['b_lower'], data['b_upper']))
    s = prob.sample(uniform(0, 1))
    for i in range(data['N']):
        prob.observe(gaussian(a * data['x'][i] + b, s), data['y'][i])
    return a, b, s


def test_linear_regression(model, data, method, model_name, n=1000, only_best_values=None, plot_only_mean=True):
    m = method(model, data)
    dist = m.infer(n=n)
    supp = dist.get_support()

    res = [((supp.values[i][0], supp.values[i][1]), supp.probs[i]) for i in range(len(supp.values))]
    sorted_res = sorted(res, key=lambda x: x[1], reverse=True)

    if not plot_only_mean:
        for i in range(n if only_best_values is None else only_best_values):
            x = np.linspace(data['lower'], data['upper'], 2)
            a, b = sorted_res[i][0]
            plt.plot(x, a * x + b, color='blue', alpha=0.1, zorder=0)

    if only_best_values is not None:
        a = np.average([r[0][0] for r in sorted_res[:only_best_values]], 
                    weights=[r[1] for r in sorted_res[:only_best_values]])
        b = np.average([r[0][1] for r in sorted_res[:only_best_values]],
                    weights=[r[1] for r in sorted_res[:only_best_values]])
    else:
        a = np.average([r[0][0] for r in sorted_res], 
                    weights=[r[1] for r in sorted_res])
        b = np.average([r[0][1] for r in sorted_res],
                    weights=[r[1] for r in sorted_res])

    x = np.linspace(data['lower'], data['upper'], 2)
    plt.plot(x, a * x + b, color='green', alpha=1, zorder=1)
    
    plt.scatter(data['x'], data['y'], color='red', zorder=1)
    plt.title(f"{model_name} - {method.name()}")
    plt.show()


if __name__ == '__main__':
    model = linear_regression
    name = "Linear Regression"

    N = 8
    noise = 0.25
    lower = 0
    upper = 10

    x_obs = np.linspace(lower, upper, N)
    y_obs = 2 * np.tanh(4 * (x_obs - upper) / upper) + noise * np.random.randn(N)

    data = {
        'N': N,
        'lower': lower,
        'upper': upper,
        'x': x_obs,
        'y': y_obs,
        'a_lower': 0,
        'a_upper': 1,
        'b_lower': -10,
        'b_upper': 10
    }
    
    test_linear_regression(model, data, ImportanceSampling, name, 2000, 100, True)
    test_linear_regression(model, data, MetropolisHastings, name, 2000, 100, True)