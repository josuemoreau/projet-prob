from distribution import binomial, uniform
from test_inference import *

def laplace(prob, data):
    p = prob.sample(uniform(0., 1.))
    """g = prob.sample(binomial(p, 493472)
    prob.assume(g = 241945)"""
    prob.observe(binomial(p, 493472), 241945)
    return p

if __name__ == '__main__':
    model = laplace
    data = None
    name = "Laplace"
    options = {
        'shrink': False,
        'plot_with_support': True,
        'plot_style': 'line'
    }
    #Tourne beaucoup trop longtemps
    #rejsamp_test(model, data, name, plot_with_support, plot_style)

    #Fonctionne
    impsamp_test(model, data, name, **options)
    mh_test(model, data, name, **options)

    #N'est pas applicable car uniforme n'a pas de support fini.
    #enumsamp_test(model, data, name, plot_with_support, plot_style)
