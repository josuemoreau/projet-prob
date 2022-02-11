from distribution import bernoulli, uniform
from test_inference import *

def coin(prob, data):
    z = prob.sample(uniform(0., 1.))
    for elem in data:
        prob.observe(bernoulli(z), elem)
    return z

if __name__ == '__main__':
    model = coin
    data = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    name = "Funny Bernoulli"
    plot_with_support = True
    plot_style = 'line'

    #Tourne indéfiniement
    #rejsamp_test(model, data, name, plot_with_support, plot_style)
    
    #Fonctionne
    impsamp_test(model, data, name, plot_with_support)
    
    #N'est pas applicable car uniforme n'a pas de support fini.
    #enumsamp_test(model, data, name, plot_with_support, plot_style)