from distribution import bernoulli, uniform, uniform_support
from test_inference import *
from numpy import linspace

def coin(prob, data):
    z = prob.sample(uniform(0., 1.))
    for elem in data:
        prob.observe(bernoulli(z), elem)
    return z

def discrete_coin(prob, data):
    z = prob.sample(uniform_support(linspace(0, 1, 20)))
    for elem in data:
        prob.observe(bernoulli(z), elem)
    return z

if __name__ == '__main__':
    model = coin
    data = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    name = "Coin"
    plot_with_support = True
    plot_style = 'line'
    remove_first_iterations = 300

    metropolis_hastings_test(model, data, name,
                             remove_first_iterations,
                             plot_with_support, plot_style)

    ## Version continue

    #Tourne indéfiniement
    #rejsamp_test(model, data, name, plot_with_support, plot_style)
    
    #Fonctionne
    #impsamp_test(model, data, name, plot_with_support, plot_style)
    
    #N'est pas applicable car uniforme n'a pas de support fini.
    #enumsamp_test(model, data, name, plot_with_support, plot_style)

    #Tourne indéfiniement
    #rejsamp_test(model, data, name, plot_with_support, plot_style)
    

    ## Version discrète
    name = "Discrete coin"
    model = discrete_coin
    plot_style = 'bar'
    #rejsamp_test(model, data, name, plot_with_support, plot_style)
    impsamp_test(model, data, name, plot_with_support, plot_style)
    enumsamp_test(model, data, name, plot_with_support, plot_style)
