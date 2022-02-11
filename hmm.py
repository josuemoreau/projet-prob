from distribution import gaussian
from test_inference import *

def laplace(prob, data):
    if len(data) == 0:
        return []
    states = [data[-1]]
    for i in range(len(data)-2, -1, -1):
        x = prob.sample(gaussian(states[-1], 1.0))
        prob.oberve(gaussian(x, 1.0), data[i])
        states.append(x)
    return states

'''if __name__ == '__main__':
    model = hmm
    data = ...
    name = "Hidden Markov Chain"
    plot_with_support = True
    plot_style = 'line'

    #Tourne beaucoup trop longtemps
    #rejsamp_test(model, data, name, plot_with_support, plot_style)
    
    #Fonctionne
    impsamp_test(model, data, name, plot_with_support, plot_style)
    
    #N'est pas applicable car uniforme n'a pas de support fini.
    #enumsamp_test(model, data, name, plot_with_support, plot_style)'''