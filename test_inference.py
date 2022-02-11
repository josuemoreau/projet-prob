from inference import RejectionSampling, ImportanceSampling, EnumerationSampling

def rejsamp_test(foo, data, name, plot_with_support=False, plot_style='scatter'):
    print("-- {}, Basic Rejection Sampling --".format(name))
    rejsamp = RejectionSampling(foo, data)
    dist = rejsamp.infer()
    supp = dist.get_support(shrink=True)
    for i in range(len(supp.values)):
        print(f"{supp.values[i]} {supp.probs[i]}")
    dist.plot(plot_with_support=plot_with_support, plot_style=plot_style)

def impsamp_test(foo, data, name, plot_with_support=False, plot_style='scatter'):
    print("-- {}, Basic Importance Sampling --".format(name))
    impsamp = ImportanceSampling(foo, data)
    dist = impsamp.infer()
    supp = dist.get_support(shrink=True)
    for i in range(len(supp.values)):
        print(f"{supp.values[i]} {supp.probs[i]}")
    dist.plot(plot_with_support=plot_with_support, plot_style=plot_style)

def enumsamp_test(foo, data, name, plot_with_support=False, plot_style='scatter'):
    print("-- {}, Enumeration Sampling --".format(name))
    dist = EnumerationSampling.infer(foo, data)
    supp = dist.get_support(shrink=True)
    for i in range(len(supp.values)):
        print(f"{supp.values[i]} {supp.probs[i]}")
    dist.plot(plot_with_support=plot_with_support, plot_style=plot_style)