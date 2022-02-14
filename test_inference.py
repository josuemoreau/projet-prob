from inference import RejectionSampling, ImportanceSampling, EnumerationSampling, MetropolisHastings


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


def metropolis_hastings_test(foo, data, name,
                             remove_first_iterations=0,
                             plot_with_support=False,
                             plot_style='scatter'):
    print("-- {}, Metropolis-Hastings --".format(name))
    mh = MetropolisHastings(foo, data)
    dist = mh.infer(remove_first_iterations=remove_first_iterations)
    supp = dist.get_support(shrink=True)
    for i in range(len(supp.values)):
        print(f"{supp.values[i]} {supp.probs[i]}")
    dist.plot(plot_with_support=plot_with_support, plot_style=plot_style)
