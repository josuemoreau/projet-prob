from inference import RejectionSampling, ImportanceSampling, EnumerationSampling, MetropolisHastings


def rejsamp_test(foo, data, name, shrink=False,
                 plot_with_support=False, plot_style='scatter'):
    print("-- {}, Basic Rejection Sampling --".format(name))
    rejsamp = RejectionSampling(foo, data)
    dist = rejsamp.infer()
    if shrink:
        dist.shrink_support()
    supp = dist.get_support()
    for i in range(len(supp.values)):
        print(f"{supp.values[i]} {supp.probs[i]}")
    dist.plot(plot_with_support=plot_with_support, plot_style=plot_style)


def impsamp_test(foo, data, name, shrink=False,
                 plot_with_support=False, plot_style='scatter'):
    print("-- {}, Basic Importance Sampling --".format(name))
    impsamp = ImportanceSampling(foo, data)
    dist = impsamp.infer()
    if shrink:
        dist.shrink_support()
    supp = dist.get_support()
    for i in range(len(supp.values)):
        print(f"{supp.values[i]} {supp.probs[i]}")
    dist.plot(plot_with_support=plot_with_support, plot_style=plot_style)


def enumsamp_test(foo, data, name, shrink=False,
                  plot_with_support=False, plot_style='scatter'):
    print("-- {}, Enumeration Sampling --".format(name))
    dist = EnumerationSampling.infer(foo, data)
    if shrink:
        dist.shrink_support()
    supp = dist.get_support()
    for i in range(len(supp.values)):
        print(f"{supp.values[i]} {supp.probs[i]}")
    dist.plot(plot_with_support=plot_with_support, plot_style=plot_style,
              model_name=name, method_name='Enumeration Sampling')


def mh_test(foo, data, name, shrink=False,
            plot_with_support=False, plot_style='scatter'):
    print("-- {}, Metropolis-Hastings --".format(name))
    mh = MetropolisHastings(foo, data)
    dist = mh.infer()
    if shrink:
        dist.shrink_support()
    supp = dist.get_support()
    # for i in range(len(supp.values)):
    #     print(f"{supp.values[i]} {supp.probs[i]}")
    dist.plot(plot_with_support=plot_with_support, plot_style=plot_style)


def test(model, data, name, method=ImportanceSampling,
         print_support=False,
         shrink=False, plot_with_support=False, plot_style='scatter'):
    print("-- {}, {} --".format(name, method.name()))
    mh = method(model, data)
    dist = mh.infer()
    if shrink:
        dist.shrink_support()
    if print_support:
        supp = dist.get_support()
        for i in range(len(supp.values)):
            print(f"{supp.values[i]} {supp.probs[i]}")
    dist.plot(plot_with_support=plot_with_support, plot_style=plot_style,
              model_name=name, method_name=method.name())
