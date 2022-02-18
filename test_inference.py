from typing import Callable, TypeVar, Type
from inference import Prob, InferenceMethod, RejectionSampling, \
    ImportanceSampling, EnumerationSampling, MetropolisHastings, \
    HamiltonianMonteCarlo

A = TypeVar('A')
B = TypeVar('B')

def test(model: Callable[[Prob, A], B], data: A, name: str,
         method: Type[InferenceMethod[A, B]] = ImportanceSampling,
         n: int = 1000,
         print_support: bool = False,
         shrink: bool = False,
         plot_with_support: bool = False,
         plot_style: str = 'scatter',
         eps: float = 0.01, L: int = 20) \
         -> None:
    print("-- {}, {} --".format(name, method.name()))
    mh = method(model, data)
    if method == HamiltonianMonteCarlo:
        dist = mh.infer(n=n, eps=eps, L=L)  # type: ignore
    else:
        dist = mh.infer(n=n)
    if shrink:
        dist.shrink_support()
    if print_support:
        supp = dist.get_support()
        assert (supp is not None)
        for i in range(len(supp.values)):
            print(f"{supp.values[i]} {supp.probs[i]}")
    dist.plot(plot_with_support=plot_with_support, plot_style=plot_style,
              model_name=name, method_name=method.name())
