from distribution import bernoulli
from inference import RejectionSampling, ImportanceSampling, EnumerationSampling


def funny_bernoulli(prob, _x):
    a = prob.sample(bernoulli(0.5))
    b = prob.sample(bernoulli(0.5))
    c = prob.sample(bernoulli(0.5))
    prob.assume(a == 1 or b == 1)
    return a + b + c


if __name__ == '__main__':
    print("-- Funny Bernoulli, Basic Rejection Sampling --")
    rs = RejectionSampling(funny_bernoulli, None)
    dist = rs.infer()
    supp = dist.get_support(shrink=True)
    for i in range(len(supp.values)):
        print(f"{supp.values[i]} {supp.probs[i]}")
    dist.plot()

    print("-- Funny Bernoulli, Basic Importance Sampling --")
    impsamp = ImportanceSampling(funny_bernoulli, None)
    dist = impsamp.infer()
    supp = dist.get_support(shrink=True)
    for i in range(len(supp.values)):
        print(f"{supp.values[i]} {supp.probs[i]}")
    dist.plot()
