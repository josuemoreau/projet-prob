from distribution import bernoulli
from inference import RejectionSampling


def funny_bernoulli(prob, _x):
    a = prob.sample(bernoulli(0.5))
    b = prob.sample(bernoulli(0.5))
    c = prob.sample(bernoulli(0.5))
    prob.assume(a == 1 or b == 1)
    return a + b + c


if __name__ == '__main__':
    print("-- Funny Bernoulli, Basic Rejection Sampling --")
    rs = RejectionSampling(funny_bernoulli, None)
    dist = rs.infer(n=100)
    supp = dist.get_support(shrink=True)
    dist.plot()
