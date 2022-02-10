from distribution import bernoulli
from inference import RejectionSampling

def funny_bernoulli(sampler, _x):
    a = sampler.sample(bernoulli(0.5))
    b = sampler.sample(bernoulli(0.5))
    c = sampler.sample(bernoulli(0.5))
    sampler.assume(a == 1 or b == 1)
    return a + b + c

if __name__ == '__main__':
    print("-- Funny Bernoulli, Basic Rejection Sampling --")
    rs = RejectionSampling(funny_bernoulli, None)
    dist = rs.infer()
    supp = dist.get_support(shrink=True)
    dist.plot()