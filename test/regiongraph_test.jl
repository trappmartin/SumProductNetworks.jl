using SumProductNetworks
using Test, Random

Random.seed!(1)

# some test data
x = hcat(randn(100,1), randn(100,1)*0.2, randn(100,1)*2.0)

# build a region graph
spn = ratspn(x)
