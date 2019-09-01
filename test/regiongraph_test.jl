using SumProductNetworks
using Test, Random

Random.seed!(1)

# some test data
x = hcat(randn(100,1), rand(1:10, 100,1), rand(Bool,100,1))

likelihoods = Type[Normal, Categorical, Bernoulli]
priors = Distribution[NormalInverseGamma(), Dirichlet(2, 1.0), Beta()]

@testset "graph nodes" begin

    @testset "atmoic region" begin

        K = 10
        parameters = map(prior -> rand(prior, K), priors)

        node = FactorizedDistributionGraphNode(
                                               gensym(),
                                               trues(3),
                                               likelihoods,
                                               priors,
                                               parameters
                                              )


    end

end

# build a region graph
#spn = ratspn(x)
