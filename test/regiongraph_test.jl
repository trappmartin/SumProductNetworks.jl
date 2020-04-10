using SumProductNetworks
using StatsFuns
using BenchmarkTools
using Test, Random

Random.seed!(1)

# some test data
x = hcat(randn(100,1), rand(1:10, 100,1), rand(Bool,100,1))

catlogpdf(x::Int, p...) = log(p[x])
catlogpdf(x::AbstractFloat, p...) = catlogpdf(Int(x), p...)
bernlogpdf(x::Int, p) = x == 0 ? 1-p : p
bernlogpdf(x::AbstractFloat, p) = bernlogpdf(Int(x), p)

likelihoods = Function[(x, μ, σ) -> normlogpdf(μ, σ, x), catlogpdf, bernlogpdf]
priors = Distribution[NormalInverseGamma(), Dirichlet(10, 1.0), Beta()]

@testset "graph nodes" begin
    @testset "atomic region" begin

        K = 10
        parameters = draw(priors, K)
        node = FactorizedAtomicRegion( likelihoods, parameters, 3 )
        @test node isa FactorizedAtomicRegion

        lp = logpdf(node, x)
        @test size(lp) == (100, K)

        @test all(lp .== 0.0)

        setscope!(node, [1])
        lp = logpdf(node, x)
        lp2 = zero(lp)
        for (p,l) in zip(eachslice(parameters[1], dims=2), eachslice(lp2, dims=2))
            l[:] = normlogpdf.(p[1], p[2], x[:,1])
        end
        @test all(lp .== lp2)

        setscope!(node, [1,2])
        @test scope(node) == [1,2]
        lp = logpdf(node, x)
        lp2 = zero(lp)
        for (p,l) in zip(eachslice(parameters[1], dims=2), eachslice(lp2, dims=2))
            l[:] = normlogpdf.(p[1], p[2], x[:,1])
        end
        for (p,l) in zip(eachslice(parameters[2], dims=2), eachslice(lp2, dims=2))
            l[:] += catlogpdf.(x[:,2], p...)
        end
        @test all(lp .== lp2)

    end
    @testset "sum region" begin



    end
end

# build a region graph
#spn = ratspn(x)
