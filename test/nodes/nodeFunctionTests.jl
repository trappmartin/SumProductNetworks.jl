using SumProductNetworks
using Distributions
using Test

@testset "indicator node" begin
    value = 1
    dim = 2
    node = IndicatorNode(value, dim)

    @test logpdf(node, [1, 2]) == -Inf
    @test logpdf(node, [1, 1]) == 0.
end

@testset "univariate node" begin
    dist = Normal()
    node = UnivariateNode(dist, 2)

    @test logpdf(node, [0.1, 1.0]) ≈ logpdf(dist, 1.0)
end

@testset "multivariate node" begin
    dist = MvNormal(ones(2))
    node = MultivariateNode(dist, [1, 2])

    @test logpdf(node, [1., 0.2, 2.]) ≈ logpdf(dist, [1., 0.2])
end

@testset "sum node" begin

    node = FiniteSumNode{Float32}()

    add!(node, IndicatorNode(0, 1), log(0.3))
    add!(node, IndicatorNode(1, 1), log(0.7))

    @test logpdf(node, [0]) ≈ log(0.3)
    @test logpdf(node, [1]) ≈ log(0.7)
end

@testset "product node" begin

    node = FiniteProductNode()
    add!(node, IndicatorNode(1, 1))
    add!(node, IndicatorNode(1, 2))

    @test logpdf(node, [0, 1]) == -Inf
    @test logpdf(node, [1, 0]) == -Inf
    @test logpdf(node, [1, 1]) == 0
end
