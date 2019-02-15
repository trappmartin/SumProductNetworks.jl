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

    node = FiniteSumNode()

    add!(node, IndicatorNode(0, 1), log(0.3))
    add!(node, IndicatorNode(1, 1), log(0.7))
    updatescope!(node)

    @test logpdf(node, [0]) ≈ log(0.3)
    @test logpdf(node, [1]) ≈ log(0.7)

    @test first(logpdf(node, ones(1, 1))) ≈ log(0.7)
    @test logpdf(node, reshape([1, 0], 2, 1)) ≈ log.([0.7, 0.3])

    spn = SumProductNetwork(node)

    @test exp.(logpdf(spn, reshape([1, 0], 2, 1))) ≈ [0.7, 0.3]
    @test exp.(logpdf(spn, reshape([1, 2], 2, 1))) ≈ [0.7, 0]

    n = FiniteSumNode(D = 2)
    p1 = FiniteProductNode(D = 2)
    add!(p1, IndicatorNode(1, 1))
    add!(p1, IndicatorNode(1, 2))

    p2 = FiniteProductNode(D = 2)
    add!(p2, IndicatorNode(2, 1))
    add!(p2, IndicatorNode(2, 2))

    add!(n, p1, log(0.8))
    add!(n, p2, log(0.2))

    updatescope!(n)

    @test logpdf(n, [1, 1]) == log(0.8)
    @test logpdf(n, [2, 2]) == log(0.2)
    @test logpdf(n, [1, 2]) == log(0.0)
end

@testset "product node" begin

    node = FiniteProductNode()
    add!(node, IndicatorNode(1, 1))
    add!(node, IndicatorNode(1, 2))

    # Product nodes without scope always return p(x) = 1.
    @test logpdf(node, [0, 0]) == 0

    updatescope!(node)

    @test logpdf(node, [0, 1]) == -Inf
    @test logpdf(node, [1, 0]) == -Inf
    @test logpdf(node, [1, 1]) == 0

    n = FiniteProductNode(D = 2)
    s1 = FiniteSumNode(D = 1)
    add!(s1, IndicatorNode(1, 1), log(0.8))
    add!(s1, IndicatorNode(2, 1), log(0.2))

    s2 = FiniteSumNode(D = 1)
    add!(s2, IndicatorNode(1, 2), log(0.2))
    add!(s2, IndicatorNode(2, 2), log(0.8))

    add!(n, s1)
    add!(n, s2)

    updatescope!(n)

    @test logpdf(n, [1, 1]) == log(0.8) + log(0.2)
    @test logpdf(n, [2, 1]) == log(0.2) + log(0.2)
end
