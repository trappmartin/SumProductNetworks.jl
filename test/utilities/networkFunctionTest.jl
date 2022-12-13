using SumProductNetworks
using Test
using Random

@testset "utility functions" begin
    root = FiniteSumNode();

    Random.seed!(1)

    prod1 = FiniteProductNode()
    for d in 1:2 # Assume 2-D data
        add!(prod1, UnivariateNode(Normal(), d))
    end
    add!(root, prod1, log(rand()))

    prod2 = FiniteProductNode()
    for d in 1:2 # Assume 2-D data
        leaf = FiniteSumNode()
        weights = log.(rand(2))
        add!(leaf, IndicatorNode(0, d), weights[1])
        add!(leaf, IndicatorNode(1, d), weights[1])
        add!(prod2, leaf)
    end
    add!(root, prod2, log(rand()))

    spn = SumProductNetwork(root)
    updatescope!(spn)

    @test isnormalized(root) == false

    normalize!(spn)

    @test isnormalized(root) == true
end
