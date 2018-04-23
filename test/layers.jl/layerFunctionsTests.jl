using SumProductNetworks, Distributions
import RDatasets.dataset
using Base.Test

@testset "Topological Order Test" begin
    C = 10
    Ch = 4
    D = 10

    layer1 = MultivariateFeatureLayer(collect(1:C), rand(C, D), rand(Bool, C, D), nothing)
    layer2 = MultivariateFeatureLayer(collect(1:C), rand(C, D), rand(Bool, C, D), nothing)
    layer3 = ProductLayer(collect(1:C), rand(Int, Ch, C), SPNLayer[], nothing)

    layer4 = MultivariateFeatureLayer(collect(1:C), rand(C, D), rand(Bool, C, D), nothing)
    layer5 = MultivariateFeatureLayer(collect(1:C), rand(C, D), rand(Bool, C, D), nothing)
    layer6 = ProductLayer(collect(1:C), rand(Int, Ch, C), SPNLayer[], nothing)

    spn = SumLayer(collect(1:C), rand(Int, Ch, C), rand(Ch, C), SPNLayer[], nothing)

    # connect layers
    push!(spn.children, layer3)
    push!(spn.children, layer6)

    layer3.parent = spn
    layer6.parent = spn

    push!(layer3.children, layer1)
    push!(layer3.children, layer2)
    push!(layer6.children, layer4)
    push!(layer6.children, layer5)

    layer1.parent = layer3
    layer2.parent = layer3
    layer4.parent = layer6
    layer5.parent = layer6

    # actual test
    computationOrder = getOrderedLayers(spn)

    # expected computation order
    # 1, 2, 3, 4, 5, 6, spn

    @test computationOrder[end] == spn
    @test computationOrder[1] == layer1
    @test computationOrder[2] == layer2
    @test computationOrder[3] == layer3
    @test computationOrder[4] == layer4
    @test computationOrder[5] == layer5
    @test computationOrder[6] == layer6
end

@testset "Structure Generation" begin

    # create dummy data
    iris = convert(Array, dataset("datasets", "iris"))
    X = iris[:, 1:4]
    Y = Int[findfirst(unique(iris[:,5]) .== yi) for yi in iris[:,5]]

    (N, D) = size(X)
    C = length(unique(Y))
    G = 2
    K = 1

    @testset "Filter Structure" begin
        P = 10
        M = 2
        W = 0
        spn = SumLayer([1], Array{Int,2}(0, 0), Array{Float32, 2}(0, 0), SPNLayer[], nothing)
        imageStructure!(spn, C, D, G, K; parts = P, mixtures = M, window = W)

        computationOrder = order(spn)
        @test length(computationOrder) == 5

        @test size(computationOrder[1]) == (4*M*P*C, 1)
        @test size(computationOrder[2]) == (M*P*C, 4)
        @test size(computationOrder[3]) == (P*C, M)
        @test size(computationOrder[4]) == (C, P)
        @test size(computationOrder[5]) == (1, C)
    end
end
