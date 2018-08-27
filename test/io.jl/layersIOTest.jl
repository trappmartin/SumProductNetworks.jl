using SumProductNetworks, Distributions
using Test

@testset "Export Layered SPN Test" begin

    N = 100
    D = 4
    M = 4 # number of children under a sum node
    K = 4 # number of children under a product node
    L = 2 # number of product-sum layers (excluding the root)
    S = 2 # states

    spn = create_bayesian_discrete_layered_spn(M, K, L, N, D, S; α = 1.0, β = 1.0, γ = 1.0)

    outDir = tempdir()
    filename_dot = randstring()
    filename_param = randstring()
    output_dot = joinpath(outDir, filename_dot)
    output_param = joinpath(outDir, filename_param)

    info(outDir)

    exportNetwork(spn, output_dot, output_param; nodeObsDegeneration = false, excludeDegenerated = false)

    @test isfile(string(output_dot, ".dot"))
    @test isfile(string(output_param, "_layer7.jld2"))

    mv(string(output_dot, ".dot"), "exportedNetwork.dot", force=true)
    mv(string(output_param, "_layer7.jld2"), "spn.jld2", force=true)

end
