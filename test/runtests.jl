using SumProductNetworks
using Test

testcases = Dict(
        "utilities"  => ["utilityFunctionTest"],
        "layers"     => ["layersTests", "bayesianLayerTests", "layerFunctionsTests"],
        "nodes"     => ["nodeFunctionTests"]
)


# Run tests
path = dirname(@__FILE__)
cd(path)

@testset "Sum-product network tests" begin
    for (target, list) in testcases
        @testset "$target" begin
            for t in list
                filename = string(t, ".jl")
                include( joinpath(target, filename) );
            end
        end
    end
end
