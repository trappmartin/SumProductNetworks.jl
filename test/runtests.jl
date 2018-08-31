using SumProductNetworks

testcases = Dict(
        "utilities"  => ["utilityFunctionTest"],
        "layers"     => ["layersTests", "bayesianLayerTests", "layerFunctionsTests"]
)


# Run tests
path = dirname(@__FILE__)
cd(path)

for (target, list) in testcases
  for t in list
    filename = string(t, ".jl")
    include( joinpath(target, filename) );
  end
end
