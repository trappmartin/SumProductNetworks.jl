using SumProductNetworks

@info("[runtests.jl] runtests.jl loaded")

testcases = Dict(
        "utilities.jl"  => ["utilityFunctionTest"],
        "layers.jl"     => ["layersTests", "bayesianLayerTests", "layerFunctionsTests"]
)


# Run tests
path = dirname(@__FILE__)
cd(path)

@info("[runtests.jl] CDed test path")
@info("[runtests.jl] testing starts")

for (target, list) in testcases
  for t in list
    filename = string(t, ".jl")
    @info("[runtests.jl] \"$target/$t.jl\" is running")
    include( joinpath(target, filename) );
    @info("[runtests.jl] \"$target/$t.jl\" is successful")
  end
end
@info("[runtests.jl] all tests pass")
