using SumProductNetworks

println("[runtests.jl] runtests.jl loaded")

testcases = Dict(
        "utilities.jl"  => ["utilityFunctionTest"],
        "layers.jl"     => ["layersTests", "bayesianLayerTests", "layerFunctionsTests"]
)


# Run tests
path = dirname(@__FILE__)
cd(path)

println("[runtests.jl] CDed test path")
println("[runtests.jl] testing starts")

for (target, list) in testcases
  for t in list
    println("[runtests.jl] \"$target/$t.jl\" is running")
    include(target*"/"t*".jl");
    println("[runtests.jl] \"$target/$t.jl\" is successful")
  end
end
println("[runtests.jl] all tests pass")
