using SumProductNetworks
using FactCheck
FactCheck.setstyle(:compact)

println("Running test: ", now())

facts("Layers Test") do
    context("Mutlivariate Feature Layer") do
        w = rand(50, 10)
        scopes = rand(Bool, 50, 10)
        layer = MultivariateFeatureLayer(1, w, scopes, nothing)
        
        X = rand(100, 10)
        

        @fact 1 --> 1
    end

end
