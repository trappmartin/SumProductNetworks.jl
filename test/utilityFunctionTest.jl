using SumProductNetworks
using FactCheck
using BenchmarkTools

println("Running test: ", now())

facts("Utility Functions") do
    context("logsumexp Test") do

        X = rand(Float64, 10, 5)

        function logsumexp_batch(X::Vector)
            alpha = maximum(X)  # Find maximum value in X
            log(sum(exp.(X-alpha))) + alpha
        end

        r1 = vec(mapslices(logsumexp_batch, X, 1))
        r2 = logsumexp(X, dim = 1)

        @fact all(r1 .== r2) --> true

    end
end
