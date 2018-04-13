using SumProductNetworks
using FactCheck
using BenchmarkTools

println("Running test: ", now())

facts("Utility Functions") do
    context("logsumexp Test") do

        # log sum exp function as in literatur
        function logsumexp_batch(X::Vector)
            alpha = maximum(X)  # Find maximum value in X
            log(sum(exp.(X-alpha))) + alpha
        end

        for r in 1:100 # run the test for multiple random values

            X = log.(rand(Float64, 10, 5))

            r1 = vec(mapslices(logsumexp_batch, X, 1))
            r2 = logsumexp(X, dim = 1)

            # fast log sum exp in first dimension
            @fact sum(abs.(r1 - r2)) --> roughly(0.; atol=1e-10)

            X = X'
            # fast log sum exp in second dimension
            r2 = logsumexp(X, dim = 2)
            @fact sum(abs.(r1 - r2)) --> roughly(0.; atol=1e-10)
        end
    end
end
