using SumProductNetworks
using FactCheck
using Distributions, StatsFuns

println("Running test: ", now())

facts("Layers Test") do
    context("Mutlivariate Feature Layer") do

        C = 50 # number of nodes
        D = 10 # dimensionality
        N = 100 # number of samples

        weights = rand(C, D)
        scopes = rand(Bool, C, D)
        layer = MultivariateFeatureLayer(collect(1:C), weights, scopes, -1)

        @fact size(layer) --> (C, D)

        X = rand(D, N)
        llhvals = zeros(C, N)
        @fact all(llhvals .== 0.) --> true

        eval!(layer, X, llhvals)
        W = weights .* scopes # C x D
        @fact llhvals --> W * X
    end

    context("Sum Layer") do

        C = 10 # number of nodes
        Ch = 5 # number of children
        D = 10 # dimensionality
        N = 100 # number of samples

        childIds = reduce(hcat, [[i+(j-1)*Ch for i in 1:Ch] for j in 1:C]) + C
        weights = rand(Dirichlet([1./Ch for j in 1:Ch]), C)
        layer = SumLayer(collect(1:C), childIds, weights, -1)

        @fact size(layer) --> (C, Ch)
        @fact size(layer.weights) --> (Ch, C)
        @fact size(layer.childIds) --> (Ch, C)

        X = rand(D, N)
        llhvals = zeros(C + C*Ch, N)
        @fact all(llhvals .== 0.) --> true

        llhvals[C+1:end, :] = rand(C*Ch, N)

        eval!(layer, X, llhvals)

        Y = zeros(C + C*Ch, N)
        Y[C+1:end, :] = llhvals[C+1:end, :]
        for c in 1:C
          for n in 1:N
            Y[c,n] = logsumexp(Y[childIds[:,c],n] + log(weights[:,c]))
          end
        end
        @fact llhvals[1:C,:] --> Y[1:C,:]

        # set some weights to 0. should still validate to llhvals > -Inf
        layer.weights[2:end, :] = 0.
        eval!(layer, X, llhvals)
        @fact all(isfinite(llhvals[1:C,:])) --> true
        
        # set all weights to 0. should validate to llhvals = -Inf
        layer.weights[:, :] = 0.
        eval!(layer, X, llhvals)
        @fact all(isfinite(llhvals[1:C,:])) --> false
    end

    context("Product Layer") do

        C = 10 # number of nodes
        Ch = 5 # number of children
        D = 10 # dimensionality
        N = 100 # number of samples

        childIds = reduce(hcat, [[i+(j-1)*Ch for i in 1:Ch] for j in 1:C]) + C
        layer = ProductLayer(collect(1:C), childIds, -1)

        @fact size(layer) --> (C, Ch)
        @fact size(layer.childIds) --> (Ch, C)

        X = rand(D, N)
        llhvals = zeros(C + C*Ch, N)
        @fact all(llhvals .== 0.) --> true

        llhvals[C+1:end, :] = rand(C*Ch, N)

        eval!(layer, X, llhvals)

        Y = zeros(C + C*Ch, N)
        Y[C+1:end, :] = llhvals[C+1:end, :]
        for c in 1:C
          for n in 1:N
            Y[c,n] = sum(Y[childIds[:,c],n])
          end
        end
        @fact llhvals[1:C,:] --> Y[1:C,:]
        @fact all(isfinite(llhvals)) --> true
    end

    context("Product Class Layer") do

        C = 10 # number of nodes
        Ch = 5 # number of children
        D = 10 # dimensionality
        N = 100 # number of samples

        childIds = reduce(hcat, [[i+(j-1)*Ch for i in 1:Ch] for j in 1:C]) + C
        clabels = collect(1:C)
        layer = ProductCLayer(collect(1:C), childIds, clabels, -1)

        @fact size(layer) --> (C, Ch)
        @fact size(layer.childIds) --> (Ch, C)

        X = rand(D, N)
        y = rand(1:C, N)
        llhvals = zeros(C + C*Ch, N)
        @fact all(llhvals .== 0.) --> true

        llhvals[C+1:end, :] = rand(C*Ch, N)

        eval!(layer, X, y, llhvals)

        Y = zeros(C + C*Ch, N)
        Y[C+1:end, :] = llhvals[C+1:end, :]
        for c in 1:C
          for n in 1:N
            Y[c,n] = sum(Y[childIds[:,c],n]) + log(y[n] == clabels[c])
          end
        end
        @fact llhvals[1:C,:] --> Y[1:C,:]
    end

end
