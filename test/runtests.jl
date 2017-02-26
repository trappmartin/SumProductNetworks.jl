using SumProductNetworks
using FactCheck
using Distributions, StatsFuns

println("Running test: ", now())

facts("Layers Test") do
    context("Multivariate Feature Layer") do

        C = 50 # number of nodes
        D = 10 # dimensionality
        N = 100 # number of samples

        weights = rand(C, D)
        scopes = rand(Bool, C, D)
        layer = MultivariateFeatureLayer(collect(1:C), weights, scopes, nothing)

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
        layer = SumLayer(collect(1:C), childIds, weights, nothing, nothing)

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
        layer = ProductLayer(collect(1:C), childIds, nothing, nothing)

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
        layer = ProductCLayer(collect(1:C), childIds, clabels, nothing, nothing)

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

facts("Topological Order Test") do
    context("Layers") do
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
        computationOrder = order(spn)

        # expected computation order
        1, 2, 3, 4, 5, 6, spn

        @fact computationOrder[end] --> spn
        @fact computationOrder[1] --> layer1
        @fact computationOrder[2] --> layer2
        @fact computationOrder[3] --> layer3
        @fact computationOrder[4] --> layer4
        @fact computationOrder[5] --> layer5
        @fact computationOrder[6] --> layer6
    end

    context("Nodes") do
        println("tbd")
        @fact 1 --> 1
    end


end
