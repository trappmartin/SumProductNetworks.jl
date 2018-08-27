using SumProductNetworks, Distributions
using Test

@testset "SPN layers" begin

	@testset "Multivariate Feature Layer" begin

		C = 50 # number of nodes
		D = 10 # dimensionality
		N = 100 # number of samples

		weights = rand(D, C)
		scopes = rand(Bool, D, C)
		layer = MultivariateFeatureLayer(collect(1:C), weights, scopes, nothing)

		@test size(layer) == (C, 1)

		X = rand(N, D)
		llhvals = zeros(N, C)
		@test all(llhvals .== 0.) == true

		@inferred evaluate!(layer, X, llhvals)

		@test all( llhvals .≈  -log.(1. + exp.( X * (layer.weights .* layer.scopes) )))
	end

	@testset "Sum Layer" begin

		C = 10 # number of nodes
		Ch = 5 # number of children
		D = 10 # dimensionality
		N = 100 # number of samples

		childIds = reduce(hcat, [[i+(j-1)*Ch for i in 1:Ch] for j in 1:C]) + C
		w = log.(rand(Dirichlet([1./Ch for j in 1:Ch]), C))
		layer = SumLayer(collect(1:C), childIds, w, nothing, nothing)

		@test size(layer) == (C, Ch)
		@test size(weights(layer)) == (Ch, C)
		@test size(cids(layer)) == (Ch, C)

		X = rand(N, D)
		llhvals = zeros(N, C + C*Ch)

		@test all(llhvals .== 0.)

		llhvals[:, C+1:end] = log.(rand(N, C*Ch))

		@inferred evaluate!(layer, X, llhvals)

		Y = zeros(N, C + C*Ch)
		Y[:, C+1:end] = llhvals[:, C+1:end]
		for c in 1:C
		  for n in 1:N
			Y[n, c] = StatsFuns.logsumexp(Y[n, childIds[:,c]] + w[:,c])
		  end
		end

		@test all(llhvals[:,1:C] .≈ Y[:,1:C])

		# set some weights to 0. should still validate to llhvals > -Inf
		weights(layer)[2:end, :] = -Inf
		evaluate!(layer, X, llhvals)

		@test all(isfinite.(llhvals[:,1:C]))
	end

	@testset "Product Layer" begin
		C = 10 # number of nodes
		Ch = 5 # number of children
		D = 10 # dimensionality
		N = 100 # number of samples

		childIds = reduce(hcat, [[i+(j-1)*Ch for i in 1:Ch] for j in 1:C]) + C
		layer = ProductLayer(collect(1:C), childIds, nothing, nothing)

		@test size(layer) == (C, Ch)
		@test size(cids(layer)) == (Ch, C)

		X = rand(D, N)
		llhvals = zeros(N, C + C*Ch)

		@test all(llhvals .== 0.)

		llhvals[:, C+1:end] = rand(N, C*Ch)

		@inferred evaluate!(layer, X, llhvals)

		Y = zeros(N, C + C*Ch)
		Y[:, C+1:end] = llhvals[:, C+1:end]
		for c in 1:C
		  for n in 1:N
			Y[n, c] = sum(Y[n, childIds[:,c]])
		  end
		end

		@test all(llhvals[:,1:C] .≈ Y[:,1:C])
		@test all(isfinite.(llhvals))
	end

	@testset "Product Class Layer" begin

	    C = 10 # number of nodes
	    Ch = 5 # number of children
	    D = 10 # dimensionality
	    N = 100 # number of samples

	    childIds = reduce(hcat, [[i+(j-1)*Ch for i in 1:Ch] for j in 1:C]) + C
	    clabels = collect(1:C)
	    layer = ProductCLayer(collect(1:C), childIds, clabels, nothing, nothing)

	    @test size(layer) == (C, Ch)
	    @test size(cids(layer)) == (Ch, C)

	    X = rand(D, N)
	    y = rand(1:C, N)
	    llhvals = zeros(N,C + C*Ch)

	    @test all(llhvals .== 0.)

	    llhvals[:,C+1:end] = rand(N,C*Ch)

	    @inferred evaluateCLLH!(layer, X, y, llhvals)

	    Y = zeros(N,C + C*Ch)
	    Y[:,C+1:end] = llhvals[:,C+1:end]
	    for c in 1:C
	      for n in 1:N
	        Y[n,c] = sum(Y[n,childIds[:,c]]) + log(y[n] == clabels[c])
	      end
	    end
	    @test all(llhvals[:,1:C] .≈ Y[:,1:C])
	end

	@testset "Indicator Layer" begin
	  C = 5 # number of nodes (values)
	  D = 10 # dimensionality
	  N = 100 # number of samples

	  ids = collect(1:D*C) # reshape ids to D * C
	  scopes = randperm(D)
	  values = collect(1:C)
	  layer = IndicatorLayer(ids, scopes, values, nothing)

	  @test size(layer) == (D, C)

	  X = rand(1:C, N, D)
	  llhvals = zeros(N, D * C)

	  @inferred evaluate!(layer, X, llhvals)
	  @test all(llhvals[1, :] .≈ reduce(vcat, [log.(X[1, scopes] .== c) for c in 1:C]))

	end
end
