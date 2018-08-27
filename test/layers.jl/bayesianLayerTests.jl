using SumProductNetworks, Distributions
using Test

@testset "Bayesian SPN layers" begin

	@testset "Bayesian Sum Layer" begin

		C = 10 # number of nodes
		Ch = 5 # number of children
		D = 10 # dimensionality
		N = 100 # number of samples
		α = 1.

		layer = BayesianSumLayer(collect(1:C), Ch, N, D, α)
		layer.childIds[:,:] = reduce(hcat, [[i+(j-1)*Ch for i in 1:Ch] for j in 1:C]) + C

		@test size(layer) == (C, Ch)
		@test size(sstats(layer)) == (Ch, C)
		@test size(cids(layer)) == (Ch, C)

		X = rand(Int, D, N)
		llhvals = zeros(N, C + C*Ch)

		@test all(llhvals .== 0.)

		llhvals[:, C+1:end] = log.(rand(N, C*Ch))

		@inferred evaluate!(layer, X, llhvals)
		@test all(isfinite.(llhvals[:,1:C]))
	end

	@testset "Bayesian Product Layer" begin
		C = 10 # number of nodes
		Ch = 5 # number of children
		D = 10 # dimensionality
		N = 100 # number of samples

		childIds = reduce(hcat, [[i+(j-1)*Ch for i in 1:Ch] for j in 1:C]) + C
		β = 1.

		layer = BayesianProductLayer(collect(1:C), Ch, N, D, β)
		layer.childIds[:,:] = reduce(hcat, [[i+(j-1)*Ch for i in 1:Ch] for j in 1:C]) + C

		@test size(layer) == (C, Ch)
		@test size(sstats(layer)) == (Ch, C)
		@test size(cids(layer)) == (Ch, C)
	end

	@testset "Bayesian Categorical Layer" begin

		C_ = 10
		D = 5
		S = 2
		N = 100 # number of samples

		C = C_ * D

		ids = collect(1:C)
		scopes = vec(repmat(collect(1:D), 1, C_))
		γ = 1.

		layer = BayesianCategoricalLayer(ids, scopes, S, N, D, γ)

		@test size(layer) == (C, 1)
		@test size(sstats(layer)) == (S, C)

		X = rand(1:S, N, D)
		llhvals = zeros(N, C)

		@test all(llhvals .== 0.)

		@inferred evaluate!(layer, X, llhvals)
		@test all(isfinite.(llhvals[:,1:C]))
	end
end
