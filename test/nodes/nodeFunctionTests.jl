using SumProductNetworks
using FactCheck
using BenchmarkTools

facts("Node Function Functions") do

	context("sum node evaluate test") do

		S = FiniteSumNode{Float32}(1, 1, 1)
		S.cids = Int[2, 3]
		S.logweights = map(Float32, log.([0.3, 0.7]))

		llhvals = log.(rand(3, 3))
		llhvals[:,1] = 0.

		evaluate!(S, rand(), llhvals)

		for i in 1:3
			L = llhvals[i,2:3] + S.logweights
			a = maximum(L)
			@fact llhvals[i,1] --> log(sum(exp.(L - a))) + a
		end
	end

	context("product node evaluate test") do

		P = FiniteProductNode(1, 2, 1)
		P.cids = Int[2, 3]
		setScope!(P, [1, 2])
		add!(P, IndicatorNode{Int}(2, 1, 2))
		add!(P, IndicatorNode{Int}(3, 1, 2))

		setScope!(children(P)[1], 1)
		setScope!(children(P)[2], 2)

		llhvals = log.(rand(3, 3))
		llhvals[:,1] = 0.

		evaluate!(P, rand(), llhvals)

		for i in 1:3
			@fact llhvals[i,1] --> sum(llhvals[i,2:3])
		end
	end
end
