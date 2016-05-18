function generateRandomProduct(sumWidth::Int, depth::Int, μ::Vector{Float64}, Σ::AbstractArray, σ::Float64, currentDepth::Int, scope::Vector{Int}, idCounter::Int)

	idCounter += 1
	P = ProductNode(idCounter)

	dist = MvNormal(μ[scope], Σ[scope,scope])
	means = rand(dist, 1)

	if currentDepth >= depth

		for (i, mean) in enumerate(means)

			idCounter += 1
			node = NormalDistributionNode(idCounter, scope[i], μ = mean, σ = σ)
			add!(P, node)

		end

	else

		@assert length(scope) >= 2

		s = convert(Array{Bool}, rand(Bernoulli(), length(scope)))

		while (sum(s) == length(scope)) | (sum(s) == 0)
			s = convert(Array{Bool}, rand(Bernoulli(), length(scope)))
		end

		if sum(s) >= 2
			(node1, idCounter) = generateRandomSum(sumWidth, depth, μ, Σ, σ, currentDepth + 1, scope[s], idCounter)
			add!(P, node1)
		else
			idCounter += 1
			node1 = NormalDistributionNode(idCounter, scope[s][1], μ = means[s][1], σ = σ)
			add!(P, node1)
		end

		if sum(!s) >= 2
			(node2, idCounter) = generateRandomSum(sumWidth, depth, μ, Σ, σ, currentDepth + 1, scope[!s], idCounter)
			add!(P, node2)
		else
			idCounter += 1
			node2 = NormalDistributionNode(idCounter, scope[!s][1], μ = means[!s][1], σ = σ)
			add!(P, node2)
		end
	end

	return (P, idCounter)

end

function generateRandomSum(sumWidth::Int, depth::Int, μ::Vector{Float64}, Σ::AbstractArray, σ::Float64, currentDepth::Int, scope::Vector{Int}, idCounter::Int)

	idCounter += 1
	S = SumNode(idCounter)

	for child in 1:sumWidth
		(node, idCounter) = generateRandomProduct(sumWidth, depth, μ, Σ, σ, currentDepth, scope, idCounter)
		add!(S, node)
	end

	return (S, idCounter)

end

function generateRandomStructure(X::AbstractArray, sumWidth::Int, depth::Int, σ::Float64, idCounter::Int)

	(D, N) = size(X)

	μ = vec(mean(X, 2))
	Σ = cov(X, vardim = 2)

	(S, idCounter) = generateRandomSum(sumWidth, depth, μ, Σ, σ, 1, collect(1:D), idCounter)

	return (S, idCounter)

end

function randomStructure(X::AbstractArray, Classes::Vector{Int}, sumWidth::Int, depth::Int; σ = 0.25)

	idCounter = 1

	S = SumNode(idCounter)

	for cclass in Classes

		idCounter += 1
		C = ProductNode(idCounter)
		push!(C.classes, ClassNode(cclass))

		(child, idCounter) = generateRandomStructure(X, sumWidth, depth, σ, idCounter)
		add!(C, child)
		add!(S, C, 1.0/length(Classes))
	end

	normalize!(S)

	return S

end
