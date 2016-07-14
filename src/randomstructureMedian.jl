function generateRandomProduct(X::AbstractArray, sumWidth::Int, depth::Int, σ::Float64, currentDepth::Int, scope::Vector{Int}, idCounter::Int)

	idCounter += 1
	P = ProductNode(idCounter)

	(D, N) = size(X)

	if currentDepth >= depth

		for s in scope

			id = rand(1:N)

			idCounter += 1
			node = NormalDistributionNode(idCounter, s, μ = X[s, id], σ = σ)
			add!(P, node)

		end

	else

		@assert length(scope) >= 2

		s = convert(Array{Bool}, rand(Bernoulli(), length(scope)))

		while (sum(s) == length(scope)) | (sum(s) == 0)
			s = convert(Array{Bool}, rand(Bernoulli(), length(scope)))
		end

		if sum(s) >= 2
			(node1, idCounter) = generateRandomSum(X, sumWidth, depth, σ, currentDepth + 1, scope[s], idCounter)
			add!(P, node1)
		else
			idCounter += 1
			node1 = NormalDistributionNode(idCounter, scope[s][1], μ = X[scope[s][1], rand(1:N)], σ = σ)
			add!(P, node1)
		end

		if sum(!s) >= 2
			(node2, idCounter) = generateRandomSum(X, sumWidth, depth, σ, currentDepth + 1, scope[!s], idCounter)
			add!(P, node2)
		else
			idCounter += 1
			node2 = NormalDistributionNode(idCounter, scope[!s][1], μ = X[scope[!s][1], rand(1:N)], σ = σ)
			add!(P, node2)
		end
	end

	return (P, idCounter)

end

function generateRandomSum(X::AbstractArray, sumWidth::Int, depth::Int, σ::Float64, currentDepth::Int, scope::Vector{Int}, idCounter::Int)

	idCounter += 1
	S = SumNode(idCounter)

	for child in 1:sumWidth
		(node, idCounter) = generateRandomProduct(X::AbstractArray, sumWidth, depth, σ, currentDepth, scope, idCounter)
		add!(S, node)
	end

	return (S, idCounter)

end

function generateRandomStructure(X::AbstractArray, sumWidth::Int, depth::Int, σ::Float64, idCounter::Int)

	(D, N) = size(X)

	(S, idCounter) = generateRandomSum(X, sumWidth, depth, σ, 1, collect(1:D), idCounter)

	return (S, idCounter)

end

function randomStructureMedian(X::AbstractArray, Y::Vector{Int}, sumWidth::Int, depth::Int; σ = 0.25)

	idCounter = 1

	S = SumNode(idCounter)

	K = unique(Y)

	for yi in K

		idCounter += 1
		C = ProductNode(idCounter)
		push!(C.classes, ClassNode(yi))

		(child, idCounter) = generateRandomStructure(X[:,Y .== yi], sumWidth, depth, σ, idCounter)
		add!(C, child)
		add!(S, C, 1.0/length(K))
	end

	normalize!(S)

	return S

end
