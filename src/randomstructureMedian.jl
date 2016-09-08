export randomStructureMedian

function generateRandomProduct(X::AbstractArray, sumWidth::Int, depth::Int, σ::Float64, currentDepth::Int, scope::Vector{Int}, idCounter::Int)

	idCounter += 1
	P = ProductNode(idCounter)

	(N, D) = size(X)

	if currentDepth >= depth

		for s in scope

			id = rand(1:N)

			idCounter += 1

			S = SumNode(idCounter)
			for child in 1:sumWidth
				node = NormalDistributionNode(idCounter, s, μ = X[id, s], σ = σ)
				idCounter += 1
				add!(S, node)
			end

			add!(P, S)
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

			S = SumNode(idCounter)
			for child in 1:sumWidth
				node1 = NormalDistributionNode(idCounter, scope[s][1], μ = X[rand(1:N), scope[s][1]], σ = σ)
				idCounter += 1
				add!(S, node1)
			end

			add!(P, S)
		end

		if sum(!s) >= 2
			(node2, idCounter) = generateRandomSum(X, sumWidth, depth, σ, currentDepth + 1, scope[!s], idCounter)
			add!(P, node2)
		else
			idCounter += 1
			S = SumNode(idCounter)
			for child in 1:sumWidth
				node2 = NormalDistributionNode(idCounter, scope[!s][1], μ = X[rand(1:N), scope[!s][1]], σ = σ)
				idCounter += 1
				add!(S, node2)
			end

			add!(P, S)
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

	(N, D) = size(X)

	(S, idCounter) = generateRandomSum(X, sumWidth, depth, σ, 1, collect(1:D), idCounter)

	return (S, idCounter)

end

function randomStructureMedian(X::AbstractArray, sumWidth::Int, depth::Int; classdim = -1, σ = 0.25)

	(N, D) = size(X)

	idCounter = 1

	S = SumNode(idCounter)

	if classdim != -1
		K = unique(X[:,classdim])

		for yi in K

			idCounter += 1
			C = ProductNode(idCounter)

			(child, idCounter) = generateRandomStructure(X[X[:,classdim] .== yi,setdiff(1:D, classdim)], sumWidth, depth, σ, idCounter)
			add!(C, child)
			idCounter += 1
			add!(C, ClassIndicatorNode(idCounter, yi, classdim))
			add!(S, C, 1.0/length(K))
		end
	else
		(S, idCounter) = generateRandomStructure(X, sumWidth, depth, σ, idCounter)
	end

	normalize!(S)

	return S

end
