function generateRandomProduct(sumWidth::Int, depth::Int, μ::Vector{Float64}, Σ::AbstractArray, σ::Float64, currentDepth::Int, scope::Vector{Int})

	P = ProductNode()

	if currentDepth >= depth

		dist = MvNormal(μ[scope], Σ[scope,scope])

		means = rand(dist, 1)
		for (i, mean) in enumerate(means)

			node = NormalDistributionNode(scope[i], μ = mean, σ = σ)
			add!(P, node)

		end

	else

		@assert length(scope) >= 2

		s = convert(Array{Bool}, rand(Bernoulli(), length(scope)))

		node1 = generateRandomSum(sumWidth, depth, μ, Σ, σ, currentDepth, scope[s])
		add!(P, node1)

		node2 = generateRandomSum(sumWidth, depth, μ, Σ, σ, currentDepth, scope[!s])
		add!(P, node2)

	end


	return P

end

function generateRandomSum(sumWidth::Int, depth::Int, μ::Vector{Float64}, Σ::AbstractArray, σ::Float64, currentDepth::Int, scope::Vector{Int})

	S = SumNode()

	for child in 1:sumWidth
		node = generateRandomProduct(sumWidth, depth, μ, Σ, σ, currentDepth, scope)
		add!(S, node)
	end

	normalize!(S)

	return S

end

function generateRandomStructure(X::AbstractArray, sumWidth::Int, depth::Int, σ::Float64)

	(D, N) = size(X)

	μ = vec(mean(X, 2))
	Σ = cov(X, vardim = 2)

	S = generateRandomSum(sumWidth, depth, μ, Σ, σ, 1, collect(1:D))

end

function randomStructure(X::AbstractArray, Classes::Vector{Int}, sumWidth::Int, depth::Int; σ = 1.0)

	S = SumNode()

	for cclass in Classes

		C = ProductNode()
		push!(C.classes, ClassNode(cclass))

		child = generateRandomStructure(X, sumWidth, depth, σ)
		add!(C, child)

		add!(S, C, 1.0/length(Classes))
	end

	return S

end
