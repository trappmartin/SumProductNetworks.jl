function generateRandomProduct(X::AbstractArray, sumWidth::Int, productWidth::Int)

end

function generateRandomSum(X::AbstractArray, sumWidth::Int, productWidth::Int)

	S = SumNode()

	(D, N) = size(X)
	idx = collect(1:N)

	shuffle!(idx)

	P = N ÷ sumWidth

	@assert P >= 1

	s = 1
	e = P
	for child in 1:sumWidth
		node = generateRandomProduct(X[:,idx[s:e]], sumWidth, productWidth)
		add!(S, node)

		s += P
		e = maximum([e + P, N])
	end

	return S

end

function generateRandomStructure(X::AbstractArray, sumWidth::Int, productWidth::Int, σ::Float64)

	(D, N) = size(X)

	μ = vec(mean(X, 2))
	Σ = cov(X, 2)

	d = MvNormal(μ, Σ)

	S = generateRandomSum()


end

function randomStructure(X::AbstractArray, Classes::Vector{Int}, sumWidth::Int, productWidth::Int; σ = 1.0)

	S = SumNode()

	for cclass in Classes

		C = ProductNode()
		push!(C.classes, ClassNode(cclass))

		children = generateRandomStructure(X, sumWidth, productWidth, σ)
		for child in children
			add!(C, child)
		end

		add!(S, C, 1.0/length(Classes))
	end

	return S

end
