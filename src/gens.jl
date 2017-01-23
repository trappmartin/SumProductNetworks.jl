export learnSPN

function learnSumNode{T <: AbstractFloat}(X::AbstractArray{T}; iterations = 1000, minN = 10)

	(N, D) = size(X)

	if N < minN
		return ([1.0], ones(Int, N))
	end

	μ0 = vec(mean(X, 1))
	κ0 = 5.0
	ν0 = 9.0
	Σ0 = cov(X) + (10 * eye(D))

	models = train(init(X, DPM(GaussianDiagonal{NormalNormal}(NormalNormal[NormalNormal(μ0 = mean(X[:,d])) for d in 1:D])), KMeansInitialisation(k = round(Int, log(N)))), DPMHyperparam(), Gibbs(maxiter = iterations))

	ass = models[end].assignments

	# number of child nodes
	uidx = unique(ass)

	idx = Int[findfirst(uidx .== i) for i in ass]

	# compute cluster weights
	w = Float64[sum(idx .== i) / convert(Float64, N) for i in sort(uidx)]

	return (w, idx)
end

function learnSumNode{T <: AbstractFloat}(X::AbstractVector{T}; iterations = 1000, minN = 10)

	N = length(X)

	if N < minN
		return ([1.0], ones(Int, N))
	end

	μ0 = mean(X)

	# this is a quick hack!
	XX = reshape(X, N, 1)
	models = train(init(XX, DPM(NormalNormal(μ0 = μ0)), KMeansInitialisation(k = round(Int, log(N)))), DPMHyperparam(), Gibbs(maxiter = iterations))

	ass = models[end].assignments

	# number of child nodes
	uidx = unique(ass)

	idx = Int[findfirst(uidx .== i) for i in ass]

	# compute cluster weights
	w = Float64[sum(idx .== i) / convert(Float64, N) for i in sort(uidx)]

	return (w, idx)
end

function learnSumNode(X::AbstractArray{Int}; iterations = 1000, minN = 10)

	(N, D) = size(X)

	if N < minN
		return ([1.0], ones(Int, N))
	end

	idClusterId = runNaiveBayes(X')
	ass = [idClusterId[i] for i in 1:N]

	# number of child nodes
	uidx = unique(ass)
	idx = Int[findfirst(uidx .== i) for i in ass]

	# compute cluster weights
	w = Float64[sum(idx .== i) / convert(Float64, N) for i in sort(uidx)]

	return (w, idx)
end

function isNotIndependent(x::Vector{Int}, y::Vector{Int}; pvalue = 0.05)
	(p, logP) = BMITest.test(vec(x), vec(y), α = 1)
	value = 1-p
	return value > 0.5
end

function isNotIndependent{T <: AbstractFloat}(x::Vector{T}, y::Vector{T}; pvalue = 0.05)
	(value, threshold) = gammaHSIC(x', y', α = pvalue, kernelSize = 0.5)
	return value > threshold
end

function learnProductNode(X::AbstractArray; pvalue = 0.05, minN = 10)

	(N, D) = size(X)

	if N < minN
		return collect(1:D)
	end

	# create set of variables
	varset = collect(1:D)

	indset = Vector{Int}(0)
	toprocess = Vector{Int}(0)
	toremove = Vector{Int}(0)

	d = rand(1:D)

	push!(indset, d)
	push!(toprocess, d)
	deleteat!(varset, findfirst(varset .== d))

	x = zeros(D)
	y = zeros(D)

	while length(toprocess) > 0

		d = pop!(toprocess)

		for d2 in varset

			value = 0
			threshold = 1

			if issparse(X)
				x = full(X[:,d])
				y = full(X[:,d2])
			else
				x = X[:,d]
				y = X[:,d2]
			end

			if isNotIndependent(vec(x), vec(y), pvalue = pvalue)
				# values are not independent
				push!(indset, d2)
				push!(toprocess, d2)
				push!(toremove, d2)
			end

		end

		while length(toremove) > 0
			deleteat!(varset, findfirst(varset .== pop!(toremove)))
		end
	end

	return indset
end

function fitLeafDistribution{T <: AbstractFloat}(X::AbstractArray{T}, id::Int, scope::Int, obs::Vector{Int})
	sigma = std(X[obs, scope])
	sigma = isnan(sigma) ? 1e-6 : sigma + 1e-6
	mu = mean(X[obs, scope])

	@assert !isnan(sigma)
	@assert !isnan(mu)
	return NormalDistributionNode(id, scope, μ = mu, σ = sigma)
end

#TODO: Change such that sum of indicators is returned and node id's are incremented!
function fitLeafDistribution(X::AbstractArray{Int}, id::Int, scope::Int, obs::Vector{Int})

	K = maximum(X[:, scope])
	p = Float64[sum(X[obs, scope] .== k) / length(obs) for k in 1:K]

	@assert all(!isnan(p))
	return UnivariateNode{Categorical}(id, Categorical(p), scope)
end

function learnSPN(X::AbstractArray; minSamples = 10, maxiter = 500, maxDepth = Inf)

	(N, D) = size(X)

	observations = Vector{Int}[]
	dimensions = Vector{Int}[]

	# temp SPN structure
	nodeDepths = [0]
	modes = [:sum]
	ids = [1]
	usedids = []
	cids = Dict{Int, Vector}()
	weights = Dict{Int, Vector}()
	scopes = Dict{Int, Vector}()

	nodes = SPNNode[]

	# push data
	push!(observations, collect(1:N))
	push!(dimensions, collect(1:D))

	while !isempty(observations)

		nodeDepth = pop!(nodeDepths)
		mode = pop!(modes)
		id = pop!(ids)
		obs = sort(pop!(observations))
		dims = sort(pop!(dimensions))

		push!(usedids, id)

		isuniv = length(dims) == 1

		if mode == :sum

			# if depth has been reached, push back
			if nodeDepth >= maxDepth

				cid = Int[]
				w = [1.0]

				ccid = maximum(usedids)
				push!(cid, ccid + 1)
				push!(ids, ccid + 1)
				push!(observations, obs)
				push!(dimensions, dims)
				push!(modes, :product)
				push!(nodeDepths, nodeDepth + 1)
			else

				if isuniv
					(w, assignments) = learnSumNode(X[obs, dims[1]], minN = minSamples, iterations = maxiter)
				else
					(w, assignments) = learnSumNode(X[obs, dims], minN = minSamples, iterations = maxiter)
				end
				numchildren = length(w)

				cid = Int[]
				for c in 1:numchildren
					ccid = maximum(usedids)
					push!(cid, ccid + c)
					push!(ids, ccid + c)
					push!(observations, obs[find(assignments .== c)])
					push!(dimensions, dims)
					push!(modes, :product)
					push!(nodeDepths, nodeDepth + 1)
				end

			end

			weights[id] = w
			cids[id] = cid
			scopes[id] = dims

		elseif mode == :product

			# if univariate, then push back as Leaf
			if isuniv
				push!(ids, id)
				push!(observations, obs)
				push!(dimensions, dims)
				push!(modes, :leaf)
				push!(nodeDepths, nodeDepth + 1)
				continue
			end

			# if depth has been reached, push back
			if nodeDepth >= maxDepth
				cid = Int[]
				ccid = maximum(usedids)
				for (c, d) in enumerate(dims)
					push!(cid, ccid + c)
					push!(ids, ccid + c)
					push!(observations, obs)
					push!(dimensions, [d])
					push!(modes, :leaf)
					push!(nodeDepths, nodeDepth + 1)
				end

				cids[id] = cid

			else

				assignments = learnProductNode(X[obs, dims], minN = minSamples)

				p0 = dims[assignments]
				p1 = setdiff(dims, p0)

				if isempty(p1)

					cid = Int[]
					ccid = maximum(usedids)
					for (c, d) in enumerate(p0)
						push!(cid, ccid + c)
						push!(ids, ccid + c)
						push!(observations, obs)
						push!(dimensions, [d])
						push!(modes, :leaf)
						push!(nodeDepths, nodeDepth + 1)
					end

					cids[id] = cid
				else

					cid = Int[]
					ccid = maximum(usedids)
					push!(cid, ccid + 1)
					push!(ids, ccid + 1)
					push!(observations, obs)
					push!(dimensions, p0)
					push!(modes, :sum)
					push!(nodeDepths, nodeDepth + 1)

					push!(cid, ccid + 2)
					push!(ids, ccid + 2)
					push!(observations, obs)
					push!(dimensions, p1)
					push!(modes, :sum)
					push!(nodeDepths, nodeDepth + 1)

					cids[id] = cid
				end
			end

			scopes[id] = dims

		elseif mode == :leaf
			node = fitLeafDistribution(X, id, dims[1], obs)
			push!(nodes, node)
		else
			throw(ErrorException("Unknown mode: $mode"))
		end
	end

	for k in keys(cids)
		for cid in cids[k]
			@assert cid in usedids
		end
	end

	# construct SPN
	while !isempty(cids)
		for id in sort(collect(keys(cids)), rev = true)

			# check if all chidren exist
			ncids = Int[n.id for n in nodes]
			if all(Bool[ccid in ncids for ccid in cids[id]])

				if haskey(weights, id) # sum
					S = SumNode(id, scope = scopes[id])
					w = weights[id]
					for (i, ccid) in enumerate(cids[id])
						add!(S, nodes[findfirst(ncids .== ccid)], w[i])
					end
					push!(nodes, S)
					delete!(cids, id)
					delete!(weights, id)

				else # product
					P = ProductNode(id, scope = scopes[id])

					for (i, ccid) in enumerate(cids[id])
						add!(P, nodes[findfirst(ncids .== ccid)])
					end
					push!(nodes, P)
					delete!(cids, id)
					delete!(weights, id)
				end
			end

		end
	end

	ncids = Int[n.id for n in nodes]
	return nodes[findfirst(ncids .== 1)]
end
