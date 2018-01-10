export learnSPN

function learnSumNode(X::Vector; iterations = 100, minN = 10, k = 2, method = :GMM, α = 1.0)

	N = length(X)

	if N < minN
		return ([1.0], ones(Int, N))
	end

	R = kmeans(Float64.(reshape(X, 1, N)), k; maxiter=iterations)
	ass = assignments(R)

	# number of child nodes
	uidx = unique(ass)

	idx = Int[findfirst(uidx .== i) for i in ass]

	# compute cluster weights
	w = Float64[sum(idx .== i) + α for i in sort(uidx)]

	return (w / sum(w), idx)
end


function learnSumNode(X::Matrix; iterations = 100, minN = 10, k = 2, method = :MAPDP, α = 1.0)

	(N, D) = size(X)

	if N < minN
		return ([1.0], ones(Int, N))
	end

	ass = if method == :GMM
		R = kmeans(Float64.(X)', k; maxiter=iterations)
		assignments(R)
    elseif method == :MAPDP

        if isa(X, Matrix{Int})
            N0 = α
            alpha = ones(D) / 2.
            mapDP_Cat(X, N0, alpha, maxIter = iterations)
        else
            N0 = 0.1
            m0 = mean(X, 1)
            a0 = 5.
            c0 = 9. / N
            B0 = eye(D) .* (1./0.05*var(X, 1))

            mapDP_NW(X, N0, m0, a0, c0, B0, maxIter = iterations)
        end
	elseif method == :DPM

		μ0 = vec(mean(X, 1))
		κ0 = 5.0
		ν0 = 9.0
		Σ0 = cov(X) + (10 * eye(D))

		models = train(init(X, DPM(GaussianDiagonal{NormalNormal}(NormalNormal[NormalNormal(μ0 = mean(X[:,d])) for d in 1:D])), KMeansInitialisation(k = round(Int, log(N)))), DPMHyperparam(), SliceSampler(maxiter = iterations))

		models[end].assignments
	elseif method == :NaiveBayes
		idClusterId = runNaiveBayes(X')
		[idClusterId[i] for i in 1:N]
	else
		warn("Unknown method for clustering, returning one group!")
		ones(N)
	end

	# number of child nodes
	uidx = unique(ass)

	idx = Int[findfirst(uidx .== i) for i in ass]

	# compute cluster weights
	w = Float64[sum(idx .== i) + α for i in sort(uidx)]

	return (w / sum(w), idx)
end

function learnProductNode(X::AbstractArray; pvalue = 0.05, minN = 10, method = :GTest, η = 0.5, gfactor = 5.0)
	(N, D) = size(X)

	if N < minN
		return (collect(1:D), Int[])
	end

	(scope1, scope2) = if method == :EBVS
		ebvs(X, 1:D, η = η)
	elseif method == :GTest
		greedySplit(X, 1:D, factor = gfactor)
	else
		warn("Unknown method for variable selection")
		(collect(1:D), Int[])
	end

	return (scope1, scope2)
end

function fitLeafDistribution{T <: AbstractFloat}(X::AbstractArray{T}, id::Int, scope::Int, obs::Vector{Int})
	sigma = std(X[obs, scope])
	sigma = isnan(sigma) ? 1e-6 : sigma + 1e-6
	mu = mean(X[obs, scope])

	@assert !isnan(sigma) "Estimated variance is NaN, check if your data contains NaNs!"
	@assert !isnan(mu) "Estimated mean is NaN, check if your data contains NaNs!"
	return [NormalDistributionNode(id, scope, μ = mu, σ = sigma)]
end

function fitLeafDistribution(X::AbstractArray{Int}, id::Int, scope::Int, obs::Vector{Int}, ϵ = 0.1)
	K = unique(X[:,scope])
    N = length(obs)
    p = Dict(k => Float32((sum(X[obs, scope] .== k) + ϵ) / N) for k in K)

	iid = id

	S = FiniteSumNode{Float32}(iid, Int[scope])
	L = SPNNode[]
	for k in K
		iid += 1
		push!(L, IndicatorNode{Int}(iid, k, scope))
		add!(S, L[end], log(p[k]))
	end

	S.logweights -= logsumexp(S.logweights)

	return vcat(L, [S])
end

function learnSPN(X::AbstractArray; minSamples = 10, 
				  maxiter = 100, 
				  k = 2,
				  clusteringMethod = :GMM,
				  varsplitMethod = :GTest,
				  gfactor = 5.0,
                  ϵ = 0.1,
                  α = 1.0,
				  maxDepth = Inf)

	(N, D) = size(X)

	observations = Vector{Int}[]
	dimensions = Vector{Int}[]

	# temp SPN structure
	nodeDepths = [0]
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
		id = pop!(ids)
		obs = sort(pop!(observations))
		dims = sort(pop!(dimensions))

		push!(usedids, id)

		isuniv = length(dims) == 1
		nodeConstructed = false

		# if univariate, then push back as Leaf
		if !isuniv
			(p0, p1) = learnProductNode(X[obs, dims], minN = minSamples, method = varsplitMethod, gfactor = gfactor)

			if (!isempty(p0) & !isempty(p1))
				scope1 = dims[p0]
				scope2 = dims[p1]

				cid = Int[]
				ccid = maximum(union(usedids, ids))

				push!(cid, ccid + 1)
				push!(ids, ccid + 1)
				push!(observations, obs)
				push!(dimensions, scope1)
				push!(nodeDepths, nodeDepth + 1)
				push!(cid, ccid + 2)
				push!(ids, ccid + 2)
				push!(observations, obs)
				push!(dimensions, scope2)
				push!(nodeDepths, nodeDepth + 1)

				cids[id] = cid

				scopes[id] = dims
				nodeConstructed = true
			end
		end

		if !nodeConstructed

			if isuniv
				(w, assignments) = learnSumNode(X[obs, dims[1]], minN = minSamples, iterations = maxiter, method = clusteringMethod, k = k, α = α)
			else
				(w, assignments) = learnSumNode(X[obs, dims], minN = minSamples, iterations = maxiter, method = clusteringMethod, k = k, α = α)
			end

			numchildren = length(w)

			if (numchildren == 1) | (nodeDepth > maxDepth)
				cid = Int[]
				for (c, d) in enumerate(dims)
					ccid = maximum(union(usedids, ids)) + 1
					leafnodes = fitLeafDistribution(X, ccid, d, obs)

					push!(cid, ccid)

					for node in leafnodes
						push!(nodes, node)
						push!(usedids, node.id)
					end
				end

				cids[id] = cid

			else

				cid = Int[]
				for c in 1:numchildren
					ccid = maximum(union(ids, usedids)) + 1
					push!(cid, ccid)
					push!(ids, ccid)
					push!(observations, obs[find(assignments .== c)])
					push!(dimensions, dims)
					push!(nodeDepths, nodeDepth + 1)
				end

				cids[id] = cid
				weights[id] = log.(w)
			end
			scopes[id] = dims

		end
	end

	# construct SPN
	while !isempty(cids)
		for id in sort(collect(keys(cids)), rev = true)

			# check if all chidren exist
			ncids = Int[n.id for n in nodes]
			if all(Bool[ccid in ncids for ccid in cids[id]])

				if haskey(weights, id) # sum
					S = FiniteSumNode{Float32}(id, scopes[id])
					w = weights[id]
					for (i, ccid) in enumerate(cids[id])
						add!(S, nodes[findfirst(ncids .== ccid)], w[i])
					end
					push!(nodes, S)
					delete!(cids, id)
					delete!(weights, id)

				else # product
					P = FiniteProductNode(id, scopes[id])

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
	root = nodes[findfirst(ncids .== 1)]

    # simplify the network
    simplify!(root)

    return root
end
