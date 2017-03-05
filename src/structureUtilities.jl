export learnSPNStructure, transformToLogisticLeaves

function learnSPNStructure(X, Y, maxiterations, maxdepth, randomseed; method = :gens, removedegenerated = false, maxChildren = 4, minChildren = 2, allowScopeOverlap = false)
	srand(randomseed)

	if method == :gens
		SStructure = learnSPN(X, maxiter = maxiterations, maxDepth = maxdepth)
	elseif method == :random
		SStructure = randomSPN(X, maxDepth = maxdepth, allowScopeOverlap = allowScopeOverlap)
	else
		error("Unknown structure learning method: ", method)
	end

	@assert all(isfinite(llh(SStructure, X)))

	nodeidincrement = 1
	Sinit = SumNode(nodeidincrement)
	for label in unique(Y)
	  nodeidincrement += 1
	  P = ProductNode(nodeidincrement)
	  add!(P, copySPN(SStructure, idIncrement = nodeidincrement))
	  nodeidincrement = maximum([node.id for node in SumProductNetworks.order(P)]) + 1
	  add!(P, IndicatorNode(nodeidincrement, label, size(X, 2) + 1))
	  add!(Sinit, P, sum(Y.==label) / length(Y))
	end

	if removedegenerated
		dists = pairwise(Euclidean(), X')
		dists .+= (eye(size(X, 1)) * (maximum(dists) + 10))
		pi = findfirst([percentile(vec(minimum(dists, 1)), i) for i in 1:50] .> 0)
		prune!(Sinit, percentile(vec(minimum(dists, 1)), pi))
	end

	SumProductNetworks.normalize!(Sinit)
	simplify!(Sinit)
	SumProductNetworks.normalize!(Sinit)

	return Sinit
end

function transformToLogisticLeaves(SPN::SumNode; filterInitialisation = :zeros)

	# transform product nodes and leaves of product nodes
	for node in filter(n -> isa(n, ProductNode), SumProductNetworks.order(SPN))
	  if all(n -> isa(n, NormalDistributionNode), children(node))
	    fnode = MultivariateFeatureNode(node.id, node.scope)

			if filterInitialisation == :random
	    	k = rand()+0.2
	    	wstd = k/sqrt(length(fnode.weights))
	    	fnode.weights[:] = [rand(Normal(0., wstd)) for i in 1:length(fnode.weights)]
			else
				fnode.weights[:] = zeros(size(fnode.weights))
			end

	    # we assume a tree structure
	    parent = parents(node)[1]
	    cid = findfirst(children(parent) .== node)
	    weight = parent.weights[cid]
	    remove!(parent, cid)
	    add!(parent, fnode, weight)
	  elseif any(n -> isa(n, NormalDistributionNode), children(node))

	    ids = find(n -> isa(n, NormalDistributionNode), children(node))
	    scopes = [n.scope for n in children(node)[ids]]

	    fnode = MultivariateFeatureNode(children(node)[ids[1]].id, scopes)

			if filterInitialisation == :random
		    k = rand()+0.2
		    wstd = k/sqrt(length(fnode.weights))
		    fnode.weights[:] = [rand(Normal(0., wstd)) for i in 1:length(fnode.weights)]
			else
				fnode.weights[:] = zeros(size(fnode.weights))
			end

	    for id in ids
	      remove!(node, id)
	    end
	    add!(node, fnode)
	  end
	end

	# transform sum nodes
	for node in filter(n -> isa(n, SumNode), SumProductNetworks.order(SPN))
	  if all(n -> isa(n, NormalDistributionNode), children(node))
	    fnode = MultivariateFeatureNode(node.id, node.scope)

			if filterInitialisation == :random
		    k = rand()+0.2
		    wstd = k/sqrt(length(fnode.weights))
		    fnode.weights[:] = [rand(Normal(0., wstd)) for i in 1:length(fnode.weights)]
			else
				fnode.weights[:] = zeros(size(fnode.weights))
			end

	    # we assume a tree structure
	    parent = parents(node)[1]
	    cid = findfirst(children(parent) .== node)
	    remove!(parent, cid)
	    add!(parent, fnode)
	  end
	end
end
