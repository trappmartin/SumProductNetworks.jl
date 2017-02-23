export randomSPN

function randomSPN(X::AbstractArray; minSamples = 10, maxDepth = Inf, minChildren = 1, maxChildren = 10, allowScopeOverlap = false)

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

        ass = rand(minChildren:maxChildren, length(obs))

        # number of child nodes
      	uidx = unique(ass)

      	assignments = Int[findfirst(uidx .== i) for i in ass]

      	# compute cluster weights
      	w = Float64[sum(assignments .== i) / convert(Float64, N) for i in sort(uidx)]
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

        if !allowScopeOverlap
				  assignments = learnProductNode(X[obs, dims], minN = minSamples)

          p0 = dims[assignments]
          p1 = setdiff(dims, p0)
          p2 = []
        else
          k = rand(minChildren:maxChildren)
          p0 = rand(dims, k)
          p1 = rand(dims, k)
          p2 = setdiff(dims, union(p0, p1))
        end

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

          if !isempty(p2)
            push!(cid, ccid + 3)
            push!(ids, ccid + 3)
            push!(observations, obs)
            push!(dimensions, p2)
            push!(modes, :sum)
            push!(nodeDepths, nodeDepth + 1)
          end

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
