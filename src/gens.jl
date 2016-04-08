function generateSumLayer(layerSizes::Vector{Int}, windowSize::Int, featureSize::Int, leafnodes::Vector{SPNNode}; currentdepth = 1)

	if currentdepth == length(layerSizes) + 1
		locNodes = Vector{SPNNode}(0)
		for loc in 1:(featureSize-windowSize)+1
			locNode = SumNode()
			locNode.isFilter = true

			# add children
			for i in loc:loc+windowSize-1
				add!(locNode, leafnodes[i], 1e-6)
			end

			push!(locNodes, locNode)
		end
		return locNodes
	end

	nodes = Vector{SPNNode}(layerSizes[currentdepth])

	for node in 1:layerSizes[currentdepth]

		layerNode = SumNode()
		children = generateSumLayer(layerSizes, windowSize, featureSize, leafnodes, currentdepth = currentdepth + 1)

		for child in children
			add!(layerNode, child)
		end

		normalize!(layerNode)

		nodes[node] = layerNode

	end

	return nodes

end

function partStructure(Classes::Vector{Int}, windowSize::Int, featureSize::Int, layerSizes::Vector{Int})

	S = SumNode()

	for cclass in Classes

		C = ProductNode()
		push!(C.classes, ClassNode(cclass))

		nodes = Vector{SPNNode}(0)
		for i in 1:featureSize
			push!(nodes, UnivariateFeatureNode(i))
		end

		children = generateSumLayer(layerSizes, windowSize, featureSize, nodes)
		for child in children
			add!(C, child)
		end

		add!(S, C, 1.0/length(Classes))
	end

	return S

end


function learnSumNode(X, G0::ConjugatePostDistribution; iterations = 100, minN = 10, pointestimate = false, α = 0.01, debug = false, method = :KMeans)

	(D, N) = size(X)

	if N < minN
		return (1 => 1.0, ones(Int, N))
	end

	idx = zeros(Int, N)

	if method == :NB
		idClusterId = SPN.runNaiveBayes(X)

		idx = [idClusterId[i] for i in 1:N]

	elseif method == :KMeans

		if debug
			println("   # [learnSPN]: starting K-Means training $(now()) - iterations: $(iterations), number of clusters: $(minimum([10, round(Int, log(N))]))")
		end

		R = Clustering.kmeans(X, minimum([10, round(Int, log(N))]); maxiter = iterations)
		idx = assignments(R)

	elseif method == :DPM

		if debug
			println("   # [learnSPN]: starting DPMM training $(now()) - burnin: 0, iterations: $(iterations), initial number of clusters: $(minimum([10, round(Int, log(N))]))")
		end

		models = train(DPM(G0), Gibbs(burnin = 100, maxiter = iterations, thinout = 2), RandomInitialisation(k = minimum([10, round(Int, log(N))]) ), X)

		if debug
			println("   # [learnSPN]: finished DPMM training $(now())")
		end

		# get assignments
		Z = reduce(hcat, map(model -> vec(model.assignments), models))

		# make sure assignments are in range
		nz = zeros(Int, N)
		for i in 1:size(Z)[2]
			uz = unique(Z[:,i])

			for (zi, z) in enumerate(uz)
				idx = find(Z[:,i] .== z)
				nz[idx] = zi
			end

			Z[:,i] = nz
		end

		if pointestimate == true

			if debug
				println("   # [learnSPN]: compute PSM $(now())")
			end

			psm = compute_psm(Z)

			if debug
				println("   # [learnSPN]: compute point estimate $(now())")
			end

			# NOTE: This is very slow as it does hclust for a N * N matrix.
			(idx, value) = point_estimate(psm, method = :comp)

		else

			p = reduce(hcat, map(model -> model.energy, models))
			p = exp(p - maximum(p))

			j = BNP.rand_indices(p)

			if debug
				println("   # [learnSPN]: take random model $(j) with $(length(unique(Z[:,j]))) cluster(s)")
			end

			idx = Z[:,j]

		end
	end

	if debug
		println("   # [learnSPN]: construct node $(now())")
	end

	# number of child nodes
	uidx = unique(idx)

	# compute cluster weights
	w = [i => sum(idx .== i) / convert(Float64, N) for i in uidx]

	return (w, idx)

end


function learnProductNode(X::AbstractArray; method = :HSIC, pvalue = 0.05, minN = 10)

	(D, N) = size(X)

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
				x = full(X[d,:])'
				y = full(X[d2,:])'
			else
				x = X[d,:]'
				y = X[d2,:]'
			end

			if method == :HSIC
				(value, threshold) = gammaHSIC(x, y, α = pvalue, kernelSize = 0.5)
			elseif method == :BM
				(p, logP) = BMITest.test(vec(x), vec(y), α = 1)
				value = 1-p
				threshold = 0.5
			end

			if value > threshold
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

function learnSPN(X, dimMapping::Dict{Int, Int}, obsMapping::Dict{Int, Int}, assignments::Assignment;
	parents = Vector{ProductNode}(0), minSamples = 10, method = :HSIC, G0Type = GaussianWishart, L0Type = NormalGamma, α = 0.01, debug = false)

	# learn SPN using Gens learnSPN
	(D, N) = size(X)

	if debug
		println("   # [learnSPN]: learn using $(N) samples with $(D) dimensions ($(now()))..")
	end

	# define G0
	G0 = BNP.fit(G0Type, X)

	# learn sum nodes
	(w, ids) = learnSumNode(X, G0, minN = minSamples, α = α, debug = debug)

	# create sum node
	scope = [dimMapping[d] for d in 1:D]
	snode = SumNode(scope = scope)

	# set assignments
	for n in 1:N
		set!(assignments, snode, obsMapping[n])
	end

	# check if this is supposed to be the root
	if length(parents) != 0
		for parent in parents
			add!(parent, snode)
		end
	end

	uidset = unique(ids)

	for uid in uidset

		Xhat = X[:,ids .== uid]

		# add product node
		node = ProductNode(scope = scope)
		add!(snode, node, convert(Float64, w[uid]))

		# set assignments
		for n in find(ids .== uid)
			set!(assignments, node, obsMapping[n])
		end

		# compute product nodes
		Dhat = Set(learnProductNode(Xhat, minN = minSamples, method = method))
		Dset = Set(1:D)

		Ddiff = setdiff(Dset, Dhat)

		# get list of children
		if length(Ddiff) > 0
			# split has been found

			# don't recurse if only one dimension is inside the bucket
			if length(Ddiff) == 1
				d = pop!(Ddiff)

				L0 = BNP.fit(L0Type, Xhat[d,:])
				leaf = UnivariateNode{ConjugatePostDistribution}(BNP.add_data(L0, Xhat[d,:]), dimMapping[d])
				add!(node, leaf)

				# set assignments
				for n in find(ids .== uid)
					set!(assignments, leaf, obsMapping[n])
				end
			else
				# recurse

				# update mappings
				dimMappingC = Dict{Int, Int}([di => dimMapping[d] for (di, d) in enumerate(collect(Ddiff))])
				obsMappingC = Dict{Int, Int}([ni => obsMapping[n] for (ni, n) in enumerate(find(ids .== uid))])

				learnSPN(Xhat[collect(Ddiff),:], dimMappingC, obsMappingC, assignments, parents = vec([node]), method = method, G0Type = G0Type, L0Type = L0Type)
			end

			# don't recurse if only one dimension is inside the bucket
			if length(Dhat) == 1
				d = pop!(Dhat)

				L0 = BNP.fit(L0Type, Xhat[d,:])
				leaf = UnivariateNode{ConjugatePostDistribution}(BNP.add_data(L0, Xhat[d,:]), dimMapping[d])
				add!(node, leaf)

				# set assignments
				for n in find(ids .== uid)
					set!(assignments, leaf, obsMapping[n])
				end
			else

				# recurse

				# update mappings
				dimMappingC = Dict{Int, Int}([di => dimMapping[d] for (di, d) in enumerate(collect(Dhat))])
				obsMappingC = Dict{Int, Int}([ni => obsMapping[n] for (ni, n) in enumerate(find(ids .== uid))])

				learnSPN(Xhat[collect(Dhat),:], dimMappingC, obsMappingC, assignments, parents = vec([node]), method = method, G0Type = G0Type, L0Type = L0Type)
			end

		else

			# construct leaf nodes
			for d in Dhat

				L0 = BNP.fit(L0Type, Xhat[d,:])
				leaf = UnivariateNode{ConjugatePostDistribution}(BNP.add_data(L0, Xhat[d,:]), dimMapping[d])
				add!(node, leaf)

				# set assignments
				for n in find(ids .== uid)
					set!(assignments, leaf, obsMapping[n])
				end
			end

		end

	end

	if length(parents) == 0
		return snode
	end

end
