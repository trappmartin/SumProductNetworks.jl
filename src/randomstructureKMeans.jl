export learnSPNKMeans

function learnSumNodeDPMM(X; iterations = 100, minN = 10, k = 2)

	(N, D) = size(X)

	if N < minN
		return (Dict(1 => 1.0), ones(Int, N))
	end

	G0 = BNP.fit(GaussianWishart, X')
	models = train(DPM(G0), Gibbs(burnin = 1, maxiter = iterations, thinout = 2), KMeansInitialisation(k = k), X')

	idx = models[end].assignments

	# number of child nodes
	uidx = unique(idx)

	# compute cluster weights
	w = Dict([i => sum(idx .== i) / convert(Float64, N) for i in uidx])

	return (w, idx)

end

function learnSumNodeKMeans(X; iterations = 1000, minN = 10, k = 2)

	(N, D) = size(X)

	if N < minN
		return (Dict(1 => 1.0), ones(Int, N))
	end

	R = Clustering.kmeans(X', k; maxiter = iterations)
	idx = assignments(R)

	# number of child nodes
	uidx = unique(idx)

	# compute cluster weights
	w = Dict([i => sum(idx .== i) / convert(Float64, N) for i in uidx])

	return (w, idx)

end

function learnProductNodeKMeans(X::AbstractArray; method = :Random, pvalue = 0.05, minN = 10)

	(N, D) = size(X)

	if N < minN
		return collect(1:D)
	elseif D == 1
		return [D]
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

	if method == :Random
		varset = collect(1:D)
		indset = varset[rand(Bool, D)]

		if length(indset) == 0
			indset = Int[d]
		elseif length(indset) == D
			deleteat!(indset, findfirst(indset .== d))
		end

	else

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

				if method == :HSIC
					(value, threshold) = gammaHSIC(x, y, α = pvalue, kernelSize = 0.5)
				elseif method == :BM
					(p, logP) = BMITest.test(x, y, α = 1)
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

	end

	return indset

end

function learnSPNKMeans(X, dimMapping::Dict{Int, Int}, obsMapping::Dict{Int, Int},
	lastID::Int, depth::Int;
	parents = Vector{ProductNode}(0), minSamples = 10,
	method = :Random, maxDepth = 2, minSigma = 0.2, dpmmRoot = true)

	idCounter = lastID

	# learn SPN using Gens learnSPN
	(N, D) = size(X)

	# learn sum nodes with k-means
	if (depth != 0) | !dpmmRoot
		(w, ids) = learnSumNodeKMeans(X, minN = minSamples, k = min(N, D+1))
	else
		(w, ids) = learnSumNodeDPMM(X, minN = minSamples, k = min(N, D+1))
	end

	# create sum node
	idCounter += 1
	scope = Int[dimMapping[d] for d in 1:D]
	snode = SumNode(idCounter, scope = scope)

	# check if this is supposed to be the root
	if length(parents) != 0
		for parent in parents
			add!(parent, snode)
		end
	end

	uidset = unique(ids)

	for uid in uidset

		Xhat = X[ids .== uid, :]

		idCounter += 1

		# add product node
		node = ProductNode(idCounter, scope = scope)
		add!(snode, node, convert(Float64, w[uid]))

		# compute product nodes
		Dset = Set(1:D)
		Dhat = Set(learnProductNodeKMeans(Xhat, minN = minSamples, method = method))

		Ddiff = setdiff(Dset, Dhat)
		# get list of children
		if (length(Ddiff) > 0) & (depth < maxDepth)
			# split has been found

			# don't recurse if only one dimension is inside the bucket
			# recurse

			# update mappings
			dimMappingC = Dict{Int, Int}([di => dimMapping[d] for (di, d) in enumerate(collect(Ddiff))])
			obsMappingC = Dict{Int, Int}([ni => obsMapping[n] for (ni, n) in enumerate(find(ids .== uid))])

			(idCounter, child) = learnSPNKMeans(Xhat[:, collect(Ddiff)], dimMappingC, obsMappingC, idCounter, depth + 1, parents = vec([node]),
				method = method, maxDepth = maxDepth, minSigma = minSigma)
			# recurse

			# update mappings
			dimMappingC = Dict{Int, Int}([di => dimMapping[d] for (di, d) in enumerate(collect(Dhat))])
			obsMappingC = Dict{Int, Int}([ni => obsMapping[n] for (ni, n) in enumerate(find(ids .== uid))])

			(idCounter, child) = learnSPNKMeans(Xhat[:, collect(Dhat)], dimMappingC, obsMappingC, idCounter, depth + 1, parents = vec([node]),
				method = method, maxDepth = maxDepth, minSigma = minSigma)

		else

			# construct leaf nodes
			for d in Dhat

				# argmax parameters
				μ = mean(Xhat[:, d])
				σ = std(Xhat[:, d])

				if (σ < minSigma) | isnan(σ)
					σ = minSigma
				end

				if isnan(μ)
					println("size: ", size(Xhat) )
					println("data dim: ", Xhat[:, d])
					println("all data: ", Xhat)
				end

				@assert !isnan(μ)
				@assert !isnan(σ)

				idCounter += 1

				leaf = NormalDistributionNode(idCounter, dimMapping[d], μ = μ, σ = σ, logz = normlogpdf(μ, σ, μ))
				add!(node, leaf)

			end

		end

	end

	return (idCounter, snode)

end
