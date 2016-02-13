function learnSumNode(X, G0::ConjugatePostDistribution; iterations = 50, minN = 10, pointestimate = false, α = 1.0, debug = false, method = :DPM)

	(D, N) = size(X)

	if N < minN
		return (1 => 1.0, ones(Int, N))
	end

	idx = zeros(Int, N)

	if method == :NB
		idClusterId = SPN.runNaiveBayes(X)

		idx = [idClusterId[i] for i in 1:N]

	elseif method == :DPM

		models = train(DPM(G0), Gibbs(burnin = 200, maxiter = iterations, thinout = 2), KMeansInitialisation(k = minimum([10, round(Int, sqrt(N))]) ), X)

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
	w = [i => (sum(idx .== i) / convert(Float64, N)) for i in uidx]

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
				(value, threshold) = gammaHSIC(x, y, α = pvalue)
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
	parent = Nullable{ProductNode}(), minSamples = 10, method = :HSIC, G0Type = GaussianWishart, L0Type = NormalGamma, α = 1.0, debug = false)

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
	snode = SumNode(0, scope = scope)

	# set assignments
	for n in 1:N
		set!(assignments, snode, obsMapping[n])
	end

	# check if this is supposed to be the root
	if !isnull(parent)
		p = get(parent)
		add!(p, snode)
	end

	uidset = unique(ids)

	for uid in uidset

		Xhat = X[:,ids .== uid]

		# add product node
		node = ProductNode(uid, scope = scope)
		add!(snode, node, w[uid])

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
				leaf = UnivariateNode{ConjugatePostDistribution}(BNP.add_data(L0, Xhat[d,:]), scope = dimMapping[d])
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

				learnSPN(Xhat[collect(Ddiff),:], dimMappingC, obsMappingC, assignments, parent = Nullable(node), method = method, G0Type = G0Type, L0Type = L0Type)
			end

			# don't recurse if only one dimension is inside the bucket
			if length(Dhat) == 1
				d = pop!(Dhat)

				L0 = BNP.fit(L0Type, Xhat[d,:])
				leaf = UnivariateNode{ConjugatePostDistribution}(BNP.add_data(L0, Xhat[d,:]), scope = dimMapping[d])
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

				learnSPN(Xhat[collect(Dhat),:], dimMappingC, obsMappingC, assignments, parent = Nullable(node), method = method, G0Type = G0Type, L0Type = L0Type)
			end

		else

			# construct leaf nodes
			for d in Dhat

				L0 = BNP.fit(L0Type, Xhat[d,:])
				leaf = UnivariateNode{ConjugatePostDistribution}(BNP.add_data(L0, Xhat[d,:]), scope = dimMapping[d])
				add!(node, leaf)

				# set assignments
				for n in find(ids .== uid)
					set!(assignments, leaf, obsMapping[n])
				end
			end

		end

	end

	if isnull(parent)
		return snode
	end

end
