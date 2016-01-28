function learnSumNode{T <: Real}(X::Array{T}, G0::ConjugatePostDistribution; iterations = 100, minN = 10)

	(D, N) = size(X)

	if N < minN
		return ([1.0], collect(1:N))
	end

	models = train(DPM(G0), Gibbs(maxiter = iterations), KMeansInitialisation(k = 10), X)

	# get assignments
	Z = reduce(hcat, map(model -> vec(model.assignments), models))

	# make sure assignments are in range
	nz = zeros(N)
	for i in 1:size(Z)[2]
		uz = unique(Z[:,i])

		for (zi, z) in enumerate(uz)
			idx = find(Z[:,i] .== z)
			nz[idx] = zi
		end

		Z[:,i] = nz
	end

	# NOTE: This seems to break sometimes.. not sure why and when
	psm = compute_psm(Z)

	# NOTE: The average method of hclust seems to be buggy, but the complete works fine.
	(idx, value) = point_estimate(psm, method = :comp)

	# number of child nodes
	C = length(unique(idx))

	# compute cluster weights
	w = [sum(idx .== i) / N for i in 1:C]

	return (w, idx)

end

function learnProductNode{T <: Real}(X::Array{T}; method = :HSCI, pvalue = 0.1)

	(D, N) = size(X)

	# create set of variables
	varset = collect(1:D)

	indset = Vector{Int}(0)
	toprocess = Vector{Int}(0)
	toremove = Vector{Int}(0)

	d = rand(1:D)

	push!(indset, d)
	push!(toprocess, d)
	deleteat!(varset, findfirst(varset .== d))

	while length(toprocess) > 0

		d = pop!(toprocess)

		for d2 in varset

			(value, threshold) = gammaHSIC(X[d,:]', X[d2,:]', α = pvalue)
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

function learnSPN{T <: Real}(X::Array{T, 2}, mapping::Dict{Int, Int}; parent = Nullable{ProductNode}(), minSamples = 10)

	# learn SPN using Gens learnSPN
	(D, N) = size(X)
	dimMapping = [convert(Int, d) => mapping[d] for d in 1:D]

	# define G0
	μ0 = vec( mean(X, 2) )
	κ0 = 1.0
	ν0 = 4.0
	Ψ = eye(D) * 10
	G0 = GaussianWishart(μ0, κ0, ν0, Ψ)

	# learn sum nodes
	(w, ids) = learnSumNode(X, G0, minN = minSamples)

	# create sum node
	snode = SumNode(0)

	# check if this is supposed to be the root
	if !isnull(parent)
		p = get(parent)
		add!(p, snode)
	end

	uidset = unique(ids)

	for uid in uidset

		# add product node
		node = ProductNode(uid)
		add!(snode, node, w[uid])

		Xhat = X[:,ids .== uid]

		# compute product nodes
		Dhat = Set(learnProductNode(Xhat))
		Dset = Set(1:D)

		Ddiff = setdiff(Dset, Dhat)

		# get list of children
		if length(Ddiff) > 0
			# split has been found

			# don't recurse if only one dimension is inside the bucket
			if length(Ddiff) == 1
				d = pop!(Ddiff)
				leaf = UnivariateNode{ConjugatePostDistribution}(BNP.add_data(NormalGamma(μ = mean(Xhat[d,:])), Xhat[d,:]), scope = dimMapping[d])
				add!(node, leaf)
			else
				# recurse
				learnSPN(Xhat[collect(Ddiff),:], dimMapping, parent = Nullable(node))
			end

			# don't recurse if only one dimension is inside the bucket
			if length(Dhat) == 1
				d = pop!(Dhat)
				leaf = UnivariateNode{ConjugatePostDistribution}(BNP.add_data(NormalGamma(μ = mean(Xhat[d,:])), Xhat[d,:]), scope = dimMapping[d])
				add!(node, leaf)
			else
				# recurse
				learnSPN(Xhat[collect(Dhat),:], dimMapping, parent = Nullable(node))
			end

		else

			# construct leaf nodes
			for d in Dhat
				leaf = UnivariateNode{ConjugatePostDistribution}(BNP.add_data(NormalGamma(μ = mean(Xhat[d,:])), Xhat[d,:]), scope = dimMapping[d])
				add!(node, leaf)
			end

		end

	end

	if isnull(parent)
		return snode
	end

end
