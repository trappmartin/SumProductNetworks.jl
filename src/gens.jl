function learnSumNode{T <: Real}(X::Array{T}, G0::ConjugatePostDistribution; iterations = 100, minN = 10)

	(D, N) = size(X)

	if N < minN
		return (1 => 1.0, ones(Int, N))
	end

	models = train(DPM(G0), Gibbs(maxiter = iterations), KMeansInitialisation(k = 10), X)

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

	psm = compute_psm(Z)

	# NOTE: The average method of hclust seems to be buggy, but the complete works fine.
	(idx, value) = point_estimate(psm, method = :comp)

	# number of child nodes
	uidx = unique(idx)

	# compute cluster weights
	w = [i => (sum(idx .== i) / convert(Float64, N)) for i in uidx]

	return (w, idx)

end

function learnProductNode{T <: Real}(X::Array{T}; method = :HSIC, pvalue = 0.5, minN = 10)

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

	while length(toprocess) > 0

		d = pop!(toprocess)

		for d2 in varset

			value = 0
			threshold = 1

			if method == :HSIC
				(value, threshold) = gammaHSIC(X[d,:]', X[d2,:]', α = pvalue)
			elseif method == :BM
				(p, logP) = BMITest.test(X[d,:]', X[d2,:]')
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

function learnSPN{T <: Real}(X::Array{T, 2}, dimMapping::Dict{Int, Int}, obsMapping::Dict{Int, Int}, assignments::Assignment;
	parent = Nullable{ProductNode}(), minSamples = 10, method = :HSIC)

	# learn SPN using Gens learnSPN
	(D, N) = size(X)

	# update mappings
	dimMapping = [convert(Int, d) => dimMapping[d] for d in 1:D]
	obsMapping = [convert(Int, n) => obsMapping[n] for n in 1:N]

	# define G0
	#μ0 = vec( mean(X, 2) )
	#κ0 = 1.0
	#ν0 = 4.0
	#Ψ = eye(D) * 10
	#G0 = GaussianWishart(μ0, κ0, ν0, Ψ)
	G0 = MultinomialDirichlet(D, 1.0)

	# learn sum nodes
	(w, ids) = learnSumNode(X, G0, minN = minSamples)

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

				#leaf = UnivariateNode{ConjugatePostDistribution}(BNP.add_data(NormalGamma(μ = mean(Xhat[d,:])), Xhat[d,:]), scope = dimMapping[d])
				leaf = UnivariateNode{ConjugatePostDistribution}(BNP.add_data(BinomialBeta(D = 1), Xhat[d,:]), scope = dimMapping[d])
				add!(node, leaf)

				# set assignments
				for n in find(ids .== uid)
					set!(assignments, leaf, obsMapping[n])
				end
			else
				# recurse
				learnSPN(Xhat[collect(Ddiff),:], dimMapping, obsMapping, assignments, parent = Nullable(node), method = method)
			end

			# don't recurse if only one dimension is inside the bucket
			if length(Dhat) == 1
				d = pop!(Dhat)
				#leaf = UnivariateNode{ConjugatePostDistribution}(BNP.add_data(NormalGamma(μ = mean(Xhat[d,:])), Xhat[d,:]), scope = dimMapping[d])
				leaf = UnivariateNode{ConjugatePostDistribution}(BNP.add_data(BinomialBeta(D = 1), Xhat[d,:]), scope = dimMapping[d])
				add!(node, leaf)

				# set assignments
				for n in find(ids .== uid)
					set!(assignments, leaf, obsMapping[n])
				end
			else

				# recurse
				learnSPN(Xhat[collect(Dhat),:], dimMapping, obsMapping, assignments, parent = Nullable(node), method = method)
			end

		else

			# construct leaf nodes
			for d in Dhat

				#leaf = UnivariateNode{ConjugatePostDistribution}(BNP.add_data(NormalGamma(μ = mean(Xhat[d,:])), Xhat[d,:]), scope = dimMapping[d])
				leaf = UnivariateNode{ConjugatePostDistribution}(BNP.add_data(BinomialBeta(D = 1), Xhat[d,:]), scope = dimMapping[d])
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
