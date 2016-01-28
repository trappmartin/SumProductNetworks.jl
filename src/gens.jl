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
