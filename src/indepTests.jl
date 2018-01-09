using InformationMeasures, StatsBase

# Entropy based variable splitting
# See: Alternative variable splitting methods to learn SPNs by Di Mauro et al.
#
# η = entropie threshold
function ebvs(X, rvs; η = 0.5)

	(N,D) = size(X)

	g1 = Int[]
	g2 = Int[]

	H = map(d -> get_entropy(vec(X[:,d])), rvs)
	meanH = mean(H)

	println(meanH)

	for d in rvs
		if H[d] < η
			push!(g1, d)
		else
			push!(g2, d)
		end
	end

	return (g1, g2)
end

# Greedy variable splitting
# See: learnSPN by Gens et al.
#
#
function greedySplit(X, rvs; factor = 5.0)

	(N, _) = size(X)

	# create set of variables
	varset = collect(rvs)

	indset = Int[]
	toprocess = Int[]
	toremove = Int[]

	d = rand(rvs)

	push!(indset, d)
	push!(toprocess, d)
	deleteat!(varset, findfirst(varset .== d))


	# G-test, Adaptation from learnSPN code by Robert Gens.
	#
	function gStats(x, y, gfactor)

		N = length(x)

		nX = countmap(x)
		nY = countmap(y)

		gval = 0;
		for kx in keys(nX)
			for ky in keys(nY)
				cXiY = (1. * nX[kx] * nY[ky] ) / N
				cXY = sum((x .== kx) .& (y .== ky))

				if cXY > 0
					gval += 1. * cXY * log(1.0 * cXY / cXiY)
				end
			end
		end

		gval *= 2.;
		dof = (length(keys(nX)) - 1) * (length(keys(nX)) - 1)

		# If less than threshold, observed values could've been produced by noise on top of independent vars
		return gval < 2. * dof * gfactor + 0.001;
	end 

	while length(toprocess) > 0

		d = pop!(toprocess)
		x = vec(X[:,d])

		for d2 in varset

			value = 0
			threshold = 1

			y = vec(X[:,d2])

			if !gStats(x, y, factor)
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

    println(indset)

	return (indset, setdiff(rvs, indset))
end
