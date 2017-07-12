function em_run(X::AbstractArray, maxIters::Int, newClusterLLPenalized::Float64)

	clusters = Vector(0)

	(D, N) = size(X)

    idClusterId = Dict{Int, Int}()
    ll = -Inf

    it = 0
    while (it < maxIters)

        # randomize data
		idx = shuffle(collect(1:N))

		ll = 0
        minBest = -Inf

        for xi in idx
			# remove sufficient statistics from prev. assigned cluster
            if haskey(idClusterId, xi)
				cId = idClusterId[xi]

                #if cId < len(clusters): # in case the cluster has been removed
				(counts, sums) = clusters[cId]
				if sums == 1
					for i in 1:N
						if haskey(idClusterId, i)
                            if idClusterId[i] > cId
                                idClusterId[i] -= 1
							end
						end
					end
					deleteat!(clusters, cId)
				end
			end

			bestCLL = newClusterLLPenalized
            bestClusterId = -1

			for (idx, c) in enumerate(clusters)

				counts = c[1] + 0.00001
				sums = c[2]
				c = counts ./ sums
				p = c ./ sum(c)

				cll = logpdf(Multinomial(1, vec(p)), vec(full(X[:,xi])))
				if cll > bestCLL
                    bestCLL = cll
                    bestClusterId = idx
				end
			end

            if bestCLL < minBest
                minBest = bestCLL
			end

            # add instance to exiting or new cluster
            if bestClusterId == -1
                counts = X[:,xi]
				sums = 1
				push!(clusters, (counts, sums))
                idClusterId[xi] = length(clusters)
            else
                (counts, sums) = clusters[bestClusterId]
				counts += X[:,xi]
				sums += 1

				clusters[bestClusterId] = (counts, sums)
                idClusterId[xi] = bestClusterId
			end

			ll += bestCLL
		end

        it += 1
        if length(clusters) == 1
            newClusterLLPenalized *= 0.5
        end
    end

    return idClusterId
end

"""
Apply navie Bayes clustering approach proposed by Gens et al. for learnSPN().

# Arguments
* `X`: discrete data matrix (in D × N format) used for the computation.
"""
function runNaiveBayes(X::AbstractArray; clusterPenalty = 2, maxRuns = 10, maxIters = 4, ϵ = 0.00001)

	(D, N) = size(X)

    bestLL = -Inf

	newClusterLL = 0
	for d in 1:D
        newClusterLL += log((1. + ϵ) / (1. + sum(X[d,:])*ϵ))
	end
    newClusterLLPenalized = -clusterPenalty * D + newClusterLL

	result = Dict{Int, Int}()
	for em in maxRuns
		result = em_run(X, maxIters, newClusterLLPenalized)
	end

	return result
end
