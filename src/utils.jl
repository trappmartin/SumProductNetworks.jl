function generate_bloobs(n_features = 2, n_centers = 3, n_samples = 100)
    centers = @parallel (hcat) for i = 1:n_features
        rand(Uniform(-10, 10), n_centers)
    end

    cluster_std = @parallel (hcat) for i = 1:n_features
      rand(Uniform(0.5, 4), n_centers)
    end

    n_samples_per_center = ones(Integer, n_centers) * int(n_samples / n_centers)

    for i = 1:(n_centers % n_samples)
        n_samples_per_center[i] += 1
    end

    X = @parallel (hcat) for i = 1:n_centers
      rand(MvNormal(ones(n_features) .* centers[i], eye(n_features) .* cluster_std[i]), n_samples_per_center[i])
    end

    X = X';

    Y = @parallel (vcat) for i = 1:n_centers
      y = ones(n_samples_per_center[i]) * i
    end

    ids = [1:size(X)[1]]
    shuffle!(ids)

    X = X[ids,:]
    Y = Y[ids];

    return (X, Y)
end

@doc doc"""
The number of combinations of N things taken 2 at a time.
This is often expressed as "N choose 2".



""" ->
function comb2(N::Int)

    if 2 > N
        return 0
    end

    val = 1.0

    for j in 1:minimum([2, N-2])
        val = (val*(N-j+1)) / j
    end

    return val
end

@doc doc"""
The Rand Index computes a similarity measure between two clusterings
by considering all pairs of samples and counting pairs that are
assigned in the same or different clusters in the predicted and
true clusterings.

The adjusted Rand index is thus ensured to have a value close to
0.0 for random labeling independently of the number of clusters and
samples and exactly 1.0 when the clusterings are identical (up to
a permutation).

""" ->
function adjustedRandIndex(c1::Vector{Int}, c2::Vector{Int})

    N = size(c1, 1)
	M = hcat(c1, c2)

	(e1, e2, contingency) = hist2d(M)

	# Compute the ARI using the contingency data
	combC = sum([comb2(n) for n in sum(contingency, 1)])
    combK = sum([comb2(n) for n in sum(contingency, 2)])

    sumComb = 0.0

    for n in contingency
        for i in n
            sumComb += comb2(i)
        end
    end

    prodComb = (combC * combK) / comb2(N)
    meanComb = (combC + combK) / 2

    return ((sumComb - prodComb) / (meanComb - prodComb))
end
