println(" * generating test data..")

# data
X = rand(MultivariateNormal([5.0, 5.0], [1.0 0.0; 0.0 2.0]), 100) # 1
X = cat(2, X, rand(MultivariateNormal([-5.0, 5.0], [0.5 -0.2; -0.2 1.0]), 100)) # 2
X = cat(2, X, rand(MultivariateNormal([-5.0, -5.0], [1.0 0.0; 0.0 0.5]), 100)) # 3
X = cat(2, X, rand(MultivariateNormal([5.0, -5.0], [1.0 0.5; 0.5 0.5]), 100)) # 4

(D, N) = size(X)

println(" * generated data with ", N, " samples and ", D, " variables.")

println(" * learn single sum node using DP-MM..")

# define G0

μ0 = vec( mean(X, 2) )
κ0 = 1.0
ν0 = 4.0
Ψ = eye(D) * 10
G0 = GaussianWishart(μ0, κ0, ν0, Ψ)

# learn sum nodes
(w, ids) = SPN.learnSumNode(X, G0)

@test length(w) == 4

println(" * learn single product node using HSIC..")

# learn product nodes
Dhat = SPN.learnProductNode(X)
@test 1 in Dhat
@test 2 in Dhat

println(" * learn SPN using learnSPN..")

# learn SPN using Gens learnSPN
(D, N) = size(X)
dimMapping = [d => d for d in 1:D]

# define G0
μ0 = vec( mean(X, 2) )
κ0 = 1.0
ν0 = 4.0
Ψ = eye(D) * 10
G0 = GaussianWishart(μ0, κ0, ν0, Ψ)

# learn sum nodes
(w, ids) = SPN.learnSumNode(X, G0)

# create sum node
root = SumNode(0)

uidset = unique(ids)

for uid in uidset

	# add product node
	node = ProductNode(uid)
	add!(root, node, w[uid])

	Xhat = X[:,ids .== uid]

	# compute product nodes
	Dhat = Set(SPN.learnProductNode(Xhat))
	Dset = Set(1:D)

	Ddiff = setdiff(Dset, Dhat)

	# get list of children
	if length(Ddiff) > 0
		# split has been found

		# recurse

	else

		# construct leaf nodes
		for d in Dhat
			leaf = UnivariateNode{ConjugatePostDistribution}(BNP.add_data(NormalGamma(μ = mean(Xhat[d,:])), Xhat[d,:]), scope = dimMapping[d])
			add!(node, leaf)
		end

	end

end

draw(root, file = "learnSPN.svg")
