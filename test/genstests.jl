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

# he should find at least two clusters.. 4 is better though :D
@test length(w) > 1

println(" * learn single product node using HSIC..")

# learn product nodes
Dhat = SPN.learnProductNode(X)

# there should be no independence as it is a MultivariateNormal
@test 1 in Dhat
@test 2 in Dhat

println(" * learn SPN using learnSPN..")

(D, N) = size(X)

# initialisation stuff
dimMapping = Dict{Int, Int}([convert(Int, d) => convert(Int, d) for d in 1:D])
obsMapping = Dict{Int, Int}([convert(Int, n) => convert(Int, n) for n in 1:N])
assignments = Assignment()

# learn SPN using Gens Approach
root = SPN.learnSPN(X, dimMapping, obsMapping, assignments)

dists = Vector{MvNormal}(length(root.children))

for (ci, child) in enumerate(root.children)
	m = mean(X[:,assignments(child)], 2)
	c = cov(X[:,assignments(child)]')
	dists[ci] = MvNormal(vec(m), c)
end

llhv = 0.0

for i in 1:N
	llhvi = 0
	for j in 1:length(root.children)
		llhvi += root.weights[j] * (pdf(dists[j], vec(X[:,i])) / pdf(dists[j], mean(dists[j])))
	end
	llhv += log(llhvi)
end

println(llhv / N)

println(mean([llh(root, X[:,i])[1] for i in 1:N]))

println("fixing SPN")
SPN.fixSPN!(root)

println(mean([llh(root, X[:,i])[1] for i in 1:N]))
