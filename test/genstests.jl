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

(D, N) = size(X)

dimMapping = Dict{Int, Int}([convert(Int, d) => convert(Int, d) for d in 1:D])
obsMapping = Dict{Int, Int}([convert(Int, n) => convert(Int, n) for n in 1:N])
assignments = Assignment()

root = SPN.learnSPN(X, dimMapping, obsMapping, assignments)

println(" * draw SPN")

drawSPN(root, file = "learnSPN.svg")
