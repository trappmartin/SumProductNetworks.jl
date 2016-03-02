println(" * test UnivariateNode...")

N = 100
X = randn(N)
root = SumNode()
add!(root, UnivariateNode{Normal}(fit(Normal, X), 1))
normalize!(root);

@test length(root.children) == 1
@test root.weights[1] == 1.0
@test SPN.order(root)[end] == root
@test llh(root, [0.0])[1] > llh(root, [1.0])[1]

println(" * test Multivariate...")

N = 100
D = 2
X = randn(D, N)
root = SumNode()
add!(root, MultivariateNode{MvNormal}(fit(MvNormal, X), collect(1:2)))
normalize!(root);

@test length(root.children) == 1
@test root.weights[1] == 1.0
@test SPN.order(root)[end] == root
@test llh(root, zeros(D, 1))[1] > llh(root, ones(D, 1))[1]
llh(root, zeros(D, 2))

println(" * test Multivariate with conjugate prior")

μ0 = vec( mean(X, 2) )
κ0 = 1.0
ν0 = convert(Float64, D)
Ψ = eye(D) * 10

G0 = GaussianWishart(μ0, κ0, ν0, Ψ);

root = SumNode()
add!(root, MultivariateNode{ConjugatePostDistribution}(BNP.add_data(G0, X), collect(1:D)))
normalize!(root);

@test llh(root, zeros(D, 1))[1] > llh(root, ones(D, 1))[1]
