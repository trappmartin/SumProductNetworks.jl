using Distributions

println(" * test UnivariateNode...")

N = 100
X = randn(N)
root = SumNode(0)
add!(root, UnivariateNode(fit(Normal, X)))
normalize!(root);

@test length(root.children) == 1
@test root.weights[1] == 1.0
@test order(root)[end] == root
@test llh(root, [0.0])[1] > llh(root, [1.0])[1]

println(" * test Multivariate...")

N = 100
D = 2
X = randn(D, N)
root = SumNode(0)
add!(root, MultivariateNode{MvNormal}(fit(MvNormal, X), collect(1:2)))
normalize!(root);

@test length(root.children) == 1
@test root.weights[1] == 1.0
@test order(root)[end] == root
@test llh(root, zeros(D, 1))[1] > llh(root, ones(D, 1))[1]
