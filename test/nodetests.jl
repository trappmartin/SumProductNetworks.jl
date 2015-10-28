using Distributions

N = 100

X = randn(N)

root = SumNode(0)
add!(root, UnivariateNode(fit(Normal, X)))
normalize!(root);

@test length(root.children) == 1
@test root.weights[1] == 1.0
@test order(root)[end] == root
