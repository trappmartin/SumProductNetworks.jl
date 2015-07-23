using SPN
using Base.Test
using Distributions

N = 1000
X1 = randn(N)
X2 = randn(N) + 10

# build SPN with one multivariate leaf node
node = build_sum(0)

d = MvNormal(vec(mean(hcat(X1, X2), 1)), cov(hcat(X1, X2)))
child3 = build_multivariate(d, [1, 2])

add(node, child3)
normalize(node)

@test mean(llh(node, hcat(X1, X2))) > -3.0
@time mean(llh(node, hcat(X1, X2)))

# build SPN assuming independence between variables
node = build_sum(0)
pnode = build_prod(1)

child1 = build_univariate(fit(Normal, X1), [1])
child2 = build_univariate(fit(Normal, X2), [2])

add(pnode, child1)
add(pnode, child2)
add(node, pnode)
normalize(node)

@test mean(llh(node, hcat(X1, X2))) > -3.0
@time mean(llh(node, hcat(X1, X2)))
