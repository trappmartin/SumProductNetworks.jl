println(" * DataAssignments test...")
# 50% 1, 50% 2
Z = 1 + round(Int, rand(10) .>= 0.5)

d = SPN.DataAssignments(Z, collect(1:10), 1, 10)

for (i, z) in enumerate(Z)
    @test d(i) == z
end

@test sum(find(Z .== 1) - d[1]) == 0
@test d.c == 2

# test functions on Buffer

N = 10
D = 2
X = rand(Float64, 2, 10)
idx = collect(1:10)
Z = 1 + vec( round(Int, X[1,:] .>= 0.5) )

da = SPN.DataAssignments(Z, collect(1:10), 2, 10)

Z1 = ones(Int, length(da[1]))
da1 = SPN.DataAssignments(Z1, da[1], 2, length(da[1]))

Z2 = ones(Int, length(da[2]))
da2 = SPN.DataAssignments(Z2, da[2], 2, length(da[2]))

@test da1.N + da2.N == da.N

println(" * SPNBuffer tests...")

using Distributions

# node assiciated with data assignment
dist = MvNormal(vec(mean(X, 2)), cov(X, vardim=2))
node = MultivariateNode{MvNormal}(fit(MvNormal, X), collect(1:2))

n2d = Dict{SPNNode, SPN.DataAssignments}(node => da)

# create Buffer Object
B = SPN.SPNBuffer(X, n2d)

@test B.Z[node] == da
@test sum(SPN.get(B, 1) - X[:,1]) ≈ 0

println(" * deep add & remove (distribution) tests...")
# test deepadd_data!
N = 100
D = 2

X = randn(D, N)

μ0 = vec( mean(X, 2) )
κ0 = 1.0
ν0 = convert(Float64, D)
Ψ = eye(D) * 10

G0 = GaussianWishart(μ0, κ0, ν0, Ψ);

node = MultivariateNode{ConjugatePostDistribution}(BNP.add_data(G0, X[:,1:end-1]), collect(1:D))
da = SPN.DataAssignments(ones(Int, N), collect(1:N), D, N)

n2d = Dict{SPNNode, SPN.DataAssignments}(node => da)

# create Buffer Object
B = SPN.SPNBuffer(X, n2d)

x = X[:,end]

llh1 = llh(node, x)[1]

# add data to leaf
SPN.deepadd_data!(node, B, N)
llh2 = llh(node, x)[1]

@test llh1 < llh2
@test B.Z[node].N == N
@test sum(B.Z[node].ids .== N) == 1
@test B.Z[node](N) == 1

# remove data from leaf
SPN.deepremove_data!(node, B, N)

@test length(B.Z[node][1]) == N-1
@test B.Z[node].active[N] == false

llh1 = llh(node, x)[1]
SPN.deepadd_data!(node, B, N)
llh2 = llh(node, x)[1]

@test llh1 < llh2

# add data to internal node
println(" * deep add & remove (internal node) tests...")

N = 100
D = 2
X1 = randn(D, N)
X2 = randn(D, N) + 10
root = SumNode(0)
add!(root, MultivariateNode{ConjugatePostDistribution}(BNP.add_data(G0, X1), collect(1:D)))
add!(root, MultivariateNode{ConjugatePostDistribution}(BNP.add_data(G0, X2), collect(1:D)))

da0 = SPN.DataAssignments(ones(Int, N*2), collect(1:N*2), D, N*2)
da1 = SPN.DataAssignments(da0, ones(Int, N), collect(1:N))
da2 = SPN.DataAssignments(da0, ones(Int, N), collect(N+1:N*2))

n2d = Dict{SPNNode, SPN.DataAssignments}(root => da0, root.children[1] => da1, root.children[2] => da2)

# create Buffer Object
B = SPN.SPNBuffer(X, n2d)

SPN.deepremove_data!(root, B, 1)

# item should be in-active for all data assignments
@test da0.active[1] == false
@test da1.active[1] == false
@test da2.active[1] == false

SPN.deepadd_data!(root, B, 1)

# item should be only active for data assignment 0 and 1
@test da0.active[1] == true
@test da1.active[1] == true
@test da2.active[1] == false
