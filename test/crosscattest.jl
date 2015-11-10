using SPN
using Base.Test

# 50% 1, 50% 2
Z = 1 + round(Int, rand(10) .>= 0.5)

d = DataAssignments(Z)

for (i, z) in enumerate(Z)
    @test d(i) == z
end

@test sum(find(Z .== 1) - d[1]) == 0
@test d.c == 2

# test functions on Buffer

N = 10
D = 2
X = rand(2, 10)
idx = collect(1:10)
Z = 1 + vec( round(Int, X[1,:] .>= 0.5) )

da = DataAssignments(Z)

Z1 = ones(Int, length(da[1]))
da1 = DataAssignments(Z1)

Z2 = ones(Int, length(da[2]))
da2 = DataAssignments(Z2)

push!(da.children, da1)
push!(da.children, da2)

# create Buffer Object
B = SPNBuffer(D, N, collect(1:10), X, da)

# create sub Buffer
B2 = B(1)

@test B2.N == length(da1.Z)

# test deepadd_data!

D = 2

μ0 = vec( zeros(D) )
κ0 = 1.0
ν0 = convert(Float64, D)
Ψ = eye(D)

G0 = GaussianWishart(μ0, κ0, ν0, Ψ);

n = MultivariateNode{ConjugatePostDistribution}(d, collect(1:D))

x = ones(D, 1)

llh1 = llh(n, x)

# add data
deepadd_data!(n, x)

llh2 = llh(n, x)

@test llh1 < llh2
