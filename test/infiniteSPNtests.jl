
println(" * simple 2D assignment tests")
# simple test
N = 10
D = 2
X = randn(D, N)

# create Assignments
assign = SPN.Assignments(N)

# create simple structure (sum node and one leaf)
root = SumNode(0)
add!(root, MultivariateNode{MvNormal}(fit(MvNormal, X), collect(1:2)))

SPN.add!(assign, root)

# bucket size should be 0
@test assign(root) == 0

SPN.increment!(assign, root, i = N)

# bucket size should be N
@test assign(root) == N

# assign data points to leaf node
for i in collect(1:N)
	SPN.assign!(assign, i, root.children[1])
end

# check if all data points are assigned correctly
for i in collect(1:N)
	@test length(assign[i]) == 1
	@test assign[i][1] == root.children[1]
end

println(" * infinite GMM test")

srand(41234)

using Distributions

# data
X = rand(MultivariateNormal([5.0, 5.0], [1.0 0.0; 0.0 2.0]), 100) # 1
X = cat(2, X, rand(MultivariateNormal([-5.0, 5.0], [0.5 -0.2; -0.2 1.0]), 100)) # 2
X = cat(2, X, rand(MultivariateNormal([-5.0, -5.0], [1.0 0.0; 0.0 0.5]), 100)) # 3
X = cat(2, X, rand(MultivariateNormal([5.0, -5.0], [1.0 0.5; 0.5 0.5]), 100)) # 4

(D, N) = size(X)

μ0 = vec( mean(X, 2) )
κ0 = 1.0
ν0 = convert(Float64, D)
Ψ = cov(X, vardim = 2)

G0 = GaussianWishart(μ0, κ0, ν0, Ψ)

μ0 = vec( mean(X, 1) )
κ0 = 1.0
ν0 = convert(Float64, N)
Ψ = eye(N) * 100

G0Mirror = GaussianWishart(μ0, κ0, ν0, Ψ)

# create SPN
root = SumNode(0, scope = collect(1:D))
dist = MultivariateNode{ConjugatePostDistribution}(BNP.add_data(G0, X), collect(1:D))
add!(root, dist)

# create Assignments
assign = Assignments(N)
for i in collect(1:N)
    assign!(assign, i, dist)
end

increment!(assign, root, i = N)
increment!(assign, dist, i = N)

println(" * - Assignments on Root: ", assign(root))
println(" * - Assignments on Leaf: ", assign(dist))

println(" * - Finished initialisation for Gibbs test")

# get list of all leaf nodes
nodes = SPN.order(root)

# list of leafs
leafs = Vector{Leaf}(0)

for node in nodes
	if isa(node, Leaf)
		push!(leafs, node)
	end
end

@test length(leafs) == 1

# get N / D

(D, N) = size(X)

println(" * - Start parallel worker")


for id in [1] #randperm(N)

	x = X[:, id]
	kdists = assign[id]
	toremove = Vector{SPNNode}(0)

	# remove data point and withdraw nodes from datum
	for dist in kdists

		decrement!(assign, dist)
		SPN.withdraw!(assign, id, dist)

		remove_data!(dist.dist, x[dist.scope,:])

		@test assign(dist) >= 0

		if assign(dist) == 0
			push!(toremove, dist)
		end
	end

	# compute likelihoods of distribution nodes
	llh = Base.map( k -> SPN.eval(k, x)[1][1], leafs )
	println(llh)

	# compute prior for all selective SPNs
	

end


# run Gibbs steps
#println(" * - Gibbs sweep test")




#for i in collect(1:100)
#    println(" * - Iteration #", i)
#    gibbs_iteration!(root, assign, G0, G0Mirror, X, internalIters = 50)
#    println(" * - Draw SPN on iteration #", i)
#    SPN.draw(root, file="SPN_iteration_$(i).svg")
#    println(" * - Recompute weights")
#    SPN.update_weights(root, assign)
#    println(" * - LLH: ", llh(root, X)[1])
#end
