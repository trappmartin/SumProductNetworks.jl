
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

# create simple SPN
root = SumNode(0, scope = collect(1:D))
assign = Assignments(N)

# product nodes
p1 = ProductNode(1)
p2 = ProductNode(2)

# assign parent ship
add!(root, p1)
add!(root, p2)

# test
@test parent(p1) == root
@test parent(p2) == root

@test length(root) == 2

increment!(assign, root, i = N)

# construct nodes

μ0 = mean(X[1,1:100])

G0 = NormalGamma(μ = μ0)

d1 = UnivariateNode{ConjugatePostDistribution}(BNP.add_data(G0, X[1,1:100]), scope = 1)
d2 = UnivariateNode{ConjugatePostDistribution}(BNP.add_data(G0, X[2,1:100]), scope = 2)

# create Assignments
for i in collect(1:100)
    assign!(assign, i, d1)
		assign!(assign, i, d2)
end

increment!(assign, d1, i = 100)
increment!(assign, d2, i = 100)

add!(p1, d1)
add!(p1, d2)

# construct further nodes

μ0 = mean(X[1,1:100])

G0 = NormalGamma(μ = μ0)

d3 = UnivariateNode{ConjugatePostDistribution}(BNP.add_data(G0, X[1,101:N]), scope = 1)
d4 = UnivariateNode{ConjugatePostDistribution}(BNP.add_data(G0, X[2,101:N]), scope = 2)

# create Assignments
for i in collect(101:N)
    assign!(assign, i, d3)
		assign!(assign, i, d4)
end

increment!(assign, d3, i = N-100)
increment!(assign, d4, i = N-100)

add!(p2, d3)
add!(p2, d4)

println(" * - Assignments on Root: ", assign(root))

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

@test length(leafs) == 4

# get N / D

(D, N) = size(X)

for id in [1 2] #randperm(N)

	x = X[:, id]
	kdists = assign[id]
	toremove = Vector{SPNNode}(0)

	# remove data point and withdraw
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
	@time llh = pmap( k -> SPN.eval(k, x)[1][1], leafs )
	@time llh = map( k -> SPN.eval(k, x)[1][1], leafs )


	println("x: ", x)

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
