
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
	SPN.assign!(assign, i, root.children[1], X[:, i])
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

# child nodes
p1 = ProductNode(1)
p2 = ProductNode(1)
s1 = SumNode(2)

p11 = ProductNode(3)
p12 = ProductNode(4)

# assign parent ship
add!(root, p1)
add!(root, p2)

# test
@test parent(p1) == root
@test parent(p2) == root

@test length(root) == 2

increment!(assign, root, i = N)

# construct nodes

d1 = UnivariateNode{ConjugatePostDistribution}(BNP.add_data(NormalGamma(μ = mean(X[1,1:100])), X[1,1:100]), scope = 1)
d2 = UnivariateNode{ConjugatePostDistribution}(BNP.add_data(NormalGamma(μ = mean(X[2,1:100])), X[2,1:100]), scope = 2)

# create Assignments
for i in collect(1:100)
    assign!(assign, i, d1, X[:,i])
		assign!(assign, i, d2, X[:,i])

		assign!(assign, i, p2, X[:,i])

		assign!(assign, i, root, X[:,i])
end

increment!(assign, p1, i = 100)
increment!(assign, d1, i = 100)
increment!(assign, d2, i = 100)

add!(p1, d1)
add!(p1, d2)

# construct further nodes

d11 = UnivariateNode{ConjugatePostDistribution}(BNP.add_data(NormalGamma(μ = mean(X[1,101:200])), X[1,101:200]), scope = 1)
d12 = UnivariateNode{ConjugatePostDistribution}(BNP.add_data(NormalGamma(μ = mean(X[1,101:200])), X[2,101:200]), scope = 2)

d21 = UnivariateNode{ConjugatePostDistribution}(BNP.add_data(NormalGamma(μ = mean(X[1,201:N])), X[1,201:N]), scope = 1)
d22 = UnivariateNode{ConjugatePostDistribution}(BNP.add_data(NormalGamma(μ = mean(X[1,201:N])), X[2,201:N]), scope = 2)

# create Assignments
for i in collect(101:200)
    assign!(assign, i, d11, X[:,i])
		assign!(assign, i, d12, X[:,i])

		assign!(assign, i, p11, X[:,i])

		assign!(assign, i, s1, X[:,i])
		assign!(assign, i, p1, X[:,i])

		assign!(assign, i, root, X[:,i])
end

increment!(assign, p11, i = 100)
increment!(assign, d11, i = 100)
increment!(assign, d12, i = 100)

add!(p11, d11)
add!(p11, d12)

for i in collect(201:N)
    assign!(assign, i, d21, X[:,i])
		assign!(assign, i, d22, X[:,i])

		assign!(assign, i, p12, X[:,i])

		assign!(assign, i, s1, X[:,i])
		assign!(assign, i, p1, X[:,i])

		assign!(assign, i, root, X[:,i])
end

increment!(assign, p12, i = N-200)
increment!(assign, d21, i = N-200)
increment!(assign, d22, i = N-200)

add!(p12, d21)
add!(p12, d22)

increment!(assign, s1, i = N-100)
increment!(assign, p2, i = N-100)

add!(s1, p11)
add!(s1, p12)

add!(p2, s1)

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

@test length(leafs) == 6

function remove!(node::Node, assign::Assignments, id::Int, x)

	decrement!(assign, node)
	SPN.withdraw!(assign, id, node, x)

	@assert assign(node) >= 0

	return false

end

function remove!(node::Leaf, assign::Assignments, id::Int, x)

	decrement!(assign, node)
	SPN.withdraw!(assign, id, node, x)

	remove_data!(node.dist, x[node.scope,:])

	@assert assign(node) >= 0

	return assign(node) == 0

end

"Copmute posterior predictive using selective tree CRP and conjugate priors."
function posterior_predictive(node::SumNode, assign::Assignments, x; α = 1.0, mirror = false)

	if !mirror

		# use pmap at this position for parallel computations
		ch = map(child -> eval_topdown(child, x, assign, α = α, mirror = mirror), children(node))

		p = map(child -> child[1], ch)

		# compute selective tree paths
		selectiveTrees = map(child -> child[2], ch)
		selectiveTrees = reduce(vcat, selectiveTrees)

		# construct "new" child
		nch = ProductNode(0)
		nch.parent = node

		# get G0 parameter
		μ0 = assign.ZZ[node] ./ assign(node)

		for scope in node.scope
			add!(nch, UnivariateNode{ConjugatePostDistribution}(NormalGamma(μ = μ0[scope]), scope = scope))
		end

		nchp = eval_topdown(nch, x, assign)

		push!(p, nchp[1])
		println(length(nchp[2]))
		push!(selectiveTrees, nchp[2])

		# add node to all selectiveTrees
		for tree in selectiveTrees
			push!(tree, node)
		end

		# compute weights
		w = map(child -> log(assign(child)), children(node))
		push!(w, log(α) )

		return (reduce(vcat, w .+ p), selectiveTrees)

	end

end

function eval_topdown{T<:Real}(node::SumNode, x::AbstractArray{T},
		assign::Assignments;
		α = 1.0,
		mirror = false)

		if !mirror

			ch = map(child -> eval_topdown(child, x, assign, α = α, mirror = mirror), children(node))

			p = map(child -> child[1], ch)

			# compute pathes (selective spns)
			selectiveTrees = map(child -> child[2], ch)
			selectiveTrees = reduce(vcat, selectiveTrees)
			push!(selectiveTrees, Set{SPNNode}())

			# add node to all selectiveTrees
			for tree in selectiveTrees
				push!(tree, node)
			end

			# get G0 parameter
			μ0 = assign.ZZ[node] ./ assign(node)

			# get G0 prediction
			pp = sum(map(scope -> logpred(NormalGamma(μ = μ0[scope]), x[scope, :])[1], node.scope))
			p = push!(p, collect(pp))

			# compute weights
			w = map(child -> log(assign(child)), children(node))
			push!(w, log(α) )

			return (reduce(vcat, w .+ p), selectiveTrees)
		end
end

function elwlogprod(data::Array)
	if length(data) >= 2

		d1 = data[1]
		d2 = elwlogprod(data[2:end])

		r = zeros(length(d1) * length(d2))
		i = 1

		for x1 in d1
			for x2 in d2
				r[i] = x1 + x2
			end
		end

		return r

	else
		return data[1]
	end
end

function elwpathext(data::Array, node::SPNNode)

	result = Array{Set{SPNNode}}(0)

	if length(data) == 1
		return data[1]
	end

	counter = ones(Int, length(data))
	max = map(child -> length(child), data)

	canIncrease = true
	while canIncrease

		println(counter)
		println(data[1][counter[1]])
		t = copy(data[1][counter[1]])

		for child in collect(2:length(data))
			# join sets
			union!(t, data[child][counter[child]])
		end

		# push to results
		push!(result, t)

		# increase counter
		increased = false
		pos = length(counter)
		while !increased | pos != 0

			if (counter[pos] + 1) <= max[pos]
				counter[pos] += 1
				increased = true
			else
				counter[pos] = 1
				pos -= 1
			end

		end

		canIncrease = increased

	end

	println(length(result))

	return result
end

"Each product node produces prod_{j ∈ children(node)} dim_j dimensional output."
function eval_topdown{T<:Real}(node::ProductNode, x::AbstractArray{T},
		assign::Assignments;
		α = 1.0,
		mirror = false)

		if !mirror

			ch = map(child -> eval_topdown(child, x, assign, α = α, mirror = mirror), children(node))

			# compute element wise product in log space
			p = elwlogprod(map(child -> child[1], ch))

			# compute pathes (selective spns)
			s = elwpathext(map(child -> child[2], ch), node)
			#println(s)

			return (p, s)
		end

end


function eval_topdown{T<:Real, U<:ConjugatePostDistribution}(node::UnivariateNode{U},
   data::AbstractArray{T},
   assign::Assignments;
   α = 1.0,
   mirror = false)

	 llh = logpred(node.dist, sub(data, node.scope, :))
	 @assert !isnan(llh[1])

	 tree = Array{Set{SPNNode}}(1)
	 tree[1] = Set{SPNNode}(collect([node]))

	 return (llh, tree)

end

function eval_topdown{T<:Real, U<:ConjugatePostDistribution}(node::MultivariateNode{U},
   data::AbstractArray{T},
   assign::Assignments;
   α = 1.0,
   mirror = false)

	 llh = logpred(node.dist, data[node.scope,:])
	 @assert !isnan(llh[1])

	 tree = Array{Set{SPNNode}}(1)
	 tree[1] = Set{SPNNode}(collect([node]))

	 return (llh, tree)

end

# get N / D

(D, N) = size(X)

for id in [1 2] #randperm(N)

	x = X[:, id]
	nodes = assign[id]
	toremove = Vector{SPNNode}(0)

	# remove data point and withdraw
	for node in nodes
		if remove!(node, assign, id, x)
			push!(toremove, dist)
		end
	end

	draw(root)

	(llh, selectiveTrees) = posterior_predictive(root, assign, x)

	println("x: ", x)

	println("llh: ", llh)
	println("# trees: ", size(selectiveTrees))

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
