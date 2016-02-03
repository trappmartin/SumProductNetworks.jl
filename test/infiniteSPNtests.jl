println(" * create initial SPN using learnSPN")

#using RDatasets
#iris = dataset("datasets", "iris")

#X = convert(Array, iris[[:SepalLength, :SepalWidth, :PetalWidth]])'

# data
X = rand(MultivariateNormal([5.0, 5.0], [1.0 0.0; 0.0 2.0]), 100) # 1
X = cat(2, X, rand(MultivariateNormal([-2.5, 2.5], [0.5 -0.2; -0.2 1.0]), 100)) # 2
X = cat(2, X, rand(MultivariateNormal([-2.5, -2.5], [1.0 0.0; 0.0 0.5]), 100)) # 3
X = cat(2, X, rand(MultivariateNormal([5.0, -5.0], [1.0 0.5; 0.5 0.5]), 100)) # 4

(D, N) = size(X)

println(" * using dataset 'test' with ", N, " observations and ", D, " variables.")

dimMapping = Dict{Int, Int}([convert(Int, d) => convert(Int, d) for d in 1:D])
obsMapping = Dict{Int, Int}([convert(Int, n) => convert(Int, n) for n in 1:N])
assignments = Assignment()

root = SPN.learnSPN(X, dimMapping, obsMapping, assignments)

# draw initial solution
println(" * draw initial SPN")
drawSPN(root, file = "initialSPN.svg")

# transform SPN to regions and partitions
println(" * transform SPN into regions and partitions")
(spn, assign) = transformToRegionPartition(root, assignments, N)

@test size(spn.partitions, 1) == 1
@test size(spn.regions, 1) >= 3

# draw region graph (TODO)
println(" * draw transformed SPN")
drawSPN(spn, file = "transformedSPN.svg")

println(" * run Gibbs sweep on a sample using SPN in regions and partitions representation")

observation = 1

x = X[:,observation]

# parameters
α = 1.0

# evaluate all leaf regions

# 0.) remove observation from SPN
activeRegions = assign.regionAssignments[observation]
regionIdMapping = Dict{Region, Int}()
activePartitions = assign.partitionAssignments[observation]

# remove observation from regions and Distributions
for (region, cNode) in activeRegions

	if isa(region, LeafRegion)

		# decrease popularity
		region.popularity[cNode] -= 1
		region.N -= 1

		# remove from Distribution
		remove_data!(region.nodes[cNode].dist, x[region.nodes[cNode].scope,:])

		# remove node if the node is now empty
		if region.popularity[cNode] == 0
			# TODO ?!?
		end

	end

	if isa(region, SumRegion)

		# decrease popularity
		region.popularity[cNode] -= 1
		region.N -= 1

		# we need this for removal of partition assignments
		regionIdMapping[region] = cNode

	end

end

# remove observation from regions -> partition assignments
for (region, partition) in activePartitions

	if isa(region, SumRegion)

		# decrease popularity
		region.partitionPopularity[regionIdMapping[region]][partition] -= 1

	end

end

# 1.) get sample trees in the SPN
c = Dict{Region, Vector{Int}}()
cMax = Dict{Region, Vector{Int}}()

for region in spn.regions

	if isa(region, LeafRegion)
		c[region] = ones(Int, 1)
		cMax[region] = [size(region.nodes, 1)] # all nodes
	else
		c[region] = ones(Int, 2)
		cMax[region] = [size(region.partitionPopularity, 1), # all pseudo-nodes
										size(spn.regionConnections[region], 1)]
	end

end

@doc doc"""
Extract sample tree from configuration.
""" ->
function extractSampleTree(config::Dict{Region, Vector{Int}}, spn::SPNStructure)

	tree = Vector{Region}(0)

	for region in spn.regions

		# check if region is root (heuristic: no previous partition)
		isRoot = true

		for partition in spn.partitions
			isRoot &= !(region in spn.partitionConnections[partition])
		end

		if isRoot
			push!(tree, region)
		else

			# find out if region is inside tree
			foundSelection = false
			for partition in spn.partitions
				if region in spn.partitionConnections[partition]
					# check if the partition is selected by any region
					for r2 in spn.regions
						if partition in spn.regionConnections[r2]
							# partition is connected to region r2

							# is it selected?
							foundSelection |= (config[r2][2] == findfirst(spn.regionConnections[r2], partition))
						end
					end
				end
			end

			if foundSelection
				push!(tree, region)
			end
		end

	end

	return tree

end

# 2.) iterate over sample trees in the SPN
LLH = Vector{Float64}(0)
configurations = Vector{Dict{Region, Vector{Int}}}(0)

canIncrease = true

while canIncrease

	llh = 0.0
	lw = 0.0

	# get list of regions in sample tree
	sampleTree = extractSampleTree(c, spn)

	config = Dict{Region, Vector{Int}}()

	for region in sampleTree

		# get selection
		cRs = c[region]

		config[region] = cRs

		if isa(region, LeafRegion)

			cNode = cRs[1]
			# get llh values
			llh += logpred(region.nodes[cNode].dist, sub(x, region.nodes[cNode].scope, :))[1]

			# TODO: replace with correct formula

			# p(c_{i, Rg} = j | c_{-i, Rg}, α)
			lw += log(region.popularity[cNode] / (region.N - 1 + α) )
		else

			cNode = cRs[1]
			cPartition = cRs[2]

			# TODO: replace with correct formula

			# p(c_{i, Rg} = j | c_{-i, Rg}, α)
			lw =+ log(region.popularity[cNode] / (region.N - 1 + α) )

			#p(c_{i, S} = j | c_{-i, S}, \alpha)
			lw += log(region.partitionPopularity[cNode][spn.regionConnections[region][cPartition]] / (region.popularity[cNode] - 1 + α) )

		end

	end

	if (llh + lw) in LLH
		println(" # exact same value exists already...")
		println(" # skipping")
	else
		push!(LLH, llh + lw)
		push!(configurations, config)
	end

	#println("llh: ", llh + lw)

	# increase configuration if possible
	increased = false
	pos = size(spn.regions, 1)

	while (!increased) & (pos != 0)

		# check all settings in a region (1 if leafRegion, and 2 if sumRegion)
		posI = size(c[spn.regions[pos]], 1)
		for i in 1:posI

			if increased
				continue
			end

			if (c[spn.regions[pos]][i] + 1) <= cMax[spn.regions[pos]][i]
				c[spn.regions[pos]][i] += 1
				increased = true
			else
				c[spn.regions[pos]][i] = 1
			end
		end

		if !increased
			pos -= 1
		end

	end

	canIncrease = increased

end

println(" * finished computation of llh values for existing sample trees")
println(" * - finished ", length(LLH), " computations of sample trees")
println(" * - p(x, T | Θ) = ", exp(LLH))
println(" * - best configutation: ", configurations[indmax(LLH)])


#=
# create simple SPN
root = SumNode(0, scope = collect(1:D))
assign = Assignments(N)

# child nodes
p1 = ProductNode(1)
p2 = ProductNode(1)
s1 = SumNode(2, scope = collect(1:D))

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

		assign!(assign, i, p1, X[:,i])

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
		assign!(assign, i, p2, X[:,i])

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
		assign!(assign, i, p2, X[:,i])

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

	if assign(node) < 0
		println(node)
	end

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

function sumeval_topdown(node::Node, assign::Assignments, x; α = 1.0, mirror = false, parallel = false)

	# use pmap at this position for parallel computations
	if parallel
		ch = pmap(child -> eval_topdown(child, x, assign, α = α, mirror = mirror), children(node))
	else
		ch = map(child -> eval_topdown(child, x, assign, α = α, mirror = mirror), children(node))
	end

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
	push!(selectiveTrees, nchp[2][1])

	# add node to all selectiveTrees
	for tree in selectiveTrees
		push!(tree, node)
	end

	# compute weights
	w = map(child -> log(assign(child)), children(node))
	push!(w, log(α) )

	return (reduce(vcat, w .+ p), selectiveTrees)

end

"Copmute posterior predictive using selective tree CRP and conjugate priors."
function posterior_predictive(node::SumNode, assign::Assignments, x; α = 1.0, mirror = false)

	if !mirror
		return sumeval_topdown(node, assign, x, α = α, mirror = mirror)#, parallel = true)
	end

end

function eval_topdown{T<:Real}(node::SumNode, x::AbstractArray{T},
		assign::Assignments;
		α = 1.0,
		mirror = false)

		if !mirror
			return sumeval_topdown(node, assign, x, α = α, mirror = mirror)
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

		for tree in data[1]
			push!(tree, node)
		end

		return data[1]
	end

	idx = ones(Int, length(data))
	max = map(child -> length(child), data)

	canIncrease = true
	while canIncrease

		t = copy(data[1][idx[1]])

		for child in collect(2:length(data))
			# join sets
			union!(t, data[child][idx[child]])
		end

		# push to results
		push!(t, node)
		push!(result, t)

		# increase counter
		increased = false
		pos = length(idx)
		while !increased & pos != 0

			if (idx[pos] + 1) <= max[pos]
				idx[pos] += 1
				increased = true
			else
				idx[pos] = 1
				pos -= 1
			end

		end

		canIncrease = increased

	end

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

			# compute (selective trees)
			s = elwpathext(map(child -> child[2], ch), node)

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

println("processing ", N, " samples..")
i = 0

for id in randperm(N)

	i += 1

	x = X[:, id]
	nodes = assign[id]
	toremove = Vector{SPNNode}(0)

	# remove data point and withdraw
	for node in nodes
		if remove!(node, assign, id, x)
			push!(toremove, dist)
		end
	end

	# actually remove node from structure
	for node in toremove
		remove!(get(node.parent), node)
	end

	@time (llh, selectiveTrees) = posterior_predictive(root, assign, x)

	# coin tossing
	max = maximum(llh)
	k = BNP.rand_indices(exp(llh - max))

	if (i % 50) == 0
		draw(root, selectiveTrees[k], file = "selectived_tree_$(i).svg")
		draw(root, file = "spn_$(i).svg", showBucket = true, assign = assign)
	end

	# add datum to selective tree (add sub tree if required)
	for node in selectiveTrees[k]
		if isa(node, Leaf)
			add_data!(node.dist, x[node.scope,:])
		end

		assign!(assign, id, node, x)
		increment!(assign, node)

		# add to spn if its a new subtree
		if !isnull(node.parent) & !node.inSPN
			add!(get(node.parent), node)

			println(" # SPN chang..")
			draw(root, selectiveTrees[k], file = "selectived_tree_$(i).svg")
			draw(root, file = "spn_$(i).svg", showBucket = true, assign = assign)
		end

	end

end
=#
