
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

# data
M = 10
N = M * 2
D = 2

X = cat(2, randn(D, M) + 10, randn(D, M) - 10)

# G0
μ0 = vec( mean(X, 2) )
κ0 = 1.0
ν0 = convert(Float64, D)
Ψ = eye(D)

G0 = GaussianWishart(μ0, κ0, ν0, Ψ)

root = SumNode(0, scope = collect(1:D))
dist = MultivariateNode{ConjugatePostDistribution}(BNP.add_data(G0, X), collect(1:D))
add!(root, dist)

# create Assignments
assign = SPN.Assignments(N)
for i in collect(1:N)
	SPN.assign!(assign, i, dist)
end

SPN.add!(assign, root)
SPN.increment!(assign, root, i = N)

SPN.add!(assign, dist)
SPN.increment!(assign, dist, i = N)

# single Gibbs step
println(" * Gibbs sweep test")

for id in randperm(N)

	x = X[:, id]

	kdists = assign[id]

	for dist in kdists
		# - remove "random" data point
		SPN.decrement!(assign, dist)
		remove_data!(dist.dist, x[dist.scope,:])
	end

	# get k's

	toporder = SPN.order(root)

	llhval = Dict{SPNNode, Array{Float64}}()
	kvals = Dict{SPNNode, Int}()

	for node in toporder

			(llh, newk) = SPN.evalWithK(node, x, llhval, assign, G0)

			# always open a new table

			llhval[node] = llh
			kvals[node] = newk
	end

	# assign datum to
	SPN.recurseCondK!(root, kvals, x, id, assign, G0)

end

# test removal
#kdists = assign[id]

#for dist in kdists
	# - remove "random" data point
#	SPN.decrement!(assign, dist)

#	remove_data!(dist.dist, x)

#	if assign(dist) == 0
#		SPN.killChild!(dist, assign)
#	end
#end

#@test length(root.children) == 1

println(" * Extend SPN with Product")

SPN.extend!(root, assign)

println(" * Gibbs on mirror SPN")

toporder = SPN.order(root)

println(" * Compute mirrored Assignment for first time...")

assignMirror = SPN.Assignments(D)
assignMirror.S = assign.S

for node in toporder

	if isa(node, SPN.Leaf)

		for i in node.scope
			SPN.assign!(assignMirror, i, dist)
		end

	else
		break
	end
end

# mirror all Leafs
println(" * mirror all leafs")

for node in toporder

	if isa(node, SPN.Leaf)
		SPN.mirror!(node, assign, X, G0)
	else
		break
	end

end

println(" * parallel sweeps")

# for each product node
for child in root.children

	if isa(child, ProductNode)

		for id in randperm(D)

			# define G0 for mirrored SPN rooted at child

			# G0Mirror
			μ0 = vec( mean(X[:,child.scope], 1) )
			κ0 = 1.0
			ν0 = convert(Float64, length(child.scope))
			Ψ = eye(length(child.scope)) * 100

			G0Mirror = GaussianWishart(μ0, κ0, ν0, Ψ)

			x = X[id, :]'

			kdists = assignMirror[id]

			for dist in kdists
				# - remove data point
				SPN.decrement!(assign, dist)
				remove_data!(dist.dist, x[dist.scope,:])
			end

			toporder = SPN.order(child)

			llhval = Dict{SPNNode, Array{Float64}}()
			kvals = Dict{SPNNode, Int}()

			for node in toporder

					(llh, newk) = SPN.evalWithK(node, x, llhval, assign, G0Mirror, mirror = true)

					llhval[node] = llh
					kvals[node] = newk
			end

		end

	end

end


println(" * Draw SPN")
#SPN.draw(root)
