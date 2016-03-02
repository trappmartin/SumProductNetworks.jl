"Assignment Data Object"
type Assignment

	# bucket list
	S::Dict{SPNNode, Vector{Int}}

	Assignment(;bucket = Dict{SPNNode, Vector{Int}}() ) = new( bucket )

end

call(p::Assignment, n::SPNNode) = p.S[n]

"Add new node to bucket list"
function add!(p::Assignment, n::SPNNode)
	if !haskey(p.S, n)
		p.S[n] = Vector{Int}(0)
	end
end

"Set item into bucket"
function set!(p::Assignment, n::SPNNode, idx::Int)

	if !haskey(p.S, n)
		add!(p, n)
	end

	push!(p.S[n], idx)
end

"Set items into bucket"
function set!(p::Assignment, n::SPNNode, idx::Vector{Int})
	for id in idx
		set!(p, n, id)
	end
end

@doc doc"""
A partition containing a index function over its scope.
""" ->
type Partition

	scope::Set{Int}
	indexFunction::Dict{Int64, Int64}
	subscopes::Vector{Set{Int}}

	Partition() = new( Set{Int}(), Dict{Int64, Int64}(), Vector{Set{Int}}(0) )

end

@doc doc"""
A abstract region.
""" ->
abstract Region

@doc doc"""
A region containing sum node definied over the same scope.
""" ->
type SumRegion <: Region

	scope::Set{Int}
	partitionPopularity::Vector{Dict{Partition, Int64}}
	popularity::Vector{Int64}
	N::Int
	isRoot::Bool

	SumRegion(;root = false) = new( Set{Int}(), Vector{Dict{Partition, Int64}}(0), Vector{Int64}(0), 0, root )

end

@doc doc"""
A region containing Distributions (leaf nodes).
""" ->
type LeafRegion <: Region

	scope::Int
	nodes::Vector{Leaf}
	popularity::Vector{Int64}
	N::Int

	LeafRegion(scope) = new( scope, Vector{Leaf}(0), Vector{Int64}(0), 0 )

end

@doc doc"""
SPN structure object in regions / partitions representation.
""" ->
type SPNStructure

	regions::Vector{Region}
	regionConnections::Dict{Region, Vector{Partition}}
	regionConnectionsBottomUp::Dict{Region, Vector{Partition}}
	partitions::Vector{Partition}
	partitionConnections::Dict{Partition, Vector{Region}}
	partitionConnectionsBottomUp::Dict{Partition, Vector{Region}}

	SPNStructure() = new( Vector{Region}(0), Dict{Region, Vector{Partition}}(), Dict{Region, Vector{Partition}}(), Vector{Partition}(0), Dict{Partition, Vector{Region}}(), Dict{Partition, Vector{Region}}() )

end

@doc doc"""
Assignment Data Object for Region / Partition Representation
""" ->
type AssignmentRegionGraph

	regionAssignments::Vector{Dict{Region, Int}}
	partitionAssignments::Vector{Dict{Region, Partition}}
	observationRegionAssignments::Dict{Region, Set{Int}}
	observationPartitionAssignments::Dict{Partition, Set{Int}}

	AssignmentRegionGraph(observations::Int) = new( vec([Dict{Region, Int}() for i in 1:observations]),
			vec([Dict{Region, Partition}() for i in 1:observations]),
			Dict{Region, Set{Int}}(), Dict{Partition, Set{Int}}() )

end

@doc doc"""
Extend set of partitions with children and parent of sum node.
""" ->
function extendPartitions(node::SumNode, spn::SPNStructure, region::SumRegion, assignments::Assignment, assign::AssignmentRegionGraph)

	c = Dict{Node, Partition}()

	# connect children
	for child in node.children

		id = findPartition(child, spn, assignments)

		c[child] = spn.partitions[id]

		if !(region in spn.partitionConnectionsBottomUp[spn.partitions[id]])
			push!(spn.partitionConnectionsBottomUp[spn.partitions[id]], region)
		end

		if !(spn.partitions[id] in spn.regionConnections[region])
			push!(spn.regionConnections[region], spn.partitions[id])
		end

		for observation in assignments(child)
			assign.partitionAssignments[observation][region] = spn.partitions[id]
		end

		# append assignments
		if !haskey(assign.observationPartitionAssignments, spn.partitions[id])
			assign.observationPartitionAssignments[spn.partitions[id]] = Set(assignments(node))
		else
			union!(assign.observationPartitionAssignments[spn.partitions[id]], Set(assignments(node)))
		end

	end

	# connect parents
	if !isnull(node.parent)

		parent = get(node.parent)

		id = findPartition(parent, spn, assignments, bottomUp = true)

		if !(region in spn.partitionConnections[spn.partitions[id]])
			push!(spn.partitionConnections[spn.partitions[id]], region)
		end

		if !(spn.partitions[id] in spn.regionConnectionsBottomUp[region])
			push!(spn.regionConnectionsBottomUp[region], spn.partitions[id])
		end

		for observation in assignments(parent)
			assign.partitionAssignments[observation][region] = spn.partitions[id]
		end

		# append assignments
		if !haskey(assign.observationPartitionAssignments, spn.partitions[id])
			assign.observationPartitionAssignments[spn.partitions[id]] = Set(assignments(node))
		else
			union!(assign.observationPartitionAssignments[spn.partitions[id]], Set(assignments(node)))
		end

	end

	return c

end

@doc doc"""
Extend set of partitions with parent of leaf node.
""" ->
function extendPartitions(node::Leaf, spn::SPNStructure, region::LeafRegion, assignments::Assignment, assign::AssignmentRegionGraph)

	# connect parents
	if !isnull(node.parent)

		parent = get(node.parent)

		id = findPartition(parent, spn, assignments, bottomUp = true)

		if !(region in spn.partitionConnections[spn.partitions[id]])
			push!(spn.partitionConnections[spn.partitions[id]], region)
		end

		if !(spn.partitions[id] in spn.regionConnectionsBottomUp[region])
			push!(spn.regionConnectionsBottomUp[region], spn.partitions[id])
		end

		for observation in assignments(parent)
			assign.partitionAssignments[observation][region] = spn.partitions[id]
		end

		# append assignments
		if !haskey(assign.observationPartitionAssignments, spn.partitions[id])
			assign.observationPartitionAssignments[spn.partitions[id]] = Set(assignments(node))
		else
			union!(assign.observationPartitionAssignments[spn.partitions[id]], Set(assignments(node)))
		end

	end

	return

end

@doc doc"""
Find a matching partition (same scope and indexing function).

Returns index of matched partition or -1 if no match could be found.
""" ->
function findPartition(scope::Set{Int}, indexFunction::Dict{Int64, Int64}, spn::SPNStructure)

	for (pi, partition) in enumerate(spn.partitions)

		if scope == partition.scope
#
			# check index functions (clusterings)
			c1 = vec([partition.indexFunction[s] for s in scope])
			c2 = vec([indexFunction[s] for s in scope])

			r = adjustedRandIndex(c1, c2)

			# equal if r == 1
			if r == 1.0
				return pi
			end

		end
	end

	return -1
end

@doc doc"""
Find partition for node.
""" ->
function findPartition(node::Node, spn::SPNStructure, assignments::Assignment; bottomUp = false)
	# get indexing function
	idxFun = Dict{Int64, Int64}()

	for (ci, c) in enumerate(node.children)
		for s in c.scope
			idxFun[s] = ci
		end
	end

	id = findPartition(Set(node.scope), idxFun, spn)

	if id == -1
		# create new one
		partition = Partition()
		partition.scope = Set(node.scope)
		partition.indexFunction = idxFun

		# generate sub scopes
		uvs = unique(values(partition.indexFunction))
		partition.subscopes = [Set(collect(partition.scope)[find(uv .== collect(values(partition.indexFunction)))]) for uv in uvs]

		push!(spn.partitions, partition)
		spn.partitionConnections[partition] = Vector{Region}(0)
		spn.partitionConnectionsBottomUp[partition] = Vector{Region}(0)
		id = size(spn.partitions, 1)
	end

	return id
end

@doc doc"""
Extend set of regions with sum node. Create new region if necessary
and extend set of partitions with children of sum node.
""" ->
function extendRegions!(node::SumNode, spn::SPNStructure, assignments::Assignment, assign::AssignmentRegionGraph, isRoot::Bool)

	nscope = Set(node.scope)

	for region in spn.regions

		# check nodes have same scope
		if nscope == region.scope
			# scope matches

			products_partitions = extendPartitions(node, spn, region, assignments, assign)

			id = size(region.partitionPopularity, 1) + 1

			push!(region.partitionPopularity, Dict{Partition, Int64}())
			for child in node.children
				region.partitionPopularity[id][products_partitions[child]] = length(assignments(child))
			end

			push!(region.popularity, length(assignments(node)))
			region.N += length(assignments(node))

			for observation in assignments(node)
				assign.regionAssignments[observation][region] = id
			end

			# append assignments
			if !haskey(assign.observationRegionAssignments, region)
				assign.observationRegionAssignments[region] = Set(assignments(node))
			else
				union!(assign.observationRegionAssignments[region], Set(assignments(node)))
			end

			return

		end

	end

	# not found => make new region

	region = SumRegion(root = isRoot)
	spn.regionConnections[region] = Vector{Partition}(0)
	spn.regionConnectionsBottomUp[region] = Vector{Partition}(0)

	region.scope = nscope

	# add new partition
	products_partitions = extendPartitions(node, spn, region, assignments, assign)

	id = size(region.partitionPopularity, 1) + 1

	push!(region.partitionPopularity, Dict{Partition, Int64}())
	for child in node.children
		region.partitionPopularity[id][products_partitions[child]] = length(assignments(child))
	end

	push!(region.popularity, length(assignments(node)))
	region.N += length(assignments(node))

	for observation in assignments(node)
		assign.regionAssignments[observation][region] = id
	end

	# append assignments
	if !haskey(assign.observationRegionAssignments, region)
		assign.observationRegionAssignments[region] = Set(assignments(node))
	else
		union!(assign.observationRegionAssignments[region], Set(assignments(node)))
	end

	push!(spn.regions, region)

end

@doc doc"""
Extend set of regions with leaf node. Create new region if necessary
and extend set of partitions with parent of the node.
""" ->
function extendRegions!(node::Leaf, spn::SPNStructure, assignments::Assignment, assign::AssignmentRegionGraph, isRoot::Bool)

	nscope = node.scope[1]

	for region in spn.regions

		# check nodes have same scope
		if nscope == region.scope
			# scope matches

			idx = size(region.nodes, 1) + 1

			push!(region.nodes, node)
			push!(region.popularity, length(assignments(node)))
			region.N += length(assignments(node))
			extendPartitions(node, spn, region, assignments, assign)

			for observation in assignments(node)
				assign.regionAssignments[observation][region] = idx
			end

			# append assignments
			if !haskey(assign.observationRegionAssignments, region)
				assign.observationRegionAssignments[region] = Set(assignments(node))
			else
				union!(assign.observationRegionAssignments[region], Set(assignments(node)))
			end

			return

		end

	end

	# not found => make new region

	region = LeafRegion(nscope)
	spn.regionConnections[region] = Vector{Partition}(0)
	spn.regionConnectionsBottomUp[region] = Vector{Partition}(0)

	idx = size(region.nodes, 1) + 1

	push!(region.nodes, node)
	push!(region.popularity, length(assignments(node)))
	region.N += length(assignments(node))
	extendPartitions(node, spn, region, assignments, assign)

	for observation in assignments(node)
		assign.regionAssignments[observation][region] = idx
	end

	# append assignments
	if !haskey(assign.observationRegionAssignments, region)
		assign.observationRegionAssignments[region] = Set(assignments(node))
	else
		union!(assign.observationRegionAssignments[region], Set(assignments(node)))
	end

	push!(spn.regions, region)

end

@doc doc"""
Extend set of regions with leaf node. Create new region if necessary
and extend set of partitions with parent of the node.
""" ->
function transformToRegionPartition(root::SumNode, assignments::Assignment, N::Int)
	nodes = SPN.order(root)

	spn = SPNStructure()
	assign = AssignmentRegionGraph(N)

	# apply transformation to every sum or leaf node
	for node in nodes

		if isa(node, SumNode) | isa(node, Leaf)
			extendRegions!(node, spn, assignments, assign, node == root)
		end

	end

	return (spn, assign)
end
