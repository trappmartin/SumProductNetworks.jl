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
	popularity::Int

	Partition() = new( Set{Int}(), Dict{Int64, Int64}(), )

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
	weights::Vector{Dict{Partition, Float64}}
	popularity::Dict{Int64, Int64}
	N::Int

	SumRegion() = new( Set{Int}(), Vector{Dict{Partition, Float64}}(0), Dict{Int64, Int64}(), 0 )

end

@doc doc"""
A region containing Distributions (leaf nodes).
""" ->
type LeafRegion <: Region

	scope::Int
	nodes::Vector{Leaf}
	popularity::Dict{Int64, Int64}
	N::Int

	LeafRegion(scope) = new( scope, Vector{Leaf}(0), Dict{Int64, Int64}(), 0 )

end

@doc doc"""
SPN structure object in regions / partitions representation.
""" ->
type SPNStructure

	regions::Vector{Region}
	regionConnections::Dict{Region, Vector{Partition}}
	partitions::Vector{Partition}
	partitionConnections::Dict{Partition, Vector{Region}}

	SPNStructure() = new( Vector{Region}(0), Dict{Region, Vector{Partition}}(), Vector{Partition}(0), Dict{Partition, Vector{Region}}() )

end

@doc doc"""
Find a matching partition (same scope and indexing function).

Returns index of matched partition or -1 if no match could be found.
""" ->
function findPartition(scope::Set{Int}, indexFunction::Dict{Int64, Int64}, spn::SPNStructure)

	for (pi, partition) in enumerate(spn.partitions)

		if scope == partition.scope

			# check index functions (clusterings)
			c1 = [partition.indexFunction[s] for s in scope]
			c2 = [indexFunction[s] for s in scope]

			r = adjustedRandIndex(c1, c2)

			# equal if r == 1
			if r == 1.0
				return pi
			end

		end
	end

	return -1
end
