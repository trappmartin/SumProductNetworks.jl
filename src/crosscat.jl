import Base.getindex

type DataAssignments

    # number of unique clusters
    c::Int

    # assignments
    Z::Vector{Int}

    # ids
    ids::Vector{Int}

    # active ids
    active::Vector{Bool}

    # dimensionality of data
    D::Int
    N::Int

    # constructor
    DataAssignments(Z::Vector{Int}, ids::Vector{Int}, D::Int, N::Int) = new(
        Base.length(unique(Z)),
        Z,
        ids,
        ones(Bool, N),
        D,
        N)

    function DataAssignments(p::DataAssignments, Z::Vector{Int}, ids::Vector{Int})
        ZZ = copy(p.Z)
        ZZ[ids] = Z
        active = zeros(Bool, p.N)
        active[ids] = true

        new( Base.length(unique(Z)),
            ZZ,
            p.ids,
            active,
            p.D,
            p.N)
    end

end

# functions on DataAssignments
call(p::DataAssignments, idx::Int) = p.Z[idx]

"Get index function for DataAssignments"
function getindex(p::DataAssignments, i::Int)
    p.ids[find(p.Z[p.active] .== i)]
end

"add data assignment"
function add!(p::DataAssignments, id::Int, z::Int)

    if findfirst(p.Z .== z) == 0
        # we need to increase the cluster count
        p.c += 1
    end

    p.Z[id] = z
    p.active[id] = true

end

"delete assignment for DataAssignments"
function remove!(p::DataAssignments, id::Int)

    if !p.active[id]
        return p
    end

    if Base.length(p[p(id)]) == 1
        # we need to remove cluster count
        p.c -= 1
    end

    p.active[id] = false
end

type SPNBuffer

    # data matrix
    X::AbstractArray{Float64, 2}

    # assignment tree
    Z::Dict{SPNNode, DataAssignments}

end

"Get datum from Buffer"
function get(B::SPNBuffer, idx::Int)
    sub(B.X, :, idx)
end

## utility functions

"Add data point to Distribution"
function deepadd_data!(node::MultivariateNode{ConjugatePostDistribution}, B::SPNBuffer, id::Int)

    x = get(B, id)
    add!(B.Z[node], id, 1)

    add_data!(node.dist, x)
end

"Alias function for deepadd_data!"
function traverse_deepadd!(path::Dict{SPNNode, Array{SPNNode}}, node::Leaf, B::SPNBuffer, id::Int)
    deepadd_data!(node, B, id)
end

"Traverse through MAP tree and add datum."
function traverse_deepadd!(path::Dict{SPNNode, Array{SPNNode}}, node::Node, B::SPNBuffer, id::Int)

    children = path[node]

    if Base.length(children) > 1 # product node

        for (z, child) in enumerate(children)
            traverse_deepadd!(path, child, B, id)

            # add to Buffer
            add!(B.Z[node], id, z)
        end

    elseif Base.length(children) == 1 # sum node
        z = findfirst(node.children .== children[1])

        traverse_deepadd!(path, children[1], B, id)

        add!(B.Z[node], id, z)
    end
end

"Add data point to Distribution (intermediate node)"
function deepadd_data!(node::Node, B::SPNBuffer, id::Int; z = 1)

    x = get(B, id)

    # compute map path to find distribution
    path = map(node, x)[2]

    # traverse through path and add data
    traverse_deepadd!(path, node, B, id)

end

"Remove data point from Distribution"
function deepremove_data!(node::MultivariateNode{ConjugatePostDistribution}, B::SPNBuffer, id::Int)
    x = get(B, id)
    remove!(B.Z[node], id)

    remove_data!(node.dist, x)
end

"Remove data point from Distribution (intermediate node)"
function deepremove_data!(node::Node, B::SPNBuffer, id::Int)

    remove!(B.Z[node], id)

    for child in node.children
        deepremove_data!(child, B, id)
    end
end
