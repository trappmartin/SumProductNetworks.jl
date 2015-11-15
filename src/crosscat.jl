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

"Add data point to Distribtion"
function deepadd_data!(node::MultivariateNode{ConjugatePostDistribution}, B::SPNBuffer, id::Int)

    x = get(B, id)
    add!(B.Z[node], id, 1)

    add_data!(node.dist, x)
end

"Add data point to Distribtion (intermediate node)"
function deepadd_data!(node::Node, B::SPNBuffer, id::Int)

    x = get(B, id)

    # compute map path to find distribution
    (mapval, path) = map(node, x)

    for key in keys(path)
        for c in path[key]
            if isa(c, Leaf)
                deepadd_data!(c, x)
            end
        end
    end

end

"Remove data point from Distribtion"
function deepremove_data!(node::MultivariateNode{ConjugatePostDistribution}, B::SPNBuffer, id::Int)
    x = get(B, id)
    remove!(B.Z[node], id)

    remove_data!(node.dist, x)
end

"Remove data point from Distribtion (intermediate node)"
function deepremove_data!(node::SumNode, B::SPNBuffer, id::Int)
    deepremove_data!(c, B, x)
end

"Remove data point from Distribtion (intermediate node)"
function deepremove_data!(node::ProductNode, B::SPNBuffer, id::Int)
    deepremove_data!(c, B, x)
end
