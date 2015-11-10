import Base.getindex

type DataAssignments

    # number of unique clusters
    c::Int

    # assignments
    Z::Vector{Int}

    # ids
    ids::Vector{Int}

    # children
    children::Vector{DataAssignments}

    # constructor
    DataAssignments(Z::Vector{Int}, ids::Vector{Int}) = new(Base.length(unique(Z)), Z, ids, DataAssignments[])

end

# functions on DataAssignments
call(p::DataAssignments, idx::Int) = p.Z[idx]
call(p::DataAssignments, ids::Vector{Int}) = p.Z[ids]

"Get index function for DataAssignments"
function getindex(p::DataAssignments, i::Int)
    p.ids[find(p.Z .== i)]
end

"delete assignment for DataAssignments"
function delete!(p::DataAssignments, id::Int)
    println(id)
end

type SPNBuffer

    # dimensionality of data
    D::Int
    N::Int

    # data indecies
    idx::Vector{Int}

    # data matrix
    X::AbstractArray{Float32}

    # assignment tree
    Z::DataAssignments

end

# functions on SPNBuffer
call(p::SPNBuffer, i::Int) = SPNBuffer(p.D, Base.length(p.Z[i]), collect(1:Base.length(p.Z[i])), sub(p.X, :, p.Z[i]), p.Z.children[i])

## utility functions

"Add data point to Distribtion"
function deepadd_data!{T<:Real}(node::SPN.MultivariateNode{ConjugatePostDistribution}, x::Array{T})
    add_data!(node.dist, x)
end

"Add data point to Distribtion (intermediate node)"
function deepadd_data!(node::SPN.Node, x::Array)
    # compute map path to find distribution
    (mapval, path) = SPN.map(node, x)

    for key in keys(path)
        for c in path[key]
            if isa(c, SPN.Leaf)
                deepadd_data!(c, x)
            end
        end
    end

end

"Remove data point from Distribtion"
function deepremove_data!(node::MultivariateNode{ConjugatePostDistribution}, B::SPNBuffer, id::Int)
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
