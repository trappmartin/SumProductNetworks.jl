export simplify!, complexity, depth, prune!, getOrderedNodes, getOrderedLayers, copySPN, logsumexp

"""
    Fast and numerical stable computation of logsumexp.

    Parameters:
    * X: Data matrix

    Optional Parameters:
    * dim: dimension used to sum over in logsumexp

"""
function logsumexp(X::Matrix; dim = 1)

    T = typeof(first(X))
    odim = setdiff([1, 2], dim)[1]

    alpha = one(T) * map(T, -Inf)
    r = zeros(T, size(X, odim))

    @inbounds @fastmath for i in 1:size(X, odim)
        Xi = slicedim(X, odim, i)
        for j in 1:length(Xi)
            if isinf(Xi[j])
                continue
            elseif Xi[j] <= alpha
                r[i] += exp(Xi[j] - alpha)
            else
                r[i] *= exp(alpha - Xi[j])
                r[i] += one(T)
                alpha = Xi[j]
            end
        end
        r[i] = log(r[i]) + alpha
        alpha = one(T) * map(T, -Inf)
    end

    return r
end

"""
Get all nodes in topological order using Tarjan's algoritm.
"""
function getOrderedNodes(root)
    visitedNodes = Vector{SPNNode}()
    visitNode!(root, visitedNodes)
    return visitedNodes
end

function getOrderedLayers(root)
    visitedLayers = Vector{SPNLayer}()
    visitLayer!(root, visitedLayers)
    return visitedLayers
end

function visitNode!(node::Node, visitedNodes)
    # check if we have already visited this node
    if !(node in visitedNodes)

        # visit node
        for child in children(node)
            visitNode!(child, visitedNodes)
        end

        push!(visitedNodes, node)
    end
end
visitNode!(node::Leaf, visitedNodes) = push!(visitedNodes, node)

function visitLayer!(layer::AbstractInternalLayer, visitedLayers)
    # check if we have already visited this layer
    if !(layer in visitedLayers)

        # visit layer
        for child in children(layer)
            visitLayer!(child, visitedLayers)
        end

        push!(visitedLayers, layer)
    end
end
visitLayer!(layer::AbstractLeafLayer, visitedLayers) = push!(visitedLayers, layer)

"""

depth(S)

Compute the depth of the SPN rooted at S.
"""
depth(S::Node) = maximum(ndepth(child, 1) for child in children(S))
depth(S::Leaf) = 0

function ndepth(S::Node, d::Int)
    return maximum(ndepth(child, d+1) for child in children(S))
end

function ndepth(S::Leaf, d::Int)
    return d
end

"""

complexity(S)

Compute the complexity (number of free parameters) of the SPN rooted at S.
"""
function complexity(S)
    return sum(map(n -> length(n), filter(n -> isa(n, SumNode), order(S))))
end

"""

simplify!

Simplify the structure of an SPN.
"""
function simplify!(S::FiniteSumNode)

    for child in children(S)
        simplify!(child)
    end

    childrentoremove = Int[]

    for (i, child) in enumerate(children(S))
        if isa(child, SumNode) & (length(parents(child)) == 1)
            # collaps child if its a sum
            toremove = Int[]
            for (j, k) in enumerate(children(child))
                add!(S, k, child.logweights[j] + S.logweights[i])
                push!(toremove, j)
            end

            for k in reverse(toremove)
                remove!(child, k)
            end

            push!(childrentoremove, i)
        elseif isa(child, ProductNode) & (length(parents(child)) == 1) & (length(child) == 1)
            # collaps child if its a product over one child
            add!(S, child.children[1], S.logweights[i])
            remove!(child, 1)
            push!(childrentoremove, i)
        end
    end

    for child in children(S)
        @assert findfirst(S .== child.parents) > 0
    end

    for child in reverse(childrentoremove)
        remove!(S, child)
    end

    for child in children(S)
        @assert findfirst(S .== child.parents) > 0
    end
end

function simplify!(S::FiniteProductNode)

    for child in children(S)
        simplify!(child)
    end

    childrentoremove = Int[]

    for (i, child) in enumerate(children(S))
        if isa(child, ProductNode) & (length(parents(child)) == 1)
            # collaps child if its a product
            toremove = Int[]
            for (j, k) in enumerate(children(child))
                add!(S, k)
                push!(toremove, j)
            end

            for k in reverse(toremove)
                remove!(child, k)
            end

            push!(childrentoremove, i)
        elseif isa(child, SumNode) & (length(parents(child)) == 1) & (length(child) == 1)
            # collaps child if its a sum over one child
            add!(S, child.children[1])
            remove!(child, 1)
            push!(childrentoremove, i)
        end
    end

    for child in reverse(childrentoremove)
        remove!(S, child)
    end

    for child in children(S)
        @assert findfirst(S .== child.parents) > 0
    end
end

function simplify!(S)
end

"""

prune!(S, σ)

Prune away leaf nodes & sub-trees with std lower than σ.
"""
function prune!(S::FiniteSumNode, σ::Float64)

    for node in filter(n -> isa(n, SumNode), order(S))

        toremove = Int[]

        for (ci, child) in enumerate(children(node))
            if isa(child, NormalDistributionNode)
                if child.σ < σ
                    push!(toremove, ci)
                end
            elseif isa(child, ProductNode)
                if any([isa(childk, NormalDistributionNode) for childk in children(child)])
                    drop = false
                    for childk in children(child)
                        if isa(childk, NormalDistributionNode)
                            if childk.σ < σ
                                drop = true
                            end
                        end
                    end
                    if drop
                        push!(toremove, ci)
                    end
                end
            end
        end
        reverse!(toremove)

        for ci in toremove
            remove!(node, ci)
        end
    end

    for node in filter(n -> isa(n, Node), order(S))

        toremove = Int[]
        for (ci, child) in enumerate(children(node))
            if isa(child, Node)
                if length(child) == 0
                    push!(toremove, ci)
                end
            end
        end
        reverse!(toremove)

        for ci in toremove
            remove!(node, ci)
        end
    end

end

"""

prune!(S, σ)

Prune away leaf nodes & sub-trees with std lower than σ.
"""
function prune_llh!(S::FiniteSumNode, data::AbstractArray; minrange = 0.0)

    nodes = order(S)

    maxId = maximum(Int[node.id for node in nodes])
    llhval = Matrix{Float64}(size(data, 1), maxId)

    for node in nodes
        eval!(node, data, llhval)
    end

    llhval -= maximum(vec(mean(llhval, 1)))

    rd = minrange + (rand(maxId) * (1-minrange))

    drop = rd .> exp(vec(mean(llhval, 1)))

    for node in filter(n -> isa(n, Node), order(S))

        toremove = Int[]

        for (ci, child) in enumerate(children(node))
            if isa(child, ProductNode)
                if any([isa(childk, NormalDistributionNode) for childk in children(child)])
                    for childk in children(child)
                        if drop[childk.id]
                            drop[child.id] = true
                        end
                    end
                end
            end

            if drop[child.id]
                push!(toremove, ci)
            end
        end

        reverse!(toremove)

        for ci in toremove
            remove!(node, ci)
        end
    end

    for node in filter(n -> isa(n, Node), order(S))

        toremove = Int[]
        for (ci, child) in enumerate(children(node))
            if isa(child, Node)
                if length(child) == 0
                    push!(toremove, ci)
                end
            end
        end
        reverse!(toremove)

        for ci in toremove
            remove!(node, ci)
        end
    end

end

function prune_uniform!(S::FiniteSumNode, p::Float64)

    nodes = order(S)
    maxId = maximum(Int[node.id for node in nodes])

    drop = rand(maxId) .> p

    for node in filter(n -> isa(n, Node), order(S))

        toremove = Int[]

        for (ci, child) in enumerate(children(node))
            if isa(child, ProductNode)
                if any([isa(childk, NormalDistributionNode) for childk in children(child)])
                    for childk in children(child)
                        if drop[childk.id]
                            drop[child.id] = true
                        end
                    end
                end
            end

            if drop[child.id]
                push!(toremove, ci)
            end
        end

        reverse!(toremove)

        for ci in toremove
            remove!(node, ci)
        end
    end

    for node in filter(n -> isa(n, Node), order(S))

        toremove = Int[]
        for (ci, child) in enumerate(children(node))
            if isa(child, Node)
                if length(child) == 0
                    push!(toremove, ci)
                end
            end
        end
        reverse!(toremove)

        for ci in toremove
            remove!(node, ci)
        end
    end

end

function copySPN(source::FiniteSumNode; idIncrement = 0)

    nodes = order(source)
    destinationNodes = Vector{SPNNode}()
    id2index = Dict{Int, Int}()

    for node in nodes

        if isa(node, NormalDistributionNode)
            dnode = NormalDistributionNode(copy(node.id) + idIncrement, copy(node.scope))
            dnode.μ = copy(node.μ)
            dnode.σ = copy(node.σ)
            push!(destinationNodes, dnode)
            id2index[dnode.id] = length(destinationNodes)
        elseif isa(node, IndicatorNode)
            dnode = IndicatorNode(copy(node.id) + idIncrement, copy(node.value), copy(node.scope))
            push!(destinationNodes, dnode)
            id2index[dnode.id] = length(destinationNodes)
        elseif isa(node, SumNode)
            dnode = SumNode(copy(node.id) + idIncrement, scope = copy(node.scope))
            cids = Int[child.id for child in children(node)]
            for (i, cid) in enumerate(cids)
                add!(dnode, destinationNodes[id2index[cid + idIncrement]], copy(node.weights[i]))
            end
            push!(destinationNodes, dnode)
            id2index[dnode.id] = length(destinationNodes)
        elseif isa(node, ProductNode)
            dnode = ProductNode(copy(node.id) + idIncrement, scope = copy(node.scope))
            cids = Int[child.id for child in children(node)]
            for (i, cid) in enumerate(cids)
                add!(dnode, destinationNodes[id2index[cid + idIncrement]])
            end

            push!(destinationNodes, dnode)
            id2index[dnode.id] = length(destinationNodes)

        else
            throw(TypeError(node, "Node type not supported."))
            end

        end

        return destinationNodes[end]
    end
