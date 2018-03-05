"""

llh(S, data) -> logprobvals::Vector{T}

"""
function llh(S::Node, data, nodes, maxID)
    llhval = Matrix{Float32}(size(data, 1), maxID)

    fill!(llhval, Float32(0.))

    for node in nodes
        evaluate!(node, data, llhval)
    end

    return llhval[:, S.id]
end

function llh(S::Node, data, nodes)
    maxID = maximum(node.id for node in nodes)
    return llh(S, data, nodes, maxID)
end

function llh(S::Node, data)
    nodes = getOrderedNodes(S)
    return llh(S, data, nodes)
end

"""

normalizeSPN!(S)

Localy normalize the weights of a SPN using Algorithm 1 from Peharz et al.

##### Parameters:
* `node::FiniteSumNode`: Sum Product Network

##### Optional Parameters:
* `ϵ::Float64`: Lower bound to ensure we don't devide by zero. (default 1e-10)
"""
function normalizeSPN!(S::Node; ϵ = 1e-10)

    nodes = order(S)
    αp = ones(length(nodes))

    for (nid, node) in enumerate(nodes)

        if isa(node, Leaf)
            continue
        end

        α = 0.0

        if isa(node, FiniteSumNode)
            α = sum(exp.(node.logweights))

            if α < ϵ
                α = ϵ
            end
            node.logweights[:] .-= log(α)
            node.logweights[exp.(node.logweights) .< ϵ] = ϵ

        elseif isa(node, FiniteProductNode)
            α = αp[nid]
            αp[nid] = 1
        end

        for fnode in parents(node)

            if isa(fnode, FiniteSumNode)
                id = findfirst(children(fnode) .== node)
                @assert id > 0
                fnode.logweights[id] = fnode.logweights[id] + log(α)
            elseif isa(fnode, FiniteProductNode)
                id = findfirst(nodes .== fnode)
                if id == 0
                    println("parent of the following node not found! ", nid)
                end
                @assert id > 0
                αp[id] = α * αp[id]
            end
        end
    end
end
