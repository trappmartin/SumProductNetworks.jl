export normalize!

"""
    normalize!(S)

Localy normalize the weights of a SPN using Algorithm 1 from Peharz et al.
"""
function normalize!(spn::SumProductNetwork; ϵ = 1e-10)
    αp = ones(length(spn))

    for (nid, node) in enumerate(values(spn))

        if isa(node, Leaf)
            continue
        end

        α = 0.0

        if isa(node, SumNode)
            α = Float64(sum(weights(node)))
            if α < ϵ
                α = ϵ
            end
            T = eltype(node)
            logweights(node)[:] .-= T(log(α))
            logweights(node)[weights(node) .< ϵ] .= T(ϵ)
        elseif isa(node, ProductNode)
            α = αp[nid]
            αp[nid] = 1.
        end

        for fnode in parents(node)
            if isa(fnode, SumNode)
                id = findfirst(children(fnode) .== node)
                @assert id > 0
                T = eltype(fnode)
                logweights(fnode)[id] = logweights(fnode)[id] + T(log(α))
            elseif isa(fnode, ProductNode)
                id = findfirst(nodes .== fnode)
                if id == 0
                    @error("Parent of the following node not found! ", nid)
                end
                @assert id > 0
                αp[id] = α * αp[id]
            end
        end
    end
end
