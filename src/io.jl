export exportNetwork

function getLabel(node::AbstractSumLayer)
    return "\"+\""
end

function getLabel(node::BayesianSumLayer)
    return "\"+\""
end

function getLabel(node::AbstractProductLayer)
    return "<&times;>"
end

function getLabel(node::BayesianProductLayer)
    return "<&times;>"
end

function getLabel(node::AbstractLeafLayer)
    return "L"
end

function getLabel(node::SumNode)
    return "\"+\""
end

function getLabel(node::ProductNode)
    return "<&times;>"
end

function getLabel(node::Leaf)
    return "\"X$(scope(node))\""
end

function getLabel(node::IndicatorNode)
    return "\"X$(scope(node)) = $(node.value)\""
end

function getShape(node::SPNNode)
    return "circle"
end

function getShape(node::IndicatorNode)
    return "doublecircle"
end

function getShape(node::SPNLayer)
    return "circle"
end

function getShape(node::AbstractLeafLayer)
    return "doublecircle"
end

function getFontSize(node::SPNNode)
    return 40
end

function getFontSize(node::IndicatorNode)
    return 14
end

function getParameters(node::SPNNode)
    return "nothing"
end

function getParameters(node::IndicatorNode)
    return "\"value=$(node.value)\""
end

function isDegenerated(node::Node, nodeObsDegeneration)
    return nodeObsDegeneration ? !(hasscope(node) && hasobs(node)) : !hasscope(node)
end

function isDegenerated(node::Leaf, nodeObsDegeneration)
    return all(map(p -> isDegenerated(p, nodeObsDegeneration), parents(node)))
end

function exportNetwork(spn::SumProductNetwork,
                       filename::AbstractString;
                       nodeObsDegeneration = false,
                       excludeDegenerated = false)

    gstring = ["digraph SPN {"]
    push!(gstring, "node [margin=0, width=0.7, fixedsize=true];")

    if excludeDegenerated
        @warn("Excluding degenerated nodes, these nodes cannot be recovered from the DOT file!")
    end

    for node in values(spn)
        label = getLabel(node)
        shape = getShape(node)
        fontsize = getFontSize(node)
        parameters = getParameters(node)
        nodeType = "\"$(string(typeof(node)))\""
        nodeScope = "\"$(string(scope(node)))\""
        nodeNumObs = isa(node, Node) ? string(sum(node.obsVec)) : "\"\""

        degeneratedNode = isDegenerated(node, nodeObsDegeneration)
        style = !degeneratedNode ? "" : "style=dotted,"

        if excludeDegenerated && degeneratedNode
            continue
        end

        nodestring = "$(spn.idx[node.id]) [label=$(label), shape=$(shape), $(style) fontsize=$(fontsize), "*
                        "nodeType=$(nodeType), nodeScope=$(nodeScope), nodeNumObs=$(nodeNumObs), "*
                        "nodeParameters=$(parameters)];"

        push!(gstring, nodestring)
    end

    for node in filter(n -> isa(n, Node), values(spn))
        for (ci, child) in enumerate(children(node))
            degeneratedChild = isDegenerated(child, nodeObsDegeneration)
            degeneratedNode = isDegenerated(node, nodeObsDegeneration)
            style = !degeneratedChild ? "" : "style=dotted,"

            if excludeDegenerated & (degeneratedNode || degeneratedChild)
                continue
            end

            if hasweights(node)
                w = weights(node)[ci]
                #logw = @sprintf("%0.4f", logw)
                w = @sprintf("%0.2f", w)
                push!(gstring, "$(spn.idx[node.id]) -> $(spn.idx[child.id]) [label=$(w), $(style) logWeight=\"$(logweights(node))\"];")

            else
                style = !degeneratedChild ? "" : "style=dotted"
                push!(gstring, "$(spn.idx[node.id]) -> $(spn.idx[child.id]) [$(style)];")
            end
        end
    end

    push!(gstring, "}")

    open(filename, "w") do f
        write(f, join(gstring, '\n'))
    end
end
