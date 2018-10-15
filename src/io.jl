export export_network

getLabel(node::SumNode) = "\"+\""
getLabel(node::ProductNode) = "<&times;>"
getLabel(node::Leaf) = "\"X$(scope(node))\""
getLabel(node::IndicatorNode) = "\"X$(scope(node)) = $(node.value)\""

getShape(node::SPNNode) = "circle"
getShape(node::IndicatorNode) = "doublecircle"

getFontSize(node::SPNNode) = 40
getFontSize(node::IndicatorNode) = 14

getParameters(node::SPNNode) = "nothing"
getParameters(node::IndicatorNode) = "\"value=$(node.value)\""

function isDegenerated(node::Node, nodeObsDegeneration)
    return nodeObsDegeneration ? !(hasscope(node) && hasobs(node)) : !hasscope(node)
end

function isDegenerated(node::Leaf, nodeObsDegeneration)
    return all(map(p -> isDegenerated(p, nodeObsDegeneration), parents(node)))
end

function export_network(spn::SumProductNetwork,
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
