export exportNetwork

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
    return "\"X$(scope(node)) = $(node.value-1)\""
end

function getShape(node::SPNNode)
    return "circle"
end

function getShape(node::IndicatorNode)
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
    return nodeObsDegeneration ? ( isempty(scope(node)) | isempty(obs(node)) ) : isempty(scope(node))
end

function isDegenerated(node::Leaf, nodeObsDegeneration)
    return isDegenerated(parents(node)[1], nodeObsDegeneration)
end

function exportNetwork(root::Node, filename::String; nodeObsDegeneration = false, excludeDegenerated = false)

	gstring = ["digraph SPN {"]

    push!(gstring, "node [margin=0, width=0.7, fixedsize=true];")

    nodes = getOrderedNodes(root)

    if excludeDegenerated
        warn("Excluding degenerated nodes, these nodes cannot recovered from the DOT file!")
    end

    for node in nodes
        label = getLabel(node)
        shape = getShape(node)
        fontsize = getFontSize(node)
        parameters = getParameters(node)
        nodeType = "\"$(string(typeof(node)))\""
        nodeScope = "\"$(string(scope(node)))\""
        nodeNumObs = string(sum(node.obsVec))

        degeneratedNode = isDegenerated(node, nodeObsDegeneration)
        style = !degeneratedNode ? "" : "style=dotted,"

        if excludeDegenerated & degeneratedNode
            continue
        end

        nodestring = "$(node.id) [label=$(label), shape=$(shape), $(style) fontsize=$(fontsize), "*
                        "nodeType=$(nodeType), nodeScope=$(nodeScope), nodeNumObs=$(nodeNumObs), "*
                        "nodeParameters=$(parameters)];"

        push!(gstring, nodestring)
    end

    for node in filter(n -> isa(n, Node), nodes)
        for (ci, child) in enumerate(children(node))
            degeneratedChild = isDegenerated(child, nodeObsDegeneration)
            style = !degeneratedChild ? "" : "style=dotted,"

            if excludeDegenerated & degeneratedChild
                continue
            end

            if hasWeights(node)
                logw = getWeights(node)[ci]
                #logw = @sprintf("%0.4f", logw)
                w = @sprintf("%0.2f", exp(logw))

                push!(gstring, "$(node.id) -> $(child.id) [label=$(w), $(style) logWeight=\"$(logw)\"];")

            else
                style = !degeneratedChild ? "" : "style=dotted"
                push!(gstring, "$(node.id) -> $(child.id) [$(style)];")
            end
        end
    end

    push!(gstring, "}")

    open(filename, "w") do f
        write(f, join(gstring, '\n'))
    end
end
