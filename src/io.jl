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
    return "\"X$(scope(node)) = $(node.value-1)\""
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
        @warn("Excluding degenerated nodes, these nodes cannot be recovered from the DOT file!")
    end

    for node in nodes
        label = getLabel(node)
        shape = getShape(node)
        fontsize = getFontSize(node)
        parameters = getParameters(node)
        nodeType = "\"$(string(typeof(node)))\""
        nodeScope = "\"$(string(scope(node)))\""
        nodeNumObs = isa(node, Node) ? string(sum(node.obsVec)) : "\"\""

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

function exportNetwork(root::SPNLayer, dot_filename::String, param_filename::String; nodeObsDegeneration = false, excludeDegenerated = false)

	gstring = ["digraph SPN {"]

    push!(gstring, "node [margin=0.5, width=0.7, fixedsize=true];")
    push!(gstring, "compound=true;")

    layers = getOrderedLayers(root)

    if excludeDegenerated
        @warn("Excluding degenerated nodes, these nodes cannot be recovered from the DOT file!")
    end

    for (li, layer) in enumerate(layers)

        label = getLabel(layer)
        shape = getShape(layer)
        #fontsize = getFontSize(node)
        #parameters = getParameters(node)
        layer_type = "\"$(string(typeof(layer)))\""
        #nodeScope = "\"$(string(scope(node)))\""
        #nodeNumObs = isa(node, Node) ? string(sum(node.obsVec)) : "\"\""

        #degeneratedNode = isDegenerated(node, nodeObsDegeneration)
        #style = !degeneratedNode ? "" : "style=dotted,"
        style = ""

        #if excludeDegenerated & degeneratedNode
        #    continue
        #end

        if isa(layer, AbstractInternalLayer)

            (C, Ch) = size(layer)

            layer_string = ["subgraph cluster$(li) { "]

            nodestring = "$(layer.ids[1]) [label=$(label), shape=$(shape), $(style) xlabel=$(layer.ids[1])];"
            push!(layer_string, nodestring)

            if C > 1
                if C > 2
                    nodestring = "$(layer.ids[2]) [shape=point, width=0.1];"
                    push!(layer_string, nodestring)
                end

                nodestring = "$(layer.ids[end]) [label=$(label), shape=$(shape), $(style) xlabel=$(layer.ids[end])];"
                push!(layer_string, nodestring)
            end

            push!(layer_string, "label = \"$(string(typeof(layer))) - [$(li)]\";")
            push!(layer_string, "number_of_nodes = \"C\";")
            push!(layer_string, "number_of_children = \"Ch\";")

            push!(layer_string, "}")
            append!(gstring, layer_string)

            # construct connections
            childids = cids(layer)

            if C > 1
                push!(gstring, "$(layer.ids[1]) -> $(childids[1, 1]) [ltail=cluster$(li),lhead=cluster$(li-1)];")
                push!(gstring, "$(layer.ids[end]) -> $(childids[end, end]) [ltail=cluster$(li),lhead=cluster$(li-1)];")
            else
                push!(gstring, "$(layer.ids[1]) -> $(childids[2, 1]) [ltail=cluster$(li),lhead=cluster$(li-1)];")
            end

            jldopen(string(param_filename, "_layer$(li).jld2"), "w") do file
                file["layer_id"] = li
                file["ids"] = layer.ids
                file["child_ids"] = cids(layer)
                file["parameters"] = parameters(layer)
            end

        else

            (C, _) = size(layer)

            layer_string = ["subgraph cluster$(li) { "]

            nodestring = "$(layer.ids[1]) [label=$(label), shape=$(shape), $(style) xlabel=$(layer.ids[1])];"
            push!(layer_string, nodestring)

            nodestring = "$(layer.ids[end]) [label=$(label), shape=$(shape), $(style) xlabel=$(layer.ids[end])];"
            push!(layer_string, nodestring)

            push!(layer_string, "label = \"$(string(typeof(layer))) - [$(li)]\";")
            push!(layer_string, "number_of_nodes = \"C\";")

            push!(layer_string, "}")
            append!(gstring, layer_string)

            jldopen(string(param_filename, "_layer$(li).jld2"), "w") do file
                file["layer_id"] = li
                file["ids"] = layer.ids
                file["scopes"] = layer.scopes
                file["parameters"] = parameters(layer)
            end

        end



    end

    push!(gstring, "}")

    open(string(dot_filename, ".dot"), "w") do f
        write(f, join(gstring, '\n'))
    end
end
