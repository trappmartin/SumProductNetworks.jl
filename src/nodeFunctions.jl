export hasweights, weights, setscope!, setobs!, addobs!, scope, obs, isnormalized
export addscope!
export removescope!
export hasscope
export logweights
export classes, children, parents, length, add!, remove!, logpdf!, logpdf

function isnormalized(node::Node)
    if !hasweights(node)
        return mapreduce(child -> isnormalized(child), &, children(node))
    else
        return sum(weights(node)) ≈ 1.0
    end
end
isnormalized(node::Leaf) = true

hasweights(node::SumNode) = true
hasweights(node::FiniteAugmentedProductNode) = true
hasweights(node::Node) = false

weights(node::SumNode) = exp.(node.logweights)
logweights(node::SumNode) = node.logweights
weights(node::FiniteAugmentedProductNode) = exp.(node.logomega)
logweights(node::FiniteAugmentedProductNode) = node.logomega

function setscope!(node::SPNNode, scope::Vector{Int})
    if length(scope) > 0
        @assert maximum(scope) <= length(node.scopeVec)

        fill!(node.scopeVec, false)
        node.scopeVec[scope] .= true
    else
        fill!(node.scopeVec, false)
    end
end

function setscope!(node::SPNNode, scope::Int)
    @assert scope <= length(node.scopeVec)

    fill!(node.scopeVec, false)
    node.scopeVec[scope] = true
end

function addscope!(node::Node, scope::Int)
    @assert scope <= length(node.scopeVec)
    node.scopeVec[scope] = true
end

function removescope!(node::Node, scope::Int)
    @assert scope <= length(node.scopeVec)
    node.scopeVec[scope] = false
end

function scope(node::SPNNode)
    return findall(node.scopeVec)
end

function hasscope(node::Node)
    return sum(node.scopeVec) > 0
end
hasscope(node::Leaf) = true


function hasSubScope(node1::ProductNode, node2::SPNNode)
    return any(node1.scopeVec .& node2.scopeVec)
end

function addobs!(node::SPNNode, obs::Int)
    @assert obs <= length(node.obsVec)
    node.obsVec[obs] = true
end

function setobs!(node::Node, obs::AbstractVector{Int})
    if length(obs) > 0
        @assert maximum(obs) <= length(node.obsVec)

        fill!(node.obsVec, false)
        node.obsVec[obs] .= true
    else
        fill!(node.obsVec, false)
    end
end

obs(node::Node) = findall(node.obsVec)
hasobs(node::Node) = any(node.obsVec)


"""

classes(node) -> classlabels::Vector{Int}

Returns a list of class labels the Node is associated with.

##### Parameters:
* `node::FiniteProductNode`: node to be evaluated
"""
function classes(node::FiniteProductNode)

    classNodes = Vector{Int}()

    for classNode in filter(c -> isa(c, IndicatorNode), node.children)
        push!(classNodes, classNode.value)
    end

    for parent in node.parents
        classNodes = cat(1, classNodes, classes(parent))
    end

    return unique(classNodes)
end

function classes(node::SPNNode)

    classNodes = Vector{Int}()

    for parent in node.parents
        classNodes = cat(1, classNodes, classes(parent))
    end

    return unique(classNodes)
end

"""

children(node) -> children::SPNNode[]

Returns the children of an internal node.

##### Parameters:
* `node::Node`: Internal SPN node to be evaluated.
"""
function children(node::Node)
    node.children
end

"""

parents(node) -> parents::SPNNode[]

Returns the parents of an SPN node.

##### Parameters:
* `node::SPNNode`: SPN node to be evaluated.
"""
function parents(node::SPNNode)
    node.parents
end

"""
Add a node to a finite sum node with given weight in place.
add!(node::FiniteSumNode, child::SPNNode, weight<:Real)
"""
function add!(parent::SumNode, child::SPNNode, logweight::T) where T <: Real
    if !(child in parent.children)
        push!(parent.children, child)
        push!(parent.logweights, logweight)
        push!(child.parents, parent)
    end
end

function add!(parent::ProductNode, child::SPNNode)
    if !(child in parent.children)
        push!(parent.children, child)
        push!(child.parents, parent)
    end
end

function add!(parent::FiniteAugmentedProductNode, child::SPNNode, logomega::Float32)
    if !(child in parent.children)
        push!(parent.children, child)
        push!(child.parents, parent)
        push!(parent.logomega, logomega)
    end
end

"""
Remove a node from the children list of a sum node in place.
remove!(node::FiniteSumNode, index::Int)
"""
function remove!(parent::SumNode, index::Int)
    pid = findfirst(parent .== parent.children[index].parents)
    @assert pid > 0 "Could not find parent ($(node.id)) in list of parents ($(parent.children[index].parents))!"
    deleteat!(parent.children[index].parents, pid)
    deleteat!(parent.children, index)
    deleteat!(parent.logweights, index)
end

function remove!(parent::ProductNode, index::Int)
    pid = findfirst(parent .== parent.children[index].parents)
    @assert pid > 0 "Could not find parent ($(node.id)) in list of parents ($(parent.children[index].parents))!"
    deleteat!(parent.children[index].parents, pid)
    deleteat!(parent.children, index)
end

function length(node::Node)
    Base.length(node.children)
end

function logpdf(node::Node, x::AbstractVector{T}) where T<:Real
    idx = Axis{:id}([n.id for n in getOrderedNodes(node)])
    llhvals = AxisArray(Vector{Float32}(undef, length(idx)), idx)
    
    # Call inplace function.
    logpdf!(node, x, llhvals)
    return llhvals[node.id]
end

"""
Evaluate Sum-Node on data.
This function updates the llh of the data under the model.
"""
function logpdf!(node::SumNode, x::AbstractVector{T}, llhvals::AxisArray) where T<:Real
    alpha = -Inf
    r = 0.0
    y = -Inf32
    for (i, child) in enumerate(children(node))
        if !isdefined(llhvals, child.id)
            logpdf!(child, x, llhvals)
        end

        y = llhvals[child.id] + logweights(node)[i]

        if isinf(y)
            continue
        elseif y <= alpha
            r += exp(y - alpha)
        else
            r *= exp(alpha - y)
            r += 1.
            alpha = y
        end
    end

    llhvals[node.id] = log(r) + alpha
    return llhvals
end

function logpdf!(node::ProductNode, x::AbstractVector{T}, llhvals::AxisArray) where T<:Real
    r = 0.0
    for child in children(node)
        if !isdefined(llhvals, child.id)
            logpdf!(child, x, llhvals)
        end
        r += llhvals[child.id]
    end
    llhvals[node.id] = r
    return llhvals
end

function logpdf(node::Leaf, x::AbstractVector{T}) where T<:Real
    llhvals = AxisArray(Vector{Float32}(undef, 1), Axis{:id}([node.id]))
    logpdf!(node, x, llhvals)
    return llhvals[node.id]
end

function logpdf!(node::IndicatorNode, x::AbstractVector{T}, llhvals::AxisArray) where T<:Real
    llhvals[node.id] = x[node.scope] == node.value ? zero(Float32) : -Inf32
    return llhvals
end

function logpdf!(node::UnivariateNode, x::AbstractVector{T}, llhvals::AxisArray) where T<:Real
    llhvals[node.id] = convert(Float32, logpdf(node.dist, x[node.scope]))
    return llhvals
end

function logpdf!(node::MultivariateNode, x::AbstractVector{T}, llhvals::AxisArray) where T<:Real
    llhvals[node.id] = convert(Float32, logpdf(node.dist, x[node.scope]))
    return llhvals
end
