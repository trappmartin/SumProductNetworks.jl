export hasWeights, getWeights, setScope!, setObservations!, scope, obs, update!, classes, children, parents, length, add!, remove!, evaluate!

function hasWeights(node::SumNode)
    return true
end

function hasWeights(node::FiniteAugmentedProductNode)
    return true
end

function hasWeights(node::Node)
    return false
end

function getWeights(node::SumNode)
    return node.logweights
end

function getWeights(node::FiniteAugmentedProductNode)
    return node.logomega
end

function setScope!(node::SPNNode, scope::Vector{Int})
    if length(scope) > 0
        @assert maximum(scope) <= length(node.scopeVec)

        fill!(node.scopeVec, false)
        node.scopeVec[scope] = true
    end
end

function setScope!(node::SPNNode, scope::Int)
    @assert scope <= length(node.scopeVec)

    fill!(node.scopeVec, false)
    node.scopeVec[scope] = true
end

function addScope!(node::SPNNode, scope::Int)
    @assert scope <= length(node.scopeVec)
    node.scopeVec[scope] = true
end

function removeScope!(node::SPNNode, scope::Int)
    @assert scope <= length(node.scopeVec)
    node.scopeVec[scope] = false
end

function scope(node::SPNNode)
    return find(node.scopeVec)
end

function hasSubScope(node1::ProductNode, node2::SPNNode)
    return any(node1.scopeVec .& node2.scopeVec)
end

function setObservations!(node::Node, obs::Vector{Int})
    if length(obs) > 0
        @assert maximum(obs) <= length(node.obsVec)

        fill!(node.obsVec, false)
        node.obsVec[obs] = true
    end
end

function obs(node::Node)
    return find(node.obsVec)
end

function update!(node::Node)
    node.cids = Int[child.id for child in children(node)]
end

"""

classes(node) -> classlabels::Vector{Int}

Returns a list of class labels the Node is associated with.

##### Parameters:
* `node::FiniteProductNode`: node to be evaluated
"""
function classes(node::FiniteProductNode)

    classNodes = Vector{Int}(0)

    for classNode in filter(c -> isa(c, IndicatorNode), node.children)
        push!(classNodes, classNode.value)
    end

    for parent in node.parents
        classNodes = cat(1, classNodes, classes(parent))
    end

    return unique(classNodes)
end

function classes(node::SPNNode)

    classNodes = Vector{Int}(0)

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

"""
Evaluate Sum-Node on data.
This function updates the llh of the data under the model.
"""
function evaluate!(node::SumNode, data, llhvals)
    @simd for ii in 1:size(llhvals, 1)
        @inbounds llhvals[ii,node.id] = logsumexp(view(llhvals, ii, node.cids) + node.logweights)
    end

    if any(isnan.(llhvals[:,node.id]))
        ids = find(isnan.(llhvals[:,node.id]))
        println("cild llh: ", llhvals[ids, node.cids])
    end
    @assert !any(isnan.(llhvals[:,node.id])) "Found NaN in output, w: $(node.logweights)"
end

function evaluate!(node::ProductNode, data, llhvals)
    for k in 1:length(node)
        @simd for ii in 1:size(llhvals, 1)
            @inbounds llhvals[ii, node.id] += llhvals[ii, node.cids[k]] * hasSubScope(node, node.children[k])
        end
    end
    @assert !any(isnan.(llhvals[:,node.id]))
end

function evaluate!(node::IndicatorNode, data, llhvals)
    @inbounds idx = find(data[:, node.scopeVec] .!= node.value)
    @inbounds llhvals[idx, node.id] = -Inf32
    @assert !any(isnan.(llhvals[:,node.id]))
end

function evaluate!(node::NormalDistributionNode, data, llhvals)
    @simd for i in 1:size(data, 1)
        @inbounds llhvals[i, node.id] = normlogpdf(node.μ, node.σ, data[i, node.scope])
    end
end

function evaluate!{U}(node::UnivariateNode{U}, data, llhvals)
    @inbounds llhvals[:, node.id] = logpdf(node.dist, data[:, node.scope])
end

function evaluate!{U}(node::MultivariateNode{U}, data, llhvals)
    @inbounds llhvals[:, node.id] = logpdf(node.dist, data[:, node.scope]')
end

function evaluate!{U<:ConjugatePostDistribution}(node::UnivariateNode{U}, data, llhvals)
    @inbounds llhvals[:, node.id] = logpostpred(node.dist, data[:, node.scope])
end

function evaluate!{U<:ConjugatePostDistribution}(node::MultivariateNode{U}, data, llhvals)
    @inbounds llhvals[:, node.id] = logpostpred(node.dist, data[:, node.scope])
end
