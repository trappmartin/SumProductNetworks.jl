export hasweights, weights
export setscope!, setobs!, addobs!, scope, obs
export nobs, nscope
export isnormalized
export addscope!
export removescope!
export hasscope, hasobs
export logweights
export updatescope!
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

getindex(node::Node, i...) = getindex(node.children, i...)

function setscope!(node::SPNNode, scope::Vector{Int})
    if length(scope) > 0

        if maximum(scope) > length(node.scopeVec)
            @warn "New scope is larger than original scope, resize node scope..."
            resize!(node.scopeVec, maximum(scope))
        end

        fill!(node.scopeVec, false)
        node.scopeVec[scope] .= true
    else
        fill!(node.scopeVec, false)
    end
end

function setscope!(node::SPNNode, scope::Int)

    if scope <= length(node.scopeVec)
        @warn "New scope is larger than original scope, resize node scope..."
        resize!(node.scopeVec, scope)
    end

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

@inline scope(node::Node) = findall(node.scopeVec)
@inline scope(node::Leaf) = node.scope
@inline nscope(node::Node) = sum(node.scopeVec)
@inline nscope(node::Leaf) = length(node.scope)
@inline hasscope(node::Node) = sum(node.scopeVec) > 0
@inline hasscope(node::Leaf) = true
hassubscope(node1::ProductNode, node2::SPNNode) = any(node1.scopeVec .& node2.scopeVec)

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

@inline nobs(node::Node) = sum(node.obsVec)
@inline obs(node::Node) = findall(node.obsVec)
@inline hasobs(node::Node) = any(node.obsVec)


"""
    updatescope!(spn)
Update the scope of all nodes in the SPN.
"""
function updatescope!(spn::SumProductNetwork)
    updatescope!(spn.root)
end

function updatescope!(node::SumNode)
    for child in children(node)
        updatescope!(child)
    end
    setscope!(node, scope(first(children(node))))
    return node
end

function updatescope!(node::ProductNode)
    for child in children(node)
        updatescope!(child)
    end
    setscope!(node, mapreduce(c -> scope(c), vcat, children(node)))
    return node
end

function updatescope!(node::Leaf)
    return node
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
    pid = findfirst(map(p -> p == parent, parent.children[index].parents))

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

function logpdf(
                spn::SumProductNetwork,
                x::AbstractMatrix{T};
                idx = Axis{:id}(collect(keys(spn))),
                llhvals = AxisArray(Matrix{Float32}(undef, size(x, 1), length(spn)), 1:size(x,1), idx)
              ) where T<:Real
    for layer in spn.layers
        Threads.@threads for node in layer
            logpdf!(node, x, llhvals)
        end
    end
    return llhvals[:,spn.root.id]
end

function logpdf(spn::SumProductNetwork, x::AbstractVector{T}) where T<:Real
    idx = Axis{:id}(collect(keys(spn)))
    llhvals = AxisArray(Vector{Float32}(undef, length(idx)), idx)
    
    # Call inplace function.
    for node in values(spn)
        logpdf!(node, x, llhvals)
    end
    return llhvals[spn.root.id]
end

function logpdf(node::Node, x::AbstractVector{T}) where T<:Real
    idx = Axis{:id}([n.id for n in getOrderedNodes(node)])
    llhvals = AxisArray(Vector{Float32}(undef, length(idx)), idx)
    
    # Call inplace function.
    logpdf!(node, x, llhvals)
    return llhvals[node.id]
end

function logpdf!(node::SumNode, x::AbstractMatrix{T}, llhvals::AxisArray) where T<:Real
    alpha = ones(Float32, size(x, 1)) * -Inf32
    r = zeros(Float32, size(x, 1))
    y = ones(Float32, size(x, 1)) * -Inf32
    for (i, child) in enumerate(children(node))
        if !isdefined(llhvals, child.id)
            logpdf!(child, x, llhvals)
        end

        y[:] .= llhvals[:, child.id] .+ Float32(logweights(node)[i])

        ind = .!isinf.(y)
        ind2 = y .<= alpha
        
        j = findall(ind .& ind2)
        r[j] .+= exp.(y[j] .- alpha[j])
        j = findall(ind .& .!ind2)
        r[j] .*= exp.(alpha[j] - y[j])
        r[j] .+= 1.f0
        alpha[j] .= y[j]
    end

    llhvals[:, node.id] = log.(r) .+ alpha
    return llhvals
end

function logpdf!(node::SumNode, x::AbstractVector{T}, llhvals::AxisArray) where T<:Real
    alpha = -Inf32
    r = 0.0f0
    y = -Inf32
    for (i, child) in enumerate(children(node))
        if !isdefined(llhvals, child.id)
            logpdf!(child, x, llhvals)
        end

        y = llhvals[child.id] + Float32(logweights(node)[i])

        if isinf(y)
            continue
        elseif y <= alpha
            r += exp(y - alpha)
        else
            r *= exp(alpha - y)
            r += 1.f0
            alpha = y
        end
    end

    llhvals[node.id] = log(r) + alpha
    return llhvals
end

function logpdf!(node::ProductNode, x::AbstractMatrix{T}, llhvals::AxisArray) where T<:Real
    r = zeros(Float32, size(x, 1))
    for child in filter(c -> hasscope(c), children(node))
        if !isdefined(llhvals, child.id)
            logpdf!(child, x, llhvals)
        end
        r .+= llhvals[:, child.id]
    end
    llhvals[:, node.id] .= r
    return llhvals
end

function logpdf!(node::ProductNode, x::AbstractVector{T}, llhvals::AxisArray) where T<:Real
    r = 0.0f0
    for child in filter(c -> hasscope(c), children(node))
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

function logpdf!(node::IndicatorNode, x::AbstractMatrix{T}, llhvals::AxisArray) where T<:Real
    llhvals[:, node.id] .= map(b -> b ? 0.f0 : -Inf32, x[:, node.scope] .== node.value)
    return llhvals
end

function logpdf!(node::IndicatorNode, x::AbstractVector{T}, llhvals::AxisArray) where T<:Real
    llhvals[node.id] = x[node.scope] == node.value ? zero(Float32) : -Inf32
    return llhvals
end

function logpdf!(node::UnivariateNode, x::AbstractMatrix{T}, llhvals::AxisArray) where T<:Real
    llhvals[:, node.id] .= convert(Vector{Float32}, logpdf.(node.dist, x[:, node.scope]))
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
