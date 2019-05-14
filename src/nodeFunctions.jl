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
export rand

function isnormalized(node::Node)
    if !hasweights(node)
        return mapreduce(child -> isnormalized(child), &, children(node))
    else
        return sum(weights(node)) ≈ 1.0
    end
end
isnormalized(node::Leaf) = true

hasweights(node::SumNode) = true
hasweights(node::Node) = false

weights(node::SumNode) = exp.(node.logweights)
logweights(node::SumNode) = node.logweights

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

    if scope > length(node.scopeVec)
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
@inline hassubscope(n1::ProductNode, n2::SPNNode) = any(n1.scopeVec .& n2.scopeVec)

function addobs!(node::SPNNode, obs::Int)
    @assert obs <= length(node.obsVec)
    node.obsVec[obs] = true
end

function setobs!(node::Node, obs::AbstractVector{Int})
    if length(obs) > 0
        if maximum(obs) > length(node.obsVec)
            @warn "New obs is larger than original obs field, resize node obs..."
            resize!(node.obsVec, maximum(obs))
        end

        fill!(node.obsVec, false)
        node.obsVec[obs] .= true
    else
        fill!(node.obsVec, false)
    end
end

function setobs!(node::Node, obs::Int)
    if obs > length(node.obsVec)
        @warn "New obs is larger than original obs field, resize node obs..."
        resize!(node.obsVec, obs)
    end

    fill!(node.obsVec, false)
    node.obsVec[obs] = true
end

@inline nobs(node::Node) = sum(node.obsVec)
@inline obs(node::Node) = findall(node.obsVec)
@inline hasobs(node::Node) = any(node.obsVec)

"""
    updatescope!(spn)
Update the scope of all nodes in the SPN.
"""
updatescope!(spn::SumProductNetwork) = updatescope!(spn.root)

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

updatescope!(node::Leaf) = node

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

function logpdf!(spn::SumProductNetwork, X::AbstractMatrix{<:Real}, llhvals::AxisArray)

    fill!(llhvals, -Inf)

    # Call inplace functions.
    for layer in spn.layers
        Threads.@threads for node in layer
            logpdf!(node, X, llhvals)
        end
    end

    spn.info[:llh] = mean(llhvals[:,spn.root.id])

    return llhvals[:,spn.root.id]
end

function logpdf(spn::SumProductNetwork, X::AbstractMatrix{<:Real})
    idx = Axis{:id}(collect(keys(spn)))
    llhvals = AxisArray(Matrix{Float64}(undef, size(X, 1), length(idx)), 1:size(X, 1), idx)
    return logpdf!(spn, X, llhvals)
end

function logpdf!(spn::SumProductNetwork, x::AbstractVector{<:Real}, llhvals::AxisArray)

    fill!(llhvals, -Inf)

    # Call inplace functions.
    for layer in spn.layers
        Threads.@threads for node in layer
            logpdf!(node, x, llhvals)
        end
    end

    spn.info[:llh] = llhvals[spn.root.id]

    return llhvals[spn.root.id]
end

function logpdf(spn::SumProductNetwork, x::AbstractVector{<:Real})
    idx = Axis{:id}(collect(keys(spn)))
    llhvals = AxisArray(Vector{Float64}(undef, length(idx)), idx)
    return logpdf!(spn, x, llhvals)
end

"""
    logpdf(n::SumNode, x)

Compute the logpdf of a sum node.
"""
logpdf(n::SumNode, x::AbstractArray{<:Real}; lw::AbstractVector{<:Real}=logweights(n)) = n(x, lw=lw)

function logpdf!(n::SumNode, x::AbstractVector{<:Real}, llhvals::AxisArray{U,1}) where {U<:Real}
    @inbounds y = map(c -> llhvals[c.id], children(n))
    llhvals[n.id] = map(U, _logpdf(n, x, y, logweights(n)))
    return llhvals
end

function logpdf!(n::SumNode, x::AbstractMatrix{<:Real}, llhvals::AxisArray{U,2}) where {U<:Real}
    @inbounds Y = mapreduce(c -> llhvals[:, c.id], hcat, children(n))
    @inbounds llhvals[:,n.id] = map(i -> map(U, _logpdf(n, x, view(Y, i, :), logweights(n))), 1:size(x,1))
    return llhvals
end

# ############################################ #
# Concrete implementation for finite sum node. #
# ############################################ #
function (n::FiniteSumNode)(x::AbstractArray{<:Real}, y::AbstractVector{<:Real}; lw::AbstractVector{<:Real}=logweights(n))
    return _logpdf(n, x, y, lw)
end

function _logpdf(n::SumNode, x::AbstractArray{<:Real}, y::AbstractVector{<:Real}, lw::AbstractVector{<:Real})
    l = lw+y
    m = maximum(l)
    lp = log(mapreduce(v -> exp(v-m), +, l)) + m
    return isfinite(lp) ? lp : -Inf
end

function (n::FiniteSumNode)(x::AbstractVector{<:Real}; lw::AbstractVector{<:Real} = logweights(n))
    y = map(c -> c(x), children(n))
    return _logpdf(n, x, y, lw)
end

function (n::FiniteSumNode)(x::AbstractMatrix{<:Real}; lw::AbstractVector{<:Real}=logweights(n))
    @inbounds Y = mapreduce(i -> map(c -> c(view(x, i, :)), children(n)), hcat, 1:size(x,1))
    return @inbounds map(i -> _logpdf(n, x, view(Y, :, i), lw), 1:size(x,1))
end

"""
    logpdf(n::ProductNode, x)

Compute the logpdf of a product node.
"""
logpdf(n::ProductNode, x::AbstractArray{<:Real}) = n(x)
(n::FiniteProductNode)(x::AbstractArray{<:Real}, y::AbstractVector{<:Real}) = _logpdf(n, x, y)
function (n::FiniteProductNode)(x::AbstractVector{<:Real})
    y = map(c -> c(x), children(n))
    return _logpdf(n, x, y)
end

function (n::FiniteProductNode)(x::AbstractMatrix{<:Real})
    @inbounds Y = mapreduce(i -> map(c -> c(view(x, i, :)), children(n)), hcat, 1:size(x,1))
    return _logpdf(n, x, Y)
end

function _logpdf(n::ProductNode, x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    if !hasscope(n)
        return 0.0
    else
        return mapreduce(k -> hasscope(n[k]) ? y[k] : 0.0, +, 1:length(n))
    end
end

function _logpdf(n::ProductNode, x::AbstractMatrix{<:Real}, y::AbstractMatrix{<:Real})
    N = size(x,1)
    if !hasscope(n)
        return zeros(N)
    else
        return mapreduce(k -> hasscope(n[k]) ? y[:,k] : zeros(N), +, 1:length(n))
    end
end

function logpdf!(n::ProductNode, x::AbstractVector{<:Real}, llhvals::AxisArray{U}) where {U<:Real}
    @inbounds y = map(c -> llhvals[c.id], children(n))
    llhvals[n.id] = map(U, _logpdf(n, x, y))
    return llhvals
end

function logpdf!(n::ProductNode, x::AbstractMatrix{<:Real}, llhvals::AxisArray{U}) where {U<:Real}
    @inbounds Y = mapreduce(c -> llhvals[:, c.id], hcat, children(n))
    llhvals[:,n.id] = map(U, _logpdf(n, x, Y))
    return llhvals
end

# ################## #
# Leaf distributions #
# ################## #
"""
    logpdf(n::Leaf, x)

Compute the logpdf of a leaf node.
"""
function logpdf!(n::Leaf, x::AbstractVector{<:Real}, llhvals::AxisArray{U}) where {U<:Real}
    llhvals[n.id] = map(U, n(x))
    return llhvals
end

function logpdf!(n::Leaf, x::AbstractMatrix{<:Real}, llhvals::AxisArray{U}) where {U<:Real}
    llhvals[:,n.id] .= map(U, n(x))
    return llhvals
end

@inline (n::IndicatorNode)(x::AbstractVector{<:Real}) = n(x, n.value)
@inline (n::IndicatorNode)(x::AbstractVector{<:Real}, θ::T) where {T<:Real} = _logpdf(n, x, θ)

function _logpdf(n::IndicatorNode, x::AbstractVector{<:Real}, θ::T) where {T<:Real}
    return isnan(x[scope(n)]) ? 0.0 : (x[scope(n)] == θ ? 0.0 : -Inf)
end

@inline (n::UnivariateNode)(x::AbstractVector{<:Real}) = n(x, Distributions.params(n.dist)...)
@inline (n::UnivariateNode)(x::AbstractVector{<:Real}, θ...) = _logpdf(n, x, θ...)

function _logpdf(n::UnivariateNode, x::AbstractVector{<:Real}, θ...)
    if isnan(x[scope(n)])
        return 0.0
    end

    if θ == params(n)
        return logpdf(n.dist, x[scope(n)])
    else
        return logpdf((typeof(n.dist))(θ...), x[scope(n)])
    end
end

@inline (n::MultivariateNode)(x::AbstractVector{<:Real}) = n(x, Distributions.params(n.dist)...)
@inline (n::MultivariateNode)(x::AbstractVector{<:Real}, θ...) = _logpdf(n, x, θ...)

function _logpdf(n::MultivariateNode, x::AbstractVector{<:Real}, θ...)

    @assert !any(isnan, x[scope(n)]) "NaN values for multivariate leaves is not supported."

    if θ == params(n)
        return logpdf(n.dist, x[scope(n)])
    else
        return logpdf((typeof(n.dist))(θ...), x[scope(n)])
    end
end

(n::IndicatorNode)(x::AbstractMatrix{<:Real}) = @inbounds map(i -> n(view(x,i,:)), 1:size(x,1))
(n::IndicatorNode)(x::AbstractMatrix{<:Real}, θ...) = @inbounds map(i -> n(view(x,i,:), θ...), 1:size(x,1))
(n::UnivariateNode)(x::AbstractMatrix{<:Real}) = @inbounds map(i -> n(view(x,i,:)), 1:size(x,1))
(n::UnivariateNode)(x::AbstractMatrix{<:Real}, θ...) = @inbounds map(i -> n(view(x,i,:), θ...), 1:size(x,1))
(n::MultivariateNode)(x::AbstractMatrix{<:Real}) = @inbounds map(i -> n(view(x,i,:)), 1:size(x,1))
(n::MultivariateNode)(x::AbstractMatrix{<:Real}, θ...) = @inbounds map(i -> n(view(x,i,:), θ...), 1:size(x,1))

@inline logpdf(n::Leaf, x::AbstractArray{<:Real}) = n(x)
@inline logpdf(n::Leaf, x::AbstractArray{<:Real}, y...) = n(x, y...)

rand(spn::SumProductNetwork) = rand(spn.root)

function rand(node::Node)
    @assert isnormalized(node)
    @assert hasscope(node)
    v = rand_(node)
    return map(d -> v[d], sort(scope(node)))
end

rand(node::IndicatorNode) = node.value
rand(node::UnivariateNode) = rand(node.dist)
rand(node::MultivariateNode) = rand(node.dist)

function rand_(node::ProductNode)
    @assert isnormalized(node)
    @assert hasscope(node)
    return mapreduce(c -> rand_(c), merge, filter(c -> hasscope(c), children(node)))
end

function rand_(node::SumNode)
    @assert isnormalized(node)
    @assert hasscope(node)
    w = Float64.(weights(node))
    z = rand(Categorical(w / sum(w))) # Normalisation due to precision errors.
    # Generate observation by drawing from a child.
    return rand_(children(node)[z])
end

rand_(node::IndicatorNode) = Dict(node.scope => node.value)
rand_(node::UnivariateNode) = Dict(node.scope => rand(node.dist))
function rand_(node::MultivariateNode)
	x = rand(node.dist)
	return Dict(d[2] => x[d[1]] for d in enumerate(node.scope))
end
