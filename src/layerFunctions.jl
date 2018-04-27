export size, weights, cids, sstats, posterior_sstats,
evaluate!, evaluate, evaluateLLH!, evaluateCLLH!,
observations, scopes, set_observations, observation_isactive, set_observation_active,
set_scopes, set_scope_active, scope_isactive,
llh, cllh

"""
Several helper functions
"""
size(layer::MultivariateFeatureLayer) = (size(layer.scopes, 2), 1)
size(layer::AbstractInternalLayer) = size(layer.childIds')
size(layer::IndicatorLayer) = (length(layer.scopes), length(layer.values))
size(layer::GaussianLayer) = (length(layer.scopes), 1)
size(layer::BayesianCategoricalLayer) = (length(layer.ids), 1)

children(layer::AbstractInternalLayer) = layer.children
weights(layer::AbstractSumLayer) = layer.logweights
cids(layer::AbstractInternalLayer) = layer.childIds

sstats(layer::AbstractBayesianLayer) = layer.sufficientStats
sstats(layer::AbstractBayesianLeafLayer) = layer.sufficientStats

posterior_sstats(layer::BayesianSumLayer) = layer.sufficientStats .+ layer.α
posterior_sstats(layer::BayesianProductLayer) = layer.sufficientStats .+ layer.β
posterior_sstats(layer::BayesianCategoricalLayer) = layer.sufficientStats .+ layer.γ

observations(layer::AbstractBayesianLayer) = map(c -> find(layer.activeObservations[:,c]), 1:size(layer.activeObservations, 2))
observations(layer::AbstractBayesianLeafLayer) = map(c -> find(layer.activeObservations[:,c]), 1:size(layer.activeObservations, 2))

set_observations(layer::AbstractBayesianLayer, c::Int, obs::Vector{Int}) = layer.activeObservations[obs,c] = true
set_observations(layer::AbstractBayesianLeafLayer, c::Int, obs::Vector{Int}) = layer.activeObservations[obs,c] = true

set_observation_active(layer::AbstractBayesianLayer, flags::Vector{Bool}, obs::Int) = layer.activeObservations[obs,:] = flags
set_observation_active(layer::AbstractBayesianLeafLayer, flags::Vector{Bool}, obs::Int) = layer.activeObservations[obs,:] = flags
set_observation_active(layer::AbstractBayesianLayer, flag::Bool, c::Int, obs::Int) = layer.activeObservations[obs,c] = flag
set_observation_active(layer::AbstractBayesianLeafLayer, flag::Bool, c::Int, obs::Int) = layer.activeObservations[obs,c] = flag

observation_isactive(layer::AbstractBayesianLayer, obs::Int) = layer.activeObservations[obs,:]
observation_isactive(layer::AbstractBayesianLeafLayer, obs::Int) = layer.activeObservations[obs,:]
observation_isactive(layer::AbstractBayesianLayer, c::Int, obs::Int) = layer.activeObservations[obs,c]
observation_isactive(layer::AbstractBayesianLeafLayer, c::Int, obs::Int) = layer.activeObservations[obs,c]

scopes(layer::AbstractBayesianLayer) = map(c -> find(layer.activeDimensions[:,c]), 1:size(layer.activeDimensions, 2))
scopes(layer::AbstractBayesianLeafLayer) = layer.scopes

set_scopes(layer::AbstractBayesianLayer, c::Int, dims::Vector{Int}) = layer.activeDimensions[dims,c] = true
set_scopes(layer::AbstractBayesianLeafLayer, c::Int, dim::Int) = layer.scopes[c] = dim

set_scope_active(layer::AbstractBayesianLeafLayer, c::Int, flag::Bool) = layer.activeDimensions[c] = flag

scope_isactive(layer::AbstractBayesianLeafLayer, c::Int, dim::Int) = layer.activeDimensions[dim,c]
scope_isactive(layer::AbstractBayesianLeafLayer, c::Int) = layer.activeDimensions[c]

"""
Compute log likelihood of the network given the data.
"""
function llh{T<:Real}(spn::SPNLayer, X::AbstractArray{T})
    # get topological order
    computationOrder = order(spn)

    maxId = maximum(maximum(layer.ids) for layer in computationOrder)
    (N, D) = size(X)

    llhval = Matrix{Float32}(N, maxId)

    fill!(llhval, -Inf)

    for layer in computationOrder
        eval!(layer, X, llhval)
    end

    return vec(llhval[:, spn.ids])
end

"""
Compute conditional log likelihood of the network given the data.
"""
function cllh{T<:Real}(spn::SPNLayer, X::AbstractArray{T}, y::Vector{Int})
    # get topological order
    computationOrder = order(spn)

    maxId = maximum(maximum(layer.ids) for layer in computationOrder)
    (N, D) = size(X)

    llhval = Matrix{Float32}(N, maxId)

    fill!(llhval, -Inf)

    # -- compute S[x, y] --
    for layer in computationOrder
        eval!(layer, X, y, llhval)
    end

    Sy = vec(llhval[:, spn.ids])

    fill!(llhval, -Inf)

    # -- compute S[x, 1] --
    for layer in computationOrder
        eval!(layer, X, llhval)
    end

    S1 = vec(llhval[:, spn.ids])

    return Sy - S1
end

# evaluation of a layer
const evaluate!(layer::SPNLayer, X, llhvals) = evaluateLLH!(layer::SPNLayer, X, llhvals)
const evaluateLLH!(layer::SPNLayer, X::AbstractArray, llhvals::Matrix) = evaluateLLH!(layer, X, llhvals, view(llhvals, :,layer.ids))
const evaluateCLLH!(layer::SPNLayer, X::AbstractArray, y::Vector{Int}, llhvals::Matrix) = evaluateCLLH!(layer, X, y, llhvals, view(llhvals, :,layer.ids))

function evaluate(layer::SPNLayer, X::AbstractArray, llhvals::Matrix)
    llhvalOut = copy(llhvals[:,layer.ids])
    evaluateLLH!(layer, X, llhvals, llhvalOut)
    return llhvalOut
end

"""
Evaluates a SumLayer using its weights on the data matrix.

# Arguments
* `layer`: layer used for the computation.
* `X`: data matrix (in N × D format) used for the computation.
* `llhvals`: resulting llh values (in C × N format).
"""
function evaluateLLH!(layer::SumLayer, X::AbstractArray, llhvals::Matrix, llhval::AbstractArray)

    (C, Ch) = size(layer)

    # clear data
    @inbounds fill!(llhval, map(typeof(first(llhval)), -Inf))
    @inbounds cids = layer.childIds[:,1]
    @inbounds logw = reshape(layer.logweights[:,1], 1, Ch)

    @inbounds for c in 1:C
        cids[:] = layer.childIds[:,c]
        logw[:] = reshape(layer.logweights[:,c], 1, Ch)
        @fastmath llhval[:, c] = logsumexp(llhvals[:, cids] .+ logw, dim = 2)
    end

    llhval
end

"""
Evaluates a Bayesian SumLayer by drawing its weights from the posterior of a Dirichlet.

# Arguments
* `layer`: layer used for the computation.
* `X`: data matrix (in N × D format) used for the computation.
* `llhvals`: resulting llh values (in C × N format).
"""
function evaluateLLH!(layer::AbstractBayesianLayer, X::AbstractArray, llhvals::Matrix, llhval::AbstractArray)

    (C, Ch) = size(layer)

    # clear data
    @inbounds fill!(llhval, map(typeof(first(llhval)), -Inf))
    @inbounds cids = layer.childIds[:,1]
    @inbounds @fastmath logw = mapslices(sstat -> log.(rand(Dirichlet(vec(sstat)))), posterior_sstats(layer), [1]) # Ch x C

    @inbounds for c in 1:C
        cids[:] = layer.childIds[:,c]
        @fastmath llhval[:, c] = logsumexp(llhvals[:, cids] .+ reshape(logw[:,c], 1, Ch), dim = 2)
    end

    llhval
end

"""
Evaluates a ProductLayer on the data matrix.

# Arguments
* `layer`: layer used for the computation.
* `X`: data matrix (in N × D format) used for the computation.
* `llhvals`: resulting llh values (in C × N format).
"""
function evaluateLLH!(layer::AbstractProductLayer, X::AbstractArray, llhvals::Matrix, llhval::AbstractArray)

    (C, Ch) = size(layer)

    # clear data
    @inbounds fill!(llhval, map(typeof(first(llhval)), -Inf))

    @inbounds for c in 1:C
        cids = layer.childIds[:,c]
        llhval[:,c] = sum(llhvals[:,cids], 2)
    end

    llhval
end

"""
Evaluates a ProductCLayer with its class labels on the data matrix.

# Arguments
* `layer`: layer used for the computation.
* `X`: data matrix (in N × D format) used for the computation.
* `y`: labels vector (in N format, starting with 1) used for the computation.
* `llhvals`: resulting llh values (in C × N format).
"""
function evaluateCLLH!(layer::ProductCLayer, X::AbstractArray, y::Vector{Int}, llhvals::Matrix, llhval::AbstractArray)

    (C, Ch) = size(layer)

    # clear data
    @inbounds fill!(llhval, map(typeof(first(llhval)), -Inf))

    @inbounds for c in 1:C
        label = layer.clabels[c]
        cids = layer.childIds[:,c]

        @fastmath llhval[:,c] = sum(llhvals[:,cids], 2) .+ log.(y .== label)
    end
    llhval
end

"""
Evaluates a MultivariateFeatureLayer using its weights on the data matrix.

# Arguments
* `layer`: layer used for the computation.
* `X`: data matrix (in N × D format) used for the computation.
* `llhvals`: resulting llh values (in C × N format).
"""
function evaluateLLH!(layer::MultivariateFeatureLayer, X::AbstractArray, llhvals::Matrix, llhval::AbstractArray)

    (N, Dx) = size(X)
    (Dl, C) = size(layer.scopes)

    @assert Dl == Dx "data dimensionality $(Dx) does not match layer dimensionality $(Dl)"

    # clear data
    @inbounds fill!(llhval, map(typeof(first(llhval)), 0.))

    # apply the standard logistic function: $\frac{1}{1+exp(-x*w')}$
    # derivative: $\frac{x * exp(w*x)}{(1+exp(w*x))^2}$
    @fastmath llhval[:,:] = -log.(1. + exp.( X * (layer.weights .* layer.scopes) ))
    llhval
end

"""
Evaluates a IndicatorLayer on the data matrix.

# Arguments
* `layer`: layer used for the computation.
* `X`: data matrix (in N × D format) used for the computation.
* `llhvals`: resulting llh values (in C × N format).
"""
function evaluateLLH!(layer::IndicatorLayer, X::AbstractArray, llhvals::Matrix, llhval::AbstractArray)

    (N, Dx) = size(X)
    (Dl, C) = size(layer)

    @assert Dl == Dx "data dimensionality $(Dx) does not match layer dimensionality $(Dl)"

    # clear data
    @inbounds fill!(llhval, map(typeof(first(llhval)), -Inf))

    @inbounds for c in 1:C
        idx = Int[sub2ind((Dl, C), i, c) for i in 1:Dl]
        @fastmath llhval[:, idx] = log.(X[:, layer.scopes] .== layer.values[c])
    end
    llhval

end

"""
Evaluates a GaussianLayer on the data matrix.

# Arguments
* `layer`: layer used for the computation.
* `X`: data matrix (in N × D format) used for the computation.
* `llhvals`: resulting llh values (in C × N format).
"""
function evaluateLLH!(layer::GaussianLayer, X::AbstractArray, llhvals::Matrix, llhval::AbstractArray)

    (C, _) = size(layer)

    # clear data
    @inbounds fill!(llhval, map(typeof(first(llhval)), -Inf))

    for c in 1:C
        @inbounds llhval[:, c] = normlogpdf.(layer.μ[c], layer.σ[c], X[:, layer.scopes[c]])
    end
    llhval
end

"""
Evaluates a BayesianCategoricalLayer on the data matrix.

# Arguments
* `layer`: layer used for the computation.
* `data`: data matrix (in N × D format) used for the computation.
* `llhvals`: resulting llh values (in C × N format).
"""
function evaluateLLH!(layer::BayesianCategoricalLayer, X::AbstractArray, llhvals::Matrix, llhval::AbstractArray)

    # clear data
    @inbounds fill!(llhval, map(typeof(first(llhval)), -Inf))
    @inbounds llhval[:,:] = posterior_sstats(layer)[ @view X[:, layer.scopes] ]
    llhval
end
