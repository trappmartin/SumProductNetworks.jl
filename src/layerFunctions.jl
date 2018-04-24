export size, weights, cids, sstats, posterior_sstats,
        evaluate!, evaluate, evaluateLLH!, evaluateCLLH!,
        llh, cllh

"""
Several helper functions
"""
size(layer::MultivariateFeatureLayer) = (size(layer.scopes, 2), 1)
size(layer::AbstractInternalLayer) = size(layer.childIds')
size(layer::IndicatorLayer) = (length(layer.scopes), length(layer.values))
size(layer::GaussianLayer) = (length(layer.scopes), 1)

children(layer::AbstractInternalLayer) = layer.children
weights(layer::AbstractSumLayer) = layer.logweights
cids(layer::AbstractInternalLayer) = layer.childIds

sstats(layer::AbstractBayesianLayer) = layer.sufficientStats
sstats(layer::AbstractBayesianLeafLayer) = layer.sufficientStats

posterior_sstats(layer::BayesianSumLayer) = layer.sufficientStats .+ layer.α
posterior_sstats(layer::BayesianProductLayer) = layer.sufficientStats .+ layer.β
posterior_sstats(layer::BayesianCategoricalLayer) = layer.sufficientStats .+ layer.γ

"""
  Compute log likelihood of the network given the data.
"""
function llh{T<:Real}(spn::SPNLayer, data::AbstractArray{T})
    # get topological order
    computationOrder = order(spn)

		maxId = maximum(maximum(layer.ids) for layer in computationOrder)
    (D, N) = size(data)

    llhval = Matrix{Float32}(N, maxId)

		fill!(llhval, -Inf)

    for layer in computationOrder
        eval!(layer, data, llhval)
    end

    return vec(llhval[:, spn.ids])
end

"""
  Compute conditional log likelihood of the network given the data.
"""
function cllh{T<:Real}(spn::SPNLayer, data::AbstractArray{T}, labels::Vector{Int})
    # get topological order
    computationOrder = order(spn)

		maxId = maximum(maximum(layer.ids) for layer in computationOrder)
    (D, N) = size(data)

    llhval = Matrix{Float32}(N, maxId)

		fill!(llhval, -Inf)

    # -- compute S[x, y] --
    for layer in computationOrder
        eval!(layer, data, labels, llhval)
    end

    Sy = vec(llhval[:, spn.ids])

    fill!(llhval, -Inf)

    # -- compute S[x, 1] --
    for layer in computationOrder
        eval!(layer, data, llhval)
    end

    S1 = vec(llhval[:, spn.ids])

    return Sy - S1
end

# evaluation of a layer
const evaluate!(layer::SPNLayer, data, llhvals) = evaluateLLH!(layer::SPNLayer, data, llhvals)
const evaluateLLH!(layer::SPNLayer, data::AbstractArray, llhvals::Matrix) = evaluateLLH!(layer, data, llhvals, view(llhvals, :,layer.ids))
const evaluateCLLH!(layer::SPNLayer, data::AbstractArray, labels::Vector{Int}, llhvals::Matrix) = evaluateCLLH!(layer, data, labels, llhvals, view(llhvals, :,layer.ids))

function evaluate(layer::SPNLayer, data::AbstractArray, llhvals::Matrix)
  llhvalOut = copy(llhvals[:,layer.ids])
  evaluateLLH!(layer, data, llhvals, llhvalOut)
  return llhvalOut
end

"""
Evaluates a SumLayer using its weights on the data matrix.

# Arguments
* `layer`: layer used for the computation.
* `data`: data matrix (in D × N format) used for the computation.
* `llhvals`: resulting llh values (in C × N format).
"""
function evaluateLLH!(layer::SumLayer, data::AbstractArray, llhvals::Matrix, llhval::AbstractArray)

  (Dx, N) = size(data)
  (C, Ch) = size(layer)

  # clear data
  llhval[:] = -Inf
  @inbounds cids = layer.childIds[:,1]
  @inbounds logw = reshape(layer.logweights[:,1], 1, Ch)

  @inbounds for c in 1:C
    cids[:] = layer.childIds[:,c]
    logw[:] = reshape(layer.logweights[:,c], 1, Ch)
    @fastmath llhval[:, c] = logsumexp(llhvals[:, cids] .+ logw, dim = 2)
  end

end

"""
Evaluates a Bayesian SumLayer by drawing its weights from the posterior of a Dirichlet.

# Arguments
* `layer`: layer used for the computation.
* `data`: data matrix (in D × N format) used for the computation.
* `llhvals`: resulting llh values (in C × N format).
"""
function evaluateLLH!(layer::AbstractBayesianLayer, data::AbstractArray, llhvals::Matrix, llhval::AbstractArray)

  (Dx, N) = size(data)
  (C, Ch) = size(layer)

  # clear data
  llhval[:] = -Inf
  @inbounds cids = layer.childIds[:,1]
  @inbounds @fastmath logw = mapslices(sstat -> log.(rand(Dirichlet(vec(sstat)))), posterior_sstats(layer), [1]) # Ch x C

  @inbounds for c in 1:C
    cids[:] = layer.childIds[:,c]
    @fastmath llhval[:, c] = logsumexp(llhvals[:, cids] .+ reshape(logw[:,c], 1, Ch), dim = 2)
  end

end

"""
Evaluates a ProductLayer on the data matrix.

# Arguments
* `layer`: layer used for the computation.
* `data`: data matrix (in D × N format) used for the computation.
* `llhvals`: resulting llh values (in C × N format).
"""
function evaluateLLH!(layer::AbstractProductLayer, data::AbstractArray, llhvals::Matrix, llhval::AbstractArray)

  (Dx, N) = size(data)
  (C, Ch) = size(layer)

  # clear data
  llhval[:] = -Inf

  @inbounds for c in 1:C
    cids = layer.childIds[:,c]
    llhval[:,c] = sum(llhvals[:,cids], 2)
  end

end

"""
Evaluates a ProductCLayer with its class labels on the data matrix.

# Arguments
* `layer`: layer used for the computation.
* `data`: data matrix (in D × N format) used for the computation.
* `labels`: labels vector (in N format, starting with 1) used for the computation.
* `llhvals`: resulting llh values (in C × N format).
"""
function evaluateCLLH!(layer::ProductCLayer, data::AbstractArray, labels::Vector{Int}, llhvals::Matrix, llhval::AbstractArray)

  (Dx, N) = size(data)
  (C, Ch) = size(layer)

  # clear data
  llhval[:] = -Inf

  @inbounds for c in 1:C
    label = layer.clabels[c]
    cids = layer.childIds[:,c]
    for n in 1:N
      @fastmath llhval[n,c] = sum(llhvals[n,cids]) + log(labels[n] == label)
    end
  end
end

"""
Evaluates a MultivariateFeatureLayer using its weights on the data matrix.

# Arguments
* `layer`: layer used for the computation.
* `data`: data matrix (in D × N format) used for the computation.
* `llhvals`: resulting llh values (in C × N format).
"""
function evaluateLLH!(layer::MultivariateFeatureLayer, data::AbstractArray, llhvals::Matrix, llhval::AbstractArray)

  (Dx, N) = size(data)
  (Dl, C) = size(layer.scopes)

  @assert Dl == Dx "data dimensionality $(Dx) does not match layer dimensionality $(Dl)"

  # clear data
  llhval[:] = 0.

  # apply the standard logistic function: $\frac{1}{1+exp(-x*w')}$
  # derivative: $\frac{x * exp(w*x)}{(1+exp(w*x))^2}$

  @inbounds for c in 1:C
      w = layer.weights[:, c] .* layer.scopes[:, c]
      for n in 1:N
          x = dot(data[:,n], w)
          # compute log(1) - log(1+exp(-wx))
          @fastmath llhval[n, c] = -log(1. + exp(-x))
      end
  end

end

"""
Evaluates a IndicatorLayer on the data matrix.

# Arguments
* `layer`: layer used for the computation.
* `data`: data matrix (in D × N format) used for the computation.
* `llhvals`: resulting llh values (in C × N format).
"""
function evaluateLLH!(layer::IndicatorLayer, data::AbstractArray, llhvals::Matrix, llhval::AbstractArray)

  (Dx, N) = size(data)
  (Dl, C) = size(layer)

  @assert Dl == Dx "data dimensionality $(Dx) does not match layer dimensionality $(Dl)"

  # clear data
  llhval[:] = -Inf

  @inbounds for c in 1:C
    idx = Int[sub2ind((Dl, C), i, c) for i in 1:Dl]
    @fastmath llhval[:, idx] = log.(data[layer.scopes,:]' .== layer.values[c])
  end

end

"""
Evaluates a GaussianLayer on the data matrix.

# Arguments
* `layer`: layer used for the computation.
* `data`: data matrix (in D × N format) used for the computation.
* `llhvals`: resulting llh values (in C × N format).
"""
function evaluateLLH!(layer::GaussianLayer, data::AbstractArray, llhvals::Matrix, llhval::AbstractArray)

  (Dx, N) = size(data)
  (C, _) = size(layer)

  # clear data
  llhval[:] = -Inf

  for c in 1:C
    @inbounds llhval[n, c] = normlogpdf.(node.μ, node.σ, data[i, node.scope])
  end

end
