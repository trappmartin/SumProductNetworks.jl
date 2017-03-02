export size, eval!, eval

"""
Returns the size of a SPN layer object.
"""
function size(layer::MultivariateFeatureLayer)
  return (size(layer.scopes, 2), 1)
end

function size(layer::SumLayer)
  return size(layer.childIds')
end

function size(layer::AbstractProductLayer)
  return size(layer.childIds')
end

function size(layer::SPNLayer)
  error("Not implemented!")
end

function eval!(layer::SPNLayer, data::AbstractArray, labels::Vector{Int}, llhvals::Matrix)
  eval!(layer, data, labels, llhvals, view(llhvals, :,layer.ids))
end

function eval!(layer::SPNLayer, data::AbstractArray, labels::Vector{Int}, llhvals::Matrix, llhvalOut::AbstractArray)
  eval!(layer, data, llhvals, llhvalOut)
end

function eval(layer::SPNLayer, data::AbstractArray, labels::Vector{Int}, llhvals::Matrix)
  llhvalOut = copy(llhvals[:,layer.ids])
  eval!(layer, data, labels, llhvals, llhvalOut)
  return llhvalOut
end

function eval!(layer::SPNLayer, data::AbstractArray, llhvals::Matrix)
  eval!(layer, data, llhvals, view(llhvals, :,layer.ids))
  @assert all(!isnan(view(llhvals, :,layer.ids))) "$(typeof(layer)) outputs NaN values, something is broken!"
end

function eval(layer::SPNLayer, data::AbstractArray, llhvals::Matrix)
  llhvalOut = copy(llhvals[:,layer.ids])
  eval!(layer, data, llhvals, llhvalOut)
  return llhvalOut
end

"""
Evaluates a SumLayer using its weights on the data matrix.

# Arguments
* `layer`: layer used for the computation.
* `data`: data matrix (in D × N format) used for the computation.
* `llhvals`: resulting llh values (in C × N format).
"""
function eval!(layer::SumLayer, data::AbstractArray, llhvals::Matrix, llhval::AbstractArray)

  (Dx, N) = size(data)
  (C, Ch) = size(layer)

  # clear data
  llhval[:] = -Inf
  logw = log(layer.weights)

  @simd for c in 1:C
    @inbounds cids = layer.childIds[:,c]
    @inbounds w = logw[:,c]
    @simd for n in 1:N
      @inbounds llhval[n, c] = logsumexp(llhvals[n, cids] + w)
    end
  end
end

"""
Evaluates a ProductLayer on the data matrix.

# Arguments
* `layer`: layer used for the computation.
* `data`: data matrix (in D × N format) used for the computation.
* `llhvals`: resulting llh values (in C × N format).
"""
function eval!(layer::AbstractProductLayer, data::AbstractArray, llhvals::Matrix, llhval::AbstractArray)

  (Dx, N) = size(data)
  (C, Ch) = size(layer)

  # clear data
  llhval[:] = -Inf

  @simd for c in 1:C
    @inbounds cids = layer.childIds[:,c]
    @inbounds llhval[:,c] = sum(llhvals[:,cids], 2)
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
function eval!(layer::ProductCLayer, data::AbstractArray, labels::Vector{Int}, llhvals::Matrix, llhval::AbstractArray)

  (Dx, N) = size(data)
  (C, Ch) = size(layer)

  # clear data
  llhval[:] = -Inf

  @simd for c in 1:C
    @inbounds label = layer.clabels[c]
    @inbounds cids = layer.childIds[:,c]
    @simd for n in 1:N
      @inbounds llhval[n,c] = sum(llhvals[n,cids]) + log(labels[n] == label)
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
function eval!(layer::MultivariateFeatureLayer, data::AbstractArray, llhvals::Matrix, llhval::AbstractArray)

  (Dx, N) = size(data)
  (Dl, C) = size(layer.scopes)

  @assert Dl == Dx "data dimensionality $(Dx) does not match layer dimensionality $(Dl)"

  # clear data
  llhval[:] = 0.

  # apply the standard logistic function: $\frac{1}{1+exp(-x*w')}$
  # derivative: $\frac{x * exp(w*x)}{(1+exp(w*x))^2}$

    @simd for c in 1:C
        @inbounds w = layer.weights[:, c] .* layer.scopes[:, c]
        @simd for n in 1:N
            x = dot(data[:,n], w)
            # compute log(1) - log(1+exp(-wx))
            @inbounds llhval[n, c] = -(1. + exp(-x))
        end
    end

end
