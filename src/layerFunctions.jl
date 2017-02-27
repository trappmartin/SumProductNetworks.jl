export size, eval!

"""
Returns the size of a SPN layer object.
"""
function size(layer::MultivariateFeatureLayer)
  return (size(layer.scopes, 1), 1)
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

"""
Evaluates a SumLayer using its weights on the data matrix.

# Arguments
* `layer`: layer used for the computation.
* `data`: data matrix (in D × N format) used for the computation.
* `llhvals`: resulting llh values (in C × N format).
"""
function eval!(layer::SumLayer, data::AbstractArray, llhvals::AbstractArray)

  (Dx, N) = size(data)
  (C, Ch) = size(layer)

  # clear data
  fill!(llhvals[:,layer.ids], -Inf)

  logw = log(layer.weights)

  id = 0
  @simd for c in 1:C
    @inbounds id = layer.ids[c]
    @inbounds cids = layer.childIds[:,c]
    @inbounds w = logw[:,c]
    @simd for n in 1:N
      @inbounds llhvals[id, n] = logsumexp(llhvals[cids, n] + w)
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
function eval!(layer::AbstractProductLayer, data::AbstractArray, llhvals::AbstractArray)

  (Dx, N) = size(data)
  (C, Ch) = size(layer)

  # clear data
  fill!(llhvals[:,layer.ids], -Inf)

  id = 0
  @simd for c in 1:C
    @inbounds id = layer.ids[c]
    @inbounds cids = layer.childIds[:,c]
    @simd for n in 1:N
      @inbounds llhvals[id, n] = sum(llhvals[cids, n])
    end
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
function eval!(layer::ProductCLayer, data::AbstractArray, labels::Vector{Int}, llhvals::AbstractArray)

  (Dx, N) = size(data)
  (C, Ch) = size(layer)

  # clear data
  fill!(llhvals[:,layer.ids], -Inf)

  id = 0
  @simd for c in 1:C
    @inbounds id = layer.ids[c]
    @inbounds label = layer.clabels[c]
    @inbounds cids = layer.childIds[:,c]
    @simd for n in 1:N
      @inbounds llhvals[id, n] = sum(llhvals[cids, n]) + log(labels[n] == label)
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
function eval!(layer::MultivariateFeatureLayer, data::AbstractArray, llhvals::AbstractArray)

  (Dx, N) = size(data)
  (C, Dl) = size(layer.scopes)

  @assert Dl == Dx

  # clear data
  fill!(llhvals[:,layer.ids], 0.)

  id = 0
  @simd for d in 1:Dx
    @simd for c in 1:C
      @inbounds id = layer.ids[c]
      @inbounds w = layer.weights[c, d] * layer.scopes[c, d]
      @simd for n in 1:N
	       @inbounds llhvals[id, n] += w * data[d, n]
      end
    end
	end

end
