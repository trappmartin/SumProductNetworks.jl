# abstract definition of a SPN layer
abstract SPNLayer
abstract InternalLayer <: SPNLayer
abstract LeafLayer <: SPNLayer

# Definition of a layered Sum-Product Network.
# The Network contains of Sum, Product and Leaf Layers.
type LayeredSPN

  rootWeights::Vector{Float64}
  child::Nullable{InternalLayer}

  layers::Vector{SPNLayer}

  LayeredSPN() = new(Float64[], nothing, SPNLayer[])
end

# Sum Layers are layers containing only sum nodes.
# Per definition each sum inside a sum layer
# connects with all nodes in the next layer.
type SumLayer <: InternalLayer

  # Fields
	parent::Nullable{InternalLayer}
  child::Nullable{SPNLayer}
  weights::Array{Float64, 2}

	# This flags determins if the layer
	# should act as a filter or not.
	# If the flag is true, exp(x * w') will be evaluated
	# instead of x * w'!
  isFilter::Bool

  SumLayer(size::Int; filter = false) = new(nothing, nothing, Array{Float64}(size,0), filter)

end

# Product Layers are layers containing only product nodes.
# Per definition each product inside a product layer
# connects with exactly one layer below.
type ProductLayer <: InternalLayer

  # Fields
	parent::Nullable{InternalLayer}
  children::Vector{SPNLayer}
  classes::Vector{ClassNode}

  ProductLayer() = new(nothing, SPNLayer[], ClassNode[])
end

# Feature Layers are layers containing feature nodes.
type FeatureLayer <: LeafLayer

	parents::Vector{SPNLayer}
  scope::Vector{Int}

  FeatureLayer() = new(SPNLayer[], Int[])
end

# Normal Distribution Layers are layers containing feature nodes.
type NormalDistributionLayer <: LeafLayer

	parents::Vector{SPNLayer}
  nodes::Vector{NormalDistributionNode}

  NormalDistributionLayer() = new(SPNLayer[], NormalDistributionNode[])
end
