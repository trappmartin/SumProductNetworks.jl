# abstract definition of a SPN layer
abstract SPNLayer
abstract InternalLayer <: SPNLayer
abstract LeafLayer <: SPNLayer

# Sum Layers are layers containing only sum nodes.
# Per definition each sum inside a sum layer
# connects with all nodes in the next layer.
type SumLayer <: InternalLayer

  # Fields
	parent::InternalLayer
  child::SPNLayer
  weights::Array{Float64, 2}

	# This flags determins if the layer
	# should act as a filter or not.
	# If the flag is true, exp(x * w') will be evaluated
	# instead of x * w'!
  isFilter::Bool

  SumLayer(parent, child, weights; filter = false) = new(parent, child, weights, filter)

end

# Product Layers are layers containing only product nodes.
# Per definition each product inside a product layer
# connects with exactly one layer below.
type ProductLayer <: InternalLayer

  # Fields
	parent::InternalLayer
  children::Vector{SPNLayer}
  classes::Vector{ClassNode}

  ProductLayer(parent; children = SPNLayer[], classes = ClassNode[]) = new(parent, children, classes)
end

# Feature Layers are layers containing feature nodes.
type FeatureLayer <: LeafLayer

	parents::Vector{SPNLayer}
  scope::Vector{Int}

  FeatureLayer(scope::Vector{Int}; parents = SPNLayer[]) = new(parents, scope)
end
