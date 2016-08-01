"""

	connect!(layer, child, class)

	Connect a product layer with a child layer and assigning the given class.

##### Parameters:

* `layer::ProductLayer`: The parent layer.
* `child::SPNLayer`: The new child layer.

"""
function connect!(layer::ProductLayer, child::InternalLayer)
	push!(layer.children, child)
	child.parent = layer
end

function connect!(layer::ProductLayer, child::LeafLayer)
	push!(layer.children, child)
	push!(layer.parents, layer)
end

"""

	connect!(layer, child, class)

	Connect a sum layer with a child layer.

##### Parameters:

* `layer::SumLayer`: The parent layer.
* `child::SPNLayer`: The new child layer.
* `weights::Array{Float64, 2}`: Weights of the connections (#sum nodes * #child nodes)

"""
function connect!(layer::SumLayer, child::LeafLayer, weights::Array{Float64, 2})
	layer.child = child
	layer.weights = weights
	layer.parent = layer
end

function connect!(layer::SumLayer, child::InternalLayer, weights::Array{Float64, 2})
	layer.child = child
	layer.weights = weights
	layer.parent = layer
end

"""

	add!(layer, node)

	Add a node to a distribution layer.

##### Parameters:

* `layer::NormalDistributionLayer`: The parent layer.
* `node::NormalDistributionNode`: The new NormalDistributionNode.

"""
function add!(layer::NormalDistributionLayer, node::NormalDistributionNode)
	push!(layer.nodes, node)
end

"""

	update!(spn)

	Update the list of layers available in the SPN.

##### Parameters:

* `spn::LayeredSPN`: The layered Sum-Product Network.

"""
function update!(spn::LayeredSPN)


	if has(spn.child)

		layers = Vector{SPNLayer}(0)

		push!(layers, get(spn.child))

		for c in children(get(spn.child))

		end

	end

end

"""

	set!(spn, layer)

	Set layer to be the successive layer or the root.
	The method also updates the list of layers available in the SPN.

##### Parameters:

* `spn::LayeredSPN`: The layered Sum-Product Network.
* `layer::InternalLayer`: The layer.

"""
function set!(spn::LayeredSPN, layer::InternalLayer)

end

"""

	children(layer) -> Vector{SPNLayer}

	Get all children of the given layer.

##### Parameters:

* `layer::InternalLayer`: The layer.
"""
function children(layer::ProductLayer)
	return layer.children
end

function children(layer::SumLayer)
	return SPNLayer[get(layer.child)]
end

"""

	length(layer) -> Integer

	Get the number of nodes in the given layer.

##### Parameters:

* `layer::ProductLayer`: The layer.
"""
function length(layer::SumLayer)
	return size(layer.weights, 2)
end

function length(layer::ProductLayer)
	return length(layer.children)
end

function length(layer::NormalDistributionLayer)
	return length(layer.nodes)
end
