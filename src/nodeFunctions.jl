export classes, children, parents, length, add!, remove!, normalize!, llh

"""

	classes(node) -> classlabels::Vector{Int}

Returns a list of class labels the Node is associated with.

##### Parameters:
* `node::ProductNode`: node to be evaluated
"""
function classes(node::ProductNode)

	classNodes = Vector{Int}(0)

  for classNode in filter(c -> isa(c, IndicatorNode), node.children)
    push!(classNodes, classNode.value)
  end

  for parent in node.parents
    classNodes = cat(1, classNodes, classes(parent))
  end

	return unique(classNodes)
end

"""

	classes(node) -> classlabels::Vector{Int}

Returns a list of class labels the Node is associated with.

##### Parameters:
* `node::SPNNode`: Node to be evaluated.
"""
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

	normalize!(S)

Localy normalize the weights of a SPN using Algorithm 1 from Peharz et al.

##### Parameters:
* `node::SumNode`: Sum Product Network

##### Optional Parameters:
* `ϵ::Float64`: Lower bound to ensure we don't devide by zero. (default 1e-10)
"""
function normalize!(S::SumNode; ϵ = 1e-10)

	nodes = order(S)
	αp = ones(length(nodes))

	for (nid, node) in enumerate(nodes)

		if isa(node, Leaf)
			continue
		end

		α = 0.0

		if isa(node, SumNode)
			α = sum(node.weights)

			if α < ϵ
				α = ϵ
			end
			node.weights[:] ./= α
			node.weights[node.weights .< ϵ] = ϵ

		elseif isa(node, ProductNode)
			α = αp[nid]
			αp[nid] = 1
		end

		for fnode in parents(node)

			if isa(fnode, SumNode)
				id = findfirst(children(fnode) .== node)
				@assert id > 0
				fnode.weights[id] = fnode.weights[id] * α
			elseif isa(fnode, ProductNode)
				id = findfirst(nodes .== fnode)
				if id == 0
					println("parent of the following node not found! ", nid)
				end
				@assert id > 0
				αp[id] = α * αp[id]
			end

		end

	end


end

"""

	normalizeNode!(node) -> parents::SPNNode[]

Normalize the weights of a sum node in place.

##### Parameters:
* `node::SumNode`: Sum node to be normnalized.

##### Optional Parameters:
* `ϵ::Float64`: Additional noise to ensure we don't devide by zero. (default 1e-8)
"""
function normalizeNode!(node::SumNode; ϵ = 1e-8)
  node.weights /= sum(node.weights) + ϵ
  node
end

"""
Add a node to a sum node with random weight in place.
add!(node::SumNode, child::SPNNode) -> SumNode
"""
function add!(parent::SumNode, child::SPNNode)
  add!(parent, child, rand())
end

"""
Add a node to a sum node with given weight in place.
add!(node::SumNode, child::SPNNode, weight::Float64) -> SumNode
"""
function add!(parent::SumNode, child::SPNNode, weight::Float64)
	if !(child in parent.children)
	  push!(parent.children, child)
	  push!(parent.weights, weight)
	  push!(child.parents, parent)
	end
end

"""
Add a node to a product node in place.
add!(node::ProductNode, child::SPNNode) -> ProductNode
"""
function add!(parent::ProductNode, child::SPNNode)
	if !(child in parent.children)
	  push!(parent.children, child)
	  push!(child.parents, parent)
	end
end

"""
Remove a node from the children list of a sum node in place.
remove!(node::SumNode, index::Int) -> SumNode
"""
function remove!(parent::SumNode, index::Int)
	pid = findfirst(parent .== parent.children[index].parents)

	deleteat!(parent.children[index].parents, pid)
	deleteat!(parent.children, index)
  deleteat!(parent.weights, index)

  parent
end

"""
Remove a node from the children list of a product node in place.
remove!(node::ProductNode, index::Int) -> ProductNode
"""
function remove!(parent::ProductNode, index::Int)
	pid = findfirst(parent .== parent.children[index].parents)

	deleteat!(parent.children[index].parents, pid)
  deleteat!(parent.children, index)

  parent
end


"Recursively get number of children including children of children..."
function deeplength(node::SPNNode)

    if isa(node, Leaf)
        return 1
    else
				if Base.length(node.children) > 0
        	return sum([deeplength(child) for child in node.children])
				else
					return 0
				end
    end

end

function length(node::SPNNode)

    if isa(node, Node)
        return Base.length(node.children)
    else
        return 0
    end

end

"""

	llh(S, data) -> logprobvals::Vector{T}

"""
function llh{T<:Real}(S::Node, data::AbstractArray{T})
    # get topological order
    nodes = order(S)

		maxId = maximum(Int[node.id for node in nodes])
    llhval = Matrix{Float64}(size(data, 1), maxId)

    for node in nodes
        eval!(node, data, llhval)
    end

    return llhval[:, S.id]
end

"""
Evaluate Sum-Node on data.
This function updates the llh of the data under the model.
"""
function evalSum!(M::AbstractMatrix{Float64}, iRange::Range, cids::Vector{Int}, nid::Int, logw::Vector{Float64})
	@simd for ii in iRange
		@inbounds M[ii, nid] = logsumexp(view(M, ii, cids) + logw)
	end
end

function eval!{T<:Real}(node::SumNode, data::AbstractMatrix{T}, llhvals::AbstractMatrix{Float64}; id2index::Function = (id) -> id)
	cids = id2index.(Int[child.id for child in children(node)])
	logw = log.(node.weights)
	nid = id2index(node.id)
	evalSum!(llhvals, 1:size(data, 1), cids, nid, logw)
	@assert !any(isnan(logw)) "low weights of sum node: $(node.id) contains NaN's!"
end

"""
Evaluate Product-Node on data.
This function updates the llh of the data under the model.
"""
function eval!{T<:Real}(node::ProductNode, data::AbstractMatrix{T}, llhvals::AbstractMatrix{Float64}; id2index::Function = (id) -> id)
	cids = id2index.(Int[child.id for child in children(node)])
	nid = id2index(node.id)
	@inbounds llhvals[:, nid] = sum(llhvals[:, cids], 2)
	@assert !any(isnan(view(llhvals, 1:size(data, 1), nid))) "result computed by product node: $(node.id) contains NaN's!"
end

"""
Evaluate IndicatorNode on data.
This function updates the llh of the data under the model.
"""
function eval!{T<:Real}(node::IndicatorNode, data::AbstractArray{T}, llhvals::AbstractArray{Float64}; id2index::Function = (id) -> id)
	nid = id2index(node.id)
	@simd for ii in 1:size(data, 1)
		@inbounds llhvals[ii, nid] = isnan(data[ii,node.scope]) ? 0.0 : log(data[ii,node.scope] == node.value)
	end
	@assert !any(isnan(view(llhvals, 1:size(data, 1), nid))) "result computed by indicator node: $(node.id) contains NaN's!"
end

"""
Evaluate UnivariateFeatureNode on data.
"""
function eval!{T<:Real}(node::UnivariateFeatureNode, data::AbstractArray{T}, llhvals::AbstractArray{Float64}; id2index::Function = (id) -> id)
	N = size(data, 1)
	@inbounds llhvals[1:N, id2index(node.id)] = node.weight * data[:, node.scope]
  @assert !any(isnan(view(llhvals, 1:size(data, 1), id2index(node.id)))) "result computed by univariate feature node: $(node.id) contains NaN's!"
end

"""
Evaluate MultivariateFeatureNode on data.
"""
function eval!{T<:Real}(node::MultivariateFeatureNode, data::AbstractArray{T}, llhvals::AbstractArray{Float64}; id2index::Function = (id) -> id)
	N = size(data, 1)
	D = length(node.weights)
	nid = id2index(node.id)
	@simd for ii in 1:N
		@inbounds llhvals[ii, nid] = dot(node.weights, @view data[ii, node.scope]) # formulation by Gens et al.
		# @inbounds llhvals[ii, nid] = -log(1+exp(-dot(node.weights, @view data[ii, node.scope]))) # standard logistic function 1/(1+exp(-x))
	end
  @assert !any(isnan(view(llhvals, 1:size(data, 1), id2index(node.id)))) "result computed by univariate feature node: $(node.id) contains NaN's!"
end

"""
Evaluate NormalDistributionNode on data.
This function updates the llh of the data under the model.
"""
function eval!(node::NormalDistributionNode, data::AbstractMatrix{Float64}, llhvals::AbstractMatrix{Float64}; id2index::Function = (id) -> id)
	nid = id2index(node.id)
	@simd for i in 1:size(data, 1)
		@inbounds llhvals[i, nid] = isnan(data[i,node.scope]) ? 0.0 : normlogpdf(node.μ, node.σ, data[i, node.scope])
	end

	@assert !any(isnan(view(llhvals, 1:size(data, 1), nid))) "result computed by normal distribution node: $(node.id) with μ: $(node.μ) and σ: $(node.σ) contains NaN's!"
end

"""
Evaluate UnivariateNode on data.
This function updates the llh of the data under the model.
"""
function eval!{T<:Real, U}(node::UnivariateNode{U}, data::AbstractArray{T}, llhvals::AbstractArray{Float64}; id2index::Function = (id) -> id)
	@inbounds llhvals[:, id2index(node.id)] = logpdf(node.dist, data[:, node.scope])
	@assert !any(isnan(view(llhvals, 1:size(data, 1), id2index(node.id)))) "result computed by univariate distribution node: $(node.id) with distribution: $(node.dist) contains NaN's!"
end

"""
Evaluate MultivariateNode on data.
This function updates the llh of the data under the model.
"""
function eval!{T<:Real, U}(node::MultivariateNode{U}, data::AbstractArray{T}, llhvals::AbstractArray{Float64}; id2index::Function = (id) -> id)
  @inbounds llhvals[:, id2index(node.id)] = logpdf(node.dist, data[:, node.scope]')
	@assert !any(isnan(view(llhvals, 1:size(data, 1), id2index(node.id)))) "result computed by multivariate distribution node: $(node.id) with distribution: $(node.dist) contains NaN's!"
end
