"""

	classes(node) -> classlabels::Vector{Int}

Returns a list of class labels the Node is associated with.

##### Parameters:
* `node::ProductNode`: node to be evaluated
"""
function classes(node::ProductNode)

  if length(node.classes) > 0
    return [classNode.class for classNode in node.classes]
  end

  classNodes = Vector{Int}(0)

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

@doc doc"""
Add a node to a sum node with random weight in place.
add!(node::SumNode, child::SPNNode) -> SumNode
""" ->
function add!(parent::SumNode, child::SPNNode)
  if parent.isFilter
    add!(parent, child, 1e-6)
  else
    add!(parent, child, rand())
  end
  parent
end

@doc doc"""
Add a node to a sum node with given weight in place.
add!(node::SumNode, child::SPNNode, weight::Float64) -> SumNode
""" ->
function add!(parent::SumNode, child::SPNNode, weight::Float64)
  push!(parent.children, child)
  push!(parent.weights, weight)
  push!(child.parents, parent)
  parent
end

@doc doc"""
Add a node to a product node in place.
add!(node::ProductNode, child::SPNNode) -> ProductNode
""" ->
function add!(parent::ProductNode, child::SPNNode)
  push!(parent.children, child)
  push!(child.parents, parent)
  parent
end

@doc doc"""
Remove a node from the children list of a sum node in place.
remove!(node::SumNode, index::Int) -> SumNode
""" ->
function remove!(parent::SumNode, index::Int)
	pid = findfirst(parent .== parent.children[index].parents)

	deleteat!(parent.children[index].parents, pid)

	deleteat!(parent.children, index)
  deleteat!(parent.weights, index)

  parent
end

@doc doc"""
Remove a node from the children list of a product node in place.
remove!(node::ProductNode, index::Int) -> ProductNode
""" ->
function remove!(parent::ProductNode, index::Int)
	pid = findfirst(parent .== parent.children[index].parents)

	deleteat!(parent.children[index].parents, pid)
  deleteat!(parent.children, index)

  parent
end


@doc doc"""
Type definition for topological ordering.
Naming could be improved.
""" ->
type SPNMarking
  ordering::Array{SPNNode}
  unmarked::Array{SPNNode}
end

@doc doc"""
Compute topological order of SPN using Tarjan's algoritm.
order(spn::Node) -> SPNNode[] in topological order
""" ->
function order(root::Node)

    function visit!(node::SPNNode, data::SPNMarking)

        if node in data.unmarked

            if isa(node, Node)
                for n in node.children
                    data = visit!(n, data)
                end
            end

            idx = findfirst(data.unmarked, node)
            splice!(data.unmarked, idx)
            push!(data.ordering, node)
        end

        data
    end

    N = deeplength(root)

    marking = SPNMarking(Array{SPNNode}(0), Array{SPNNode}(0))
    flat!(marking.unmarked, root)

    while(Base.length(marking.unmarked) > 0)
        n = marking.unmarked[end]

        visit!(n, marking)

    end

    marking.ordering
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

function flat!(nodes::Array{SPNNode}, node::SPNNode)
    if isa(node, Node)
        for n in node.children
            flat!(nodes, n)
        end
    end

    if !(node in nodes)
        push!(nodes, node)
    end

    nodes
end

@doc doc"""
Collapse useless node inside the SPN.
collapse!(root::SumNode) -> SumNode
""" ->
function collapse!(root::SumNode)

	# get topological order
	toporder = order(root)

	for node in toporder
		# TODO
	end

end

@doc doc"""
Fix the SPN by removing prior on Distributions.
""" ->
function fixSPN!(root::SumNode)

	# get topological order
	toporder = order(root)

	for node in toporder
		if isa(node, Leaf)
			if isa(node.dist, BinomialBeta)

				d = BNP.convert(node.dist)

        # find node in parent
        hasParents = length(node.parents) > 0
        while hasParents

          parent = node.parents[1]
          newNode = UnivariateNode{Binomial}(d, node.scope)
          id = findfirst(node .== parent.children)
          remove!(parent, id)
          add!(parent, newNode)

          hasParents = length(node.parents) > 0
        end

      elseif isa(node.dist, NormalGamma)

				d = BNP.convert(node.dist)

        hasParents = length(node.parents) > 0
        while hasParents

          parent = node.parents[1]
          newNode = NormalDistributionNode(node.scope, μ = mean(d), σ = std(d))
          id = findfirst(node .== parent.children)
          remove!(parent, id)
          add!(parent, newNode)

          hasParents = length(node.parents) > 0
        end

			end
		end
	end

end

"""

	llh(S, data) -> logprobvals::Vector{T}

"""
function llh{T<:Real}(S::Node, data::AbstractArray{T})
    # get topological order
    nodes = order(S)

		maxId = maximum(Int[node.id for node in nodes])
    llhval = Matrix{Float64}(maxId, size(data, 2))

    for node in nodes
        eval!(node, data, llhval)
    end

    return vec(llhval[S.id, :])
end

function cmllh{T<:Real}(root::Node, query::Dict{Int, T}, evidence::Dict{Int, T})
    # get topological order
    toporder = order(root)


    llhEQ = 0
		# compute P(evidence, query)
    for q in keys(query)

      llhval = Dict{SPNNode, Array{Float64}}()
  		data = ones(length(root.scope), 1) * NaN
  		for d in root.scope
  			if q == d
  				data[d] = query[d]
  			elseif haskey(evidence, d)
  				data[d] = evidence[d]
  			end
  		end

      for node in toporder
          # take only llh values. Eval function returns: (llh, map, mappath)
          llhval[node] = eval(node, data, llhval)[1]
      end

      llhEQ += llhval[toporder[end]][1]

    end

		# compute P(evidence)

    llhval = Dict{SPNNode, Array{Float64}}()
		data = ones(length(root.scope), 1) * NaN
		for d in root.scope
			if haskey(evidence, d)
				data[d] = evidence[d]
			end
		end

    for node in toporder
        # take only llh values. Eval function returns: (llh, map, mappath)
        llhval[node] = eval(node, data, llhval)[1]
    end

		llhE = llhval[toporder[end]][1]

    return llhEQ - (llhE * length(keys(query)))
end

function map_path!(root::SPNNode, allpath::Dict{SPNNode, Array{SPNNode}}, mappath::Dict{SPNNode, Array{SPNNode}})

    if haskey(allpath, root)
        for child in allpath[root]
            map_path!(child, allpath, mappath)
        end

        mappath[root] = allpath[root]

    end

end

function map{T<:Real}(root::Node, data::AbstractArray{T})

    # get topological order
    toporder = order(root)

    path = Dict{SPNNode, Array{SPNNode}}()
    mapval = Dict{SPNNode, Array{Float64}}()

    for node in toporder
        (llh, v, p) = eval(node, data, mapval)
        mapval[node] = v

        if !isempty(p)
            path[node] = p
        end
    end

    # construct MAP path
    mappath = Dict{SPNNode, Array{SPNNode}}()
    map_path!(toporder[end], path, mappath)

    return (mapval[toporder[end]], mappath)
end

function filterEval!{T<:Real}(node::SumNode, data::AbstractArray{T}, llhvals::AbstractArray{Float64})
  for i = 1:length(node)
    BLAS.axpy!(node.weights[i], sub(llhvals, node.children[i].id, :), sub(llhvals, node.id, :))
  end
end

function sumEval!{T<:Real}(node::SumNode, data::AbstractArray{T}, llhvals::AbstractArray{Float64})
  cids = Int[child.id for child in children(node)]
  llhvals[node.id, :] = NumericExtensions.logsumexp(sub(llhvals, cids, :) .+ log(node.weights), 1)# .- log(sum(node.weights))

	#logsumexp!(llhvals, node.id, cids, log(node.weights))

end

function eval!{T<:Real}(node::SumNode, data::AbstractArray{T}, llhvals::AbstractArray{Float64})
		if node.isFilter
      filterEval!(node, data, llhvals)
    else
      sumEval!(node, data, llhvals)
		end
end

"""
Evaluate Product-Node on data.
This function returns the llh of the data under the model, the maximum a posterior (equal to llh), and all child nodes of the maximum a posterior path.
"""
function eval!{T<:Real}(node::ProductNode, data::AbstractArray{T}, llhvals::AbstractArray{Float64})
	cids = Int[child.id for child in children(node)]
	llhvals[node.id, :] = sum(sub(llhvals, cids, :), 1)
end

function eval!{T<:Real}(node::UnivariateFeatureNode, data::AbstractArray{T}, llhvals::AbstractArray{Float64})
  @inbounds llhvals[node.id,:] = eval(node, data)
end

function eval{T<:Real}(node::UnivariateFeatureNode, data::AbstractArray{T})
    if ndims(data) > 1
      if node.bias
        return zeros(1, size(data, 2))
      else
        return data[node.scope,:]
      end
    else
      if node.bias
        return zeros(1, 1)
      else
        return reshape(data[node.scope], 1, 1)
      end
    end
end

function eval!{T<:Real}(node::NormalDistributionNode, data::AbstractArray{T}, llhvals::AbstractArray{Float64})
  @inbounds llhvals[node.id,:] = eval(node, data)
end

function eval{T<:Real}(node::NormalDistributionNode, data::AbstractArray{T})

    if ndims(data) > 1
			llh = zeros(Float64, 1, size(data, 2))
			for i in 1:size(data, 2)
		  	llh[i] = normlogpdf(node.μ, node.σ, data[node.scope,i]) - node.logz
		  end
      return llh
    else
      llh = normlogpdf(node.μ, node.σ, data[node.scope]) - node.logz
      return reshape([llh], 1, 1)
    end

end

function eval!{T<:Real, U}(node::UnivariateNode{U}, data::AbstractArray{T}, llhvals::AbstractArray{Float64})
	@inbounds llhvals[node.id,:] = eval(node, data)
end

@doc doc"""
This function returns the llh of the data under a univariate node.

eval(node, X) -> llh::Array{Float64, 2}, map::Array{Float64, 2}, children::Array{SPNNode}
""" ->
function eval{T<:Real, U<:DiscreteUnivariateDistribution}(node::UnivariateNode{U}, data::AbstractArray{T})
    if ndims(data) > 1
      llh = logpdf(node.dist, data[node.scope,:]) - logpdf(node.dist, mean(node.dist))
      return llh
    else
        llh = logpdf(node.dist, data[node.scope])
        return reshape([llh], 1, 1)
    end
end

@doc doc"""
This function returns the llh of the data under a univariate node.

eval(node, X) -> llh::Array{Float64, 2}, map::Array{Float64, 2}, children::Array{SPNNode}
""" ->
function eval{T<:Real, U<:ContinuousUnivariateDistribution}(node::UnivariateNode{U}, data::AbstractArray{T})

    if ndims(data) > 1
        llh = logpdf(node.dist, data[node.scope,:]) - logpdf(node.dist, mean(node.dist))
        return reshape(llh, 1, size(data, 2))
    else
        llh = logpdf(node.dist, data[node.scope]) - logpdf(node.dist, mean(node.dist))
        return reshape([llh], 1, 1)
    end

end

@doc doc"""
This function returns the llh of the data under a univariate node.

eval(node, X) -> llh::Array{Float64, 2}, map::Array{Float64, 2}, children::Array{SPNNode}
""" ->
function eval{T<:Real, U<:ConjugatePostDistribution}(node::UnivariateNode{U}, data::AbstractArray{T})
    if ndims(data) > 1
        x = sub(data, node.scope, :)

				if all(isnan(x))
					return [0]
				end

        llh = logpred(node.dist, x)
        return llh
    else
        llh = logpred(node.dist, data[node.scope])
        return reshape([llh], 1, 1)
    end

end

function eval!{T<:Real, U}(node::MultivariateNode{U}, data::AbstractArray{T}, llhvals::AbstractArray{Float64})
  @inbounds llhvals[node.id,:] = eval(node, data)
end

"""
Evaluate Multivariate Node with ContinuousMultivariateDistribution.
This function returns the llh of the data under the model, the maximum a posterior (equal to llh), and itself.
"""

function eval{T<:Real, U<:ContinuousMultivariateDistribution}(node::MultivariateNode{U}, data::AbstractArray{T})
  return logpdf(node.dist, data[node.scope,:])
end

"""
Evaluate Multivariate Node with ConjugatePostDistribution.
This function returns the llh of the data under the model, the maximum a posterior (equal to llh), and itself.
"""
function eval{T<:Real, U<:ConjugatePostDistribution}(node::MultivariateNode{U}, data::AbstractArray{T})
  return logpred(node.dist, data[node.scope,:])
end
