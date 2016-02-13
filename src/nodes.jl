# abstract definition of a SPN node
abstract SPNNode
abstract Node <: SPNNode
abstract Leaf{T} <: SPNNode

# definition of class indicater Node
type ClassNode <: Leaf

    class::Int

    ClassNode(class::Int) = new(class)
end

# definition of a Sum Node
type SumNode <: Node

  # SumNode fields
	inSPN::Bool
  uid::Int
	parent::Nullable{SPNNode}
  children::Vector{SPNNode}
  weights::Vector{Float32}

  scope::Vector{Int}

  SumNode(id::Int; parent = Nullable{SPNNode}(), scope = Vector{Int}(0)) = new(false, id, parent, SPNNode[], Float64[], scope)
  SumNode(id::Int, children::Vector{SPNNode}, w::Vector{Float64}; parent = Nullable{SPNNode}()) = new(false, id, parent, children, w, Vector{Int}(0))

end

# definition of a Product Node
type ProductNode <: Node

  # ProductNode fields
	inSPN::Bool
  uid::Int32
	parent::Nullable{SPNNode}
  children::Vector{SPNNode}
  class::Nullable{ClassNode}

  scope::Vector{Int}

  ProductNode(id::Int; parent = Nullable{SPNNode}(), children = SPNNode[], class = Nullable{ClassNode}(), scope = Vector{Int}(0)) = new(false, id, parent, children, class, scope)
end

# definition of a Univariate Node
type UnivariateNode{T} <: Leaf

	inSPN::Bool
	parent::Nullable{SPNNode}
  dist::T
  scope::Int

  UnivariateNode{T}(D::T; parent = Nullable{SPNNode}(), scope = 0) = new(false, parent, D, scope)
end

# definition of a Multivariate Node
type MultivariateNode{T} <: Leaf

	inSPN::Bool
	parent::Nullable{SPNNode}
  dist::T
  scope::Vector{Int}

  MultivariateNode{T}(D::T, scope::Vector{Int}; parent = Nullable{SPNNode}()) = new(false, parent, D, scope)

end

## -------------------------------------------------- ##
## accessing function                                 ##
## -------------------------------------------------- ##

# get children
function children(node::Node)
	node.children
end

# get parent
function parent(node::SPNNode)
	get(node.parent)
end

# normalize sum node
function normalize!(node::SumNode)
  node.weights /= sum(node.weights)
  node
end

# add node
function add!(parent::SumNode, child::SPNNode)
  add!(parent, child, rand())
  parent
end

# add node with weight
function add!(parent::SumNode, child::SPNNode, weight::Float64)
  push!(parent.children, child)
  push!(parent.weights, weight)
  child.parent = parent
	child.inSPN = true
  parent
end

# add node
function add!(parent::ProductNode, child::SPNNode)
  push!(parent.children, child)
  child.parent = parent
  child.inSPN = true
  parent
end

# remove node with index
function remove!(parent::SumNode, index::Integer)
  deleteat!(parent.children, index)
  deleteat!(parent.weights, index)

  parent
end

# remove node with index
function remove!(parent::ProductNode, index::Integer)
  deleteat!(parent.children, index)

  parent
end

" Type definition for topological ordering, naming could be improved"
type SPNMarking
  ordering::Array{SPNNode}
  unmarked::Array{SPNNode}

end

"""
Compute topological order of SPN using Tarjan's algoritm.
"""
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
Fix the SPN by removing prior on Distributions.
""" ->
function fixSPN!(root::Node)

	# get topological order
	toporder = order(root)

	for node in toporder
		if isa(node, Leaf)
			if isa(node.dist, BinomialBeta)

				d = BNP.convert(node.dist)

        # find node in parent
        parent = get(node.parent)
        id = findfirst(node .== parent.children)
        remove!(parent, id)
        newNode = UnivariateNode{Binomial}(d, scope = node.scope)
        add!(parent, newNode)

      elseif isa(node.dist, NormalGamma)

				d = BNP.convert(node.dist)

        # find node in parent
        parent = get(node.parent)
        id = findfirst(node .== parent.children)
        remove!(parent, id)
        newNode = UnivariateNode{Normal}(d, scope = node.scope)
        add!(parent, newNode)

			end
		end
	end

end

"""
Compute the log likelihood of the data under the model.
The result is computed considering the topological order of the SPN.
"""
function llh{T<:Real}(root::Node, data::AbstractArray{T})
    # get topological order
    toporder = order(root)

    llhval = Dict{SPNNode, Array{Float64}}()

    for node in toporder
        # take only llh values. Eval function returns: (llh, map, mappath)
        llhval[node] = eval(node, data, llhval)[1]
    end

    return llhval[toporder[end]]
end

"""
Compute the log likelihood of the data under the model.
This function evaluates leaf nodes only.
"""
function llh{T<:Real}(root::Leaf, data::AbstractArray{T})
    return eval(root, data)[1]
end

@doc doc"""
Computes the conditional marginal log-likelihood.
E.g.: P(query | evidence) = P(evidence, query) / P(evidence)

cmllh(spn, query, evidence) -> [P(q1 | e1)]
""" ->
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

"Extract MAP path, this implementation is possibly slow!"
function map_path!(root::SPNNode, allpath::Dict{SPNNode, Array{SPNNode}}, mappath::Dict{SPNNode, Array{SPNNode}})

    if haskey(allpath, root)
        for child in allpath[root]
            map_path!(child, allpath, mappath)
        end

        mappath[root] = allpath[root]

    end

end

"Compute MAP and MAP path, this implementation is possibly slow!"
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

"""
Evaluate Sum-Node on data.
This function returns the llh of the data under the model, the maximum a posterior, and the child node of the maximum a posterior path.
"""
function eval{T<:Real}(root::SumNode, data::AbstractArray{T}, llhvals::Dict{SPNNode, Array{Float64}})

    _llh = [llhvals[c] for c in root.children]
    _llh = reduce(hcat, _llh)
    w = repmat( log(root.weights)', size(_llh, 1), 1)

    _llh = _llh + w
    _llh = _llh'

    maxlog = maximum(_llh, 1)

    _llh = _llh .- maxlog
    prob = sum(exp(_llh), 1)
    (map, mapidx) = findmax(exp(_llh), 1)

    map = log(map) .+ maxlog
    map -= log( sum(root.weights) )

    _llh = log(prob) .+ maxlog
    _llh -= log(sum(root.weights))

    # get map path
    ids = length(root) - (mapidx % length(root))
    mappath = repmat(root.children, 1, size(data, 2))[ids]

    return (_llh', map, mappath)
end

"""
Evaluate Product-Node on data.
This function returns the llh of the data under the model, the maximum a posterior (equal to llh), and all child nodes of the maximum a posterior path.
"""
function eval{T<:Real}(root::ProductNode, data::AbstractArray{T}, llhvals::Dict{SPNNode, Array{Float64}})
    _llh = [llhvals[c] for c in root.children]
    _llh = reduce(vcat, _llh)
    return (sum(_llh, 1)', sum(_llh, 1), root.children)
end

function eval{T<:Real, U}(node::UnivariateNode{U}, data::AbstractArray{T}, llhvals::Dict{SPNNode, Array{Float64}})
  return eval(node, data)
end

"""
Evaluate Univariate Node.
This function returns the llh of the data under the model, the maximum a posterior (equal to log-likelihood), and itself.
"""
function eval{T<:Real, U<:DiscreteUnivariateDistribution}(node::UnivariateNode{U}, data::AbstractArray{T})
    if ndims(data) > 1
        x = sub(data, node.scope, :)

				if all(isnan(x))
					return ([0], [0], Array{SPNNode}(0))
				end

        llh = logpdf(node.dist, x)
        return (llh, llh, Array{SPNNode}(0))
    else
        llh = logpdf(node.dist, data)

        return (llh, llh, Array{SPNNode}(0))
    end

end

"""
Evaluate Univariate Node.
This function returns the llh of the data under the model, the maximum a posterior (equal to log-likelihood), and itself.
"""
function eval{T<:Real, U<:ContinuousUnivariateDistribution}(node::UnivariateNode{U}, data::AbstractArray{T})
    if ndims(data) > 1
        x = sub(data, node.scope, :)

				if all(isnan(x))
					return ([0], [0], Array{SPNNode}(0))
				end

        llh = logpdf(node.dist, x) - logpdf(node.dist, mean(node.dist))
        return (llh, llh, Array{SPNNode}(0))
    else
        llh = logpdf(node.dist, data) - logpdf(node.dist, mean(node.dist))

        return (llh, llh, Array{SPNNode}(0))
    end

end

"""
Evaluate Univariate Node.
This function returns the llh of the data under the model, the maximum a posterior (equal to llh), and itself.
"""
function eval{T<:Real, U<:ConjugatePostDistribution}(node::UnivariateNode{U}, data::AbstractArray{T})
    if ndims(data) > 1
        x = sub(data, node.scope, :)

				if all(isnan(x))
					return ([0], [0], Array{SPNNode}(0))
				end

        llh = logpred(node.dist, x)
        return (llh, llh, Array{SPNNode}(0))
    else
        llh = logpred(node.dist, data)
        return (llh, llh, Array{SPNNode}(0))
    end

end

function eval{T<:Real, U}(node::MultivariateNode{U}, data::AbstractArray{T}, llhvals::Dict{SPNNode, Array{Float64}})
  return eval(node, data)
end

"""
Evaluate Multivariate Node with ContinuousMultivariateDistribution.
This function returns the llh of the data under the model, the maximum a posterior (equal to llh), and itself.
"""

function eval{T<:Real, U<:ContinuousMultivariateDistribution}(node::MultivariateNode{U}, data::AbstractArray{T})
  llh = logpdf(node.dist, data[node.scope,:])
  return (llh, llh, Array{SPNNode}(0))
end

"""
Evaluate Multivariate Node with ConjugatePostDistribution.
This function returns the llh of the data under the model, the maximum a posterior (equal to llh), and itself.
"""
function eval{T<:Real, U<:ConjugatePostDistribution}(node::MultivariateNode{U}, data::AbstractArray{T})
  llh = logpred(node.dist, data[node.scope,:])
  return (llh, llh, Array{SPNNode}(0))
end
