# abstract definition of a SPN node
abstract SPNNode
abstract Node <: SPNNode
abstract Leaf <: SPNNode

# definition of class indicater Node
type ClassNode <: Leaf

    class::Int

    ClassNode(class::Int) = new(class)
end

# definition of a Sum Node
type SumNode <: Node

  # SumNode fields
  uid::Int
  children::Vector{SPNNode}
  weights::Vector{Float32}

  # additional fields
  deepChildrenCount::Int

  SumNode(id::Int) = new(id, SPNNode[], Float64[], 0)
  SumNode(id::Int, children::Vector{SPNNode}, w::Vector{Float64}) = new(id, children, w, sum())

end

# definition of a Product Node
type ProductNode <: Node

  # ProductNode fields
  uid::Int32
  children::Vector{SPNNode}
  class::Nullable{ClassNode}

  ProductNode(id::Int) = new(id, SPNNode[], Nullable{ClassNode}())
  ProductNode(id::Int, class::ClassNode) = new(id, SPNNode[], Nullable(class))
  ProductNode(id::Int, children::Vector{SPNNode}) = new(id, children, Nullable{ClassNode}())
  ProductNode(id::Int, children::Vector{SPNNode}, class::ClassNode) = new(id, children, Nullable(class))

end

# definition of a Univariate Node
type UnivariateNode <: Leaf

  dist::UnivariateDistribution
  variable::Int

  UnivariateNode(D::UnivariateDistribution) = new(D, 0)
  UnivariateNode(D::UnivariateDistribution, var::Int) = new(D, var)

end

# definition of a Multivariate Node
type MultivariateNode{T} <: Leaf

  dist::T
  variables::Vector{Int}

  MultivariateNode{T}(D::T, vars::Vector{Int}) = new(D, vars)

end

## -------------------------------------------------- ##
## accessing function                                 ##
## -------------------------------------------------- ##

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
  parent
end

# add node
function add!(parent::ProductNode, child::SPNNode)
  push!(parent.children, child)
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

type SPNMarking
  ordering::Array{SPNNode}
  unmarked::Array{SPNNode}

end

# get topological order of SPN
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

# get number of children including (deep)
function deeplength(node::SPNNode)

    if isa(node, Leaf)
        return 1
    else
        return sum([deeplength(child) for child in node.children])
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

function llh{T<:Real}(root::Node, data::Array{T})
    # get topological order
    toporder = order(root)

    llhval = Dict{SPNNode, Array{Float64}}()

    for node in toporder
        llhval[node] = eval(node, data, llhval)
    end

    return llhval[toporder[end]]
end

function llh{T<:Real}(root::Leaf, data::Array{T})
    llhval = Dict{SPNNode, Array{Float64}}()
    return eval(root, data, llhval)
end

# evaluate SumNode
function eval{T<:Real}(root::SumNode, data::Array{T}, llhvals::Dict{SPNNode, Array{Float64}})

    _llh = [llhvals[c] for c in root.children]

    if ndims(data) != 1
        _llh = reduce(vcat, _llh)
        w = repmat( log(root.weights), 1, size(_llh, 2))

        _llh = _llh + w
    else
        _llh = reduce(hcat, _llh)
        w = repmat( log(root.weights)', size(_llh, 1), 1)

        _llh = _llh + w
        _llh = _llh'
    end

    maxlog = maximum(_llh, 1)

    _llh = _llh .- maxlog
    prob = sum(exp(_llh), 1)

    _llh = log(prob) .+ maxlog
    _llh -= log(sum(root.weights))

    return _llh
end

# evaluate ProductNode
function eval{T<:Real}(root::ProductNode, data::Array{T}, llhvals::Dict{SPNNode, Array{Float64}})
    _llh = [llhvals[c] for c in root.children]
    _llh = reduce(vcat, _llh)
    return sum(_llh, 1)
end

# evaluate Univariate Node
function eval{T<:Real}(node::UnivariateNode, data::Array{T}, llhvals::Dict{SPNNode, Array{Float64}})
    if ndims(data) > 1
        x = sub(data, node.variable, :)
        return logpdf(node.dist, x)
    else
        return logpdf(node.dist, data)
    end
end

"Evaluate Multivariate Node with ContinuousMultivariateDistribution."
function eval{T<:Real, U<:ContinuousMultivariateDistribution}(node::MultivariateNode{U}, data::Array{T}, llhvals::Dict{SPNNode, Array{Float64}})
    return logpdf(node.dist, data[node.variables])
end

"Evaluate Multivariate Node with ConjugatePostDistribution."
function eval{T<:Real, U<:ConjugatePostDistribution}(node::MultivariateNode{U}, data::Array{T}, llhvals::Dict{SPNNode, Array{Float64}})
  if ndims(data) < 2
      return collect(logpred(node.dist, data[node.variables]))
  else
      return logpred(node.dist, data[node.variables])
  end
end
