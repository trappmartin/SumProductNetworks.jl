using Distributions

# abstract definition of a SPN node
abstract SPNNode
abstract Node <: SPNNode
abstract Leaf <: SPNNode

# definition of a Sum Node
type SumNode <: Node

  # SumNode fields
  uid::Int                    # unique identifier
  children::Vector{SPNNode}   # children of sum node
  weights::Vector{Float32}    # weights / priors for children
   
  # additional fields
  deepChildrenCount::Int
    
  SumNode(id::Int) = new(id, SPNNode[], Float64[], 0)
    SumNode(id::Int, children::Vector{SPNNode}, w::Vector{Float64}) = new(id, children, w, sum())

end

# definition of a Product Node
type ProductNode <: Node

  # ProductNode fields
  uid::Int32                 # unique identifier
  children::Vector{SPNNode}  # children of product node
    
  ProductNode(id::Int) = new(id, SPNNode[])
  ProductNode(id::Int, children::Vector{SPNNode}) = new(id, children)

end

# definition of a Univariate Node
type UnivariateNode <: Leaf

  dist::UnivariateDistribution
    
  UnivariateNode(D::UnivariateDistribution) = new(D)

end

# definition of a Multivariate Node
type MultivariateNode <: Leaf

  dist::MultivariateDistribution
    
  MultivariateNode(D::MultivariateDistribution) = new(D)

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

# get topological order of SPN
function order(root::Node)

    function visit!(node::SPNNode, data::(Array{SPNNode}, Array{SPNNode}))

        if node in data[1]

            if isa(node, Node)
                for n in node.children
                    data = visit!(n, data)
                end
            end
            idx = findfirst(data[1], node)
            splice!(data[1], idx)
            push!(data[2], node)
        end

        data
    end

    N = deeplength(root)

    ordering = SPNNode[]
    unmarked = SPNNode[]
    flat!(unmarked, root)

    while(Base.length(unmarked) > 0)

        n = unmarked[end]
        (unmarked, ordering) = visit!(n, (unmarked, ordering))
    end

    ordering
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

function llh{T<:Real}(root::SumNode, data::Array{T})
    # get topological order
    toporder = SPN.order(root)
    
    llhval = Dict{SPNNode, Array{Float64}}()
    
    for node in toporder
        llhval[node] = eval(node, data, llhval)
    end
    
    return llhval[toporder[end]]
end

# evaluate SumNode
function eval{T<:Real}(root::SumNode, data::Array{T}, llhvals::Dict{SPNNode, Array{Float64}})

    _llh = [llhvals[c] for c in root.children]
    _llh = reduce(vcat, _llh)
    
    if ndims(data) != 1
        w = repmat( log(root.weights)', size(_llh, 1), 1)
    else
        w = log(root.weights)
    end
  
    _llh = _llh + w
    
    maxlog = maximum(_llh)
    _llh -= maxlog
    prob = sum(exp(_llh), 1)

    _llh = log(prob) + maxlog
    _llh -= log(sum(root.weights))
    
    return _llh
end

# evaluate ProductNode
function eval{T<:Real}(root::ProductNode, data::Array{T}, llhvals::Dict{SPNNode, Array{Float64}})
  _llh = [llhvals[c] for c in root.children]
  return sum(_llh, 2)
end

# evaluate Univariate Node
function eval{T<:Real}(node::UnivariateNode, data::Array{T}, llhvals::Dict{SPNNode, Array{Float64}})
  return logpdf(node.dist, data)
end

# evaluate Univariate Node
function llh{T<:Real}(node::MultivariateNode, data::Array{T}, llhvals::Dict{SPNNode, Array{Float64}})
  return logpdf(node.dist, data)
end
