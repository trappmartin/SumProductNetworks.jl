# abstract definition of a SPN node
abstract SPNNode

# definition of a Sum Node
type SumNode <: SPNNode

  # SumNode fields
  uid::Int                    # unique identifier
  children::Vector{SPNNode}   # children of sum node
  weights::Vector{Float64}    # weights / priors for children

end

# definition of a Product Node
type ProductNode <: SPNNode

  # ProductNode fields
  uid::Int32                 # unique identifier
  children::Vector{SPNNode}  # children of product node

end

# definition of a Univariate Node
immutable UnivariateNode <: SPNNode

  dist::UnivariateDistribution
  variables::Vector{Int}

end

# definition of a Multivariate Node
immutable MultivariateNode <: SPNNode

  dist::MultivariateDistribution
  variables::Vector{Int}

end

## -------------------------------------------------- ##
## accessing function                                 ##
## -------------------------------------------------- ##

# build nodes
function build_sum(id)
  return SumNode(id, SPNNode[], Float64[])
end

function build_sum(id, children::Vector{SPNNode})
  w = rand(length(children))
  w /= sum(w)

  return SumNode(id, children, w)
end

function build_sum(id, children::Vector{SPNNode}, w::Vector{Float64})
  return SumNode(id, children, w)
end

function build_prod(id)
  return ProductNode(id, SPNNode[])
end

function build_prod(id, children::Vector{SPNNode})
  return ProductNode(id, children)
end

# build UnivariateNode
function build_univariate(D::UnivariateDistribution, var::Vector{Int})
  return UnivariateNode(D, var)
end

# build MultivariateNode
function build_multivariate(D::MultivariateDistribution, var::Vector{Int})
  return MultivariateNode(D, var)
end

# normalize sum node
function normalize(node::SumNode)
  node.weights /= sum(node.weights)
end

# add node
function add(parent::SumNode, child::SPNNode)
  add(parent, child, rand())
end

# add node with weight
function add(parent::SumNode, child::SPNNode, weight::Float64)
  push!(parent.children, child)
  push!(parent.weights, weight)
  return length(parent.children)
end

# add node
function add(parent::ProductNode, child::SPNNode)
  push!(parent.children, child)
  return length(parent.children)
end

# remove node with index
function remove(parent::SumNode, index::Integer)
  deleteat!(parent.children, index)
  deleteat!(parent.weights, index)
end

# remove node with index
function remove(parent::ProductNode, index::Integer)
  deleteat!(parent.children, index)
end

# evaluate SumNode
function llh{T<:Real}(root::SumNode, data::Array{T})

  _llh = mapreduce( x -> llh(x, data), hcat, root.children )

  w = repmat( log(root.weights)', size(_llh, 1), 1)
  _llh = _llh + w

  maxlog = maximum(_llh)
  _llh -= maxlog
  prob = sum(exp(_llh), 2)

  _llh = log(prob) + maxlog
  _llh -= log(sum(root.weights))
  return _llh
end

# evaluate SumNode
function llh_map{T<:Real}(root::SumNode, data::Array{T})

  _llh = mapreduce( x -> llh(x, data), hcat, root.children )

  w = repmat( log(root.weights)', size(_llh, 1), 1)
  _llh = _llh + w

  maxlog = maximum(_llh)
  _llh -= maxlog
  prob = maximum(exp(_llh))

  _llh = log(prob) + maxlog

  return _llh
end

# evaluate ProductNode
function llh{T<:Real}(root::ProductNode, data::Array{T})
  _llh = mapreduce( x -> llh(x, data), hcat, root.children )
  return sum(_llh, 2)
end

# evaluate Univariate Node
function llh{T<:Real}(node::UnivariateNode, data::Array{T})
  x = data[:,node.variables]
  return vec(logpdf(node.dist, x))
end

# evaluate Univariate Node
function llh{T<:Real}(node::MultivariateNode, data::Array{T})
  x = data[:,node.variables]
  return vec(logpdf(node.dist, x'))
end
