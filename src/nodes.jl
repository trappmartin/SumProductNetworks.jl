# abstract definition of an SPN node
abstract SPNNode
abstract Node <: SPNNode
abstract Leaf{T} <: SPNNode

# definition of class indicater Node
type ClassNode <: Leaf
    class::Int
    ClassNode(class::Int) = new(class)
end

#
# A sum node computes a weighted sum of its children.
#
type SumNode <: Node

  # unique node identifier
  id::Int

  # Fields
	inSPN::Bool
	parents::Vector{SPNNode}
  children::Vector{SPNNode}
  weights::Vector{Float64}
  isFilter::Bool

  scope::Vector{Int}

  SumNode(id; parents = SPNNode[], scope = Int[]) = new(id, false, parents, SPNNode[], Float64[], false, scope)
  SumNode(id, children::Vector{SPNNode}, weights::Vector{Float64}; parents = SPNNode[], scope = Int[]) = new(id, false, parents, children, weights, false, scope)

end

#
# A product node computes a the product of its children.
#
type ProductNode <: Node

  # unique node identifier
  id::Int

  # Fields
	inSPN::Bool
	parents::Vector{SPNNode}
  children::Vector{SPNNode}
  classes::Vector{ClassNode}

  scope::Vector{Int}

  ProductNode(id; parents = SPNNode[], children = SPNNode[], classes = ClassNode[], scope = Int[]) = new(id, false, parents, children, classes, scope)
end

@doc doc"""
A univariate node computes the likelihood of x under a univariate distribution.
""" ->
type UnivariateFeatureNode <: Leaf

  # unique node identifier
  id::Int

  # Fields
	inSPN::Bool
	parents::Vector{SPNNode}
  bias::Bool
  scope::Int

  UnivariateFeatureNode(id, scope::Int; parents = SPNNode[], isbias = false) = new(id, false, parents, isbias, scope)
end

@doc doc"""
A univariate node computes the likelihood of x under a univariate distribution.
""" ->
type UnivariateNode{T} <: Leaf

  # unique node identifier
  id::Int

  # Fields
	inSPN::Bool
	parents::Vector{SPNNode}
  dist::T
  scope::Int

  UnivariateNode{T}(id, distribution::T, scope::Int; parents = SPNNode[]) = new(id, false, parents, distribution, scope)
end

type NormalDistributionNode <: Leaf

  # unique node identifier
  id::Int

  # Fields
	inSPN::Bool
	parents::Vector{SPNNode}
  μ::Float64
  σ::Float64
  logz::Float64
  scope::Int

  NormalDistributionNode(id, scope::Int; parents = SPNNode[], μ = 0.0, σ = 1.0, logz = 0.0) = new(id, false, parents, μ, σ, logz, scope)
end

@doc doc"""
A multivariate node computes the likelihood of x under a multivariate distribution.
""" ->
type MultivariateNode{T} <: Leaf

  # unique node identifier
  id::Int

  # Fields
	inSPN::Bool
	parents::Vector{SPNNode}
  dist::T
  scope::Vector{Int}

  MultivariateNode{T}(id, distribution::T, scope::Vector{Int}; parents = SPNNode[]) = new(id, false, parents, distribution, scope)

end
