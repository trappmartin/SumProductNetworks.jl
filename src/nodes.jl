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

  # Fields
	inSPN::Bool
	parents::Vector{SPNNode}
  children::Vector{SPNNode}
  weights::Vector{Float64}
  isFilter::Bool

  scope::Vector{Int}

  SumNode(; parents = SPNNode[], scope = Int[]) = new(false, parents, SPNNode[], Float64[], false, scope)
  SumNode(children::Vector{SPNNode}, weights::Vector{Float64}; parents = SPNNode[], scope = Int[]) = new(false, parents, children, weights, false, scope)

end

#
# A product node computes a the product of its children.
#
type ProductNode <: Node

  # Fields
	inSPN::Bool
	parents::Vector{SPNNode}
  children::Vector{SPNNode}
  classes::Vector{ClassNode}

  scope::Vector{Int}

  ProductNode(; parents = SPNNode[], children = SPNNode[], classes = ClassNode[], scope = Int[]) = new(false, parents, children, classes, scope)
end

@doc doc"""
A univariate node computes the likelihood of x under a univariate distribution.
""" ->
type UnivariateFeatureNode <: Leaf

	inSPN::Bool
	parents::Vector{SPNNode}
  bias::Bool
  scope::Int

  UnivariateFeatureNode(scope::Int; parents = SPNNode[], isbias = false) = new(false, parents, isbias, scope)
end

@doc doc"""
A univariate node computes the likelihood of x under a univariate distribution.
""" ->
type UnivariateNode{T} <: Leaf

	inSPN::Bool
	parents::Vector{SPNNode}
  dist::T
  scope::Int

  UnivariateNode{T}(distribution::T, scope::Int; parents = SPNNode[]) = new(false, parents, distribution, scope)
end

@doc doc"""
A multivariate node computes the likelihood of x under a multivariate distribution.
""" ->
type MultivariateNode{T} <: Leaf

	inSPN::Bool
	parents::Vector{SPNNode}
  dist::T
  scope::Vector{Int}

  MultivariateNode{T}(distribution::T, scope::Vector{Int}; parents = SPNNode[]) = new(false, parents, distribution, scope)

end
