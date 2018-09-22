export SPNNode, Node, Leaf, SumNode, ProductNode
export FiniteSumNode, FiniteProductNode, FiniteAugmentedProductNode
export InfiniteSumNode, InfiniteProductNode
export IndicatorNode, UnivariateNode 
export MultivariateNode

# Abstract definition of an SumProductNetwork node.
abstract type SPNNode end
abstract type Node <: SPNNode end
abstract type SumNode{T} <: Node end
abstract type ProductNode <: Node end
abstract type Leaf <: SPNNode end

# A finite sum node.
mutable struct FiniteSumNode{T <: Real} <: SumNode{T}
    id::Symbol
    parents::Vector{SPNNode}
    children::Vector{SPNNode}
    logweights::Vector{T}
    α::Float64
    scopeVec::BitArray{1}
    obsVec::BitArray{1}
end

function FiniteSumNode{T}(;D = 0, N = 0, parents = SPNNode[], α = 1.) where T <: Real
    return FiniteSumNode(gensym(), parents, SPNNode[], T[], α, falses(D), falses(N))
end

# A finite split node.
mutable struct FiniteSplitNode <: ProductNode
    id::Symbol
    parents::Vector{SPNNode}
    children::Vector{SPNNode}
    split::Vector{Float64}
end

function FiniteSplitNode(split::Vector{Float64}; parents = SPNNode[])
    return FiniteProductNode(gensym(), parents, SPNNode[], split, Int[])
end

# A finite product node.
struct FiniteProductNode <: ProductNode
    id::Symbol
    parents::Vector{SPNNode}
    children::Vector{SPNNode}
    scopeVec::BitArray{1}
    obsVec::BitArray{1}
end

function FiniteProductNode(;D = 0, N = 0, parents = SPNNode[])
    return FiniteProductNode(gensym(), parents, SPNNode[], falses(D), falses(N))
end

mutable struct FiniteAugmentedProductNode{T <: Real} <: ProductNode
    id::Symbol
    parents::Vector{SPNNode}
    children::Vector{SPNNode}
    logomega::Vector{T}
    scopeVec::BitArray{1}
    obsVec::BitArray{1}
end

function FiniteAugmentedProductNode{T}(; D = 0, N = 0, parents = SPNNode[]) where T<:Real
    return FiniteAugmentedProductNode(
                               gensym(),
                               parents,
                               SPNNode[],
                               Int[],
                               T[],
                               falses(D),
                               falses(N)
    )
end

# Definition of an indicater Node.
mutable struct IndicatorNode <: Leaf
    id::Symbol
    value::Int
    scope::Int
    parents::Vector{SPNNode}
end

function IndicatorNode(value::Int, dim::Int; parents = SPNNode[])
    return IndicatorNode(gensym(), value, dim, parents)
end

# A univariate node computes the likelihood of x under a univariate distribution.
mutable struct UnivariateNode <: Leaf
    id::Symbol
    parents::Vector{SPNNode}
    dist::UnivariateDistribution
    scope::Int
end

function UnivariateNode(distribution::T, dim::Int; parents = SPNNode[]) where {T<:UnivariateDistribution}
    return UnivariateNode(gensym(), parents, distribution, dim)
end

# A multivariate node computes the likelihood of x under a multivariate distribution.
mutable struct MultivariateNode <: Leaf
    id::Symbol
    parents::Vector{SPNNode}
    dist::MultivariateDistribution
    scope::Vector{Int}
end

function MultivariateNode(distribution::T, dims::Vector{Int}; parents = SPNNode[]) where {T<:MultivariateDistribution}
    return MultivariateNode(gensym(), parents, distribution, dims)
end
