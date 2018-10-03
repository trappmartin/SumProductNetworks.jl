export SumProductNetwork
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

struct SumProductNetwork
    root::SumNode
    nodes::Vector{SPNNode}
    leaves::Vector{Leaf}
    idx::Dict{Symbol,Int}
    topological_order::Vector{Int}
end

Base.keys(spn::SumProductNetwork) = keys(spn.idx)
Base.values(spn::SumProductNetwork) = spn.nodes
Base.getindex(spn::SumProductNetwork, i...) = getindex(spn.nodes, spn.idx[i...])
Base.setindex!(spn::SumProductNetwork, v, i...) = setindex!(spn.nodes, v, spn.idx[i...])
Base.length(spn::SumProductNetwork) = length(spn.nodes)
function Base.show(io::IO, spn::SumProductNetwork)
    println(io, summary(spn))
    println(io, "\t#nodes = $(length(spn))")
    println(io, "\t#leaves = $(length(spn.leaves))")
    println(io, "\troot = $(spn.root.id)")
end

# A finite sum node.
mutable struct FiniteSumNode{T <: Real} <: SumNode{T}
    id::Symbol
    parents::Vector{SPNNode}
    children::Vector{SPNNode}
    logweights::Vector{T}
    α::Union{Float64, Vector{Float64}}
    scopeVec::BitArray{1}
    obsVec::BitArray{1}
end

function FiniteSumNode{T}(;D = 0, N = 0, parents = SPNNode[], α = 1.) where T <: Real
    return FiniteSumNode{T}(gensym(), parents, SPNNode[], T[], α, falses(D), falses(N))
end

function header(node::SPNNode)
    return "$(summary(node))($(node.id))"
end

function Base.show(io::IO, node::FiniteSumNode)
    println(io, header(node))
    println(io, "\tparents = $(map(p -> header(p), node.parents))")
    println(io, "\tchildren = $(map(c -> header(c), node.children))")
    println(io, "\t(log) weights = $(node.logweights)")
    println(io, "\tweights = $(exp.(node.logweights))")
    println(io, "\tnormalized = $(isnormalized(node))")
    println(io, "\talpha = $(node.α)")
    println(io, "\tscope = $(findall(node.scopeVec))")
    println(io, "\tassigns = $(findall(node.obsVec))")
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

function Base.show(io::IO, node::FiniteProductNode)
    println(io, header(node))
    println(io, "\tparents = $(map(p -> header(p), node.parents))")
    println(io, "\tchildren = $(map(c -> header(c), node.children))")
end

mutable struct FiniteAugmentedProductNode{T <: Real} <: ProductNode
    id::Symbol
    parents::Vector{SPNNode}
    children::Vector{SPNNode}
    logomega::Vector{T}
    α::Union{Float64, Vector{Float64}}
    scopeVec::BitArray{1}
    obsVec::BitArray{1}
end

function FiniteAugmentedProductNode{T}(; D = 0, N = 0, parents = SPNNode[]) where T<:Real
    return FiniteAugmentedProductNode{T}(
                               gensym(),
                               parents,
                               SPNNode[],
                               T[],
                               1.,
                               falses(D),
                               falses(N)
    )
end

function Base.show(io::IO, node::FiniteAugmentedProductNode)
    if get(io, :compact, true)
        print(io, """augmented product node ($(node.id) : parents = $(map(p -> p.id, node.parents)), 
              children = $(map(c -> c.id, node.children))""")
    else
        println(io, "augmented product node ($(node.id))")
        println(io, "\tparents = $(map(p -> p.id, node.parents))")
        println(io, "\tchildren = $(map(c -> c.id, node.children))")
    end
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

function Base.isequal(n1::IndicatorNode, n2::IndicatorNode)
    return (n1.value == n2.value) && (n1.scope == n2.scope)
end

function Base.show(io::IO, node::IndicatorNode)
    println(io, "indicator node ($(node.id))")
    println(io, "\tparents = $(map(p -> p.id, node.parents))")
    println(io, "\tfunction = 1(x[$(node.scope)] = $(node.value))")
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

function Base.show(io::IO, node::UnivariateNode)
    if get(io, :compact, true)
        print(io, """univariate node ($(node.id) : parents = $(map(p -> p.id, node.parents)), 
              distribution function = $(node.dist)""")
    else
        println(io, "univariate node ($(node.id))")
        println(io, "\tparents = $(map(p -> p.id, node.parents))")
        println(io, "\tdistribution function = $(node.dist)")
    end
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
