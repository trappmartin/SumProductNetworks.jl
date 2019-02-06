export SumProductNetwork
export SPNNode, Node, Leaf, SumNode, ProductNode
export FiniteSumNode, FiniteProductNode
export IndicatorNode, UnivariateNode 
export MultivariateNode

# Abstract definition of an SumProductNetwork node.
abstract type SPNNode end
abstract type Node <: SPNNode end
abstract type SumNode <: Node end
abstract type ProductNode <: Node end
abstract type Leaf <: SPNNode end

header(node::SPNNode) = "$(summary(node))($(node.id))"

function Base.show(io::IO, node::SPNNode)
    println(io, header(node))
    if hasweights(node)
        println(io, "\tweights = $(weights(node))")
        println(io, "\tnormalized = $(isnormalized(node))")
    end
    if hasscope(node)
        println(io, "\tscope = $(scope(node))")
    else
        println(io, "\tNo scope set!")
    end
    if hasobs(node)
        println(io, "\tassigns = $(obs(node))")
    end
end

function Base.show(io::IO, node::Leaf)
    println(io, header(node))
    println(io, "\tscope = $(scope(node))")
    println(io, "\tparameters = $(params(node))")
end

struct SumProductNetwork
    root::Node
    nodes::Vector{<:SPNNode}
    leaves::Vector{<:SPNNode}
    idx::Dict{Symbol,Int}
    topological_order::Vector{Int}
    layers::Vector{AbstractVector{SPNNode}}
    info::Dict{Symbol,<:Real}
end

function SumProductNetwork(root::Node)
    nodes = getOrderedNodes(root)
    leaves = filter(n -> isa(n, Leaf), nodes)
    idx = Dict(n.id => indx for (indx, n) in enumerate(nodes))
    toporder = collect(1:length(nodes))

    maxdepth = depth(root)
    nodedepth = map(n -> depth(n), nodes)
    layers = Vector{Vector{SPNNode}}(undef, maxdepth+1)
    for d in 0:maxdepth
        layers[d+1] = nodes[findall(nodedepth .== d)]
    end

    return SumProductNetwork(root, nodes, leaves, idx, toporder, layers, Dict{Symbol, Real}())
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
    println(io, "\tdepth = $(length(spn.layers))")
end

"""
   FiniteSumNode <: SumNode
   
A sum node computes a convex combination of its weight and the pdf's of its children.

## Usage:

```julia
node = FiniteSumNode{Float64}(;D=4, N=1)
add!(node, ..., log(0.5))
add!(node, ..., log(0.5))
logpdf(node, rand(4))
```

"""
mutable struct FiniteSumNode{T<:Real} <: SumNode
    id::Symbol
    parents::Vector{<:Node}
    children::Vector{<:SPNNode}
    logweights::Vector{T}
    scopeVec::BitArray{1}
    obsVec::BitArray{1}
end

function FiniteSumNode{T}(;D=1, N=1, parents::Vector{<:Node}=Node[]) where {T<:Real}
    return FiniteSumNode{T}(gensym(:sum), parents, SPNNode[], T[], falses(D), falses(N))
end
function FiniteSumNode(;D=1, N=1, parents::Vector{<:Node}=Node[])
    return FiniteSumNode{Float64}(;D=D, N=N, parents=parents)
end

eltype(::Type{FiniteSumNode{T}}) where {T<:Real} = T
eltype(n::SPNNode) = eltype(typeof(n))

"""
   FiniteProductNode <: ProductNode
   
A product node computes a product of the pdf's of its children.

## Usage:

```julia
node = FiniteProductNode(;D=4, N=1)
add!(node, ...)
add!(node, ...)
logpdf(node, rand(4))
```

"""
mutable struct FiniteProductNode <: ProductNode
    id::Symbol
    parents::Vector{<:Node}
    children::Vector{<:SPNNode}
    scopeVec::BitArray{1}
    obsVec::BitArray{1}
end

function FiniteProductNode(;D=1, N=1, parents::Vector{<:Node} = Node[])
    return FiniteProductNode(gensym(:prod), parents, SPNNode[], falses(D), falses(N))
end

"""
   IndicatorNode <: Leaf
   
An indicator node evaluates an indicator function.

## Usage:

```julia
value = 1
dimension = 1
node = IndicatorNode(value, dimension)
logpdf(node, [1, 2, 1]) # == 0.0
logpdf(node, [2, 2, 1]) # == -Inf
```

"""
mutable struct IndicatorNode <: Leaf
    id::Symbol
    value::Int
    scope::Int
    parents::Vector{<:Node}
end

function IndicatorNode(value::Int, dim::Int; parents::Vector{<:Node} = Node[])
    return IndicatorNode(gensym(:indicator), value, dim, parents)
end

function Base.isequal(n1::IndicatorNode, n2::IndicatorNode)
    return (n1.value == n2.value) && (n1.scope == n2.scope)
end
params(n::IndicatorNode) = (n.value)

"""
   UnivariateNode <: Leaf
   
A univariate node evaluates the logpdf of its univariate distribution function.

## Usage:

```julia
distribution = Normal()
dimension = 1
node = UnivariateNode(distribution, dimension)
logpdf(node, rand(2)) # == logpdf(Normal(), rand(2))
```

"""
mutable struct UnivariateNode <: Leaf
    id::Symbol
    parents::Vector{<:Node}
    dist::UnivariateDistribution
    scope::Int
end

function UnivariateNode(distribution::T, dim::Int; parents::Vector{<:Node} = Node[]) where {T<:UnivariateDistribution}
    return UnivariateNode(gensym(:univ), parents, distribution, dim)
end
params(n::UnivariateNode) = Distributions.params(n.dist)

"""
   MultivariateNode <: Leaf
   
A multivariate node evaluates the logpdf of its multivariate distribution function.

## Usage:

```julia
distribution = MvNormal()
dimensions = [1, 2]
node = UnivariateNode(distribution, dimensions)
logpdf(node, rand(2)) # == logpdf(MvNormal(), rand(2))
```

"""
mutable struct MultivariateNode <: Leaf
    id::Symbol
    parents::Vector{<:Node}
    dist::MultivariateDistribution
    scope::Vector{Int}
end

function MultivariateNode(distribution::T, dims::Vector{Int}; parents::Vector{<:Node} = Node[]) where {T<:MultivariateDistribution}
    return MultivariateNode(gensym(:multiv), parents, distribution, dims)
end
params(n::MultivariateNode) = Distributions.params(n.dist)
