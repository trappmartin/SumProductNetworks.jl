export SPNNode, Node, Leaf, SumNode, ProductNode
export FiniteSumNode, FiniteProductNode
export InfiniteSumNode, InfiniteProductNode
export IndicatorNode, UnivariateNode, NormalDistributionNode
export MultivariateNode

# abstract definition of an SumProductNetwork node
abstract type SPNNode end
abstract type Node <: SPNNode end
abstract type SumNode{T} <: Node end
abstract type ProductNode <: Node end
abstract type Leaf{T} <: SPNNode end

#
# A finite sum node.
#
struct FiniteSumNode{T <: Real} <: SumNode{T}

    # * immutable fields * #
    id::Int

    # * mutable fields * #
    parents::Vector{SPNNode}
    children::Vector{SPNNode}
    weights::Vector{T}
    scope::Vector{Int}

    function FiniteSumNode{T}(id::Int, scope::Vector{Int}; parents = SPNNode[]) where T <: Real

        if id < 1
            error("invalid id, expecting id >= 1")
        end

        if isempty(scope)
            error("invalid value for node scope")
        end
        new(id, parents, SPNNode[], T[], scope)
    end
end

#
# An infinite sum node.
#
struct InfiniteSumNode{T <: Real} <: SumNode{T}

    # * immutable fields * #
    id::Int

    # * mutable fields * #
    parents::Vector{SPNNode}
    children::Vector{SPNNode}
    α::T
    πremain::T
    π::Vector{T}
    scope::Vector{Int}

    function InfiniteSumNode{T}(id::Int, scope::Vector{Int}; parents = SPNNode[], α = one(T)) where T <: Real
        if id < 1
            error("invalid id, expecting id >= 1")
        end

        if isempty(scope)
            error("invalid value for node scope")
        end
        
        if α == 0
            error("invalid value for alpha")
        end

        new(id, parents, SPNNode[], α, one(T), Vector{T}(0), scope)
    end
end

#
# A finite product node.
#
struct FiniteProductNode <: ProductNode

    # * immutable fields * #
    id::Int

    # * mutable fields * #
	parents::Vector{SPNNode}
    children::Vector{SPNNode}
    scope::Vector{Int}

    function FiniteProductNode(id::Int, scope::Vector{Int}; parents = SPNNode[])
        if id < 1
            error("invalid id, expecting id >= 1")
        end

        if isempty(scope)
            error("invalid value for node scope")
        end
        new(id, parents, SPNNode[], scope)
    end
end

#
# An infinite product node.
#
struct InfiniteProductNode{T <: Real} <: ProductNode

    # * immutable fields * #
    id::Int

    # * mutable fields * #
	parents::Vector{SPNNode}
    children::Vector{SPNNode}
    α::T
    ωremain::T
    ω::Vector{T}
    scope::Vector{Int}

    function InfiniteProductNode{T}(id; parents = SPNNode[], scope = Int[]) where T <: Real
        if id < 1
            error("invalid id, expecting id >= 1")
        end

        if isempty(scope)
            error("invalid value for node scope")
        end
        
        if α == 0
            error("invalid value for alpha")
        end

        new(id, parents, SPNNode[], α, one(T), Vector{T}(0), scope)
    end
end

# definition of indicater Node
type IndicatorNode{T <: Integer} <: Leaf{T}

    # * immutable fields * #
    id::Int
    value::T
    scope::Int

    # * mutable fields * #
	parents::Vector{SPNNode}

    function IndicatorNode{T}(id::Int, value::T, scope::Int; parents = SPNNode[]) where T <: Integer
        if id < 1
            error("invalid id, expecting id >= 1")
        end

        if isempty(scope)
            error("invalid value for node scope")
        end
        
        new(id, value, scope, parents)
    end
end

#
# A univariate node computes the likelihood of x under a univariate distribution.
#
type UnivariateNode{T} <: Leaf{Any}

    # unique node identifier
    id::Int

    # Fields
	parents::Vector{SPNNode}
    dist::T
    scope::Int

    function UnivariateNode{T}(id::Int, distribution::T, scope::Int; parents = SPNNode[]) where T <: Any 
        if id < 1
            error("invalid id, expecting id >= 1")
        end

        if scope < 1
            error("invalid value for node scope")
        end
        new(id, parents, distribution, scope)
    end
end

type NormalDistributionNode <: Leaf{Any}

    # unique node identifier
    id::Int

    # Fields
	parents::Vector{SPNNode}
    μ::Float64
    σ::Float64
    scope::Int

    function NormalDistributionNode(id::Int, scope::Int; parents = SPNNode[], μ = 0.0, σ = 1.0)
        if id < 1
            error("invalid id, expecting id >= 1")
        end

        if scope < 1
            error("invalid value for node scope")
        end
        new(id, parents, μ, σ, scope)
    end
end

#
# A multivariate node computes the likelihood of x under a multivariate distribution.
#
type MultivariateNode{T} <: Leaf{Any}

    # unique node identifier
    id::Int

    # Fields
	parents::Vector{SPNNode}
    dist::T
    scope::Vector{Int}

    function MultivariateNode{T}(id::Int, distribution::T, scope::Vector{Int}; parents = SPNNode[]) where T <: Any
        if id < 1
            error("invalid id, expecting id >= 1")
        end

        if isempty(scope)
            error("invalid value for node scope")
        end
        new(id, parents, distribution, scope)
    end

end
