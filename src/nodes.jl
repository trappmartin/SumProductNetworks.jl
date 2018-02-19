export SPNNode, Node, Leaf, SumNode, ProductNode
export FiniteSumNode, FiniteProductNode, FiniteAugmentedProductNode
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
mutable struct FiniteSumNode{T <: Real} <: SumNode{T}

	# * immutable fields * #
	id::Int

	# * mutable fields * #
	parents::Vector{SPNNode}
	children::Vector{SPNNode}
    cids::Vector{Int}
	logweights::Vector{T}

	α::Float64

	scopeVec::BitArray
	obsVec::BitArray

	function FiniteSumNode{T}(id::Int, D::Int, N::Int; parents = SPNNode[], α = 1.) where T <: Real

		if id < 1
			error("invalid id, expecting id >= 1")
		end

		@assert D >= 1
		@assert N >= 0

        new(id, parents, SPNNode[], Int[], T[], α, falses(D), falses(N))
	end
end

#
# A finite split node.
#
mutable struct FiniteSplitNode <: ProductNode

	# * immutable fields * #
	id::Int

	# * mutable fields * #
	parents::Vector{SPNNode}
	children::Vector{SPNNode}
	split::Vector{Float64}

	function FiniteProductNode(id::Int, split::Vector{Float64}; parents = SPNNode[])
		if id < 1
			error("invalid id, expecting id >= 1")
		end

		new(id, parents, SPNNode[], split, Int[])
	end
end

#
# A finite product node.
#
mutable struct FiniteProductNode <: ProductNode

	# * immutable fields * #
	id::Int

	# * mutable fields * #
	parents::Vector{SPNNode}
	children::Vector{SPNNode}
    cids::Vector{Int}

	scopeVec::BitArray
	obsVec::BitArray

	function FiniteProductNode(id::Int, D::Int, N::Int; parents = SPNNode[])
		if id < 1
			error("invalid id, expecting id >= 1")
		end

		@assert D >= 1
		@assert N >= 0

        new(id, parents, SPNNode[], Int[], falses(D), falses(N))
	end
end

mutable struct FiniteAugmentedProductNode{T <: Real} <: ProductNode

	# * immutable fields * #
	id::Int

	# * mutable fields * #
	parents::Vector{SPNNode}
	children::Vector{SPNNode}
    cids::Vector{Int}
    logomega::Vector{T}

	scopeVec::BitArray
	obsVec::BitArray

    function FiniteAugmentedProductNode{T}(id::Int, D::Int, N::Int; parents = SPNNode[]) where T <: Real
		if id < 1
			error("invalid id, expecting id >= 1")
		end

        new(id, parents, SPNNode[], Int[], T[], falses(D), falses(N))
	end
end

# definition of indicater Node
type IndicatorNode{T <: Integer} <: Leaf{T}

	# * immutable fields * #
	id::Int
	value::T
	scopeVec::BitArray

	# * mutable fields * #
	parents::Vector{SPNNode}

	function IndicatorNode{T}(id::Int, value::T, D::Int; parents = SPNNode[]) where T <: Integer
		if id < 1
			error("invalid id, expecting id >= 1")
		end

		new(id, value, falses(D), parents)
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
	scopeVec::BitArray

	function UnivariateNode{T}(id::Int, distribution::T, D::Int; parents = SPNNode[]) where T <: Any
		if id < 1
			error("invalid id, expecting id >= 1")
		end
		new(id, parents, distribution, falses(D))
	end
end

type NormalDistributionNode <: Leaf{Any}

	# unique node identifier
	id::Int

	# Fields
	parents::Vector{SPNNode}
	μ::Float64
	σ::Float64
	scopeVec::BitArray

	function NormalDistributionNode(id::Int, D::Int; parents = SPNNode[], μ = 0.0, σ = 1.0)
		if id < 1
			error("invalid id, expecting id >= 1")
		end
		new(id, parents, μ, σ, falses(D))
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
	scopeVec::BitArray

	function MultivariateNode{T}(id::Int, distribution::T, D::Int; parents = SPNNode[]) where T <: Any
		if id < 1
			error("invalid id, expecting id >= 1")
		end
		new(id, parents, distribution, falses(D))
	end

end
