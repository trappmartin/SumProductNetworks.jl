export SPNLayer,
    MultivariateFeatureLayer,
    SumLayer,
    BayesianSumLayer,
    ProductLayer,
    BayesianProductLayer,
    ProductCLayer,
    AbstractProductLayer,
    AbstractSumLayer,
    IndicatorLayer,
    BayesianCategoricalLayer

abstract type SPNLayer end
abstract type AbstractInternalLayer <: SPNLayer end
abstract type AbstractSumLayer <: AbstractInternalLayer end
abstract type AbstractBayesianLayer <: AbstractInternalLayer end

# Layer with Sum Nodes
type SumLayer <: AbstractSumLayer

    ids::Vector{Int}
    childIds::Matrix{Int} # Ch x C child ids
    logweights::Matrix{AbstractFloat} # Ch x C filter matrix

    children
    parent

end

# Layer with Bayesian Sum Nodes, weights collapsed out
type BayesianSumLayer <: AbstractBayesianLayer

    ids::Vector{Int}
    childIds::Matrix{Int} # Ch x C child ids
    sufficientStats::Matrix{Int} # Ch x C matrix with sufficient statistics
    activeObservations::SparseMatrixCSC{Bool, Int} # N x C matrix
    activeDimensions::SparseMatrixCSC{Bool, Int} # D x C matrix
    α::Matrix{AbstractFloat} # 1 x C matrix or Ch x C matrix

    children
    parent

    function BayesianSumLayer(ids::Vector{Int}, Ch::Int, N::Int, D::Int, α::AbstractFloat; parent_node = nothing)

        C = length(ids)
        childIds_ = zeros(Int, Ch, C)
        sufficientStats_ = zeros(Int, Ch, C)
        activeObservations_ = spzeros(Bool, N, C)
        activeDimensions_ = spzeros(Bool, D, C)
        α_ = ones(1, C) * (α / Ch)

        new(ids, childIds_, sufficientStats_, activeObservations_, activeDimensions_, α_, SPNLayer[], parent_node)
    end

end

abstract type AbstractProductLayer <: AbstractInternalLayer end

# Layer with Product Nodes
type ProductLayer <: AbstractProductLayer

    ids::Vector{Int}
    childIds::Matrix{Int} # Ch x C child ids

    children
    parent

end

# Layer with Bayesian Product Nodes, augmented weights collapsed out
type BayesianProductLayer <: AbstractBayesianLayer

    ids::Vector{Int}
    childIds::Matrix{Int} # Ch x C child ids
    sufficientStats::Matrix{Int} # Ch x C with sufficient statistics
    activeObservations::SparseMatrixCSC{Bool, Int} # N x C matrix
    activeDimensions::SparseMatrixCSC{Bool, Int} # N x C matrix
    β::Matrix{AbstractFloat} # 1 x C matrix or Ch x C matrix

    children
    parent

    function BayesianProductLayer(ids::Vector{Int}, Ch::Int, N::Int, D::Int, β::AbstractFloat; parent_node = nothing)

        C = length(ids)
        childIds_ = zeros(Int, Ch, C)
        sufficientStats_ = zeros(Int, Ch, C)
        activeObservations_ = spzeros(Bool, N, C)
        activeDimensions_ = spzeros(Bool, D, C)
        β_ = ones(1, C) * (β / Ch)

        new(ids, childIds_, sufficientStats_, activeObservations_, activeDimensions_, β_, SPNLayer[], parent_node)
    end

end

# Layer with Product Nodes, each equiped with a class label
type ProductCLayer <: AbstractProductLayer

    ids::Vector{Int}
    childIds::Matrix{Int} # Ch x C child ids
    clabels::Vector{Int} # C class labels

    children
    parent

end

abstract type AbstractLeafLayer <: SPNLayer end
abstract type AbstractBayesianLeafLayer <: AbstractLeafLayer end

# Layer with MultivariateFeature Nodes
type MultivariateFeatureLayer <: AbstractLeafLayer

    ids::Vector{Int}
    weights::Matrix{AbstractFloat} # C x D filter matrix
    scopes::Matrix{Bool} # C x D mask

    parent

end

# Layer with indicator nodes
type IndicatorLayer <: AbstractLeafLayer

    ids::Vector{Int} # flatten vector representing C x D ids matrix
    scopes::Vector{Int} # D dimensional vector
    values::Vector # C dimensional vector of values used for the indicator functions

    parent

end

# Layer with univariate Gauss distributions
type GaussianLayer <: AbstractLeafLayer

    ids::Vector{Int} # C dimensional vector
    scopes::Vector{Int} # C dimensional vector
    μ::Vector{Float32} # C dimensional vector of values location parameters
    σ::Vector{Float32} # C dimensional vector of values scale parameters

    parent

end

# Layer with categorical distributions with Dirichlet as prior
type BayesianCategoricalLayer <: AbstractBayesianLeafLayer

    ids::Vector{Int} # C dimensional vector
    scopes::Vector{Int} # C dimensional vector
    sufficientStats::Matrix{Int} # Ch x C with sufficient statistics
    activeObservations::SparseMatrixCSC{Bool, Int} # N x C matrix
    γ::Matrix{AbstractFloat} # 1 x C matrix or Ch x C matrix

    parent

    function BayesianCategoricalLayer(ids::Vector{Int}, scopes::Vector{Int}, S::Int, N::Int, γ::AbstractFloat; parent_node = nothing)

        C = length(ids)
        sufficientStats_ = zeros(Int, S, C)
        activeObservations_ = spzeros(Bool, N, C)
        γ_ = ones(1, C) * (γ / S)

        new(ids, scopes, sufficientStats_, activeObservations_, γ_, parent_node)
    end

end
