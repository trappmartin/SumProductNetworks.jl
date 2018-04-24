export SPNLayer,
    MultivariateFeatureLayer,
    SumLayer,
    BayesianSumLayer,
    ProductLayer,
    BayesianProductLayer,
    ProductCLayer,
    AbstractProductLayer,
    AbstractSumLayer,
    IndicatorLayer

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
    α::Matrix{AbstractFloat} # 1 x C matrix or Ch x C matrix

    children
    parent

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
    β::Matrix{AbstractFloat} # 1 x C matrix or Ch x C matrix

    children
    parent

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
    γ::Matrix{AbstractFloat} # 1 x C matrix or Ch x C matrix

    parent

end
