export SPNLayer, MultivariateFeatureLayer, SumLayer, ProductLayer, ProductCLayer, AbstractProductLayer, IndicatorLayer, GaussianLayer

abstract SPNLayer

# Layer with Sum Nodes
type SumLayer <: SPNLayer

    ids::Vector{Int}
    childIds::Matrix{Int} # Ch x C child ids
    weights::Matrix{AbstractFloat} # Ch x C filter matrix

    children
    parent

end

abstract AbstractProductLayer <: SPNLayer
# Layer with Product Nodes
type ProductLayer <: AbstractProductLayer

    ids::Vector{Int}
    childIds::Matrix{Int} # Ch x C child ids

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

# Layer with MultivariateFeature Nodes
type MultivariateFeatureLayer <: SPNLayer

    ids::Vector{Int}
    weights::Matrix{AbstractFloat} # C x D filter matrix
    scopes::Matrix{Bool} # C x D mask

    parent

end

# Layer with indicator nodes
type IndicatorLayer <: SPNLayer

    ids::Vector{Int} # flatten vector representing C x D ids matrix
    scopes::Vector{Int} # D dimensional vector
    values::Vector # C dimensional vector of values used for the indicator functions

    parent

end

# Layer with univariate Gauss distributions
type GaussianLayer <: SPNLayer

    ids::Vector{Int} # C dimensional vector
    scopes::Vector{Int} # C dimensional vector
    μ::Vector{Float32} # C dimensional vector of values location parameters
    σ::Vector{Float32} # C dimensional vector of values scale parameters

    parent

end
