export SPNLayer, MultivariateFeatureLayer, SumLayer, ProductLayer, ProductCLayer

abstract SPNLayer

# Layer with Sum Nodes
type SumLayer <: SPNLayer

    ids::Vector{Int}
    childIds::Matrix{Int} # Ch x C child ids
    weights::Matrix{AbstractFloat} # Ch x C filter matrix

    parentId::Int

end

abstract AbstractProductLayer <: SPNLayer
# Layer with Product Nodes
type ProductLayer <: AbstractProductLayer

    ids::Vector{Int}
    childIds::Matrix{Int} # Ch x C child ids

    parentId::Int

end

# Layer with Product Nodes, each equiped with a class label
type ProductCLayer <: AbstractProductLayer

    ids::Vector{Int}
    childIds::Matrix{Int} # Ch x C child ids
    clabels::Vector{Int} # C class labels

    parentId::Int

end

# Layer with MultivariateFeature Nodes
type MultivariateFeatureLayer <: SPNLayer

    ids::Vector{Int}
    weights::Matrix{AbstractFloat} # C x D filter matrix
    scopes::Matrix{Bool} # C x D mask

    parentId::Int

end
