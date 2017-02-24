export MultivariateFeatureLayer

# Layer with C MultivariateFeature Nodes
immutable MultivariateFeatureLayer
    
    id::Int
    weights::Matrix{AbstractFloat} # C x D
    scope::Matrix{Bool} # C x D mask
    
    parent

end
