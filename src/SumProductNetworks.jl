__precompile__()

module SumProductNetworks

  # loading dependencies into workspaces
  using Clustering,
        Distances,
        Distributions,
        StatsFuns,
		Base,
        HilbertSchmidtIndependenceCriterion,
        BayesianNonparametrics

    import Base.getindex
    import Base.map
    import Base.parent
    import Base.length
    import Base.size
    import Base.show

    # include general implementations
    include("nodes.jl")
    include("nodeFunctions.jl")
    include("networkFunctions.jl")
    include("layers.jl")
    include("layerFunctions.jl")

    # include approach specific implementations
    include("clustering.jl")
    include("indepTests.jl")
    include("naiveBayesClustering.jl")
    include("gens.jl")
    include("randomStructure.jl")
    include("imageStructure.jl")
    include("structureUtilities.jl")

    # include utilities
    include("utilityFunctions.jl")
end # module
