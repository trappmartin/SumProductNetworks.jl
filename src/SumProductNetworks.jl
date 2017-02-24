__precompile__(true)

module SumProductNetworks

  # loading dependencies into workspaces
  using Clustering,
        Distances,
        Distributions,
        StatsFuns,
				Base,
        HSIC,
        BayesianNonparametrics,
				BMITest

    import Base.getindex
	import Base.map
    import Base.parent
    import Base.length
	import Base.show

    # include general implementations
    include("nodes.jl")
    include("nodeFunctions.jl")
    include("layers.jl")

    # include approach specific implementations
	include("naiveBayesClustering.jl")
	include("gens.jl")
    include("randomStructure.jl")
    include("imageStructure.jl")

	# include utilities
	include("utilityFunctions.jl")

end # module
