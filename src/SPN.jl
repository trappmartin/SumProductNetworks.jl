module SPN

  # loading dependencies into workspaces
  using Clustering,
				JuMP,
        Distances,
        Distributions,
        StatsFuns,
				DistributedArrays,
				Base,
        BNP,
        HSIC,
        GraphLayout,
        Compose,
        Colors,
				BMITest
       # FunctionalData

	import Base.getindex
	import Base.map
  import Base.parent
  import Base.length
	import Base.show

  # include general implementations
  include("nodes.jl")
  include("nodeFunctions.jl")
  include("layers.jl")
  include("layerFunctions.jl")
  include("utils.jl")

  # include approach specific implementations
	include("naiveBayesClustering.jl")
	include("gens.jl")
  include("randomstructureMedian.jl")
  include("randomstructureKMeans.jl")

  # include visualization implementations
  include("draw.jl")
	include("show.jl")


  export
    # types
    SPNNode,
		Node,
		Leaf,
    ClassIndicatorNode,
    SumNode,
    ProductNode,
    UnivariateNode,
    NormalDistributionNode,
		UnivariateFeatureNode,
    MultivariateNode,
		Assignment,

    # spn functions
		children,
		parent,
    length,
    classes,
    add!,
    remove!,
    normalize!,
    llh,
    cmllh,
    map,

    # structure learning
		partStructure,
    learnSPN,

    # utilities
    drawSPN,
    adjustedRandIndex

end # module
