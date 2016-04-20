#__precompile__(true)

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
				BMITest#,
       # FunctionalData

	import Base.getindex
	import Base.map
  import Base.parent
  import Base.length
	import Base.show

  # include general implementations
  include("nodes.jl")
  include("nodeFunctions.jl")
  include("utils.jl")

  # include approach specific implementations
	include("infiniteSPN.jl")
  include("infiniteSPNGibbs.jl")
	include("naiveBayesClustering.jl")
	include("gens.jl")

  # include visualization implementations
  include("draw.jl")
	include("show.jl")


  export
    # types
    SPNNode,
		Node,
		Leaf,
    SumNode,
    ProductNode,
		ClassNode,
    UnivariateNode,
    NormalDistributionNode,
		UnivariateFeatureNode,
    MultivariateNode,
		Assignment,
    #Partition,
    #Region,
    #SumRegion,
    #LeafRegion,
    #SPNStructure,
		#SPNConfiguration,
    #AssignmentRegionGraph,

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
    fixSPN!,

		# infinite SPN functions
		#increment!,
		#decrement!,
		#assign!,
		#evalWithK,
		#recurseCondK!,
		#extend!,
		#mirror!,
		#draw,
		#gibbs_iteration!,
    #transformToRegionPartition,

    # utilities
    drawSPN,
    adjustedRandIndex

end # module
