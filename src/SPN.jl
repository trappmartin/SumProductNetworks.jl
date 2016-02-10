#VERSION >= v"0.4.2" && __precompile__(true)

module SPN

  # loading dependencies into workspaces
  using JuMP,
        Distributions,
        Base,
        BNP,
        HSIC,
        GraphLayout,
        Compose,
        Colors,
				BMITest

	import Base.getindex
	import Base.map
  import Base.parent
  import Base.length
	import Base.show

  # include general implementations
  include("nodes.jl")
  include("utils.jl")

  # include approach specific implementations
	include("infiniteSPN.jl")
  include("infiniteSPNGibbs.jl")
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
    UnivariateNode,
    MultivariateNode,
		Assignment,
    Partition,
    Region,
    SumRegion,
    LeafRegion,
    SPNStructure,
		SPNConfiguration,

    # spn functions
		children,
		parent,
    length,
    add!,
    remove!,
    normalize!,
    llh,
    cmllh,
    map,

		# infinite SPN functions
		increment!,
		decrement!,
		assign!,
		evalWithK,
		recurseCondK!,
		extend!,
		mirror!,
		#draw,
		gibbs_iteration!,
    transformToRegionPartition,

    # utilities
    drawSPN,
    adjustedRandIndex

end # module
