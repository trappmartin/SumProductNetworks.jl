VERSION >= v"0.4.0" && __precompile__(true)

module SPN

  # load dependencies into workspace
  using Distributions,
        Base,
        BNP,
        HSIC,
        JuMP,
        GraphLayout,
        Compose,
        Colors

	import Base.getindex
	import Base.map
  import Base.parent
  import Base.length

  # include implementations
  include("nodes.jl")
  include("utils.jl")
	include("draw.jl")
	include("infiniteSPN.jl")
	include("gens.jl")

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

    # spn functions
		children,
		parent,
    length,
    add!,
    remove!,
    normalize!,
    llh,
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
    findPartition,

    # utilities
    drawSPN,
    adjustedRandIndex

end # module
