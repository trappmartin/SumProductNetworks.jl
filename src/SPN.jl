VERSION >= v"0.4.0" && __precompile__(true)

module SPN

  # load dependencies into workspace
  using Distributions,
        Base,
        BNP,
        GraphLayout

	import Base.getindex
	import Base.map
  import Base.parent
  import Base.length

  # include implementations
  include("nodes.jl")
  include("utils.jl")
  include("infiniteSPN.jl")

  export
    # types
    SPNNode,
		Node,
		Leaf,
    SumNode,
    ProductNode,
    UnivariateNode,
    MultivariateNode,
		Assignments,

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
		draw,
		gibbs_iteration!

end # module
