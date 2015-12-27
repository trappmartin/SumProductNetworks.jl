VERSION >= v"0.4.0" && __precompile__(true)

module SPN

  # load dependencies into workspace
  using Distributions,
        Base,
        BNP,
        GraphLayout

	import Base.getindex

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
