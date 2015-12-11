module SPN

  using Distributions
  using Base
  using BNP
  using Graphs

  include("nodes.jl")
  include("utils.jl")
  include("infiniteSPN.jl")

  export
    # types
    SPNNode,
    SumNode,
    ProductNode,
    UnivariateNode,
    MultivariateNode,

    # spn functions
    add!,
    remove!,
    normalize!,
    llh,
    map

end # module
