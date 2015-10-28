module SPN

  using Distributions
  using Base
  using BNP

  include("nodes.jl")
  include("utils.jl")

  export
    # types
    SPNNode,
    SumNode,
    ProductNode,
    UnivariateNode,

    # spn functions
    add!,
    remove!,
    normalize!,
    llh,
    eval

end # module
