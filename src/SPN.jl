module SPN

  using Distributions
  using Base
  using BNP

  include("nodes.jl")
  include("utils.jl")
  include("crosscat.jl")

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
    convert,
    llh,
    map,
    eval

end # module
