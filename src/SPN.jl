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
    MultivariateNode,

    # spn functions
    add!,
    remove!,
    normalize!,
    convertNode,
    llh,
    map,
    order,
    eval

end # module
