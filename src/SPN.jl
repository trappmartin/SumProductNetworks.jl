require("Distributions")

module SPN

  using Distributions
  using Base

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
    llh_map
  
end # module
