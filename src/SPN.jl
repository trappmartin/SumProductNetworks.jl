require("Distributions")

module SPN

  using Distributions
  using Base

  export
    # types
    SPNNode,
    SumNode,
    ProductNode,
    UnivariateNode,

    # spn functions
    add,
    remove,
    build_sum,
    build_prod,
    build_univariate,
    build_multivariate,
    normalize,
    llh,
    llh_map,

    # utils
    generate_bloobs,

    # gibbs
    gibbs_iteration!

  include("nodes.jl")
  include("utils.jl")
  include("gibbs.jl")

end # module
