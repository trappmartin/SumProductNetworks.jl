require("Distributions")

module SPN

  using Distributions
  using Base

  export
    SumNode,
    ProductNode,
    UnivariateNode,
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
    generate_bloobs

  include("nodes.jl")
  include("utils.jl")

end # module
