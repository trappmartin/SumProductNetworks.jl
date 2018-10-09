# Sum-Product Networks in Julia
[![Build Status](https://travis-ci.org/trappmartin/SumProductNetworks.jl.svg?branch=master)](https://travis-ci.org/trappmartin/SumProductNetworks.jl)
[![Coverage Status](https://coveralls.io/repos/github/trappmartin/SumProductNetworks.jl/badge.svg?branch=master)](https://coveralls.io/github/trappmartin/SumProductNetworks.jl?branch=master)

This software package implements node and layer wise Sum-Product Networks (SPNs). Further, this code provides high level routines to work with SPNs.

### News
* 24.09.2018 - SumProductNetworks.jl now works under Julia 1.0. 

## Installation
Make sure you have julia running. Currently the package is not registered (this will change soon) and you have to run the following inside of julia's package mode. (You can enter the package mode by typing ] in the REPL.)

```julia
pkg> add https://github.com/trappmartin/SumProductNetworks.jl.git
```

## Usage
The following example is a minimal example.
```julia
using SumProductNetworks

# Create a root sum node.
root = FiniteSumNode{Float64}()

# Add two product nodes to the root.
add!(root, FiniteProductNode(), log(0.3)) # Use a weight of 0.3
add!(root, FiniteProductNode(), log(0.7)) # Use a weight of 0.7

# Add Normal distributions to the product nodes, i.e. leaves.
for prod in children(root)
    for d in 1:2 # Assume 2-D data
        add!(prod, UnivariateNode(Normal(), d))
    end
end

# Evaluate the network on some data.
x = [0.8, 1.2]
logp = logpdf(root, x)
```

## Documentation
The documentation is currently work in progress.

### Contribute
Feel free to open a PR if you want to contribute!

### References
Please consider citing any of the following publications if you use this package.

```
@inproceedings{trapp2018,
  title={Safe Semi-Supervised Learning of Sum-Product Networks},
  author={Trapp, Martin and Madl, Tamas and Peharz, Robert and Pernkopf, Pernkopf and Trappl, Robert},
  booktitle={Conference on Uncertainty in Artificial Intelligence},
  year={2017}
}

@misc{trapp2016,
  title={Structure Inference in Sum-Product Networks using Infinite Sum-Product Trees},
  author={Trapp, Martin and Peharz, Robert and Skowron, Marcin and Madl, Tamas and Pernkopf, Pernkopf and Trappl, Robert},
  booktitle={NIPS Workshop on Practical Bayesian Nonparametrics},
  year={2016}
}
```
