# Sum-Product Networks in Julia
[![Build Status](https://travis-ci.org/trappmartin/SumProductNetworks.jl.svg?branch=master)](https://travis-ci.org/trappmartin/SumProductNetworks.jl)
[![Coverage Status](https://coveralls.io/repos/github/trappmartin/SumProductNetworks.jl/badge.svg?branch=master)](https://coveralls.io/github/trappmartin/SumProductNetworks.jl?branch=master)

This software package implements node and layer wise Sum-Product Networks (SPNs). Further, this code provides high level routines to work with SPNs.

** This package is currently wip to get ready for Julia 1.0. Several things might break! **

## Requireements
* julia 1.0
* packages listed in REQUIRE

## Installation
Make sure you have julia running. Currently the package is not registered (this will change soon) and you have to run the following inside of julia's package mode. (You can enter the package mode by typing ] in the REPL.)

```julia
pkg> add https://github.com/trappmartin/SumProductNetworks.jl.git
```

## Documentation
Please check the doc folder for a documentation on this software package. Note that this package is under constant development and the documentation might be behind things.

### Developing the source code
To ensure correctness of the implementation, the source code is developed using a test-driven approach by automatically rerunning all test. Please post an issue if you find a bug or run into trouble using this package!

### Contribute
This package is currently maintained only by myself. Feel free to contact me if you want to contribute!

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
