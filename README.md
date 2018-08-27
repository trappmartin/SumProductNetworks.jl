# Sum-Product Networks in Julia
[![Build Status](https://travis-ci.org/trappmartin/SumProductNetworks.jl.svg?branch=master)](https://travis-ci.org/trappmartin/SumProductNetworks.jl)
[![Coverage Status](https://coveralls.io/repos/github/trappmartin/SumProductNetworks.jl/badge.svg?branch=master)](https://coveralls.io/github/trappmartin/SumProductNetworks.jl?branch=master)

This software package implements node and layer wise Sum-Product Networks (SPNs). Further, this code provides high level routines to work with SPNs.


## Requireements
* julia 0.6
* packages listed in REQUIRE

## Installation
Make sure you have julia 0.6 running.
Inside of julia run:

```julia
Pkg.clone("https://github.com/trappmartin/SumProductNetworks.jl.git")
```

## Documentation
Please check the doc folder for a documentation on this software package. Note that this package is under constant development and the documentation might be behind things.

### Developing the source code
To ensure correctness of the implementation, the source code is developed using a test-driven approachby automatically rerunning all test using:

```bash
find . -name '*.jl' | entr julia test/runtests.jl
```
