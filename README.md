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
The following minimal example illustrates the use of the package.

```julia
using SumProductNetworks

# Create a root sum node.
root = FiniteSumNode{Float64}();

# Add two product nodes to the root.
add!(root, FiniteProductNode(), log(0.3)); # Use a weight of 0.3
add!(root, FiniteProductNode(), log(0.7)); # Use a weight of 0.7

# Add Normal distributions to the product nodes, i.e. leaves.
for prod in children(root)
    for d in 1:2 # Assume 2-D data
        add!(prod, UnivariateNode(Normal(), d));
    end
end

# Compile the constructed network to an SPN type
spn = SumProductNetwork(root);

# Print statistics on the network.
println(spn)

# Evaluate the network on some data.
x = [0.8, 1.2];
logp = logpdf(spn, x)
```

## Documentation

#### Datatypes
The following types are implemented and supported in this package. The abstract type hierarchy is designed such that it is easy to extend the existing types and that efficient implementations using type dispatching is possible.

```julia
# Abstract type hierarchy.
SPNNode
Node <: SPNNode
Leaf <: SPNNode
SumNode{T} <: Node
ProductNode <: Node

# Node types.
FiniteSumNode() <: SumNode
FiniteProductNode() <: ProductNode
IndicatorNode() <: Leaf
UnivariateNode() <: Leaf
MultivariateNode() <: Leaf
```

To get more details on the individual node type, please use the internal documentation system of Julia.

In addition to this types, the package also provides a composite type to represent an SPN, i.e.:

```julia
SumProductNetwork(root::Node)
```

#### Utility Functions on an SumProductNetwork
The following utility functions can be used on an instance of a SumProductNetwork.

```julia
# Get all nodes of the network.
values(spn::SumProductNetwork)

# Get the ids of all nodes in the network.
keys(spn::SumProductNetwork)

# Number of nodes in the network.
length(spn::SumProductNetwork)

# Indexing using an id.
spn[id::Symbol]
```

#### Utility Functions on Nodes
The following utility functions can be used on an instance of an SPN Node.

```julia
# Add a child to an internal node (with or without weight).
add!(node::Node, child::SPNNode)
add!(node::Node, child::SPNNode, logw::Real)

# Remove a child from an internal node.
remove!(node::Node, child::SPNNode)

# The depth of the SPN rooted at the node.
depth(node::SPNNode)

# Get all children of a node.
children(node::Node)

# Get the number of children of node.
length(node::Node)

# Get all parents of a node.
parents(node::SPNNode)

# Has the node a weights field.
hasweights(node::Node)

# Get all weights of the node.
weights(node::Node) = exp.(logweights(node))

# Get all log weights of the node
logweights(node::Node)

# Is the SPN rooted at the node normalized?
isnormalized(node::SPNNode)

```

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
