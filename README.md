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

# Update the scope of all nodes, i.e. propagate the scope bottom-up.
updatescope!(spn)

# Evaluate the network on some data.
x = [0.8, 1.2];
logp = logpdf(spn, x)

# Save the network to a DOT file.
exportNetwork(spn, "mySPN.dot")
```

## Advanced Usage
Besides the basic functionality for nodes and SPNs, this package additionally provides helper functions that are useful for more advanced use-cases. The following examples illustrates a more advanced tasks.

```julia
using SumProductNetworks
using AxisArrays

N = 100
D = 2

x = rand(N, D)

# Compute the logpdf value for every node in the SPN.
idx = Axis{:id}(collect(keys(spn)))
llhvals = AxisArray(Matrix{Float32}(undef, N, length(spn)), 1:N, idx)

# Using SPN of the minimal example.
logpdf(spn, x; idx, llhvals)

# Print the logpdf value for each leaf.
for node in spn.leaves
    println("logpdf at $(node.id) = $(llhvals[:,node.id])")
end

# Assign all observations to their most likely child under the root.
j = argmax(llhvals[:, map(c -> c.id, children(spn.root))], dims = 2)

# Set observations for the root.
observations = collect(1:N)
setobs!(spn.root, observations)

# Set observations for the children of the root.
for k in length(spn.root)
    setobs!(spn.root[k], observations[findall(j .== k)])
end

# Get the parametric type of the root.
T = eltype(spn.root)

# Update the weights of the root.
w = map(c -> nobs(spn.root) / nobs(c), children(spn.root))
for k in 1:length(spn.root)
    logweights(spn.root)[k] = T(log(w[k]))
end
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
remove!(node::Node, childIndex::Int)

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

* Martin Trapp, Tamas Madl, Robert Peharz, Franz Pernkopf, Robert Trappl: **Safe Semi-Supervised Learning of Sum-Product Networks.** UAI 2017. [pdf](auai.org/uai2017/proceedings/papers/108.pdf) [bibtex](https://dblp.uni-trier.de/rec/bibtex/conf/uai/TrappMPPT17)
* Martin Trapp, Robert Peharz, Marcin Skowron, Tamas Madl, Franz Pernkopf, Robert Trappl: **Structure Inference in Sum-Product Networks using Infinite Sum-Product Trees.** NIPS 2016 - Workshop on Practical Bayesian Nonparametrics. [paper](https://www.spsc.tugraz.at/sites/default/files/BNPNIPS_2016_paper_9.pdf) [bibtex](https://www.spsc.tugraz.at/biblio/export/bibtex/3537)