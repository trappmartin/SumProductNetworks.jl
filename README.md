# Sum-Product Networks in Julia
[![Build Status](https://travis-ci.org/trappmartin/SumProductNetworks.jl.svg?branch=master)](https://travis-ci.org/trappmartin/SumProductNetworks.jl)
[![Coverage Status](https://coveralls.io/repos/github/trappmartin/SumProductNetworks.jl/badge.svg?branch=master)](https://coveralls.io/github/trappmartin/SumProductNetworks.jl?branch=master)

This software package implements the tractable probabilistic model sum-product network (SPN) in Julia.
The package provides a clean and modular interface for SPNs and implements various helper and utility functions to efficienty work with SPN models.

### News
* 18.10.2018 - The package is officialy registered.
* 10.10.2018 - The package now provides more efficient logpdf routines and allows for multithreaded computations.
* 24.09.2018 - SumProductNetworks now works under Julia 1.0.

## Installation
Make sure you have Julia 1.0 running. The package can be installed using Julia's package mode. (You can enter the package mode by typing ] in the REPL.)

```julia
pkg> add SumProductNetworks
```

To install this package in its `master` branch version, use `PackageSpec`:

```bash
julia -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/trappmartin/SumProductNetworks.jl"))'
```

## Usage
The following minimal example illustrates the use of the package.

```julia
using SumProductNetworks

# Create a root sum node.
root = FiniteSumNode();

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

# Access the stored log likelihood
llh = spn.info[:llh]

# Evaluate the network by marginalising out a RV.
x = [0.8, NaN]
logp = logpdf(spn, x)

# Save the network to a DOT file.
export_network(spn, "mySPN.dot")
```

## Advanced Usage
Besides the basic functionality for nodes and SPNs, this package additionally provides helper functions that are useful for more advanced use-cases. The following examples illustrates a more advanced tasks.

```julia
using SumProductNetworks
using AxisArrays

N = 100
D = 2

x = rand(N, D)

# Create a root sum node.
root = FiniteSumNode{Float32}();

# Add two product nodes to the root.
add!(root, FiniteProductNode(), Float32(log(0.3))); # Use a weight of 0.3
add!(root, FiniteProductNode(), Float32(log(0.7))); # Use a weight of 0.7

# Add Normal distributions to the product nodes, i.e. leaves.
for prod in children(root)
    for d in 1:2 # Assume 2-D data
        add!(prod, UnivariateNode(Normal(), d));
    end
end

# Compile the constructed network to an SPN type
spn = SumProductNetwork(root);

# Update the scope of all nodes.
updatescope!(spn)

# Store the logpdf value for every node in the SPN.
llhvals = initllhvals(spn, x)

# Compute logpdf values for all nodes in the network.
logpdf!(spn, x, llhvals)

# Print the logpdf value for each leaf.
for node in spn.leaves
    println("logpdf at $(node.id) = $(llhvals[:,node.id])")
end

# Assign observations to their most likely child under a sum node.
function assignobs!(node::SumNode, observations::Vector{Int})
    j = argmax(llhvals[observations, map(c -> c.id, children(node))], dims = 2)

    # Set observations to the node.
    setobs!(node, observations)

    # Set observations for the children of the node.
    for k in length(node)
        setobs!(node[k], observations[findall(j .== k)])
    end

    # Get the parametric type of the sum node, i.e. Float32.
    T = eltype(node)

    # Update the weights of the root.
    w = map(c -> nobs(c) / nobs(node), children(node))
    for k in 1:length(node)
        logweights(node)[k] = map(T, log(w[k]))
    end
end

# Call assignment function for the root.
assignobs!(spn.root, collect(1:N))
```

## Examples
The following examples illustrate the use of this package: (WIP)

* [Parameter optimization using ForwardDiff](examples/parameterOptimization.jl)

## Documentation

#### Datatypes
The following types are implemented and supported in this package. The abstract type hierarchy is designed such that it is easy to extend the existing types and that efficient implementations using type dispatching is possible.

```julia
# Abstract type hierarchy.
SPNNode
Node <: SPNNode
Leaf <: SPNNode
SumNode <: Node
ProductNode <: Node

# Node types.
FiniteSumNode() <: SumNode
FiniteProductNode() <: ProductNode
IndicatorNode(value::Int, scope::Int) <: Leaf
UnivariateNode(dist::UnivariateDistribution, scope::Int) <: Leaf
MultivariateNode(dist::MultivariateDistribution, scope::Vector{Int}) <: Leaf
```

To get more details on the individual node type, please use the internal documentation system of Julia.

In addition to this types, the package also provides a composite type to represent an SPN, i.e.:

```julia
SumProductNetwork(root::Node)
```

#### Structure Learning
Utility functions for structure learning.

The interface for learning SPN structure is:

```julia
generate_spn(X::Matrix, algo::Symbol; params...)

# learnSPN algorithm by Gens et al.
generate_spn(X, :learnspn)
```

#### Utility Functions on an SumProductNetwork
The following utility functions can be used on an instance of a SumProductNetwork.

```julia
# Get all nodes of the network in topological order.
values(spn::SumProductNetwork)

# Get the ids of all nodes in the network.
keys(spn::SumProductNetwork)

# Number of nodes in the network.
length(spn::SumProductNetwork)

# Indexing using an id.
spn[id::Symbol]

# Locally normalize an SPN.
normalize!(spn::SumProductNetwork)

# Number of free parameters in the SPN.
complexity(spn::SumProductNetwork)

# Export the SPN to a DOT file.
export_network(spn::SumProductNetwork, filename::String)

# Draw a random sample for the SPN.
rand(spn::SumProductNetwork)
```

#### Utility Functions on Nodes
The following utility functions can be used on an instance of an SPN Node.

```julia
# Indexing an internal node returns the respective child.
node[i::Int]

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

# Compute the log pdf.
logpdf(node::SPNNode, x::AbstractArray)

# Alternatively, you can call the object with some data.
node(x::AbstractArray)

# Compute the log pdf but use the passed parameters instead.
logpdf(node::Leaf, x::AbstractArray, p...)
logpdf(node::SumNode, x::AbstractArray; lw::AbstractVector=logweights(node))

# Draw a random sample from a node.
rand(node::SPNNode)
```

#### General utility functions
The following functions are general utility functions.

```julia
# Independence test by Margaritis and Thurn for discrete sets.
bmitest(X::Vector{Int}, Y::Vector{Int})

# Efficient projections onto the L 1-ball.
projectToPositiveSimplex!(q::AbstractVector{<:Real}; lowerBound = 0.0, s = 1.0)

# Construct a log likelihoods data-structure.
initllhvals(spn::SumProductNetwork, X::AbstractMatrix)
```

### Contribute
Feel free to open a PR if you want to contribute!

### References
Please consider citing any of the following publications if you use this package.

* Martin Trapp, Robert Peharz, Hong Ge, Franz Pernkopf, Zoubin Ghahramani: **Bayesian learning of sum-product networks.** In proceedings of NeurIPS, 2019.
* Martin Trapp, Tamas Madl, Robert Peharz, Franz Pernkopf, Robert Trappl: **Safe Semi-Supervised Learning of Sum-Product Networks.** In proceedings of UAI, 2017. [pdf](http://auai.org/uai2017/proceedings/papers/108.pdf) [bibtex](https://dblp.uni-trier.de/rec/bibtex/conf/uai/TrappMPPT17)
* Martin Trapp, Robert Peharz, Marcin Skowron, Tamas Madl, Franz Pernkopf, Robert Trappl: **Structure Inference in Sum-Product Networks using Infinite Sum-Product Trees.** In proceedings of NeurIPS workshop on BNP, 2016. [paper](https://www.spsc.tugraz.at/sites/default/files/BNPNIPS_2016_paper_9.pdf) [bibtex](https://www.spsc.tugraz.at/biblio/export/bibtex/3537)
