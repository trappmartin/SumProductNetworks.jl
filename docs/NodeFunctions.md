# Node Functions

[\< back](README.md)

## Network Functions

#### Local Normalization of SPNs
An SPN can be locally normalized using:

```julia
normalize!(SPN)
```
where `SPN` is the root node of the network.

#### Compute log likelihood
The log likelihood of some data can be computed using:

```julia
llhValues::Vector{Float64} = llh(SPN::Node, X)
```

#### Depth of an SPN
We can compute the depth of an SPN using:

```julia
d::Int = depth(SPN::Node)
```


## Node Specific Functions

#### Add Children to Internal Nodes
Children can be added to internal nodes, e.g. product nodes or sum nodes, using:

```julia
add!(parent::Node, child::Leaf)
add!(parent::Node, child::Leaf, weight::Float64)
```

#### Remove Child from Internal Node
Children can be removed from internal nodes using:

```julia
remove!(parent::Node, index::Int)
```

#### Number of Children of a Node
We can access the number of children of a node using:

```julia
length::Int = length(node::SPNNode)
```

#### Evaluate a Node
We can evaluate a node inplace (assuming all subsequent nodes has been evaluated) using:

```julia
eval!(node::SPNNode, X, llhValues::Matrix{Float64})
```
where `llhValues` contains precomputed log density values of all sub-sequent nodes. Indexing the matrix is done using the `id` field of the nodes.
