# Nodes

## Internal Nodes
### Sum Nodes
Implementation of sum nodes. Those nodes mix the distributions of their children using (normalized) non-negative weights.

```julia
n = SumNode(id::Int)
```

Optional arguments:

```julia
parents = SPNNode[]
scope = Int[]
```

Example:

```julia
n = SumNode(1, scope = [1, 2, 3])
```

### Product Nodes
Implementation of product nodes. Those nodes combine independent sub-scopes.

```julia
n = ProductNode(id::Int)
```

Optional arguments:

```julia
parents = SPNNode[]
children = SPNNode[]
scope = Int[]
```

Example:

```julia
n = ProductNode(1)
```

## Leaf Nodes
### Indicator Nodes
Implementation of indicator functions as introduced in Poon and Domingos 2011.

```julia
n = IndicatorNode(id::Int, value::Int, scope::Int)
```

Optional arguments:

```julia
parents = SPNNode[]
```

Example:

```julia
n = IndicatorNode(1, 0, 1)
```

### Univariate Feature Nodes
Implementation of feature nodes as used by Gens 2013 for discriminative SPNs.

```julia
n = UnivariateFeatureNode(id::Int, scope::Int)
```

Optional arguments:

```julia
parents = SPNNode[]
weight = 0.
```

Example:

```
n = UnivariateFeatureNode(1, 1)
```

### Multivariate Feature Nodes
Implementation of multivariate feature nodes.

```julia
n = MultivariateFeatureNode(id::Int, scope::Vector{Int})
```

Optional arguments:

```julia
parents = SPNNode[]
```

Example:

```julia
n = MultivariateFeatureNode(1, [1, 2, 3])
```

### Generalized Univariate Nodes
Implementation of a univariate leaf node used in generalized SPNs.

```julia
n = UnivariateNode{T}(id::Int, distribution::T, scope::Int)
```
where `T` is the type of univariate distribution used, see Distribution.jl for univariate distribution types.

Optional arguments:

```julia
parents = SPNNode[]
```

Example:

```julia
n = UnivariateNode{Cauchy}(1, Chauchy(), 1)
```

### Generalized Multivariate Nodes
Implementation of a multivariate leaf node used in generalized SPNs.

```julia
n = MultivariateNode{T}(id::Int, distribution::T, scope::Vector{Int})
```
where `T` is the type of multivariate distribution used, see Distribution.jl.

Optional arguments:

```julia
parents = SPNNode[]
```

Example:

```julia
n = MultivariateNode{MvNormal}(1, MvNormal(mu, sigma), 1)
```
