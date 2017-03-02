# SumProductNetworks.jl

## Nodes

### Sum Nodes
tbd
### Product Nodes
tbd
### Indicator Nodes
tbd
## Layers

Sub-types of SPNs, e.g. trees, can be easily defined using layers. The following layers are implemented in this package (assuming tree structured SPNs):

### Sum Layer
Sum Layers consist of a weight matrix of size $Ch \times C$, where $C$ is the number of nodes and $Ch$ is the number of child nodes per node. This layer applies the following computation:

$$
Y_c = \sum_{j \in ch(c)} w_{cj} S_j[X]
$$

where $ch(c)$ are the children of node $c$ and $S_{j}[X]$ denotes the value of node $j$ on input $X$.

#### Example
```julia
C = 50 		# number of nodes
Ch = 5 		# number of children

parentLayer = ... 	# specify the reference to the parent layer, or nothing
children = ... 	# specify the references to the children, or nothing
ids = ... 		# specify the unique ids, e.g. collect(1:C)
childIds = ... 	# specify a Ch x C matrix of all child ids
weights = rand(Dirichlet([1./Ch for j in 1:Ch]), C) # random weights
layer = SumLayer(ids, childIds, weights, children, parentLayer)

```

### Product Layer
Product Layers apply the following computation:

$$
Y_c = \prod_{j \in ch(c)} S_j[X]
$$

where $ch(c)$ are the children of node $c$ and $S_{j}[X]$ denotes the value of node $j$ on input $X$.

#### Example

```julia
C = 50 		# number of nodes
Ch = 5 		# number of children

parentLayer = ... 	# specify the reference to the parent layer, or nothing
children = ... 	# specify the references to the children, or nothing
ids = ... 		# specify the unique ids, e.g. collect(1:C)
childIds = ... 	# specify a Ch x C matrix of all child ids
layer = ProductLayer(ids, childIds, children, parentLayer)

```

### Product Layer with class labels
In Product Layer with class labels each node is additionally asociated with a class label. This layer applies the following computation:

$$
Y_c = \prod_{j \in ch(c)} \mathcal{1}(y_c)  S_j[X]
$$

where $ch(c)$ are the children of node $c$ and $S_{j}[X]$ denotes the value of node $j$ on input $X$ and $\mathcal{1}(y_c)$ is a indicator function retuning one if the class label of node $c$ and those of the observation are equivalent.

#### Example

```julia
C = 50 		# number of nodes
Ch = 5 		# number of children

parentLayer = ... 	# specify the reference to the parent layer, or nothing
children = ... 	# specify the references to the children, or nothing
ids = ... 		# specify the unique ids, e.g. collect(1:C)
childIds = ... 	# specify a Ch x C matrix of all child ids
clabels = collect(1:C) # class labels have to start at 1
layer = ProductCLayer(ids, childIds, clabels, children, parentLayer)

```

### Multivariate Feature Layer (Convolution Layer)
This layer consists of filter matrix of size $C \times D$, where $C$ is the number of nodes and $D$ is the dimensionality. This layer applies the following computation:

$$
Y = \exp( W \cdot X )
$$

where we assume that $X \in \mathcal{R}^{D \times N}$.

#### Example

```julia
C = 50 					# number of nodes
D = 10 					# dimensionality

parentLayer = ... 				# specify the reference to the parent layer, or nothing
ids = ... 					# specify the unique ids, e.g. collect(1:C)
weights = zeros(D, C)		# use zero initialised filter weights
scopes = rand(Bool, D, C)	# mask
layer = MultivariateFeatureLayer(ids, weights, scopes, parentLayer)

```

## Layer Functions
The following helper functions are accesable for all layers:

```julia
size(layer) => C x Ch matrix				# the size of a layer in nodes x children
eval(layer, X, llhvalues) => N x C matrix 	# computes the output llh values of the layer
eval!(layer, X, llhvalues) 				# computes the output llh values of the layer in-place
eval(layer, X, y, llhvalues) => N x C matrix	# computes the output llh values of the layer conditioned on y
eval!(layer, X, y, llhvalues) 				# computes the output llh values of the layer conditioned on y in-place
```

## Structure Generation
The SumProductNetworks package provides implementations of the following heuristic structure learning approaches.

### Image Convolution Structure
This approach is suitable if the features represent responses from convolutions neural network (or similar kind) and spatial relationships can be assumes.

The input data $X \in R^{K \times G^2}$ where $G$ represents some feature map dimensionality and each input vector is assumed to be of size $1 \times D$ where $D = K \times G^2$.

#### Example
This example shows how to generate an image convolution structure.

```jl
C = 10 		# number of classes in the data set
G = 8 		# feature map size
K = 100 		# feature vector size, e.g. number of nodes in CNN
D = G^2 * K

P = 10 		# number of part sum nodes that should be generated
M = 2 		# number of mixture sum nodes that should be generated
W = 4 		# window size used for the sliding window approach

spn = SumLayer(1) # the root node
imageStructure!(spn, C, D, G, K; parts = P, mixtures = M, window = W)
```

Alternatively to the generated nodes based representation, the structure can also be generated using a sum layers as root, see:

```jl
...
spn = SumLayer(...)
imageStructure!(spn, C, D, G, K; parts = P, mixtures = M, window = W)

```

### learnSPN (Discrete and Continuous Data)
TBD

### Developing the source code
To ensure correctness of the implementation, the source code can be developed with while automatically rerunning all test using:

```
find . -name '*.jl' | entr julia test/runtests.jl
```
