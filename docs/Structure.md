# Structure Generation
The SumProductNetworks package provides implementations of the following heuristic structure learning approaches.

## Image Convolution Structure
This approach is suitable if the features represent responses from convolutions neural network (or similar kind) where spatial relationships can be assumes.

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

## Random Structure
This approach is allow to generate a random structure in node or layer form.

#### Example
This example shows how to generate a random structure.

tbd

## learnSPN (Discrete and Continuous Data)
tbd
