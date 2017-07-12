# Layers

Sub-types of SPNs, e.g. trees, can be easily defined using layers. The following layers are implemented in this package (assuming tree structured SPNs):

## Sum Layer
Sum Layers consist of a weight matrix of size <img src="svgs/dc11c801cf18acc49f31087b0540eda4.svg?invert_in_darkmode" align=middle width=55.73386500000001pt height=22.063469999999988pt/>, where <img src="svgs/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode" align=middle width=13.24702500000001pt height=21.697829999999996pt/> is the number of nodes and <img src="svgs/3419e778dfd9c1961b8f080e5f4f1bed.svg?invert_in_darkmode" align=middle width=22.71802500000001pt height=22.063469999999988pt/> is the number of child nodes per node. This layer applies the following computation:

<p align="center"><img src="svgs/5ca06646171b16f56a88962dbfd8a212.svg?invert_in_darkmode" align=middle width=151.19659499999997pt height=40.548089999999995pt/></p>

where <img src="svgs/6ce7c8d9e329516e2559c365fd6d34cf.svg?invert_in_darkmode" align=middle width=36.806385000000006pt height=23.889689999999977pt/> are the children of node <img src="svgs/3e18a4a28fdee1744e5e3f79d13b9ff6.svg?invert_in_darkmode" align=middle width=7.43612100000001pt height=13.387440000000009pt/> and <img src="svgs/b3e4614343ffbabf3048521cfca15dc7.svg?invert_in_darkmode" align=middle width=41.36979000000001pt height=23.889689999999977pt/> denotes the value of node <img src="svgs/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode" align=middle width=8.03272800000001pt height=20.9154pt/> on input <img src="svgs/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode" align=middle width=15.230985000000008pt height=21.697829999999996pt/>.

#### Example
```julia
C = 50              # number of nodes
Ch = 5              # number of children

parentLayer = ...   # specify the reference to the parent layer, or nothing
children = ...      # specify the references to the children, or nothing
ids = ...           # specify the unique ids, e.g. collect(1:C)
childIds = ...      # specify a Ch x C matrix of all child ids
weights = rand(Dirichlet([1./Ch for j in 1:Ch]), C) # random weights
layer = SumLayer(ids, childIds, weights, children, parentLayer)

```

## Product Layer
Product Layers apply the following computation:

<p align="center"><img src="svgs/425f4415ea14f4dfcfa876b3e69db63b.svg?invert_in_darkmode" align=middle width=126.62693999999999pt height=40.548089999999995pt/></p>

where <img src="svgs/6ce7c8d9e329516e2559c365fd6d34cf.svg?invert_in_darkmode" align=middle width=36.806385000000006pt height=23.889689999999977pt/> are the children of node <img src="svgs/3e18a4a28fdee1744e5e3f79d13b9ff6.svg?invert_in_darkmode" align=middle width=7.43612100000001pt height=13.387440000000009pt/> and <img src="svgs/b3e4614343ffbabf3048521cfca15dc7.svg?invert_in_darkmode" align=middle width=41.36979000000001pt height=23.889689999999977pt/> denotes the value of node <img src="svgs/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode" align=middle width=8.03272800000001pt height=20.9154pt/> on input <img src="svgs/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode" align=middle width=15.230985000000008pt height=21.697829999999996pt/>.

#### Example

```julia
C = 50              # number of nodes
Ch = 5              # number of children

parentLayer = ...   # specify the reference to the parent layer, or nothing
children = ...      # specify the references to the children, or nothing
ids = ...           # specify the unique ids, e.g. collect(1:C)
childIds = ...      # specify a Ch x C matrix of all child ids
layer = ProductLayer(ids, childIds, children, parentLayer)

```

## Product Layer with class labels
In Product Layer with class labels each node is additionally asociated with a class label. This layer applies the following computation:

<p align="center"><img src="svgs/401dfc8885634357800c81f46b41be1b.svg?invert_in_darkmode" align=middle width=170.6067pt height=40.548089999999995pt/></p>

where <img src="svgs/6ce7c8d9e329516e2559c365fd6d34cf.svg?invert_in_darkmode" align=middle width=36.806385000000006pt height=23.889689999999977pt/> are the children of node <img src="svgs/3e18a4a28fdee1744e5e3f79d13b9ff6.svg?invert_in_darkmode" align=middle width=7.43612100000001pt height=13.387440000000009pt/> and <img src="svgs/b3e4614343ffbabf3048521cfca15dc7.svg?invert_in_darkmode" align=middle width=41.36979000000001pt height=23.889689999999977pt/> denotes the value of node <img src="svgs/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode" align=middle width=8.03272800000001pt height=20.9154pt/> on input <img src="svgs/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode" align=middle width=15.230985000000008pt height=21.697829999999996pt/> and <img src="svgs/7e059afa3f48a12aef668669f19f66a7.svg?invert_in_darkmode" align=middle width=44.302170000000004pt height=23.889689999999977pt/> is a indicator function retuning one if the class label of node <img src="svgs/3e18a4a28fdee1744e5e3f79d13b9ff6.svg?invert_in_darkmode" align=middle width=7.43612100000001pt height=13.387440000000009pt/> and those of the observation are equivalent.

#### Example

```julia
C = 50                  # number of nodes
Ch = 5                  # number of children

parentLayer = ...       # specify the reference to the parent layer, or nothing
children = ... 	        # specify the references to the children, or nothing
ids = ... 		        # specify the unique ids, e.g. collect(1:C)
childIds = ... 	        # specify a Ch x C matrix of all child ids
clabels = collect(1:C) # class labels have to start at 1
layer = ProductCLayer(ids, childIds, clabels, children, parentLayer)

```

## Multivariate Feature Layer (Convolution Layer)
This layer consists of filter matrix of size <img src="svgs/f52d6ad69c5488960d279c9ae7d6dc07.svg?invert_in_darkmode" align=middle width=47.40433500000001pt height=21.697829999999996pt/>, where <img src="svgs/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode" align=middle width=13.24702500000001pt height=21.697829999999996pt/> is the number of nodes and <img src="svgs/78ec2b7008296ce0561cf83393cb746d.svg?invert_in_darkmode" align=middle width=14.388495000000008pt height=21.697829999999996pt/> is the dimensionality. This layer applies the following computation:

<p align="center"><img src="svgs/b9f290ad3787da65f6a6184b583e5802.svg?invert_in_darkmode" align=middle width=117.602595pt height=16.438356pt/></p>

where we assume that <img src="svgs/f4503fed1fe8477e329e171e6259c383.svg?invert_in_darkmode" align=middle width=82.27593pt height=26.889059999999983pt/>.

#### Example

```julia
C = 50 					    # number of nodes
D = 10 					    # dimensionality

parentLayer = ...           # specify the reference to the parent layer, or nothing
ids = ...                   # specify the unique ids, e.g. collect(1:C)
weights = zeros(D, C)		# use zero initialised filter weights
scopes = rand(Bool, D, C)	# mask
layer = MultivariateFeatureLayer(ids, weights, scopes, parentLayer)
```

## Indicator Layer
This layer consists of indicator functions.

#### Example

```julia

D = 10 					    # dimensionality
V = 0:1                     # values, e.g. 0:1 if data is binary

parentLayer = ...           # specify the reference to the parent layer, or nothing
ids = ... 					# specify the unique ids, e.g. collect(1:C)
scopes = randperm(D)
values = V

layer = IndicatorLayer(ids, scopes, values, parentLayer)
```

## Gaussian Layer
tbd

# Layer Functions
The following helper functions are accesable for all layers:

```julia
size(layer) => C x Ch matrix				# the size of a layer in nodes x children
eval(layer, X, llhvalues) => N x C matrix 	# computes the output llh values of the layer
eval!(layer, X, llhvalues) 				# computes the output llh values of the layer in-place
eval(layer, X, y, llhvalues) => N x C matrix	# computes the output llh values of the layer conditioned on y
eval!(layer, X, y, llhvalues) 				# computes the output llh values of the layer conditioned on y in-place
```
