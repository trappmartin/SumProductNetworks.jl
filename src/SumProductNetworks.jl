module SumProductNetworks

# loading dependencies into workspaces
using Clustering
using Reexport
using Distances
@reexport using Distributions
using StatsFuns
using HilbertSchmidtIndependenceCriterion
using BayesianNonparametrics
using JLD2
using FileIO
using AxisArrays
# using InformationMeasures

using StatsBase: countmap

import Base: getindex, map, parent, length, size, show, isequal, getindex, keys, eltype
import Distributions.logpdf

# add Base modules
@reexport using Statistics
using LinearAlgebra
using SparseArrays
using Random
using Printf

# include general implementations
include("nodes.jl")
include("nodeFunctions.jl")
include("networkFunctions.jl")
include("layers.jl")
include("layerFunctions.jl")

# include approach specific implementations
include("clustering.jl")
include("indepTests.jl")
include("naiveBayesClustering.jl")
include("gens.jl")
include("randomStructure.jl")
include("imageStructure.jl")
include("layerStructure.jl")
include("structureUtilities.jl")

# include utilities
include("utilityFunctions.jl")
include("io.jl")

end # module
