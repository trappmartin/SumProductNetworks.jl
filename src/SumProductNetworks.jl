module SumProductNetworks

# loading dependencies into workspaces
using Clustering
using Distances
using Distributions
using StatsFuns
using Base
using HilbertSchmidtIndependenceCriterion
using BayesianNonparametrics
using JLD2
using FileIO
# using InformationMeasures

using StatsBase: countmap

import Base.getindex
import Base.map
import Base.parent
import Base.length
import Base.size
import Base.show

# add Base modules
using Statistics
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
