module SumProductNetworks

# loading dependencies into workspaces
using Reexport
@reexport using Distributions
using StatsFuns
using SpecialFunctions
using AxisArrays

using StatsBase: countmap

import Base: getindex, map, parent, length, size, show, isequal, getindex, keys, eltype, rand
import Distributions.logpdf
import StatsBase.nobs

# add Base modules
@reexport using Statistics
using LinearAlgebra
using SparseArrays
using Random
using Printf

using DataStructures
using Distances
using Clustering

# include general implementations
include("nodes.jl")
include("nodeFunctions.jl")
include("networkFunctions.jl")

# include approach specific implementations
include("bmiTest.jl")
include("structureUtilities.jl")

# include utilities
include("utilityFunctions.jl")
include("io.jl")

end # module
