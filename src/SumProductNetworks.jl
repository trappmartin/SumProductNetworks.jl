module SumProductNetworks

# loading dependencies into workspaces
using Reexport
@reexport using Distributions

using AxisArrays
using Clustering
using DataStructures
using Distances
using SpecialFunctions
using StatsFuns
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

# include general implementations
include("nodes.jl")
include("nodeFunctions.jl")
include("networkFunctions.jl")

# include approach specific implementations
include("bmiTest.jl")
include("structureUtilities.jl")
include("parameterUtilities.jl")

# include utilities
include("utilityFunctions.jl")
include("io.jl")

end # module
