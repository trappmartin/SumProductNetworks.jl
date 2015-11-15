using BNP
using SPN 
using Base.Test

# run nodes tests
println("# starting: node tests...")
include("nodetests.jl")
println("# finished: node test...")

println("# starting: cross cat tests...")
include("crosscattest.jl")
println("# finished: cross cat test...")
