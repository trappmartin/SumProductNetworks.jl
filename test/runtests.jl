using BNP
using SPN
using Base.Test

# run nodes tests
println("# starting: node tests...")
include("nodetests.jl")
println("# finished: node test...")

println("# starting: infinite SPN tests...")
include("infiniteSPNtests.jl")
println("# finished: infinite SPN tests...")
