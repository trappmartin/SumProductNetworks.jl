using SumProductNetworks
using Test

@testset "utility functions" begin
    q = rand(2)
    @test sum(projectToPositiveSimplex!(q)) == 1.0

    # check corner cases
    @test sum(projectToPositiveSimplex!(zeros(2))) == 1.0

    # check corner cases
    @test projectToPositiveSimplex!([0.5, 0.5]) == [0.5, 0.5]
end
