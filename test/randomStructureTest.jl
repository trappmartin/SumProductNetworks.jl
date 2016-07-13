using SPN
using Base.Test

X = randn(2, 100)
S = SPN.randomStructure(X, [1, 2], 2, 1)

@test length(SPN.order(S)) == 17

X = randn(4, 100)
S = SPN.randomStructure(X, [1, 2], 2, 2)

SPN.order(S)

Y = rand(1:2, 100)

S = SPN.randomStructureMedian(X, Y, 200, 1)

Mu = Float64[]
for node in SPN.order(S)

  if isa(node, NormalDistributionNode)
    append!(Mu, [node.μ])
  end

end

using Gadfly

plot(x=Mu, Geom.histogram)
