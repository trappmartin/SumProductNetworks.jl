using SPN
using Base.Test

X = randn(2, 100)
S = SPN.randomStructure(X, [1, 2], 2, 1)

@test length(SPN.order(S)) == 17

X = randn(4, 100)
S = SPN.randomStructure(X, [1, 2], 2, 2)

SPN.order(S)
