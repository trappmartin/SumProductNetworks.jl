
# test adjusted rand index / score

c1 = vec([ 1 2 1 2 ])
c2 = vec([ 2 1 2 1])
c3 = vec([ 1 1 2 2])

println(" * test adjusted rand index")

r1 = adjustedRandIndex(c1, c2)
r2 = adjustedRandIndex(c1, c3)

@test r1 == 1.0
@test r1 > r2
@test r2 < 0
