using Random
using SumProductNetworks

println("make dataset A and B")

A = vcat(
    randn(1, 100) .+ 0.0,
    randn(1, 100) .+ 0.5,
    randn(1, 100) .+ 1.0,
    randn(1, 100) .+ 1.5,
)

B = vcat(
    randn(1, 100) .+ 2.0,
    randn(1, 100) .+ 2.5,
    randn(1, 100) .+ 3.0,
    randn(1, 100) .+ 3.5,
)

println("generate an SPN by LearnSPN with dataset A")
spn = generate_spn(A, :learnspn; minclustersize=20)
updatescope!(spn)

println("mean log-likelihood for...")
println("   dataset A: ", mean(logpdf(spn, A)))
println("   dataset B: ", mean(logpdf(spn, B)))
