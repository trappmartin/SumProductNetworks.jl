using Random
using SumProductNetworks

println("make dataset A and B")

N = 1000
D = 4

A = hcat([randn(N) .+ (2 + 0.5*i) for i in 1:D]...)
B = hcat([randn(N) .+ (4 + 0.5*i) for i in 1:D]...)

println("generate an SPN by LearnSPN with dataset A")
spn = generate_spn(A, :learnspn)
updatescope!(spn)

println("mean log-likelihood for...")
println("   dataset A: ", mean(logpdf(spn, A)))
println("   dataset B: ", mean(logpdf(spn, B)))
