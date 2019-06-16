using Random
using SumProductNetworks


N = 1000
D = 4

@info("Make dataset A and its shifted version B")
A = hcat([randn(N) .+ (2 + 0.5*i) for i in 1:D]...)
B = hcat([randn(N) .+ (4 + 0.5*i) for i in 1:D]...)

@info("Generate an SPN by LearnSPN with dataset A")
spn = generate_spn(A, :learnspn)
updatescope!(spn)

@info("Mean log-likelihood:")
@info("   dataset A: $(mean(logpdf(spn, A)))")
@info("   dataset B: $(mean(logpdf(spn, B)))")
