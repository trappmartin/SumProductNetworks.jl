using Random
using SumProductNetworks


N = 1000
D = 4
X = randn(N, D)

spn = generate_spn(X, :learnspn)
updatescope!(spn)

@info("Initial mean log-likelihood:")
@info("   $(mean(logpdf(spn, X)))")

fit!(spn, X, :em)
@info("Mean log-likelihood after Expectation-Maximization:")
@info("   $(mean(logpdf(spn, X)))")
