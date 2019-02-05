using SumProductNetworks

# Load an automatic differentiation (AD) package, this case we use ForwardDiff.
# Note that using a reverse mode AD is more efficient.
using ForwardDiff

# Loading some utility functions.
using Random, LinearAlgebra

# Generate some training data.
N = 1000
D = 4
x = rand(1:2, N, D)

# Helper function to construct an SPN structure.
function buildSPN(T::Type{<:Real}, x; K = 2)
    N, D = size(x)
    root = FiniteSumNode{T}(;D = D);

    for k in 1:K
        add!(root, FiniteProductNode(D = D), map(T, log(1.0/K)))
        for d in 1:D
            add!(root[k], FiniteSumNode{T}(D = D))
            for l in 1:2
                add!(root[k][d], IndicatorNode(l, d), map(T, log(0.5)))
            end
        end
    end
    updatescope!(root)
    return SumProductNetwork(root)
end

# Number of children under a sum node.
K = 2

# Construct the SPN.
spn = buildSPN(Real, x, K = K)

# Collect all sum nodes.
snodes = filter(n -> isa(n, SumNode), values(spn))

# Helper function used by ForwardDiff.
function f(θ)

    N, D = size(x)
    c = 1
    for i in 1:length(snodes)
        K = length(snodes[i])

        # Extraction of parameters.
        ϕ = θ[c:(c+K-1)]

        # Copy parameters to SPN node.
        snodes[i].logweights[:] = log.(projectToPositiveSimplex!(ϕ))

        c += length(snodes[i])
    end

    # Return llh.
    return -mean(logpdf(spn.root, x))
end

# Create an initial guess for all parameters and project them to the positive simplex.
q = mapreduce(n -> projectToPositiveSimplex!(rand(length(n))), vcat, snodes)

@info "Number of parameters to optimize: $(length(q))"

# Configure the AD.
chunk = ForwardDiff.Chunk(10)
config = ForwardDiff.GradientConfig(f, q, chunk)

# Construct AD function.
g(θ) = ForwardDiff.gradient!(similar(θ), f, θ, config)

# Number of iterations.
τ = 100

# log likelihood values.
ℓ = zeros(τ)

@info "Starting optimization for $(τ) iterations..."
@info "Initial log likelihood: $(-f(q))"

for i in 1:τ

    # Save LLH.
    ℓ[i] = -f(q)
    # Gradient step.
    δq = g(q)
    q .-= 0.1 * δq
end

@info "Log likelihood of optimized model: $(last(ℓ))"