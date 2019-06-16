export fit!

"""
    fit!(spn::SumProductNetwork, X::Matrix, algo::Symbol; ...)

Fit an SPN to data using a parameter learning algorithm.

Arguments:
* `spn`: Sum Product Network to fit.
* `X`: Data matrix.
* `algo`: Algorithm, one out of [:gd, :em].
"""
function fit!(spn::SumProductNetwork, X::Matrix, algo::Symbol; params...)
    if algo == :gd
        return gd!(spn, X)
    elseif algo == :em
        return em!(spn, X)
    else
        @error("Unknown structure learning algorithm: ", algo)
    end
end

gd!(spn, X; params...) = @error("Gradient descent learning is currently not supported.", 
                                "Refer to example/parameterOptimization.jl for an example.")

"""
    em!(spn, X; params...)

Run Expectation-Maximization algorithm to learn the parameters of SPN.

Peharz et al. "On the Latent Variable Interpretation in Sum-Product Networks." PAMI (2016).
"""
function em!(spn::SumProductNetwork, X::Matrix;
             maxiters::Integer=10, tol::Real=1e-3, epsilon::Real=1.0)
    # store log-likelihoods and gradient
    llhvals = initllhvals(spn, X)
    gradvals = initgradvals(spn, X)

    # store sufficient statistics of sum nodes and leaves
    suffstats = Dict{Symbol, Vector{Real}}()
    for n in values(spn)
        if isa(n, SumNode)
            suffstats[n.id] = Vector{Real}(undef, length(n))
        elseif isa(n, Leaf)
            suffstats[n.id] = Vector{Real}(undef, length(params(n)) + 1)
        end
    end

    # expectation-maximization
    prevlogpdf = -Inf
    for i in 1:maxiters
        # pre-compute log-likelihoods and gradient
        logpdf = mean(logpdf!(spn, X, llhvals))
        gradient!(spn, llhvals, gradvals)

        # initialize sufficient statistiscs
        for key in keys(suffstats)
            suffstats[key][:] .= 0.0
        end

        # accumulate sufficient statistics of sum nodes and leaves
        for n in values(spn)
            coeff = exp.(gradvals[:, n.id] + llhvals[:, n.id] - llhvals[:, spn.root.id])
            if isa(n, SumNode)
                suffstats[n.id] .+= sum(coeff.*transpose(exp.(weights(n))))
            elseif isa(n, Leaf)
                if isa(n, UnivariateNode)
                    if isa(n.dist, Normal)
                        suffstats[n.id][1] += sum(coeff)
                        suffstats[n.id][2] += sum(coeff.*X[:, n.scope])
                        suffstats[n.id][3] += sum(coeff.*X[:, n.scope].^2)
                        continue
                    end
                end
                @error("currently not supported leaf type: $(n)")
            end
        end

        # update parameters given the sufficient statistics calculated above
        for n in values(spn)
            if isa(n, SumNode)
                counts = suffstats[n.id] .+ epsilon
                n.logweights[:] = log.(counts./sum(counts))
            elseif isa(n, Leaf)
                if isa(n, UnivariateNode)
                    if isa(n.dist, Normal)
                        mean = suffstats[n.id][2]/suffstats[n.id][1]
                        var = suffstats[n.id][3]/suffstats[n.id][1] - mean^2
                        std = var > 0 ? sqrt(var) : sqrt(epsilon)
                        n.dist = Normal(mean, std)
                        continue
                    end
                end
                @error("currently not supported leaf type: $(n)")
            end
        end

        # test convergence
        if logpdf - prevlogpdf < tol
            break
        end
        prevlogpdf = logpdf
    end
end
