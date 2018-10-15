export bmitest

"""
    Test if discrete sets are independent.

    test(X, Y) -> p

    Computes the Independence Test by Margaritis and Thrun for X and Y.
    The test returns the probability of X and Y being independent.
"""
function bmitest(X::Vector{Int}, Y::Vector{Int}; α = 1.0)

    uniqueX = unique(X)
    uniqueY = unique(Y)

    if any(uniqueX .≠ 1:length(uniqueX))
        t = zeros(Int, size(X));
        for k = 1:length(uniqueX)
            t[X .== uniqueX[k]] .= Int(k)
        end
        X = copy(t)
    end

    if any(uniqueY .≠ 1:length(uniqueY))
        t = zeros(Int, size(Y));
        for k = 1:length(uniqueY)
            t[Y .== uniqueY[k]] .= Int(k)
        end
        Y = copy(t)
    end

    L  = length(X)
    NX = length(uniqueX)
    NY = length(uniqueY)

    hX = counts(X, 0:NX)
    hY = counts(Y, 0:NY)
    h  = counts((X.-1)*NY .+ Y, 0:(NX*NY))

    PRD = lgamma(α*NX*NY) - lgamma(α*NX*NY + L) + sum( lgamma.(h .+ α) ) - NX*NY*lgamma(α)
    PRI = lgamma(α*NX) - lgamma(α*NX + L) + sum( lgamma.(hX .+ α) ) - NX * lgamma(α) +
            lgamma(α*NY) - lgamma(α*NY + L) + sum( lgamma.(hY .+ α) ) - NY * lgamma(α)

    m = max(PRD, PRI)

    logP = PRI - m - log(exp(PRD-m) + exp(PRI-m))
    return exp(logP)
end
