"""
	Test if discrete sets are independent.

	test(X, Y) -> p, log(P)

	Computes the Independence Test by Margaritis and Thrun for X and Y.
	The result p gives the probability of X and Y being independent. And
	log(p) is the logarithm of p.

"""
function bmiTest(X::Vector{Int}, Y::Vector{Int}; α = 1.0)

	uniqueX = unique(X)
	uniqueY = unique(Y)

	if any(uniqueX .≠ 1:length(uniqueX))
	    t = zeros(size(X));
	    for k = 1:length(uniqueX)
	        t[X .== uniqueX[k]] = k
	    end
	    X = copy(t)
	end

	if any(uniqueY .≠ 1:length(uniqueY))
	    t = zeros(size(Y));
	    for k = 1:length(uniqueY)
	        t[Y .== uniqueY[k]] = k
	    end
	    Y = copy(t)
	end

	L  = length(X)
	NX = length(uniqueX)
	NY = length(uniqueY)

	(eX, hX) = hist(X, 0:NX)
	(eY, hY) = hist(Y, 0:NY)
	(e, h)  = hist((X-1)*NY + Y, 0:(NX*NY))

	PRD = lgamma(α*NX*NY) - lgamma(α*NX*NY + L) + sum( lgamma(h  + α) ) - NX*NY*lgamma(α)
	PRI = lgamma(α*NX)    - lgamma(α*NX    + L) + sum( lgamma(hX + α) ) - NX * lgamma(α) +
	      lgamma(α*NY)    - lgamma(α*NY    + L) + sum( lgamma(hY + α) ) - NY * lgamma(α)

	m = maximum([PRD, PRI])

	logP = PRI-m - log(exp(PRD-m) + exp(PRI-m))
	p = exp(logP)

	return (p, logP)
end
