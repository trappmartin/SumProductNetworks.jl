export NormalInverseGamma

## Gaussian with Normal Inverse Gamma prior ##
struct NormalInverseGamma{T<:Real} <: Distribution{Multivariate,Continuous}
    μ::T
    ν::T
    a::T
    b::T
end

Distributions.length(d::NormalInverseGamma) = 2

function Distributions.logpdf(d::NormalInverseGamma, μ, σ²)
    lp = _invgammalogpdf(d.a, d.b, σ²)
    lp += normlogpdf(d.μ, sqrt(σ²) / d.ν, μ)
    return l
end

@inline _invgammalogpdf(a::T, b::T, x::T) where {T<:Real} = a*log(b)-lgamma(a)-(a+one(T))*log(x)-b/x
