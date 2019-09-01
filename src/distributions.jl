export NormalInverseGamma

## Gaussian with Normal Inverse Gamma prior ##
struct NormalInverseGamma{T<:Real} <: Distribution{Multivariate,Continuous}
    μ::T
    ν::T
    a::T
    b::T
end
NormalInverseGamma() = NormalInverseGamma(0.0, 1.0, 1.0, 1.0)

Distributions.length(d::NormalInverseGamma) = 2

function Distributions.logpdf(d::NormalInverseGamma, μ::T, σ²::T) where {T}
    lp = _invgammalogpdf(d.a, d.b, σ²)
    lp += normlogpdf(d.μ, sqrt(σ²) / d.ν, μ)
    return l
end

@inline _invgammalogpdf(a::T, b::T, x::T) where {T<:Real} = a*log(b)-lgamma(a)-(a+one(T))*log(x)-b/x

function Distributions._rand!(rng::AbstractRNG, d::NormalInverseGamma, out::AbstractArray{T,2}) where {T}
    for p in eachslice(out, dims=[2])
        @inbounds begin
            p[2] = rand(InverseGamma(d.a, d.b))
            p[1] = rand(Normal(d.μ, sqrt(p[2] / d.ν )))
        end
    end
    return out
end

@inline Distributions.params(d::NormalInverseGamma) = (d.μ, d.ν, d.a, d.b)
