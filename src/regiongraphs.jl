export draw
export FactorizedAtomicRegion

abstract type AbstractRegionGraphNode end

function setscope!(n::AbstractRegionGraphNode, dims::Vector{Int})
    fill!(n.scopeVecs, false)
    @inbounds n.scopeVecs[dims] .= true
    empty!(n.scope)
    append!(n.scope, dims)
end
nscope(d::AbstractRegionGraphNode) = sum(d.scopeVecs)
hasscope(d::AbstractRegionGraphNode) = any(d.scopeVecs)
scope(d::AbstractRegionGraphNode) = d.scope

function setscope!(n::AbstractRegionGraphNode, i::Int, s::Bool)
    @inbounds n.scopeVecs[i] = s
    push!(n.scope, s)
end

id(d::AbstractRegionGraphNode) = d.id

"""
    topoligicalorder(root::AbstractRegionGraphNode)

Return topological order of the nodes in the SPN.
"""
function topoligicalorder(root::AbstractRegionGraphNode)
  visitedNodes = Vector{AbstractRegionGraphNode}()
  visitNode!(root, visitedNodes)
  return visitedNodes
end

function visitNode!(node::AbstractRegionGraphNode, visitedNodes)
  # check if we have already visited this node
  if !(node in visitedNodes)

    # visit node
    if haschildren(node)
        for child in children(node)
            visitNode!(child, visitedNodes)
        end
    end
    push!(visitedNodes, node)
  end
end

"""
    Region node.

Parameters:

* ids::Symbol                                           Id
* scopeVecs::Vector{Bool}                               Active dimensions (D)
* logweights::Matrix{<:Real}                            Log weights of sum nodes (Ch x K)
* prior::Dirichlet                                      Prior for sum nodes
* children::Vector{AbstractRegionGraphNode}             Children of region

"""
struct SumRegion{T<:Real,V<:AbstractRegionGraphNode} <: AbstractRegionGraphNode
    id::Symbol
    scopeVecs::Vector{Bool}
    logweights::Matrix{T}
    children::Vector{V}
end

haschildren(n::SumRegion) = !isempty(n.children)
children(n::SumRegion) = n.children

function updatescope!(n::SumRegion)
    updatescope!.(children(n))
    setscope!(n, scope(first(n.children)))
end

# Multi-threaded LSE
function _logsumexp(y::AbstractMatrix{T}, lw::AbstractMatrix{T}) where {T<:Real}
    r = zeros(T,size(lw,2),size(y,2))
    Threads.@threads for j in 1:size(y,2)
        for k in 1:size(lw,2)
            @inbounds begin
                yi_max = y[1,j] + lw[1,k]
                for i in 2:size(y,1)
                    yi_max = max(y[i,j]+lw[i,k], yi_max)
                end
                s = zero(T)
                for i in 1:size(y,1)
                    s += exp(y[i,j]+lw[i,k] - yi_max)
                end
                r[k,j] = log(s) + yi_max
            end
        end
    end
    return transpose(r)
end

function _getchildlogpdf(child::AbstractRegionGraphNode, out::AxisArray{V}) where {V<:AbstractMatrix}
    return out[id(child)]
end

function _getchildlogpdf(child::PartitionGraphNode, out::AxisArray{V}) where {V<:AbstractMatrix}
    childids = map(id, children(child))
    lp_ = out[childids]
    return reduce(_cross_prod, lp_)
end

function logpdf(n::SumRegion, x::AbstractMatrix{T}) where {T<:Real}
    out = zeros(T, size(x,1), lengh(n))
    logdf!(n, x, out)
    return out
end

function logpdf!(n::SumRegion{T,Tc},
                 x::AbstractMatrix{T},
                 out::AbstractMatrix{T}) where {T<:Real,Tc<:FactorizedAtomicRegion}

    for k1 = 1:length()




    lp_ = _getchildlogpdf.(children(d), Ref(out))
    out[id(d)] = _logsumexp(reduce(hcat, lp_)', d.logweights)
    return out
end

"""
    logmllh!(n::RegionGraphNode, x::AbstractMatrix{T}, out::AxisArray{V}, y::AxisArray, d::Int)

Log marginal likelihood of a region node in a region graph for dimension d. (in-place)
"""
function logmllh!(n::RegionGraphNode,
                  x::AbstractMatrix{T},
                  out::AxisArray{V},
                  y::AxisArray,
                  d::Int) where {T<:Real,V<:AbstractVector}
    K = length(n)
    lp_ = out[id.(children(n))]
    lp_ = reduce(vcat, lp_) .* n.active
    out[id(n)] = vec(sum(lp_, dims = 1))
    return out
end

function _cross_prod(x1::AbstractMatrix{T}, x2::AbstractMatrix{T}) where {T<:Real}
    nx, ny = size(x1,2), size(x2,2)
    r = zeros(T, size(x1,1), nx*ny)
    Threads.@threads for j in 1:nx*ny
        @inbounds begin
            j1, j2 = Base._ind2sub((nx,ny), j)
            r[:,j] .= x1[:,j1] .+ x2[:,j2]
        end
    end
    return r
end

"""
    Factorized distribution of the scope.

Parameters:

* ids::Symbol                               Id
* scopeVecs::Vector{Bool}                   Active dimensions (D)
* likelihoods::Vector{Type{Distribution}}   Likelihood functions (D)
* priors::Vector{Distribution}              Priors for each dimension (D)
* parameters::Vector{Matrix}                Parameters for each k≦K for each dimension (D)

"""
struct FactorizedAtomicRegion <: AbstractRegionGraphNode
    id::Symbol
    K::Int
    scopeVecs::Vector{Bool}
    scope::Vector{Int}
    likelihoods::Vector{Function}
    priors::Vector{Distribution}
    parameters::Vector{Matrix}
end

function FactorizedAtomicRegion(likelihoods::Vector{Function},
                                         parameters::Vector{<:Matrix},
                                         D::Int;
                                         priors::Vector{Distribution}=Vector{Distribution}()
                                        )
    K = size(first(parameters), 2)
    for p in parameters
        @assert size(p,2) == K
    end
    svec = falses(D)
    return FactorizedAtomicRegion(gensym(), K, Int[], svec, likelihoods, priors, parameters)
end

function draw(priors::Vector{Distribution}, K::Int)
    return map(prior -> prior isa UnivariateDistribution ? reshape(rand(prior,K),1,:) : rand(prior,K), priors)
end

@inline haschildren(n::FactorizedAtomicRegion) = false
@inline updatescope!(n::FactorizedAtomicRegion) = scope(d)
@inline length(n::FactorizedAtomicRegion) = n.K

function apply!(n::FactorizedAtomicRegion,
                x::AbstractMatrix{T},
                out::AbstractVector{V},
                k::Int
               ) where {T,V}
    for d in scope(n)
        @inbounds begin
            θ = @view(n.parameters[d][:,k])
            out[:] += n.likelihoods[d].(@view(x[:,d]), θ...)
        end
    end
end

function _logpdf!(n::FactorizedAtomicRegion,
                  x::AbstractMatrix{T},
                  out::AbstractMatrix{V}) where {T<:Real,V<:Real}
    fill!(out, zero(V))
    for k in 1:lenght(n)
        apply!(n, x, @view(out[:,k]), k)
    end
    return out
end

function logpdf(n::FactorizedAtomicRegion, x::AbstractMatrix{T}) where {T<:Real}
    N = size(x,1)
    K = length(n)
    lp_ = zeros(Float64, N, K)
    return _logpdf!(n, x, lp_)
end

function _pdf!(n::FactorizedAtomicRegion,
                  x::AbstractMatrix{T},
                  out::AbstractMatrix{V}) where {T<:Real,V<:Real}
    fill!(out, zero(V))
    ds = scope(n)
    for d in ds
        for (θ, lp) in zip(eachslice(n.parameters[d], dims=2), eachslice(out, dims=2))
            @inbounds lp[:] .*= exp.(n.likelihoods[d].(@view(x[:,d]), θ...))
        end
    end
    return out
end

function pdf(n::FactorizedAtomicRegion, x::AbstractMatrix{T}) where {T<:Real}
    N = size(x,1)
    K = length(n)
    lp_ = zeros(Float64, N, K)
    return _pdf!(n, x, lp_)
end
