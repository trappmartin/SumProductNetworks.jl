abstract type AbstractRegionGraphNode end

function setscope!(n::AbstractRegionGraphNode, dims::Vector{Int})
    fill!(n.scopeVecs, false)
    n.scopeVecs[dims] .= true
end
nscope(d::AbstractRegionGraphNode) = sum(d.scopeVecs)
hasscope(d::AbstractRegionGraphNode) = any(d.scopeVecs)
scope(d::AbstractRegionGraphNode) = findall(d.scopeVecs)

function setscope!(d::AbstractRegionGraphNode, i::Int, s::Bool)
    d.scopeVecs[i] = s
end

length(d::AbstractRegionGraphNode) = size(d.obsVecs,2)
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
* obsVecs::Matrix{Bool}                                 Active observations (N x K)
* logweights::Matrix{<:Real}                            Log weights of sum nodes (Ch x K)
* prior::Dirichlet                                      Prior for sum nodes
* children::Vector{AbstractRegionGraphNode}             Children of region

"""
struct RegionGraphNode{T<:Real} <: AbstractRegionGraphNode
    id::Symbol
    scopeVecs::Vector{Bool}
    obsVecs::Matrix{Bool}
    logweights::Matrix{T}
    active::Matrix{Bool}
    prior::Dirichlet
    children::Vector{<:AbstractRegionGraphNode}
end

"""
    Partition node.

Parameters:

* ids::Symbol                                   Id
* scopeVecs::Vector{Bool}                       Active dimensions (D)
* obsVecs::Matrix{Bool}                         Active observations (N x K)
* prior::Dirichlet                              Prior on product nodes
* children::Vector{<:AbstractRegionGraphNode}   Child region nodes

"""
struct PartitionGraphNode <: AbstractRegionGraphNode
    id::Symbol
    scopeVecs::Vector{Bool}
    obsVecs::Matrix{Bool}
    prior::Dirichlet
    children::Vector{<:AbstractRegionGraphNode}
end

haschildren(d::RegionGraphNode) = !isempty(d.children)
children(d::RegionGraphNode) = d.children

function updatescope!(d::RegionGraphNode)
    updatescope!.(children(d))
    setscope!(d, scope(first(d.children)))
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

"""
    logpdf(d::RegionGraphNode, x::AbstractMatrix{T})

Log pdf of a region node in a region graph.
"""
function logpdf(d::RegionGraphNode, x::AbstractMatrix{T}) where {T<:Real}
    lp_ = logpdf.(d.children, Ref(x))
    return _logsumexp(reduce(hcat, lp_)', d.logweights)
end

"""
    logpdf!(d::RegionGraphNode, x::AbstractMatrix{T}, out::AxisArray{V})

Log pdf of a region node in a region graph. (in-place)
"""
function logpdf!(d::RegionGraphNode, x::AbstractMatrix{T}, out::AxisArray{V}) where {T<:Real,V<:AbstractMatrix}
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

## Partition Graph Node

haschildren(d::PartitionGraphNode) = !isempty(d.children)
children(d::PartitionGraphNode) = d.children

function updatescope!(d::PartitionGraphNode)
    s_ = mapreduce(updatescope!, vcat, children(d))
    setscope!(d, unique(s_))
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
    logpdf(d::PartitionGraphNode, x::AbstractMatrix{T})

Log pdf of a partition node in a region graph.
"""
function logpdf(d::PartitionGraphNode, x::AbstractMatrix{T}) where {T<:Real}
    childrn_ = children(d)
    lp_ = logpdf.(childrn_, Ref(x))
    return reduce(_cross_prod, lp_)
end

"""
    logpdf!(d::RegionGraphNode, x::AbstractMatrix{T}, out::AxisArray{V})

Log pdf of a partition node in a region graph. (in-place)
"""
function logpdf!(d::PartitionGraphNode, x::AbstractMatrix{T}, out::AxisArray{V}) where {T<:Real,V<:AbstractMatrix}
    return out
end

"""
    Factorized distribution of the scope.

Parameters:

* ids::Symbol                               Id
* scopeVecs::Vector{Bool}                   Active dimensions (D)
* obsVecs::Matrix{Bool}                     Active observations (N x K)
* priors::Vector{Distribution}              Priors for each dimension (D)
* likelihoods::Vector{Distribution}         Likelihood functions (D)
* sstats::Matrix{AbstractSufficientStats}   Sufficient stats (D x K)

"""
struct FactorizedDistributionGraphNode <: AbstractRegionGraphNode
    id::Symbol
    scopeVecs::Vector{Bool}
    obsVecs::Matrix{Bool}
    likelihoods::Vector{<:Distribution}
    parameters::Matrix
end

haschildren(d::FactorizedDistributionGraphNode) = false
updatescope!(d::FactorizedDistributionGraphNode) = scope(d)
obs(d::FactorizedDistributionGraphNode, k::Int) = findall(d.obsVecs[:,k])

function _logpdf!(n::FactorizedDistributionGraphNode,
                  x::AbstractMatrix{T},
                  out::AbstractMatrix{V}) where {T<:Real,V<:Real}
    fill!(out, zero(V))
    ds = scope(n)
    for d in ds
        sstats_ = n.sstats[d,:]
        θ = postparams.(Ref(n.priors[d]), sstats_)
        lpd_ = _logpostpdf.(Ref(n.likelihoods[d]), Ref(view(x,:,d)), θ)
        out .+= reduce(hcat, lpd_)
    end
    return out
end

"""
    logpdf(d::FactorizedDistributionGraphNode, x::AbstractMatrix{T})

Log pdf of an atomic region node in a region graph.
"""
function logpdf(n::FactorizedDistributionGraphNode, x::AbstractMatrix{T}) where {T<:Real}
    (N, D) = size(x)
    K = length(n)
    lp_ = zeros(Float64, N, K)
    return _logpdf!(n, x, lp_)
end

"""
    logpdf!(d::FactorizedDistributionGraphNode, x::AbstractMatrix{T}, out::AxisArray{V})

Log pdf of an atomic node in a region graph. (in-place)
"""
function logpdf!(n::FactorizedDistributionGraphNode, x::AbstractMatrix{T}, out::AxisArray{V}) where {T<:Real,V<:AbstractMatrix}
    (N, D) = size(x)
    K = length(n)

    i = findfirst(out.axes[1].val .== id(n))
    if !isassigned(out, i)
        out[id(n)] = zeros(Float64, N, K)
    end

    _logpdf!(n, x, out[id(n)])
    return out
end

