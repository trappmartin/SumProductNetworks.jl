export ratspn
#export templateLeaves, templatePartition, templateRegion

function templateLeaves(likelihoods::AbstractVector{<:Distribution},
                        priors::AbstractVector{<:Distribution},
                        N::Int, D::Int, K::Int)
    scopeVec = zeros(Bool, D)
    parameters = map(prior -> rand(prior,K), priors)

    return FactorizedDistributionGraphNode(
                                    gensym("fact"),
                                    scopeVec,
                                    likelihoods,
                                    priors,
                                    parameters
                                   )
end

function templatePartition(likelihoods::AbstractVector,
                           N::Int, D::Int,
                           K_sum::Int, K_prod::Int,
                           J::Int, K::Int,
                           depth::Int, maxdepth::Int)

    children = if depth == maxdepth
        map(k -> templateLeaves(alpha_leaf_prior, priors_leaf, likelihoods, sstats, N, D, K), 1:K_prod)
    else
        map(k -> templateRegion(alpha_region_prior, alpha_partition_prior, alpha_leaf_prior,
                                priors_leaf, likelihoods, sstats, N, D, K_sum, K_prod, J, K, depth+1, maxdepth), 1:K_prod)
    end

    K_ = mapreduce(child -> length(child), *, children)
    scopeVec = zeros(Bool, D)
    obsVec = zeros(Bool, N, K_)

    return PartitionGraphNode(
                              gensym("partition"),
                              scopeVec,
                              obsVec,
                              prior,
                              children
                             )
end

function buildRegion(llhs::AbstractVector, priors::AbstractVector,
                        N::Int, D::Int,
                        K_sum::Int, K_prod::Int,
                        J::Int, K::Int,
                        depth::Int, maxdepth::Int; root = false)

    K_ = root ? 1 : K_sum
    children = if depth == maxdepth
        map(k -> buildLeaves(likelihoods, priors, N, D, K), 1:J)
    else
        map(k -> buildPartition(likelihoods, priors, N, D, K_sum, K_prod, J, K, depth+1, maxdepth), 1:J)
    end

    Ch = sum(length.(children))
    scopeVec = zeros(Bool, D)
    logweights = rand(Dirichlet(Ch, 1.0), K_)
    active = zeros(Bool, size(logweights)...)
    @assert size(logweights) == (Ch, K_)

    return RegionGraphNode( gensym("region"), scopeVec, logweights, children )
end


function ratspn(x::AbstractMatrix{T};
               Ksplits::Int=2,
               Kparts::Int=2,
               Ksums::Int=2,
               Kdists::Int=5,
               maxdepth::Int=2
              ) where {T<:Real}

    N,D = size(x)
    isdiscrete = map(d -> all(isinteger, x[:,d]), 1:D)

    K = map(d -> isdiscrete[d] ? length(unique(x[:,d])) : Inf, 1:D)

    llhs = map(d -> isdiscrete[d] ? Categorical(K[d]) : Normal(), 1:D)
    priors = map(d -> isdiscrete[d] ? Dirichlet(K[d], 1.0) : NormalInverseGamma(), 1:D)

    return ratspn(N,D, llhs, priors, Ksplits, Kparts, Ksums, Kdists, maxdepth)
end

function ratspn(N::Int, D::Int,
                llhs::AbstractVector{<:Distribution},
                priors::AbstractVector{<:Distribution},
                Ksplits::Int, # Number of partitions under each region
                Kparts::Int, # Number of sub-regions under each partition
                Ksums::Int, # Number of sum nodes per region
                Kdists::Int, # Number of distibutions per terminal region
                maxdepth::Int # Maximum number of consecutive region-partition pairs
                )

    # sanity checks
    @assert length(llhs) == D == length(priors)

    

    #return templateRegion()
end
