# Implementation of collapsed Gibbs sampling with conjugate prior.
# Note: This current implementation assumes a Gaussian Distribution and a Normal-Inverse-Wishart as prior.

function gibbs_iteration!{T<:Float64, U<:Bool}(root::SPNNode, X::Matrix{T}, Z::Matrix{U}, base::MultivariateDistribution, N::Integer, D::Integer, alpha::Float64)

  covar_prior = 10
  change = 0.0

  @inbounds for i = 1:N

    # find last assignment
    idx = findin(Z[i,:], true)[1]

    # remove assignment
    Z[i,idx] = 0

    # remove if no sample assigned to node
    if (sum(Z[:,idx]) == 0)
      remove(root, idx)
      Z = Z[:,[1:idx-1, idx+1:end]]
    else
      # TODO: this should be an update not a reassignment

      m = mean(X[Z[:,idx],:], 1)
      if (sum(Z[:,idx]) == 1)
        cc = eye(D)
      else
        cc = cov(X[Z[:,idx],:]) + (covar_prior * eye(D) * 0.0001)
      end

      d = MvNormal(vec(m), cc)
      c = build_multivariate(d, [1:D])

      # assign updated node
      root.children[idx] = c

      # update weight
      root.weights[idx] = float(sum(Z[:,idx])) / (alpha + N - 1)
    end

    n_clusters = size(Z)[2]

    # compute posterior
    _posterior =  @parallel (vcat) for child = 1:n_clusters
      log(root.weights[child]) + llh(root.children[child], X[i,:])
    end

    base_posterior = log(alpha / (alpha + N - 1)) + logpdf(base, X[i,:]')
    _posterior = vcat(_posterior, base_posterior)

    # to probabilities
    max = maximum(_posterior)
    prob = exp(_posterior - (log(sum(exp(_posterior - max))) + max))

    # find new assignment
    rd = rand()
    _k = sum(cumsum(prob) .< rd) + 1

    if _k > n_clusters

      if _k != idx
        change += sum(abs(X[i,:]))
      end

      # set assignment
      z = bool(zeros(N))
      z[i] = true
      Z = hcat(Z, z)

      d = MvNormal(vec(X[i,:]), eye(D) * covar_prior)
      c = build_multivariate(d, [1:D])
      add(root, c, 1.0 / (alpha + N - 1))
    else

      # set assignment
      Z[i, _k] = true

      m = mean(X[Z[:,_k],:], 1)
      cc = cov(X[Z[:,_k],:]) + (covar_prior * eye(D) * 0.0001)

      if _k != idx
        change += sum(abs(mean(root.children[_k].dist) - m'))
      end

      d = MvNormal(vec(m), cc)
      c = build_multivariate(d, [1:D])

      root.children[_k] = c
      root.weights[_k] = float(sum(Z[:,_k])) / (alpha + N - 1)
    end

  end

  return (root, Z, llh(root, X), change / N)

end

function gibbs_sampling{T<:Float64}(root::SPNNode, X::Matrix{T}, base::MultivariateDistribution, alpha::Float64, max_iterations::Integer)

  (N, D) = size(X)
  Z = bool(eye(N))
  n_clusters = N

  iter = 1


end
