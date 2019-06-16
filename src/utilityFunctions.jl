export complexity, depth, projectToPositiveSimplex!, initllhvals, initgradvals

"""
Get all nodes in topological order using Tarjan's algoritm.
"""
function getOrderedNodes(root)
  visitedNodes = Vector{SPNNode}()
  visitNode!(root, visitedNodes)
  return visitedNodes
end

function visitNode!(node::Node, visitedNodes)
  # check if we have already visited this node
  if !(node in visitedNodes)

    # visit node
    for child in children(node)
      visitNode!(child, visitedNodes)
    end

    push!(visitedNodes, node)
  end
end
visitNode!(node::Leaf, visitedNodes) = push!(visitedNodes, node)

"""

depth(S)

Compute the depth of the SPN rooted at S.
"""
depth(S::Node) = maximum(ndepth(child, 1) for child in children(S))
depth(S::Leaf) = 0

ndepth(S::Node, d::Int) = maximum(ndepth(child, d+1) for child in children(S))
ndepth(S::Leaf, d::Int) = d

"""

    complexity(S)

Compute the complexity (number of free parameters) of the SPN rooted at S.
"""
complexity(spn::SumProductNetwork) = complexity(spn.root)
complexity(node::SumNode) = mapreduce(c -> complexity(c), +, children(node)) + length(node)
complexity(node::ProductNode) = mapreduce(c -> complexity(c), +, children(node))
complexity(node::IndicatorNode) = 0
complexity(node::UnivariateNode) = length(fieldnames(node.d))
complexity(node::MultivariateNode) = length(fieldnames(node.d))

"""
  sub2ind2(size1, ind1, ind2) -> linear index
"""
sub2ind2(s1, i, j) = i + (j-1)*s1

"""
  sub2ind3(size1, size2, ind1, ind2, ind3) -> linear index
"""
sub2ind3(s1, s2, i, j, k) = i + (j-1)*s1 + (k-1)*s1*s2

"""

    projectToPositiveSimplex!(q::AbstractVector{<:Real}; lowerBound = 0.0, s = 1.0)

Project q to the positive simplex to ensure sum(q) == s.

##### Details
See Algorithm 1 in:
    Duchi, J., Shalev-Shwartz, S., Singer, Y., and Chandra, T.: Efficient projections onto the L 1-ball for learning in high dimensions. In proceeding of ICML 2008

"""
function projectToPositiveSimplex!(q::AbstractVector{<:Real}; lowerBound = 0.0, s = 1.0)

    if sum(q) == 0.0
        q[:] = ones(length(q)) / length(q)
        return q
    end

    if (sum(q) == s) & all(q .> lowerBound)
        return q
    end

    N = length(q)

    U = sort(q, rev=true)
    CSU = cumsum(U)
    CSUU = U .* collect(1:N) .>= (CSU .- s)
    ρ = maximum(findall(CSUU))
    θ = (CSU[ρ] - s) / ρ

    q[:] .-= θ
    q[q .< lowerBound] .= lowerBound
    return q
end


"""
    initllhvals(spn::SumProductNetwork, X::AbstractMatrix)

Construct a log likelihoods data-structure using `spn` and `X`.
"""
function initllhvals(spn::SumProductNetwork, X::AbstractMatrix)
    return initllhvals(spn, size(X, 1))
end

function initllhvals(spn::SumProductNetwork, X::AbstractVector)
    return initllhvals(spn, 1)
end

function initllhvals(spn::SumProductNetwork, N::Int)
    idx = Axis{:id}(collect(keys(spn)))
    if N == 1
        return AxisArray(ones(length(idx)) * -Inf, idx)
    else
        return AxisArray(ones(N, length(idx)) * -Inf, 1:N, idx)
    end
end


"""
    initgradvals(spn::SumProductNetwork)

Construct a gradient (w.r.t. nodes and leaves) data-structure using `spn` and `X`.
"""
function initgradvals(spn::SumProductNetwork, X::AbstractMatrix)
    return initllhvals(spn, size(X, 1))
end

function initgradvals(spn::SumProductNetwork, X::AbstractVector)
    return initllhvals(spn, 1)
end

function initgradvals(spn::SumProductNetwork, N::Int)
    idx = Axis{:id}(collect(keys(spn)))
    if N == 1
        return AxisArray(ones(length(idx)) * -Inf, idx)
    else
        return AxisArray(ones(N, length(idx)) * -Inf, 1:N, idx)
    end
end
