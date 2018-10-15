export complexity, depth

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
