export simplify!, complexity, depth, prune!, order, copySPN

"""
Type definition for topological ordering.
"""
type SPNMarking
  ordering::Array
  unmarked::Array
end

"""
Compute topological order of an SPN using Tarjan's algoritm.
"""
function order(root)

    function visit!(node, data)

        if node in data.unmarked

            if in(:children, fieldnames(node))
                for n in node.children
                    data = visit!(n, data)
                end
            end

            idx = findfirst(data.unmarked, node)
            splice!(data.unmarked, idx)
            push!(data.ordering, node)
        end

        data
    end

    marking = SPNMarking(Any[], Any[])
    flat!(marking.unmarked, root)

    while(Base.length(marking.unmarked) > 0)
        n = marking.unmarked[end]

        visit!(n, marking)

    end

    marking.ordering
end

function flat!(nodes::Array, node)
    if in(:children, fieldnames(node))
        for n in node.children
            flat!(nodes, n)
        end
    end

    if !(node in nodes)
        push!(nodes, node)
    end

    nodes
end

"""

	depth(S)

Compute the depth of the SPN rooted at S.
"""
function depth(S::Node)
	return maximum(ndepth(child, 1) for child in children(S))
end

function depth(S::Leaf)
	return 0
end

function ndepth(S::Node, d::Int)
	return maximum(ndepth(child, d+1) for child in children(S))
end

function ndepth(S::Leaf, d::Int)
	return d
end

"""

	complexity(S)

Compute the complexity (number of free parameters) of the SPN rooted at S.
"""
function complexity(S)
	return sum(map(n -> length(n), filter(n -> isa(n, SumNode), order(S))))
end

"""

	simplify!

Simplify the structure of an SPN.
"""
function simplify!(S::SumNode)

	for child in children(S)
		simplify!(child)
	end

	childrentoremove = Int[]

	for (i, child) in enumerate(children(S))
		if isa(child, SumNode) & (length(parents(child)) == 1)
			# collaps child if its a sum
			toremove = Int[]
			for (j, k) in enumerate(children(child))
				add!(S, k, child.weights[j] * S.weights[i])
				push!(toremove, j)
			end

			for k in reverse(toremove)
				remove!(child, k)
			end

			push!(childrentoremove, i)
		elseif isa(child, ProductNode) & (length(parents(child)) == 1) & (length(child) == 1)
			# collaps child if its a product over one child
			add!(S, child.children[1], S.weights[i])
			remove!(child, 1)
			push!(childrentoremove, i)
		end
	end

	for child in children(S)
		@assert findfirst(S .== child.parents) > 0
	end

	for child in reverse(childrentoremove)
		remove!(S, child)
	end

	for child in children(S)
		@assert findfirst(S .== child.parents) > 0
	end
end

function simplify!(S::ProductNode)

	for child in children(S)
		simplify!(child)
	end

	childrentoremove = Int[]

	for (i, child) in enumerate(children(S))
		if isa(child, ProductNode) & (length(parents(child)) == 1)
			# collaps child if its a product
			toremove = Int[]
			for (j, k) in enumerate(children(child))
				add!(S, k)
				push!(toremove, j)
			end

			for k in reverse(toremove)
				remove!(child, k)
			end

			push!(childrentoremove, i)
		elseif isa(child, SumNode) & (length(parents(child)) == 1) & (length(child) == 1)
			# collaps child if its a sum over one child
			add!(S, child.children[1])
			remove!(child, 1)
			push!(childrentoremove, i)
		end
	end

	for child in reverse(childrentoremove)
		remove!(S, child)
	end

	for child in children(S)
		@assert findfirst(S .== child.parents) > 0
	end
end

function simplify!(S)
end

"""

	prune!(S, σ)

Prune away leaf nodes & sub-trees with std lower than σ.
"""
function prune!(S::SumNode, σ::Float64)

	for node in filter(n -> isa(n, SumNode), order(S))

	  toremove = Int[]

	  for (ci, child) in enumerate(children(node))
	    if isa(child, NormalDistributionNode)
	      if child.σ < σ
	        push!(toremove, ci)
	      end
	    elseif isa(child, ProductNode)
	      if any([isa(childk, NormalDistributionNode) for childk in children(child)])
	        drop = false
	        for childk in children(child)
	          if isa(childk, NormalDistributionNode)
	            if childk.σ < σ
	              drop = true
	            end
	          end
	        end
	        if drop
	          push!(toremove, ci)
	        end
	      end
	    end
	  end
	  reverse!(toremove)

	  for ci in toremove
	    remove!(node, ci)
	  end
	end

	for node in filter(n -> isa(n, Node), order(S))

	  toremove = Int[]
	  for (ci, child) in enumerate(children(node))
	    if isa(child, Node)
	      if length(child) == 0
	        push!(toremove, ci)
	      end
	    end
	  end
	  reverse!(toremove)

	  for ci in toremove
	    remove!(node, ci)
	  end
	end

end

"""

	prune!(S, σ)

Prune away leaf nodes & sub-trees with std lower than σ.
"""
function prune_llh!(S::SumNode, data::AbstractArray; minrange = 0.0)

	nodes = order(S)

	maxId = maximum(Int[node.id for node in nodes])
	llhval = Matrix{Float64}(size(data, 1), maxId)

	for node in nodes
			eval!(node, data, llhval)
	end

	llhval -= maximum(vec(mean(llhval, 1)))

	rd = minrange + (rand(maxId) * (1-minrange))

	drop = rd .> exp(vec(mean(llhval, 1)))

	for node in filter(n -> isa(n, Node), order(S))

	  toremove = Int[]

	  for (ci, child) in enumerate(children(node))
	    if isa(child, ProductNode)
				if any([isa(childk, NormalDistributionNode) for childk in children(child)])
	        for childk in children(child)
	          if drop[childk.id]
	            drop[child.id] = true
	          end
	        end
				end
			end

			if drop[child.id]
				push!(toremove, ci)
			end
	  end

	  reverse!(toremove)

	  for ci in toremove
	    remove!(node, ci)
	  end
	end

	for node in filter(n -> isa(n, Node), order(S))

	  toremove = Int[]
	  for (ci, child) in enumerate(children(node))
	    if isa(child, Node)
	      if length(child) == 0
	        push!(toremove, ci)
	      end
	    end
	  end
	  reverse!(toremove)

	  for ci in toremove
	    remove!(node, ci)
	  end
	end

end

function prune_uniform!(S::SumNode, p::Float64)

	nodes = order(S)
	maxId = maximum(Int[node.id for node in nodes])

	drop = rand(maxId) .> p

	for node in filter(n -> isa(n, Node), order(S))

	  toremove = Int[]

	  for (ci, child) in enumerate(children(node))
	    if isa(child, ProductNode)
				if any([isa(childk, NormalDistributionNode) for childk in children(child)])
	        for childk in children(child)
	          if drop[childk.id]
	            drop[child.id] = true
	          end
	        end
				end
			end

			if drop[child.id]
				push!(toremove, ci)
			end
	  end

	  reverse!(toremove)

	  for ci in toremove
	    remove!(node, ci)
	  end
	end

	for node in filter(n -> isa(n, Node), order(S))

	  toremove = Int[]
	  for (ci, child) in enumerate(children(node))
	    if isa(child, Node)
	      if length(child) == 0
	        push!(toremove, ci)
	      end
	    end
	  end
	  reverse!(toremove)

	  for ci in toremove
	    remove!(node, ci)
	  end
	end

end

function copySPN(source::SumNode; idIncrement = 0)

	nodes = order(source)
  destinationNodes = Vector{SPNNode}()
  id2index = Dict{Int, Int}()

  for node in nodes

    if isa(node, NormalDistributionNode)
      dnode = NormalDistributionNode(copy(node.id) + idIncrement, copy(node.scope))
      dnode.μ = copy(node.μ)
      dnode.σ = copy(node.σ)
      push!(destinationNodes, dnode)
      id2index[dnode.id] = length(destinationNodes)
		elseif isa(node, MultivariateFeatureNode)
			dnode = MultivariateFeatureNode(copy(node.id) + idIncrement, copy(node.scope))
			dnode.weights[:] = node.weights
			push!(destinationNodes, dnode)
			id2index[dnode.id] = length(destinationNodes)
    elseif isa(node, IndicatorNode)
      dnode = IndicatorNode(copy(node.id) + idIncrement, copy(node.value), copy(node.scope))
      push!(destinationNodes, dnode)
      id2index[dnode.id] = length(destinationNodes)
    elseif isa(node, SumNode)
      dnode = SumNode(copy(node.id) + idIncrement, scope = copy(node.scope))
      cids = Int[child.id for child in children(node)]
      for (i, cid) in enumerate(cids)
        add!(dnode, destinationNodes[id2index[cid + idIncrement]], copy(node.weights[i]))
      end
			push!(destinationNodes, dnode)
			id2index[dnode.id] = length(destinationNodes)
		elseif isa(node, ProductNode)
			dnode = ProductNode(copy(node.id) + idIncrement, scope = copy(node.scope))
			cids = Int[child.id for child in children(node)]
			for (i, cid) in enumerate(cids)
				add!(dnode, destinationNodes[id2index[cid + idIncrement]])
			end

			push!(destinationNodes, dnode)
			id2index[dnode.id] = length(destinationNodes)

		else
			throw(TypeError(node, "Node type not supported."))
		end

	end

  return destinationNodes[end]
end
