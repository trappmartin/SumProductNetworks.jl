export simplify!, complexity, depth, prune!

"""

	depth(S)

Compute the depth of the SPN rooted at S.
"""
function depth(S::Node)
	return maximum(ndepth(child, 1) for child in children(S))
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
