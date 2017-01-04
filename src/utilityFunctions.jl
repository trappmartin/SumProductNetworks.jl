export simplify!, complexity, depth

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

	for child in reverse(childrentoremove)
		remove!(S, child)
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
end

function simplify!(S)
end
