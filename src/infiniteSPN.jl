"Assignment Data Object"
type Assignments

	# datum to leaf assignments
	Z::Vector{Vector{Leaf}}

	# bucket sizes
	S::Dict{SPNNode, Int}

	Assignments(N::Int) = new( Base.map(i -> Vector{Leaf}(0), collect(1:N)), Dict{SPNNode, Int}())

end

call(p::Assignments, n::SPNNode) = p.S[n]

"Add new node to bucket list"
function add!(p::Assignments, n::SPNNode)
	if !haskey(p.S, n)
		p.S[n] = 0
	end
end

"Increment number of items in bucket"
function increment!(p::Assignments, n::SPNNode; i = 1)

	if !haskey(p.S, n)
		add!(p, n)
	end

	p.S[n] += i
end

"Decrement number of items in bucket"
function decrement!(p::Assignments, n::SPNNode; i = 1)

	p.S[n] -= i

end

"Assign datum to leaf node"
function assign!(p::Assignments, id::Int, n::Leaf)

	if isdefined(p.Z[id])
		push!(p.Z[id], n)
	else
		p.Z[id] = [n]
	end
end

"Get index function"
function getindex(p::Assignments, i::Int)
    p.Z[i]
end

#=
 *** Accutal infinite SPN code ***
 =#

 "Evaluate Sum-Node on data"
 function evalWithK{T<:Real}(root::SumNode,
    data::AbstractArray{T},
    llhvals::Dict{SPNNode, Array{Float64}},
    assign::Assignments,
    G0::ConjugatePostDistribution;
    α = 1.0,
    mirror = false)

   if !mirror
      evalSumInternal(root, data, llhvals, assign, G0, α = α)
   else
      evalProductInternal(root, data, llhvals, assign, G0, α = α)
   end
end

"Internal function of SumNode evaluation"
function evalSumInternal{T<:Real}(root::Node,
   data::AbstractArray{T},
   llhvals::Dict{SPNNode, Array{Float64}},
   assign::Assignments,
   G0::ConjugatePostDistribution;
   α = 1.0)

   p = ones(size(data, 2), Base.length(root.children) + 1) * -Inf

   for (ci, c) in enumerate(root.children)
      p[:,ci] = llhvals[c] + log(assign(c) / (assign(root) + α - 1))
   end

   p[:,end] = logpred(G0, data[root.scope,:]) + log( α / (assign(root) + α - 1) )

   # get k
   k = 0

   maxp = maximum(p, 2)
   p = exp(p .- maxp)

   k = BNP.rand_indices(p)

   # get node llh
   p = sum(p, 2)
   p = log(p) .+ maxp

   #p -= log(sum(root.weights))

   return (p, k)
end

"Evaluate Product-Node on data"
function evalWithK{T<:Real}(root::ProductNode,
   data::AbstractArray{T},
   llhvals::Dict{SPNNode, Array{Float64}},
   assign::Assignments,
   G0::ConjugatePostDistribution;
   α = 1.0,
   mirror = false)

   if mirror
      evalSumInternal(root, data, llhvals, assign, G0, α = α)
   else
      evalProductInternal(root, data, llhvals, assign, G0, α = α)
   end

end

"Evaluate Product-Node on data"
function evalProductInternal{T<:Real}(root::Node,
      data::AbstractArray{T},
      llhvals::Dict{SPNNode, Array{Float64}},
      assign::Assignments,
      G0::ConjugatePostDistribution;
      α = 1.0)

   _llh = [llhvals[c] for c in root.children]
   _llh = reduce(vcat, _llh)

   return (sum(_llh, 1), -1)
end

"Evaluate Univariate Node"
function evalWithK{T<:Real}(node::UnivariateNode,
   data::AbstractArray{T},
   llhvals::Dict{SPNNode, Array{Float64}},
   assign::Assignments,
   G0::ConjugatePostDistribution;
   α = 1.0,
   mirror = false)

   if ndims(data) > 1
       x = sub(data, node.scope, :)
       llh = logpdf(node.dist, x)
       return (llh, -1)
   else
       llh = logpdf(node.dist, data)
       return (llh, -1)
   end

end

"Evaluate Multivariate Node with ConjugatePostDistribution"
function evalWithK{T<:Real, U<:ConjugatePostDistribution}(node::MultivariateNode{U},
   data::AbstractArray{T},
   llhvals::Dict{SPNNode, Array{Float64}},
   assign::Assignments,
   G0::ConjugatePostDistribution;
   α = 1.0,
   mirror = false)

   llh = logpred(node.dist, data[node.scope,:])

 return (llh, -1)
end

"Spawn new child into SPN"
function spawnChild!(node::Node, G0::ConjugatePostDistribution)


   println("Spawning child")


   G = GaussianWishart(G0.mu0[node.scope], G0.kappa0, G0.nu0, G0.Sigma0[node.scope, node.scope])
   add!(node, MultivariateNode{ConjugatePostDistribution}(G, copy(node.scope)))
end

"Kill child from SPN"
function killChild!(node::SPNNode, assign::Assignments)

   p = get(node.parent)

   remove!(p, findfirst(p.children .== node))

   # apply to parent if necessary
   if assign(p) == 0
      killChild!(p, assign)
   end

end

"Recurse on Nodes, add node if necessary."
function recurseCondK!{T<:Real}(node::Node, ks::Dict{SPNNode, Int},
   data::AbstractArray{T}, idx::Int, assign::Assignments, G0::ConjugatePostDistribution)

   # check if we need to extend the number of children
   if ks[node] > Base.length(node.children)
      spawnChild!(node, G0)
   end

	if haskey(ks, node)
		recurseCondK!(node.children[ks[node]], ks, data, idx, assign, G0)
	else
		for child in node.children
			recurseCondK!(child, ks, data, idx, assign, G0)
		end
	end

   increment!(assign, node)
end

"Recurse on Leafs"
function recurseCondK!{T<:Real}(node::Leaf, ks::Dict{SPNNode, Int},
   data::AbstractArray{T}, idx::Int, assign::Assignments, G0::ConjugatePostDistribution)

   assign!(assign, idx, node)
   add_data!(node.dist, data)
   increment!(assign, node)

end

"Extend SPN"
function extend!(node::Node, assign::Assignments; depth = 1, cutoff = 2)

   d = depth
   if Base.length(node.children) == 1
      d + 1
   end

   for child in node.children
      extend!(child, assign, depth = d, cutoff = cutoff)
   end
end

"Extend SPN on Leaf"
function extend!(node::Leaf, assign::Assignments; depth = 1, cutoff = 2)
   if depth <= cutoff
      # extend SPN
      p = get(node.parent)

      # compute scope
      scope = Int[]
      for (di, datum) in enumerate(assign.Z)

         if sum(datum .== node) > 0
            push!(scope, di)
         end

      end

      n = isa(p, SumNode) ? ProductNode(0, scope = scope) : SumNode(0, scope = scope)

      add!(p, n)
      increment!(assign, n, i = assign(node))
      add!(n, node)
      remove!(p, findfirst(p.children .== node))
   end
end

"Mirror Leaf node"
function mirror!(node::Leaf, assign::Assignments, X::Array, G0::ConjugatePostDistribution; mirrored = false)

	scope = Int[]

	for (zi, z) in enumerate(assign.Z)
		if node in z
			push!(scope, zi)
		end
	end

	if !mirrored
		d = BNP.add_data(G0, X[node.scope, scope])
	else
		d = BNP.add_data(G0, X[scope, node.scope])
	end

	node.dist = d

end

"Visualize SPN"
function draw(root::Node; adjMatrix = Vector{Int}[], labels = Vector{AbstractString}(0), level = 1)
   thislevel = copy(level)
   adjList = Int[]

   if isa(root, SumNode)
      push!(labels, "+")
   else
      push!(labels, "*")
   end

   for (ci, child) in enumerate(root.children)
      push!(adjList, level + 1)
      level = draw(child, adjMatrix = adjMatrix, labels = labels, level = level + 1)
   end

   if thislevel == 1
      GraphLayout.layout_tree(adj_list,labels,filename="spn.svg",
            cycles = false, ordering = :optimal, coord = :optimal,
            background = nothing)
   end

   level

end

function draw(root::Leaf; adjMatrix = Vector{Int}[], labels = Vector{AbstractString}(0), level = 1)
   push!(labels, "N")
   level
end
