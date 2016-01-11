"Assignment Data Object"
type Assignments

	# datum to node assignments
	Z::Vector{Vector{SPNNode}}

   # node sum of squares
   ZZ::Dict{SPNNode, Vector{Float64}}

	# bucket sizes
	S::Dict{SPNNode, Int}

	Assignments(N::Int) = new( Base.map(i -> Vector{SPNNode}(0), collect(1:N)),  Dict{SPNNode, Vector{Float64}}(), Dict{SPNNode, Int}())

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
function assign!(p::Assignments, id::Int, n::SPNNode, x::Vector{Float64})

	if isdefined(p.Z[id])

      if !(n in p.Z[id])
         push!(p.Z[id], n)

         zz = get(p.ZZ, n, zeros(size(x)))
         p.ZZ[n] = zz + x
      end
	else
		p.Z[id] = [n]
	end

end

"Withdraw node assignment"
function withdraw!(p::Assignments, id::Int, n::SPNNode, x::Vector{Float64})

	if isdefined(p.Z[id])
      idx = findfirst(p.Z[id] .== n)
      p.Z[id] = p.Z[id][[1:idx-1; idx+1:end]]

      zz = get(p.ZZ, n, zeros(size(x)))
      p.ZZ[n] = zz - x
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

   G = GaussianWishart(G0.mu0[root.scope], G0.kappa0, float(Base.length(root.scope)), G0.Sigma0[root.scope, root.scope])

   p = ones(size(data, 2), Base.length(root.children) + 1) * -Inf

   for (ci, c) in enumerate(root.children)
      try
         p[:,ci] = llhvals[c] + log(assign(c) / (assign(root) + α - 1))
      catch
         println("ERROR!")
         println("assign(c): ", assign(c))
         println("c*: ", pointer_from_objref(c))
         println("node*: ", pointer_from_objref(root))
         println("# children: ", Base.length(root.children))

         for k in keys(assign.S)
            println("assign(k): ", assign(k))
            println("k*: ", pointer_from_objref(k))
         end

      end

   end

   p[:,end] = logpred(G, data[root.scope,:]) + log( α / (assign(root) + α - 1) )

   # get k
   k = 0

   maxp = maximum(p, 2)
   p = exp(p .- maxp)

   k = BNP.rand_indices(p)

   # get node llh
   p = sum(p, 2)
   p = log(p) .+ maxp

   @assert k != nothing

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

   if Base.length(root.children) == 0
      println(assign(root))
   end

   _llh = [llhvals[c] for c in root.children]
   _llh = reduce(vcat, _llh)

   @assert !isnan(_llh[1])

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

   @assert !isnan(llh[1])

 return (llh, -1)
end

"Spawn new child into SPN"
function spawnChild!(node::Node, G0::ConjugatePostDistribution)

   G = GaussianWishart(G0.mu0[node.scope], G0.kappa0, float(Base.length(node.scope)), G0.Sigma0[node.scope, node.scope])
   add!(node, MultivariateNode{ConjugatePostDistribution}(G, copy(node.scope)))
end

"Kill child from SPN"
function killChild!(node::SPNNode, assign::Assignments)

   p = get(node.parent)

   remove!(p, findfirst(p.children .== node))

   # apply to parent if necessary
   #if assign(p) == 0
   #   killChild!(p, assign)
   #end

end

"Recurse on Nodes, add node if necessary."
function recurseCondK!{T<:Real}(node::ProductNode, ks::Dict{SPNNode, Int},
   data::AbstractArray{T}, idx::Int, assign::Assignments, G0::ConjugatePostDistribution; mirror = false)

   # check if we need to extend the number of children
   if ks[node] > Base.length(node.children)
      spawnChild!(node, G0)
   end

	if mirror
      oldNodes = assign[idx]
		recurseCondK!(node.children[ks[node]], ks, data, idx, assign, G0, mirror = mirror)
      increment!(assign, node)
	else
		for child in node.children
			recurseCondK!(child, ks, data, idx, assign, G0, mirror = mirror)
         increment!(assign, node)
		end
	end

   assign!(assign, idx, node)
end

"Recurse on Nodes, add node if necessary."
function recurseCondK!{T<:Real}(node::SumNode, ks::Dict{SPNNode, Int},
   data::AbstractArray{T}, idx::Int, assign::Assignments, G0::ConjugatePostDistribution; mirror = false)

   # check if we need to extend the number of children
   if ks[node] > Base.length(node.children)
      spawnChild!(node, G0)
   end

	if !mirror
      oldNodes = assign[idx]
		recurseCondK!(node.children[ks[node]], ks, data, idx, assign, G0, mirror = mirror)
      increment!(assign, node)
	else
		for child in node.children
			recurseCondK!(child, ks, data, idx, assign, G0, mirror = mirror)
         increment!(assign, node)
		end
	end

   assign!(assign, idx, node)
end

"Recurse on Leafs"
function recurseCondK!{T<:Real}(node::Leaf, ks::Dict{SPNNode, Int},
   data::AbstractArray{T}, idx::Int, assign::Assignments, G0::ConjugatePostDistribution; mirror = false)

   assign!(assign, idx, node)
   add_data!(node.dist, data[node.scope,:])
   increment!(assign, node)

end

"Extend SPN"
function extend!(node::Node, assign::Assignments; depth = 1, cutoff = 2, tomirror = true)

   d = depth
   if Base.length(node.children) == 1
      d = depth + 1
   end

   childrens = SPNNode[]

   for child in node.children
      push!(childrens, child)
   end

   for child in childrens
      extend!(child, assign, depth = d, cutoff = cutoff, tomirror = tomirror)
   end
end

"Extend SPN on Leaf"
function extend!(node::Leaf, assign::Assignments; depth = 1, cutoff = 2, tomirror = true)

   if depth <= cutoff

      # extend SPN
      p = get(node.parent)

      n = isa(p, SumNode) ? ProductNode(0, scope = node.scope) : SumNode(0, scope = node.scope)

      if isa(n, ProductNode) & tomirror
         return
      elseif isa(n, SumNode) & !tomirror
         return
      end

      add!(p, n)
      increment!(assign, n, i = assign(node))
      add!(n, node)

      for (di, datum) in enumerate(assign.Z)
         if node in datum
            assign!(assign, di, n)
         end
      end

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

   G = GaussianWishart(G0.mu0[scope], G0.kappa0, float(Base.length(scope)), G0.Sigma0[scope, scope])

	if !mirrored
		d = BNP.add_data(G, X[node.scope, scope]')
	else
		d = BNP.add_data(G, X[scope, node.scope])
	end

   sort!(scope)

   node.scope = scope
	node.dist = d

end

"Mirror Internal node"
function mirror!(node::Node, assign::Assignments, X::Array, G0::ConjugatePostDistribution; mirrored = false)

	function recurseInternal(node::Node)
		mapreduce(c -> c.scope, vcat, node.children)
	end

	function recurseInternal(node::Leaf)
		node.scope
	end

	scope = unique(recurseInternal(node))
   sort!(scope)

   node.scope = scope

end

"Update Weights on Sum Node"
function update_weights(root::SumNode, assign::Assignments; α = 1.0)

   root.weights = vec( [assign(c) / (assign(root) + α - 1) for c in root.children] )

   # update children
   for c in root.children
      update_weights(c, assign, α = α)
   end

end

"Update Weights"
function update_weights(root::ProductNode, assign::Assignments; α = 1.0)

   # update children
   for c in root.children
      update_weights(c, assign, α = α)
   end

end

"Update Weights"
function update_weights(root::Leaf, assign::Assignments; α = 1.0)
   # nothing to do
end

"Visualize SPN"
function draw(spn::SumNode; file="spn.svg", showBucket = false, assign = nothing)

   nodes = order(spn)

   labels = AbstractString[]
   A = zeros(Base.length(nodes), Base.length(nodes))

   reverse!(nodes)

   for i in collect(1:Base.length(nodes))
      if isa(nodes[i], Node)
         for j in collect(1:Base.length(nodes))
            if nodes[j] in nodes[i].children
               A[i, j] = 1
            end
         end

         if nodes[i] == spn
            if showBucket
               push!(labels, "R+ ($(assign(nodes[i])))")
            else
               push!(labels, "R+")
            end
         else
            if isa(nodes[i], SumNode)
               if showBucket
                  push!(labels, "+ ($(assign(nodes[i])))")
               else
                  push!(labels, "+")
               end
            else
               if showBucket
                  push!(labels, "x ($(assign(nodes[i])))")
               else
                  push!(labels, "x")
               end
            end
         end
      else
         if showBucket
            push!(labels, "O ($(assign(nodes[i])))")
         else
            push!(labels, "O")
         end
      end
   end

   labSize = showBucket ? 10.0 : 20.0

   loc_x, loc_y = layout_spring_adj(A)
   draw_layout_adj(A, loc_x, loc_y, labels=labels, labelsize=labSize, filename=file)

   adj_list = Vector{Int}[]
   for i in 1:size(A,1)
       new_list = Int[]
       for j in 1:size(A,2)
           if A[i,j] != zero(eltype(A))
               push!(new_list,j)
           end
       end
       push!(adj_list, new_list)
   end

   #layout_tree(adj_list, labels, cycles=false, filename="tree.svg")

end

"Run a single Gibbs iteration"
function gibbs_iteration!(root::Node, assign::Assignments,
   G0::ConjugatePostDistribution, G0Mirror::ConjugatePostDistribution,
   X::Array; internalIters = 100)

	(D, N) = size(X)

   # for testing
   toporder = order(root)

   for it in collect(1:internalIters)

   	for id in randperm(N)

   		x = X[:, id]
   		kdists = assign[id]
   		toremove = Vector{SPNNode}(0)

         # remove data point and withdraw nodes from datum
   		for dist in kdists

            if assign(dist) == 0
               println("woot")
               #draw(root, file="debug2.svg", showBucket = true, assign = assign)
            end

            if isa(dist, ProductNode)
               decrement!(assign, dist, i = Base.length(dist.children))
            else
               decrement!(assign, dist)
            end
            withdraw!(assign, id, dist)

            if isa(dist, Leaf)
               remove_data!(dist.dist, x[dist.scope,:])
            end

            if assign(dist) < 0
               println(typeof(dist))
               println(assign(dist))

               println("all: ", Base.length(kdists), " uniques: ", Base.length(unique(kdists)))
            end
            #
            @assert assign(dist) > -1

   			if assign(dist) == 0
   				push!(toremove, dist)
   			end
   		end

   		# clean up

         # TODO: Make this "softer"

   		#for node in toremove
   	   #	killChild!(node, assign)
   		#end

         # evaluate all leaf nodes in paralel


         # evaluate SPN on datum (including integration and sampling over latent variables)
   		for node in toporder

   				(llh, newk) = evalWithK(node, x, llhval, assign, G0)

   				llhval[node] = llh
   				kvals[node] = newk

               if isa(node, SumNode)
                  #println(newk)
               end
   		end

   		# assign datum to
   		recurseCondK!(root, kvals, x, id, assign, G0)

   	end

   end

	# extend SPN
	extend!(root, assign)

	# get topological order
	toporder = order(root)

	# for each product node
	for child in root.children

      if assign(child) == 1
         continue
      end

		toporder = order(child)
		assignMirror = Assignments(D)

		# set up assignments
		for node in toporder

			# flip assignment of data to leafs
			for i in node.scope
				assign!(assignMirror, i, node)
			end

			# flip scopes and transpose distributions
			mirror!(node, assign, X, G0Mirror)

			# flip bucket sizes
			if isa(node, Leaf)

				bucket = Int[]

				for (zi, z) in enumerate(assignMirror.Z)
					if node in z
						push!(bucket, zi)
					end
				end

				assignMirror.S[node] = Base.length(bucket)
			else
            assignMirror.S[node] = sum(Base.map(c -> assignMirror(c), node.children))
			end
		end

         for it in collect(1:internalIters)

   		# gibbs loop on mirrored SPN
   		for id in randperm(D)

            # current data point
   			x = X[id, :]'
   			kdists = assignMirror[id]
   			toremove = Vector{SPNNode}(0)

            # remove data point / withdraw node from list of nodes for datum
   			for dist in kdists

               if isa(dist, SumNode)
                  decrement!(assignMirror, dist, i = Base.length(dist.children))
               else
                  decrement!(assignMirror, dist)
               end

               withdraw!(assignMirror, id, dist)

               # this makes only sense for leaf nodes
               if isa(dist, Leaf)
                  remove_data!(dist.dist, x[dist.scope,:])
               end

               if assignMirror(dist) < 0
                  println(typeof(dist))
                  println(assignMirror(dist))
               end
               @assert assignMirror(dist) > -1

               # remove empty nodes
   				if assignMirror(dist) == 0
                  push!(toremove, dist)
   				end
   			end

   			# clean up
   			for node in toremove

               # clean up references in assign structure
               for j in child.scope
                  withdraw!(assign, j, node)
               end

               killChild!(node, assignMirror)
   			end

            # compute topological order and initialise data structures
            toporder = order(child)
   			llhval = Dict{SPNNode, Array{Float64}}()
   			kvals = Dict{SPNNode, Int}()

            # evaluate SPN on datum
   			for node in toporder

   					(llh, newk) = evalWithK(node, x, llhval, assignMirror, G0Mirror, mirror = true)

   					llhval[node] = llh
   					kvals[node] = newk

   			end

   			# assign datum to nodes
   			recurseCondK!(child, kvals, x, id, assignMirror, G0Mirror, mirror = true)

   		end
      end

      toporder = order(child)

      # reassign data points accordingly (houskeeping...)
		for node in toporder

         inc = Base.length(node.scope)

			mirror!(node, assignMirror, X, G0, mirrored = true)

         if isa(node, Leaf)
            bucket = get(node.parent).scope

            for item in bucket
               assign!(assign, item, node)
            end

            if !haskey(assign.S, node)
               increment!(assign, node, i = inc)
            end

         end

		end

	end

   extend!(root, assign, tomirror = false)

end
