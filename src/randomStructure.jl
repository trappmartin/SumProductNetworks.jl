export randomSPN, randomStructure!

function randomSPN(X::AbstractArray; maxDepth = Inf, minChildren = 1, maxChildren = 10, allowScopeOverlap = false)

  (N, D) = size(X)

  observations = Vector{Int}[]
  dimensions = Vector{Int}[]

  # temp SPN structure
  nodeDepths = [0]
  modes = [:sum]
  ids = [1]
  usedids = []
  cids = Dict{Int, Vector}()
  weights = Dict{Int, Vector}()
  scopes = Dict{Int, Vector}()

  nodes = SPNNode[]

  # push data
  push!(observations, collect(1:N))
  push!(dimensions, collect(1:D))

  while !isempty(observations)

    nodeDepth = pop!(nodeDepths)
    mode = pop!(modes)
    id = pop!(ids)
    obs = sort(pop!(observations))
    dims = sort(pop!(dimensions))

    push!(usedids, id)

    isuniv = length(dims) == 1

    if mode == :sum

      # if depth has been reached, push back
      if nodeDepth >= maxDepth

        cid = Int[]
        w = [1.0]

        ccid = maximum(usedids)
        push!(cid, ccid + 1)
        push!(ids, ccid + 1)
        push!(observations, obs)
        push!(dimensions, dims)
        push!(modes, :product)
        push!(nodeDepths, nodeDepth + 1)
      else

        ass = rand(minChildren:maxChildren, length(obs))

        # number of child nodes
        uidx = unique(ass)

        assignments = Int[findfirst(uidx .== i) for i in ass]

        # compute cluster weights
        w = Float64[sum(assignments .== i) / convert(Float64, N) for i in sort(uidx)]
        numchildren = length(w)

        cid = Int[]
        for c in 1:numchildren
          ccid = maximum(usedids)
          push!(cid, ccid + c)
          push!(ids, ccid + c)
          push!(observations, obs[findall(assignments .== c)])
          push!(dimensions, dims)
          push!(modes, :product)
          push!(nodeDepths, nodeDepth + 1)
        end

      end

      weights[id] = w
      cids[id] = cid
      scopes[id] = dims

    elseif mode == :product

      # if univariate, then push back as Leaf
      if isuniv
        push!(ids, id)
        push!(observations, obs)
        push!(dimensions, dims)
        push!(modes, :leaf)
        push!(nodeDepths, nodeDepth + 1)
        continue
      end

      # if depth has been reached, push back
      if nodeDepth >= maxDepth
        cid = Int[]
        ccid = maximum(usedids)
        for (c, d) in enumerate(dims)
          push!(cid, ccid + c)
          push!(ids, ccid + c)
          push!(observations, obs)
          push!(dimensions, [d])
          push!(modes, :leaf)
          push!(nodeDepths, nodeDepth + 1)
        end

        cids[id] = cid

      else

        if !allowScopeOverlap
          assignments = rand(Bool, length(dims))
          p0 = dims[assignments]
          p1 = setdiff(dims, p0)
          p2 = []
        else
          k = rand(minChildren:maxChildren)
          p0 = rand(dims, k)
          p1 = rand(dims, k)
          p2 = setdiff(dims, union(p0, p1))
        end

        if isempty(p1)

          cid = Int[]
          ccid = maximum(usedids)
          for (c, d) in enumerate(p0)
            push!(cid, ccid + c)
            push!(ids, ccid + c)
            push!(observations, obs)
            push!(dimensions, [d])
            push!(modes, :leaf)
            push!(nodeDepths, nodeDepth + 1)
          end

          cids[id] = cid
        else

          cid = Int[]
          ccid = maximum(usedids)
          push!(cid, ccid + 1)
          push!(ids, ccid + 1)
          push!(observations, obs)
          push!(dimensions, p0)
          push!(modes, :sum)
          push!(nodeDepths, nodeDepth + 1)

          push!(cid, ccid + 2)
          push!(ids, ccid + 2)
          push!(observations, obs)
          push!(dimensions, p1)
          push!(modes, :sum)
          push!(nodeDepths, nodeDepth + 1)

          if !isempty(p2)
            push!(cid, ccid + 3)
            push!(ids, ccid + 3)
            push!(observations, obs)
            push!(dimensions, p2)
            push!(modes, :sum)
            push!(nodeDepths, nodeDepth + 1)
          end

          cids[id] = cid
        end
      end

      scopes[id] = dims

    elseif mode == :leaf
      node = fitLeafDistribution(X, id, dims[1], obs)
      push!(nodes, node)
    else
      throw(ErrorException("Unknown mode: $mode"))
    end
  end

  # construct SPN
  while !isempty(cids)
    for id in sort(collect(keys(cids)), rev = true)

      # check if all chidren exist
      ncids = Int[n.id for n in nodes]
      if all(Bool[ccid in ncids for ccid in cids[id]])

        if haskey(weights, id) # sum
          S = SumNode(id, scope = scopes[id])
          w = weights[id]
          for (i, ccid) in enumerate(cids[id])
            add!(S, nodes[findfirst(ncids .== ccid)], w[i])
          end
          push!(nodes, S)
          delete!(cids, id)
          delete!(weights, id)

        else # product
          P = ProductNode(id, scope = scopes[id])

          for (i, ccid) in enumerate(cids[id])
            add!(P, nodes[findfirst(ncids .== ccid)])
          end
          push!(nodes, P)
          delete!(cids, id)
          delete!(weights, id)
        end
      end

    end
  end

  ncids = Int[n.id for n in nodes]
  return nodes[findfirst(ncids .== 1)]
end

function randomStructure!(spn::SumLayer, values::Vector{Int}, D::Int; mixtureSizes = 6, maxDepth = -1, randomSeed = 0)

  productSizes = 2

  # set scope for root sum node
  nodeScopes = Dict{Int, Any}(spn.ids[1] => collect(1:D))

  spn.childIds = zeros(Int, mixtureSizes, 1)

  # construct internal layers
  layerType = :product
  lastLayer = spn
  for depth in 1:maxDepth

    (C, Ch) = size(lastLayer)

    lastID = maximum(lastLayer.ids)
    if layerType == :product

      # construct layer
      ids = Vector{Int}()

      for id in lastLayer.ids
        scope = nodeScopes[id]

        @assert length(scope) > 1

        # construct children
        for child in 1:mixtureSizes

          # draw partition: s.t. sum_i v_{j,i} > 0
          partition = rand(1:productSizes, length(scope))
          while(!all([sum(partition .== c) for c in 1:productSizes] .> 0))
            partition = rand(1:productSizes, length(scope))
          end

          push!(ids, lastID+1)
          nodeScopes[lastID+1] = [scope[partition .== c] for c in 1:productSizes]
          lastID += 1
        end

        delete!(nodeScopes, id)
      end

      layer = ProductLayer(ids, zeros(Int, productSizes, length(ids)), SPNLayer[], lastLayer)
      push!(lastLayer.children, layer)
      lastLayer.childIds = reshape(layer.ids, Ch, C)
      lastLayer = layer

    elseif layerType == :sum

      # construct layer
      ids = Vector{Int}()

      for id in lastLayer.ids
        subscopes = nodeScopes[id] # assuming product layer

        for subscope in subscopes
          push!(ids, lastID+1)
          nodeScopes[lastID+1] = subscope
          lastID += 1
        end
        delete!(nodeScopes, id)
      end

      weights = rand(Dirichlet([1. / mixtureSizes for j in 1:mixtureSizes]), length(ids))
      layer = SumLayer(ids, zeros(Int, mixtureSizes, length(ids)), weights, SPNLayer[], lastLayer)
      push!(lastLayer.children, layer)
      lastLayer.childIds = reshape(layer.ids, Ch, C)
      lastLayer = layer
    end

    # switch types
    layerType = layerType == :sum ? :product : :sum
  end

  # construct multinomial distributions
  if (layerType == :product) & any(map(x -> length(nodeScopes[x]) > 1, keys(nodeScopes)))

    mvIDs = collect(filter(x -> length(nodeScopes[x]) > 1, keys(nodeScopes)))
    @info("Need to construct product layer for: ", mvIDs)

    uvIDs = setdiff(lastLayer.ids, mvIDs)
    @info("Dont need to construct product layer for: ", uvIDs)


  end

end
