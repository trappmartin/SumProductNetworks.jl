export imageStructure!

function imageStructure!(spn::SumNode, C::Int, D::Int, G::Int, K::Int; parts = 200, mixtures = 2, window = 3)

  @assert D == (G^2 * K) "dimensions don't match"
  @assert window <= G "window size is larger than image"

  # create scopes of leaves
  posx = 1
  posy = 1
  scopes = Vector{Vector{Int}}()
  while (posy + window) <= G
    while (posx + window) <= G
      scope = Int[]
      for k in 1:K
        for w2 in posx:window+posx
          for w1 in posy:window+posy
            push!(scope, sub2ind3(K, G, k, w2, w1))
          end
        end
      end
      push!(scopes, scope)
      posx += 1
    end
    posx = 1
    posy += 1
  end

  numberOfLeaves = length(scopes)

  id = spn.id
  id += 1

  # create classes
  for c in 1:C

    Pc = ProductNode(id)
    id += 1
    add!(Pc, IndicatorNode(id, c, D + 1))
    id += 1

    # create parts
    for p in 1:parts

      Sp = SumNode(id)
      id += 1
      # create mixtures
      for m in 1:mixtures

        Sm = SumNode(id)
        id += 1
        # create locations
        for l in 1:numberOfLeaves
          L = MultivariateFeatureNode(id, scopes[l])
          id += 1
          add!(Sm, L)
        end
        add!(Sp, Sm)
      end
      add!(Pc, Sp)
    end
    add!(spn, Pc)
  end
end

function imageStructure!(spn::SumLayer, C::Int, D::Int, G::Int, K::Int; parts = 200, mixtures = 2, window = 3)

  @assert D == (G^2 * K) "dimensions don't match"
  @assert window <= G "window size is larger than image"

  # create scopes of leaves
  posx = 1
  posy = 1
  scopes = Vector{Vector{Bool}}()
  while (posy + window) <= G
    while (posx + window) <= G
      mask = zeros(Bool, D)
      for k in 1:K
        for w2 in posx:window+posx
          for w1 in posy:window+posy
            mask[sub2ind3(K, G, k, w2, w1)] = true
          end
        end
      end
      push!(scopes, mask)
      posx += 1
    end
    posx = 1
    posy += 1
  end

  scopes = reduce(hcat, scopes)
  locations = size(scopes, 2)

  # construct structure represented as a layer path

  maxId = maximum(spn.ids)

  # add classes
  classesLayer = ProductCLayer(collect(maxId + 1:maxId + C), Array{Int,2}(undef, 0, 0), collect(1:C), SPNLayer[], spn)
  push!(spn.children, classesLayer)
  spn.childIds = reshape(classesLayer.ids, length(classesLayer.ids), 1)
  spn.logweights = log.(rand(Dirichlet([1. / C for j in 1:C]), 1))

  # add parts
  maxId = maximum(classesLayer.ids)
  P = C * parts
  logw = log.(rand(Dirichlet([1. / mixtures for j in 1:mixtures]), P))
  partsLayer = SumLayer(collect(maxId + 1:maxId + P),  Array{Int,2}(undef, 0, 0), logw, SPNLayer[], classesLayer)
  push!(classesLayer.children, partsLayer)
  classesLayer.childIds = reshape(partsLayer.ids, parts, C)

  # add mixtures
  maxId = maximum(partsLayer.ids)
  M = C * parts * mixtures
  logw = log.(rand(Dirichlet([1. /locations for j in 1:locations]), M))
  mixturesLayer = SumLayer(collect(maxId + 1:maxId + M),  Array{Int,2}(undef, 0, 0), logw, SPNLayer[], partsLayer)
  push!(partsLayer.children, mixturesLayer)
  partsLayer.childIds = reshape(mixturesLayer.ids, mixtures, P)

  # number of nodes in the MultivariateFeatureLayer
  maxId = maximum(mixturesLayer.ids)
  L = C * parts * mixtures * locations
  filterLayer = MultivariateFeatureLayer(collect(maxId + 1:maxId + L), zeros(Float32, D, L), repeat(scopes, 1, M), mixturesLayer)
  push!(mixturesLayer.children, filterLayer)
  mixturesLayer.childIds = reshape(filterLayer.ids, locations, M)

end
