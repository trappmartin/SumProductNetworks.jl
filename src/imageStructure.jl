export imageStructure!

function imageStructure!(spn::SumNode, C::Int, D::Int, G::Int, K::Int; parts = 200, mixtures = 2, window = 3)

  @assert D == (G^2 * K) "dimensions don't match"
  @assert window <= G "window size is larger than image"

  # create scopes of leaves
  pos = 1
  scopes = Vector{Vector{Int}}(0)
  while (pos + window) <= G
    scope = Int[]
    for k in 1:K
      for w2 in pos:window+pos
        for w1 in pos:window+pos
          push!(scope, sub2ind((K, G, G), k, w2, w1))
        end
      end
    end
    push!(scopes, scope)
    pos += 1
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
  pos = 1
  scopes = Vector{Vector{Int}}(0)
  while (pos + window) <= G
    scope = Int[]
    for k in 1:K
      for w2 in pos:window+pos
        for w1 in pos:window+pos
          push!(scope, sub2ind((K, G, G), k, w2, w1))
        end
      end
    end
    push!(scopes, scope)
    pos += 1
  end

  numberOfLeaves = length(scopes)

  id = 1

  SPN = SumNode(id)
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
    add!(SPN, Pc)
  end

  return SPN
end
