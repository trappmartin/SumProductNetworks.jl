function Base.show(io::IO, m::SumRegion)
  print(io, "SumRegion => [scope: $(m.scope), #children: $(size(m.partitionPopularity, 1)), #observations: $(m.N)]")
end

function Base.show(io::IO, m::LeafRegion)
  print(io, "LeafRegion => [scope: $(m.scope), #children: $(size(m.nodes, 1)), #observations: $(m.N)]")
end

function Base.show(io::IO, m::Partition)
  print(io, "Partition => [scope: $(m.scope), indexFunction: $(m.indexFunction)]")
end

function Base.show(io::IO, m::SPNStructure)
  print(io, "SPNStructure => [regions: $(m.regions), partitions: $(m.partitions)]")
end
