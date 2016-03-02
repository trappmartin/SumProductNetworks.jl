function Base.show(io::IO, m::SumRegion)
  print(io, "SumRegion => [scope: $(m.scope), #children: $(size(m.partitionPopularity, 1)), #observations: $(m.N)]")
end

function Base.show(io::IO, m::LeafRegion)
  print(io, "LeafRegion => [scope: $(m.scope), #nodes: $(size(m.nodes, 1)), #observations: $(m.N)]")
end

function Base.show(io::IO, m::Partition)
  print(io, "Partition => [scope: $(m.scope), indexFunction: $(m.indexFunction)]")
end

function Base.show(io::IO, m::SPNStructure)
  println(io, "SPNStructure => [ regions: $(m.regions), ")
	println(io, "                  partitions: $(m.partitions), ")
	println(io, "                  regionConnections: $(m.regionConnections), ")
	println(io, "                  partitionConnections: $(m.partitionConnections) ]")
end

function Base.show(io::IO, m::LeafRegionResultObject)
	println(io, "LeafRegionResult => [ P(x | Θ)P(Θ): $(m.postpredNodes), ")
	println(io, "                      configuration: $(m.configNodes), ")
	println(io, "                     ∫P(x | Θ*)H(Θ* | Y) dΘ*: $(m.postpredInformedNodes), ")
	println(io, "                      configuration: $(m.configInformedNodes), ")
	println(io, "                     ∫P(x | Θ*)H(Θ*) dΘ*: $(m.postpredUninformedNode) ]")
end

function Base.show(io::IO, m::SumRegionResultObject)
	println(io, "SumRegionResult => [ P(x | Θ)P(cP | cP -i)P(cN | cN -i): $(m.postpred), ")
	println(io, "                     P(x | Θ)P(cP*)P(cN | cN -i): $(m.postpredNewPartitions), ")
	println(io, "                     partitions: $(m.configNewPartitionPartitions), ")
	println(io, "                     regions: $(m.configNewPartitionRegions), ")
	println(io, "                     P(x | Θ)P(cP | cP -i)P(cN*): $(m.postpredNewNode), ")
	println(io, "                     P(x | Θ)P(cP*)P(cN*): $(m.postpredNewNodeNewPartition), ")
	println(io, "                     partitions: $(m.configNewNodeNewPartitionPartitions), ")
	println(io, "                     regions: $(m.configNewNodeNewPartitionRegions) ]")
end
