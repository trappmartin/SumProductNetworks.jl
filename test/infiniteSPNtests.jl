println(" * create initial SPN using learnSPN")

using RDatasets
iris = dataset("datasets", "iris")

X = convert(Array, iris[[:SepalLength, :SepalWidth, :PetalWidth]])'

# data
#X = rand(MultivariateNormal([5.0, 5.0], [1.0 0.0; 0.0 2.0]), 100) # 1
#X = cat(2, X, rand(MultivariateNormal([-2.5, 2.5], [0.5 -0.2; -0.2 1.0]), 100)) # 2
#X = cat(2, X, rand(MultivariateNormal([-2.5, -2.5], [1.0 0.0; 0.0 0.5]), 100)) # 3
#X = cat(2, X, rand(MultivariateNormal([5.0, -5.0], [1.0 0.5; 0.5 0.5]), 100)) # 4

(D, N) = size(X)

println(" * using dataset 'test' with ", N, " observations and ", D, " variables.")

dimMapping = Dict{Int, Int}([convert(Int, d) => convert(Int, d) for d in 1:D])
obsMapping = Dict{Int, Int}([convert(Int, n) => convert(Int, n) for n in 1:N])
assignments = Assignment()

root = SPN.learnSPN(X, dimMapping, obsMapping, assignments)

# draw initial solution
println(" * draw initial SPN")
drawSPN(root, file = "initialSPN.svg")

# transform SPN to regions and partitions
println(" * transform SPN into regions and partitions")
(spn, assign) = transformToRegionPartition(root, assignments, N)

@test size(spn.partitions, 1) == 1
@test size(spn.regions, 1) >= 3

# check if assign object and observation counts are equal
for region in spn.regions
	@test region.N == length(assign.observationRegionAssignments[region])
end

# draw region graph (TODO)
println(" * draw transformed SPN")
drawSPN(spn, file = "transformedSPN.svg")

println(" * run Gibbs sweep on a sample using SPN in regions and partitions representation")

observation = 1

x = X[:,observation]

# evaluate all leaf regions

# 0.) remove observation from SPN
activeRegions = assign.regionAssignments[observation]
activePartitions = assign.partitionAssignments[observation]

# list of regions to remove
regionsToRemove = Vector{Region}(0)

# list of partitions to remove
partitionsToRemove = Vector{Partition}(0)

# remove observation from regions and Distributions
for (region, cNode) in activeRegions

	# decrease popularity
	region.popularity[cNode] -= 1
	region.N -= 1
	delete!(assign.observationRegionAssignments[region], observation)

	if isa(region, LeafRegion)

		# remove from Distribution
		remove_data!(region.nodes[cNode].dist, x[region.nodes[cNode].scope,:])

	elseif isa(region, SumRegion)

		# removal of partition assignments
		region.partitionPopularity[cNode][activePartitions[region]] -= 1
		delete!(assign.observationPartitionAssignments[activePartitions[region]], observation)
	end

	# remove node if the node is now empty
	if region.popularity[cNode] == 0

		delete!(region.popularity, cNode)

		if isa(region, LeafRegion)
			deleteat!(region.nodes, cNode)
		elseif isa(region, SumRegion)
			deleteat!(region.partitionPopularity, cNode)
		end

		# get all observations sharing the same region
		obsR = assign.observationRegionAssignments[region]

		# decrease node index for all
		for obs in obsR
			if assign.regionAssignments[obs][region] > cNode
				assign.regionAssignments[obs][region] -= 1
			end
		end
	end

	# remove region if region is now empty
	if region.N == 0
		push!(regionsToRemove, region)
	end

	# remove partition if is now empty
	if length(assign.observationPartitionAssignments[activePartitions[region]]) == 0
		push!(partitionsToRemove, activePartitions[region])
	end

end

# clean up assignments
assign.regionAssignments[observation] = Dict{Region, Int}()
assign.partitionAssignments[observation] = Dict{Region, Partition}()

# check if we have to remove regions
if length(regionsToRemove) > 0

	for region in regionsToRemove

		# loop over all partitions and remove partitionConnections
		for partition in spn.partitions
			if region in spn.partitionConnections[partition]
				deleteat!(spn.partitionConnections[partition], findfirst(region .== spn.partitionConnections[partition]))
			end
		end

		# remove regionConnections
		delete!(spn.regionConnections, region)

	end

end

# check if we have to remove partitions
if length(partitionsToRemove) > 0

	for partition in partitionsToRemove

		# loop over all regions and remove regionConnections
		for region in spn.regions
			if partition in spn.regionConnections[region]
				deleteat!(spn.regionConnections[region], findfirst(partition .== spn.regionConnections[region]))
			end
		end

		# remove partitionConnections
		delete!(spn.partitionConnections, partition)

	end
end

# 1.) get sample trees in the SPN
c = SPNConfiguration(Vector{Vector{Int}}(size(spn.regions, 1)))
cMax = SPNConfiguration(Vector{Vector{Int}}(size(spn.regions, 1)))

for (ri, region) in enumerate(spn.regions)
	if isa(region, LeafRegion)
		c.c[ri] = [1]
		cMax.c[ri] = [size(region.nodes, 1)] # all nodes
	else
		c.c[ri] = [1, 1]
		cMax.c[ri] = [size(region.partitionPopularity, 1), # all pseudo-nodes
															size(spn.regionConnections[region], 1)]
	end
end

# 2.) iterate over sample trees in the SPN

configs = SPN.findConfigurations(c, cMax, spn)
LLH = Vector{Float64}(length(configs))

for (i, configuration) in enumerate(configs)

	postpred = 0.0

	# get list of regions in sample tree
	sampleTree = SPN.extractSampleTree(configuration, spn)

	for regionId in sampleTree # LOOP
		postpred += SPN.posteriorPredictive(spn.regions[regionId], regionId, sampleTree, configuration, cMax, spn, x)
	end

	LLH[i] = postpred

end

println(" * finished computation of llh values for existing sample trees")
println(" * - finished ", length(LLH), " computations of sample trees")

p = exp(LLH - maximum(LLH))

println(" * - p(x, T | Θ) = ", p)

# 2.) roll the dice...

k = BNP.rand_indices(p)

println("new config: ", k)

# 3.) add sample to new sample tree
config = configs[k]
sampleTree = SPN.extractSampleTree(config, spn)

# add sample to existing structure
for regionId in sampleTree

	region = spn.regions[regionId]
	c = config.c[regionId]

	# increase popularity
	region.popularity[c[1]] += 1
	region.N += 1
	push!(assign.observationRegionAssignments[region], observation)

	if isa(region, LeafRegion)

		# remove from Distribution
		add_data!(region.nodes[c[1]].dist, x[region.nodes[c[1]].scope,:])

	elseif isa(region, SumRegion)

		# removal of partition assignments
		region.partitionPopularity[c[1]][spn.partitions[c[2]]] += 1
		push!(assign.observationPartitionAssignments[spn.partitions[c[2]]], observation)
	end

end

# add additional structure if necessary
for regionId in sampleTree
	if haskey(config.newPartitions, regionId) > 0

		region = spn.regions[regionId]
		c = config.c[regionId]

		for newPartition in config.newPartitions[regionId]

			newRID = size(spn.partitions, 1) + 1
			push!(spn.partitions, newPartition)
			spn.partitionConnections[newPartition] = Vector{Region}(0)

			if region.scope == newPartition.scope

				# connect as the new partition is a child of the current region
				push!(spn.regionConnections[region], newPartition)

				# add popularity count
				region.partitionPopularity[c[1]][newPartition] = 1

			# loop over all partitions and remove partitionConnections
			for partition in spn.partitions
				if region in spn.partitionConnections[partition]
					deleteat!(spn.partitionConnections[partition], findfirst(region .== spn.partitionConnections[partition]))
				end
			end

			# remove regionConnections
			delete!(spn.regionConnections, region)

		end

		println("todo")
	end

	if haskey(config.newRegions, regionId) > 0
		println("todo")
	end

end
