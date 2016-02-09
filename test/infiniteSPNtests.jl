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

# 0.) remove observation from SPN
@time SPN.removeObservation!(observation, x, spn, assign)

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

configs = SPN.findConfigurations(c, cMax, spn)

# 2.) iterate over sample trees in the SPN
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
p = p ./ sum(p)

println(" * - p(x | T , W, Θ) = ", p)

# 2.) roll the dice...
k = BNP.rand_indices(p)
k = length(p)
println("new config: ", k)

# 3.) add sample to new sample tree
config = configs[k]
sampleTree = SPN.extractSampleTree(config, spn)

# add sample to existing structure
for regionId in sampleTree

	region = spn.regions[regionId]
	c = config.c[regionId]

	# check if we are out of the range (new node)
	if c[1] > cMax.c[regionId][1]

		# new node
		region.popularity[c[1]] = 1

		if isa(region, LeafRegion)

			SS = 0.0
			C = 0.0

			for partition in spn.partitions # LOOP
				if region in spn.partitionConnections[partition]
					# check if the partition is selected by any region in the tree
					for r2Id in sampleTree # LOOP
						if partition in spn.regionConnections[region]
							# partition is connected to region r2
							# is it selected?
							if (configuration.c[r2Id][2] == findfirst(spn.regionConnections[region], partition))
								# get all observations that are inside that partition and region r2
								for obs in 1:N # LOOP
									if (region, partition) in assign.partitionAssignments[obs]
										SS += X[region.scope, obs]
										C += 1
									end
								end
							end
						end
					end
				end
			end

			if C == 0
				SS = 0.0
				C = 1.0
			end

			leaf = UnivariateNode{ConjugatePostDistribution}(NormalGamma(μ = SS / C), scope = region.scope)
			push!(region.nodes, leaf)

		elseif isa(region, SumRegion)

			push!(region.partitionPopularity, Dict{Partition, Int}())

		end

	end

	partitionAdded = false

	# check if we have a new partition selected
	if isa(region, SumRegion)
		if c[2] > cMax.c[regionId][2]

			# new partition
			println("new partitions")
			for newpartition in config.newPartitions[regionId]

				println("adding new partition with scope ", newpartition.scope)

				partitionAdded = true

				# create new partition
				pid = size(spn.partitions, 1) + 1
				push!(spn.partitions, newpartition)

				# check if this partition should be connected to the region
				if region.scope == newpartition.scope
					region.partitionPopularity[c[1]][newpartition] = 1

					push!(spn.regionConnections[region], newpartition)
					assign.partitionAssignments[observation][region] = newpartition
				else # find region that should connect to the region

					for sregion in spn.regions
						if sregion.scope == newpartition.scope

							# connect partition to region (assume this is a new region, -> number of children = 0)
							push!(sregion.partitionPopularity, Dict{Partition, Int}())
							@assert size(sregion.partitionPopularity, 1) == 1
							sregion.partitionPopularity[1][newpartition] = 1

							push!(spn.regionConnections[sregion], newpartition)
							assign.partitionAssignments[observation][sregion] = newpartition
						end
					end

				end

				assign.observationPartitionAssignments[newpartition] = Set{Int}(observation)
				spn.partitionConnections[newpartition] = Vector{Region}()

				scopes = collect(keys(newpartition.indexFunction))
				parts = collect(values(newpartition.indexFunction))
				partIds = unique(parts)

				for partId in partIds
					idx = find(partId .== parts)

					subscope = Set(scopes[idx])

					splitFound = false
					for sregion in spn.regions
						if sregion.scope == subscope
							splitFound = true

							# connect partition to region
							push!(spn.partitionConnections[newpartition], sregion)
						end
					end

					if splitFound
						continue
					else

						# check new regions
						newregions = config.newRegions[regionId]

						for newregion in newregions
							if newregion.scope == subscope

								# add new region!
								println("adding new region with scope ", newregion.scope)
								push!(spn.regions, newregion)
								spn.regionConnections[newregion] = Vector{Partition}(0)
								assign.observationRegionAssignments[newregion] = Set{Int}(observation)
							end
						end

					end

				end

			end

		end

	end

	# increase popularity
	region.popularity[c[1]] += 1
	region.N += 1
	push!(assign.observationRegionAssignments[region], observation)

	if isa(region, LeafRegion)

		# add to Distribution
		add_data!(region.nodes[c[1]].dist, x[region.nodes[c[1]].scope,:])

	elseif isa(region, SumRegion) & !partitionAdded

		# add to partition assignments
		region.partitionPopularity[c[1]][spn.partitions[c[2]]] += 1
		push!(assign.observationPartitionAssignments[spn.partitions[c[2]]], observation)
	end

end
#
# # add additional structure if necessary
# for regionId in sampleTree
# 	if haskey(config.newPartitions, regionId) > 0
#
# 		region = spn.regions[regionId]
# 		c = config.c[regionId]
#
# 		for newPartition in config.newPartitions[regionId]
#
# 			newRID = size(spn.partitions, 1) + 1
# 			push!(spn.partitions, newPartition)
# 			spn.partitionConnections[newPartition] = Vector{Region}(0)
#
# 			if region.scope == newPartition.scope
#
# 				# connect as the new partition is a child of the current region
# 				push!(spn.regionConnections[region], newPartition)
#
# 				# add popularity count
# 				region.partitionPopularity[c[1]][newPartition] = 1
# 			end
#
# 			# remove regionConnections
# 			delete!(spn.regionConnections, region)
#
# 		end
#
# 		println("todo")
# 	end
#
# 	if haskey(config.newRegions, regionId) > 0
# 		println("todo")
# 	end
#
# end
