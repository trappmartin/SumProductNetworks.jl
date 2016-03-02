type SPNConfiguration

	c::Vector{Vector{Int}}
	newPartitions::Dict{Int, Vector{Partition}}
	newRegions::Dict{Int, Vector{Region}}

	SPNConfiguration(c::Vector{Vector{Int}}; nP = Dict{Int, Vector{Partition}}(), nR = Dict{Int, Vector{Region}}()) = new(c, nP, nR)

end

@doc doc"""
Helper function for merging configurations in place.
""" ->
function merge!(c1::SPNConfiguration, c2::SPNConfiguration)
	for i in 1:length(c1.c)
		for j in 1:length(c1.c[i])
			c1.c[i][j] = maximum([c1.c[i][j], c2.c[i][j]])
		end

		if haskey(c2.newPartitions, i)
			if haskey(c1.newPartitions, i)
				c1.newPartitions[i] = append!(c1.newPartitions[i], c2.newPartitions[i])
			else
				c1.newPartitions[i] = c2.newPartitions[i]
			end
		end

		if haskey(c2.newRegions, i)
			if haskey(c1.newRegions, i)
				c1.newRegions[i] = append!(c1.newRegions[i], c2.newRegions[i])
			else
				c1.newRegions[i] = c2.newRegions[i]
			end
		end
	end

	c1

end

abstract RegionResultObject

type LeafRegionResultObject <: RegionResultObject

	postpredNodes::Vector{Float64}
	configNodes::Vector{Int}
	postpredInformedNodes::Vector{Float64}
	configInformedNodes::Vector{Tuple{Int, Int}}
	postpredUninformedNode::Float64

	LeafRegionResultObject(postpredNodes, configNodes;
		postpredInformedNodes = Vector{Float64}(0), configInformedNodes = Vector{Tuple{Int, Int}}(0)) = new(postpredNodes, configNodes,
		postpredInformedNodes, configInformedNodes, -Inf)

end

type SumRegionResultObject <: RegionResultObject

	postpred::Array{Float64, 2}
	postpredNewPartitions::Vector{Float64}
	configNewPartitionPartitions::Vector{Vector{Partition}}
	configNewPartitionRegions::Vector{Vector{Region}}
	postpredNewNode::Float64
	postpredNewNodeNewPartition::Float64
	configNewNodeNewPartitionPartitions::Vector{Partition}
	configNewNodeNewPartitionRegions::Vector{Region}

	SumRegionResultObject(size11::Int, size12::Int, size2::Int) = new(Array{Float64, 2}(size11, size12),
		Vector{Float64}(size2), Vector{Vector{Partition}}(size2), Vector{Vector{Region}}(size2), -Inf, -Inf,
		Vector{Partition}(0), Vector{Region}(0))

end

function buildPartitionsAndRegions!(region::Region, regionId::Int, spn::SPNStructure)

	newPartitions = Vector{Partition}(0)
	newRegions = Vector{Region}(0)

	# new partition
	p = Partition()

	# check scope
	indexFunction = Dict{Int, Int}()
	scope = collect(region.scope)

	p.scope = region.scope

	if length(scope) == 2
		p.indexFunction = Dict{Int, Int}(scope[1] => 1, scope[2] => 2)
		push!(newPartitions, p)
	else

		pL = length(partitions(scope))
		parts = collect(partitions(scope))[rand(2:pL)]
		p.indexFunction = Dict{Int, Int}()
		p.subscopes = [Set(part) for part in parts]
		push!(newPartitions, p)

		# check if indexFunction splits into new region
		for (pi, part) in enumerate(parts)

			for v in part
				p.indexFunction[v] = pi
			end

			if length(part) == 1
				continue
			end

			partSet = Set(part)

			allSplitsFound = false
			for region in spn.regions
				if region.scope == partSet
					allSplitsFound = true
					# save new connection
				end
			end

			if !allSplitsFound
				# create region
				r = SumRegion()
				r.scope = partSet
				r.partitionPopularity = Vector{Dict{Partition, Int64}}(0)
				r.popularity = Vector{Int}(0)
				r.N = 1

				push!(newRegions, r)
				(nP, nR) = buildPartitionsAndRegions!(r, regionId, spn)

				append!(newRegions, nR)
				append!(newPartitions, nP)

			end
		end
	end

	(newPartitions, newRegions)
end

@doc doc"""
posterior predictive for leaf region.
""" ->
function posteriorPredictive(region::LeafRegion, regionId::Int, initialConfig::SPNConfiguration, cMax::SPNConfiguration,
	spn::SPNStructure, assign::AssignmentRegionGraph, X::AbstractArray, xi::Int; α = 1.0, G0Type = NormalGamma, allowNew = true)

	# get selection
	initialcNode = initialConfig.c[regionId][1]
	cMax = cMax.c[regionId][1]

	if initialcNode == -1
		return LeafRegionResultObject(Vector{Float64}(0), Vector{Int}(0))
	end

	result = LeafRegionResultObject(Vector{Float64}((1 + cMax - initialcNode)), Vector{Int}((1 + cMax - initialcNode)))

	# for all nodes in the region
	for (i, cNode) in enumerate(initialcNode:cMax)
		# get llh values
		llh = logpred(region.nodes[cNode].dist, X[region.scope, xi])[1]

		# p(c_{i, Rg} = j | c_{-i, Rg}, α)
		lc = log(region.popularity[cNode] / (region.N - 1 + α) )

		result.postpredNodes[i] = llh + lc
		result.configNodes[i] = cNode
	end

	if !allowNew
		return result
	end

	# for all possible new nodes with informed prior
	for partition in spn.regionConnectionsBottomUp[region]
		obs1 = assign.observationPartitionAssignments[partition]

		# p(c_{i, Rg} = j | c_{-i, Rg}, α)
		lc = log(α / (region.N - 1 + α) )

		for parentRegion in spn.partitionConnectionsBottomUp[partition]

			parentRegionId = findfirst(parentRegion .== spn.regions)
			partitionId = findfirst(partition .== spn.regionConnections[parentRegion])

			obs = intersect(obs1, assign.observationRegionAssignments[parentRegion])

			# .) compute llh using adjusted mean0
			G0 = BNP.fit(G0Type, X[region.scope, collect(obs)])
			llh = logpred(G0, X[region.scope, xi])[1]

			push!(result.postpredInformedNodes, llh + lc)
			push!(result.configInformedNodes, (parentRegionId, partitionId))
		end
	end

	# posterior predictive for uninformed prior
	G0 = BNP.fit(G0Type, [X[region.scope, xi]]', computeScale = false)
	llh = logpred(G0, X[region.scope, xi])[1]

	# p(c_{i, Rg} = j | c_{-i, Rg}, α)
	lc = log(α / (region.N - 1 + α) )

	result.postpredUninformedNode = llh + lc

	return result

end

@doc doc"""
posterior predictive for leaf region.
""" ->
function posteriorPredictive(region::SumRegion, regionId::Int, initialConfig::SPNConfiguration, cMax::SPNConfiguration,
		spn::SPNStructure, assign::AssignmentRegionGraph, X::AbstractArray, xi::Int; α = 1.0, G0Type = NormalGamma, allowNew = true)

	# get selection
	initialcNode = initialConfig.c[regionId][1]
	cMaxNode = cMax.c[regionId][1]

	if initialcNode == -1
		return SumRegionResultObject(0, 0, 0)
	end

	cMaxPartition = cMax.c[regionId][2]

	# result object
	result = SumRegionResultObject(cMaxNode, cMaxPartition, cMaxNode)

	# compute existing node / partition pairs
	for cNode in 1:cMaxNode
		for cPartition in 1:cMaxPartition

			# p(c_{i, Rg} = j | c_{-i, Rg}, α)
			result.postpred[cNode, cPartition] = log(region.popularity[cNode] / (region.N - 1 + α) )

			# p(c_{i, S} = j | c_{-i, S}, α)
			# check if the node already has a popularity value for this partition otherwise its α
			if haskey(region.partitionPopularity[cNode], spn.regionConnections[region][cPartition])
				result.postpred[cNode, cPartition] += log(region.partitionPopularity[cNode][spn.regionConnections[region][cPartition]] / (region.popularity[cNode] - 1 + α) )
			else
				result.postpred[cNode, cPartition] += log(α / (region.popularity[cNode] - 1 + α) )
			end
		end
	end

	if !allowNew
		return result
	end

	logpNewConnection = log(α / (1 - 1 + α) )

	# compute existing node new partition pairs
	for cNode in 1:cMaxNode

		# p(c_{i, Rg} = j | c_{-i, Rg}, α)
		result.postpredNewPartitions[cNode] = log(region.popularity[cNode] / (region.N - 1 + α) )

		# p(c_{i, S} = j | c_{-i, S}, α)
		result.postpredNewPartitions[cNode] += log(α / (region.popularity[cNode] - 1 + α) )

		# build new partition
		(newPartitions, newRegions) = buildPartitionsAndRegions!(region, regionId, spn)

		result.configNewPartitionPartitions[cNode] = newPartitions
		result.configNewPartitionRegions[cNode] = newRegions

		# additional p(c_{i, S} = j | c_{-i, S}, α)
		if length(newRegions) > 0
			result.postpredNewPartitions[cNode] += logpNewConnection * length(newRegions)
			result.postpredNewPartitions[cNode] += logpNewConnection * (length(newPartitions) - 1)
		end

	end

	# compute new node and existing partition pairs
	# p(c_{i, Rg} = j | c_{-i, Rg}, α)
	result.postpredNewNode = log(α / (region.N - 1 + α) )

	# p(c_{i, S} = j | c_{-i, S}, α)
	# NOTE: this is a sequential process
	# therefore, count for observation in the node is 1.
	result.postpredNewNode += logpNewConnection

	# compute new node and new partition pairs
	# p(c_{i, Rg} = j | c_{-i, Rg}, α)
	result.postpredNewNodeNewPartition = log(α / (region.N - 1 + α) )

	# p(c_{i, S} = j | c_{-i, S}, α)
	# NOTE: this is a sequential process
	# therefore, count for observation in the node is 1.
	result.postpredNewNodeNewPartition += logpNewConnection

	# build new partition
	(newNodeNewPartitions, newNodeNewRegions) = buildPartitionsAndRegions!(region, regionId, spn)

	result.configNewNodeNewPartitionPartitions = newNodeNewPartitions
	result.configNewNodeNewPartitionRegions = newNodeNewRegions

	# additional p(c_{i, S} = j | c_{-i, S}, α)
	if length(newNodeNewRegions) > 0
		result.postpredNewNodeNewPartition += logpNewConnection * length(newNodeNewRegions)
		result.postpredNewNodeNewPartition += logpNewConnection * (length(newNodeNewPartitions) - 1)
	end

	return result

end

@doc doc"""

""" ->
function computeSampleTreePosteriors(regionId::Int, initialConfiguration::SPNConfiguration, cMax::SPNConfiguration, spn::SPNStructure,
	precomputations::Vector{RegionResultObject}; allowNew = true, initialScope = Set{Int}())

	postpreds = Vector{Float64}(0)
	configurations = Vector{SPNConfiguration}(0)

	config = deepcopy(initialConfiguration)
	canIncrease = true

	# 2.1) get all configurations
	while canIncrease

		# increase configuration if possible
		increased = false
		pos = size(spn.regions, 1)

		while (!increased) & (pos != 0)

			if config.c[pos][1] == -1
				pos -= 1
				continue
			end

			if isa(spn.regions[pos], LeafRegion)
				if allowNew
					cMaxValue = cMax.c[pos][1] + 1
				else
					cMaxValue = cMax.c[pos][1]
				end

				if (config.c[pos][1] + 1) <= cMaxValue # +1 is new node!
					config.c[pos][1] += 1
					increased = true
				else
					config.c[pos][1] = 1
				end

			else # SumRegion

				# node increase
				# except for root...
				if !spn.regions[pos].isRoot

					if allowNew
						cMaxValue = cMax.c[pos][1] + 1
					else
						cMaxValue = cMax.c[pos][1]
					end

					if (config.c[pos][1] + 1) <= cMaxValue
						config.c[pos][1] += 1
						increased = true
					else
						config.c[pos][1] = 1
					end

				end

				if increased
					continue
				end

				# partition increase
				if allowNew
					cMaxValue = cMax.c[pos][2] + 1
				else
					cMaxValue = cMax.c[pos][2]
				end

				if (config.c[pos][2] + 1) <= cMaxValue
					config.c[pos][2] += 1
					increased = true
				else
					config.c[pos][2] = 1
				end

			end

			if !increased
				pos -= 1
			end

		end

		if increased

			newconfig = deepcopy(config)

			postpred = 0.0

			# list of all selected partition scopes define a sample tree as there is only one region with scope Y!
			sampleTreeScopes = Vector{Set{Int}}(0)
			activeRegions = Vector{Int}(0)

			for regionId in 1:length(spn.regions)

				if initialConfiguration.c[regionId][1] == -1
					continue
				end

				if isa(spn.regions[regionId], SumRegion)

					if !spn.regions[regionId].isRoot
						# check if the region is inside the sample tree
						if !(spn.regions[regionId].scope in sampleTreeScopes) & !(spn.regions[regionId].scope == initialScope)
							newconfig.c[regionId][1] = -1
							continue
						end
					end

					push!(activeRegions, regionId)

					cNode = newconfig.c[regionId][1]
					cPartition = newconfig.c[regionId][2]

					if (cNode > cMax.c[regionId][1]) & (cPartition > cMax.c[regionId][2])
						postpred += precomputations[regionId].postpredNewNodeNewPartition
						newconfig.newRegions[regionId] = precomputations[regionId].configNewNodeNewPartitionRegions
						newconfig.newPartitions[regionId] = precomputations[regionId].configNewNodeNewPartitionPartitions

						for connectedPartition in newconfig.newPartitions[regionId]
							for subscope in connectedPartition.subscopes
								push!(sampleTreeScopes, subscope)
							end
						end
					elseif (cNode > cMax.c[regionId][1])
						postpred += precomputations[regionId].postpredNewNode

						for subscope in spn.regionConnections[spn.regions[regionId]][cPartition].subscopes
							push!(sampleTreeScopes, subscope)
						end

					elseif (cPartition > cMax.c[regionId][2])
						postpred += precomputations[regionId].postpredNewPartitions[cNode]
						newconfig.newRegions[regionId] = precomputations[regionId].configNewPartitionRegions[cNode]
						newconfig.newPartitions[regionId] = precomputations[regionId].configNewPartitionPartitions[cNode]

						for connectedPartition in newconfig.newPartitions[regionId]
							for subscope in connectedPartition.subscopes
								push!(sampleTreeScopes, subscope)
							end
						end

					else
						postpred += precomputations[regionId].postpred[cNode, cPartition]

						for subscope in spn.regionConnections[spn.regions[regionId]][cPartition].subscopes
							push!(sampleTreeScopes, subscope)
						end
					end

				else

					push!(activeRegions, regionId)

					cNode = newconfig.c[regionId][1]

					if (cNode > cMax.c[regionId][1])

						# find out which to take
						foundInformed = false
						for (ci, cInformed) in enumerate(precomputations[regionId].configInformedNodes)

							if foundInformed
								continue
							end

							(pRegionId, pPartitionId) = cInformed

							if pRegionId in activeRegions
								if newconfig.c[pRegionId][2] == findfirst(spn.partitions[activePartitions] .== spn.regionConnections[spn.regions[pRegionId]])
									postpred += precomputations[regionId].postpredInformedNodes[ci]
									foundInformed = true
								end
							end

						end

						if !foundInformed
							postpred += precomputations[regionId].postpredUninformedNode
						end

					else
						postpred += precomputations[regionId].postpredNodes[cNode]
					end

				end

			end

			push!(postpreds, postpred)
			push!(configurations, newconfig )

		end

		canIncrease = increased

	end

	return (postpreds, configurations)

end

@doc doc"""
Process all possible sample trees inside an infinite SPN.

The approach uses two independent steps.
1.) evaluate all regions in increasing scope size order
2.) compute posterior of sample trees using precomputed values

processAllConfigurations() -> [p(x | T1), p(x | T2), ... p(x | T*)]

""" ->
function processAllConfigurations(initialConfiguration::SPNConfiguration, cMax::SPNConfiguration, spn::SPNStructure,
	assign::AssignmentRegionGraph, X::AbstractArray, xi::Int; allowNew = true, initialScope = Set{Int}())

	regionsIds = collect(1:length(spn.regions))
	sort!(regionsIds, by = id -> length(spn.regions[id].scope))

	#f(id, initC, maxC, spnStruct, assignments, Data, observation) = posteriorPredictive(spnStruct.regions[id], id, initC, maxC, spnStruct, assignments, Data, observation)

	#println("run parallel: $(now())")
	#@time results = ptableany(f, regionsIds, Any[initialConfiguration], Any[cMax], Any[spn], Any[assign], Any[X], Any[xi])

	results = Vector{RegionResultObject}(length(regionsIds))
	#results = FunctionalData.typed(results)

	#println("run normal: $(now())")
	for id in regionsIds
		results[id] = posteriorPredictive(spn.regions[id], id, initialConfiguration, cMax, spn, assign, X, xi, allowNew = allowNew)
	end

	# compute sample tree posterior values
	return computeSampleTreePosteriors(regionsIds[end], initialConfiguration, cMax, spn, results, allowNew = allowNew, initialScope = initialScope)

end

@doc doc"""
sampleConfiguration(id, result) for LeafRegions
""" ->
function sampleConfiguration(id::Int, region::LeafRegion, result::LeafRegionResultObject, spn::SPNStructure, config::SPNConfiguration; allowNew = true)

	cNodeMax = length(region.nodes)
	cNodeMaxIter = cNodeMax + 1
	size = length(result.postpredNodes) + 1

	if !allowNew
		size -= 1
		cNodeMaxIter -= 1
	end

	llhval = ones(size) * -Inf

	# configurations
	configs = Vector{Int}(size)

	i = 1
	for cNode in 1:cNodeMaxIter

		configs[i] = cNode

		if cNode <= cNodeMax
			llhval[i] = result.postpredNodes[cNode]
		else
			foundInformed = false
			for (ci, cInformed) in enumerate(result.configInformedNodes)

				if foundInformed
					continue
				end

				(pRegionId, pPartitionId) = cInformed

				if config.c[pRegionId][2] == pPartitionId
					llhval[i] = result.postpredInformedNodes[ci]
					foundInformed = true
				end
			end

			if !foundInformed
				llhval[i] = result.postpredUninformedNode
			end
		end

		i += 1

	end

	p = exp(llhval - maximum(llhval))
	k = BNP.rand_indices(p)

	return configs[k]

end

@doc doc"""
sampleConfiguration(id, result) for SumRegions
""" ->
function sampleConfiguration(id::Int, region::SumRegion, result::SumRegionResultObject, spn::SPNStructure; allowNew = true)

	cPartitionMax = length(spn.regionConnections[region])
	cNodeMax = length(region.partitionPopularity)

	cPartitionMaxIter = cPartitionMax + 1
	cNodeMaxIter = cNodeMax + 1

	size = (cPartitionMax * cNodeMax) + length(result.postpredNewPartitions) + cPartitionMax + 1

	if !allowNew
		cPartitionMaxIter -= 1
		cNodeMaxIter -= 1
		size = cPartitionMax * cNodeMax
	end

	llhval = ones(size) * -Inf

	# configurations
	configs = Vector{Tuple{Int, Int}}(size)
	newRegions = Vector{Vector{Region}}(size)
	newPartitions = Vector{Vector{Partition}}(size)

	i = 1
	for cPartition in 1:cPartitionMaxIter
		for cNode in 1:cNodeMaxIter

			newRegions[i] = Vector{Region}(0)
			newPartitions[i] = Vector{Partition}(0)
			configs[i] = (cNode, cPartition)

			if (cNode <= cNodeMax) & (cPartition <= cPartitionMax)
				llhval[i] = result.postpred[cNode, cPartition]
			elseif (cNode <= cNodeMax)
				llhval[i] = result.postpredNewPartitions[cNode]
				newRegions[i] = result.configNewPartitionRegions[cNode]
				newPartitions[i] = result.configNewPartitionPartitions[cNode]
			elseif (cPartition <= cPartitionMax)
				llhval[i] = result.postpredNewNode
			else
				llhval[i] = result.postpredNewNodeNewPartition
				newRegions[i] = result.configNewNodeNewPartitionRegions
				newPartitions[i] = result.configNewNodeNewPartitionPartitions
			end

			i += 1

		end
	end

	p = exp(llhval - maximum(llhval))
	k = BNP.rand_indices(p)

	return (configs[k], newRegions[k], newPartitions[k])

end

@doc doc"""
Process all possible sample trees inside an infinite SPN.

The approach uses two independent steps.
1.) evaluate all regions in increasing scope size order
2.) compute posterior of sample trees using precomputed values

processAllConfigurations() -> [p(x | T1), p(x | T2), ... p(x | T*)]

""" ->
function processIterative(spn::SPNStructure, assign::AssignmentRegionGraph, X::AbstractArray, xi::Int; allowNew = true, initialScope = Set{Int}())

	regionsIds = collect(1:length(spn.regions))
	sort!(regionsIds, by = id -> length(spn.regions[id].scope), rev = true)

	initialConfiguration = SPNConfiguration(Vector{Vector{Int}}(size(spn.regions, 1)))
	cMax = SPNConfiguration(Vector{Vector{Int}}(size(spn.regions, 1)))

	for (ri, region) in enumerate(spn.regions)
		if isa(region, LeafRegion)
			initialConfiguration.c[ri] = [-1]
			cMax.c[ri] = [size(region.nodes, 1)] # all nodes
		else
			if region.isRoot & isempty(initialScope)
				initialConfiguration.c[ri] = [1, 1]
			elseif region.scope == initialScope
				initialConfiguration.c[ri] = [1, 1]
			else
				initialConfiguration.c[ri] = [-1, 1]
			end
			cMax.c[ri] = [size(region.partitionPopularity, 1), size(spn.regionConnections[region], 1)]
		end
	end

	for region in spn.regions
		if length(collect(Set(region.scope))) < 1
			println("---")
			println(spn.regions)
		end
	end

	for id in regionsIds

		if initialConfiguration.c[id][1] == -1
			continue
		end

		region = spn.regions[id]

		result = posteriorPredictive(region, id, initialConfiguration, cMax, spn, assign, X, xi, allowNew = allowNew)

		if isa(region, SumRegion)

			(config, newRegions, newPartitions) = sampleConfiguration(id, region, result, spn, allowNew = allowNew)

			initialConfiguration.c[id][1] = config[1]
			initialConfiguration.c[id][2] = config[2]
			initialConfiguration.newRegions[id] = newRegions
			initialConfiguration.newPartitions[id] = newPartitions

			# activate children
			if config[2] <= length(spn.regionConnections[region])
				partition = spn.regionConnections[region][config[2]]
				for childRegion in spn.partitionConnections[partition]
					childId = findfirst(childRegion .== spn.regions)
					if childId == 0
						println(spn.regions)
						println(childRegion)
					end
					initialConfiguration.c[childId][1] = 1
				end
			else
				for (cid, childRegion) in enumerate(spn.regions)
					for partition in newPartitions
						if Set(childRegion.scope) in partition.subscopes
							initialConfiguration.c[cid][1] = maximum([initialConfiguration.c[cid][1], 1])
						end
					end
				end
			end
		else

			config = sampleConfiguration(id, region, result, spn, initialConfiguration, allowNew = allowNew)
			initialConfiguration.c[id][1] = config
		end

	end

	println(initialConfiguration)

	return (initialConfiguration, cMax)

end

@doc doc"""
Remove a observation from the infinite SPN.
""" ->
function removeObservation!(observation::Int, x::AbstractArray, spn::SPNStructure, assign::AssignmentRegionGraph; regionsSubset = Vector{Region}(0))

	activeRegions = assign.regionAssignments[observation]
	activePartitions = assign.partitionAssignments[observation]

	# list of regions to remove
	regionsToRemove = Vector{Region}(0)

	# list of partitions to remove
	partitionsToRemove = Vector{Partition}(0)

	# remove observation from regions and Distributions
	for (region, cNode) in activeRegions

		if !isempty(regionsSubset)
			if !(region in regionsSubset)
				continue
			end
		end

		# decrease popularity
		region.popularity[cNode] -= 1
		region.N -= 1
		delete!(assign.observationRegionAssignments[region], observation)
		delete!(assign.regionAssignments[observation], region)

		if isa(region, LeafRegion)

			# remove from Distribution
			remove_data!(region.nodes[cNode].dist, x[region.nodes[cNode].scope,:])

		elseif isa(region, SumRegion)

			# removal of partition assignment
			region.partitionPopularity[cNode][activePartitions[region]] -= 1
			delete!(assign.observationPartitionAssignments[activePartitions[region]], observation)

			if length(assign.observationPartitionAssignments[activePartitions[region]]) == 0
				push!(partitionsToRemove, activePartitions[region])
			end

			delete!(assign.partitionAssignments[observation], region)
		end

		# remove node if the node is now empty
		if region.popularity[cNode] == 0

			deleteat!(region.popularity, cNode)

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

	end

	# check if we have to remove regions
	if length(regionsToRemove) > 0

		for region in regionsToRemove

			# loop over all partitions and remove partitionConnections
			for partition in spn.partitions
				if region in spn.partitionConnections[partition]
					println("..")
					println(length(assign.observationPartitionAssignments[partition]))
					println(region)
					println(partition)
					deleteat!(spn.partitionConnections[partition], findfirst(region .== spn.partitionConnections[partition]))
				end
				if region in spn.partitionConnectionsBottomUp[partition]
					deleteat!(spn.partitionConnectionsBottomUp[partition], findfirst(region .== spn.partitionConnectionsBottomUp[partition]))
				end
			end

			# remove regionConnections
			delete!(spn.regionConnections, region)
			delete!(spn.regionConnectionsBottomUp, region)
			deleteat!(spn.regions, findfirst(region .== spn.regions))
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
				if partition in spn.regionConnectionsBottomUp[region]
					deleteat!(spn.regionConnectionsBottomUp[region], findfirst(partition .== spn.regionConnectionsBottomUp[region]))
				end
			end

			# remove partitionConnections
			delete!(spn.partitionConnections, partition)
			delete!(spn.partitionConnectionsBottomUp, partition)
			deleteat!(spn.partitions, findfirst(partition .== spn.partitions))
		end
	end
end

@doc doc"""
Add the observation and if necessary the new sub structure.
""" ->
function addObservation!(observation::Int, X::AbstractArray, config::SPNConfiguration, cMax::SPNConfiguration, spn::SPNStructure, assign::AssignmentRegionGraph; G0Type = NormalGamma)

	regionIdMapping = Dict{Region, Int}()

	for (regionId, region) in enumerate(spn.regions)
		regionIdMapping[region] = regionId
	end

	# add sample to existing structure
	for region in spn.regions

		if !haskey(regionIdMapping, region)
			continue
		end

		regionId = regionIdMapping[region]

		if config.c[regionId][1] == -1
			continue
		end

		c = config.c[regionId]

		# check if we are out of the range(new node)
		if c[1] > cMax.c[regionId][1]

			# new node
			push!(region.popularity, 1)

			if isa(region, LeafRegion)

				# find out which to take
				foundInformed = false
				G0 = BNP.fit(G0Type, [X[region.scope, observation]]', computeScale = false)

				# for all possible new nodes with informed prior
				for partition in spn.regionConnectionsBottomUp[region]

					if foundInformed
						continue
					end

					obs1 = assign.observationPartitionAssignments[partition]

					for parentRegion in spn.partitionConnectionsBottomUp[partition]

						partitionId = findfirst(partition .== spn.regionConnections[parentRegion])
						parentRegionId = findfirst(parentRegion .== spn.regions)

						if config.c[parentRegionId][1] == -1
							continue
						end

						if config.c[parentRegionId][2] == partitionId
							foundInformed = true
						end

						obs = intersect(obs1, assign.observationRegionAssignments[parentRegion])

						# .) compute llh using adjusted mean0
						G0 = BNP.fit(G0Type, X[region.scope, collect(obs)])
					end
				end

				leaf = UnivariateNode{ConjugatePostDistribution}(G0, scope = region.scope)
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
				for newpartition in config.newPartitions[regionId]

					partitionAdded = true

					# create new partition
					pid = size(spn.partitions, 1) + 1
					push!(spn.partitions, newpartition)

					# create assignment holders
					spn.partitionConnectionsBottomUp[newpartition] = Vector{Region}(0)
					spn.partitionConnections[newpartition] = Vector{Region}(0)

					# check if this partition should be connected to the region
					if region.scope == newpartition.scope
						region.partitionPopularity[c[1]][newpartition] = 1

						push!(spn.partitionConnectionsBottomUp[newpartition], region)
						push!(spn.regionConnections[region], newpartition)
						assign.partitionAssignments[observation][region] = newpartition

					else # find region that should connect to the region

						for sregion in spn.regions
							if sregion.scope == newpartition.scope

								# connect partition to region (assume this is a new region, -> number of children = 0)
								push!(sregion.partitionPopularity, Dict{Partition, Int}())
								@assert size(sregion.partitionPopularity, 1) == 1
								sregion.partitionPopularity[1][newpartition] = 1
								push!(sregion.popularity, 1)

								push!(spn.partitionConnectionsBottomUp[newpartition], sregion)
								push!(spn.regionConnections[sregion], newpartition)
								assign.partitionAssignments[observation][sregion] = newpartition

							end
						end

					end

					assign.observationPartitionAssignments[newpartition] = Set{Int}(observation)

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
								push!(spn.regionConnectionsBottomUp[sregion], newpartition)
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
									push!(spn.regions, newregion)
									spn.regionConnections[newregion] = Vector{Partition}(0)
									spn.regionConnectionsBottomUp[newregion] = Vector{Partition}(0)
									assign.observationRegionAssignments[newregion] = Set{Int}(observation)
									assign.regionAssignments[observation][newregion] = 1
								end
							end

						end

					end

				end

			end

		end

		# increase popularity ,
		if !(observation in assign.observationRegionAssignments[region])
			region.popularity[c[1]] += 1
			region.N += 1
			push!(assign.observationRegionAssignments[region], observation)
			assign.regionAssignments[observation][region] = c[1]

			if isa(region, LeafRegion)

				# add to Distribution
				add_data!(region.nodes[c[1]].dist, [X[region.scope, observation]]')

			elseif isa(region, SumRegion) & !partitionAdded

				# add to partition assignments
				if !haskey(region.partitionPopularity[c[1]], spn.regionConnections[region][c[2]])
					region.partitionPopularity[c[1]][spn.regionConnections[region][c[2]]] = 1
				else
					region.partitionPopularity[c[1]][spn.regionConnections[region][c[2]]] += 1
				end
				assign.partitionAssignments[observation][region] = spn.regionConnections[region][c[2]]
				push!(assign.observationPartitionAssignments[spn.regionConnections[region][c[2]]], observation)
			end

		end

	end

end

@doc doc"""
Assign observations to regions and partitions, and create region & partitions
if necessary.

assignAndBuildRegionsPartitions!(observations, scope, spn, assign) -> (spn, assign)

""" ->
function assignAndBuildRegionsPartitions!(observation::Int, X::AbstractArray, scope::Set{Int}, spn::SPNStructure, assign::AssignmentRegionGraph; onRecurse = false)

	returnedRegion = SumRegion()

	#check if there exists such a region
	regionFound = false
	for region in spn.regions
		if regionFound
			continue
		end

	 if Set(region.scope) == scope
		 regionFound = true
		 returnedRegion = region
	 end
	end

	if regionFound
		# add obseravtions to region

			(config, cMax) = processIterative(spn, assign, X, observation; allowNew = !onRecurse, initialScope = scope)

			addObservation!(observation, X, config, cMax, spn, assign)

	else
		# construct new region and proceed
		region = SumRegion()
		region.scope = scope
		region.partitionPopularity = Vector{Dict{Partition, Int64}}(0)
		region.popularity = Vector{Int}(0)
		region.N = 1

		# add new region!
		push!(spn.regions, region)
		returnedRegion = region
		spn.regionConnections[region] = Vector{Partition}(0)
		spn.regionConnectionsBottomUp[region] = Vector{Partition}(0)
		assign.observationRegionAssignments[region] = Set{Int}(observation)
		assign.regionAssignments[observation][region] = 1

		# get new partitions and regions
		(newPartitions, newRegions) = buildPartitionsAndRegions!(region, 1, spn)

		# actual construct the regions and partitions
		for newpartition in newPartitions

			partitionAdded = true

			# create new partition
			pid = size(spn.partitions, 1) + 1
			push!(spn.partitions, newpartition)
			spn.partitionConnections[newpartition] = Vector{Region}()
			spn.partitionConnectionsBottomUp[newpartition] = Vector{Region}()

			# check if this partition should be connected to the region
			if region.scope == newpartition.scope
				push!(region.partitionPopularity, Dict{Partition, Int}())
				@assert size(region.partitionPopularity, 1) == 1
				region.partitionPopularity[1][newpartition] = 1
				push!(region.popularity, 1)

				push!(spn.partitionConnectionsBottomUp[newpartition], region)
				push!(spn.regionConnections[region], newpartition)
				assign.partitionAssignments[observation][region] = newpartition

			else # find region that should connect to the partition

				for sregion in spn.regions
					if sregion.scope == newpartition.scope

						# connect partition to region (assume this is a new region, -> number of children = 0)
						push!(sregion.partitionPopularity, Dict{Partition, Int}())
						@assert size(sregion.partitionPopularity, 1) == 1
						sregion.partitionPopularity[1][newpartition] = 1
						push!(region.popularity, 1)

						push!(spn.partitionConnectionsBottomUp[newpartition], sregion)
						push!(spn.regionConnections[sregion], newpartition)
						assign.partitionAssignments[observation][sregion] = newpartition

					end
				end

			end

			assign.observationPartitionAssignments[newpartition] = Set{Int}(observation)

			scopes = collect(keys(newpartition.indexFunction))
			parts = collect(values(newpartition.indexFunction))
			partIds = unique(parts)

			for partId in partIds
				idx = find(partId .== parts)

				subscope = Set(scopes[idx])

				splitFound = false
				for sregion in spn.regions
					if Set(sregion.scope) == subscope
						splitFound = true

						# connect partition to region
						push!(spn.partitionConnections[newpartition], sregion)
						push!(spn.regionConnectionsBottomUp[sregion], newpartition)
					end
				end

				if splitFound
					continue
				else

					# check new regions
					newregions = newRegions

					for newregion in newregions
						if newregion.scope == subscope

							# add new region!
							push!(spn.regions, newregion)
							spn.regionConnections[newregion] = Vector{Partition}(0)
							spn.regionConnectionsBottomUp[newregion] = Vector{Partition}(0)
							assign.observationRegionAssignments[newregion] = Set{Int}(observation)
							assign.regionAssignments[observation][newregion] = 1
						end
					end

				end

			end

		end

		# recurse
		assignAndBuildRegionsPartitions!(observation, X, scope, spn, assign, onRecurse = true)

	end

	return returnedRegion

end

@doc doc"""
Update partitions of the partition-regions in an infinite SPN.
""" ->
function updatePartitions!(X::AbstractArray, spn::SPNStructure, assign::AssignmentRegionGraph; partitionPrior = :CRP, G0Type = GaussianWishart)

	# sort partitions by scope
	sortedPartitions = sort(spn.partitions, by=p -> length(p.scope))

	# update each partition if sample count is sufficiently highy
	for partition in sortedPartitions

		# get number of assignments
		initK = length(unique(values(partition.indexFunction)))

		Ds = collect(partition.scope)
		Ns = collect(assign.observationPartitionAssignments[partition])

		if (length(Ns) > 0) & (length(Ds) >= initK)

			oldIdxFun = Array{Int}([partition.indexFunction[s] for s in partition.scope])
			idxFun = copy(oldIdxFun)

			# construct data matrix
			Xhat = X[Ds,Ns]'

			(D, N) = size(Xhat)

			if partitionPrior == :CRP

				G0 = BNP.fit(G0Type, Xhat, computeScale = false)

				models = train(DPM(G0), Gibbs(burnin = 0, maxiter = 1, thinout = 1), PrecomputedInitialisation(idxFun), Xhat)

				# get assignment
				idx = vec(models[end].assignments)

				if length(unique(idx)) == 1
					# this means there is no partition -> just keep the old one...
					idxFun = oldIdxFun
				else
					idxFun = idx
				end
			elseif partitionPrior == :VCM

				models = train(VCM(), Gibbs(burnin = 0, maxiter = 1, thinout = 1), IncrementalInitialisation(), Xhat)

				# get assignments
				for model in models
					println(size(full(model.C)))
				end
				#Z = reduce(hcat, map(model -> vec(model.C), models))
				#

			end

			# make sure assignments are in range
			newIdxFun = zeros(Int, length(idxFun))
			uz = unique(idxFun)
			for (zi, z) in enumerate(uz)
				idx = find(idxFun .== z)
				newIdxFun[idx] = zi
			end

			if adjustedRandIndex(newIdxFun, oldIdxFun) == 1
				continue
			else

				# splitting has changed
				groups = unique(newIdxFun)

				for group in groups

					idx = find(group .== newIdxFun)

					if length(idx) == 1
						continue
					end

					subscope = Set(collect(partition.scope)[idx])

					# try to find region with such scope in list of connected regions
					foundExistingConnection = false
					for region in spn.partitionConnections[partition]
						if Set(region.scope) == subscope
							foundExistingConnection = true
						end
					end

					if foundExistingConnection
						continue
					end

					# get list of observations

					# get relevant regions
					relevantRegions = Vector{Region}(0)

					oldGroups = unique(values(partition.indexFunction))
					vs = collect(values(partition.indexFunction))
					for g in oldGroups
						ids = find(g .== vs)
						subscope2 = Set(collect(keys(partition.indexFunction))[ids])

						if ⊆(subscope2, subscope) | ⊆(subscope, subscope2)
							println(subscope2)
							println(subscope)
							for region in spn.partitionConnections[partition]
								println(region)
								if Set(region.scope) == subscope2

									# memorize this region
									push!(relevantRegions, region)
								end
							end
						end
					end

					# extract relevant observations
					obs = Set{Int}()
					for region in relevantRegions
						union!(obs, intersect(assign.observationRegionAssignments[region], assign.observationPartitionAssignments[partition]))
					end

					newRegion = SumRegion()

					@assert length(collect(obs)) > 0


					# remove observations from regions
					for observation in collect(obs)
						println(relevantRegions)
						println(subscope)

						removeObservation!(observation, X[:,observation], spn, assign, regionsSubset = relevantRegions)
						# reassign observations to regions and partitions
						newRegion = assignAndBuildRegionsPartitions!(observation, X, subscope, spn, assign)
					end

					# remove old connections and add new connection
					for region in relevantRegions
						if region in spn.partitionConnections[partition]
							deleteat!(spn.partitionConnections[partition], findfirst(region .== spn.partitionConnections[partition]))
						end
					end

					println(newRegion)

					# adding
					push!(spn.partitionConnections[partition], newRegion)


				end
			end

			# update index Function of partition
			partition.indexFunction = [s => newIdxFun[si] for (si, s) in enumerate(partition.scope)]

		end
	end

	(spn, assign)

end
