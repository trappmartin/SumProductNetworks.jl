type SPNConfiguration

	c::Vector{Vector{Int}}
	newPartitions::Dict{Int, Vector{Partition}}
	newRegions::Dict{Int, Vector{Region}}

	SPNConfiguration(c::Vector{Vector{Int}}; nP = Dict{Int, Vector{Partition}}(), nR = Dict{Int, Vector{Region}}()) = new(c, nP, nR)

end

function buildPartitionsAndRegions!(region::Region, regionId::Int, newConfig::SPNConfiguration, spn::SPNStructure)

	if !haskey(newConfig.newPartitions, regionId)
		newConfig.newPartitions[regionId] = Vector{Partition}(0)
		newConfig.newRegions[regionId] = Vector{Region}(0)
	end

	# new partition
	p = Partition()

	# check scope
	indexFunction = Dict{Int, Int}()
	scope = collect(region.scope)

	p.scope = region.scope

	if length(scope) == 2
		p.indexFunction = Dict{Int, Int}(scope[1] => 1, scope[2] => 2)
		push!(newConfig.newPartitions[regionId], p)
	else
		pL = length(partitions(scope))
		parts = collect(partitions(scope))[rand(2:pL)]
		p.indexFunction = Dict{Int, Int}()
		push!(newConfig.newPartitions[regionId], p)

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
				r.popularity = Dict{Int64, Int64}()
				r.N = 1

				push!(newConfig.newRegions[regionId], r)
				buildPartitionsAndRegions!(r, regionId, newConfig, spn)

			end
		end
	end

	newConfig
end

function findConfigurations(c::SPNConfiguration, cMax::SPNConfiguration, spn::SPNStructure; allowNew = true)

	canIncrease = true
	configs = Vector{SPNConfiguration}(0)
	push!(configs, c) # push initial configuration

	# 2.1) get all configurations
	while canIncrease

		# last configuration
		newConfig = SPNConfiguration(deepcopy(configs[end].c))

		# increase configuration if possible
		increased = false
		pos = size(spn.regions, 1)

		while (!increased) & (pos != 0)

			if newConfig.c[pos][1] == -1
				pos -= 1
				continue
			end

			if isa(spn.regions[pos], LeafRegion)
				if (newConfig.c[pos][1] + 1) <= (cMax.c[pos][1] + 1) # +1 is new node!
					newConfig.c[pos][1] += 1
					increased = true
				else
					newConfig.c[pos][1] = 1
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

					if (newConfig.c[pos][1] + 1) <= cMaxValue
						newConfig.c[pos][1] += 1
						increased = true
					else
						newConfig.c[pos][1] = 1
					end

				end

				if increased & (newConfig.c[pos][2] > cMax.c[pos][2])
					newConfig = buildPartitionsAndRegions!(spn.regions[pos], pos, newConfig, spn)
					@assert haskey(newConfig.newPartitions, pos)
					@assert length(newConfig.newPartitions[pos]) > 0
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

				if (newConfig.c[pos][2] + 1) <= cMaxValue
					newConfig.c[pos][2] += 1
					increased = true
				else
					newConfig.c[pos][2] = 1
				end

				if newConfig.c[pos][2] > cMax.c[pos][2]
					newConfig = buildPartitionsAndRegions!(spn.regions[pos], pos, newConfig, spn)
					@assert haskey(newConfig.newPartitions, pos)
					@assert length(newConfig.newPartitions[pos]) > 0
				end

			end

			if !increased
				pos -= 1
			end

		end

		if increased

			# check if all regions have new partitions if necessary
			for pos in 1:size(spn.regions, 1)
				if isa(spn.regions[pos], SumRegion)
					if newConfig.c[pos][1] == -1
						continue
					end
					if newConfig.c[pos][2] > cMax.c[pos][2]
						if !haskey(newConfig.newPartitions, pos)
							newConfig = buildPartitionsAndRegions!(spn.regions[pos], pos, newConfig, spn)
						end
					end
				end
			end

			push!(configs, newConfig)
		end

		canIncrease = increased
	end

	return configs
end

@doc doc"""
Extract sample tree from configuration.
""" ->
function extractSampleTree(configs::SPNConfiguration, spn::SPNStructure)

	tree = Vector{Int}(0)

	for (rId, region) in enumerate(spn.regions)

		# skip if config is -1
		if configs.c[rId][1] == -1
			continue
		end

		# check if region is root (heuristic: no previous partition)
		isRoot = true

		for partition in spn.partitions
			isRoot &= !(region in spn.partitionConnections[partition])
		end

		# check if region is leaf region
		isLeaf = isa(region, LeafRegion)

		if isRoot | isLeaf
			push!(tree, rId)
		else

			# find out if region is inside tree
			foundSelection = false
			for partition in spn.partitions
				if region in spn.partitionConnections[partition]
					# check if the partition is selected by any region
					for (r2Id, r2) in enumerate(spn.regions)
						if partition in spn.regionConnections[r2]
							# partition is connected to region r2

							# is it selected?
							foundSelection |= (configs.c[r2Id][2] == findfirst(spn.regionConnections[r2], partition))
						end
					end
				end
			end

			# check if region can be found in "extended" tree
			#if configuration.newRegions[regionId]
			for (r2Id, r2) in enumerate(spn.regions)
				if haskey(configs.newPartitions, r2Id)

					for partition in configs.newPartitions[r2Id]

						# check if scopes match
						if ⊈(region.scope, partition.scope)
							continue
						end

						# get sub-scope of partition
						v = [partition.indexFunction[s] for s in region.scope]

						# check if sub scope matches in index function
						if !all(v .== v[1])
							continue
						end

						# check if sub scope is complete
						if sum(collect(values(partition.indexFunction)) .== v[1]) == length(v)
							foundSelection = true
						end

					end
				end
			end

			if foundSelection
				push!(tree, rId)
			end
		end

	end

	return tree

end

@doc doc"""
posterior predictive for leaf region.
""" ->
function posteriorPredictive(region::LeafRegion, regionId::Int, sampleTree::Vector{Int},
	configuration::SPNConfiguration, cMax::SPNConfiguration,
	spn::SPNStructure, x::AbstractArray; α = 1.0)

	postpred = 0.0

	llh = 0.0
	lc = 0.0

	# get selection
	cNode = configuration.c[regionId][1]
	cRMax = cMax.c[regionId][1]

	# check this is a new node
	if cNode > cRMax

		# get llh value
		# 1. get all observation that

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

		# 2. compute llh using adjusted mean0
		llh += logpred(NormalGamma(μ = SS / C), sub(x, region.scope, :))[1]

		# p(c_{i, Rg} = j | c_{-i, Rg}, α)
		lc += log(α / (region.N - 1 + α) )

	else

		# get llh values
		llh += logpred(region.nodes[cNode].dist, sub(x, region.nodes[cNode].scope, :))[1]

		# p(c_{i, Rg} = j | c_{-i, Rg}, α)
		@assert haskey(region.popularity, cNode)
		lc += log(region.popularity[cNode] / (region.N - 1 + α) )

	end

	postpred = llh + lc

	return postpred

end

@doc doc"""
posterior predictive for leaf region.
""" ->
function posteriorPredictive(region::SumRegion, regionId::Int, sampleTree::Vector{Int},
	configuration::SPNConfiguration, cMax::SPNConfiguration,
	spn::SPNStructure, x::AbstractArray; α = 1.0)

	postpred = 0.0

	# get selection
	cNode = configuration.c[regionId][1]
	cMaxNode = cMax.c[regionId][1]

	cPartition = configuration.c[regionId][2]
	cMaxPartition = cMax.c[regionId][2]

	# NOTE: We assume in allways that region.N > 0 !!!
	if (cNode > cMaxNode) & (cPartition <= cMaxPartition)

		# .) new sum node
		# p(c_{i, Rg} = j | c_{-i, Rg}, α)
		postpred += log(α / (region.N - 1 + α) )

		# p(c_{i, S} = j | c_{-i, S}, α)
		# NOTE: this is a sequential process
		# therefore, count for observation in the node is 1.
		postpred += log(α / (1 - 1 + α) )

	elseif (cNode <= cMaxNode) & (cPartition <= cMaxPartition)

		# p(c_{i, Rg} = j | c_{-i, Rg}, α)
		postpred += log(region.popularity[cNode] / (region.N - 1 + α) )

		# p(c_{i, S} = j | c_{-i, S}, α)
		# check if the node already has a popularity value for this partition otherwise its α
		if haskey(region.partitionPopularity[cNode], spn.regionConnections[region][cPartition])
			postpred += log(region.partitionPopularity[cNode][spn.regionConnections[region][cPartition]] / (region.popularity[cNode] - 1 + α) )
		else
			postpred += log(α / (region.popularity[cNode] - 1 + α) )
		end

	elseif (cNode <= cMaxNode) & (cPartition > cMaxPartition)

		# p(c_{i, Rg} = j | c_{-i, Rg}, α)
		postpred += log(region.popularity[cNode] / (region.N - 1 + α) )

		# p(c_{i, S} = j | c_{-i, S}, α)
		postpred += log(α / (region.popularity[cNode] - 1 + α) )

		# additional p(c_{i, S} = j | c_{-i, S}, α)
		if haskey(configuration.newRegions, regionId)
			postpred += log(α / (1 - 1 + α) ) * length(configuration.newRegions[regionId])
			postpred += log(α / (1 - 1 + α) ) * (length(configuration.newPartitions[regionId]) - 1)
		end

	else

		# p(c_{i, Rg} = j | c_{-i, Rg}, α)
		postpred += log(α / (region.N - 1 + α) )

		# p(c_{i, S} = j | c_{-i, S}, α)
		# NOTE: this is a sequential process
		# therefore, count for observation in the node is 1.
		postpred += log(α / (1 - 1 + α) )

		# additional p(c_{i, S} = j | c_{-i, S}, α)
		if haskey(configuration.newRegions, regionId)
			postpred += log(α / (1 - 1 + α) ) * length(configuration.newRegions[regionId])
			postpred += log(α / (1 - 1 + α) ) * (length(configuration.newPartitions[regionId]) - 1)
		end

	end

	return postpred

end

function processConfiguration(configuration::SPNConfiguration, cMax::SPNConfiguration, spn::SPNStructure, x::AbstractArray)
	postpred = 0.0

	# get list of regions in sample tree
	sampleTree = SPN.extractSampleTree(configuration, spn)

	for regionId in sampleTree # LOOP
		postpred += SPN.posteriorPredictive(spn.regions[regionId], regionId, sampleTree, configuration, cMax, spn, x)
	end

	return postpred
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
		if isa(region, SumRegion)
			if length(assign.observationPartitionAssignments[activePartitions[region]]) == 0
				push!(partitionsToRemove, activePartitions[region])
			end
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
end

@doc doc"""
Add the observation and if necessary the new sub structure.
""" ->
function addObservation!(observation::Int, x::AbstractArray, config::SPNConfiguration, cMax::SPNConfiguration, spn::SPNStructure, assign::AssignmentRegionGraph)

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
				for newpartition in config.newPartitions[regionId]

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
								sregion.popularity[1] = 1

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
									push!(spn.regions, newregion)
									spn.regionConnections[newregion] = Vector{Partition}(0)
									assign.observationRegionAssignments[newregion] = Set{Int}(observation)
									assign.regionAssignments[observation][newregion] = 1
								end
							end

						end

					end

				end

			end

		end

		# increase popularity
		if !(observation in assign.observationRegionAssignments[region])
			region.popularity[c[1]] += 1
			region.N += 1
			push!(assign.observationRegionAssignments[region], observation)
			assign.regionAssignments[observation][region] = c[1]

			if isa(region, LeafRegion)

				# add to Distribution
				add_data!(region.nodes[c[1]].dist, x[region.nodes[c[1]].scope,:])

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
function assignAndBuildRegionsPartitions!(observation::Int, x::AbstractArray, scope::Set{Int}, spn::SPNStructure, assign::AssignmentRegionGraph; onRecurse = false)

	returnedRegion = SumRegion()

	#check if there exists such a region
	regionFound = false
	for region in spn.regions
	 if Set(region.scope)== scope
		 regionFound = true
		 returnedRegion = region
	 end
	end

	if regionFound
		# add obseravtions to region

			c = SPNConfiguration(Vector{Vector{Int}}(size(spn.regions, 1)))
			cMax = SPNConfiguration(Vector{Vector{Int}}(size(spn.regions, 1)))

			for (ri, region) in enumerate(spn.regions)

				if ⊆(Set(region.scope), scope)

					if isa(region, LeafRegion)
						c.c[ri] = [1]
						cMax.c[ri] = [size(region.nodes, 1)] # all nodes
					else
						c.c[ri] = [1, 1]
						cMax.c[ri] = [size(region.partitionPopularity, 1), # all pseudo-nodes
																			size(spn.regionConnections[region], 1)]
					end
				else
					c.c[ri] = [-1]
					cMax.c[ri] = [-1]
				end

			end

			configs = SPN.findConfigurations(c, cMax, spn, allowNew = !onRecurse)

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

			p = exp(LLH - maximum(LLH))
			p = p ./ sum(p)

			k = BNP.rand_indices(p)

			# add to sampleTree
			config = configs[k]

			SPN.addObservation!(observation, x, config, cMax, spn, assign)

	else
		# construct new region and proceed
		region = SumRegion()
		region.scope = scope
		region.partitionPopularity = Vector{Dict{Partition, Int64}}(0)
		region.popularity = Dict{Int64, Int64}()
		region.N = 1

		# add new region!
		push!(spn.regions, region)
		returnedRegion = region
		spn.regionConnections[region] = Vector{Partition}(0)
		assign.observationRegionAssignments[region] = Set{Int}(observation)
		assign.regionAssignments[observation][region] = 1

		# get new partitions and regions
		newConfig = SPNConfiguration(Vector{Vector{Int}}(1))
		newConfig.newRegions[1] = Vector{Region}(0)
		newConfig.newPartitions[1] = Vector{Partition}(0)
		push!(newConfig.newRegions[1], region)
		SPN.buildPartitionsAndRegions!(region, 1, newConfig, spn)

		# actual construct the regions and partitions
		for newpartition in newConfig.newPartitions[1]

			partitionAdded = true

			# create new partition
			pid = size(spn.partitions, 1) + 1
			push!(spn.partitions, newpartition)

			# check if this partition should be connected to the region
			if region.scope == newpartition.scope
				push!(region.partitionPopularity, Dict{Partition, Int}())
				@assert size(region.partitionPopularity, 1) == 1
				region.partitionPopularity[1][newpartition] = 1
				region.popularity[1] = 1

				push!(spn.regionConnections[region], newpartition)
				assign.partitionAssignments[observation][region] = newpartition

			else # find region that should connect to the partition

				for sregion in spn.regions
					if sregion.scope == newpartition.scope

						# connect partition to region (assume this is a new region, -> number of children = 0)
						push!(sregion.partitionPopularity, Dict{Partition, Int}())
						@assert size(sregion.partitionPopularity, 1) == 1
						sregion.partitionPopularity[1][newpartition] = 1
						region.popularity[1] = 1

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
					newregions = newConfig.newRegions[1]

					for newregion in newregions
						if newregion.scope == subscope

							# add new region!
							push!(spn.regions, newregion)
							spn.regionConnections[newregion] = Vector{Partition}(0)
							assign.observationRegionAssignments[newregion] = Set{Int}(observation)
							assign.regionAssignments[observation][newregion] = 1
						end
					end

				end

			end

		end

		# recurse
		assignAndBuildRegionsPartitions!(observation, x, scope, spn, assign, onRecurse = true)

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

				G0 = BNP.fit(G0Type, Xhat, useCov = false)

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
							for region in spn.partitionConnections[partition]
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

					for observation in collect(obs)
						for region in spn.regions
							l1 = haskey(assign.regionAssignments[observation], region)
							l2 = (observation in assign.observationRegionAssignments[region])

							@assert !(l1 $ l2) "inconsistency for obseravtion $(observation) -> has region: $(l1), is in region: $(l2)"
						end
					end

					newRegion = SumRegion()

					# remove observations from regions
					for observation in collect(obs)
						SPN.removeObservation!(observation, X[:,observation], spn, assign, regionsSubset = relevantRegions)

						# reassign observations to regions and partitions
						newRegion = assignAndBuildRegionsPartitions!(observation, X[:,observation], subscope, spn, assign)
					end

					# remove old connections and add new connection
					for region in relevantRegions
						if region in spn.partitionConnections[partition]
							deleteat!(spn.partitionConnections[partition], findfirst(region .== spn.partitionConnections[partition]))
						end
					end

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
