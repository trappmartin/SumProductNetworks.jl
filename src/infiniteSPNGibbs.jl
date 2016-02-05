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
		p.indexFunction = Dict{Int, Int}(scope[1] => scope[1], scope[2] => scope[2])
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
end

function findConfigurations(c::SPNConfiguration, cMax::SPNConfiguration, spn::SPNStructure)

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

			if isa(spn.regions[pos], LeafRegion)
				if (newConfig.c[pos][1] + 1) <= (cMax.c[pos][1] + 1) # +1 is new node!
					newConfig.c[pos][1] += 1
					increased = true
				else
					newConfig.c[pos][1] = 1
				end

			else # SumRegion

				# node increase
				if (newConfig.c[pos][1] + 1) <= (cMax.c[pos][1] + 1)
					newConfig.c[pos][1] += 1
					increased = true
				else
					newConfig.c[pos][1] = 1
				end

				if increased
					continue
				end

				# partition increase
				if (newConfig.c[pos][2] + 1) <= (cMax.c[pos][2] + 1)
					newConfig.c[pos][2] += 1
					increased = true
				else
					newConfig.c[pos][2] = 1
				end

				if newConfig.c[pos][2] > cMax.c[pos][2]
					buildPartitionsAndRegions!(spn.regions[pos], pos, newConfig, spn)
				end

			end

			if !increased
				pos -= 1
			end

		end

		if increased
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

		# check if region is root (heuristic: no previous partition)
		isRoot = true

		for partition in spn.partitions
			isRoot &= !(region in spn.partitionConnections[partition])
		end

		if isRoot
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
	spn::SPNStructure, x; α = 1.0)

	postpred = 0.0

	llh = 0.0
	lc = 0.0

	# get selection
	cNode = configuration.c[regionId][1]
	cRMax = cMax.c[regionId][1]

	# check this is a new node
	if cNode > cRMax

		# get llh values
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
	spn::SPNStructure, x; α = 1.0)

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
		postpred += log(region.partitionPopularity[cNode][spn.regionConnections[region][cPartition]] / (region.popularity[cNode] - 1 + α) )

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
