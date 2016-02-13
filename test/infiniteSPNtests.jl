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
#println(" * draw initial SPN")
#drawSPN(root, file = "initialSPN.svg")

# transform SPN to regions and partition
println(" * transform SPN into regions and partitions")
(spn, assign) = transformToRegionPartition(root, assignments, N)

@test size(spn.partitions, 1) >= 1
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

@time configs = SPN.findConfigurations(c, cMax, spn)

# 2.) iterate over sample trees in the SPN
LLH = Vector{Float64}(length(configs))

@time for (i, configuration) in enumerate(configs)

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

# 2.) roll the dice...
k = BNP.rand_indices(p)
println("new config: ", k)

# 3.) add sample to new sample tree
config = configs[k]
@time SPN.addObservation!(observation, x, config, cMax, spn, assign)

println(spn)

# 4.) resample partition-regions

# 4.1) assign observations and create new regions and partitions if necessary
@doc doc"""
Assign observations to regions and partitions, and create region & partitions
if necessary.

assignAndBuildRegionsPartitions!(observations, scope, spn, assign) -> (spn, assign)

""" ->
function assignAndBuildRegionsPartitions!(observation::Int, x::AbstractArray, scope::Set{Int}, spn::SPNStructure, assign::AssignmentRegionGraph; onRecurse = false)

	#check if there exists such a region
	regionFound = false
	for region in spn.regions
	 if Set(region.scope)== scope
		 regionFound = true
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


end


# config:
partitionPrior = :CRP

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

			println("running CRP to find partition of scope")

			μ0 = vec( mean(Xhat, 2) )
			κ0 = 1.0
			ν0 = convert(Float64, D)
			Ψ = eye(D) * 10

			G0 = GaussianWishart(μ0, κ0, ν0, Ψ)

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

			println("running VCM to find partition of scope")

			models = train(VCM(), Gibbs(burnin = 0, maxiter = 1, thinout = 1), IncrementalInitialisation(), Xhat)

			# get assignments
			for model in models
				println(size(full(model.C)))
			end
			#Z = reduce(hcat, map(model -> vec(model.C), models))
			#
			#println(Z)

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

				# remove observations from regions
				for observation in collect(obs)
					SPN.removeObservation!(observation, X[:,observation], spn, assign, regionsSubset = relevantRegions)

					# reassign observations to regions and partitions
					assignAndBuildRegionsPartitions!(observation, X[:,observation], subscope, spn, assign)
				end
			end
		end

		# update index Function of partition
		partition.indexFunction = [s => newIdxFun[si] for (si, s) in enumerate(partition.scope)]

		# reset connections
		#TODO !!!

	end
end

println(spn)
