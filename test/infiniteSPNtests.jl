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

println(" * using dataset 'iris' with ", N, " observations and ", D, " variables.")

dimMapping = Dict{Int, Int}([convert(Int, d) => convert(Int, d) for d in 1:D])
obsMapping = Dict{Int, Int}([convert(Int, n) => convert(Int, n) for n in 1:N])
assignments = Assignment()

root = SPN.learnSPN(X, dimMapping, obsMapping, assignments)

llhtest = [llh(root, X[:,i])[1] for i in 1:N]
println("* initial llh: ", mean(llhtest))

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
#println(" * draw transformed SPN")
#drawSPN(spn, file = "transformedSPN.svg")

println(" * run Gibbs sweep using infinite SPN in regions and partition-regions representation")

@time for observation in 1:N

	println(" # observation: ", observation)

	x = X[:,observation]

	# 0.) remove observation from SPN
	SPN.removeObservation!(observation, x, spn, assign)

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

	@time LLH = map(configuration -> SPN.processConfiguration(configuration, cMax, spn, x), configs)

	#LLH = @parallel for configuration in configs
#		SPN.processConfiguration(configuration, cMax, spn, x)
#	end

#	println(LLH)
	p = float(LLH)
	p = exp(p - maximum(p))
	p = p ./ sum(p)

	# 2.) roll the dice...
	k = BNP.rand_indices(p)

	# 3.) add sample to new sample tree
	config = configs[k]
	SPN.addObservation!(observation, x, config, cMax, spn, assign)

	# 4.) resample partition-regions

	# 4.1) assign observations and create new regions and partitions if necessary
	@time (spn, assign) = SPN.updatePartitions!(X, spn, assign)

	# print some stats
	numDists = 0
	numInternalNodes = 0
	for region in spn.regions
		if isa(region, LeafRegion)
			numDists += length(region.nodes)
		else
			numInternalNodes += length(region.popularity)
		end
	end

	println(" * number of distributions: ", numDists)
	println(" * number of internal node: ", numInternalNodes)
	println(" * number of partition-regions: ", length(spn.partitions))

end



println(spn)
