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

println(" * - p(x | T , W, Θ) = ", p)

# 2.) roll the dice...
k = BNP.rand_indices(p)
k = length(p)
println("new config: ", k)

# 3.) add sample to new sample tree
config = configs[k]
@time SPN.addObservation!(observation, x, config, cMax, spn, assign)

# 4.) resample partitions
