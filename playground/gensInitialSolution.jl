using BNP
using SPN
using Distributions

# data
dataDir = "/home/martint/git/data/"

files = readdir(dataDir)

# get tests
tests = unique([split(file, '.')[1] for file in collect(files)])

# learn initial SPN for each task
for test in tests

	println("* testing: ", test, " - [$(now())]")

	# load data
	X = readdlm(joinpath(dataDir, "$(test).ts.data"), ',')

	# data is N * D so we have to flip it
	X = X'

	(D, N) = size(X)

	# initialisation stuff
	dimMapping = Dict{Int, Int}([convert(Int, d) => convert(Int, d) for d in 1:D])
	obsMapping = Dict{Int, Int}([convert(Int, n) => convert(Int, n) for n in 1:N])
	assignments = Assignment()

	# learn SPN using Gens Approach
	root = SPN.learnSPN(X, dimMapping, obsMapping, assignments, method = :BM)
	drawSPN(root, file = "$(test).svg")

	# compute llh values
	llhtrain = 0.0
	for i in 1:N
		llhtrain += llh(root, X[:,i])
	end

	llhtrain /= N
	println("* train llh: ", llhtrain)

	# compute llh values
	X = readdlm(joinpath(dataDir, "$(test).valid.data"), ',')
	X = X'
	(D, N) = size(X)

	llhvalid = 0.0
	for i in 1:N
		llhvalid += llh(root, X[:,i])
	end

	llhvalid /= N
	println("* validation llh: ", llhvalid)

	# compute llh values
	X = readdlm(joinpath(dataDir, "$(test).test.data"), ',')
	X = X'
	(D, N) = size(X)

	llhtest = 0.0
	for i in 1:N
		llhtest += llh(root, X[:,i])
	end

	llhtest /= N
	println("* test llh: ", llhtest)


end
