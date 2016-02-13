using BNP
using SPN
using Distributions
using JuMP

# data
dataDir = "/home/martint/git/data/GensData/data"
splitDir = "/home/martint/git/data/GensData/discretized"

# split definition for cmll eval
splits = "VQ_Q0.10_E0.30"

# alpha values from Simplifying, Regularizing and Strengthening Sum-Product Network Structure Learning
αs = Dict("nltcs" => 2.0,
"msnbc" => 1.0,
"kdd" => 2.0,
"plants" => 1.0,
"baudio" => 2.0,
"jester" => 2.0,
"bnetflix" => 2.0,
"accidents" => 0.2,
"tretail" => 2.0,
"pumsb_star" => 0.1,
"dna" => 0.1,
"kosarek" => 2.0,
"msweb" => 1.0,
"book" => 2.0,
"tmovie" => 2.0,
"cwebkb" => 2.0,
"cr52" => 1.0,
"bbc" => 1.0,
"ad" => 0.1
)

# number of validation runs
numberOfRuns = 5

# learn the spn?
learn = true

# evaluate the spn?
evaluate = true

files = readdir(dataDir)

# get tests
tests = unique([split(file, '.')[1] for file in collect(files)])
tests = filter(t -> length(t) > 1, tests)

fsizes = [test => filesize(joinpath(dataDir, "$(test).ts.data")) for test in tests]

tests = sort(tests, by = test -> fsizes[test])

# learn initial SPN for each task
for testid in 1:5

	test = tests[testid]

	bestLLH = -1000.0

	println("* testing: ", test, " - [$(now())]")

	if learn
		for run in 1:numberOfRuns

			println("  # run $(run)")

			# load data
			X = round(Int, readdlm(joinpath(dataDir, "$(test).ts.data"), ','))

			# data is N * D so we have to flip it
			X = sparse(X)'

			(D, N) = size(X)

			# initialisation stuff
			dimMapping = Dict{Int, Int}([convert(Int, d) => convert(Int, d) for d in 1:D])
			obsMapping = Dict{Int, Int}([convert(Int, n) => convert(Int, n) for n in 1:N])
			assignments = Assignment()

			# learn SPN using Gens Approach
			α = 1.0
			if haskey(αs, test)
				α = αs[test]
			end

			root = SPN.learnSPN(X, dimMapping, obsMapping, assignments, method = :BM, G0Type = MultinomialDirichlet, L0Type = BinomialBeta, α = α)

			X = round(Int, readdlm(joinpath(dataDir, "$(test).valid.data"), ','))
			X = sparse(X)'
			(D, N) = size(X)

			# compute llh value
			llhvalid = mean([llh(root, X[:,i])[1] for i in 1:N])

			if llhvalid > bestLLH
				bestLLH = llhvalid

				outFile = open("$(test).jd","w")
				serialize(outFile,root)
				close(outFile)
			end

		end
	end

	if evaluate
		# load best and SPN
		inFile = open("$(test).jd","r")
		root = deserialize(inFile)
		close(inFile)

		SPN.fixSPN!(root)

		#println("* drawing structure")
		#drawSPN(root, file = "$(test).svg")

		X = round(Int, readdlm(joinpath(dataDir, "$(test).test.data"), ','))
		X = sparse(X)'
		(D, N) = size(X)

		# compute llh values
		llhtest = [llh(root, X[:,i])[1] for i in 1:N]
		println("* test llh: ", mean(llhtest))

		# load query and evidence information by Rob Gens
		f = open(joinpath(splitDir, test, "$(splits).ev"))

		evidence = Vector{Dict{Int, Int}}(0)
		for ln in eachline(f)
		       columns = split(ln, ',')

		       x = Dict{Int, Int}()
		       for (ci, column) in enumerate(columns)
		              if '*' in column
		                     continue
		              end
		              x[ci] = parse(Int, column)

		       end

		       push!(evidence, x)
		end

		f = open(joinpath(splitDir, test, "$(splits).q"))

		query = Vector{Dict{Int, Int}}(0)
		for ln in eachline(f)
		       columns = split(ln, ',')

		       x = Dict{Int, Int}()
		       for (ci, column) in enumerate(columns)
		              if '*' in column
		                     continue
		              end
		              x[ci] = parse(Int, column)

		       end

		       push!(query, x)
		end

		result = 0.0
		for i in 1:length(query)
			result += cmllh(root, query[i], evidence[i])
		end

		result /= N
		println("* cmllh: ", result)
	end
end
