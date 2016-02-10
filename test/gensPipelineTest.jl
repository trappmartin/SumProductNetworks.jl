# data
dataDir = "/home/martint/git/data/GensData/data"
splitDir = "/home/martint/git/data/GensData/discretized"

test = "nltcs"
splits = "VQ_Q0.10_E0.30"

X = round(Int, readdlm(joinpath(dataDir, "$(test).ts.data"), ','))

# data is N * D so we have to flip it
X = sparse(X)'

(D, N) = size(X)

# initialisation stuff
dimMapping = Dict{Int, Int}([convert(Int, d) => convert(Int, d) for d in 1:D])
obsMapping = Dict{Int, Int}([convert(Int, n) => convert(Int, n) for n in 1:N])
assignments = Assignment()

println(" * loaded $(N) samples with $(D) variables")

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

println(evidence)

# learn SPN using Gens Approach
#root = SPN.learnSPN(X, dimMapping, obsMapping, assignments, method = :BM, G0Type = MultinomialDirichlet, L0Type = BinomialBeta)
