using SumProductNetworks, BenchmarkTools

N = 10000
D = 10000
C = 100

llhval = rand(N, C+1)
X = rand(N, D)

S = SumNode(1)

for c in 1:C
  add!(S, ProductNode(c+1))
end

bsum = @benchmark SumProductNetworks.eval!(S, X, llhval)
println("Sum Node")
println(bsum)

P = ProductNode(1)

for c in 1:C
  add!(P, ProductNode(c+1))
end

bprod = @benchmark SumProductNetworks.eval!(P, X, llhval)

println("Product Node")
println(bprod)

N = MultivariateFeatureNode(1, collect(1:D))

bfeat = @benchmark SumProductNetworks.eval!(N, X, llhval)

println("Multivariate Feature Node")
println(bfeat)

# evaluation for layer structured computation
C = 10
G = 10
K = 100
N = 100
D = K*G*G
S = imageStructuredSPN(C, D, G, K)

# perperations
nodes = order(S)
maxDepth = maximum(depth(n) for n in nodes)
layerMapping = Dict(node.id => depth(node) for node in nodes)
layers = [filter(n -> layerMapping[n.id] == d, nodes) for d in 0:maxDepth ]

X = hcat(rand(N, D), rand(1:C, N))
llhvals = rand(N, maximum(node.id for node in nodes))

# first implementation
function llh1(nodes, X, llhvals)
  for node in nodes
    SumProductNetworks.eval!(node, X, llhvals)
  end
end

bllh1 = @benchmark llh1(nodes, X, llhvals)
println(bllh1)

# second implementation
function llh2(layers, X, llhvals)
  for layer in layers
    for node in layer
      SumProductNetworks.eval!(node, X, llhvals)
    end
  end
end

bllh2 = @benchmark llh2(layers, X, llhvals)
println(bllh2)
