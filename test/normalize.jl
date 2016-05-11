using SPN

X = rand(2, 100)

S = SPN.randomStructure(X, [1, 2], 2, 2)

nodes = SPN.order(S)

for node in nodes
  if isa(node, SumNode)
    println(node.weights)
  end
end
