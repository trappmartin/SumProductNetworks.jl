using SPN
using StatsFuns, NumericExtensions

node = NormalDistributionNode(1, 1)
data = ones(10000, 10000)
@time SPN.eval(node, data);

id = 1
node = ProductNode(id)

id += 1
for i in 1:1000
  SPN.add!(node, NormalDistributionNode(id, 1))
  id += 1
end

id

llhval = Matrix{Float64}(id-1, size(data, 2))
@time for child in children(node)
  SPN.eval!(child, data, llhval);
end

llhval

@time SPN.eval!(node, data, llhval);

id = 1
node = SumNode(id)

for i in 1:1000
  id += 1
  SPN.add!(node, NormalDistributionNode(id, 1))
end

llhval = Matrix{Float64}(id, size(data, 2))
@time for child in children(node)
  SPN.eval!(child, data, llhval);
end

@time SPN.eval!(node, data, llhval);
