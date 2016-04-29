reload("SPN")

using SPN

# construct layered SPN
spn = SPN.LayeredSPN()

# add class layer
classLayer = SPN.ProductLayer()

# add class to class layer
SPN.connect!(classLayer, SPN.SumLayer(10), SPN.ClassNode(1))
SPN.connect!(classLayer, SPN.SumLayer(10), SPN.ClassNode(2))

# add children to sum layers
for child in SPN.children(classLayer)
	println(child)

	distLayer = SPN.NormalDistributionLayer()
	for d in 1:10
		SPN.add!(distLayer, SPN.NormalDistributionNode(1))
	end

	SPN.connect!(child, distLayer, rand(length(child), length(distLayer)))
end

SPN.children(classLayer)
