using SumProductNetworks

uid = 1

# nodes
S = SumNode(uid)

for k in 1:10
    uid += 1
    C = IndicatorNode(uid, 0, 1)
    add!(S, C)
end

# weight set
W = rand(10, 1) # Children \times Internal Nodes 

spn = SumProductNetworks.SumProductNetwork(S, W)
