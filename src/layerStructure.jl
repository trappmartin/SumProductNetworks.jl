export create_bayesian_discrete_layered_spn

"""
	Create a layer structure with consecutive Bayesian product-sum layers.

	Parameters:
	* M: Number of children under a sum node
	* K: Number of children under a product node
	* L: Depth (Number of consecutive product-sum layers, excluding the root layer)
	* D: Number of dimensions
	* S: Number of states for each dimension (we assume the same value for all dimenions), e.g. 2 = binary
"""
function create_bayesian_discrete_layered_spn(M, K, L, D, S; α = 1.0, β = 1.0, γ = 1.0)

	# create root
	root = BayesianSumLayer([1], zeros(Int, 0, 0), zeros(Int, M, 1), ones(1, 1) * (α / M), SPNLayer[], nothing)

	parent = root
	# create layers
	for l in 1:L

		# number of required nodes in the product layer
		pids = parent.ids
		max_pid = maximum(pids)+1
		pnode_count = length(pids) * M
		pnode_ids = collect(max_pid:(max_pid+pnode_count-1))

		# add product layer
		p_layer = BayesianProductLayer(pnode_ids, zeros(Int, 0, 0), zeros(Int, K, pnode_count), ones(1, pnode_count) * (β / K), SPNLayer[], parent)
  		push!(parent.children, p_layer)
		cids(parent) = reshape(pnode_ids, M, length(pids))

		# number of required nodes in the sum layer
		sids = pnode_ids
		max_sid = maximum(sids)+1
		snode_count = length(sids) * K
		snode_ids = collect(max_sid:(max_sid+snode_count-1))

		# add sum layer
		s_layer = BayesianSumLayer(snode_ids, zeros(Int, 0, 0), zeros(Int, M, snode_count), ones(1, snode_count) * (α / M), SPNLayer[], p_layer)
  		push!(p_layer.children, s_layer)
		cids(p_layer) = reshape(snode_ids, K, length(sids))

		parent = s_layer
	end

	# create leaves (full factorized categorical distributions)
	pids = parent.ids
	max_pid = maximum(pids)+1
	pnode_count = length(pids) * M
	pnode_ids = collect(max_pid:(max_pid+pnode_count-1))

	d_layer = ProductLayer(pnode_ids, zeros(Int, 0, 0), SPNLayer[], parent)
	push!(parent.children, d_layer)
	cids(parent) = reshape(pnode_ids, M, length(pids))

	# create categorical distributions with Dirichlet as prior
	ccids = pnode_ids
	max_cid = maximum(ccids)+1
	cnode_count = length(ccids) * D
	cnode_ids = collect(max_cid:(max_cid+cnode_count-1))

	c_layer = BayesianCategoricalLayer(cnode_ids, vec(repmat(collect(1:D), 1, length(ccids))), zeros(Int, S, cnode_count), ones(1, cnode_count) * (γ / S), d_layer)
	push!(d_layer.children, c_layer)
	cids(d_layer) = reshape(cnode_ids, D, length(ccids))

	root
end
