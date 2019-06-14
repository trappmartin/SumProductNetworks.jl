export generate_spn

"""
    generate_spn(X::Matrix, algo::Symbol; ...)

Generate an SPN structure using a structure learning algorithm.

Arguments:
* `X`: Data matrix.
* `algo`: Algorithm, one out of [:learnspn, :random].
"""
function generate_spn(X::Matrix, algo::Symbol; params...)

    if algo == :learnspn
        return learnspn(X; params...)
    elseif algo == :random
        return randomspn(X; params...)
    else
        @error("Unknown structure learning method: ", method)
    end
end

"""
    learnspn(X; distribution=Normal(), minclustersize=100)

Return Sum Product Network learned by a simplified LearnSPN algorithm.
"""
function learnspn(X; distribution=Normal(), minclustersize=100)
    q = Queue{Tuple}()
    root = FiniteSumNode()
    variables = collect(1:size(X)[1])
    instances = collect(1:size(X)[2])
    enqueue!(q, (root, variables, instances))
    
    while length(q) > 0
        node, variables, instances = dequeue!(q)

        # stopping condition, one variable left, estimate distribution
        if length(variables) == 1
            v = variables[1]
            slice = X[v, instances]
            add!(node, UnivariateNode(mle(distribution, X[v, :]), v))
            continue
        end
        
        # stopping condition: too small cluster, factorize variables
        if length(instances) <= minclustersize
            for v in variables
                slice = X[v, instances]
                add!(node, UnivariateNode(mle(distribution, slice), v))
            end
            continue
        end
        
        # divide and conquer
        if typeof(node) <: SumNode
            clusterweights = cluster_instances(X, variables, instances)
            for (cluster, weight) in clusterweights
                prod = FiniteProductNode()
                add!(node, prod, log(weight))
                enqueue!(q, (prod, variables, cluster))
            end
        else  # typeof(node) <: ProductNode
            splits = split_variables(X, variables, instances)
            for split in splits
                if length(split) == 1
                    enqueue!(q, (node, split, instances))
                    continue
                end
                sum = FiniteSumNode()
                add!(node, sum)
                enqueue!(q, (sum, split, instances))
            end
        end
    end
    
    return SumProductNetwork(root)
end

"""
    split_variables(X, variables, instances)

Split variables into two groups by a G-test based method.
"""
function split_variables(X, variables, instances)
    function binarize(x)
        binary_x = zeros(Int, size(x))
        binary_x[x .> mean(x)] .= 1
        return binary_x
    end
    @assert length(variables) > 1
    slice = X[variables, instances]
    distances = zeros(length(variables))
    p = sum(binarize(slice[rand(1:length(variables)), :]))/length(instances)
    for i in 1:length(variables)
        q = sum(binarize(slice[i, :]))/length(instances)
        e = (p + q)/2
        d = evaluate(KLDivergence(), [p, (1 - p), q, (1 - q)], [e, (1 - e), e, (1 - e)])
        distances[i] = d
    end
    dependentindex = partialsortperm(distances, 1:floor(Integer, length(variables)/2))
    splitone = variables[dependentindex]
    splittwo = setdiff(variables, splitone)
    
    return (splitone, splittwo)
end

"""
    cluster_instances(X, variables, instances)

Cluster instances into two groups by k-means clustering.
"""
function cluster_instances(X, variables, instances)
    slice = X[variables, instances]
    results = kmeans(slice, 2)
    clusterone = instances[results.assignments .== 1]
    clustertwo = setdiff(instances, clusterone)
    weight = length(clusterone)/length(instances)    
    
    if length(clustertwo) == 0
        return ((clusterone, weight),)
    end
    return ((clusterone, weight), (clustertwo, 1 - weight))
end

"""
    mle(distribution, X; <keyword arguments>)

Return MLE of `distribution` given `X`.
"""
function mle(distribution::Normal, X; epsilon=0.5)
    μ_hat = mean(X)
    rawstd = std(X)
    σ_hat = isnan(rawstd) ? epsilon : rawstd + epsilon
    return Normal(μ_hat, σ_hat)
end

randomspn(X;params...) = @error("Random structure learning is currently not supported.")
