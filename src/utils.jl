function generate_bloobs(n_features = 2, n_centers = 3, n_samples = 100)
    centers = @parallel (hcat) for i = 1:n_features
        rand(Uniform(-10, 10), n_centers)
    end

    cluster_std = @parallel (hcat) for i = 1:n_features
      rand(Uniform(0.5, 4), n_centers)
    end

    n_samples_per_center = ones(Integer, n_centers) * int(n_samples / n_centers)

    for i = 1:(n_centers % n_samples)
        n_samples_per_center[i] += 1
    end

    X = @parallel (hcat) for i = 1:n_centers
      rand(MvNormal(ones(n_features) .* centers[i], eye(n_features) .* cluster_std[i]), n_samples_per_center[i])
    end

    X = X';

    Y = @parallel (vcat) for i = 1:n_centers
      y = ones(n_samples_per_center[i]) * i
    end

    ids = [1:size(X)[1]]
    shuffle!(ids)

    X = X[ids,:]
    Y = Y[ids];

    return (X, Y)
end
