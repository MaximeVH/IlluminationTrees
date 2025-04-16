
### Functions
# It might help to normalize the data, such that mutations are on the same scale and have the same impact
function forest_predictions(forest, features)
    # Directly collect predictions into a matrix
    return [apply_tree(tree.immutable_tree, features) .== true for tree in forest]
end

function features_to_forest_outvects(features, labels; num_trees=50, num_features=30, max_depth=4)
    # Build the forest
    Forestmodel = build_forest(labels, features, num_features, num_trees, 0.5, max_depth)
    forest = convert_forest(Forestmodel.trees, features, labels)

    # Collect predictions directly into a matrix of Float32
    forest_preds = forest_predictions(forest, features)

    outvect = Float32.(hcat(forest_preds...))

    return forest, Matrix(outvect')  # Convert to a matrix of Float32
end

function offspring_outvects(offspring_forest,features)
    # Collect predictions directly into a matrix of Float32
    forest_preds = forest_predictions(offspring_forest, features)

    outvect = Float32.(hcat(forest_preds...))

    return Matrix(outvect')  # Convert to a matrix of Float32
end

struct VAE{E,L,M, D}
    encoder_hidden::E
    encoder_logvar::L
    encoder_mu::M
    decoder::D
end

Flux.@functor VAE

function (vae::VAE)(x)
    h = vae.encoder_hidden(x)
    μ = vae.encoder_mu(h)
    logσ² = vae.encoder_logvar(h)
    σ = exp.(0.5 * logσ²)  # ensure this is a float calculation (use . operators for element-wise)
    ϵ = randn(Float32, size(σ))  # generate random noise for reparameterization
    z = μ .+ σ .* ϵ  # reparameterization trick
    x̂ = vae.decoder(z)
    return x̂, μ, logσ²
end

function VAE_archive(X;β=4.0, epochs=50, batchsize=16,hidden_dim = 128)
    N,M = size(X)
    latent_dim = 2
    
    loader = Flux.DataLoader(Matrix(transpose(X)), batchsize=batchsize, shuffle=true);

    # Encoder
    encoder_hidden = Chain(
        Dense(M, hidden_dim, relu),
        Dense(hidden_dim, hidden_dim, relu)
    )
    encoder_mu = Dense(hidden_dim, latent_dim)
    encoder_logvar = Dense(hidden_dim, latent_dim)

    # Decoder
    decoder = Chain(
        Dense(latent_dim, hidden_dim, relu),
        Dense(hidden_dim, M),
        sigmoid
    )


    model = VAE(encoder_hidden,encoder_logvar,encoder_mu, decoder);

    function loss_fn(model,x)
        x̂, μ, logσ² = model(x)
        # Binary cross-entropy loss
        recon = -sum(x .* log.(x̂ .+ 1e-7) + (1 .- x) .* log.(1 .- x̂ .+ 1e-7)) / size(x, 2)
        kl = -0.5 * sum(1 .+ logσ² .- μ.^2 .- exp.(logσ²)) / size(x, 2)
        return recon + β * kl
    end


    optimizer = Flux.Optimise.Adam(0.001)
    ps = Flux.params(model)

    for epoch in 1:epochs
        # epoch_loss = 0.0
        for batch in loader
            x = batch
    
            # Compute loss and gradients
            grads = Flux.gradient(ps) do
                loss_fn(model, x)
            end
    
            # Update model parameters
            Flux.Optimise.update!(optimizer, ps, grads)
        end
        println("Epoch $epoch, Loss: $(loss_fn(model, X[1,:]))")
    end

    function encode_all(m,X)
        zs = []
        for row in eachrow(X)
            h = m.encoder_hidden(row)
            μ = m.encoder_mu(h)
            push!(zs, μ)
        end
        return reduce(hcat, zs)'  # shape: N×2
    end
    # # return encode_all(X)
    function encode(x::AbstractVector)
        h = model.encoder_hidden(x)
        μ = model.encoder_mu(h)
        return μ  # This is a 2D vector (your latent coordinate)
    end
    # return encode_all(X), encode
    return encode_all(model,X), encode
end

function normalize_latents(zs, n_bins)
    mins = minimum(zs, dims=1)
    maxs = maximum(zs, dims=1)
    scale = (maxs .- mins) .+ eps()  # avoid divide-by-zero

    norm_zs = (zs .- mins) ./ scale
    grid_coords = clamp.(floor.(Int, norm_zs .* (n_bins - 1)) .+ 1, 1, n_bins)

    return grid_coords, scale, mins
end

function initialize_archive(forest,grid_coords)
    archive = DefaultDict(() -> nothing)  # position (i, j) => best classifier
    fitness_archive = DefaultDict(() -> -Inf) 

    for i in 1:size(grid_coords, 1)
        coord = Tuple(grid_coords[i, :])
        fitval = forest[i].fitness  # your own fitness function

        if fitval > fitness_archive[coord]
            archive[coord] = forest[i]
            fitness_archive[coord] = fitval
        end
    end
    return archive, fitness_archive
end


function generate_offspring(archive, features, y; n_offspring=10, mutation_rate=0.2)
    offspring = []
    featureIDs = collect(1:size(features, 2))

    for _ in 1:n_offspring
        parent_tree1, parent_tree2 = rand(archive, 2)

        child1_mutable, child2_mutable = crossover_height_matched(parent_tree1[2].mutable_tree, parent_tree2[2].mutable_tree)
        mutate!(child1_mutable, featureIDs, features, mutation_rate)
        mutate!(child2_mutable, featureIDs, features, mutation_rate)

        child1 = TreeWrapper_mutable(child1_mutable, features, y)
        child2 = TreeWrapper_mutable(child2_mutable, features, y)

        push!(offspring, child1)
        push!(offspring, child2)
    end

    return offspring
end


function get_grid_coords(zs::AbstractVector, mins, scale, n_bins)
    zs = zs'  # Ensure zs is a row vector
    norm_zs = (zs .- mins) ./ scale  # Normalize the latent vector
    grid_coords = clamp.(floor.(Int, norm_zs .* (n_bins - 1)) .+ 1, 1, n_bins)  # Map to grid coordinates
    return grid_coords
end

function update_archive!(tree,grid_coords,archive, fitness_archive)
    coord = Tuple(grid_coords)
    fitval = tree.fitness  # your own fitness function

    if fitval > fitness_archive[coord]
        archive[coord] = tree
        fitness_archive[coord] = fitval
    end
end

function generation!(archive,fitness_archive,features,y,enc,mins,scale,n_bins;n_offspring=10, mutation_rate=0.2)
    Offspring = generate_offspring(archive, features, y, n_offspring=n_offspring, mutation_rate=mutation_rate)
    Offspout = offspring_outvects(Offspring,features)
    Zs_ = [enc(Offspout[i, :]) for i in 1:size(Offspout, 1)]
    coords = [get_grid_coords(Zs_[i],mins,scale,n_bins) for i in 1:length(Zs_)]
    for i in 1:length(Offspring)
        update_archive!(Offspring[i], coords[i], archive, fitness_archive)
    end
end

function sample_ensemble(archive,n_trees)
    sampled_trees = rand(archive, n_trees)
    sampled_coords = [tree[1] for tree in sampled_trees]  # Extract the coordinates
    sampled_trees = [tree[2] for tree in sampled_trees] 
    sampled_fitnesses = [fitness_archive[sampled_coords[i]] for i in 1:n_trees] # Extract the mutable trees
    return sampled_trees, sampled_fitnesses
end



function fitness(tree::TreeWrapper, features::Matrix{Float32}, labels)
    predictions = apply_tree(tree.immutable_tree, features)
    accuracy = sum(predictions .== labels) / length(labels)
    return accuracy
end

function select_accurate_ensemble(archive, n_trees)
    # @assert n_trees <= length(archive), "Number of trees requested exceeds available trees in the archive."
    archive = collect(values(archive))
    fitnesses = [tree.fitness for tree in archive]
    sortinds = sortperm(fitnesses, rev=true) # sort in descending order
    accuracy_ensemble = archive[sortinds[1:n_trees]]
    accuracy_ensemble = TreeWrapper[accuracy_ensemble[i] for i in n_trees]
    return accuracy_ensemble
end

select_diverse_ensemble(archive, n_trees) = rand(collect(values(archive)), n_trees)

function select_hybrid_ensemble(archive, n_trees, random_proportion=0.5) 
    n_random_trees = Int(floor(n_trees * random_proportion)) 
    n_accurate_trees = n_trees - n_random_trees
    accuracy_ensemble = select_accurate_ensemble(archive, n_accurate_trees)
    diverse_ensemble = select_diverse_ensemble(archive, n_random_trees)
    hybrid_ensemble = vcat(accuracy_ensemble, diverse_ensemble)
    return hybrid_ensemble
end

function NeuroMapElites(features, labels; n_generations=1000, n_bins=20, n_offspring=10, mutation_rate=0.2, β=4.0, epochs=50, batchsize=16, hidden_dim = 128,num_trees=50, num_features=30, max_depth=4)
    trainfeat, trainlabels, testfeat, testlabels = train_test_split(features, labels, 0.2)

    forest, forestoutvect = features_to_forest_outvects(trainfeat, trainlabels,num_trees=num_trees, num_features=num_features, max_depth=max_depth)
    latents, Encoder = VAE_archive(forestoutvect;β=β, epochs=epochs, batchsize=batchsize, hidden_dim=hidden_dim)
    coords, scale, mins = normalize_latents(latents, n_bins)
    archive, fitness_archive = initialize_archive(forest, coords)
    for i in 1:n_generations
        generation!(archive, fitness_archive, trainfeat, trainlabels, Encoder, mins, scale, n_bins; n_offspring=n_offspring, mutation_rate=mutation_rate)
    end
    return archive, fitness_archive
end

function NME_exp(features, labels; n_generations=1000, n_bins=20, n_offspring=10, mutation_rate=0.2, β=4.0, epochs=50, batchsize=16, hidden_dim = 128,num_trees=50, num_features=30, max_depth=4)
    trainfeat, trainlabels, testfeat, testlabels = train_test_split(features, labels, 0.2)

    forest, forestoutvect = features_to_forest_outvects(trainfeat, trainlabels,num_trees=num_trees, num_features=num_features, max_depth=max_depth)
    latents, Encoder = VAE_archive(forestoutvect;β=β, epochs=epochs, batchsize=batchsize, hidden_dim=hidden_dim)
    coords, scale, mins = normalize_latents(latents, n_bins)
    archive, fitness_archive = initialize_archive(forest, coords)
    

    RF_testacc = sum(apply_ensemble(forest, testfeat) .== testlabels) / length(testlabels)
    RF_trainacc = sum(apply_ensemble(forest, trainfeat) .== trainlabels) / length(trainlabels)

    trainaccs_div = []
    testaccs_div = []
    trainaccs_acc = []
    testaccs_acc = []
    for i in 1:n_generations

        for i in 1:100
            generation!(archive, fitness_archive, trainfeat, trainlabels, Encoder, mins, scale, n_bins; n_offspring=n_offspring, mutation_rate=mutation_rate)
        end
        # Select ensembles
        accuracy_ensemble = select_hybrid_ensemble(archive, min(50,length(archive)))
        diverse_ensemble = select_diverse_ensemble(archive, min(50,length(archive)))
        push!(testaccs_acc, Accuracy(accuracy_ensemble, testfeat, testlabels))
        push!(trainaccs_acc, Accuracy(accuracy_ensemble, trainfeat, trainlabels))
        push!(testaccs_div, Accuracy(diverse_ensemble, testfeat, testlabels))
        push!(trainaccs_div, Accuracy(diverse_ensemble, trainfeat, trainlabels))
    end

    return testaccs_acc,trainaccs_acc,testaccs_div,trainaccs_div, RF_testacc, RF_trainacc
end

function NME_exp_val(features, labels; n_generations=50, n_bins=20, n_offspring=10, mutation_rate=0.2, β=4.0, epochs=50, batchsize=16, hidden_dim = 128,num_trees=50, num_features=30, max_depth=4)
    trainfeat, trainlabels, testfeatures, testy = train_test_split(features, labels, 0.2)
    valfeat, vallabels, testfeat, testlabels = train_test_split(testfeatures, testy, 0.5)

    forest, forestoutvect = features_to_forest_outvects(trainfeat, trainlabels,num_trees=num_trees, num_features=num_features, max_depth=max_depth)
    latents, Encoder = VAE_archive(forestoutvect;β=β, epochs=epochs, batchsize=batchsize, hidden_dim=hidden_dim)
    coords, scale, mins = normalize_latents(latents, n_bins)
    archive, fitness_archive = initialize_archive(forest, coords)
    

    RF_testacc = sum(apply_ensemble(forest, testfeat) .== testlabels) / length(testlabels)
    RF_trainacc = sum(apply_ensemble(forest, trainfeat) .== trainlabels) / length(trainlabels)

    trainaccs_div = []
    testaccs_div = []
    trainaccs_acc = []
    testaccs_acc = []
    for i in 1:n_generations

        for i in 1:100
            generation!(archive, fitness_archive, trainfeat, trainlabels, Encoder, mins, scale, n_bins; n_offspring=n_offspring, mutation_rate=mutation_rate)
        end
        # Select ensembles
        # accuracy_ensemble = select_accurate_ensemble(archive, min(50,length(archive)))
        accuracy_ensemble = select_accurate_ensemble_val(archive, min(50,length(archive)),valfeat, vallabels)
        diverse_ensemble = select_diverse_ensemble(archive, min(50,length(archive)))
        push!(testaccs_acc, Accuracy(accuracy_ensemble, testfeat, testlabels))
        push!(trainaccs_acc, Accuracy(accuracy_ensemble, trainfeat, trainlabels))
        push!(testaccs_div, Accuracy(diverse_ensemble, testfeat, testlabels))
        push!(trainaccs_div, Accuracy(diverse_ensemble, trainfeat, trainlabels))
    end

    return testaccs_acc,trainaccs_acc,testaccs_div,trainaccs_div, RF_testacc, RF_trainacc
end

function train_test_split(features, labels, test_size::Float64=0.2)
    n_samples = size(features, 1)
    indices = randperm(n_samples)
    split_index = Int(floor(n_samples * (1 - test_size)))
    train_indices = indices[1:split_index]
    test_indices = indices[(split_index + 1):end]
    return features[train_indices, :], labels[train_indices], features[test_indices, :], labels[test_indices]
end

function Accuracy(predictions, labels)
    return sum(predictions .== labels) / length(labels)
end

function select_accurate_ensemble_val(archive, n_trees,valfeats, vallabels)
    # @assert n_trees <= length(archive), "Number of trees requested exceeds available trees in the archive."
    archive = collect(values(archive))
    fitnesses = [fitness(archive[i], valfeats, vallabels) for i in 1:length(archive)]#fitnesses recalculated using validation set! [tree.fitness for tree in archive]
    sortinds = sortperm(fitnesses, rev=true) # sort in descending order
    accuracy_ensemble = archive[sortinds[1:n_trees]]
    accuracy_ensemble = TreeWrapper[accuracy_ensemble[i] for i in 1:50]
    return accuracy_ensemble
end

function Accuracy(ensemble, testfeatures, testlabels)
    preds = apply_ensemble(ensemble, testfeatures)
    return sum(preds .== testlabels) / length(testlabels)
end


function get_forest_outvects(forest::Vector{TreeWrapper}, features)

    # Collect predictions directly into a matrix of Float32
    forest_preds = forest_predictions(forest, features)

    outvect = Float32.(hcat(forest_preds...))

    return Matrix(outvect')  # Convert to a matrix of Float32
end

function NeuroMapElitesRandom(features, labels; n_generations=1000, n_bins=20, n_offspring=10, mutation_rate=0.2, β=4.0, epochs=50, batchsize=16, hidden_dim = 128,num_trees=50, num_features=30, max_depth=4)

    randtrees = generate_random_ensemble(100,4,collect(1:size(features, 2)), features)
    randtrees = Treewrap_random_ensemble(randtrees,features,labels)

    forestoutvect = features_to_forest_outvects(randtrees, features)
    latents, Encoder = VAE_archive(forestoutvect;β=β, epochs=epochs, batchsize=batchsize, hidden_dim=hidden_dim)
    coords, scale, mins = normalize_latents(latents, n_bins)
    archive, fitness_archive = initialize_archive(randtrees, coords)
    for i in 1:n_generations
        generation!(archive, fitness_archive, features, labels, Encoder, mins, scale, n_bins; n_offspring=n_offspring, mutation_rate=mutation_rate)
    end
    return archive, fitness_archive
end

function select_accurate_ensemble_random(archive, n_trees, features, y)
    # @assert n_trees <= length(archive), "Number of trees requested exceeds available trees in the archive."
    archive = collect(values(archive))
    fitnesses = [fitness(archive[i], features, y) for i in 1:length(archive)]#fitnesses recalculated using validation set! [tree.fitness for tree in archive]
    sortinds = sortperm(fitnesses, rev=true) # sort in descending order
    accuracy_ensemble = archive[sortinds[1:n_trees]]
    accuracy_ensemble = TreeWrapper[accuracy_ensemble[i] for i in 1:n_trees]
    return accuracy_ensemble
end

Treewrap_random_ensemble(trees,features,labels) = [TreeWrapper(tree,features,labels) for tree in trees]

function NeuroMapElitesRT(features,labels; n_generations=1000, n_bins=20, n_offspring=10, mutation_rate=0.2, β=4.0, epochs=50, batchsize=16, hidden_dim = 128,num_trees=50, num_features=30, max_depth=4)

    randtrees = generate_random_ensemble(num_trees,max_depth,collect(1:size(features, 2)), features)
    randarch = QD_evolution_RT(randtrees, features,labels,generations=200,mutation_rate=0.1, Δ_min=30)
    println("$(length(randarch)) diverse trees generated")
    randtrees = Treewrap_random_ensemble(randarch,features,labels)
    forestoutvect = get_forest_outvects(randtrees, features)
    latents, Encoder = VAE_archive(forestoutvect;β=β, epochs=epochs, batchsize=batchsize, hidden_dim=hidden_dim)
    println("$(length(latents)) latent vectors generated")
    coords, scale, mins = normalize_latents(latents, n_bins)
    archive, fitness_archive = initialize_archive(randtrees, coords)
    for i in 1:n_generations
        generation!(archive, fitness_archive, features, labels, Encoder, mins, scale, n_bins; n_offspring=n_offspring, mutation_rate=mutation_rate)
    end
    return archive, fitness_archive
end


function NeuroMapElitesRT_exp(features,labels; n_generations=100, n_bins=20, n_offspring=10, mutation_rate=0.2, β=4.0, epochs=50, batchsize=16, hidden_dim = 128,num_trees=50, num_features=30, max_depth=4)

    trainfeatures, trainlabs, testfeatures, testlabs = train_test_split(features, labels, 0.2)

    forest, forestoutvect = features_to_forest_outvects(trainfeatures, trainlabs,num_trees=50, num_features=num_features, max_depth=max_depth)
    RF_testacc = sum(apply_ensemble(forest, testfeatures) .== testlabs) / length(testlabs)
    RF_trainacc = sum(apply_ensemble(forest, trainfeatures) .== trainlabs) / length(trainlabs)
    RF_div = get_ensemble_diversity(forest, testfeatures)

    randtrees = generate_random_ensemble(num_trees,max_depth,collect(1:size(trainfeatures, 2)), trainfeatures)
    randarch = QD_evolution_RT(randtrees, trainfeatures,trainlabs,generations=300,mutation_rate=0.05, Δ_min=30)
    println("$(length(randarch)) diverse trees generated")
    randtrees = Treewrap_random_ensemble(randarch,trainfeatures,trainlabs)
    forestoutvect = get_forest_outvects(randtrees, trainfeatures)
    latents, Encoder = VAE_archive(forestoutvect;β=β, epochs=epochs, batchsize=batchsize, hidden_dim=hidden_dim)
    println("$(length(latents)) latent vectors generated")
    coords, scale, mins = normalize_latents(latents, n_bins)
    archive, fitness_archive = initialize_archive(randtrees, coords)

    trainaccs_acc = []
    testaccs_acc = []
    divs = []
    for i in 1:n_generations
        for i in 1:100
            generation!(archive, fitness_archive, trainfeatures, trainlabs, Encoder, mins, scale, n_bins; n_offspring=n_offspring, mutation_rate=mutation_rate)
        end
        # Select ensembles
        # accuracy_ensemble = select_accurate_ensemble(archive, min(50,length(archive)))
        accuracy_ensemble = select_accurate_ensemble_random(archive,  min(50,length(archive)), trainfeatures, trainlabs)
        push!(testaccs_acc, Accuracy(accuracy_ensemble, testfeatures, testlabs))
        push!(trainaccs_acc, Accuracy(accuracy_ensemble, trainfeatures, trainlabs))
        push!(divs, get_ensemble_diversity(accuracy_ensemble, testfeatures))
    end
    archHM = plot_archive(archive, fitness_archive, n_bins)
    return archHM, testaccs_acc,trainaccs_acc,divs, RF_testacc, RF_trainacc, RF_div
end

function get_ensemble_diversity(ensemble, features)
    treez = [ensemble[i].immutable_tree for i in 1:length(ensemble)]
    return Δ_raw_bar(treez,features)
end