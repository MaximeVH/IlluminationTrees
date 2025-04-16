
## Before proceeding to the genetic programming part, we need to implement our Quality-Diversity metrics and archive.

## prediction output dissimilarity measure for a single sample (1 if different, 0 if same)
function δ_raw(tree1,tree2,features)
    if apply_tree(tree1, features) == apply_tree(tree2, features)
        return 0
    else
        return 1
    end
end

## prediction output diversity measure (output dissimilarity over all samples)
function Δ_raw(tree1, tree2, features)
    total = 0
    for i in axes(features,1)
        total += δ_raw(tree1, tree2, features[i, :])
    end
    return total
end
## Diversity measure for the whole ensemble (average pairwise Δ_raw)
function Δ_raw_bar(trees::Vector{Union{Leaf{Bool}, Node{Float32, Bool}}}, features::Matrix{Float32})
    n = length(trees)
    sum_Δ = 0
    for i in 1:n-1
        for j in i+1:n
            sum_Δ += Δ_raw(trees[i], trees[j], features)
        end
    end
    return sum_Δ / (n * (n - 1))
end

function Δ_raw_bar(trees, features::Matrix{Float32})
    n = length(trees)
    sum_Δ = 0
    for i in 1:n-1
        for j in i+1:n
            sum_Δ += Δ_raw(trees[i], trees[j], features)
        end
    end
    return sum_Δ / (n * (n - 1))
end

#The fitness of a tree is defined here as the accuracy of the predictions it makes on the training data.
function fitness(tree::Union{Leaf{Bool}, Node{Float32, Bool}}, features::Matrix{Float32}, labels)
    predictions = apply_tree(tree, features)
    accuracy = sum(predictions .== labels) / length(labels)
    return accuracy
end

function fitness(tree, features::Matrix{Float32}, labels)
    predictions = apply_tree(tree, features)
    accuracy = sum(predictions .== labels) / length(labels)
    return accuracy
end

# Algorithm to add a new tree to the archive, a kind of novelty search (from https://doi.org/10.1007/978-3-030-72812-0_1).
function add_to_archive!(archive, t_new, features,labels, Δ_min,info=false)
    t_first = archive[1]
    t_second = archive[1]
    for t in archive
        if Δ_raw(t_new,t,features) < Δ_raw(t_new,t_first,features)
            t_second = t_first
            t_first = t
        elseif Δ_raw(t_new,t,features) < Δ_raw(t_new,t_second,features)
            t_second = t
        end
    end
    if Δ_raw(t_new,t_first,features) > Δ_min  # was written wrongly in the original paper
        push!(archive, t_new)
    elseif Δ_raw(t_new,t_second,features) > Δ_min && fitness(t_new,features,labels) > fitness(t_first,features,labels)
        # replace t_first with t_new
        push!(archive, t_new)
        filter!(x -> x != t_first, archive) #removes the closest tree from the archive
        if info
            println("Replaced tree with new tree")
        end
    end
end

function add_to_archive!(archive::Vector{TreeWrapper}, t_new::TreeWrapper, features::Matrix{Float32}, labels, Δ_min::Int) #labels unused here!
    t_first = archive[1]
    t_second = archive[1]

    for t in archive
        Δ_raw1 = Δ_raw(t_new.immutable_tree, t.immutable_tree, features)
        if Δ_raw1 < Δ_raw(t_new.immutable_tree, t_first.immutable_tree, features)
            t_second = t_first
            t_first = t
        elseif Δ_raw1 < Δ_raw(t_new.immutable_tree, t_second.immutable_tree, features)
            t_second = t
        end
    end

    tree_added = false

    if Δ_raw(t_new.immutable_tree, t_first.immutable_tree, features) > Δ_min
        push!(archive, t_new) # should lead to curiosity update
        tree_added = true
    elseif Δ_raw(t_new.immutable_tree, t_second.immutable_tree, features) > Δ_min && t_new.fitness > t_first.fitness
        push!(archive, t_new)  # should lead to curiosity update
        filter!(x -> x != t_first, archive)  
        tree_added = true
    end
    return tree_added
end

## In this approach, a random forest from DecisionTree.jl is used to construct the initial population (i.e., initialize the archive).
function init_archive(Forestmodel,features,labels, Δ_min = 30)
    forest = Forestmodel.trees
    archive = [forest[1],forest[2]]
    for tree in forest[3:end]
        add_to_archive!(archive, tree, features,labels, Δ_min)
    end
    return archive
end

function QD_evolution(Forestmodel, features,labels, generations::Int=100, mutation_rate::Float64=0.1, Δ_min::Int=30)
    # Initialize the archive with the random forest (from DecisionTree.jl)
    archive = init_archive(Forestmodel,features,labels,Δ_min)
    println("Initial archive size: ", length(archive))
    for gen in 1:generations #first attempt without curiosity scores, will be added later
        # random selection of two parents from the archive
        parent_tree1, parent_tree2 = rand(archive, 2) 
        # convert them to mutable trees
        parent_tree1m = to_mutable_tree(parent_tree1)  
        parent_tree2m = to_mutable_tree(parent_tree2)
        # apply genetic operations
        child_tree1, child_tree2 = crossover_height_matched(parent_tree1m, parent_tree2m)
        mutate!(child_tree1, collect(1:size(features, 2)), features, mutation_rate)
        mutate!(child_tree2, collect(1:size(features, 2)), features, mutation_rate)
        # convert back to immutable trees and attempt to add them to the archive
        add_to_archive!(archive, to_decision_tree(child_tree1), features, labels, Δ_min)
        add_to_archive!(archive, to_decision_tree(child_tree2), features, labels, Δ_min)
    end
    return archive
end

#### Once we have evolved our diverse archive of trees, we will sample them to create new DT-based ensemble models.

# We consider three selection strategies:
# 1. Random selection of trees from the archive (diverse ensemble)
# 2. fitness-based selection (accuracy ensemble)
# 3. Hybrid selection (half accuracy-based, half random)
function select_from_archive(archive,ensemble_size,features,labels,type::String="diverse")
    @assert type in ["diverse","accuracy","hybrid"] "Unrecognized ensemble type"
    diverse_ensemble = rand(archive, ensemble_size)
    if type == "diverse"
        return diverse_ensemble
    else
        fitnesses = [fitness(tree, features, labels) for tree in archive]
        sortinds = sortperm(fitnesses, rev=true) # sort in descending order
        accuracy_ensemble = archive[sortinds[1:ensemble_size]]
        if type == "accuracy"
            return accuracy_ensemble
        else  # hybrid selection strategy
            half_index = div(ensemble_size, 2)
            hybrid_ensemble = archive[sortinds[1:half_index]]
            rest = setdiff(1:length(archive), sortinds[1:half_index])
            random_samples = rand(rest, half_index)
            hybrid_ensemble = vcat(hybrid_ensemble, archive[random_samples])
            return hybrid_ensemble
        end
    end
end

## Implement a general (applicable to both random and illumination forests) function that performs inference 
## on a set of features using an ensemble of trees.

# implement majority voting for the ensemble 

function _hist_add!(
    counts::Dict{T,Int}, labels::AbstractVector{T}, region::UnitRange{Int}
) where {T}
    for i in region
        lbl = labels[i]
        counts[lbl] = get(counts, lbl, 0) + 1
    end
    return counts
end

function _hist(
    labels::AbstractVector{T}, region::UnitRange{Int}=1:lastindex(labels)
) where {T}
    _hist_add!(Dict{T,Int}(), labels, region)
end

function majority_vote(labels::AbstractVector)
    if length(labels) == 0
        return nothing
    end
    counts = _hist(labels)
    top_vote = labels[1]
    top_count = -1
    for (k, v) in counts
        if v > top_count
            top_vote = k
            top_count = v
        end
    end
    return top_vote
end

## Apply the ensemble of trees to a set of features and return the predicted majority vote label.
function apply_ensemble(trees, features::Vector{Float32})
    n_trees = length(trees)
    votes = Array{Bool}(undef, n_trees)
    for i in 1:n_trees
        votes[i] = apply_tree(trees[i], features)
    end
        return majority_vote(votes)
end

### Apply the ensemble of trees to the whole training set and return the predicted majority vote labels.
function apply_ensemble(trees, features::Matrix{Float32})
    n_samples = size(features, 1)
    predictions = Array{Bool}(undef, n_samples)
    for i in 1:n_samples
        predictions[i] = apply_ensemble(trees, features[i, :])
    end
    return predictions
end



function convert_forest(forest::Vector{Union{Leaf{Bool}, Node{Float32, Bool}}}, features::Matrix{Float32}, labels)
    return sort([TreeWrapper(tree, features, labels) for tree in forest], by = x -> x.fitness, rev=true)
end

function initialize_archive(Forestmodel,features,labels, Δ_min = 30)
    forest = convert_forest(Forestmodel.trees, features, labels)
    archive = [forest[1],forest[2]]
    for tree in forest[3:end]
        add_to_archive!(archive, tree, features,labels, Δ_min)
    end
    return archive
end



function QD_Evolution(Forestmodel, features,labels;generations::Int=100, mutation_rate=0.1, Δ_min::Int=30)
    # Initialize the archive with the random forest (from DecisionTree.jl)
    archive = initialize_archive(Forestmodel,features,labels,Δ_min)
    featureIDs = collect(1:size(features, 2)) # feature IDs for mutation
    println("Initial archive size: ", length(archive))
    for gen in 1:generations #first attempt without curiosity scores, will be added later
        # random selection of two parents from the archive
        # curiosity_scores = [max(tree.curiosity, 0.5) for tree in archive] 
        # parent_tree1, parent_tree2 = sample(archive, Weights(curiosity_scores), 2)
        weights = Weights(Curiosityview(archive))
        parent_tree1, parent_tree2 = sample(archive, weights, 2)
        # apply genetic operations
        # child_tree1, child_tree2 = crossover_height_matched(parent_tree1, parent_tree2, features, labels) #convert to wrapper after crossover and mutation!
        # mutate!(child_tree1, collect(1:size(features, 2)), features,labels, mutation_rate)
        # mutate!(child_tree2, collect(1:size(features, 2)), features,labels, mutation_rate)
        child1_mutable, child2_mutable = crossover_height_matched(parent_tree1.mutable_tree, parent_tree2.mutable_tree)
        mutate!(child1_mutable, featureIDs, features, mutation_rate)
        mutate!(child2_mutable, featureIDs, features, mutation_rate)
        child1 = TreeWrapper_mutable(child1_mutable, features, labels)
        child2 = TreeWrapper_mutable(child2_mutable, features, labels)
        # Attempt to add them to the archive
        addition1 = add_to_archive!(archive, child1, features, labels, Δ_min)
        addition2 = add_to_archive!(archive, child2, features, labels, Δ_min)

        if addition1 || addition2
            parent_tree1.curiosity += 1.0
            parent_tree2.curiosity += 1.0
        else 
            parent_tree1.curiosity -= 0.5
            parent_tree2.curiosity -= 0.5
        end

    end
    return archive
end

function select_from_archive(archive,ensemble_size,type::String="diverse")
    @assert type in ["diverse","accuracy","hybrid"] "Unrecognized ensemble type"
    diverse_ensemble = rand(archive, ensemble_size)
    if type == "diverse"
        return diverse_ensemble
    else
        fitnesses = [tree.fitness for tree in archive]
        sortinds = sortperm(fitnesses, rev=true) # sort in descending order
        accuracy_ensemble = archive[sortinds[1:ensemble_size]]
        if type == "accuracy"
            return accuracy_ensemble
        else  # hybrid selection strategy
            half_index = div(ensemble_size, 2)
            hybrid_ensemble = archive[sortinds[1:half_index]]
            rest = setdiff(1:length(archive), sortinds[1:half_index])
            random_samples = rand(rest, half_index)
            hybrid_ensemble = vcat(hybrid_ensemble, archive[random_samples])
            return hybrid_ensemble
        end
    end
end

function apply_ensemble(trees::Vector{TreeWrapper}, features::Vector{Float32})
    n_trees = length(trees)
    votes = Array{Bool}(undef, n_trees)
    for i in 1:n_trees
        votes[i] = apply_tree(trees[i].immutable_tree, features)
    end
        return majority_vote(votes)
end

### Apply the ensemble of trees to the whole training set and return the predicted majority vote labels.
function apply_ensemble(trees, features::Matrix{Float32})
    n_samples = size(features, 1)
    predictions = Array{Bool}(undef, n_samples)
    for i in 1:n_samples
        predictions[i] = apply_ensemble(trees, features[i, :])
    end
    return predictions
end

mutable struct hyperparameters
    ensemble_size::Int
    tree_depth::Int
    mutation_rate::Float64
    generations::Int
    threshold::Int
end
function hyperparameters(;ensemble_size::Int, tree_depth::Int, mutation_rate::Float64, generations::Int, threshold::Int)
    return hyperparameters(ensemble_size, tree_depth, mutation_rate, generations, threshold)
end

function evaluate_QD(features,labels,hyperparameters)
    EnsembleSize = hyperparameters.ensemble_size
    TreeDepth = hyperparameters.tree_depth
    MutationRate = hyperparameters.mutation_rate
    Generations = hyperparameters.generations
    Threshold = hyperparameters.threshold

    trainfeat,trainlabels, testfeat, testlabels = train_test_split(features, labels, 0.2)
    Forestmodel = build_forest(trainlabels,trainfeat, 30, EnsembleSize, 0.5, TreeDepth)
    archive =  QD_evolution(Forestmodel, trainfeat, trainlabels, Generations, MutationRate, Threshold) 
    println(typeof(archive))
    acc_ensemble = select_from_archive(archive, EnsembleSize, trainfeat, trainlabels, "accuracy")
    # random_forest = select_from_archive(Forestmodel.trees,EnsembleSize,trainfeat,trainlabels,"accuracy")
    random_forest = convert_forest(Forestmodel.trees, trainfeat, trainlabels)[1:EnsembleSize]
    preds_rf = apply_ensemble(random_forest, testfeat)
    preds_acc = apply_ensemble(acc_ensemble, testfeat)

    return Accuracy(preds_rf, testlabels), Δ_raw_bar(random_forest,testfeat), Accuracy(preds_acc, testlabels),Δ_raw_bar(acc_ensemble, testfeat)
end

function Δ_raw_bar(trees::Vector{TreeWrapper}, features::Matrix{Float32})
    n = length(trees)
    Δ = zeros(Int, n, n)
    for i in 1:n
        for j in 1:n
            if i != j
                Δ[i,j] = Δ_raw(trees[i].immutable_tree, trees[j].immutable_tree, features)
            end
        end
    end
    return sum(triu(Δ,1))/(n*(n-1))
end

function evaluate_QD_CV(features, labels, hyperparameters, Kfolds::Int)
    # Extract hyperparameters
    EnsembleSize = hyperparameters.ensemble_size
    TreeDepth = hyperparameters.tree_depth
    MutationRate = hyperparameters.mutation_rate
    Generations = hyperparameters.generations
    Threshold = hyperparameters.threshold

    # Initialize accumulators for metrics
    RF_accs = Float32[]
    RF_divs = Float32[]
    QD_accs = Float32[]
    QD_divs = Float32[]

    # Generate k-fold splits
    fold_indices = collect(1:size(features, 1))
    fold_size = div(length(fold_indices), Kfolds)
    shuffled_indices = shuffle(fold_indices)

    for k in 1:Kfolds
        # Define train and test indices for the k-th fold
        test_indices = shuffled_indices[(k - 1) * fold_size + 1:min(k * fold_size, length(shuffled_indices))]
        train_indices = setdiff(shuffled_indices, test_indices)

        # Split the data
        trainfeat = features[train_indices, :]
        trainlabels = labels[train_indices]
        testfeat = features[test_indices, :]
        testlabels = labels[test_indices]

        # Train the random forest and perform QD evolution
        Forestmodel = build_forest(trainlabels, trainfeat, 30, EnsembleSize, 0.5, TreeDepth)
        archive = QD_evolution(Forestmodel, trainfeat, trainlabels, Generations, MutationRate, Threshold)

        # Select ensembles
        acc_ensemble = select_from_archive(archive, EnsembleSize, "accuracy")
        random_forest = convert_forest(Forestmodel.trees, trainfeat, trainlabels)[1:EnsembleSize]

        # Make predictions
        preds_rf = apply_ensemble(random_forest, testfeat)
        preds_acc = apply_ensemble(acc_ensemble, testfeat)

        # Compute metrics
        push!(RF_accs, Accuracy(preds_rf, testlabels))
        push!(RF_divs, Δ_raw_bar(random_forest, testfeat))
        push!(QD_accs, Accuracy(preds_acc, testlabels))
        push!(QD_divs, Δ_raw_bar(acc_ensemble, testfeat))
    end

    # Return average metrics across all folds
    return mean(RF_accs), mean(RF_divs), mean(QD_accs), mean(QD_divs)
end