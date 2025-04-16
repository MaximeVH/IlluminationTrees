
#### Generate random decision trees for initialisation (rather than using a random forest)
function generate_random_tree(depth::Int, featureIDs::Vector{Int}, features::Matrix{Float32})::AbstractNode
    if depth == 0
        # Create a leaf node with a random majority class
        majority_class = rand([false, true])  # Replace with actual class labels if available
        values = rand([false, true], rand(1:10))  # Random values for the leaf
        return MutableLeaf{Float32, Bool}(majority_class, values)
    else
        # Create an internal node
        featid = rand(featureIDs)  # Randomly select a feature
        featval = rand(features[:, featid])  # Randomly select a threshold from the feature's values

        # Recursively generate left and right subtrees
        left_subtree = generate_random_tree(depth - 1, featureIDs, features)
        right_subtree = generate_random_tree(depth - 1, featureIDs, features)

        return MutableNode{Float32, Bool}(featid, featval, left_subtree, right_subtree)
    end
end


function generate_random_ensemble(num_trees::Int, depth::Int, featureIDs::Vector{Int}, features::Matrix{Float32})
    trees = Vector{AbstractNode}()
    for _ in 1:num_trees
        tree = generate_random_tree(depth, featureIDs, features)
        push!(trees, tree)
    end
    ensemble = to_decision_tree.(trees)
    return prune_tree.(ensemble,0.55)#prune_tree.(ensemble,0.5) #prune the ensemble
end

function initialize_archive_RT(forest,features,labels, Δ_min = 30)
    archive = [forest[1],forest[2]]
    for tree in forest[3:end]
        add_to_archive!(archive, tree, features,labels, Δ_min)
    end
    return archive
end

### Evolve DT's with random trees
function QD_evolution_RT(forest, features,labels; generations::Int=100, mutation_rate=0.1, Δ_min::Int=30)
    # Initialize the archive with the random forest (from DecisionTree.jl)
    archive = initialize_archive_RT(forest,features,labels,Δ_min)
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