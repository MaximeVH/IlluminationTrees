using DataFrames, CSV, DecisionTree 
using LinearAlgebra
using Plots
using Random  
#function to display DataFrames in VSCode
display_df(df) = VSCodeServer.vscodedisplay(df)

# load data
BC_data = CSV.read("WisconsinBreastCancer.csv", DataFrame)

# define the features and target variable
y = BC_data[!, :diagnosis]
X = select(BC_data, Not(:diagnosis,:id,:Column33))
features = Matrix(X)

# Build a random forest using all 30 features, max depth of 4, and 50 trees

Forestmodel = build_forest(y,features, 30, 50, 0.5, 4)


### DecisionTree.jl uses immutable trees, wereas we need mutable trees for genetic programming.
### We will create a mutable version of the decision tree in order to apply genetic operations.

# Define mutable structs and types
abstract type AbstractNode{S, T} end

mutable struct MutableLeaf{S, T} <: AbstractNode{S, T}
    majority :: T
    values   :: Vector{T}
end
mutable struct MutableNode{S, T} <: AbstractNode{S, T}
    featid::Int64
    featval::S
    left::AbstractNode{S, T}
    right::AbstractNode{S, T}
end

# Convert DecisionTree.jl's trees to mutable TreeSplit

function get_leaf_type(tree::Union{Leaf, Node})
    if tree isa Leaf
        return typeof(tree.majority)
    elseif tree isa Node
        # Recurse until we find a leaf to infer T
        return get_leaf_type(tree.left)
    else
        error("Invalid tree structure")
    end
end

function to_mutable_tree(tree::Union{Leaf, Node}, S = Nothing)
    if tree isa Leaf
        T = typeof(tree.majority)
        return MutableLeaf{S, T}(tree.majority, tree.values)
    elseif tree isa Node
        S = typeof(tree.featval)
        T = get_leaf_type(tree)
        return MutableNode{S, T}(
            tree.featid,
            tree.featval,
            to_mutable_tree(tree.left, S),
            to_mutable_tree(tree.right, S)
        )
    else
        error("Unknown tree type: $(typeof(tree))")
    end
end

function deepcopy_tree(tree::AbstractNode{S,T}) where {S, T} 
    if tree isa MutableLeaf
        return MutableLeaf{S,T}(tree.majority, tree.values)
    else
        return MutableNode{S, T}(tree.featid, tree.featval,
                          deepcopy_tree(tree.left),
                          deepcopy_tree(tree.right))
    end
end

# For inference and prediction, we need to convert the mutable tree back to the immutable form.
function to_decision_tree(tree::AbstractNode{S, T}) where {S, T}
    if tree isa MutableLeaf
        # Convert MutableLeaf to Leaf
        return Leaf(tree.majority, tree.values)
    elseif tree isa MutableNode
        # Convert MutableNode to Node
        return Node(
            tree.featid,
            tree.featval,
            to_decision_tree(tree.left),
            to_decision_tree(tree.right)
        )
    else
        error("Unknown node type: $(typeof(tree))")
    end
end

# Now that we have mutable trees, we can implement the crossover and mutation functions.
# Prerequisite utility functions for node replacement and tree height computation are necessary.

function replace_node!(tree::AbstractNode{S,T}, target::AbstractNode{S,T}, new_subtree::AbstractNode{S,T}) where {S, T}
    if tree === target
        tree = new_subtree
        return new_subtree
    elseif tree isa MutableNode
        if tree.left === target
            tree.left = new_subtree
        else
            replace_node!(tree.left, target, new_subtree)
        end

        if tree.right === target
            tree.right = new_subtree
        else
            replace_node!(tree.right, target, new_subtree)
        end
    end
    return tree
end

function compute_heights(tree::AbstractNode{S,T}) where {S, T}
    heights = Dict{AbstractNode{S,T}, Int}()
    _compute_heights!(tree, heights)
    return heights
end
function _compute_heights!(tree::AbstractNode{S,T}, heights::Dict{AbstractNode{S,T}, Int}) where {S, T}
    if tree isa MutableLeaf
        heights[tree] = 0
        return 0
    else
        left_h = _compute_heights!(tree.left, heights)
        right_h = _compute_heights!(tree.right, heights)
        h = 1 + max(left_h, right_h)
        heights[tree] = h
        return h
    end
end

# We want to implement a crossover function that respects the height of the trees. This way,
# subtrees are swapped while maintaining the maximum depth of the trees.

function crossover_height_matched(t1::AbstractNode{S,T}, t2::AbstractNode{S,T}) where {S, T}
    t1_copy = deepcopy_tree(t1)
    t2_copy = deepcopy_tree(t2)

    # Get node heights
    h1 = compute_heights(t1_copy)
    h2 = compute_heights(t2_copy)

    # Find common heights
    heights1 = Set(values(h1))
    heights2 = Set(values(h2))
    common_heights = intersect(heights1, heights2)

    if isempty(common_heights)
        println("No common height levels found — skipping crossover.")
        return t1_copy, t2_copy
    end

    # Choose a common height
    selected_height = rand(collect(common_heights))

    # Get nodes at that height
    nodes1 = [node for (node, h) in h1 if h == selected_height]
    nodes2 = [node for (node, h) in h2 if h == selected_height]

    subtree1 = rand(nodes1)
    subtree2 = rand(nodes2)

    # Replace subtrees
    replace_node!(t1_copy, subtree1, deepcopy_tree(subtree2))
    replace_node!(t2_copy, subtree2, deepcopy_tree(subtree1))

    return t1_copy, t2_copy
end

## As for mutation, we consider three types of mutations (similar to Boisvert et al. 2021):
# 1. Change the feature used for splitting (random sample of available features)
# 2. Change the threshold value for splitting (By adding a normally distributed random number)
# 3. Sample a new threshold value from the data (randomly select a value from the training data)

# before we can implement them, we need a prerequisite function to sample tree nodes.
function collect_internal_nodes(tree::AbstractNode)
    nodes = MutableNode[]  # empty vector to collect nodes
    _collect_internal_nodes!(tree, nodes)
    return nodes
end
function _collect_internal_nodes!(node::AbstractNode, nodes::Vector{MutableNode})
    if node isa MutableNode
        push!(nodes, node)
        _collect_internal_nodes!(node.left, nodes)
        _collect_internal_nodes!(node.right, nodes)
    end
    # do nothing for leaves
end

# The three mutation functions 
function mutate_feature!(tree::AbstractNode, all_features::Vector{Int})
    nodes = collect_internal_nodes(tree)
    if isempty(nodes)
        return
    end
    node = rand(nodes)
    node.featid = rand(all_features)
end

function mutate_threshold_noise!(tree::AbstractNode, sigma::Float64 = 0.1)
    nodes = collect_internal_nodes(tree)
    if isempty(nodes)
        return
    end
    node = rand(nodes)
    node.featval += randn() * sigma
end

function mutate_threshold_resample!(tree::AbstractNode, features::Matrix{Float64})
    nodes = collect_internal_nodes(tree)
    if isempty(nodes)
        return
    end
    node = rand(nodes)
    values = features[:,node.featid]
    node.featval = rand(values)
end

# now we collect them together into a single function, including mutation rate (probability).
function mutate!(tree,featureIDs,features, mutation_rate=0.1)
    if rand() < mutation_rate
        mutate_feature!(tree, featureIDs)
    end
    if rand() < mutation_rate
        mutate_threshold_resample!(tree, features)
    else mutation_rate
        mutate_threshold_noise!(tree)
    end
end


## Tree visualization (mutable tree version)

function PrintTree(tree::AbstractNode{S, T}, prefix::String = "", is_left::Bool = true) where {S, T}
    if tree isa MutableLeaf
        # Count the number of values equal to the majority class
        majority_count = count(x -> x == tree.majority, tree.values)
        # Print leaf node information
        println(prefix, (is_left ? "├─ " : "└─ "), tree.majority, " : ", majority_count, "/", length(tree.values))
    elseif tree isa MutableNode
        # Print internal node information
        println(prefix, (is_left ? "├─ " : "└─ "), "Feature ", tree.featid, " < ", tree.featval, " ?")

        # Prepare the prefix for child nodes
        new_prefix = prefix * (is_left ? "│   " : "    ")

        # Recursively print the left and right subtrees
        PrintTree(tree.left, new_prefix, true)
        PrintTree(tree.right, new_prefix, false)
    else
        error("Unknown node type: $(typeof(tree))")
    end
end

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
function Δ_raw(tree1,tree2,features)
    sum([δ_raw(tree1,tree2,features[i,:]) for i in 1:size(features,1)])
end

## Diversity measure for the whole ensemble (average pairwise Δ_raw)
function Δ_raw_bar(trees::Vector{Union{Leaf{String1}, Node{Float64, String1}}}, features::Matrix{Float64})
    n = length(trees)
    Δ = zeros(Int, n, n)
    for i in 1:n
        for j in 1:n
            if i != j
                Δ[i,j] = Δ_raw(trees[i], trees[j], features)
            end
        end
    end
    return sum(triu(Δ,1))/(n*(n-1))
end

function Δ_raw_bar(trees, features::Matrix{Float64})
    n = length(trees)
    Δ = zeros(Int, n, n)
    for i in 1:n
        for j in 1:n
            if i != j
                Δ[i,j] = Δ_raw(trees[i], trees[j], features)
            end
        end
    end
    return sum(triu(Δ,1))/(n*(n-1))
end

#The fitness of a tree is defined here as the accuracy of the predictions it makes on the training data.
function fitness(tree::Union{Leaf{String1}, Node{Float64, String1}}, features::Matrix{Float64}, labels)
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


## In this approach, a random forest from DecisionTree.jl is used to construct the initial population (i.e., initialize the archive).

function initialize_archive(Forestmodel,features,labels, Δ_min = 30)
    forest = Forestmodel.trees
    archive = [forest[1],forest[2]]
    for tree in forest[3:end]
        add_to_archive!(archive, tree, features,labels, Δ_min)
    end
    return archive
end

# Now that we've got our archive, our trees, and our genetic operations, we can implement the main loop of the algorithm.

function QD_evolution(Forestmodel, features,labels, generations::Int=100, mutation_rate::Float64=0.1, Δ_min::Int=30)
    # Initialize the archive with the random forest (from DecisionTree.jl)
    archive = initialize_archive(Forestmodel,features,labels,Δ_min)
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
function apply_ensemble(trees, features::Vector{Float64})
    n_trees = length(trees)
    votes = Array{String1}(undef, n_trees)
    for i in 1:n_trees
        votes[i] = apply_tree(trees[i], features)
    end
        return majority_vote(votes)
end

### Apply the ensemble of trees to the whole training set and return the predicted majority vote labels.
function apply_ensemble(trees, features::Matrix{Float64})
    n_samples = size(features, 1)
    predictions = Array{String1}(undef, n_samples)
    for i in 1:n_samples
        predictions[i] = apply_ensemble(trees, features[i, :])
    end
    return predictions
end

