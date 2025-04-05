using DataFrames, CSV, DecisionTree 

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
            to_mutable_tree5(tree.left, S),
            to_mutable_tree5(tree.right, S)
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