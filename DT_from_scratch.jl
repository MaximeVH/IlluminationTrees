using AbstractTrees, Random, CSV, DataFrames

X = rand(100, 5)  # 100 samples, 5 features
Y = rand(0:1, 100)  # Binary labels


# gini index function for binary classification
function gini_index(y::Vector{Int})
    n = length(y)
    if n == 0
        return 0.0
    end
    p = sum(y) / n
    return 1 - (p^2 + (1 - p)^2)
end


# # function to split a node 
# function initial_split(X::AbstractMatrix, y::Vector{Int})
#     best_gini = Inf
#     best_feature = -1
#     best_threshold = -1
#     best_left_indices = Int[]  
#     best_right_indices = Int[]
#     n_features = size(X, 2)

#     for feature in 1:n_features
#         thresholds = unique(X[:, feature]) # this can be improved (by taking the mean of two consecutive values)
#         for threshold in thresholds
#             left_indices = findall(X[:, feature] .<= threshold)
#             right_indices = findall(X[:, feature] .> threshold)
            
#             if isempty(left_indices) || isempty(right_indices)
#                 continue
#             end

#             left_gini = gini_index(y[left_indices])
#             right_gini = gini_index(y[right_indices])
#             weighted_gini = (length(left_indices) * left_gini + length(right_indices) * right_gini) / length(y)

#             if weighted_gini < best_gini
#                 best_gini = weighted_gini
#                 best_feature = feature
#                 best_threshold = threshold
#                 best_left_indices = left_indices
#                 best_right_indices = right_indices
#             end
#         end
#     end

#     left_indices = findall(X[:, best_feature] .<= best_threshold)
#     right_indices = findall(X[:, best_feature] .> best_threshold)
#     return X[left_indices, :], y[left_indices], X[right_indices, :], y[right_indices]
# end


# Xinitleft, yinitleft, Xinitright, yinitright = initial_split(X, Y)

# gini_index(Y)

# gini_index(yinitleft)
# gini_index(yinitright)

mutable struct TreeNode
    feature::Union{Int, Nothing}
    threshold::Float64
    samples::Vector{Int}
    index::Int
    parent::Union{TreeNode, Nothing}
    left::Union{TreeNode, Nothing}
    right::Union{TreeNode, Nothing}
    impurity::Union{Float64, Nothing}
end

function TreeNode(; feature=nothing, threshold=-1.0, samples=Int[], index=1,
    parent=nothing, left=nothing, right=nothing, impurity=nothing)
return TreeNode(feature, threshold, samples, index, parent, left, right, impurity)
end

function RootNode(X::AbstractMatrix, y::Vector{Int})
    root = TreeNode(
        feature=nothing,
        threshold=-1,
        samples=collect(1:size(X, 1)),  # Indices of all samples
        index=1,
        parent=nothing,
        left=nothing,
        right=nothing,
        impurity=gini_index(y)  # Initial impurity of the root node
    )
    return root
end

# function TreeSplit!(Treenode::TreeNode,X::AbstractMatrix, y::Vector{Int})
#     best_gini = Inf
#     best_feature = -1
#     best_threshold = -1
#     best_left_indices = Int[]  
#     best_right_indices = Int[]
#     n_features = size(X, 2)

#     for feature in 1:n_features
#         thresholds = unique(X[:, feature]) # this can be improved (by taking the mean of two consecutive values)
#         for threshold in thresholds
#             left_indices = findall(X[:, feature] .<= threshold)
#             right_indices = findall(X[:, feature] .> threshold)
            
#             if isempty(left_indices) || isempty(right_indices)
#                 continue
#             end

#             left_gini = gini_index(y[left_indices])
#             right_gini = gini_index(y[right_indices])
#             weighted_gini = (length(left_indices) * left_gini + length(right_indices) * right_gini) / length(y)

#             if weighted_gini < best_gini
#                 best_gini = weighted_gini
#                 best_feature = feature
#                 best_threshold = threshold
#                 best_left_indices = left_indices
#                 best_right_indices = right_indices
#             end
#         end
#     end

#     left_indices = findall(X[:, best_feature] .<= best_threshold)
#     right_indices = findall(X[:, best_feature] .> best_threshold)

#     leftchild = TreeNode(feature=best_feature, threshold=best_threshold, samples=left_indices, index=2*TreeNode.index, parent=Treenode, left=nothing, right=nothing, impurity=gini_index(y[left_indices]))
#     # return X[left_indices, :], y[left_indices], X[right_indices, :], y[right_indices]
#     return 
# end



function TreeSplit!(Treenode::TreeNode, X::AbstractMatrix, y::Vector{Int})
    best_gini = Inf
    best_feature = -1
    best_threshold = -1
    best_left_indices = Int[]
    best_right_indices = Int[]
    n_features = size(X, 2)

    # Get the indices of the samples in the current node
    current_indices = Treenode.samples

    # Find the best split
    for feature in 1:n_features
        thresholds = unique(X[current_indices, feature])  # Use only the samples in the current node
        for threshold in thresholds
            left_indices = current_indices[findall(X[current_indices, feature] .<= threshold)]
            right_indices = current_indices[findall(X[current_indices, feature] .> threshold)]

            if isempty(left_indices) || isempty(right_indices)
                continue
            end

            left_gini = gini_index(y[left_indices])
            right_gini = gini_index(y[right_indices])
            weighted_gini = (length(left_indices) * left_gini + length(right_indices) * right_gini) / length(current_indices)

            if weighted_gini < best_gini
                best_gini = weighted_gini
                best_feature = feature
                best_threshold = threshold
                best_left_indices = left_indices
                best_right_indices = right_indices
            end
        end
    end

    # Update the current node with the best split
    Treenode.feature = best_feature
    Treenode.threshold = best_threshold

    # Create left and right child nodes
    left_child = TreeNode(
        feature=best_feature,
        threshold=best_threshold,
        samples=best_left_indices,  # Indices with respect to the original feature matrix
        index=2 * Treenode.index,
        parent=Treenode,
        left=nothing,
        right=nothing,
        impurity=gini_index(y[best_left_indices])
    )

    right_child = TreeNode(
        feature=best_feature,
        threshold=best_threshold,
        samples=best_right_indices,  # Indices with respect to the original feature matrix
        index=2 * Treenode.index + 1,
        parent=Treenode,
        left=nothing,
        right=nothing,
        impurity=gini_index(y[best_right_indices])
    )

    # Update the current node's children
    Treenode.left = left_child
    Treenode.right = right_child
end

Tree = RootNode(X, Y)


TreeSplit!(Tree, X, Y)
Tree.left


function BuildTree(X::AbstractMatrix, Y::Vector{Int})
    # Initialize the tree with the root node
    root = RootNode(X, Y)

    # Define a queue for breadth-first traversal
    queue = [root]

    # Iteratively split nodes
    while !isempty(queue)
        # Get the next node to process
        current_node = popfirst!(queue)

        # Stop splitting if the node meets the stopping criteria
        if current_node.impurity == 0.0 || length(current_node.samples) < 10
            continue
        end

        # Apply TreeSplit! to split the current node
        TreeSplit!(current_node, X, Y)

        # Add the left and right child nodes to the queue if they exist
        if current_node.left !== nothing
            push!(queue, current_node.left)
        end
        if current_node.right !== nothing
            push!(queue, current_node.right)
        end
    end

    return root
end

Tree = BuildTree(X, Y)


function predict(tree::TreeNode, x::AbstractVector{Float64})
    current_node = tree

    while current_node.left !== nothing || current_node.right !== nothing
        if x[current_node.feature] <= current_node.threshold
            current_node = current_node.left
        else
            current_node = current_node.right
        end
    end

    return current_node.impurity  # Return the impurity of the leaf node as the prediction
end
function predict_tree(tree::TreeNode, X::AbstractMatrix{Float64})
    predictions = Float64[]
    for i in 1:size(X, 1)
        push!(predictions, predict(tree, X[i, :]))
    end
    return predictions
end