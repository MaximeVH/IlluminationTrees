display_df(df) = VSCodeServer.vscodedisplay(df)

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

## Define overall tree structure (with fitness and curiosity values)

mutable struct TreeWrapper
    immutable_tree::Node{Float32, Bool}  # Immutable DecisionTree.jl tree
    mutable_tree::AbstractNode{Float32, Bool}          # Mutable tree for genetic operations
    fitness::Float32                           # Cached fitness value
    curiosity::Float32                        # Curiosity value for exploration
end


import Base.show
import Base.length
import DecisionTree.depth

# show(mutable_tree)
length(leaf::MutableLeaf) = 1
length(tree::MutableNode) = length(tree.left) + length(tree.right)
depth(leaf::MutableLeaf) = 0
depth(tree::MutableNode) = 1 + max(depth(tree.left), depth(tree.right))

function Base.show(io::IO, tree::MutableNode{Float32, Bool})
    println(io, "Mutable Decision Tree")
    println(io, "Leaves: $(length(tree))")
    print(io, "Depth:  $(depth(tree))")
end

### A structure to hold the trees and their curiosity values
struct Curiosityview <: AbstractVector{Float32}
    trees::Vector{TreeWrapper}
end

Base.size(cv::Curiosityview) = (length(cv.trees),)
Base.getindex(cv::Curiosityview, i::Int) = max(cv.trees[i].curiosity, 0.5)
Base.IndexStyle(::Type{<:Curiosityview}) = IndexLinear()

### DecisionTree.jl uses immutable trees, wereas we need mutable trees for genetic programming.
### We will create a mutable version of the decision tree in order to apply genetic operations.

# Convert DecisionTree.jl's trees to mutable Tree
function to_mutable_tree(tree::Union{Leaf, Node}, S = Nothing)
    if tree isa Leaf
        T = typeof(tree.majority)
        return MutableLeaf{S, T}(tree.majority, tree.values)
    elseif tree isa Node
        S = typeof(tree.featval)
        T = Bool #get_leaf_type(tree)
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

## create a TreeWrapper from an immutable tree and features/labels
function TreeWrapper(immutable_tree::Union{Node{S, T}, Leaf{T}}, features::Matrix{S}, labels) where {S, T}
    mutable_tree = to_mutable_tree(immutable_tree)
    fit = fitness(immutable_tree, features, labels)
    return TreeWrapper(immutable_tree, mutable_tree, fit, 0.0)
end

function TreeWrapper_mutable(mutable_tree::MutableNode{Float32, Bool}, features::Matrix{Float32}, labels) 
    immutable_tree = to_decision_tree(mutable_tree)
    fit = fitness(immutable_tree, features, labels)
    return TreeWrapper(immutable_tree, mutable_tree, fit, 0.0)
end

function TreeWrapper_mutable(mutable_tree, features::Matrix{Float32}, labels) 
    immutable_tree = to_decision_tree(mutable_tree)
    fit = fitness(immutable_tree, features, labels)
    return TreeWrapper(immutable_tree, mutable_tree, fit, 0.0)
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

## create a TreeWrapper from a mutable tree and features/labels
function TreeWrapper_mutable(mutable_tree::MutableNode{Float32, Bool}, features::Matrix{Float32}, labels) 
    immutable_tree = to_decision_tree(mutable_tree)
    fit = fitness(immutable_tree, features, labels)
    return TreeWrapper(immutable_tree, mutable_tree, fit, 0.0)
end

# function get_leaf_type(tree::Union{Leaf, Node}) # deprecated for now.
#     if tree isa Leaf
#         return typeof(tree.majority)
#     elseif tree isa Node
#         # Recurse until we find a leaf to infer T
#         return get_leaf_type(tree.left)
#     else
#         error("Invalid tree structure")
#     end
# end

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

import DecisionTree.apply_tree

apply_tree(tree::TreeWrapper, feats::Vector{Float32}) = apply_tree(tree.immutable_tree, feats)
apply_tree(tree::TreeWrapper, feats::Matrix{Float32}) = apply_tree(tree.immutable_tree, feats)
