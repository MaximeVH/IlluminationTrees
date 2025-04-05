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

