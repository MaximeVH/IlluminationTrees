function annotate_tree_with_labels!(tree, features, labels) 
    # Initialize all MutableLeaf values as empty vectors
    function initialize_leaves!(node)
        if node isa MutableLeaf
            node.values = String[]  # Initialize as an empty String vector
        elseif node isa MutableNode
            initialize_leaves!(node.left)
            initialize_leaves!(node.right)
        end
    end

    # Traverse the tree and assign labels to the appropriate leaf
    function assign_labels!(node, feature, label)
        if node isa MutableLeaf
            push!(node.values, label)  # Add the label to the leaf's values
        elseif node isa MutableNode
            if feature[node.featid] < node.featval
                assign_labels!(node.left, feature, label)
            else
                assign_labels!(node.right, feature, label)
            end
        end
    end

    # Update the majority field of each MutableLeaf
    function update_majority!(node)
        if node isa MutableLeaf
            if !isempty(node.values)
                counts = Dict{String, Int}()
                for value in node.values
                    counts[value] = get(counts, value, 0) + 1
                end
                node.majority = argmax(counts)  # Set the majority class
            end
        elseif node isa MutableNode
            update_majority!(node.left)
            update_majority!(node.right)
        end
    end

    # Step 1: Initialize all leaf values
    initialize_leaves!(tree)

    # Step 2: Assign labels to the appropriate leaves
    for i in 1:size(features, 1)
        assign_labels!(tree, features[i, :], labels[i])
    end

    # Step 3: Update the majority field for all leaves
    update_majority!(tree)
end

PrintTree(random_tree)

annotate_tree_with_labels!(random_tree, features, y)

print_tree(prune_tree(to_decision_tree(random_tree),0.6))


