using Plots, MultivariateStats, Statistics
include("QDTrees.jl")

### Plotting the accuracy vs diversity of the ensembles created by the QD algorithm and the random forest for different ensemble sizes.
function acc_vs_diversity(trainfeat,trainlabels, testfeatures, testlabels)
    ensemble_sizes = [5,10, 20, 30, 50]
    acc_accuracies, div_accuracies, hybrid_accuracies, rf_accuracies = [], [], [], []
    acc_diversities,div_diversities, hybrid_diversities, rf_diversities = [], [], [], []
    for en_size in ensemble_sizes
        forestmodel = build_forest(trainlabels, trainfeat, 30, en_size, 0.5, 4)
        arch = QD_evolution(forestmodel, trainfeat, trainlabels, 200, 0.05, 30)
        acc_ensemble, div_ensemble, hybrid_ensemble, random_forest = get_QD_ensembles(forestmodel,arch,trainfeat,trainlabels, en_size)
        push!(hybrid_accuracies,Accuracy(apply_ensemble(hybrid_ensemble, testfeatures), testlabels))
        push!(div_accuracies,Accuracy(apply_ensemble(div_ensemble, testfeatures), testlabels))
        push!(acc_accuracies,Accuracy(apply_ensemble(acc_ensemble, testfeatures), testlabels))
        push!(rf_accuracies,Accuracy(apply_ensemble(random_forest, testfeatures), testlabels))

        push!(hybrid_diversities, Δ_raw_bar(hybrid_ensemble, testfeatures))
        push!(div_diversities, Δ_raw_bar(div_ensemble, testfeatures))
        push!(acc_diversities, Δ_raw_bar(acc_ensemble, testfeatures))
        push!(rf_diversities, Δ_raw_bar(random_forest, testfeatures))
    end
    return ensemble_sizes, acc_accuracies, div_accuracies, hybrid_accuracies, rf_accuracies, acc_diversities, div_diversities, hybrid_diversities, rf_diversities
end

function acc_vs_diversity_plot(trainfeat,trainlabels, testfeatures, testlabels)
    ensemble_sizes, acc_accuracies, div_accuracies, hybrid_accuracies, rf_accuracies, acc_diversities, div_diversities, hybrid_diversities, rf_diversities =acc_vs_diversity(trainfeat,trainlabels, testfeatures, testlabels)

    p = plot()
    p_twin = plot()

    plot!(p,ensemble_sizes, acc_accuracies, label="Accuracy Ensemble", xlabel="Ensemble Size", ylabel="Accuracy",legend=:outertop,title = "Accuracy vs Ensemble Size")
    plot!(p,ensemble_sizes, div_accuracies, label="Diverse Ensemble")
    plot!(p,ensemble_sizes, hybrid_accuracies, label="Hybrid Ensemble")
    plot!(p,ensemble_sizes, rf_accuracies, label="Random Forest")


    plot!(p_twin,ensemble_sizes, acc_diversities, label="Accuracy Ensemble", ylabel="Diversity",linestyle=:dash, xlabel="Ensemble Size",title = "Diversity vs Ensemble Size",  legend=:outertop)
    plot!(p_twin,ensemble_sizes, div_diversities, label="Diverse Ensemble",linestyle=:dash)
    plot!(p_twin,ensemble_sizes, hybrid_diversities, label="Hybrid Ensemble",linestyle=:dash)
    plot!(p_twin,ensemble_sizes, rf_diversities, label="Random Forest",linestyle=:dash)

    p_overall = plot(p, p_twin, layout=(1, 2), size = (1200, 600))

    return p_overall
end

### A heatmap of the diversity matrix of the ensembles created by the QD algorithm and the random forest.

function DiversityMatrix(trees, features::Matrix{Float64})
    n = length(trees)
    Δ = zeros(Int, n, n)
    for i in 1:n
        for j in 1:n
            if i != j
                Δ[i,j] = Δ_raw(trees[i], trees[j], features)
            end
        end
    end
    return Δ
end

function ensemble_disagreement_heatmap(forestmodel,archive,testfeat)
    ensemble_size = length(forestmodel.trees)
    acc_ensemble = select_from_archive(archive, ensemble_size, testfeat, testlabels, "accuracy")
    divmat_acc = DiversityMatrix(acc_ensemble, testfeat)
    rf_divmat = DiversityMatrix(forestmodel.trees, testfeat)

    hm_qdacc = heatmap(
        divmat_acc,
        title="Quality-Diversity forest",
        xlabel="Tree Index",
        ylabel="Tree Index",
        color=:viridis,  # Use a visually appealing colormap
        clims=(0, maximum(divmat_acc)),  # Set color limits for better contrast
        size=(800, 600)  # Adjust the size of the heatmap
    )
    
    hm_rf = heatmap(
        rf_divmat,
        title="Random forest",
        xlabel="Tree Index",
        ylabel="Tree Index",
        color=:viridis,  # Use a visually appealing colormap
        clims=(0, maximum(divmat_acc)),  # Set color limits for better contrast
        size=(800, 600)  # Adjust the size of the heatmap
    )
    
    heatmaps = plot(hm_qdacc, hm_rf, layout=(1, 2), size=(1600, 600))
    return heatmaps
end

### A PCA of the output predictions of the trees in the QD algorithm's ensembles and the random forest.

function generate_per_tree_outputvectors(ensemble,features::Matrix{Float64})
    n_trees = length(ensemble)
    output_vectors = Array{String1}(undef, n_trees, size(features, 1))
    for i in 1:n_trees
        output_vectors[i, :] = apply_tree(ensemble[i], features)
    end
    return output_vectors
end

function generate_per_tree_outputvectors(ensemble, features::Matrix{Float64})
    n_trees = length(ensemble)
    output_vectors = Array{Int}(undef, n_trees, size(features, 1))  # Use Int for binary values (1s and 0s)
    
    for i in 1:n_trees
        # Apply the tree and map "M" to 1 and "B" to 0
        output_vectors[i, :] = [label == "M" ? 1 : 0 for label in apply_tree(ensemble[i], features)]
    end
    
    return output_vectors
end

function PCA_on_output_vectors(output_vectors_ensemble::Matrix{Int},output_vectors_rf::Matrix{Int})
    output_vectors_ensemble = output_vectors_ensemble'  # Transpose the matrix to have samples as rows and features as columns
    output_vectors_rf = output_vectors_rf'  # Transpose the matrix to have samples as rows and features as columns
    pca_model = MultivariateStats.fit(PCA, output_vectors_ensemble; maxoutdim=2)
    transformed_data = MultivariateStats.transform(pca_model, output_vectors_ensemble)
    transformed_data_rf = MultivariateStats.transform(pca_model, output_vectors_rf)
    return transformed_data, transformed_data_rf
end

function QD_PCA_comparison_plot(Forestmodel,archive,testfeat,testlabels)

    ensemble_size = length(Forestmodel.trees)
    acc_ensemble = select_from_archive(archive, ensemble_size, testfeat, testlabels, "accuracy")

    aens_outvects_QDensemble = generate_per_tree_outputvectors(acc_ensemble, testfeat)
    aens_outvectsrf = generate_per_tree_outputvectors(Forestmodel.trees, testfeat)


    accs = [fitness(tree, testfeat, testlabels) for tree in acc_ensemble]
    accs_rf = [fitness(tree, testfeat, testlabels) for tree in Forestmodel.trees]

    PCA_outvects,PCA_outvects_rf = PCA_on_output_vectors(aens_outvects_QDensemble,aens_outvectsrf)

    fig_ = scatter(PCA_outvects_rf[1, :], PCA_outvects_rf[2, :], marker_z=accs_rf , title="PCA of Ensemble Output Vectors", xlabel="PC1", ylabel="PC2", legend=false, size=(800, 600), color=:viridis,markershape=:square)
    scatter!(PCA_outvects[1, :], PCA_outvects[2, :], marker_z=accs, title="PCA of Ensemble Output Vectors", xlabel="PC1", ylabel="PC2", legend=false, size=(800, 600), color=:viridis,colorbar=true) #markershape=:+
    
    return fig_
end

### Scatter plot on how the accuracy of the trees in the QD algorithm's ensembles and the random forest relate to their diversity.

function accuracy_vs_uniqueness_plot(forestmodel,archive,testfeat,testlabels)
    ensemble_size = length(forestmodel.trees)
    acc_ensemble = select_from_archive(archive, ensemble_size, testfeat, testlabels, "accuracy")

    accs = [fitness(tree, testfeat, testlabels) for tree in acc_ensemble]
    accs_rf = [fitness(tree, testfeat, testlabels) for tree in forestmodel.trees]

    ensemble_div_mat = DiversityMatrix(acc_ensemble, testfeat)
    rf_div_mat = DiversityMatrix(forestmodel.trees, testfeat)

    average_dists_per_tree_rf = [mean(rf_div_mat[i,:]) for i in eachindex(forestmodel.trees)]
    average_dists_per_tree_ensemble = [mean(rf_div_mat[i,:]) for i in eachindex(acc_ensemble)]

    fig = scatter(average_dists_per_tree_rf, accs_rf, label="Random Forest", xlabel="Average Distance to Other Trees", ylabel="Accuracy" )
    scatter!(average_dists_per_tree_ensemble, accs, label="QD Ensemble") #markershape=:+
    return fig
end