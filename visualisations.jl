##################################################
#### Novelty Search visualisation functions ####
################################################
using Plots, MultivariateStats, Statistics

function get_QD_ensembles(Forestmodel, archive,trainfeats,trainlabels, ensemble_size::Int=50)
    @assert ensemble_size <= length(archive) "Ensemble size must be less than or equal to the archive size."
    @assert ensemble_size <= length(Forestmodel.trees) "Not enough trees in the random forest."

    acc_ensemble = select_from_archive(archive, ensemble_size, trainfeats, trainlabels, "accuracy")
    div_ensemble = select_from_archive(archive, ensemble_size, trainfeats, trainlabels, "diverse")
    hybrid_ensemble = select_from_archive(archive, ensemble_size, trainfeats, trainlabels, "hybrid")
    # random_forest = Forestmodel.trees[1:ensemble_size] # selection of fittest individuals should probably be 
    # done using the training set accs
    random_forest = select_from_archive(Forestmodel.trees,ensemble_size,trainfeats,trainlabels,"accuracy")

    return acc_ensemble, div_ensemble, hybrid_ensemble, random_forest
end

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

# A heatmap of the diversity matrix of the ensembles created by the QD algorithm and the random forest.
function DiversityMatrix(trees, features::Matrix{Float32})
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

function ensemble_disagreement_heatmap(forestmodel,archive,testfeat,testlabels)
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

# A PCA of the output predictions of the trees in the QD algorithm's ensembles and the random forest.

function generate_per_tree_outputvectors(ensemble,features::Matrix{Float32})
    n_trees = length(ensemble)
    output_vectors = Array{Bool}(undef, n_trees, size(features, 1))
    for i in 1:n_trees
        output_vectors[i, :] = apply_tree(ensemble[i], features)
    end
    return output_vectors
end

function PCA_on_output_vectors(output_vectors_ensemble::Matrix{Bool},output_vectors_rf::Matrix{Bool})
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
    explained_variance = get_pca_explained_variance(aens_outvects_QDensemble)	
    fig_ = scatter(PCA_outvects_rf[1, :], PCA_outvects_rf[2, :], marker_z=accs_rf , title="PCA of Ensemble Output Vectors", xlabel="PC1 ($(explained_variance[1])%)", ylabel="PC2 ($(explained_variance[2])%)", legend=false, size=(800, 600), color=:viridis,markershape=:square)
    scatter!(PCA_outvects[1, :], PCA_outvects[2, :], marker_z=accs, title="PCA of Ensemble Output Vectors", size=(800, 600), color=:viridis,colorbar=true) #markershape=:+
    
    return fig_
end

function get_pca_explained_variance(output_vects)
    output_vects = output_vects'  # Transpose the matrix to have samples as rows and features as columns
    pca_model = MultivariateStats.fit(PCA, output_vects; maxoutdim=size(output_vects, 1));
    var_explained = Int.(round.(( pca_model.prinvars ./ sum(pca_model.prinvars) ) * 100))
    return var_explained[1:2]
end


# Scatter plot on how the accuracy of the trees in the QD algorithm's ensembles and the random forest relate to their diversity.

function accuracy_vs_uniqueness_plot(forestmodel,archive,testfeat,testlabels)
    ensemble_size = length(forestmodel.trees)
    acc_ensemble = select_from_archive(archive, ensemble_size, testfeat, testlabels, "accuracy")

    accs = [fitness(tree, testfeat, testlabels) for tree in acc_ensemble]
    accs_rf = [fitness(tree, testfeat, testlabels) for tree in forestmodel.trees]

    ensemble_div_mat = DiversityMatrix(acc_ensemble, testfeat)
    rf_div_mat = DiversityMatrix(forestmodel.trees, testfeat)

    average_dists_per_tree_rf = [mean(rf_div_mat[i,:]) for i in eachindex(forestmodel.trees)]
    average_dists_per_tree_ensemble = [mean(ensemble_div_mat[i,:]) for i in eachindex(acc_ensemble)]

    fig = scatter(average_dists_per_tree_rf, accs_rf, label="Random Forest", xlabel="Average Distance to Other Trees", ylabel="Accuracy" )
    scatter!(average_dists_per_tree_ensemble, accs, label="QD Ensemble") #markershape=:+
    return fig
end

##################################################
#### Neuro MAP-Elites visualisation functions ####
##################################################

function plot_archive(archive, fitness_archive, n_bins)
    filled = map(coord -> haskey(archive, coord) ? fitness_archive[coord] : NaN,
             Iterators.product(1:n_bins, 1:n_bins))
    HM = heatmap(reshape(collect(filled), n_bins, n_bins)', xlabel="z₁", ylabel="z₂", title="Archive")
    return HM
end
function plot_archive_fixed(archive, fitness_archive, n_bins, generation)
    filled = map(coord -> haskey(archive, coord) ? fitness_archive[coord] : NaN,
             Iterators.product(1:n_bins, 1:n_bins))
    HM = heatmap(reshape(collect(filled), n_bins, n_bins)', xlabel="z₁", ylabel="z₂", title="Archive Gen: $(generation)",clim=(0.8, 1))
    return HM
end

function create_archive_gif(archives,fitness_archives,n_bins)
    @gif for i in axes(archives,1)
        archive = archives[i]
        fitness_arch = fitness_archives[i]
        plot_archive_fixed(archive,fitness_arch,n_bins,i*100)  # Replace this with your actual plotting function
    end every 1 # Adjust `every` to control frame skipping
    # savefig(filename)  # Save the GIF to a file
end


function PlotQuality(testaccs_acc,trainaccs_acc,testaccs_div,trainaccs_div, RF_testacc, RF_trainacc)
    generations = length(testaccs_acc)
    generations = 100 .* collect(1:generations)
    p = plot(generations, testaccs_acc, label="Test Accuracy Ensemble", xlabel="Generations", ylabel="Accuracy",legend=:outertop,title = "Acc vs Gen")
    plot!(p, generations, trainaccs_acc, label="Train Accuracy Ensemble")
    plot!(p,generations, testaccs_div, label="Test Accuracy Diverse Ensemble")   #linestyle=:dash)
    plot!(p, generations, trainaccs_div, label="Train Accuracy Diverse Ensemble")#,linestyle=:dash

    hline!([RF_trainacc], label="RF_train", linestyle=:dash, color=:black)
    hline!([RF_testacc], label="RF_test", linestyle=:dash, color=:grey)

    return p
end

function plotDiversity(acc_divs,div_divss, RF_div)
    generations = length(acc_divs)
    generations = 100 .* collect(1:generations)
    p = plot(generations, acc_divs, label="Acc Ensemble diversity", xlabel="Generations", ylabel="Diversity",legend=:outertop) #title = "Diversity vs Gen"
    plot!(generations, div_divss, label="Div Ensemble diversity") 
    hline!([RF_div], label="RF diversity", linestyle=:dash, color=:black)

    return p
end


function plotEvolution(testaccs_acc,trainaccs_acc,testaccs_div,trainaccs_div,RF_testacc, RF_trainacc, RF_div, divs_acc, divs_div)
    p1 = PlotQuality(testaccs_acc,trainaccs_acc,testaccs_div,trainaccs_div, RF_testacc, RF_trainacc)
    p2 = plotDiversity(divs_acc, divs_div, RF_div)
    p = plot(p1, p2, layout=(1, 2), size=(1100, 500), title="Evolution of Quality and Diversity")
    return p
end

function plot_evolution_quality(testaccs_acc,trainaccs_acc, RF_testacc, RF_trainacc)
    generations = length(testaccs_acc)
    generations = 100 .* collect(1:generations)
    p = plot(generations, testaccs_acc, label="Test Accuracy Ensemble", xlabel="Generations", ylabel="Accuracy",legend=:outertop) #title = "Acc vs Gen"
    plot!(p, generations, trainaccs_acc, label="Train Accuracy Ensemble")

    hline!([RF_trainacc], label="RF_train", linestyle=:dash, color=:black)
    hline!([RF_testacc], label="RF_test", linestyle=:dash, color=:grey)

    return p
end

function plot_evolution_diversity(divs, RF_div)
    generations = length(divs)
    generations = 100 .* collect(1:generations)
    p = plot(generations, divs, label="Ensemble diversity", xlabel="Generations", ylabel="Diversity",legend=:outertop) #title = "Diversity vs Gen"
    hline!([RF_div], label="RF diversity", linestyle=:dash, color=:black)

    return p
end

function plot_evolution(testaccs_acc,trainaccs_acc,divs, RF_testacc, RF_trainacc, RF_div)
    quality_evolution = plot_evolution_quality(testaccs_acc,trainaccs_acc, RF_testacc, RF_trainacc)
    diversity_evolution = plot_evolution_diversity(divs, RF_div)
    p = plot(quality_evolution, diversity_evolution, layout=(1, 2), size=(1100, 500)) #title="Evolution of Quality and Diversity"
    return p
end 
