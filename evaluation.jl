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

function evaluate_ensembles(Forestmodel, archive,trainfeats,trainlabels, testfeatures, testlabels, ensemble_size::Int=50)
    @assert ensemble_size <= length(archive) "Ensemble size must be less than or equal to the archive size."
    @assert ensemble_size <= length(Forestmodel.trees) "Not enough trees in the random forest."

    acc_ensemble = select_from_archive(archive, ensemble_size, trainfeats, trainlabels, "accuracy")
    div_ensemble = select_from_archive(archive, ensemble_size, trainfeats, trainlabels, "diverse")
    hybrid_ensemble = select_from_archive(archive, ensemble_size, trainfeats, trainlabels, "hybrid")
    # random_forest = Forestmodel.trees[1:ensemble_size] # selection of fittest individuals should probably be 
    # done using the training set accs
    random_forest = select_from_archive(Forestmodel.trees,ensemble_size,trainfeats,trainlabels,"accuracy")

    preds_h = apply_ensemble(hybrid_ensemble, testfeatures)
    preds_rf = apply_ensemble(random_forest, testfeatures)
    preds_acc = apply_ensemble(acc_ensemble, testfeatures)
    preds_div = apply_ensemble(div_ensemble, testfeatures)

    println("Random Forest Accuracy: ", Accuracy(preds_rf, testlabels))
    println("Accuracy Ensemble Accuracy: ", Accuracy(preds_acc, testlabels) )
    println("Hybrid Ensemble Accuracy: ", Accuracy(preds_h, testlabels) )
    println("Diverse Ensemble Accuracy: ", Accuracy(preds_div, testlabels) )

    println("\n")

    println("Random Forest Diversity: ", Δ_raw_bar(random_forest, testfeatures))
    println("Accuracy Ensemble Diversity: ", Δ_raw_bar(acc_ensemble, testfeatures) )
    println("Hybrid Ensemble Diversity: ", Δ_raw_bar(hybrid_ensemble, testfeatures) )
    println("Diverse Ensemble Diversity: ", Δ_raw_bar(div_ensemble, testfeatures) )

end

function train_test_split(features, labels, test_size::Float64=0.2)
    n_samples = size(features, 1)
    indices = randperm(n_samples)
    split_index = Int(floor(n_samples * (1 - test_size)))
    train_indices = indices[1:split_index]
    test_indices = indices[(split_index + 1):end]
    return features[train_indices, :], labels[train_indices], features[test_indices, :], labels[test_indices]
end

function Accuracy(predictions, labels)
    return sum(predictions .== labels) / length(labels)
end
