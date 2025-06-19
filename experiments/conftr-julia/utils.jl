using MLJ
using CategoricalArrays
using ConformalPrediction

function accuracy_score_baseline(ŷ, y)
    L = ŷ.decoder.classes
    prob = pdf(ŷ, L)
    pred = reshape([x[2] for x in argmax(prob, dims=2)], :)
    pred = pred .- 1
    # print(pred)

    return MLJ.accuracy(pred, y)
end

function accuracy_score_conformal(ŷ, y)
    # Cheating
    L = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    prob = pdf(ŷ, L)
    pred = reshape([x[2] for x in argmax(prob, dims=2)], :)
    pred = pred .- 1
    # print(MLJ.accuracy(pred, y))
    return MLJ.accuracy(pred, y)
    # exit()
end

# CategoricalArrays.CategoricalValue{Int64, UInt32}[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]%                                                           
export accuracy_score_baseline, accuracy_score_conformal