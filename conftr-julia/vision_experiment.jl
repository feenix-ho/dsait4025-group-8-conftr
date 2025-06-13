

# Datasets: https://juliaml.github.io/MLDatasets.jl/dev/datasets/vision/
# Models: https://juliaai.github.io/MLJ.jl/stable/model_browser/#Model-Browser
using ConformalPrediction
using ArgParse
using MLDatasets
using MLJ
using MLJFlux
using MLJLinearModels
using Images
using Flux
using NearestNeighborModels
using DataFrames

import YAML
# import JLSO

include("builder.jl")

const available_datasets = Dict(
    "MNIST" => MNIST,
    "EMNIST" => EMNIST,
    "CIFAR10" => CIFAR10
)

const available_models = Dict(
    "cnn" => build_cnn,
    "mlp" => build_mlp,
    "logistic" => (@load LogisticClassifier pkg = MLJLinearModels),
    "evo_tree" => (@load EvoTreeClassifier pkg = EvoTrees),
    "knn" => (@load KNNClassifier pkg = NearestNeighborModels),
    "decision_tree" => (@load DecisionTreeClassifier pkg = DecisionTree),
    "random_forest" => (@load RandomForestClassifier pkg = DecisionTree),
)

function parse_commandline()
    settings = ArgParseSettings()

    @add_arg_table! settings begin
        "--config"
            help = "Path to configuration file."
            arg_type = String
            required = true
        # "--dataset"
        #     help = "Dataset name."
        #     arg_type = String
        #     required = true
        # "--model"
        #     help = "Model name."
    end

    return parse_args(settings)
end

function get_dataset(dataset_name::String; convert::Bool)
    @assert dataset_name in keys(available_datasets) "$(dataset_name) is not a valid dataset."

    X_train_raw, y_train_raw = available_datasets[dataset_name](split=:train)[:]
    X_test_raw, y_test_raw = available_datasets[dataset_name](split=:test)[:]

	if convert
    	X_train = map(x -> convert2image(available_datasets[dataset_name], x), eachslice(X_train_raw, dims=3))
    	X_test = map(x -> convert2image(available_datasets[dataset_name], x), eachslice(X_test_raw, dims=3))
	else
		X_train = permutedims(X_train_raw, [3, 1, 2])
        X_train = reshape(X_train, X_train.size[1], :)
		X_test = permutedims(X_test_raw, [3, 1, 2])
        X_test = reshape(X_test, X_test.size[1], :)
        X_train = DataFrame(X_train, :auto)
        X_test = DataFrame(X_test, :auto)
	end

    y_train = coerce(y_train_raw, Multiclass)
    y_test = coerce(y_test_raw, Multiclass)
    return X_train, y_train, X_test, y_test
end

function convert_dict_keys(dict::Dict)
    new_dict = Dict{Symbol, Any}()
    for p in dict
        k = p[1]
        v = p[2]
        setindex!(new_dict, v, Symbol(k))
    end
    return new_dict
end

function build_image_classifier(config::Dict)
    @assert :model_name in keys(config) "model_name not found."
    model_name = config[:model_name]
    @assert model_name in keys(available_models) "$(model_name) is not a valid model."

    # @assert :batch_size in keys(config) "batch_size not found."
    # batch_size = config[:batch_size]
    @assert :epochs in keys(config) "epochs not found."
    epochs = config[:epochs]


    if model_name == "cnn"
        @assert :input_dim in keys(config) "input_dim not found."
        input_dim = config[:input_dim]

        @assert :conv_dims in keys(config) "conv_dims not found."
        conv_dims = config[:conv_dims]

        @assert :mlp_dims in keys(config) "mlp_dims not found."
        mlp_dims = config[:mlp_dims]

        builder = MLJFlux.@builder build_cnn(input_dim, conv_dims, mlp_dims)

    elseif model_name == "mlp"
        @assert :input_dim in keys(config) "input_dim not found."
        input_dim = config[:input_dim]

        @assert :mlp_dims in keys(config) "mlp_dims not found."
        mlp_dims = config[:mlp_dims]

        builder = MLJFlux.@builder build_mlp(input_dim, mlp_dims)
    else
        if :kwargs in keys(config)
            kwargs = config[:kwargs]
            kwargs = convert_dict_keys(kwargs)
            clf = available_models[model_name](; kwargs...)
        else
            clf = available_models[model_name]()
        end
        return clf, false
    end
    
    ImageClassifier = @load ImageClassifier pkg=MLJFlux
    img_clf = ImageClassifier(
        builder=builder,
        # batch_size=batch_size,
        epochs=epochs,
        loss=Flux.crossentropy
    )
    return img_clf, true
end

function run_baseline(config::Dict)
    printstyled("Running baseline pipeline ...\n\n"; color=:red)
    # Load dataset and build model
    @assert :dataset in keys(config) "dataset not found."
    ds_name = config[:dataset]
    @assert ds_name in keys(available_datasets) "$(ds_name) is not a valid dataset."
    
    img_clf, need_convert = build_image_classifier(config)
    X_train, y_train, X_test, y_test = get_dataset(ds_name; convert=need_convert)
    
    mach = machine(img_clf, X_train, y_train)

    # Train and evaluate on train set
    fit!(mach)

    perf = evaluate!(
        mach,
        resampling=Holdout(rng=123, fraction_train=0.9),
        operation=predict,
        measure=[emp_coverage, ssc, ineff]
    )
    display(perf)

    println("Empirical coverage: $(round(perf.measurement[1], digits=3))")
    println("SSC: $(round(perf.measurement[2], digits=3))")
    println("Inefficiency: $(round(perf.measurement[3], digits=3))\n\n\n")

    # Predict on test set
    ŷ = predict(mach, X_test)
    ie = ConformalPrediction.ineff(ŷ)
    println("Inefficiency: $(round(ie, digits=3))")
end

function run_conformal_pipeline(config::Dict, method::Symbol)
    # Load dataset and build ConfTr machine
    @assert :dataset in keys(config) "dataset not found."
    ds_name = config[:dataset]
    @assert ds_name in keys(available_datasets) "$(ds_name) is not a valid dataset."
    
    @assert :coverage in keys(config) "coverage not found."
    cov = config[:coverage]
    img_clf, need_convert = build_image_classifier(config)
    
    X_train, y_train, X_test, y_test = get_dataset(ds_name; convert=need_convert)
    
    conf_model = conformal_model(img_clf; method=method, coverage=cov)
    conf_mach = machine(conf_model, X_train, y_train)

    # Train and evaluate on train set
    fit!(conf_mach)

    perf = evaluate!(
        conf_mach,
        resampling=Holdout(rng=123, fraction_train=0.9),
        operation=predict,
        measure=[emp_coverage, ssc, ineff]
    )
    display(perf)

    println("Empirical coverage: $(round(perf.measurement[1], digits=3))")
    println("SSC: $(round(perf.measurement[2], digits=3))")
    println("Inefficiency: $(round(perf.measurement[3], digits=3))\n\n\n")

    # Predict on test set
    ŷ = predict(conf_mach, X_test)
    ie = ConformalPrediction.ineff(ŷ)
    println("Inefficiency: $(round(ie, digits=3))")
end


function run_simple_inductive(config::Dict)
    printstyled("Running Simple Inductive ConfTr pipeline ...\n\n"; color=:red)
    run_conformal_pipeline(config, :simple_inductive)
end

function run_adaptive_inductive(config::Dict)
    printstyled("Running Adaptive Inductive ConfTr pipeline ...\n\n"; color=:red)
    run_conformal_pipeline(config, :adaptive_inductive)
end

function run_naive_transductive(config::Dict)
    printstyled("Running Naive Transductive ConfTr pipeline ...\n\n"; color=:red)
    run_conformal_pipeline(config, :naive)
end


function main()
    args = parse_commandline()

    config_path = args["config"]
    config = YAML.load_file(config_path, dicttype=Dict{Symbol,Any})
    
    @assert :save_dir in keys(config) "save_dir not found."
    save_dir = config[:save_dir]

    run_baseline(config)
    run_simple_inductive(config)
    run_adaptive_inductive(config)
    run_naive_transductive(config)

    # MLJ.save(save_dir, conf_mach)
end

main()
