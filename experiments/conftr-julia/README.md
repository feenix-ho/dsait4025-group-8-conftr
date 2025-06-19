# ConfTr - Julia Implementation

This directory contains a Julia implementation of Conformal Training (ConfTr) based on the paper ["Learning Optimal Conformal Classifiers"](https://arxiv.org/abs/2110.09192) by Google DeepMind.

## Overview

The implementation uses Julia's machine learning ecosystem, particularly the [`ConformalPrediction.jl`](https://github.com/JuliaTrustworthyAI/ConformalPrediction.jl) package, to apply conformal prediction techniques to various machine learning models and datasets.

## Files Structure

- `vision_experiment.jl`: Main experiment script for vision datasets (MNIST, EMNIST, CIFAR10)
- `builder.jl`: Model architecture builder functions (CNN, etc.)
- `utils.jl`: Helper functions for evaluation metrics and data processing
- `testing.jl`: Test code for symbolic regression using conformal prediction
- `configs/`: Configuration files for different experiments
  - `cnn_mnist.yml`: CNN model for MNIST
  - `knn_mnist.yml`: K-Nearest Neighbors model for MNIST
  - `linear_mnist.yml`: Linear model for MNIST
  - `logistic_mnist.yml`: Logistic regression model for MNIST
  - `mlp_mnist.yml`: Multi-layer perceptron for MNIST
- `checkpoints/`: Directory for saving model checkpoints
- `ConformalPrediction/`: Submodule containing the Julia conformal prediction library

## Dependencies

- Julia 1.7+
- Packages:
  - `ConformalPrediction`
  - `Flux`
  - `MLJ` and `MLJFlux`
  - `MLDatasets`
  - `ArgParse`
  - `Images`
  - `DataFrames`
  - `JLSO`
  - `YAML`
  - `NearestNeighborModels`

## Usage

To run an experiment with a specific configuration:

```bash
julia vision_experiment.jl --config configs/cnn_mnist.yml
```

### Configuration Options

The configuration files support the following settings:

- `model_name`: Type of model to use (cnn, knn, linear, etc.)
- `dataset`: Dataset to use (MNIST, EMNIST, CIFAR10)
- `input_dim`: Input dimensions [height, width, channels]
- `conv_dims`: For CNN models, list of [output_channels, kernel_size, pool_size]
- `mlp_dims`: Layer dimensions for fully connected networks
- `batch_size`: Batch size for training
- `epochs`: Number of training epochs
- `save_dir`: Directory to save model checkpoints
- `coverage`: Target coverage probability for conformal prediction (e.g., 0.99)

## Conformal Prediction

This implementation explores how to train models specifically for conformal prediction tasks. Rather than applying conformal prediction as a post-processing step, the goal is to train the model to optimize for conformal prediction performance directly.

The key metrics evaluated are:

- Prediction set size (smaller is better at the same coverage level)
- Coverage guarantee (ensuring true labels are in the prediction set with high probability)
- Accuracy of the most confident predictions
