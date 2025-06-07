#!/bin/bash

# Define the experiment directory
EXPERIMENT_DIR="./results"

# Define seed range (same as in train.sh)
SEEDS="0 1 2 3 4 5 6 7 8 9"

# Define evaluation methods
METHODS="thr aps"

# ----- MNIST -----
for SEED in $SEEDS; do
  # models experiment
  for METHOD in $METHODS; do
    python3 eval.py \
      --experiment_path=$EXPERIMENT_DIR/mnist_models_seed${SEED}/ \
      --experiment_dataset=mnist \
      --experiment_method=$METHOD
  done

  # conformal training experiment
  for METHOD in $METHODS; do
    python3 eval.py \
      --experiment_path=$EXPERIMENT_DIR/mnist_conformal.training_seed${SEED}/ \
      --experiment_dataset=mnist \
      --experiment_method=$METHOD
  done
done

# ----- Fashion‑MNIST -----
for SEED in $SEEDS; do
  # models experiment
  for METHOD in $METHODS; do
    python3 eval.py \
      --experiment_path=$EXPERIMENT_DIR/fashion_mnist_models_seed${SEED}/ \
      --experiment_dataset=fashion_mnist \
      --experiment_method=$METHOD
  done

  # conformal training experiment
  for METHOD in $METHODS; do
    python3 eval.py \
      --experiment_path=$EXPERIMENT_DIR/fashion_mnist_conformal.training_seed${SEED}/ \
      --experiment_dataset=fashion_mnist \
      --experiment_method=$METHOD
  done
done

# ----- EMNIST/byClass -----
for SEED in $SEEDS; do
  # models experiment
  for METHOD in $METHODS; do
    python3 eval.py \
      --experiment_path=$EXPERIMENT_DIR/emnist_byclass_models_seed${SEED}/ \
      --experiment_dataset=emnist_byclass \
      --experiment_method=$METHOD
  done

  # conformal experiment
  for METHOD in $METHODS; do
    python3 eval.py \
      --experiment_path=$EXPERIMENT_DIR/emnist_byclass_conformal_seed${SEED}/ \
      --experiment_dataset=emnist_byclass \
      --experiment_method=$METHOD
  done
done

# ----- CIFAR‑10 -----
for SEED in $SEEDS; do
  # models experiment
  for METHOD in $METHODS; do
    python3 eval.py \
      --experiment_path=$EXPERIMENT_DIR/cifar10_models_seed${SEED}/ \
      --experiment_dataset=cifar10 \
      --experiment_method=$METHOD
  done

  # baseline experiment
  for METHOD in $METHODS; do
    python3 eval.py \
      --experiment_path=$EXPERIMENT_DIR/cifar10_baseline_seed${SEED}/ \
      --experiment_dataset=cifar10 \
      --experiment_method=$METHOD
  done

  # conformal training experiment
  for METHOD in $METHODS; do
    python3 eval.py \
      --experiment_path=$EXPERIMENT_DIR/cifar10_conformal.training_seed${SEED}/ \
      --experiment_dataset=cifar10 \
      --experiment_method=$METHOD
  done
done

# ----- CIFAR‑100 -----
for SEED in $SEEDS; do
  # models experiment
  for METHOD in $METHODS; do
    python3 eval.py \
      --experiment_path=$EXPERIMENT_DIR/cifar100_models_seed${SEED}/ \
      --experiment_dataset=cifar100 \
      --experiment_method=$METHOD
  done

  # baseline experiment
  for METHOD in $METHODS; do
    python3 eval.py \
      --experiment_path=$EXPERIMENT_DIR/cifar100_baseline_seed${SEED}/ \
      --experiment_dataset=cifar100 \
      --experiment_method=$METHOD
  done

  # conformal training experiment
  for METHOD in $METHODS; do
    python3 eval.py \
      --experiment_path=$EXPERIMENT_DIR/cifar100_conformal.training_seed${SEED}/ \
      --experiment_dataset=cifar100 \
      --experiment_method=$METHOD
  done
done
