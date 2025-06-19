#!/bin/bash



## Setup conda
set -euo pipefail
CONDA_BASE="${HOME}/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate conformal_training

echo "Running evaluation script for conformal training experiments..."
echo "Note: output will be saved to eval_logs.txt."

exec > >(tee -a eval_logs.txt) 2>&1

EXPERIMENT_DIR="./results"
# Define seed range (same as in train.sh)
SEEDS="0 1 2 3 4 5 6 7 8 9"

# Define evaluation methods
METHODS="thr aps"

# Function to format time
format_time() {
    local seconds=$1
    local hours=$((seconds / 3600))
    local minutes=$(( (seconds % 3600) / 60 ))
    local secs=$((seconds % 60))
    printf "%02d:%02d:%02d" $hours $minutes $secs
}

# Function to echo section header
echo_section() {
    local title=$1
    echo "=========================================================="
    echo "===== $title EVALUATION ====="
    echo "=========================================================="
    echo "Started at: $(date)"
    SECTION_START_TIME=$(date +%s)
}

# Function to echo section footer
echo_section_end() {
    local title=$1
    local end_time=$(date +%s)
    local duration=$((end_time - SECTION_START_TIME))
    echo "----------------------------------------------------------"
    echo "Completed $title evaluation at: $(date)"
    echo "Duration: $(format_time $duration)"
    echo "=========================================================="
    echo ""
}

# Start overall timing
TOTAL_START_TIME=$(date +%s)
echo "Starting evaluation process at: $(date)"
echo "Evaluating results from: $EXPERIMENT_DIR"
echo "Using seeds: $SEEDS"
echo "Using methods: $METHODS"
echo ""

# ----- MNIST -----
echo_section "MNIST"

echo "Evaluating MNIST models for all seeds..."
MODELS_START_TIME=$(date +%s)
for SEED in $SEEDS; do
  echo "  Processing seed $SEED..."
  
  # models experiment
  echo "    Evaluating models experiment..."
  for METHOD in $METHODS; do
    echo "      Using method: $METHOD"
    python3 ../../conformal_training/eval.py \
      --experiment_path=$EXPERIMENT_DIR/mnist_models_seed${SEED}/ \
      --experiment_dataset=mnist \
      --experiment_method=$METHOD
  done

  # conformal training experiment
  echo "    Evaluating conformal training experiment..."
  for METHOD in $METHODS; do
    echo "      Using method: $METHOD"
    python3 ../../conformal_training/eval.py \
      --experiment_path=$EXPERIMENT_DIR/mnist_conformal.training_seed${SEED}/ \
      --experiment_dataset=mnist \
      --experiment_method=$METHOD
  done
  echo "  Completed seed $SEED"
done
MODELS_END_TIME=$(date +%s)
MODELS_DURATION=$((MODELS_END_TIME - MODELS_START_TIME))
echo "MNIST evaluation completed in $(format_time $MODELS_DURATION)"

echo_section_end "MNIST"

## ONLY evals MNIST, then exits.
exit 1


# ----- Fashion‑MNIST -----
echo_section "Fashion-MNIST"

echo "Evaluating Fashion-MNIST models for all seeds..."
MODELS_START_TIME=$(date +%s)
for SEED in $SEEDS; do
  echo "  Processing seed $SEED..."
  
  # models experiment
  echo "    Evaluating models experiment..."
  for METHOD in $METHODS; do
    echo "      Using method: $METHOD"
    python3 ../../conformal_training/eval.py \
      --experiment_path=$EXPERIMENT_DIR/fashion_mnist_models_seed${SEED}/ \
      --experiment_dataset=fashion_mnist \
      --experiment_method=$METHOD
  done

  # conformal training experiment
  echo "    Evaluating conformal training experiment..."
  for METHOD in $METHODS; do
    echo "      Using method: $METHOD"
    python3 ../../conformal_training/eval.py \
      --experiment_path=$EXPERIMENT_DIR/fashion_mnist_conformal.training_seed${SEED}/ \
      --experiment_dataset=fashion_mnist \
      --experiment_method=$METHOD
  done
  echo "  Completed seed $SEED"
done
MODELS_END_TIME=$(date +%s)
MODELS_DURATION=$((MODELS_END_TIME - MODELS_START_TIME))
echo "Fashion-MNIST evaluation completed in $(format_time $MODELS_DURATION)"

echo_section_end "Fashion-MNIST"

# ----- EMNIST/byClass -----
echo_section "EMNIST/byClass"

echo "Evaluating EMNIST/byClass models for all seeds..."
MODELS_START_TIME=$(date +%s)
for SEED in $SEEDS; do
  echo "  Processing seed $SEED..."
  
  # models experiment
  echo "    Evaluating models experiment..."
  for METHOD in $METHODS; do
    echo "      Using method: $METHOD"
    python3 ../../conformal_training/eval.py \
      --experiment_path=$EXPERIMENT_DIR/emnist_byclass_models_seed${SEED}/ \
      --experiment_dataset=emnist_byclass \
      --experiment_method=$METHOD
  done

  # conformal experiment
  echo "    Evaluating conformal experiment..."
  for METHOD in $METHODS; do
    echo "      Using method: $METHOD"
    python3 ../../conformal_training/eval.py \
      --experiment_path=$EXPERIMENT_DIR/emnist_byclass_conformal_seed${SEED}/ \
      --experiment_dataset=emnist_byclass \
      --experiment_method=$METHOD
  done
  echo "  Completed seed $SEED"
done
MODELS_END_TIME=$(date +%s)
MODELS_DURATION=$((MODELS_END_TIME - MODELS_START_TIME))
echo "EMNIST/byClass evaluation completed in $(format_time $MODELS_DURATION)"

echo_section_end "EMNIST/byClass"

# ----- CIFAR‑10 -----
echo_section "CIFAR-10"

echo "Evaluating CIFAR-10 models for all seeds..."
MODELS_START_TIME=$(date +%s)
for SEED in $SEEDS; do
  echo "  Processing seed $SEED..."
  
  # models experiment
  echo "    Evaluating models experiment..."
  for METHOD in $METHODS; do
    echo "      Using method: $METHOD"
    python3 ../../conformal_training/eval.py \
      --experiment_path=$EXPERIMENT_DIR/cifar10_models_seed${SEED}/ \
      --experiment_dataset=cifar10 \
      --experiment_method=$METHOD
  done

  # baseline experiment
  echo "    Evaluating baseline experiment..."
  for METHOD in $METHODS; do
    echo "      Using method: $METHOD"
    python3 ../../conformal_training/eval.py \
      --experiment_path=$EXPERIMENT_DIR/cifar10_baseline_seed${SEED}/ \
      --experiment_dataset=cifar10 \
      --experiment_method=$METHOD
  done

  # conformal training experiment
  echo "    Evaluating conformal training experiment..."
  for METHOD in $METHODS; do
    echo "      Using method: $METHOD"
    python3 ../../conformal_training/eval.py \
      --experiment_path=$EXPERIMENT_DIR/cifar10_conformal.training_seed${SEED}/ \
      --experiment_dataset=cifar10 \
      --experiment_method=$METHOD
  done
  echo "  Completed seed $SEED"
done
MODELS_END_TIME=$(date +%s)
MODELS_DURATION=$((MODELS_END_TIME - MODELS_START_TIME))
echo "CIFAR-10 evaluation completed in $(format_time $MODELS_DURATION)"

echo_section_end "CIFAR-10"

# ----- CIFAR‑100 -----
echo_section "CIFAR-100"

echo "Evaluating CIFAR-100 models for all seeds..."
MODELS_START_TIME=$(date +%s)
for SEED in $SEEDS; do
  echo "  Processing seed $SEED..."
  
  # models experiment
  echo "    Evaluating models experiment..."
  for METHOD in $METHODS; do
    echo "      Using method: $METHOD"
    python3 ../../conformal_training/eval.py \
      --experiment_path=$EXPERIMENT_DIR/cifar100_models_seed${SEED}/ \
      --experiment_dataset=cifar100 \
      --experiment_method=$METHOD
  done

  # baseline experiment
  echo "    Evaluating baseline experiment..."
  for METHOD in $METHODS; do
    echo "      Using method: $METHOD"
    python3 ../../conformal_training/eval.py \
      --experiment_path=$EXPERIMENT_DIR/cifar100_baseline_seed${SEED}/ \
      --experiment_dataset=cifar100 \
      --experiment_method=$METHOD
  done

  # conformal training experiment
  echo "    Evaluating conformal training experiment..."
  for METHOD in $METHODS; do
    echo "      Using method: $METHOD"
    python3 ../../conformal_training/eval.py \
      --experiment_path=$EXPERIMENT_DIR/cifar100_conformal.training_seed${SEED}/ \
      --experiment_dataset=cifar100 \
      --experiment_method=$METHOD
  done
  echo "  Completed seed $SEED"
done
MODELS_END_TIME=$(date +%s)
MODELS_DURATION=$((MODELS_END_TIME - MODELS_START_TIME))
echo "CIFAR-100 evaluation completed in $(format_time $MODELS_DURATION)"

echo_section_end "CIFAR-100"

# Calculate and display total execution time
TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))
echo "=========================================================="
echo "EVALUATION COMPLETED!"
echo "Started at: $(date -d @$TOTAL_START_TIME)"
echo "Finished at: $(date)"
echo "Total duration: $(format_time $TOTAL_DURATION)"
echo "=========================================================="
