#/bin/bash

## Setup conda
set -euo pipefail
CONDA_BASE="${HOME}/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate conformal_training

printf "Running evaluation script for conformal training experiments...\n"

# Define the experiment directory
EXPERIMENT_DIR="./results"

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
    echo "===== $title ====="
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
    echo "Completed $title at: $(date)"
    echo "Duration: $(format_time $duration)"
    echo "=========================================================="
    echo ""
}

# Start overall timing
TOTAL_START_TIME=$(date +%s)
echo "Starting training process at: $(date)"
echo "Results will be saved to: $EXPERIMENT_DIR"
echo ""

# ----- MNIST -----
echo_section "MNIST"

echo "Training MNIST models..."
MODEL_START_TIME=$(date +%s)
python3 ../../conformal_training/run.py \
    --experiment_dataset=mnist \
    --experiment_experiment=models \
    --experiment_seeds=10 \
    --experiment_path=$EXPERIMENT_DIR
MODEL_END_TIME=$(date +%s)
MODEL_DURATION=$((MODEL_END_TIME - MODEL_START_TIME))
echo "MNIST models training completed in $(format_time $MODEL_DURATION)"
echo ""

echo "Training MNIST with conformal training..."
python3 ../../conformal_training/run.py \
    --experiment_dataset=mnist \
    --experiment_experiment=conformal.training \
    --experiment_seeds=10 \
    --experiment_path=$EXPERIMENT_DIR

echo_section_end "MNIST"

# ----- Fashion‑MNIST -----
echo_section "Fashion-MNIST"

echo "Training Fashion-MNIST models..."
MODEL_START_TIME=$(date +%s)
python3 ../../conformal_training/run.py \
    --experiment_dataset=fashion_mnist \
    --experiment_experiment=models \
    --experiment_seeds=10 \
    --experiment_path=$EXPERIMENT_DIR
MODEL_END_TIME=$(date +%s)
MODEL_DURATION=$((MODEL_END_TIME - MODEL_START_TIME))
echo "Fashion-MNIST models training completed in $(format_time $MODEL_DURATION)"
echo ""

echo "Training Fashion-MNIST with conformal training..."
python3 ../../conformal_training/run.py \
    --experiment_dataset=fashion_mnist \
    --experiment_experiment=conformal.training \
    --experiment_seeds=10 \
    --experiment_path=$EXPERIMENT_DIR

echo_section_end "Fashion-MNIST"

# ----- EMNIST/byClass -----
echo_section "EMNIST/byClass"

echo "Training EMNIST/byClass models..."
MODEL_START_TIME=$(date +%s)
python3 ../../conformal_training/run.py \
    --experiment_dataset=emnist_byclass \
    --experiment_experiment=models \
    --experiment_seeds=10 \
    --experiment_path=$EXPERIMENT_DIR
MODEL_END_TIME=$(date +%s)
MODEL_DURATION=$((MODEL_END_TIME - MODEL_START_TIME))
echo "EMNIST/byClass models training completed in $(format_time $MODEL_DURATION)"
echo ""

echo "Training EMNIST/byClass with conformal prediction..."
python3 ../../conformal_training/run.py \
    --experiment_dataset=emnist_byclass \
    --experiment_experiment=conformal \
    --experiment_seeds=10 \
    --experiment_path=$EXPERIMENT_DIR

echo_section_end "EMNIST/byClass"

# ----- CIFAR‑10 -----
echo_section "CIFAR-10"

echo "Training CIFAR-10 backbone models..."
MODEL_START_TIME=$(date +%s)
python3 ../../conformal_training/run.py \
    --experiment_dataset=cifar10 \
    --experiment_experiment=models \
    --experiment_seeds=10 \
    --experiment_path=$EXPERIMENT_DIR
MODEL_END_TIME=$(date +%s)
MODEL_DURATION=$((MODEL_END_TIME - MODEL_START_TIME))
echo "CIFAR-10 backbone models training completed in $(format_time $MODEL_DURATION)"
echo ""

echo "Training CIFAR-10 baseline on the backbone..."
BASELINE_START_TIME=$(date +%s)
python3 ../../conformal_training/run.py \
    --experiment_dataset=cifar10 \
    --experiment_experiment=baseline \
    --experiment_seeds=10 \
    --experiment_path=$EXPERIMENT_DIR
BASELINE_END_TIME=$(date +%s)
BASELINE_DURATION=$((BASELINE_END_TIME - BASELINE_START_TIME))
echo "CIFAR-10 baseline training completed in $(format_time $BASELINE_DURATION)"
echo ""

echo "Training CIFAR-10 with conformal training on the backbone..."
python3 ../../conformal_training/run.py \
    --experiment_dataset=cifar10 \
    --experiment_experiment=conformal.training \
    --experiment_seeds=10 \
    --experiment_path=$EXPERIMENT_DIR

echo_section_end "CIFAR-10"

# ----- CIFAR‑100 -----
echo_section "CIFAR-100"

echo "Training CIFAR-100 backbone models..."
MODEL_START_TIME=$(date +%s)
python3 ../../conformal_training/run.py \
    --experiment_dataset=cifar100 \
    --experiment_experiment=models \
    --experiment_seeds=10 \
    --experiment_path=$EXPERIMENT_DIR
MODEL_END_TIME=$(date +%s)
MODEL_DURATION=$((MODEL_END_TIME - MODEL_START_TIME))
echo "CIFAR-100 backbone models training completed in $(format_time $MODEL_DURATION)"
echo ""

echo "Training CIFAR-100 baseline on the backbone..."
BASELINE_START_TIME=$(date +%s)
python3 ../../conformal_training/run.py \
    --experiment_dataset=cifar100 \
    --experiment_experiment=baseline \
    --experiment_seeds=10 \
    --experiment_path=$EXPERIMENT_DIR
BASELINE_END_TIME=$(date +%s)
BASELINE_DURATION=$((BASELINE_END_TIME - BASELINE_START_TIME))
echo "CIFAR-100 baseline training completed in $(format_time $BASELINE_DURATION)"
echo ""

echo "Training CIFAR-100 with conformal training on the backbone..."
python3 ../../conformal_training/run.py \
    --experiment_dataset=cifar100 \
    --experiment_experiment=conformal.training \
    --experiment_seeds=10 \
    --experiment_path=$EXPERIMENT_DIR

echo_section_end "CIFAR-100"

# Calculate and display total execution time
TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))
echo "=========================================================="
echo "TRAINING COMPLETED!"
echo "Started at: $(date -d @$TOTAL_START_TIME)"
echo "Finished at: $(date)"
echo "Total duration: $(format_time $TOTAL_DURATION)"
echo "=========================================================="


echo "Running evaluation script for conformal training experiments..."
source ./eval.sh