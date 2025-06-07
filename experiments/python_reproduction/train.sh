# Define the experiment directory
EXPERIMENT_DIR="./results"

# ----- MNIST -----
python3 ../../conformal_training/run.py \
    --experiment_dataset=mnist \
    --experiment_experiment=models \
    --experiment_seeds=10 \
    --experiment_path=$EXPERIMENT_DIR

python3 ../../conformal_training/run.py \
    --experiment_dataset=mnist \
    --experiment_experiment=conformal.training \
    --experiment_seeds=10 \
    --experiment_path=$EXPERIMENT_DIR

# ----- Fashion‑MNIST -----
python3 ../../conformal_training/run.py \
    --experiment_dataset=fashion_mnist \
    --experiment_experiment=models \
    --experiment_seeds=10 \
    --experiment_path=$EXPERIMENT_DIR

python3 ../../conformal_training/run.py \
    --experiment_dataset=fashion_mnist \
    --experiment_experiment=conformal.training \
    --experiment_seeds=10 \
    --experiment_path=$EXPERIMENT_DIR

# ----- EMNIST/byClass -----
python3 ../../conformal_training/run.py \
    --experiment_dataset=emnist_byclass \
    --experiment_experiment=models \
    --experiment_seeds=10 \
    --experiment_path=$EXPERIMENT_DIR

python3 ../../conformal_training/run.py \
    --experiment_dataset=emnist_byclass \
    --experiment_experiment=conformal \
    --experiment_seeds=10 \
    --experiment_path=$EXPERIMENT_DIR

# ----- CIFAR‑10 -----
# first train the backbone
python3 ../../conformal_training/run.py \
    --experiment_dataset=cifar10 \
    --experiment_experiment=models \
    --experiment_seeds=10 \
    --experiment_path=$EXPERIMENT_DIR

# baseline on the backbone
python3 ../../conformal_training/run.py \
    --experiment_dataset=cifar10 \
    --experiment_experiment=baseline \
    --experiment_seeds=10 \
    --experiment_path=$EXPERIMENT_DIR

# conformal training on the backbone
python3 ../../conformal_training/run.py \
    --experiment_dataset=cifar10 \
    --experiment_experiment=conformal.training \
    --experiment_seeds=10 \
    --experiment_path=$EXPERIMENT_DIR

# ----- CIFAR‑100 -----
# train the backbone
python3 ../../conformal_training/run.py \
    --experiment_dataset=cifar100 \
    --experiment_experiment=models \
    --experiment_seeds=10 \
    --experiment_path=$EXPERIMENT_DIR

# baseline on the backbone
python3 ../../conformal_training/run.py \
    --experiment_dataset=cifar100 \
    --experiment_experiment=baseline \
    --experiment_seeds=10 \
    --experiment_path=$EXPERIMENT_DIR

# conformal training on the backbone
python3 ../../conformal_training/run.py \
    --experiment_dataset=cifar100 \
    --experiment_experiment=conformal.training \
    --experiment_seeds=10 \
    --experiment_path=$EXPERIMENT_DIR
