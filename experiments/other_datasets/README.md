# Reproducing Results for Other Datasets

## German Credit and Diabetes Prediction - Python

This repository contains code for evaluation of ConfTr on **German Credit** and **Diabetes Prediction**.

### Installation

To get started, please follow the installation guidelines from the [original ConfTr repository](https://github.com/google-deepmind/conformal_training).

### Reproducing Results

#### Training

The experiment configurations are defined in the `experiments/` directory (of `conftr-python/src`) and are executed using `run.py`.

To train **Baseline**, **ConfTr**, and **ConfTr + L_class** models for **German Credit**, run the following commands (10 seeds):

```bash
# Baseline
python3 run.py \
  --experiment_dataset=german_credit \
  --experiment_experiment=models \
  --experiment_seeds=10 \
  --experiment_path=~/experiments/

# ConfTr
python3 run.py \
  --experiment_dataset=german_credit \
  --experiment_experiment=conformal.training \
  --experiment_seeds=10 \
  --experiment_path=~/experiments/

# ConfTr + L_class
python3 run.py \
  --experiment_dataset=german_credit \
  --experiment_experiment=conformal.training_Lclass \
  --experiment_seeds=10 \
  --experiment_path=~/experiments/
```

To run these for the **Diabetes Prediction** dataset,  change the `--experiment_dataset` flag:

```bash
--experiment_dataset=diabetes
```

#### Evaluation

Use `eval.py` to evaluate trained models.

**Baseline, ThrL:**

```bash
for seed in {0..9}; do
  python eval.py \
    --experiment_path=~/experiments/german_credit_models_seed${seed}/ \
    --experiment_method=thrl \
    --experiment_dataset=german_credit
done
```

**ConfTr, Thr:**

```bash
for seed in {0..9}; do
  python eval.py \
    --experiment_path=~/experiments/german_credit_conformal.training_seed${seed}/ \
    --experiment_method=thr \
    --experiment_dataset=german_credit
done
```

**ConfTr + L_class, Thr:**

```bash
for seed in {0..9}; do
  python eval.py \
    --experiment_path=~/experiments/german_credit_conformal.training_Lclass_seed${seed}/ \
    --experiment_method=thr \
    --experiment_dataset=german_credit
done
```

To switch between conformal prediction methods:  
Use `--experiment_method=aps` for Adaptive Prediction Sets (APS)  
Use `--experiment_method=thr` for Threshold CP (Thr)  
Use `--experiment_method=thrl` for thresholding on raw logits (ThrL)

To run these for the **Diabetes Prediction** dataset,  change:

```bash
--experiment_path=~/experiments/diabetes_models_seed${seed}/ \
--experiment_dataset=diabetes
```
