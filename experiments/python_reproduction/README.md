# Python reproduction

## Installation

First create a cuda environment from the `conformal_training/environment.yaml` file and activate it:

```bash
conda env create -f conformal_training/environment.yaml
conda activate conformal_training
```

The Python packages should be installed automatically.

Run in activated conda environment! Install cuda if not installed: 

```bash
pip install --upgrade pip
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## 4 EXPERIMENTS

From the original paper:

We present experiments in two parts: First, in Sec. 4.1, we demonstrate that ConfTr can reduce inefficiency of THR and APS compared to CP applied to a baseline model trained using cross- entropy loss separately (see Tab. 1 for the main results). Thereby, we outperform concurrent work of Bellotti (2021). Second, in Sec. 4.2, we show how ConfTr can be used to “shape” confidence sets, i.e., reduce class-conditional inefficiency for specific (groups of) classes or coverage confusion of two or more classes, while maintaining the marginal coverage guarantee. This is impossible using (Bellotti, 2021) and rather difficult for standard CP.

We consider several benchmark datasets as well as architectures, c.f. Tab. A, and report metrics averaged across 10 random calibration/test splits for 10 trained models for each method. We focus on (non-differentiable) THR and APS as CP methods used after training and, thus, obtain the cor- responding coverage guarantee. THR, in particular, consistently achieves lower inefficiency for a fixed confidence level α than, e.g., THRL (i.e., THR on logits) or RAPS, see Fig. 2 (left). We set α = 0.01 and use the same α during training using ConfTr. Hyper-parameters are optimized for THR or APS individually. We refer to App. F for further details on datasets, models, evaluation protocol and hyper-parameter optimization.

### 4.1 REDUCING INEFFICIENCY WITH CONFTR

In the first part, we focus on the inefficiency reductions of ConfTr in comparison to a standard cross- entropy training baseline and (Bellotti, 2021) (Bel). After summarizing the possible inefficiency reductions, we also discuss which CP method to use during training and how ConfTr can be used for ensembles and generalizes to lower α.

**Table 1**: Main Inefficiency Results, comparing (Bellotti, 2021) (Bel, trained with THRL) and ConfTr (trained with THRLP) using THR or APS at test time (with α=0.01). We also report improvements relative to the baseline, i.e., standard cross-entropy training, in percentage in parentheses. ConfTr results in a consistent improvement of inefficiency for both THR and APS. Training with L_class, using L = I_K, generally works slightly better. On CIFAR, the inefficiency reduction is smaller compared to other datasets as ConfTr is trained on pre-trained ResNet features.

| Dataset   | | THR | | | | APS | | |
|-----------|--------|--------|---------|-------------|--------|--------|-------------|
|           | Basel. | Bel    | ConfTr  | +Class      | Basel. | ConfTr | +Class      |
| MNIST     | 2.23   | 2.70   | 2.18    | 2.11 (-5.4%)| 2.50   | 2.16   | 2.14 (-14.4%)|
| F-MNIST   | 2.05   | 1.90   | 1.69    | 1.67 (-18.5%)| 2.36   | 1.82   | 1.72 (-27.1%)|
| EMNIST    | 2.66   | 3.48   | 2.66    | 2.49 (-6.4%)| 4.23   | 2.86   | 2.87 (-32.2%)|
| CIFAR10   | 2.93   | 2.93   | 2.88    | 2.84 (-3.1%)| 3.30   | 3.05   | 2.93 (-11.2%)|
| CIFAR100  |10.63   |10.91   |10.78    |10.44 (-1.8%)|16.62   |12.99   |12.73 (-23.4%)|

More results can be found in App. J.
