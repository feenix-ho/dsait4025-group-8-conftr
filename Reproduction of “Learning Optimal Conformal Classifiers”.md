# Reproduction of "Learning Optimal Conformal Classifiers"

|        Name        |               Email                | StudentID |
| :----------------: | :--------------------------------: | :-------: |
| Ho Thi Ngoc Phuong | <HoThiNgocPhuong@student.tudelft.nl> |  6172970  |
|  Juul Schnitzler   |  <j.b.schnitzler@student.tudelft.nl>   |  5094917  |
|  Razo van Berkel   |      <r.q.berkel@student.tudelft.nl> | 6330029 |

The reproduction code is available on Github [here](https://github.com/feenix-ho/dsait4025-group-8-conftr).

## Introduction

In recent years, we have seen significant progress in machine and deep learning. Models now achieve high accuracy on many tasks, and this continues to improve even further. However, high accuracy alone does not provide guarantees for safe deployment, especially in high-stakes applications. We need some measure of how certain or uncertain a model is about its predictions, particularly in classification problems. Ideally, we want a formal guarantee that quantifies this uncertainty. Conformal prediction (CP) addresses this need by using the classifier's predictions (e.g., its probability estimates) to construct confidence sets that contain the true class with a predefined probability.

CP has often been applied as a separate processing step after training, which prevents the model from adapting to the prediction of confidence sets. The paper _Learning Optimal Conformal Classifiers_ by Stutz et al [^stutz2021learning] addresses this limitation by introducing methods to include CP directly in the training process, enabling end-to-end training with the conformal wrapper. Their approach, called _conformal training (ConfTr)_, integrates conformalization steps into the mini-batch training loop. The results demonstrate that _ConfTr_ reduces inefficiency (i.e., the size of the confidence sets) of common conformal predictors. Moreover, the authors argue that _ConfTr_ allows shaping the confidence sets at test time, for example by reducing class-conditional inefficiency.

In this blog post, we present a reproduction and extension of some of the results from the _Learning Optimal Conformal Classifiers_ paper. Specifically, we focus on reproducing their results for the tabular German Credit dataset. In their codebase, no implementation was provided for experiments on this dataset. Therefore, we partially reimplemented the preprocessing pipeline and experimental setup for the German Credit dataset. Reproducing results for this dataset is particularly valuable, as it allows us to assess whether the paper provides sufficient detail to reproduce its findings, since the original code for this dataset was not included.

Additionally, we extend the experiments by evaluating _ConfTr_ on a new dataset: a medical tabular dataset for diabetes prediction based on patient information. While the paper highlights the relevance of CP for high-stakes AI applications such as medical diagnosis, no medical datasets were included in their experiments. We found it interesting to explore how _ConfTr_ performs in this domain.

## ConfTr: Recap of the Original Paper [^stutz2021learning]

As we already mentioned in the introduction, the key idea behind _ConfTr_ is to bring conformal prediction (CP) into the training loop. Here, we briefly recap how this process works during training. For full details, please consult the original paper by Stutz et al [^stutz2021learning].

_ConfTr_ simulates CP during training on each mini-batch. Specifically, each mini-batch is split in half:

- One half is used for the calibration step, where a threshold is computed based on the model's predicted probabilities for the calibration samples.
- The other half is used for the prediction step, where confidence sets are constructed using the threshold obtained in the first step.
    The model is then updated using a loss that combines a size loss, which encourages the model to produce smaller confidence sets, and optionally a classification loss that can shape the content of the confidence sets (e.g., penalizing certain classes). The following figure from the original paper illustrates this process:

<figure style="text-align: center;">
 <img src="https://s3.hedgedoc.org/hd1-demo/uploads/1f96c26c-d745-4bc6-a134-4e1f7a10fafc.png" alt="*ConfTr* diagram from Stutz et al.">
 <figcaption><em>Figure from Stutz et al, illustrating the conformal training process.</em></figcaption>
</figure>

The calibration and prediction steps are implemented in a differentiable way (using smooth approximations), so that the entire process can be optimized end-to-end with standard gradient-based methods. After training with _ConfTr_, the model can still be used with any standard CP method at test time, meaning the CP coverage guarantee is preserved.

## Datasets

### MNIST [^lecun2002gradient]

The MNIST benchmark is a collection of 70,000 $28 \times 28$-pixel grayscale images of handwritten digits (60 000 for training, 10 000 for testing) that were "re-mixed" from two earlier NIST handwriting corpora to provide a cleaner machine-learning testbed. Images are size-normalized, antialiased and centered, making the task a well-controlled 10-class classification benchmark that still serves as a quick sanity-check for new conformal-prediction methods.

### GermanCredit [^hofmann1994german]

The Statlog German Credit Data set contains 1 000 loan applications described by 20 categorical or integer attributes (e.g., checking-account status, duration, purpose, age). Each case is labeled as a good or bad credit risk and comes with an asymmetric cost matrix that penalizes false positives five times more than false negatives, reflecting the real-world stakes of credit scoring. Its small size and mixed feature types make it a popular tabular benchmark for evaluating coverage and set-size efficiency in conformal classification.

### Diabetes Prediction [^diabetes]

The Diabetes Prediction dataset released on Kaggle aggregates 100 000 patient records with nine routinely collected attributes—age, gender, body-mass index, hypertension, heart-disease status, five-level smoking history, HbA1c %, fasting blood-glucose level—and a binary label indicating a diabetes diagnosis. Because the features mirror those used in basic clinical screening, the set offers a realistic high-stakes use-case where calibrated uncertainty is critical before deployment.

## Reproduction Methodology

In this section, we describe our reproduction methodology, including the official Python implementation of _ConfTr_, our own Julia implementation, and the extension with tabular datasets. We also outline the experimental setup, hyperparameters, and results for both MNIST and tabular datasets.

## Official Python Implementation

The official Conformal Training code is available on GitHub at [google-deepmind/conformal\_training](https://github.com/google-deepmind/conformal_training/). It’s a pure-Python codebase with auxiliary shell scripts for running experiments and tests. Package dependencies are managed via Conda, with the environment specified in `environment.yml` (minor edits were required for compatibility). The official Python implementation uses JAX for end-to-end differentiable training through conformal prediction steps, with TensorFlow handling dataset operations. The codebase follows a modular structure organized with Absl (Google's Abseil library) for command-line interfaces, logging, and application flow. Configuration management is handled through `ml_collections.ConfigDict`, with hyperparameters defined in `config.py` and experiment-specific settings in the `experiments/` directory (like `run_mnist.py`). The core conformal prediction methods are implemented in `conformal_prediction.py`, with differentiable versions in `smooth_conformal_prediction.py` that enable gradient flow. Training variants are implemented across separate modules (`train_normal.py`, `train_conformal.py`, `train_coverage.py`), with training launched through `run.py` and evaluation performed via `eval.py`. This structure enables reproducible experiments across multiple datasets and conformal prediction variants.

### Reproduction Goal

We're trying to reproduce the results as shown in Table 1 of the original paper, which includes experiments on MNIST. Please find the table and subscript from the original paper below:

**Table 1:** Main Inefficiency Results, comparing (Bellotti, 2021) (Bel, trained with ThrL) and ConTr (trained with ThrLP) using THR or APS at test time (with α = 0.01). We also report improvements relative to the baseline, i.e., standard cross-entropy training, in percentage in parentheses. ConTr results in a consistent improvement of inefficiency for both THR and APS. Training with ℒ₍class₎, using ℒ = I_K, generally works slightly better. On CIFAR, the inefficiency reduction is small compared to other datasets as ConTr is trained on pre-trained ResNet features; see text. More results can be found in App. J.

| Dataset  | THR Baseline | THR Bel | THR ConfTr | THR + ℒ₍class₎ | APS Baseline | APS ConfTr |  APS + ℒ₍class₎ |
| -------- | -----------: | ------: | ---------: | -------------: | -----------: | ---------: | --------------: |
| MNIST    |         2.23 |    2.70 |       2.18 |  2.11 (−5.4 %) |         2.50 |       2.16 |  2.14 (−14.4 %) |
| F-MNIST  |         2.05 |    1.90 |       1.69 | 1.67 (−18.5 %) |         2.36 |       1.82 |  1.72 (−27.1 %) |
| EMNIST   |         2.66 |    3.48 |       2.66 |  2.49 (−6.4 %) |         4.23 |       2.86 |  2.87 (−32.2 %) |
| CIFAR10  |         2.93 |    2.93 |       2.88 |  2.84 (−3.1 %) |         3.30 |       3.05 |  2.93 (−11.2 %) |
| CIFAR100 |        10.63 |   10.91 |      10.78 | 10.44 (−1.8 %) |        16.62 |      12.99 | 12.73 (−23.4 %) |

We focus on the MNIST row, and reproduce the Baseline and the Conformal Training (ConfTr) results. We do not reproduce the Bellotti method, as it is not implemented in the codebase and out-of-scope for this reproduction.

### Installation

1. Clone the repo and navigate to `experiments/python_reproduction/`.
2. Create the Conda environment:

   ```bash
   conda env create -f environment.yml
   ```

3. Activate and verify the setup per the instructions in `README.md`.
4. Note: GPU acceleration could not be enabled; all experiments were run on CPU.

### GPU Compatibility Issues

Attempts to enable GPU support for both JAX and TensorFlow failed due to version conflicts:

- The `jax`, `jaxlib`, and `tensorflow` packages in `environment.yml` point to CPU-only builds.
- Upgrading to GPU-enabled versions created dependency clashes.
- Neither PyPI nor official archives hosted the required GPU builds for the specified JAX versions.
- Conda-Forge had compatible JAX builds, but they conflicted with other packages.
- Ultimately, we ran all experiments on CPU, which is adequate for MNIST but increases runtime.

### Experimental Setup

- **Platform:** Windows 11 Pro with WSL 2 (Ubuntu 24.04 LTS).
- **Conda:** v25.3.1; **Python:** 3.9.13.
- **Hardware:** AMD Ryzen 5 3600, 32 GB RAM, NVIDIA RTX 3060 12 GB.
- **Directory:** `experiments/python_reproduction/` contains training and evaluation scripts.
- **Parameters:** Defaults for MNIST live in `conformal_training/experiments/run_mnist.py`. Two shell scripts automate training and evaluation across seeds.

All models use a single-layer MLP (32 units, no hidden layers) trained for 50 epochs.

### Hyperparameters

| Variant                  |   LR | Batch | Temp. | Size Wt | Coverage Loss  | Loss Tf. | Size Tf. | RNG   | Method          |
| ------------------------ | ---: | ----: | ----: | ------: | -------------- | -------- | -------- | ----- | --------------- |
| **Baseline**             | 0.05 |   100 |     – |       – | –              | –        | –        | –     | –               |
| **Conformal Training**   | 0.05 |   500 |   0.5 |    0.01 | none           | log      | identity | False | threshold\_logp |
| **Group Zero / One**     | 0.01 |   100 |     1 |     0.5 | classification | log      | identity | False | threshold\_logp |
| **Singleton Zero / One** | 0.01 |   100 |     1 |     0.5 | classification | log      | identity | False | threshold\_logp |
| **Group Size 0 / 1**     | 0.05 |   500 |   0.5 |    0.01 | none           | log      | identity | False | threshold\_logp |
| **Class Size\_**\*       | 0.05 |   500 |   0.5 |    0.01 | none           | log      | identity | False | threshold\_logp |

`\*` denotes each class-specific experiment (`class_size_0` through `class_size_9`) which share the above Conformal Training settings.

### Julia Implementation [^julia]

This part of the reproduction was carried out by Phuong Ho with Julia. All code lives in the `conftr-julia/` directory of our GitHub repository and builds on the public `ConformalPrediction.jl` package plus a few lightweight helper scripts.

#### Installation pain points

Although `ConformalPrediction.jl` installs cleanly from the Julia registry, the first full run exposed several issues once we moved beyond toy tabular data:

- **Type and shape mismatches for images** The `:simple_inductive` mode presumes `Tables` input, so 3-D image tensors ($H\times W\times C$) fail when the code tries to coerce them to a matrix. Internally, `MLJ` cannot convert `Image` objects to `Matrix`, producing dimension and element-type errors.
- **Minor API drift** - A few calls broke after recent updates to `Flux` and CUDA, producing deprecation warnings or outright failures.

All fixes are clearly flagged with `# Code Modification` comments in the sources.

#### Key files we added

| File                   | Role                                                                                                                                                        |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `builder.jl`           | Assembles the base model (e.g. a small CNN for MNIST) and wraps it with the ConfTr layer.                                                                   |
| `vision_experiment.jl` | End-to-end script that loads a vision dataset, runs conformal training, and logs metrics; swapping to another dataset (e.g. CIFAR-10) is a one-line change. |
| `utils.jl`             | Lightweight helpers (metrics, logging, seeding).                                                                                                            |

#### Fix to the classification loss

The official docs never show how to enable the classification-shaping loss $\mathcal{L}_{\text{class}}$ from Stutz et al [^stutz2021learning]. Further inspection revealed a small bug that effectively disabled this term.

> **Note.** We audited only the classification pipeline. Regression paths compile but remain untested and may harbour similar issues.

## Results and Discussion

### MNIST with Official Python Implementation

We now have these **preliminary results** from the official Python implementation. Results on MNIST (seeds 0–3, α = 0.01), from `eval_results.csv`:

| Method                 | ThR Inefficiency (size) |     APS Avg. Set Size |
| ---------------------- | ----------------------: | --------------------: |
| **Baseline**           |            2.23 ± 0.015 |          2.50 ± 0.010 |
| **Conformal Training** |    2.16 ± 0.021 (–3.3%) | 8.92 ± 0.90 (+257.4%) |

> **Notes:** \
> • Statistics are means ± standard deviation over four random seeds. \
> • Inefficiency matches the original paper; APS set size appears inflated due to incomplete runs and will be verified in a full rerun.

### MNIST with Julia Implementation

#### Model Configurations

| Model                            | Architecture (input &rarr; output)                                                                                                                                                                                                 | Notes                                                                                                                                           |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Linear**                       | `Flatten` &rarr; `Dense(10)`                                                                                                                                                                                                       | A single fully-connected layer that maps the 784-dimensional pixel vector straight to the 10 logits (no hidden activations).                    |
| **2-Layer MLP**                  | `Flatten` &rarr; `Dense(128)` &rarr; `ReLU` &rarr; `Dense(10)`                                                                                                                                                                     | Adds one hidden layer with 128 units and ReLU non-linearity, giving the model capacity to learn simple nonlinear decision boundaries.           |
| **LeNet-5** [^lecun2002gradient] | `Conv(6×(5×5))` &rarr; _ReLU_ &rarr; `MaxPool(2×2)` &rarr; `Conv (16×(5×5), padding=2)` &rarr; _ReLU_ &rarr; `MaxPool(2×2)` &rarr; `Flatten` &rarr; `Dense(120)` &rarr; _ReLU_ &rarr; `Dense(84)` &rarr; `ReLU` &rarr; `Dense(10)` | Classic CNN for digit recognition. We keep the original kernel sizes but add padding to retain spatial dimensions after each convolution block. |

All models receive $28 \times 28$ grayscale MNIST images and output a logit for each of the 10 classes.

#### Ablation Results

<style type="text/css">
.tg  {border-collapse:collapse;border-color:#ccc;border-spacing:0;}
.tg td{background-color:#fff;border-bottom-width:1px;border-color:#ccc;border-style:solid;border-top-width:1px;
  border-width:0px;color:#333;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;
  word-break:normal;}
.tg th{background-color:#f0f0f0;border-bottom-width:1px;border-color:#ccc;border-style:solid;border-top-width:1px;
  border-width:0px;color:#333;font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;
  padding:10px 5px;word-break:normal;}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-amwm{font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-dzk6{background-color:#f9f9f9;text-align:center;vertical-align:top}
</style>
<table class="tg"><thead>
  <tr>
    <th class="tg-baqh" colspan="2" rowspan="3"></th>
    <th class="tg-amwm" colspan="3">Baseline</th>
    <th class="tg-amwm" colspan="6">ConfTr</th>
  </tr>
  <tr>
    <th class="tg-dzk6" rowspan="2">Linear</th>
    <th class="tg-dzk6" rowspan="2">2-layer MLP</th>
    <th class="tg-dzk6" rowspan="2">LeNet5 </th>
    <th class="tg-dzk6" colspan="2">Linear</th>
    <th class="tg-dzk6" colspan="2">2-layer MLP</th>
    <th class="tg-dzk6" colspan="2"><span style="font-style:normal">LeNet5</span></th>
  </tr>
  <tr>
    <th class="tg-baqh">THR</th>
    <th class="tg-baqh">APS</th>
    <th class="tg-baqh">THR</th>
    <th class="tg-baqh">APS</th>
    <th class="tg-baqh">THR</th>
    <th class="tg-baqh">APS</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-dzk6" rowspan="2">Inefficiency</td>
    <td class="tg-dzk6"><span style="font-style:normal">Train</span></td>
    <td class="tg-dzk6"><span style="font-style:normal">9.435</span></td>
    <td class="tg-dzk6">2.016</td>
    <td class="tg-dzk6">1.004</td>
    <td class="tg-dzk6"></td>
    <td class="tg-dzk6"></td>
    <td class="tg-dzk6"></td>
    <td class="tg-dzk6"></td>
    <td class="tg-dzk6"></td>
    <td class="tg-dzk6"></td>
  </tr>
  <tr>
    <td class="tg-baqh">Test</td>
    <td class="tg-baqh"><span style="font-style:normal">9.430</span></td>
    <td class="tg-baqh">2.104</td>
    <td class="tg-baqh">1.003</td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh"></td>
  </tr>
  <tr>
    <td class="tg-dzk6" rowspan="2">Accuracy</td>
    <td class="tg-dzk6">Train</td>
    <td class="tg-dzk6"><span style="font-style:normal">0.941</span></td>
    <td class="tg-dzk6">0.992</td>
    <td class="tg-dzk6"><span style="font-style:normal">0.978</span></td>
    <td class="tg-dzk6"></td>
    <td class="tg-dzk6"></td>
    <td class="tg-dzk6"></td>
    <td class="tg-dzk6"></td>
    <td class="tg-dzk6"></td>
    <td class="tg-dzk6"></td>
  </tr>
  <tr>
    <td class="tg-baqh">Test</td>
    <td class="tg-baqh">0.927</td>
    <td class="tg-baqh">0.973</td>
    <td class="tg-baqh">0.977</td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh"></td>
  </tr>
  <tr>
    <td class="tg-dzk6" rowspan="2">Classification loss</td>
    <td class="tg-dzk6">Train</td>
    <td class="tg-dzk6">-</td>
    <td class="tg-dzk6">-</td>
    <td class="tg-dzk6">-</td>
    <td class="tg-dzk6"></td>
    <td class="tg-dzk6"></td>
    <td class="tg-dzk6"></td>
    <td class="tg-dzk6"></td>
    <td class="tg-dzk6"></td>
    <td class="tg-dzk6"></td>
  </tr>
  <tr>
    <td class="tg-baqh">Test</td>
    <td class="tg-baqh">-</td>
    <td class="tg-baqh">-</td>
    <td class="tg-baqh">-</td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh"></td>
    <td class="tg-baqh"></td>
  </tr>
</tbody></table>

### Tabular Datasets with Official Python Implementation

#### German Credit

##### Experimental Setup

For the experiments, we wanted to reproduce the results from the German Credit table in the original paper. We decided to focus only on the baseline implemented in the codebase (`experiment_type = models`), as the Bellotti method was not provided. Moreover, we evaluated CP methods at test time using 10 random calibration/test splits, and we report results averaged across these splits. This is consistent with the evaluation protocol used in the original paper.

For the baseline (no _ConfTr_), we evaluated using Thr and APS post hoc. Additionally, we included an evaluation using ThrL, which we added to the codebase in order to compare with the corresponding results reported in the original paper. For _ConfTr_ and _ConfTr + L<sub>class</sub>_, we evaluated on Thr and APS. Although APS results for _ConfTr_ were not shown in the original paper for German Credit, we wanted to verify whether Thr would still yield slightly lower inefficiency compared to APS, as stated in the paper.

##### Results German Credit

Running our experiments resulted in the following table:

<figure style="text-align: center;">
  <img src="https://s3.hedgedoc.org/hd1-demo/uploads/f6fadcf9-bbfb-4678-9cdc-38b998ecca94.png" width="500" style="display: inline-block;">
  <figcaption><em>Table 2: Our experimental results on the German Credit dataset.</em></figcaption>
</figure>

We started by running the experiments for the **baseline** setup, to ensure our experimental procedure was correct. As can be seen from _Table 1_ and _Table 2_, the results for the different CP methods, in terms of both accuracy and inefficiency,are very similar. This suggests that our preprocessing and experimental setup are correct. As expected for a binary task, the resulting confidence sets remain small, and the absolute differences in inefficiency between methods are correspondingly limited.

When applying **ConfTr**, we observe that accuracy decreases slightly for Thr (which is expected), while inefficiency increases slightly compared to the Thr baseline (+0.04). In the original paper, a similar pattern was reported: accuracy decreased, and inefficiency increased slightly (+0.02). We therefore consider our results to be roughly consistent with those reported in the paper. Looking at the APS results, we see an increase in inefficiency, confirming that THR results in (slightly) lower inefficiency, as stated in the original paper.

Lastly, for **ConfTr + L<sub>class</sub>**, we would expect to see a decrease in inefficiency at a slight cost to accuracy. In our results, no decrease was observed when comparing to the Thr baseline. These differences could be due to random variation (we averaged results over only 10 seeds), or to slight differences in the experimental setup, as reproducing the exact setup from the paper can be challenging when the full code is not provided. When comparing to _ConfTr_ without class loss, we observe a small decrease in inefficiency (-0.04), which is consistent with the trend reported in the paper. As with the other results, Thr evaluation yielded lower inefficiency than APS evaluation, confirming the statement in the paper.

Finally, it is important to note that we did not modify any hyperparameters to improve results. The goal here was to assess whether the original paper's results could be reproduced using the same experimental setup.

#### New Data: Diabetes Prediction Dataset

To further extend our experiments, we also evaluated _ConfTr_ on the [Diabetes Prediction dataset](https://www.kaggle.com/datasets/marshalpatel3558/diabetes-prediction-dataset-legit-dataset), a publicly available medical dataset from Kaggle.

##### What did we do?

The Diabetes dataset contains patient information and classifies patients into three categories: Non-diabetic (N), Prediabetic (P), and Diabetic (D). The dataset contains 1000 rows, similar to German Credit. We chose this dataset because it represents a real-world medical prediction task, where providing reliable uncertainty estimates is particularly valuable.

We followed the same experimental setup as for German Credit, with the following minor adjustments:

- We adjusted the number of input features to match those in the Diabetes dataset.
- We updated the dataset statistics to reflect three classes (instead of two for German Credit).
- Again, some other functionalities were added so the experiments could be run on an additional dataset.

##### Results Diabetes

Running our experiments resulted in the following table:

<figure style="text-align: center;">
  <img src="https://s3.hedgedoc.org/hd1-demo/uploads/54478751-d191-483f-a225-f616a8844d20.png" width="500" style="display: inline-block;">
  <figcaption><em>Table 3: Our experimental results on the Diabetes Prediction dataset.</em></figcaption>
</figure>

The **baseline** results show high accuracy, indicating that a linear model performs well on this dataset. Inefficiency varies depending on the CP method used: Thr achieves the lowest inefficiency.

For **ConfTr** without class loss, we observe a further decrease in inefficiency when evaluated with Thr (from 1.43 to 1.37), which is a desirable outcome. As in our earlier experiments, Thr continues to yield lower inefficiency than APS.

For **ConfTr + L<sub>class</sub>**, we see inefficiency lower than the baseline for both Thr and APS. However, inefficiency is slightly higher compared to _ConfTr_ without class loss, which is similar to the trend we observed on German Credit.

Overall, the trends on this medical dataset are consistent with those observed on German Credit and in the original paper: _ConfTr_ can reduce inefficiency, Thr consistently yields lower inefficiency than APS, and adding class loss may provide mixed effects depending on the dataset.

## Assessing Reproducibility

In terms of reproducing results from the paper _Learning Optimal Conformal Classifiers_ by Stutz et al. (2022), there were some challenges in getting the original codebase running. The experiments were conducted on a 64-bit Linux workstation, which is not available to everyone. This also caused some conflicts when setting up their Conda environment.

For the German Credit results, the experimental setup in the paper was somewhat ambiguous. While the hyperparameters were listed, the small dataset leads to high variability in results. The code also contains more configuration options than are specified in the paper, which adds further ambiguity. Nevertheless, we were able to run the experiments and obtain results broadly consistent with those reported. In the context of reproducibility, it would be helpful if the authors provided code for all experiments, or a more fully specified experimental setup.

## Conclusions and Future Work

We reproduced and extended results from the paper _Learning Optimal Conformal Classifiers_ by Stutz et al. (2022). Specifically, we reproduced results on the German Credit dataset and ran additional experiments on a new dataset: the Diabetes Prediction dataset.

We reproduced the results for German Credit, and our findings were consistent with those reported in the original paper. However, some ambiguity remained, as not all parameters were clearly defined. For the sake of reproducibility, it would be helpful if the original paper provided more detail, or if complete code were included in the codebase.

In the future, it would be interesting to explore the application of _ConfTr_ to more medical datasets. In this work, we used the same experimental setup as for German Credit when running experiments on the Diabetes dataset. It could be valuable to investigate whether tuning hyperparameters for specific datasets yields further improvements in efficiency or calibration performance.

## Contributions

Razo re-ran the authors' official Python code on MNIST, verified coverage and set-size metrics, and documented all hyper-parameters and random seeds to enable exact replay (reproduction criteria). Doing this, he evaluated the original codebase.

Juul contributed by reproducing the German Credit results (which required partial reimplementation) and by extending the experiments to a new medical dataset (new data criteria).

Phuong ported ConfTr to Julia, patched `ConformalPrediction.jl`, and ran controlled ablations across multiple base models (Linear, MLP, LeNet5) (ablation study criteria).

## Reference

[^stutz2021learning]: Stutz, David, Ali Taylan Cemgil, and Arnaud Doucet. "Learning optimal conformal classifiers." arXiv preprint arXiv:2110.09192 (2021).
[^lecun2002gradient]: LeCun, Yann, et al. "Gradient-based learning applied to document recognition." Proceedings of the IEEE 86.11 (2002): 2278-2324.
[^julia]: Julia implementation <https://github.com/JuliaTrustworthyAI/ConformalPrediction.jl/tree/main>
[^hofmann1994german]: Hofmann, Hans. "Statlog (German Credit Data)." UCI Machine Learning Repository, 1994.
[^diabetes]: Diabetes Prediction dataset <https://www.kaggle.com/datasets/marshalpatel3558/diabetes-prediction-dataset-legit-dataset/data>
