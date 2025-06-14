# Reproducing and Extending Results from "Learning Optimal Conformal Classifiers"* 

|        Name        |               Email                | StudentID |
| :----------------: | :--------------------------------: | :-------: |
| Ho Thi Ngoc Phuong | <HoThiNgocPhuong@student.tudelft.nl> |  6172970  |
|  Juul Schnitzler   |  <j.b.schnitzler@student.tudelft.nl>   |  5094917  |
|  Razo van Berkel   |      <r.q.berkel@student.tudelft.nl> | 6330029 |

The reproduction code is available on Github [here](https://github.com/feenix-ho/dsait4025-group-8-conftr).

## Introduction 
In recent years, we have seen significant progress in machine and deep learning. Models now achieve high accuracy on many tasks, and this continues to improve even further. However, high accuracy alone does not provide guarantees for safe deployment, especially in high-stakes applications. We need some measure of how certain or uncertain a model is about its predictions, particularly in classification problems. Ideally, we want a formal guarantee that quantifies this uncertainty. Conformal prediction (CP) addresses this need by using the classifier’s predictions (e.g., its probability estimates) to construct confidence sets that contain the true class with a predefined probability.

CP has often been applied as a separate processing step after training, which prevents the model from adapting to the prediction of confidence sets. The paper _Learning Optimal Conformal Classifiers_ by Stutz et al. (2022) addresses this limitation by introducing methods to include CP directly in the training process, enabling end-to-end training with the conformal wrapper. Their approach, called _conformal training (ConfTr)_, integrates conformalization steps into the mini-batch training loop. The results demonstrate that _ConfTr_ reduces inefficiency (i.e., the size of the confidence sets) of common conformal predictors. Moreover, the authors argue that *ConfTr* allows shaping the confidence sets at test time, for example by reducing class-conditional inefficiency. 

In this blog post, we present a reproduction and extension of some of the results from the _Learning Optimal Conformal Classifiers_ paper. Specifically, we focus on reproducing their results for the tabular German Credit dataset. In their codebase, no implementation was provided for experiments on this dataset. Therefore, we partially reimplemented the preprocessing pipeline and experimental setup for the German Credit dataset. Reproducing results for this dataset is particularly valuable, as it allows us to assess whether the paper provides sufficient detail to reproduce its findings, since the original code for this dataset was not included. 

Additionally, we extend the experiments by evaluating *ConfTr* on a new dataset: a medical tabular dataset for diabetes prediction based on patient information. While the paper highlights the relevance of CP for high-stakes AI applications such as medical diagnosis, no medical datasets were included in their experiments. We found it interesting to explore how *ConfTr* performs in this domain.

## ConfTr: Recap of the Original Paper
As we already mentioned in the introduction, the key idea behind *ConfTr* is to bring conformal prediction (CP) into the training loop. Here, we briefly recap how this process works during training. For full details, please consult the original paper by Stutz et al. (2022).

*ConfTr* simulates CP during training on each mini-batch. Specifically, each mini-batch is split in half:
•	One half is used for the calibration step, where a threshold is computed based on the model's predicted probabilities for the calibration samples.
•	The other half is used for the prediction step, where confidence sets are constructed using the threshold obtained in the first step.
The model is then updated using a loss that combines a size loss, which encourages the model to produce smaller confidence sets, and optionally a classification loss that can shape the content of the confidence sets (e.g., penalizing certain classes). The following figure from the original paper illustrates this process:

<figure style="text-align: center;"> 
 <img src="https://s3.hedgedoc.org/hd1-demo/uploads/165c791b-ce60-4024-9d26-dec8207b43f0.png" alt="*ConfTr* diagram from Stutz et al. (2022)"> 
 <figcaption><em>Figure from Stutz et al. (2022), illustrating the conformal training process.</em></figcaption>
</figure> 

The calibration and prediction steps are implemented in a differentiable way (using smooth approximations), so that the entire process can be optimized end-to-end with standard gradient-based methods. After training with *ConfTr*, the model can still be used with any standard CP method at test time, meaning the CP coverage guarantee is preserved.

## Reproduction: Julia implementation MNIST

## Reproduction: Python implementation MNIST

## Reproduction: Python implementation German Credit
### Paper's Experiment on German Credit

The original paper evaluates *ConfTr* on several datasets, including the German Credit dataset. For this binary classification task (predicting "good" vs. "bad" credit risk), the authors compare:
- A standard cross-entropy baseline
- Bellotti (2021), trained using ThrL (thresholding on raw logits)
- ConfTr, trained with and without an additional classification loss (L_class)

At test time, two CP methods are used: threshold CP (Thr) and adaptive prediction sets (APS). While Bellotti uses ThrL, *ConfTr* uses ThrLP (thresholding on log-probabilities). The main evaluation metrics are inefficiency (average size of the confidence sets) and accuracy. Since the task is binary, expected confidence sets are small, and potential efficiency gains are limited.


<figure style="text-align: center;">
  <img src="https://s3.hedgedoc.org/hd1-demo/uploads/f67fd31d-6725-4ea6-bb79-051e1214d6e7.png" width="500" style="display: inline-block;">
  <figcaption><em>Table 1: Experimental results from Stutz et al. (2022) on the German Credit dataset.</em></figcaption>
</figure>




### What did we do?

As mentioned before, the experiments were only partially implemented in the [original codebase](https://github.com/google-deepmind/conformal_training) of the paper. Specifically, for tabular data there was no specific code available to run experiments straight away. Therefore, we needed to partially reimplement the code, which we did for the German Credit dataset. 

Why German Credit? This dataset is openly available, and the paper provides both hyperparameters and a clear description of the experimental setup. We thought it would be interesting to see whether we could reproduce the results from the paper based solely on the codebase and the descriptions provided in the paper.

<!-- Code used / modified. -->

For the reimplementation, the following steps were performed:

* Manually downloading the *German Credit* dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) (1994).
* *Preprocessing* the data in `preprocess_german.py`, which included one-hot encoding and scaling (whitening the features) as described in the original paper.
* Adding a function `create_tabular_split` to `data.py` to create train/val/test splits for tabular data, using the same 70/10/20 split defined for German Credit in the original paper.
* Adding `run_german_credit.py` to define the experimental setup for German Credit. For this, we looked at the structure of other available files (e.g., `run_mnist.py`, `run_wine_quality.py`) as well as the description in the original paper. The most important elements include the following:
    * We used the exact same *ConfTr* hyperparameters as defined in Table B of *Appendix F: Experimental Setup* from the original paper, without additional tuning.
    * We used a multilayer perceptron with 0 layers to imitate a *linear model*, as no linear architecture was available in the codebase.
    * We derived the *loss matrix* from the original documentation by Prof. Hofmann (UCI Machine Learning Repository, 1994).
    * For *ConfTr*, we used ThrLP for training by setting `conformal.method = 'threshold_logp'`, which uses log-probabilities as conformity scores.
    * For *ConfTr + L<sub>class</sub>*, we also used THR<sub>LP</sub> for training and additionally used classification loss by setting `conformal.use_class_loss = True`.

* Some smaller functionalities were added to make the experiment possible (e.g., defining the statistics for German Credit in `data_utils.py`), but the above sums up the most relevant contributions.

#### Experimental Setup

For the experiments, we wanted to reproduce the results from the German Credit table in the original paper. We decided to focus only on the baseline implemented in the codebase (`experiment_type = models`), as the Bellotti method was not provided. Moreover, we evaluated CP methods at test time using 10 random calibration/test splits, and we report results averaged across these splits. This is consistent with the evaluation protocol used in the original paper.
    
For the baseline (no *ConfTr*), we evaluated using Thr and APS post hoc. Additionally, we included an evaluation using ThrL, which we added to the codebase in order to compare with the corresponding results reported in the original paper. For *ConfTr* and *ConfTr + L<sub>class</sub>*, we evaluated on Thr and APS. Although APS results for *ConfTr* were not shown in the original paper for German Credit, we wanted to verify whether Thr would still yield slightly lower inefficiency compared to APS, as stated in the paper. 
    
### Results German Credit
Running our experiments resulted in the following table:
    
<figure style="text-align: center;">
  <img src="https://s3.hedgedoc.org/hd1-demo/uploads/f6fadcf9-bbfb-4678-9cdc-38b998ecca94.png" width="500" style="display: inline-block;">
  <figcaption><em>Table 2: Our experimental results on the German Credit dataset.</em></figcaption>
</figure>
    
We started by running the experiments for the **baseline** setup, to ensure our experimental procedure was correct. As can be seen from *Table 1* and *Table 2*, the results for the different CP methods, in terms of both accuracy and inefficiency,are very similar. This suggests that our preprocessing and experimental setup are correct. As expected for a binary task, the resulting confidence sets remain small, and the absolute differences in inefficiency between methods are correspondingly limited.
    
When applying **ConfTr**, we observe that accuracy decreases slightly for Thr (which is expected), while inefficiency increases slightly compared to the Thr baseline (+0.04). In the original paper, a similar pattern was reported: accuracy decreased, and inefficiency increased slightly (+0.02). We therefore consider our results to be roughly consistent with those reported in the paper. Looking at the APS results, we see an increase in inefficiency, confirming that THR results in (slightly) lower inefficiency, as stated in the original paper.
    
Lastly, for **ConfTr + L<sub>class</sub>**, we would expect to see a decrease in inefficiency at a slight cost to accuracy. In our results, no decrease was observed when comparing to the Thr baseline. These differences could be due to random variation (we averaged results over only 10 seeds), or to slight differences in the experimental setup, as reproducing the exact setup from the paper can be challenging when the full code is not provided. When comparing to *ConfTr* without class loss, we observe a small decrease in inefficiency (-0.04), which is consistent with the trend reported in the paper. As with the other results, Thr evaluation yielded lower inefficiency than APS evaluation, confirming the statement in the paper.
    
Finally, it is important to note that we did not modify any hyperparameters to improve results. The goal here was to assess whether the original paper’s results could be reproduced using the same experimental setup.
    
    
## New Data: Diabetes Prediction Dataset

To further extend our experiments, we also evaluated *ConfTr* on the [Diabetes Prediction dataset](https://www.kaggle.com/datasets/marshalpatel3558/diabetes-prediction-dataset-legit-dataset), a publicly available medical dataset from Kaggle.

### What did we do?

The Diabetes dataset contains patient information and classifies patients into three categories: Non-diabetic (N), Prediabetic (P), and Diabetic (D). The dataset contains 1000 rows, similar to German Credit. We chose this dataset because it represents a real-world medical prediction task, where providing reliable uncertainty estimates is particularly valuable.

We followed the same experimental setup as for German Credit, with the following minor adjustments:

* We adjusted the number of input features to match those in the Diabetes dataset.
* We updated the dataset statistics to reflect three classes (instead of two for German Credit).
* Again, some other functionalities were added so the experiments could be run on an additional dataset.
 

### Results Diabetes

Running our experiments resulted in the following table:

<figure style="text-align: center;">
  <img src="https://s3.hedgedoc.org/hd1-demo/uploads/54478751-d191-483f-a225-f616a8844d20.png" width="500" style="display: inline-block;">
  <figcaption><em>Table 3: Our experimental results on the Diabetes Prediction dataset.</em></figcaption>
</figure>

The **baseline** results show high accuracy, indicating that a linear model performs well on this dataset. Inefficiency varies depending on the CP method used: Thr achieves the lowest inefficiency. 

For **ConfTr** without class loss, we observe a further decrease in inefficiency when evaluated with Thr (from 1.43 to 1.37), which is a desirable outcome. As in our earlier experiments, Thr continues to yield lower inefficiency than APS.

For **ConfTr + L<sub>class</sub>**, we see inefficiency lower than the baseline for both Thr and APS. However, inefficiency is slightly higher compared to *ConfTr* without class loss, which is similar to the trend we observed on German Credit.

Overall, the trends on this medical dataset are consistent with those observed on German Credit and in the original paper: *ConfTr* can reduce inefficiency, Thr consistently yields lower inefficiency than APS, and adding class loss may provide mixed effects depending on the dataset.


## Assessing Reproducibility

In terms of reproducing results from the paper *Learning Optimal Conformal Classifiers* by Stutz et al. (2022), there were some challenges in getting the original codebase running. The experiments were conducted on a 64-bit Linux workstation, which is not available to everyone. This also caused some conflicts when setting up their Conda environment.

For the German Credit results, the experimental setup in the paper was somewhat ambiguous. While the hyperparameters were listed, the small dataset leads to high variability in results. The code also contains more configuration options than are specified in the paper, which adds further ambiguity. Nevertheless, we were able to run the experiments and obtain results broadly consistent with those reported. In the context of reproducibility, it would be helpful if the authors provided code for all experiments, or a more fully specified experimental setup.

   
## Conclusions and Future Work

We reproduced and extended results from the paper *Learning Optimal Conformal Classifiers* by Stutz et al. (2022). Specifically, we reproduced results on the German Credit dataset and ran additional experiments on a new dataset: the Diabetes Prediction dataset.

We reproduced the results for German Credit, and our findings were consistent with those reported in the original paper. However, some ambiguity remained, as not all parameters were clearly defined. For the sake of reproducibility, it would be helpful if the original paper provided more detail, or if complete code were included in the codebase.

In the future, it would be interesting to explore the application of *ConfTr* to more medical datasets. In this work, we used the same experimental setup as for German Credit when running experiments on the Diabetes dataset. It could be valuable to investigate whether tuning hyperparameters for specific datasets yields further improvements in efficiency or calibration performance.
    
    
## Contributions

* _Juul_ contributed by reproducing the German Credit results (which required partial reimplementation) and by extending the experiments to a new medical dataset.
* _Razo_ re-ran the authors' official Python code on MNIST, verified coverage and set-size metrics, and documented all hyper-parameters and random seeds to enable exact replay (reproduction criteria). Doing this, he evaluated the original codebase.
* _Phuong_ ported ConfTr to Julia, patched `ConformalPrediction.jl`, and ran controlled ablations across multiple base models (Linear, MLP, LeNet5) (ablation study criteria).



## References

* Stutz, D., Bates, S., Rabanser, S., Hein, M., & Ermon, S. (2022). Learning Optimal Conformal Classifiers. *ICLR 2022.* https://arxiv.org/abs/2110.09192
* Bellotti, T. (2021). Learning Probabilistic Set Predictors with Guarantees. *arXiv preprint arXiv:2103.10288.* https://arxiv.org/abs/2103.10288
* UCI Machine Learning Repository. (1994). https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
* Patel, M. (n.d.). Diabetes Prediction Dataset. Kaggle. https://www.kaggle.com/datasets/marshalpatel3558/diabetes-prediction-dataset-legit-dataset



