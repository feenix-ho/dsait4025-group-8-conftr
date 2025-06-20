# ConfTr Experiments - Group 8 (DSAIT 4025)

Reproduction of the paper by Google DeepMind: "Learning Optimal Conformal Classifiers". [Paper link](https://arxiv.org/abs/2110.09192). \
DSAIT 4025 - Group 8, TU Delft, June 2025.

## Team Members

| Name               | Email                                | Student ID |
| ------------------ | ------------------------------------ | ---------- |
| Ho Thi Ngoc Phuong | <HoThiNgocPhuong@student.tudelft.nl> | 6172970    |
| Juul Schnitzler    | <j.b.schnitzler@student.tudelft.nl>  | 5094917    |
| Razo van Berkel    | <r.q.berkel@student.tudelft.nl>      | 6330029    |

## Repository Structure

Source for `conformal_training/`: <https://github.com/google-deepmind/conformal_training>; repo was altered to also work wiht German Credit and Diabetes Prediction datasets.

## MNIST - Julia

See [experiments/conftr-julia/README.md](experiments/conftr-julia/README.md) for details on the Julia implementation and reproduction.

## MNIST - Python Reproduction

See [experiments/python_reproduction/README.md](experiments/python_reproduction/README.md) for details on the scripts used to reproduce the results from the original paper using the .

## German Credit and Diabetes Prediction - Python

See ![experiments/other_datasets/README.md](experiments/other_datasets/README.md) for details on running the experiments on the German Credit and Diabetes Prediction datasets using the Python implementation. The implementation is in `conformal_training/` (the original library, extended). The bash commands to run the experiments are provided in the `README.md` file in that directory.

## Full Blog Post

The full blog post with details on the experiments, results, and analysis can be found in this repo: ![Blog](BLOG.md).
