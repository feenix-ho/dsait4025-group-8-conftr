# dsait4025-group-8-conftr

DSAIT 4025 reproduction of the paper "Conformal Prediction for Deep Learning" by Google DeepMind. Group 8.

## Python Reproduction

Source: <https://github.com/google-deepmind/conformal_training>

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
