"""Experiment definitions for German credit."""
from typing import Tuple, Dict, Any, Optional

import ml_collections as collections

def get_parameters(
    experiment: str,
    sub_experiment: str,
    config: collections.ConfigDict,
) -> Tuple[collections.ConfigDict, Optional[Dict[str, Any]]]:
  """Get parameters for German credit experiments.

  Args:
    experiment: experiment to run
    sub_experiment: sub experiment, e.g., parameter to tune
    config: experiment configuration

  Returns:
    Training configuration and parameter sweeps
  """
  # Use a linear model, just like in the paper
  config.architecture = 'mlp'
  config.mlp.layers = 0
  config.epochs = 100

  config.dataset = 'german_credit'
  config.input_size = 25                  # Number of features in dataset (target variable not included)
  config.loss_matrix = [[0, 1], [5, 0]]   # Derived from original German credit documentation
  config.conformal.loss_transform = 'identity'

  parameter_sweep = None


  if experiment == 'models':
    config.learning_rate = 0.05
    config.batch_size = 100

  elif experiment == 'conformal':
    config.mode = 'conformal'

    config.conformal.size_transform = 'identity'
    config.conformal.rng = False
    config.conformal.loss_matrix = config.loss_matrix

    # ConfTr
    if sub_experiment == 'training':
      # Use ThrLP for training
      config.conformal.method = 'threshold_logp'

      # Same hyperparameters as defined in paper
      config.batch_size = 200
      config.learning_rate = 0.05
      config.conformal.temperature = 1.0
      config.conformal.size_weight = 5.0
      config.conformal.kappa = 1

      config.conformal.coverage_loss = 'none'
      config.conformal.cross_entropy_weight = 1.0

    # ConfTr + L_class
    elif sub_experiment == 'training_Lclass':
      # Again we use ThrLP for training
      config.conformal.method = 'threshold_logp'

      # Same hyperparameters as defined in paper
      config.batch_size = 400
      config.learning_rate = 0.05
      config.conformal.temperature = 0.1
      config.conformal.size_weight = 5.0
      config.conformal.kappa = 1

      config.conformal.cross_entropy_weight = 1.0

      # We want to add classification loss
      config.conformal.coverage_loss = 'classification'
      config.conformal.use_class_loss = True
      config.conformal.size_loss = 'valid'

    else:
      raise ValueError('Invalid conformal sub experiment.')
  else:
    raise ValueError('Experiment not implemented.')
  return config, parameter_sweep
