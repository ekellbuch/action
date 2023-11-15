"""
Test module
"""
from absl.testing import absltest
from absl.testing import parameterized
from ml_collections import config_dict
import sys
import yaml
import torch
from pathlib import Path
from omegaconf import OmegaConf
import os
from action.models.base import all_classifiers
BASE_DIR = Path(__file__).resolve().parent.parent

toy_data = {
  "ibl": str(BASE_DIR / "data/ibl")
}

def create_cfg(module, classifier, input_size=10, num_classes=3) -> dict:
  cfg = config_dict.ConfigDict()
  cfg.module = module
  cfg.classifier = classifier
  cfg.classifier_cfg = config_dict.ConfigDict()
  cfg.classifier_cfg.model_tier = "small"
  cfg.classifier_cfg.num_classes = num_classes
  cfg.classifier_cfg.input_size = input_size
  cfg.samples_per_class = None
  cfg.optimizer_cfg = config_dict.ConfigDict()
  cfg.optimizer_cfg.lr = 0.001
  return cfg

class ModelTestSimple(parameterized.TestCase):
  @parameterized.named_parameters(
    ("cls_segmenter", "segmenter")
  )
  def test_cls_model(self, classifier):
    hparams = {
      "segmenter": {
      'backbone': 'dtcn',  # architecture for encoder/decoder/predictor networks
      'input_size': 16,  # dimensionality of markers
      'num_classes': 6,  # number of classes
      'n_hid_layers': 2,  # hidden layers in network
      'n_hid_units': 32,  # hidden units per hidden layer
      'n_lags': 4,  # half-width of temporal convolution window
      'activation': 'lrelu',  # layer nonlinearity
      'lambda_weak': 0.5,  # hyperparam on classifying weak (heuristic) labels
      'lambda_strong': 1,  # hyperparam on classifying strong (hand) labels
      'lambda_pred': 0.5,  # hyperparam on one-step-ahead prediction
      'sequence_pad': 16,  # pad batches with extra data for convolutions
      'classifier': "segmenter",
      },
      "simple": {
      'model_tier' : 'small',

      }
    }

    # build model
    model = all_classifiers[classifier](hparams[classifier])

    assert model is not None

if __name__ == '__main__':
  absltest.main()