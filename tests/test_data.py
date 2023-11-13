"""
Test data laoder
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
from action.data.datamodule import BaseModule

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


class ModuleTestSimple(parameterized.TestCase):
  @parameterized.named_parameters(
    ("cls_baseline", "cls", "markers")
  )
  def test_load_datamodule(self, module, input_type):
    cfg = config_dict.ConfigDict()
    cfg.data_dir = toy_data["ibl"]
    cfg.batch_size = 32
    cfg.num_workers = os.cpu_count()
    cfg.input_type = input_type
    cfg.expt_ids = ["cortexlab_KS020_2020-02-06-001"]
    cfg.sequence_length = 5
    cfg.lambda_strong = 1
    cfg.lambda_weak = 0

    # pad before and pad after
    ind_data = BaseModule(cfg)
    ind_data.setup()
    train_dataloader = ind_data.train_dataloader()
    batch = next(iter(train_dataloader))
    assert input_type in batch.keys()



if __name__ == '__main__':
  absltest.main()