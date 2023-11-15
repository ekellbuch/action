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
  "ibl": str(BASE_DIR / "data/ibl"),
  "fly": str(BASE_DIR / "data/fly"),
}

TOY_CONFIG = {
  "fly": str(BASE_DIR / "scripts/script_configs/fly.yaml"),
  "fly_daart": str(BASE_DIR / "scripts/script_configs/fly_daart.yaml"),
  "ibl": str(BASE_DIR / "scripts/script_configs/ibl_paw.yaml"),
}

def load_cfg(data) -> dict:
  """Load all toy data config file without hydra."""
  cfg = yaml.load(open(str(TOY_CONFIG[data])), Loader=yaml.FullLoader)
  return OmegaConf.create(cfg)


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
    ("cls_baseline_fly_daart", "fly_daart", "markers"),
    ("cls_baseline_ibl", "ibl", "markers"),
    ("cls_baseline_fly", "fly", "markers"),
  )
  def test_load_datamodule(self, data, input_type):
    args = load_cfg(data)
    cfg = args.data_cfg
    cfg.num_workers = os.cpu_count()
    cfg.input_type = input_type
    cfg.lambda_strong = 1
    cfg.lambda_weak = 1
    cfg.batch_size = 1

    # add
    #cfg.classifier = "segmenter"
    extra_kwargs = {
      "sequence_pad": 1,
      "lambda_weak": 0.5,
      "lambda_strong": 1,
      "lambda_pred": 0.5,
    }

    # pad before and pad after
    ind_data = BaseModule(cfg, **extra_kwargs)
    ind_data.setup()
    train_dataloader = ind_data.train_dataloader()
    batch = next(iter(train_dataloader))
    print(batch.keys(), flush=True)
    assert input_type in batch.keys()
    assert len(iter(train_dataloader))*cfg.batch_size == ind_data.train_dataset.n_sequences


if __name__ == '__main__':
  absltest.main()