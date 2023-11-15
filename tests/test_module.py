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
from action.modules import all_modules
from action.data.datamodule import all_datasets

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
    ("cls_baseline", "cls", "baseline")
  )
  def test_cls_module(self, module, classifier):

    cfg = create_cfg(module, classifier, input_size=10, num_classes=3)
    # Load  a simple model
    module_args = {"hparams": cfg}
    model = all_modules[cfg.module](**module_args)

    # Make a dummy dataset
    dataset = torch.utils.data.TensorDataset(torch.randn(10, 10), torch.randint(0, 3, (10,)))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

    loss = model.training_step(next(iter(dataloader)), 0)

    assert not torch.isnan(loss)


class ModuleTestStruct(parameterized.TestCase):
  @parameterized.named_parameters(
    ("cls_structured_baseline", "cls_seq", "baseline")
  )
  def test_cls_struct_module(self, module, classifier):
    torch.set_default_dtype(torch.float32)
    cfg = config_dict.ConfigDict()
    cfg.data_dir = toy_data["ibl"]
    cfg.batch_size = 32
    cfg.num_workers = os.cpu_count()
    cfg.input_type = "markers"
    cfg.expt_ids = ["cortexlab_KS020_2020-02-06-001"]
    cfg.sequence_length = 5
    cfg.lambda_strong = 1
    cfg.lambda_weak = 1
    cfg.test_set = "base"

    extra_kwargs = {
      "sequence_pad": 1,
      "lambda_weak" : 0.5,
      "lambda_strong": 1,
      "lambda_pred": 0.5,
    }
    # Load  a simple model
    ind_data = all_datasets[cfg.test_set](cfg, **extra_kwargs)
    ind_data.setup()
    train_dataloader = ind_data.train_dataloader()
    batch = next(iter(train_dataloader))
    # make sure it is on the same precision
    batch["markers"] = batch["markers"].float()

    # Load  a simple model
    input_size = batch["markers"].shape[-1]
    num_classes = batch['labels_strong'].shape[-1]
    cfg = create_cfg(module, classifier, input_size=input_size, num_classes=num_classes)
    module_args = {"hparams": cfg}
    model = all_modules[cfg.module](**module_args)

    # Take a grad step
    loss = model.training_step(batch, 0)
    assert not torch.isnan(loss)


if __name__ == '__main__':
  absltest.main()