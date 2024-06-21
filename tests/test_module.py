"""
Test module
python -m unittest -k ModuleTestVitSegmenter.test_loss_reconstruction test_module.py

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
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf, open_dict

BASE_DIR = Path(__file__).resolve().parent.parent

toy_data = {
  "fly_daart": str(BASE_DIR / "data/fly_daart")
}


TOY_CONFIG = {
"fly_daart": str(BASE_DIR / "scripts/script_configs/fly_daart.yaml"),
}

def load_cfg(data) -> dict:
  """Load all toy data config file without hydra."""
  with open(str(TOY_CONFIG[data]),'r',encoding='UTF-8') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
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

"""
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
    ("cls_structured_baseline", "cls_seq", "baseline", "fly_daart")
  )
  def test_cls_struct_module(self, module, classifier, data):
    torch.set_default_dtype(torch.float32)
    cfg = config_dict.ConfigDict()
    cfg.data_dir = toy_data[data]
    cfg.batch_size = 32
    cfg.num_workers = os.cpu_count()
    cfg.input_type = "markers"
    cfg.expt_ids = ["2019_06_26_fly2"]
    cfg.sequence_length = 5
    cfg.lambda_strong = 1
    cfg.lambda_weak = 1
    cfg.test_set = "base"
    cfg.num_classes = 5

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
    self.assertTrue(not torch.isnan(loss))
"""
class ModuleTestGrad(parameterized.TestCase):
  @parameterized.named_parameters(
    ("cls_sef_fly_grad", "fly_daart", "segmenter_module"),
    ("cls_segbsoft_fly_grad", "fly_daart", "segmenterBsoft_module"),
    ("cls_segbsoftweak_fly_grad", "fly_daart", "segmenterBsoftWeak_module"),
  )
  def test_load_datamodule(self, data, module):
    args = load_cfg(data)
    if args.seed is not None:
      seed_everything(args.seed)

    cfg = args.data_cfg
    cfg.num_workers = os.cpu_count()

    cfg.lambda_strong = 1
    cfg.lambda_weak = 1
    cfg.lambda_pred = 0.5
    cfg.batch_size = 1

    sequence_pad = 1
    # add
    extra_kwargs = {
      "sequence_pad": sequence_pad,
      "lambda_weak": cfg.lambda_weak,
      "lambda_strong": cfg.lambda_strong,
      "lambda_pred": cfg.lambda_pred,
    }

    # pad before and pad after
    ind_data = all_datasets[cfg.test_set](cfg, **extra_kwargs)
    ind_data.setup()
    train_dataloader = ind_data.train_dataloader()


    # Load module

    module_args = args.module_cfg
    OmegaConf.set_struct(module_args, True)

    with open_dict(module_args):
      # inherit from module:
      module_args.samples_per_class = ind_data.train_dataset.samples_per_class.numpy().tolist()
      module_args.sequence_pad = sequence_pad
      module_args.classifier_cfg.lambda_weak = cfg.lambda_weak
      module_args.classifier_cfg.lambda_strong = cfg.lambda_strong
      module_args.classifier_cfg.lambda_pred = cfg.lambda_pred
      module_args.classifier_cfg.num_classes = args.data_cfg.num_classes
      module_args.classifier_cfg.input_size = args.data_cfg.input_size

    module_args = {"hparams": module_args}

    model = all_modules[module](**module_args)

    # Take a grad step
    print('strong labels', ind_data.train_dataset.samples_per_class.numpy().tolist(), flush=True)
    print('weak labels', ind_data.train_dataset.weak_samples_per_class.numpy().tolist(), flush=True)

    total_strong_non_zero = 0
    total_weak_non_zero = 0
    for batch_idx, batch in enumerate(iter(train_dataloader)):
      num_non_zero = (batch['labels_strong'] != 0).sum()
      num_non_zero_weak = (batch['labels_weak'] != 0).sum()
      #print(f"{module} batch {batch_idx} n_strong labels", num_non_zero, flush=True)
      total_strong_non_zero +=1 if num_non_zero > 0 else 0
      total_weak_non_zero +=1 if num_non_zero_weak > 0 else 0
      #loss = model.training_step(batch, batch_idx)
      #print(f"{module} loss", loss, flush=True)

    print(f"{module} strong_non_zero batch {total_strong_non_zero}/{batch_idx}", flush=True)
    print(f"{module} weak_non_zero batch {total_weak_non_zero}/{batch_idx}", flush=True)


    loss = model.training_step(batch, batch_idx)
    print(f"{module} label_strong {num_non_zero} label_weak {num_non_zero_weak} loss {loss}", flush=True)


if __name__ == '__main__':
  absltest.main()