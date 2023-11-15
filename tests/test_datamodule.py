"""
Test data laoder
python -m unittest -k ModuleTestMultiple.test_load_datamodule test_datamodule.py
"""
from absl.testing import absltest
from absl.testing import parameterized
import yaml
from pathlib import Path
from omegaconf import OmegaConf
import os
from action.data.datamodule import all_datasets
from pytorch_lightning import seed_everything


BASE_DIR = Path(__file__).resolve().parent.parent

TMP_NUM_CPUS = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))

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
  with open(str(TOY_CONFIG[data]),'r',encoding='UTF-8') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
  return OmegaConf.create(cfg)


class ModuleTestSimple(parameterized.TestCase):
  @parameterized.named_parameters(
    ("cls_baseline_fly_daart", "fly_daart", "markers"),
  )
  def test_load_datamodule(self, data, input_type):
    args = load_cfg(data)
    if args.seed is not None:
      seed_everything(args.seed)
    cfg = args.data_cfg
    cfg.num_workers = TMP_NUM_CPUS
    cfg.input_type = input_type
    cfg.lambda_strong = 1
    cfg.lambda_weak = 1
    cfg.batch_size = 1

    extra_kwargs = {
      "sequence_pad": 1,
      "lambda_weak": 0.5,
      "lambda_strong": 1,
      "lambda_pred": 0.5,
    }

    # construct data
    ind_data = all_datasets[cfg.test_set](cfg, **extra_kwargs)
    ind_data.setup()
    train_dataloader = ind_data.train_dataloader()
    batch = next(iter(train_dataloader))
    assert input_type in batch.keys()
    assert len(iter(train_dataloader))*cfg.batch_size == ind_data.train_dataset.n_sequences


  @parameterized.named_parameters(
    ("cls_baseline_fly_daart_pdataset", "fly", "markers"),
  )
  def test_each_datamodule(self, data, input_type):

    args = load_cfg(data)
    if args.seed is not None:
      seed_everything(args.seed)
    cfg = args.data_cfg
    cfg.input_type = input_type
    cfg.test_set = "base"
    cfg.num_workers = TMP_NUM_CPUS
    cfg.batch_size = 1
    cfg.train_split = 0
    cfg.val_split = 0

    extra_kwargs = {
      "sequence_pad": 1,
      "lambda_weak": 0.5,
      "lambda_strong": 1,
      "lambda_pred": 0.5,
    }

    expt_ids = [
      "2019_06_26_fly2",
      "2019_08_07_fly2",
      "2019_08_08_fly1",  # no cls 2
      "2019_08_20_fly2",
      "2019_10_21_fly1",  # no cls 5
       "2019_08_14_fly1",
      "2019_10_14_fly3",
      "2019_10_14_fly2",
      "2019_08_20_fly3",
      "2019_10_10_fly3",
    ]
    for expt_id in expt_ids:
      cfg.expt_ids = [expt_id]
          # construct data
      ind_data = all_datasets[cfg.test_set](cfg, **extra_kwargs)
      ind_data.setup()
      train_dataloader = ind_data.test_dataloader()
      batch = next(iter(train_dataloader))
      assert input_type in batch.keys()
      assert len(iter(train_dataloader)) * cfg.batch_size == ind_data.test_dataset.n_sequences


class ModuleTestMultiple(parameterized.TestCase):
  @parameterized.named_parameters(
    ("cls_baseline_fly", "fly", "markers"),
  )
  def test_load_datamodule(self, data, input_type):
    args = load_cfg(data)
    if args.seed is not None:
      seed_everything(args.seed)

    cfg = args.data_cfg
    cfg.num_workers = TMP_NUM_CPUS
    cfg.input_type = input_type
    cfg.lambda_strong = 1
    cfg.lambda_weak = 1
    cfg.batch_size = 1

    # add
    extra_kwargs = {
      "sequence_pad": 1,
      "lambda_weak": 0.5,
      "lambda_strong": 1,
      #"lambda_pred": 0.5,
    }

    # pad before and pad after
    ind_data = all_datasets[cfg.test_set](cfg, **extra_kwargs)
    ind_data.setup()
    train_dataloader = ind_data.train_dataloader()
    batch = next(iter(train_dataloader))
    assert input_type in batch.keys()
    assert len(iter(train_dataloader))*cfg.batch_size == len(ind_data.train_dataset)

    ind_data.setup(stage="predict")
    train_dataloader = ind_data.full_dataloader()
    batch = next(iter(train_dataloader))
    assert input_type in batch.keys()
    self.assertTrue((ind_data.train_dataset.samples_per_class>0).sum() == cfg.num_classes)
    self.assertTrue((ind_data.val_dataset.samples_per_class>0).sum() == cfg.num_classes)
    self.assertTrue((ind_data.test_dataset.samples_per_class>0).sum() == cfg.num_classes)



if __name__ == '__main__':
  absltest.main()
