"""
Train a classifier on top of the features
"""
import pytorch_lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import OneHotEncoder
import torchvision
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from ml_collections import config_dict
from torch.nn.functional import one_hot
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
import os
from torch.utils.data.sampler import Sampler
import copy
import time
from torch.nn.utils.rnn import pad_sequence

from action.data import data_utils
from action.data.dataloader import preproces_dataset


class BaseModule(pl.LightningDataModule):
  def __init__(self, args):
    super().__init__()
    self.hparams.update(args)
    # input type is either pose or features

    self.seed = self.hparams.get("seed", None)
    self._make_generator_from_seed()

  def _make_generator_from_seed(self):
      if self.seed == None:
          self.generator_from_seed = None
      else:
          self.generator_from_seed = torch.Generator().manual_seed(self.seed)

  def setup(self, stage=None):
    # TODO very slow
    print("Preparing data", flush=True)
    start_batch_time = time.time()
    # Dictionary with two keys data and signals
    self.dataset = preproces_dataset(self.hparams)[0]
    total_len = len(self.dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len

    # split data into train test and val
    self.train_dataset, \
    self.val_dataset, \
    self.test_dataset = torch.utils.data.random_split(
      self.dataset, [train_len, val_len, test_len],
    generator=self.generator_from_seed)

    self.train_dataset.n_sequences = len(self.train_dataset.indices)
    self.val_dataset.n_sequences = len(self.val_dataset.indices)
    self.test_dataset.n_sequences = len(self.test_dataset.indices)

    samples_per_class = np.unique(np.asarray(self.dataset.data['labels_strong'])[self.train_dataset.indices], return_counts=True)[1]
    self.sequence_length = len(self.dataset.data['labels_strong'][0])
    self.train_dataset.samples_per_class = torch.tensor(samples_per_class).to(torch.int32)

    batch_processing_time = time.time() - start_batch_time
    print("Finished preparing data in {}".format(batch_processing_time), flush=True)

  def train_dataloader(self, shuffle=True):
    # already shuffled
    dataloader = DataLoader(
      self.train_dataset,
      batch_size=self.hparams.batch_size,
      num_workers=self.hparams.num_workers,
      shuffle=shuffle,
      drop_last=False,
      pin_memory=True,
      generator=self.generator_from_seed,
    )
    return dataloader

  def val_dataloader(self):
    dataloader = DataLoader(
      self.val_dataset,
      batch_size=self.hparams.batch_size,
      num_workers=self.hparams.num_workers,
      drop_last=False,
      pin_memory=True,
    )
    return dataloader

  def test_dataloader(self):
    dataloader = DataLoader(
      self.test_dataset,
      batch_size=self.hparams.batch_size,
      num_workers=self.hparams.num_workers,
      drop_last=False,
      pin_memory=True,
    )
    return dataloader

all_datasets = {
  "base": BaseModule,
}


if __name__ == "__main__":
  # check datatype:
  cfg = config_dict.ConfigDict()
  cfg.data_dir = os.path.join(os.environ.get("LOCAL_PROJECTS_DIR"), "segment/action/data/ibl")
  cfg.batch_size = 32
  cfg.num_workers = 8
  cfg.input_type = "markers"
  #cfg.expt_ids = ["cortexlab_KS020_2020-02-06-001", "wittenlab_ibl_witten_26_2021-01-27-002"]
  cfg.expt_ids = ["cortexlab_KS020_2020-02-06-001"]
  cfg.sequence_length = 5
  cfg.lambda_strong = 1
  cfg.lambda_weak = 0
  # pad before and pad after

  ind_data = BaseModule(cfg)
  ind_data.setup()
  train_dataset = ind_data.train_dataset
  print(train_dataset.dataset.data['markers'][0].dtype)

  start_batch_time = time.time()
  batch = next(iter(ind_data.train_dataloader()))
  batch_processing_time = time.time() - start_batch_time
  print(f"Batch processing time: {batch_processing_time} seconds")
  print('build dataloader')
  start_batch_time = time.time()
  batch = next(iter(ind_data.train_dataloader()))
  batch_processing_time = time.time() - start_batch_time
  print(f"Batch processing time: {batch_processing_time} seconds")
