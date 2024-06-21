"""
Train a classifier on top of the features
"""
from typing import Optional
import pytorch_lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from ml_collections import config_dict
import os
import time
from action.data.dataloader import preproces_dataset
from action.data.data_utils import split_list_seq_classes

class ConcatDataset(torch.utils.data.Dataset):
  def __init__(self, *datasets):
    self.datasets = datasets

  def __getitem__(self, i):
    return tuple(d[i] for d in self.datasets)

  def __len__(self):
    return min(len(d) for d in self.datasets)

  @property
  def samples_per_class(self):
    return torch.stack([d.samples_per_class for d in self.datasets], axis=0).sum(0)


class BaseModule(pl.LightningDataModule):
  def __init__(self, args, **kwargs):
    super().__init__()
    self.hparams.update(args)
    self.extra_params = kwargs
    # input type is either pose or features

    self.seed = self.hparams.get("seed", None)
    self._make_generator_from_seed()

  def _make_generator_from_seed(self):
      if self.seed == None:
          self.generator_from_seed = None
      else:
          self.generator_from_seed = torch.Generator().manual_seed(self.seed)

  def setup(self, stage=None):
    # TODO: speed up by adding stage
    print("Begin setup data", flush=True)
    start_batch_time = time.time()

    # List of datasets with two keys: data and signals
    assert len(self.hparams.expt_ids) == 1
    self.dataset = preproces_dataset(self.hparams, self.extra_params)[0]
    #self.dataset = self.remove_empty_batches(self.dataset)
    self.split_into_train_val_test()
    self.calculate_samples_per_split()

    batch_processing_time = time.time() - start_batch_time
    print(f"Finished setup data in {batch_processing_time}", flush=True)

    if self.hparams.get("oversample_imbalanced_classes", False):
      assert 'labels_strong' in self.dataset.data.keys(), print("Oversampling imbalanced classes requires labels.", flush=True)
      self.set_train_sampler_cls()

  def set_train_sampler_cls(self):
    if hasattr(self.train_dataset, 'samples_per_class'):
      samples_weight = torch.from_numpy(1. / self.train_dataset.samples_per_class)
      self.train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    else:
      self.train_sampler = None

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

  def full_dataloader(self):
    # already shuffled
    dataloader = DataLoader(
      self.dataset,
      batch_size=self.hparams.batch_size,
      num_workers=self.hparams.num_workers,
      drop_last=False,
      pin_memory=True,
      generator=self.generator_from_seed,
    )
    return dataloader

  def _calculate_samples_p_class(self, dataset, x, label_key="labels_strong"):
    samples = np.zeros(self.hparams.num_classes)
    if len(x) > 0:
      unique_train, train_samples_per_class = np.unique(np.asarray(np.asarray(dataset.data[label_key])[np.sort(x)]), return_counts=True)
      samples[unique_train.astype(int)] = train_samples_per_class
    return samples

  def remove_empty_batches(self, dataset):
    # remove empty batches
    indices = []
    for i in range(len(dataset)):
      if dataset[i]['labels_strong'].sum() > 0:
        indices.append(i)
    return Subset(dataset, indices)

  def split_into_train_val_test(self):
    # split data into train test and val
    # TODO: merge split_trials function
    train_split = self.hparams.get("train_split", 0.8)
    val_split = self.hparams.get("val_split", 0.1)

    total_len = len(self.dataset)
    train_len = int(train_split * total_len)
    val_len = int(val_split * total_len)
    test_len = total_len - train_len - val_len

    # informed_data_split: calculate seq diversity to split data in train/val/test
    informed_data_split = self.hparams.get("informed_data_split", None)
    if 'labels_strong' in self.dataset.data.keys() and informed_data_split is not None:
      train_indices, \
      val_indices, \
      test_indices = split_list_seq_classes(self.dataset.data['labels_strong'],
                                            train_size=train_split,
                                            val_size=val_split)

      self.train_dataset = Subset(self.dataset, train_indices)
      self.val_dataset = Subset(self.dataset, val_indices)
      self.test_dataset = Subset(self.dataset, test_indices)
    else:
      self.train_dataset, \
      self.val_dataset, \
      self.test_dataset = torch.utils.data.random_split(
        self.dataset, [train_len, val_len, test_len],
      generator=self.generator_from_seed)

    self.train_dataset.n_sequences = len(self.train_dataset.indices)
    self.val_dataset.n_sequences = len(self.val_dataset.indices)
    self.test_dataset.n_sequences = len(self.test_dataset.indices)

  def calculate_samples_per_split(self):
    samples_p_class_torch = lambda x: torch.tensor(x).to(torch.int32)
    if 'labels_strong' in self.dataset.data.keys():
      train_samples = self._calculate_samples_p_class(self.dataset, self.train_dataset.indices, 'labels_strong')
      val_samples = self._calculate_samples_p_class(self.dataset, self.val_dataset.indices)
      test_samples = self._calculate_samples_p_class(self.dataset, self.test_dataset.indices)
      self.train_dataset.samples_per_class = samples_p_class_torch(train_samples)
      self.val_dataset.samples_per_class = samples_p_class_torch(val_samples)
      self.test_dataset.samples_per_class = samples_p_class_torch(test_samples)

    if 'labels_weak' in self.dataset.data.keys():
      train_samples = self._calculate_samples_p_class(self.dataset, self.train_dataset.indices, 'labels_weak')
      self.train_dataset.weak_samples_per_class = samples_p_class_torch(train_samples)


class ConcatDataModule(BaseModule):
  def __init__(self, args, **kwargs):
    super().__init__(args, **kwargs)


  def setup(self, stage=None):
    # TODO: speed up by adding stage
    # TODO: speed up by processing datasets together
    print("Preparing data", flush=True)
    start_batch_time = time.time()

    datasets = preproces_dataset(self.hparams, self.extra_params)
    #for dataset_idx, dataset in enumerate(datasets):
    #  datasets[dataset_idx] = self.remove_empty_batches(dataset)
    train_split = self.hparams.get("train_split", 0.8)
    val_split = self.hparams.get("val_split", 0.1)
    train_datasets = []
    val_datasets = []
    test_datasets = []
    train_samples_per_class = []
    val_samples_per_class = []
    test_samples_per_class = []
    train_weak_samples_per_class = []
    for dataset in datasets:
      total_len = len(dataset)
      train_len = int(train_split * total_len)
      val_len = int(val_split * total_len)
      test_len = total_len - train_len - val_len

      # split data into train test and val
      # TODO: merge split_trials function
      train_dataset, \
      val_dataset, \
      test_dataset = torch.utils.data.random_split(
        dataset, [train_len, val_len, test_len],
      generator=self.generator_from_seed)

      train_dataset.n_sequences = len(train_dataset.indices)
      val_dataset.n_sequences = len(val_dataset.indices)
      test_dataset.n_sequences = len(test_dataset.indices)

      if 'labels_strong' in dataset.data.keys():
        train_samples = self._calculate_samples_p_class(dataset, train_dataset.indices)
        val_samples = self._calculate_samples_p_class(dataset, val_dataset.indices)
        test_samples = self._calculate_samples_p_class(dataset, test_dataset.indices)

        train_samples_per_class.append(train_samples)
        val_samples_per_class.append(val_samples)
        test_samples_per_class.append(test_samples)

      train_datasets.append(train_dataset)
      val_datasets.append(val_dataset)
      test_datasets.append(test_dataset)

    self.train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    self.val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    self.test_dataset = torch.utils.data.ConcatDataset(test_datasets)
    #self.sequence_length = len(datasets[0].data['labels_strong'][0])

    samples_p_class_torch = lambda x: torch.tensor(np.stack(x, axis=0).sum(0)).to(torch.int32)

    if 'labels_strong' in dataset.data.keys():
      self.train_dataset.samples_per_class = samples_p_class_torch(train_samples_per_class)
      self.val_dataset.samples_per_class = samples_p_class_torch(val_samples_per_class)
      self.test_dataset.samples_per_class = samples_p_class_torch(test_samples_per_class)

    if 'labels_weak' in dataset.data.keys():
      train_samples_weak = self._calculate_samples_p_class(dataset, train_dataset.indices, 'labels_weak')
      train_weak_samples_per_class.append(train_samples_weak)
      self.train_dataset.weak_samples_per_class = samples_p_class_torch(train_weak_samples_per_class)

    batch_processing_time = time.time() - start_batch_time
    print("Finished preparing data in {}".format(batch_processing_time), flush=True)

    if stage == 'predict':
      self.dataset = torch.utils.data.ConcatDataset(datasets)


all_datasets = {
  "base": BaseModule,
  "concat_all": ConcatDataModule,
}


if __name__ == "__main__":
  # check datatype:
  cfg = config_dict.ConfigDict()
  cfg.data_dir = os.path.join(os.environ.get("LOCAL_PROJECTS_DIR"), "segment/action/data/ibl")
  cfg.batch_size = 32
  cfg.num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
  cfg.input_type = "markers"
  #cfg.expt_ids = ["cortexlab_KS020_2020-02-06-001", "wittenlab_ibl_witten_26_2021-01-27-002"]
  cfg.expt_ids = ["cortexlab_KS020_2020-02-06-001"]
  cfg.sequence_length = 5
  cfg.lambda_strong = 1
  cfg.lambda_weak = 0
  # pad before and pad after

  extra_cfg = config_dict.ConfigDict()
  extra_cfg.sequence_pad = 0

  ind_data = BaseModule(cfg, **extra_cfg)
  ind_data.prepare_data()
  ind_data.setup()
  train_dataset = ind_data.train_dataset

  start_batch_time = time.time()
  batch = next(iter(ind_data.train_dataloader()))
  batch_processing_time = time.time() - start_batch_time
  print(f"Batch processing time: {batch_processing_time} seconds", flush=True)
  print('build dataloader', flush=True)
  start_batch_time = time.time()
  batch = next(iter(ind_data.train_dataloader()))
  batch_processing_time = time.time() - start_batch_time
  print(f"Batch processing time: {batch_processing_time} seconds", flush=True)
