
from collections import OrderedDict
import h5py
import logging
import numpy as np
import os
import pandas as pd
import pickle
import torch
from torch.utils import data
from torch.utils.data import SubsetRandomSampler
from typing import List, Optional, Union
from typeguard import typechecked
from action.data.data_utils import load_marker_csv, load_feature_csv, load_marker_h5, load_label_csv, load_label_pkl
from action.data.data_transforms import ZScore, AvgAlongDimension, SumAlongDimension, MinMaxDimension
__all__ = [
    'compute_sequences', 'compute_sequence_pad', 'SingleDataset', 'preproces_dataset',
]


def preproces_dataset(hparams, model_params):
  """Helper function to build a data generator from hparam dict."""
  # Edited from https://github.com/themattinthehatt/daart/blob/master/daart/utils.py
  signals = []
  transforms = []
  paths = []

  for expt_id in hparams['expt_ids']:
    signals_curr = []
    transforms_curr = []
    paths_curr = []

    # DLC markers or features (i.e. from simba)
    input_type = hparams.get('input_type', 'markers')
    markers_file = os.path.join(hparams['data_dir'], input_type, expt_id + '_labeled.h5')
    if not os.path.exists(markers_file):
      markers_file = os.path.join(hparams['data_dir'], input_type, expt_id + '_labeled.csv')
    if not os.path.exists(markers_file):
      markers_file = os.path.join(hparams['data_dir'], input_type, expt_id + '_labeled.npy')
    if not os.path.exists(markers_file):
      msg = 'could not find marker file for %s at %s' % (expt_id, markers_file)
      logging.info(msg)
      raise FileNotFoundError(msg)
    signals_curr.append('markers')

    if hparams.get("normalize_markers", "standard") == "standard":
      # Apply Z score along dimension normalize_markers_dim
      normalize_markers_dim = hparams.get("normalize_markers_dim", 0)
      transforms_curr.append(ZScore(dim=normalize_markers_dim))
    elif hparams.normalize_markers == "sum_norm_adim":
      # Sum along dimension then apply z-score along dimension
      normalize_markers_sum_dim = hparams.get("normalize_markers_sum_dim", -1)
      normalize_markers_dim = hparams.get("normalize_markers_dim", 0)
      transforms_markers = []
      transforms_markers.append(SumAlongDimension(dim=normalize_markers_sum_dim))
      transforms_markers.append(ZScore(dim=normalize_markers_dim))
      transforms_curr.append(transforms_markers)
    elif hparams.normalize_markers == "avg_norm_adim":
      # avg along dimension then apply z-score along dimension
      normalize_markers_avg_dim = hparams.get("normalize_markers_avg_dim", -1)
      normalize_markers_dim = hparams.get("normalize_markers_dim", 0)
      transforms_markers = []
      transforms_markers.append(AvgAlongDimension(dim=normalize_markers_avg_dim))
      transforms_markers.append(ZScore(dim=normalize_markers_dim))
      transforms_curr.append(transforms_markers)
    elif hparams.normalize_markers == "avg_sum_norm_adim":
      # avg along dimension then sum along dim and then apply z-score along dimension
      normalize_markers_avg_dim = hparams.get("normalize_markers_avg_dim", -1)
      normalize_markers_sum_dim = hparams.get("normalize_markers_sum_dim", -1)
      normalize_markers_dim = hparams.get("normalize_markers_dim", 0)
      transforms_markers = []
      transforms_markers.append(AvgAlongDimension(dim=normalize_markers_avg_dim))
      transforms_markers.append(SumAlongDimension(dim=normalize_markers_sum_dim))
      transforms_markers.append(ZScore(dim=normalize_markers_dim))
      transforms_curr.append(transforms_markers)
    elif hparams.normalize_markers == "avg_sum_adim":
      # avg along dimension then sum along dim and then apply z-score along dimension
      normalize_markers_avg_dim = hparams.get("normalize_markers_avg_dim", -1)
      normalize_markers_sum_dim = hparams.get("normalize_markers_sum_dim", -1)
      transforms_markers = []
      transforms_markers.append(AvgAlongDimension(dim=normalize_markers_avg_dim))
      transforms_markers.append(SumAlongDimension(dim=normalize_markers_sum_dim))
      transforms_curr.append(transforms_markers)
    else:
      transforms_curr.append(None)

    paths_curr.append(markers_file)

    # hand labels
    if model_params.get('lambda_strong', 0) > 0:
      if expt_id not in hparams.get('expt_ids_to_keep', hparams['expt_ids']):
        hand_labels_file = None
      else:
        hand_labels_file = os.path.join(
          hparams['data_dir'], 'labels-hand', expt_id + '_labels.csv')
        if not os.path.exists(hand_labels_file):
          logging.warning('did not find hand labels file for %s' % expt_id)
          hand_labels_file = None
      signals_curr.append('labels_strong')
      transforms_curr.append(None)
      paths_curr.append(hand_labels_file)

    # heuristic labels
    if model_params.get('lambda_weak', 0) > 0:
      heur_labels_file = os.path.join(
        hparams['data_dir'], 'labels-heuristic', expt_id + '_labels.csv')
      signals_curr.append('labels_weak')
      transforms_curr.append(None)
      paths_curr.append(heur_labels_file)

    # tasks
    if model_params.get('lambda_task', 0) > 0:
      tasks_labels_file = os.path.join(hparams['data_dir'], 'tasks', expt_id + '.csv')
      if not os.path.exists(tasks_labels_file):
        tasks_labels_file = os.path.join(hparams['data_dir'], 'tasks', expt_id + '.npy')
      if not os.path.exists(tasks_labels_file):
        logging.warning('did not find tasks labels file for %s' % expt_id)
      signals_curr.append('tasks')

      if hparams.get("normalize_tasks", "standard") == "standard":
        transforms_curr.append(ZScore())
      elif hparams.normalize_tasks == "minmax":
        transforms_curr.append(MinMaxDimension())
      else:
        transforms_curr.append(None)

      paths_curr.append(tasks_labels_file)

    # define data generator signals
    signals.append(signals_curr)
    transforms.append(transforms_curr)
    paths.append(paths_curr)

  # compute padding needed to account for convolutions
  sequence_pad = model_params['sequence_pad']

  ids_list = hparams['expt_ids']
  sequence_length = hparams['sequence_length']
  signals_list = signals
  transforms_list = transforms
  paths_list = paths

  as_numpy = hparams.get("as_numpy", False)
  datasets = []
  for id, signals, transforms, paths in zip(
    ids_list, signals_list, transforms_list, paths_list):
    datasets.append(SingleDataset(
      id=id, signals=signals, transforms=transforms, paths=paths,
      sequence_length=sequence_length,
      as_numpy=as_numpy,
      sequence_pad=sequence_pad, input_type=input_type))

  return datasets


@typechecked
def compute_sequences(
        data: Union[np.ndarray, list],
        sequence_length: int,
        sequence_pad: int = 0
):
    """Compute sequences of temporally contiguous data points.

    Partial sequences are not constructed; for example, if the number of time points is 24, and the
    batch size is 10, only the first 20 points will be returned (in two batches).

    Parameters
    ----------
    data : array-like or list
        data to batch, of shape (T, N) or (T,)
    sequence_length : int
        number of continguous values along dimension 0 to include per batch
    sequence_pad : int, optional
        if >0, add `sequence_pad` time points to the beginning/end of each sequence (to account for
        padding with convolution layers)

    Returns
    -------
    list
        batched data

    """

    if isinstance(data, list):
        # assume data has already been batched
        return data

    if len(data.shape) == 2:
        batch_dims = (sequence_length + 2 * sequence_pad, data.shape[1])
    else:
        batch_dims = (sequence_length + 2 * sequence_pad,)

    # TODO: replace with nan
    n_batches = int(np.floor(data.shape[0] / sequence_length))
    batch_indices = [np.zeros(sequence_length + 2 * sequence_pad) for _ in range(n_batches)]
    batched_data = [np.zeros(batch_dims) for _ in range(n_batches)]
    for b in range(n_batches):
        idx_beg = b * sequence_length
        idx_end = (b + 1) * sequence_length
        if sequence_pad > 0:
            if idx_beg == 0:
                # initial vals are zeros; rest are real data
                batched_data[b][sequence_pad:] = data[idx_beg:idx_end + sequence_pad]
                batch_indices[b][sequence_pad:] = np.arange(idx_beg, idx_end+sequence_pad)
            elif (idx_end + sequence_pad) > data.shape[0]:
                batched_data[b][:-sequence_pad] = data[idx_beg - sequence_pad:idx_end]
                batch_indices[b][:-sequence_pad] = np.arange(idx_beg - sequence_pad, idx_end)
            else:
                batched_data[b] = data[idx_beg - sequence_pad:idx_end + sequence_pad]
                batch_indices[b] = np.arange(idx_beg - sequence_pad, idx_end + sequence_pad)

        else:
            batched_data[b] = data[idx_beg:idx_end]
            batch_indices[b] = np.arange(idx_beg, idx_end)

    return batched_data, batch_indices


@typechecked
def compute_sequence_pad(hparams: dict) -> int:
    """Compute padding needed to account for convolutions.

    Parameters
    ----------
    hparams : dict
        contains model architecture type and hyperparameter info (lags, n_hidden_layers, etc)

    Returns
    -------
    int
        amount of padding that needs to be added to beginning/end of each batch

    """

    if hparams['model_class'] in ('random-forest', 'xgboost'):
        pad = 0
    else:
        if hparams['backbone'].lower() == 'temporal-mlp':
            pad = hparams['n_lags']
        elif hparams['backbone'].lower() == 'tcn':
            pad = (2 ** hparams['n_hid_layers']) * hparams['n_lags']
        elif hparams['backbone'].lower() == 'dtcn':
            # dilattion of each dilation block is 2 ** layer_num
            # 2 conv layers per dilation block
            pad = sum([2 * (2 ** n) * hparams['n_lags'] for n in range(hparams['n_hid_layers'])])
        elif hparams['backbone'].lower() in ['lstm', 'gru']:
            # give some warmup timesteps
            pad = 4
        elif hparams['backbone'].lower() in ['animalst']:
            pad = 0
        elif hparams['backbone'].lower() == 'tgm':
            raise NotImplementedError
        else:
            raise ValueError('"%s" is not a valid backbone network' % hparams['backbone'])

    return pad


class SingleDataset(data.Dataset):
    """Dataset class for a single dataset."""

    @typechecked
    def __init__(
            self,
            id: str,
            signals: List[str],
            transforms: list,
            paths: List[Union[str, None]],
            as_numpy: bool = False,
            sequence_length: int = 500,
            sequence_pad: int = 0,
            input_type: str = 'markers',
            ignore_index: int = 0,
    ) -> None:
        """

        Parameters
        ----------
        id : str
            dataset id
        signals : list of strs
            e.g. 'markers' | 'labels_strong' | 'tasks' | ....
        transforms : list of transform objects
            each element corresponds to an entry in signals; for multiple transforms, chain
            together using :class:`daart.transforms.Compose` class. See
            :mod:`daart.transforms` for available transform options.
        paths : list of strs
            each element corresponds to an entry in `signals`; filename (using absolute path) of
            data
        sequence_length : int, optional
            number of contiguous data points in a sequence
        sequence_pad : int, optional
            if >0, add `sequence_pad` time points to the beginning/end of each sequence (to account
            for padding with convolution layers)
        input_type : str, optional
            'markers' | 'features'

        """

        # specify data
        self.id = id

        #
        self.ignore_index = ignore_index
        # get data paths
        self.signals = signals
        self.transforms = {}#OrderedDict()
        self.paths = {} #OrderedDict()
        self.dtypes = {} # OrderedDict()
        self.data = {} #OrderedDict()
        for signal, transform, path in zip(signals, transforms, paths):
            self.transforms[signal] = transform
            self.paths[signal] = path
            self.dtypes[signal] = None  # update when loading data

        self.sequence_pad = sequence_pad
        self.sequence_length = sequence_length

        self.as_numpy = as_numpy
        self.load_data(sequence_length, input_type)
        self.n_sequences = len(self.data[signals[0]])

        # meta data about train/test/xv splits; set by DataGenerator
        # self.batch_idxs = None
        # self.n_batches = None


    @typechecked
    def __str__(self) -> str:
        """Pretty printing of dataset info"""
        format_str = str('%s\n' % self.id)
        format_str += str('    signals: {}\n'.format(self.signals))
        format_str += str('    transforms: {}\n'.format(self.transforms))
        format_str += str('    paths: {}\n'.format(self.paths))
        return format_str

    @typechecked
    def __len__(self) -> int:
        return self.n_sequences

    @typechecked
    def __getitem__(self, idx: Union[int, int, None]) -> dict:
        """Return batch of data.

        Parameters
        ----------
        idx : int or NoneType
            trial index to load; if `NoneType`, return all data.

        Returns
        -------
        dict
            data sample

        """
        sample = {} #OrderedDict()
        for signal in self.signals:

            # collect signal
            if idx is None:
                sample[signal] = [d for d in self.data[signal]]
            else:
                sample[signal] = [self.data[signal][idx]]

            # from numpy to tensor
            if not self.as_numpy:
                if self.dtypes[signal] == 'float32':
                    sample[signal] = torch.from_numpy(sample[signal][0]).to(torch.float32)
                else:
                    sample[signal] = torch.from_numpy(sample[signal][0]).to(torch.long)

        # add batch index
        sample['batch_idx'] = idx
        # TODO: remove add sequence index (added for debugging video synchronization)
        if not self.as_numpy:
          sample['sequence_idx'] = torch.tensor(self.batch_idxs[idx]).to(torch.long)
        else:
          sample['sequence_idx'] = [self.batch_idxs[idx]]
        return sample

    @typechecked
    def load_data(self, sequence_length: int, input_type: str) -> None:
        """Load all data into memory.

        Parameters
        ----------
        sequence_length : int
            number of contiguous data points in a sequence
        input_type : str
            'markers' | 'features'

        """

        allowed_signals = ['markers', 'labels_strong', 'labels_weak', 'tasks']

        for signal in self.signals:

            if signal == 'markers':

                file_ext = self.paths[signal].split('.')[-1]

                if file_ext == 'csv':
                    if input_type == 'markers':
                        xs, ys, ls, marker_names = load_marker_csv(self.paths[signal])
                        data_curr = np.hstack([xs, ys])
                    else:
                        vals, feature_names = load_feature_csv(self.paths[signal])
                        data_curr = vals

                elif file_ext == 'h5':
                    if input_type != 'markers':
                        raise NotImplementedError
                    xs, ys, ls, marker_names = load_marker_h5(self.paths[signal])
                    data_curr = np.hstack([xs, ys])

                elif file_ext == 'npy':
                    # assume single array
                    data_curr = np.load(self.paths[signal])

                else:
                    raise ValueError('"%s" is an invalid file extension' % file_ext)

                self.dtypes[signal] = 'float32'

            elif signal == 'tasks':

                file_ext = self.paths[signal].split('.')[-1]
                if file_ext == 'csv':
                    vals, feature_names = load_feature_csv(self.paths[signal])
                    data_curr = vals

                elif file_ext == 'npy':
                    # assume single array
                    data_curr = np.load(self.paths[signal])
                else:
                    raise ValueError('"%s" is an invalid file extension' % file_ext)

                self.dtypes[signal] = 'float32'

            elif signal == 'labels_strong':

                if (self.paths[signal] is None) or not os.path.exists(self.paths[signal]):
                    # if no path given, assume same size as markers and set all to background
                    if 'markers' in self.data.keys():
                        data_curr = np.zeros(
                            (len(self.data['markers']) * sequence_length,), dtype=int)
                    else:
                        raise FileNotFoundError(
                            'Could not load "labels_strong" from None file without markers')
                else:
                    labels, label_names = load_label_csv(self.paths[signal])
                    data_curr = np.argmax(labels, axis=1)

                self.dtypes[signal] = 'int32'

            elif signal == 'labels_weak':

                file_ext = self.paths[signal].split('.')[-1]

                if file_ext == 'csv':
                    labels, label_names = load_label_csv(self.paths[signal])
                    data_curr = np.argmax(labels, axis=1)

                elif file_ext == 'pkl':
                    labels, label_names = load_label_pkl(self.paths[signal])
                    data_curr = labels

                self.dtypes[signal] = 'int32'

            else:
                raise ValueError(
                    '"{}" is an invalid signal type; must choose from {}'.format(
                        signal, allowed_signals))

            # apply transforms to ALL data
            # TODO: fix normalization: should be applied to train/val/test separately.
            # leaving as is to reproduce paper results

            # transform into tensor
            data_curr = data_curr.astype(np.float32)

            if self.transforms[signal]:
              if type(self.transforms[signal]) is list:
                for transform in self.transforms[signal]:
                  data_curr = transform(data_curr)
              else:
                data_curr = self.transforms[signal](data_curr)

            # min dim is 1
            if data_curr.ndim == 1:
                data_curr = data_curr[:, np.newaxis]
            # compute batches of temporally contiguous data points
            data_curr, idx_curr = compute_sequences(data_curr, sequence_length, self.sequence_pad)

            # add index data and data filename
            self.batch_idxs = idx_curr
            self.n_batches = len(idx_curr)
            if self.ignore_index is not None:
              pass

            self.data[signal] = data_curr


if __name__ == "__main__":
  from ml_collections import config_dict
  cfg = config_dict.ConfigDict()
  data_dir =   os.path.join(os.environ.get("LOCAL_PROJECTS_DIR"), "segment/action/data/ibl")
  cfg.id= "cortexlab_KS020_2020-02-06-001"
  cfg.signals=  ['markers', 'labels_weak', 'labels_strong']
  cfg.input_type = 'markers'
  cfg.transforms =  [ZScore(), None, None]
  markers_dir =   os.path.join(data_dir, cfg.input_type, cfg.id + '_labeled.npy')
  labels_weak_dir =os.path.join(data_dir, 'labels-heuristic', cfg.id + '_labels.csv')
  labels_strong_dir =os.path.join(data_dir, 'labels-hand', cfg.id + '_labels.csv')
  cfg.paths = [markers_dir, labels_weak_dir, labels_strong_dir]
  cfg.as_numpy = False
  cfg.sequence_length = 30
  cfg.sequence_pad = 1
  dataset = SingleDataset(**cfg)
  print(dataset)

  batch =  dataset.__getitem__(0)
  print(batch.keys())
  print(batch['markers'].dtype)