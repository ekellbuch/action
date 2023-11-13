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

__all__ = [
    'split_trials','load_marker_csv', 'load_feature_csv', 'load_marker_h5', 'load_label_csv', 'load_label_pkl',
]


@typechecked
def load_marker_csv(filepath: str) -> tuple:
    """Load markers from csv file assuming DLC format.

    --------------------------------------------------------------------------------
       scorer  | <scorer_name> | <scorer_name> | <scorer_name> | <scorer_name> | ...
     bodyparts |  <part_name>  |  <part_name>  |  <part_name>  |  <part_name>  | ...
       coords  |       x       |       y       |  likelihood   |       x       | ...
    --------------------------------------------------------------------------------
         0     |     34.5      |     125.4     |     0.921     |      98.4     | ...
         .     |       .       |       .       |       .       |       .       | ...
         .     |       .       |       .       |       .       |       .       | ...
         .     |       .       |       .       |       .       |       .       | ...

    Parameters
    ----------
    filepath : str
        absolute path of csv file

    Returns
    -------
    tuple
        - x coordinates (np.ndarray): shape (n_t, n_bodyparts)
        - y coordinates (np.ndarray): shape (n_t, n_bodyparts)
        - likelihoods (np.ndarray): shape (n_t,)
        - marker names (list): name for each column of `x` and `y` matrices

    """
    # data = np.genfromtxt(filepath, delimiter=',', dtype=None, encoding=None)
    # marker_names = list(data[1, 1::3])
    # markers = data[3:, 1:].astype('float')  # get rid of headers, etc.

    # define first three rows as headers (as per DLC standard)
    # drop first column ('scorer' at level 0) which just contains frame indices
    df = pd.read_csv(filepath, header=[0, 1, 2]).drop(['scorer'], axis=1, level=0)
    # collect marker names from multiindex header
    marker_names = [c[1] for c in df.columns[::3]]
    markers = df.values
    xs = markers[:, 0::3]
    ys = markers[:, 1::3]
    ls = markers[:, 2::3]
    return xs, ys, ls, marker_names


@typechecked
def load_feature_csv(filepath: str) -> tuple:
    """Load markers from csv file assuming the following format.

    --------------------------------------------------------------------------------
        name   |     <f1>      |     <f2>      |     <f3>      |     <f4>      | ...
    --------------------------------------------------------------------------------
         0     |     34.5      |     125.4     |     0.921     |      98.4     | ...
         .     |       .       |       .       |       .       |       .       | ...
         .     |       .       |       .       |       .       |       .       | ...
         .     |       .       |       .       |       .       |       .       | ...

    Parameters
    ----------
    filepath : str
        absolute path of csv file

    Returns
    -------
    tuple
        - x coordinates (np.ndarray): shape (n_t, n_bodyparts)
        - y coordinates (np.ndarray): shape (n_t, n_bodyparts)
        - likelihoods (np.ndarray): shape (n_t,)
        - marker names (list): name for each column of `x` and `y` matrices

    """
    df = pd.read_csv(filepath)
    # drop first column if it just contains frame indices
    unnamed_col = 'Unnamed: 0'
    if unnamed_col in list(df.columns):
        df = df.drop([unnamed_col], axis=1)
    vals = df.values
    feature_names = list(df.columns)
    return vals, feature_names


@typechecked
def load_marker_h5(filepath: str) -> tuple:
    """Load markers from hdf5 file assuming DLC format.

    Parameters
    ----------
    filepath : str
        absolute path of hdf5 file

    Returns
    -------
    tuple
        - x coordinates (np.ndarray): shape (n_t, n_bodyparts)
        - y coordinates (np.ndarray): shape (n_t, n_bodyparts)
        - likelihoods (np.ndarray): shape (n_t,)
        - marker names (list): name for each column of `x` and `y` matrices

    """
    df = pd.read_hdf(filepath)
    marker_names = [d[1] for d in df.columns][0::3]
    markers = df.to_numpy()
    xs = markers[:, 0::3]
    ys = markers[:, 1::3]
    ls = markers[:, 2::3]
    return xs, ys, ls, marker_names


@typechecked
def load_label_csv(filepath: str) -> tuple:
    """Load labels from csv file assuming a standard format.

    --------------------------------
       | <class 0> | <class 1> | ...
    --------------------------------
     0 |     0     |     1     | ...
     1 |     0     |     1     | ...
     . |     .     |     .     | ...
     . |     .     |     .     | ...
     . |     .     |     .     | ...

    Parameters
    ----------
    filepath : str
        absolute path of csv file

    Returns
    -------
    tuple
        - labels (np.ndarray): shape (n_t, n_labels)
        - label names (list): name for each column in `labels` matrix

    """
    labels = np.genfromtxt(
        filepath, delimiter=',', dtype=int, encoding=None, skip_header=1)[:, 1:]
    label_names = list(
        np.genfromtxt(filepath, delimiter=',', dtype=None, encoding=None, max_rows=1)[1:])
    return labels, label_names


@typechecked
def load_label_pkl(filepath: str) -> tuple:
    """Load labels from pkl file assuming a standard format.

    Parameters
    ----------
    filepath : str
        absolute path of pickle file

    Returns
    -------
    tuple
        - labels (np.ndarray): shape (n_t, n_labels)
        - label names (list): name for each column in `labels` matrix

    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    labels = data['states']
    try:
        label_dict = data['state_mapping']
    except KeyError:
        label_dict = data['state_labels']
    label_names = [label_dict[i] for i in range(len(label_dict))]
    return labels, label_names


@typechecked
def split_trials(
        n_trials: int,
        rng_seed: int = 0,
        train_tr: int = 8,
        val_tr: int = 1,
        test_tr: int = 1,
        gap_tr: int = 0
) -> dict:
    """Split trials into train/val/test blocks.

    The data is split into blocks that have gap trials between tr/val/test:

    `train tr | gap tr | val tr | gap tr | test tr | gap tr`

    Parameters
    ----------
    n_trials : int
        total number of trials to be split
    rng_seed : int, optional
        random seed for reproducibility
    train_tr : int, optional
        number of train trials per block
    val_tr : int, optional
        number of validation trials per block
    test_tr : int, optional
        number of test trials per block
    gap_tr : int, optional
        number of gap trials between tr/val/test; there will be a total of 3 * `gap_tr` gap trials
        per block; can be zero if no gap trials are desired.

    Returns
    -------
    dict
        Split trial indices are stored in a dict with keys `train`, `test`, and `val`

    """

    # same random seed for reproducibility
    np.random.seed(rng_seed)

    tr_per_block = train_tr + gap_tr + val_tr + gap_tr + test_tr + gap_tr

    n_blocks = int(np.floor(n_trials / tr_per_block))
    if n_blocks == 0:
        raise ValueError(
            'Not enough trials (n=%i) for the train/test/val/gap values %i/%i/%i/%i' %
            (n_trials, train_tr, val_tr, test_tr, gap_tr))

    leftover_trials = n_trials - tr_per_block * n_blocks
    idxs_block = np.random.permutation(n_blocks)

    batch_idxs = {'train': [], 'test': [], 'val': []}
    for block in idxs_block:

        curr_tr = block * tr_per_block
        batch_idxs['train'].append(np.arange(curr_tr, curr_tr + train_tr))
        curr_tr += (train_tr + gap_tr)
        batch_idxs['val'].append(np.arange(curr_tr, curr_tr + val_tr))
        curr_tr += (val_tr + gap_tr)
        batch_idxs['test'].append(np.arange(curr_tr, curr_tr + test_tr))

    # add leftover trials to train data
    if leftover_trials > 0:
        batch_idxs['train'].append(np.arange(tr_per_block * n_blocks, n_trials))

    for dtype in ['train', 'val', 'test']:
        batch_idxs[dtype] = np.concatenate(batch_idxs[dtype], axis=0)

    return batch_idxs
