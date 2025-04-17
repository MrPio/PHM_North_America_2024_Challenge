from time import time
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import Tensor, tensor

"""
Contains functions to read, split and standardize the dataset.
"""

__author__ = 'Valerio Morelli (@MrPio)'
__cached_df = {}
__dataset_suffix = {
    'train': '_train',
    'valid': '_valid',
    'test': '_test'
}
__discarded_suffix = {
    False: '',
    True: '_discarded',
}


def __load_df(path: str, use_cache=True) -> pd.DataFrame:
    """
    Read the dataframe from disk, if it is not already in cache.
    :param use_cache: Whether to fetch the dataset from cache when possible.
    """
    if use_cache and path in __cached_df:
        return __cached_df[path].copy()
    else:
        df = pd.read_csv(path)
        __cached_df[path] = df.copy()
        return df


def read_dataset(stage: Literal['original', 'preprocessed', 'regression', 'classification'],
                 normalization: Literal['normalize', 'standardize'] = None,
                 dataset: Literal['train', 'valid', 'test'] = 'train', merge=False, discarded=False, cheat=False,
                 use_cache=True) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
    """
    Read the original dataset and calculate the trq_target ground truth.
    :param stage: The kind of dataset to read. Can be 'original', 'preprocessed', 'regression' or 'classification'.
    :param normalization: If 'normalize' or 'standardize' is provided, the features will be normalized or standardized.
    :param dataset: What set to consider.
    :param merge: If True, the features and the ground truths will be merged into one dataframe.
    :param discarded: If True, loads the discarded csv.
    :param cheat: Whether, in classification, to fetch the ground truth instead of the regression predictions. Can only cheat for training set.
    :param use_cache: Whether to fetch the dataset from cache when possible.
    :return: One dataframe for features and one for regression and fault detection ground truths
    """
    stage_id = ['original', 'preprocessed', 'regression', 'classification'].index(stage)
    df_x = (__load_df(f'../dataset/{stage_id}-{stage}/X{__dataset_suffix[dataset]}{__discarded_suffix[discarded]}.csv',
                      use_cache=use_cache).drop(columns=['id'], errors='ignore'))
    if dataset == 'train':
        df_y = (__load_df(f'../dataset/{stage_id}-{stage}/y{__discarded_suffix[discarded]}.csv', use_cache=use_cache)
                .drop(columns=['id'], errors='ignore'))
        if not stage == 'classification':
            df_y['trq_target'] = df_x['trq_measured'] / (df_y['trq_margin'] / 100 + 1)

    if stage == 'classification':
        base_dir = '../dataset/2-regression/y' if cheat and dataset == 'train' else f'../2-torque_target_probabilistic_regression/predictions{__dataset_suffix[dataset]}'
        df_x['trq_margin'] = __load_df(f'{base_dir}{__discarded_suffix[discarded]}.csv',
                                       use_cache=use_cache)['trq_margin']

    if normalization == 'normalize':
        df_x = (df_x - df_x.min()) / (df_x.max() - df_x.min() + 1e-10)
        if dataset == 'train':
            cols = df_y.columns.difference(["faulty"])
            df_y[cols] = (df_y[cols] - df_y[cols].min()) / (df_y[cols].max() - df_y[cols].min() + 1e-10)
    elif normalization == 'standardize':
        df_x = (df_x - df_x.mean()) / df_x.std()
        if dataset == 'train':
            cols = df_y.columns.difference(["faulty"])
            df_y[cols] = (df_y[cols] - df_y[cols].mean()) / df_y[cols].std()

    if merge and dataset == 'train':
        return pd.concat([df_x, df_y], axis='columns')
    elif dataset != 'train':
        return df_x
    else:
        return df_x, df_y


def split_dataset(X: pd.DataFrame, y: pd.Series, train_ratio=0.25, standardize_y=True, max=1, seed=None, group_by=None,
                  device='cpu') -> tuple[
    tuple[Tensor, Tensor], tuple[Tensor, Tensor], tuple[Tensor, Tensor], dict[str, float]]:
    """Split the dataset into train, valid, and test sets.

    Standardize the sets using only training means and stds.
    :param X: The dataframe of features.
    :param y: The series of ground truths.
    :param train_ratio: The percentage of data to use for training.
    :param standardize_y: The classification task does not require standardization of the ground truths.
    :param max: How much of the dataset to use.
    :param seed: The seed to use for random sampling.
    :param group_by: The column that identify the domains in which the train and valid sets will be grouped.
    :param device: The device to use for the tensors.
    :return: train set (X, y), valid set (X, y), test set (X, y), and a dict of train means and std used to standardize the sets.
    """

    if group_by is not None:
        group_col = X[group_by]
        groups = group_col.unique()
        X = X.drop(group_by, axis=1)
    dataset_x = tensor(X.values, dtype=torch.float32, device=device)
    dataset_y = tensor(y, dtype=torch.float32, device=device).unsqueeze(1)
    test_ratio = (max - train_ratio) / 2.0
    size = len(X)
    np.random.seed(int(time()) if seed is None else seed)
    random_indices = np.random.choice(size, size=size, replace=False)

    # Train set
    train_x = dataset_x[random_indices[:int(size * train_ratio)]]
    train_y = dataset_y[random_indices[:int(size * train_ratio)]]
    train_x_mean = train_x.mean(dim=0)
    train_x_std = train_x.std(dim=0)
    train_y_mean = train_y.mean(dim=0)
    train_y_std = train_y.std(dim=0)
    normalizations = {}
    for i, col in enumerate(X.columns):
        normalizations[f'{col}_mean'] = train_x_mean[i].item()
        normalizations[f'{col}_std'] = train_x_std[i].item()
    if standardize_y:
        normalizations['y_mean'] = train_y_mean.item()
        normalizations['y_std'] = train_y_std.item()

    train_x = (train_x - train_x_mean) / train_x_std
    if standardize_y:
        train_y = (train_y - train_y_mean) / train_y_std

    # Valid set
    valid_x = dataset_x[random_indices[int(size * train_ratio):int(size * (train_ratio + test_ratio))]]
    valid_y = dataset_y[random_indices[int(size * train_ratio):int(size * (train_ratio + test_ratio))]]
    valid_x = (valid_x - train_x_mean) / train_x_std
    if standardize_y:
        valid_y = (valid_y - train_y_mean) / train_y_std

    # Test set
    test_x = dataset_x[random_indices[int(size * (train_ratio + test_ratio)):int(size * max)]]
    test_y = dataset_y[random_indices[int(size * (train_ratio + test_ratio)):int(size * max)]]
    test_x = (test_x - train_x_mean) / train_x_std
    if standardize_y:
        test_y = (test_y - train_y_mean) / train_y_std

    if group_by is None:
        trainset = (train_x, train_y)
        validset = (valid_x, valid_y)
    else:
        group_col = group_col.iloc[random_indices].reset_index(drop=True)
        trainset = {
            group: (
                train_x[tensor((group_col == group)[:len(train_x)].values)],
                train_y[tensor((group_col == group)[:len(train_x)].values)]
            )
            for group in groups
        }
        validset = {
            group: (
                valid_x[tensor((group_col == group)[len(train_x):len(train_x) + len(valid_x)].values)],
                valid_y[tensor((group_col == group)[len(train_x):len(train_x) + len(valid_x)].values)]
            )
            for group in groups
        }

    return trainset, validset, (test_x, test_y), normalizations
