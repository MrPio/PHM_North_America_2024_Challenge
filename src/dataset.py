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


def __load_cached_df(path: str) -> pd.DataFrame:
    """
    Read the dataframe from disk, if it is not already in cache.
    """
    if path in __cached_df:
        return __cached_df[path].copy()
    else:
        df = pd.read_csv(path)
        __cached_df[path] = df.copy()
        return df


def read_dataset(stage: Literal['original', 'preprocessed', 'regression', 'classification'],
                 normalization: Literal['normalize', 'standardize'] = None, testset=False, merge=False) -> tuple[
                                                                                                               pd.DataFrame, pd.DataFrame] | pd.DataFrame:
    """
    Read the original dataset and calculate the trq_target ground truth.
    :param stage: The kind of dataset to read. Can be 'original', 'preprocessed', 'regression' or 'classification'.
    :param normalization: If 'normalize' or 'standardize' is provided, the features will be normalized or standardized.
    :param testset: Whether to consider the unlabelled test set or the labelled trainset.
    :param merge: If True, the features and the ground truths will be merged into one dataframe.
    :return: One dataframe for features and one for regression and fault detection ground truths
    """
    stage_id = ['original', 'preprocessed', 'regression', 'classification'].index(stage)
    if testset:
        if stage == 'original':
            df_valid = __load_cached_df(f'../dataset/0-original/X_validation.csv').drop(columns=['id'], errors='ignore')
            df_test = __load_cached_df(f'../dataset/0-original/X_test.csv').drop(columns=['id'], errors='ignore')
            df_x = pd.concat([df_valid, df_test], axis='rows', ignore_index=True)
        else:
            df_x = __load_cached_df(f'../dataset/{stage_id}-{stage}/X_test.csv').drop(columns=['id'], errors='ignore')
    else:
        df_x = __load_cached_df(f'../dataset/{stage_id}-{stage}/X.csv').drop(columns=['id'], errors='ignore')
        df_y = __load_cached_df(f'../dataset/{stage_id}-{stage}/y.csv').drop(columns=['id'], errors='ignore')

    if stage == 'classification':
        df_x['trq_margin'] = __load_cached_df(
            f'../2-torque_target_probabilistic_regression/predictions_{"test" if testset else "train"}.csv')[
            'trq_margin']
    elif not testset:
        df_y['trq_target'] = df_x['trq_measured'] / (df_y['trq_margin'] / 100 + 1)

    if normalization == 'normalize':
        df_x = (df_x - df_x.min()) / (df_x.max() - df_x.min() + 1e-10)
        if not testset:
            cols = df_y.columns.difference(["faulty"])
            df_y[cols] = (df_y[cols] - df_y[cols].min()) / (df_y[cols].max() - df_y[cols].min() + 1e-10)
    elif normalization == 'standardize':
        df_x = (df_x - df_x.mean()) / df_x.std()
        if not testset:
            cols = df_y.columns.difference(["faulty"])
            df_y[cols] = (df_y[cols] - df_y[cols].mean()) / df_y[cols].std()
    if merge and not testset:
        return pd.concat([df_x, df_y], axis='columns')
    elif testset:
        return df_x
    else:
        return df_x, df_y


def split_dataset(X: pd.DataFrame, y: pd.Series, train_ratio=0.25, standardize_y=True, max=1, seed=None,
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
    :param device: The device to use for the tensors.
    :return: train set (X, y), valid set (X, y), test set (X, y), and a dict of train means and std used to standardize the sets.
    """

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

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y), normalizations
