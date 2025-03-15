from typing import Literal

import pandas as pd

"""
This module contains functions to read, split and standardize the dataset.
Author: Valerio Morelli (@MrPio)
"""
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
                 normalization: Literal['normalize', 'standardize'] = None, merge=False) -> tuple[
                                                                                                pd.DataFrame, pd.DataFrame] | pd.DataFrame:
    """
    Read the original dataset and calculate the trq_target ground truth.
    :param stage: The kind of dataset to read. Can be 'original', 'preprocessed', 'regression' or 'classification'.
    :param normalization: If 'normalize' or 'standardize' is provided, the features will be normalized or standardized.
    :param merge: If True, the features and the ground truths will be merged into one dataframe.
    :return: One dataframe for features and one for regression and fault detection ground truths
    """
    stage_id = ['original', 'preprocessed', 'regression', 'classification'].index(stage)
    df_x = __load_cached_df(f'../dataset/{stage_id}-{stage}/X.csv').drop(columns=['id'], errors='ignore')
    df_y = __load_cached_df(f'../dataset/{stage_id}-{stage}/y.csv').drop(columns=['id'], errors='ignore')
    df_y['trq_target'] = df_x['trq_measured'] / (df_y['trq_margin'] / 100 + 1)
    if normalization == 'normalize':
        df_x = (df_x - df_x.min()) / (df_x.max() - df_x.min() + 1e-10)
        df_y = (df_y - df_y.min()) / (df_y.max() - df_y.min() + 1e-10)
    if normalization == 'standardize':
        df_x = (df_x - df_x.mean()) / df_x.std()
        df_y = (df_y - df_y.mean()) / df_y.std()
    if merge:
        return pd.concat([df_x, df_y], axis='columns')
    else:
        return df_x, df_y
