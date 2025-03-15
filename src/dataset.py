import pandas as pd

"""
This module contains functions to read, split and standardize the dataset
"""


def read_dataset_original() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read the original dataset and calculate the trq_target ground truth.
    :return: One dataframe for features and one for regression and fault detection ground truths
    """
    df_x = pd.read_csv('../dataset/0-original/X.csv').drop(columns=['id'])
    df_y = pd.read_csv('../dataset/0-original/y.csv').drop(columns=['id'])
    df_y['trq_target'] = df_x['trq_measured'] / (df_y['trq_margin'] / 100 + 1)
    return df_x, df_y
