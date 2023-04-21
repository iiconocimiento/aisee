import os
from pathlib import Path

import numpy as np
import pandas as pd


def check_single_label_data(data):
    """
    Check if the data has the structure needed for training on single_label task.

    Parameters
    ----------
    data : str or pandas.DataFrame
        Directory path or dataframe.
        For directory, it must contain subfolders with training ('train')
        and validation ('val') samples.
        For data frame, it must contain the path, label, and fold columns
        where the fold column indicates the 'train' and 'val' samples.
    """
    if isinstance(data, pd.DataFrame):
        if "fold" not in data.columns:
            raise KeyError(
                "Missing column 'fold' to indicate separation of train and val in the dataframe.",
            )

        for fold in ["train", "val"]:
            if fold not in data["fold"].unique():
                raise ValueError(
                    f"Missing  value '{fold}' in 'fold' column to indicate separation of train and val.",
                )

        if "label" in data.columns:
            train_labels = data[data.fold == "train"].label.unique()
            val_labels = data[data.fold == "val"].label.unique()
            train_labels.sort()
            val_labels.sort()
            if not np.array_equal(
                train_labels,
                val_labels,
            ):
                raise ValueError(
                    "Train and val classes mismatch, check for names or missing classes in column 'label'.",
                )
        else:
            raise KeyError("Missing column 'label' in dataframe.")

    elif isinstance(data, (str, Path)) and Path(data).is_dir():
        for fold in ["train", "val"]:
            if not Path(data, fold).is_dir():
                raise FileNotFoundError(
                    f"Missing '{fold}' folder in {data}, to indicate data for {fold}.",
                )
        classes = [
            next(os.walk(Path(data, set_)))[1] for set_ in ["train", "val"]
        ]

        if not set(classes[0]) == set(classes[1]):
            raise ValueError(
                "Train and val classes mismatch, check for names or missing classes in folders.",
            )
    else:
        raise ValueError("Data must be a string directory path or dataframe.")


def get_data_split(data, fold):
    """
    Split data for `train` or `val.

    Parameters
    ----------
    data : str or pandas.DataFrame
        Directory path or dataframe.
        For directory, it must contain subfolders with training ('train')
        and validation ('val') samples.
        For data frame, it must contain the path, label, and fold columns
        where the fold column indicates the 'train' and 'val' samples.

    fold : str, "train" or "val"
        Directory or data split to be selected.

    Returns
    -------
    d : pandas.DataFrame or str
        Dataframe split ('train' or 'val') if data=pandas.DataFrame
        path to the directory split ('train' or 'val') if data=str
    """
    if isinstance(data, pd.DataFrame):
        return data[data["fold"] == fold]
    if isinstance(data, (str, Path)):
        return Path(data, fold)
    raise ValueError("Data must be a string directory path or dataframe.")


def get_n_classes(data):
    """
    Get the number of classes.

    Parameters
    ----------
    data : str or pandas.DataFrame
        Directory path or dataframe.
        For directory, it must contain subfolders with training ('train')
        and validation ('val') samples.
        For data frame, it must contain the path, label, and fold columns
        where the fold column indicates the 'train' and 'val' samples.

    Returns
    -------
    n : int
        Number of classes.
    """
    check_single_label_data(data)

    if isinstance(data, pd.DataFrame):
        return data[data.fold == "train"].label.nunique()
    if isinstance(data, (str, Path)):
        return len(next(os.walk(Path(data, "train")))[1])
    return None


def get_n_classes_multilabel(df):
    """
    Get the number of classes in multilabel problems.

    Parameters
    ----------
    df : pandas.DataFrame
        Multilabel image dataframe. It must contain the path and fold columns where
        the fold column indicates the 'train' and 'val' samples. Also, it must
        include a column for each possible label.

    Returns
    -------
    n : int
        Number of classes.
    """
    check_multilabel_df(df)

    return df.drop(["path", "fold"], axis=1).shape[1]


def check_multilabel_df(df):
    """
    Check if the data has the structure needed for training on multi_label task.

    Parameters
    ----------
    data : pandas.DataFrame
        Multilabel image dataframe. It must contain the path and fold columns where
        the fold column indicates the 'train' and 'val' samples. Also, it must
        include a column for each possible label.
    """
    if "path" not in df.columns:
        raise KeyError("Missing column 'path' in the DataFrame.")

    df_only_labels = df.drop(["path", "fold"], axis=1, errors="ignore")
    non_numeric_columns = df_only_labels.select_dtypes(exclude=["number"]).columns

    if len(non_numeric_columns) > 0:
        raise TypeError(
            f"{non_numeric_columns} are non-numeric columns, check data type.",
        )

    if len(set(df_only_labels.columns)) < len(df_only_labels.columns):
        raise KeyError("At least one column name is repeated.")

    for col in df_only_labels.columns:
        if any(df_only_labels[col].isna()):
            raise ValueError(
                f"Column '{col}' contains NaN, values must be between 0 and 1.",
            )

        if df_only_labels[col].max() > 1:
            raise ValueError(
                f"Column '{col}' has values greater than 1, values must be between 0 and 1.",
            )

        if df_only_labels[col].min() < 0:
            raise ValueError(
                f"Column '{col}' has values less than 0, values must be between 0 and 1.",
            )
