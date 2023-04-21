from typing import Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import datasets, transforms


class DatasetFromFolder(datasets.ImageFolder):
    """Image Dataset for return de path of the image.

    Parameters
    ----------
    data : str
        A directory where each folder is a class, and inside that class are the
        images.

        Example:

        .. code-block:: console

            ├── animals
            ├── cat
            │  ├── cat1.jpg
            │  └── cat2.jpg
            └── dog
                ├── dog1.jpg
                └── dog2.jpg

    transform : torchvision.transforms.Compose
    class_to_idx : dict[str, int], default=None
        Equivalence between the label and the index of the neural net output. This parameter
        is equivalent to `label2id` of the transformers library.
        Example:

            .. highlight:: python
            .. code-block:: python

                class_to_idx = {
                    "cat": 0,
                    "dog": 1
                }

                nerual_net_output = [0.2, 0.8]

            - Cat has probability 0.2
            - Dog has probability 0.8

    """

    def __init__(
        self,
        data: str,
        transform: transforms.Compose = None,
        class_to_idx: dict[str, int] = None,
    ) -> None:
        super().__init__(data, transform)

        if class_to_idx:
            self._idx_to_class_original = {v: k for k, v in self.class_to_idx.items()}
            self.class_to_idx = class_to_idx
        else:
            self._idx_to_class_original = None

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        """
        Return the image, label and image path.

        Parameters
        ----------
        index : int
            Index of the element to return.

        Returns
        -------
        result : tuple[torch.Tensor, int, str]
            The tuple containes: (image, label, image path)
        """
        img, label = super().__getitem__(index)

        if self._idx_to_class_original:
            label = self.class_to_idx[self._idx_to_class_original[label]]

        path = self.imgs[index][0]

        return img, label, path


class DatasetFromDataFrame(torch.utils.data.Dataset):
    """
    Image Dataset for Pandas Dataframe data.

    Parameters
    ----------
    data : pandas.DataFrame

        - If it is a multiclass problem: the dataframe must contain a `path`
        column with the full path of the image and a `label` column (optional) with
        the label assigned to the image. This `label` can be a number or a string.

        Example:

        +-------------------+------------+
        |       path        |   label    |
        +===================+============+
        |"sample/cat1.png"  |   "cat"    |
        +-------------------+------------+
        |"sample/cat2.png"  |   "cat"    |
        +-------------------+------------+
        |"sample/dog1.png"  |   "dog"    |
        +-------------------+------------+
        |"sample/cat3.png"  |   "cat"    |
        +-------------------+------------+

        or

        +-----------------+
        |        path     |
        +=================+
        |"sample/cat1.png"|
        +-----------------+
        |"sample/cat2.png"|
        +-----------------+
        |"sample/dog1.png"|
        +-----------------+
        |"sample/cat3.png"|
        +-----------------+

        - If it is a multilabel problem: the dataframe must contain a "path"
        column with the full path of the image and one column for each
        class in the problem. The classes that belong to that image will be
        indicated with a "1" and those that do not with a "0".

        Example:

        +------------------------+---------+---------------+--------+
        |        path            |   car   |   motorbike   |   bus  |
        +========================+=========+===============+========+
        |"sample/vehicles1.png"  |    1    |       1       |    0   |
        +------------------------+---------+---------------+--------+
        |"sample/vehicles2.png"  |    0    |       0       |    1   |
        +------------------------+---------+---------------+--------+
        |"sample/vehicles3.png"  |    1    |       0       |    1   |
        +------------------------+---------+---------------+--------+
        |"sample/vehicles4.png"  |    1    |       0       |    0   |
        +------------------------+---------+---------------+--------+

    task : str
        Task to be resolved. Possible values:

        - "single_label"
        - "multi_label"

    transform : torchvision.transforms.Compose
    class_to_idx : dict[str, int], default=None
        Equivalence between the label and the index of the neural net output. This parameter
        is equivalent to `label2id` of the transformers library.
        Example:

            .. highlight:: python
            .. code-block:: python

                class_to_idx = {
                    "cat": 0,
                    "dog": 1
                }

                nerual_net_output = [0.2, 0.8]

            - Cat has probability 0.2
            - Dog has probability 0.8

    """

    def __init__(
        self,
        data: Union[pd.Series, pd.DataFrame],
        task: str = "single_label",
        transform: transforms.Compose = None,
        class_to_idx: dict[str, int] = None,
    ) -> None:
        self.data = data.reset_index(drop=True)
        self.task = task
        self.transform = transform

        if "fold" in data.columns:
            self.data = self.data.drop(["fold"], axis=1)

        self.check_dataframe_columns()

        if not class_to_idx:
            if self.task == "single_label" and "label" not in self.data.columns:
                self.class_to_idx = None
            else:
                self.class_to_idx = self.generate_class_to_idx()
        else:
            self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        """Return data size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """
        Return the image, label and image path.

        Parameters
        ----------
        index : int
            Index of the element to return.

        Returns
        -------
        result : tuple[torch.Tensor, torch.Tensor, str]
            The tuple containes: (image, label, image path)
        """
        img = Image.open(self.data["path"][idx]).convert("RGB")
        if self.task == "multi_label":
            label = (
                self.data.drop(["path"], axis=1).iloc[idx].to_numpy(dtype=np.float32)
            )
        elif self.task == "single_label":
            if self.class_to_idx:
                label = self.data["label"][idx]
                label = self.class_to_idx[label]
                label = torch.tensor(label, dtype=torch.long)
            else:
                label = np.nan

        path = self.data["path"][idx]

        if self.transform:
            img = self.transform(img)

        return img, label, path

    def generate_class_to_idx(self) -> dict[str, int]:
        """
        Generate a dictionary with classes equivalences.

        Example:

        .. highlight:: python
        .. code-block:: python

            classes = ["cat", "dog]

            class_to_idx = {
                "cat": 0,
                "dog": 1
            }

        Returns
        -------
        class_to_idx : dict[str, int]
        """
        if self.task == "single_label":
            classes = sorted(self.data["label"].unique())
        elif self.task == "multi_label":
            classes = self.data.drop(["path"], axis=1).columns

        return {label: idx for idx, label in enumerate(classes)}

    def check_dataframe_columns(self) -> None:
        """Check if Pandas DataFrame is well constructed."""
        if "path" not in self.data.columns:
            raise ValueError("path column is missing")

        if self.task == "multi_label":
            duplicated_columns = self.data.columns.duplicated()
            if any(duplicated_columns):
                raise ValueError(
                    "Some columns names are duplicated: "
                    f"{', '.join(self.data.columns[duplicated_columns])}",
                )
        elif self.task != "single_label":
            raise ValueError(f"Task '{self.task}' is not supported.")


class DatasetFromSingleImage(torch.utils.data.Dataset):
    """
    Image Dataset for one label image.

    Parameters
    ----------
    data : str
        Image path

        Example:

        .. code-block:: console

            /data/cat.png

    transform : torchvision.transforms.Compose
    """

    def __init__(self, data: str, transform: transforms.Compose = None) -> None:
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        """Return data size."""
        return 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float, str]:
        """
        Return the image, label and image path.

        Parameters
        ----------
        index : int
            Index of the element to return.

        Returns
        -------
        result : tuple[torch.Tensor, float, str]
            The tuple containes: (image, label, image path)
        """
        img = Image.open(self.data).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = np.nan

        return img, label, self.data
