import inspect
import logging
import operator
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, TypeVar, Union

import numpy as np
import pandas as pd
import timm
import torch
from torch import nn
from torchvision import transforms

from .custom_datasets import (
    DatasetFromDataFrame,
    DatasetFromFolder,
    DatasetFromSingleImage,
)

T = TypeVar("T")
TM = TypeVar("TM")


class VisionClassifier:
    """
    Instantiating and predicting with a computer vision model from timm library.

    The VisionClassifier class allows loading and utilizing neural networks from
    timm library. The class provides methods for loading models with pre-trained
    or non pre-trained weights, or instantiating models with custom weights.
    It also allows training models with pre-trained weights (freezing some layers or training
    the entire network) and from scratch. Predictions can be made through a path to an image,
    a directory with images, or a dataframe. Additionally, it allows working with multiclass
    and multilabel problems.

    Parameters
    ----------
    model_name : str
        Name of the model that will be obtained from the timm library.
    num_classes : int
        Number of classes in the problem. The number of classes will
        be the number of outputs of the neural net.
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

    weights_path : str, default=None
        Directory where network weights are located. If value
        is different from None, pretrained weigths from the timm
        library will be ignored.
    learning_method : str, default="freezed"
        Possible values: `from_scratch`, `freezed`, and `unfreezed`:

        - `from_scratch`: The model will be trained from scratch, without
        using any pre-trained weights contained in the timm library.

        - `freezed`: The model will be trained using pre-trained weights
        from the timm library. For this training, all layers of the network
        will be frozen (weights will not be updated) except for the last layer,
        and the extra layer if it is added with the extra_layer parameter.

        - `unfreezed`: The model will be trained using pre-trained weights
        from the timm library. In this case, all layers of the network will be
        updated without exception.

        Note that if custom weights are passed in the `custom_weights` parameter,
        the network weights will be those, and the pre-trained weights from the
        timm library will be ignored.
    extra_layer : int, default=None
        If value is different from None, a linear layer is added before the last layer
        with `extra_layer` number of neurons. If None, this does nothing.
    dropout : float, default=None
        If dropout has a value different from None, dropout layer is added before the last layer. Otherwise this
        does nothing.
    task : str, default="single_label"
        Task to be resolved. Possible values:

        - "single_label"
        - "multi_label"

    device : str, default="cpu"
        Device where the neural network will be running.

        Example: "cuda:1", "cpu"
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        class_to_idx: dict[str, int] = None,
        weights_path: str = None,
        learning_method: str = "freezed",
        extra_layer: int = None,
        dropout: float = None,
        task: str = "single_label",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.weights_path = weights_path

        if learning_method not in ["from_scratch", "freezed", "unfreezed"]:
            raise ValueError(
                f"""learning_method must be ['from_scratch', 'freezed, 'unfreezed']
                not {learning_method}""",
            )
        self.learning_method = learning_method

        self.extra_layer = extra_layer
        self.dropout = dropout

        if task not in ["single_label", "multi_label"]:
            raise ValueError(
                f"""task must be ['single_label', 'multi_label'] not {task}""",
            )
        self.task = task

        self.device = device
        self.class_to_idx = class_to_idx
        self.num_classes = num_classes

        self._available_device = None

        self.model, self.cfg = self._load_model()
        self._model_to_device()
        self._load_custom_weights()

    def _model_to_device(self) -> None:
        """Load model on the device."""
        torch.cuda.empty_cache()

        if ("cuda" in self.device) and (not torch.cuda.is_available()):
            logging.warning("CUDA device selected but not available, using CPU")
            self._available_device = torch.device("cpu")
        else:
            try:
                self._available_device = torch.device(self.device)
            except RuntimeError:
                self._available_device = torch.device("cpu")
                logging.warning(f"Error with device:'{self.device}', using CPU")

        self.model.to(self._available_device)

    def _load_model(self) -> tuple[TM, dict[str, tuple]]:
        """
        Load model from timm library.

        Extra layer and dropout layer are added if necessary.

        Returns
        -------
        model : timm.models
            Model from timm library.
        cfg : dict[str, tuple]
            cfg contains "input_size", "mean" and "std" keys.
        """
        model = timm.create_model(
            self.model_name,
            pretrained=(self.learning_method != "from_scratch"),
        )
        model = self._freeze_pretrained_layers(model)
        head_name = model.default_cfg["classifier"]

        layer_clsf = operator.attrgetter(head_name)(model)
        if layer_clsf.__class__.__name__ == "Linear":
            num_features = layer_clsf.in_features
            if len(head_name.split(".")) == 2:
                head_name = head_name.split(".")
                head_clsf = getattr(model, head_name[0])
                new_atr = self._general_head_layer(num_features)
                setattr(head_clsf, head_name[1], new_atr)
            else:
                new_atr = self._general_head_layer(num_features)
                setattr(model, head_name, new_atr)

        cfg_ = model.default_cfg
        cfg = {
            "input_size": cfg_["input_size"][1:],
            "mean": cfg_["mean"],
            "std": cfg_["std"],
        }

        return model, cfg

    def _general_head_layer(
        self,
        num_features: int,
    ) -> Union[torch.nn.Sequential, torch.nn.Linear]:
        """
        Create new model head layer.

        Parameters
        ----------
        num_features: int
            Number of in features in the last layer of the original neural net.

        Returns
        -------
        layer: torch.nn.Sequential or else torch.nn.Linear
            Model head layer.

            If self.extra_layer or self.dropout has a value (different from None)
            the return type is torch.nn.Sequential. Otherwise the return type
            is torch.nn.linear.
        """
        if self.extra_layer or self.dropout:
            layers = []
            if self.dropout:
                layers.append(nn.Dropout(self.dropout))
            if self.extra_layer:
                layers.extend([nn.Linear(num_features, self.extra_layer), nn.ReLU()])
                num_features = self.extra_layer
            layers.append(nn.Linear(num_features, self.num_classes))
            return nn.Sequential(*layers)
        return nn.Linear(num_features, self.num_classes)

    def _freeze_pretrained_layers(self, model: timm.models) -> timm.models:
        """
        Freeze layers if `learning_method` is `freezed`.

        Parameters
        ----------
        model : timm.models
            Model from timm library.

        Returns
        -------
        model : timm.models
            Model from timm library.
        """
        if self.learning_method == "freezed":
            for param in model.parameters():
                param.requires_grad = False

        return model

    def _load_custom_weights(self) -> None:
        """
        Load weights from path.

        Parameters
        ----------
        model : timm.models
            Model from timm library.

        Returns
        -------
        model : timm.models
            Model from timm library with custom weights loaded.
        """
        if self.weights_path:
            self.model.load_state_dict(
                torch.load(self.weights_path, map_location=self._available_device),
            )

    def create_default_transform(self) -> dict[str, transforms.Compose]:
        """
        Create default transform based on timm config model.

        Returns
        -------
        dict_transform : dict[str, torchvision.transforms.Compose]
            A `torchvision.transforms.Compose` for train and another one for val/test.
            The dictionary keys are "train" and "val". Both keys contain a
            `torchvision.transforms.Compose` that contains the following layers:

            - Resize
            - ToTensor
            - Normalize
        """
        transform = transforms.Compose(
            [
                transforms.Resize(self.cfg["input_size"]),
                transforms.ToTensor(),
                transforms.Normalize(self.cfg["mean"], self.cfg["std"]),
            ],
        )

        return {"train": transform, "val": transform}

    def create_dataloader(
        self,
        data: Union[pd.Series, pd.DataFrame, str],
        num_workers: int = 2,
        data_transform: transforms.Compose = None,
        batch_size: int = 8,
        shuffle: bool = False,
    ) -> torch.utils.data.DataLoader:
        """
        Create dataloaders from data.

        Parameters
        ----------
        data : pandas.DataFrame or str
            A DataFrame or a string which contains the training data:

        - If it is a dataframe:
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

        - If it is a string, it must be:
            - A path to an image.

            Example

            .. code-block:: console

                /data/cat.png

            - A directory where each folder is a class, and inside that class are the
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

        num_workers : int, default=2
            Subprocesses to use for data loading.
        data_transform : torchvision.transforms.Compose, default=None
        batch_size : int, default=8
        shuffle : bool, default=False
            Shuffle the data.

        Returns
        -------
        image_dataloader : torch.utils.data.DataLoader
        """
        if not data_transform:
            data_transform = self.create_default_transform()["val"]

        if isinstance(data, pd.Series):
            data = data.to_frame()

        if isinstance(data, pd.DataFrame):
            image_dataset = DatasetFromDataFrame(
                data,
                self.task,
                data_transform,
                self.class_to_idx,
            )
        elif isinstance(data, (str, Path)) and Path(data).exists():
            if Path(data).is_file():
                image_dataset = DatasetFromSingleImage(data, transform=data_transform)
            elif Path(data).is_dir():
                image_dataset = DatasetFromFolder(
                    data,
                    data_transform,
                    self.class_to_idx,
                )
        else:
            raise ValueError(
                "Data must be a valid directory path (str), an image path (str) or a Pandas Dataframe.",
            )

        return torch.utils.data.DataLoader(
            image_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def predict(
        self,
        data: Union[pd.DataFrame, str],
        num_workers: int = 2,
        data_transform: transforms.Compose = None,
        batch_size: int = 8,
    ) -> list[Mapping[str, T]]:
        """
        Predict images in `data`.

        Parameters
        ----------
        data : pandas.DataFrame or str
            It must be a dataframe or a string:

            - If it is a dataframe:
                - If it is a multiclass problem: the dataframe must contain a `path`
                column with the full path of the image and a `label` column with
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

            - If it is a string, it must be:

                - A path to an image.

                Example

                .. code-block:: console

                    /data/cat.png

                - A directory where each folder is a class, and inside that class are the
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

        num_workers : int, default=2
            Subprocesses to use for data loading.
        data_transform : torchvision.transforms.Compose, default=None
        batch_size : int, default=8

        Returns
        -------
        result : list[Mapping[str, T]]
            Each position of the lists contains a dictionaty like this:

            .. code-block:: console

                {
                    "image_path": <path of the image>,
                    "probabilities": <all classes probabilities>,
                    "prediction": <class(es) predicted>,
                    "real_label": <real label(s)>
                }

        """
        dataloader = self.create_dataloader(
            data=data,
            num_workers=num_workers,
            data_transform=data_transform,
            batch_size=batch_size,
            shuffle=False,
        )

        image_paths, probabilities, predictions, real_labels = self.predict_loop(
            dataloader,
        )

        return [
            {
                "image_path": image_path,
                "probabilities": prob.numpy(force=True),
                "prediction": pred.numpy(force=True),
                "real_label": real_label.numpy(force=True),
            }
            for image_path, prob, pred, real_label in zip(
                image_paths,
                probabilities,
                predictions,
                real_labels,
            )
        ]

    def predict_loop(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> tuple[list, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make the loop for predict.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader

        Returns
        -------
        results : tuple[list, torch.Tensor, torch.Tensor, torch.Tensor]
            The tuple contains:
            - `all_paths`: Paths of the images predicted
            - `all_probabilities`: Probabilities of the images predicted
            - `all_predictions`: Classes predicted of the images.
            - `all_real_labels`:  Real labels of the images predicted
        """
        all_probabilities = torch.empty(0, device=self._available_device)
        all_predictions = torch.empty(0, device=self._available_device)
        all_real_labels = torch.empty(0, device=self._available_device)
        all_paths = []

        with torch.no_grad():
            self.model.eval()
            for batch in dataloader:
                inputs, labels, paths = batch

                inputs = inputs.to(self._available_device)
                labels = labels.to(self._available_device)

                outputs = self.model(inputs)

                if self.task == "single_label":
                    probabilities = nn.functional.softmax(outputs, dim=1)
                    _, predictions = torch.max(probabilities, 1)
                elif self.task == "multi_label":
                    probabilities = nn.functional.sigmoid(outputs)
                    predictions = probabilities > 0.5

                all_probabilities = torch.cat((all_probabilities, probabilities), 0)
                all_predictions = torch.cat((all_predictions, predictions), 0)
                all_real_labels = torch.cat((all_real_labels, labels), 0)
                all_paths.extend(paths)

        return all_paths, all_probabilities, all_predictions, all_real_labels

    def evaluate(
        self,
        data: Union[pd.Series, pd.DataFrame, str],
        metrics: list[Callable],
        metrics_kwargs: dict[str, dict[str, Any]],
        num_workers: int = 2,
        data_transform: transforms.Compose = None,
        batch_size: int = 8,
    ) -> dict[str, Any]:
        """
        Evaluate `data` over the metrics indicated in the `metrics` parameter.

        Parameters
        ----------
        data : pandas.DataFrame or str
            A DataFrame or a string which contains the training data:

        - If it is a dataframe:
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

        - If it is a string, it must be:
            - A path to an image.

            Example

            .. code-block:: console

                /data/cat.png

            - A directory where each folder is a class, and inside that class are the
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

        metrics : List[Callable]
            Each element of the list is a function that at least has the parameters
            `y_pred` and `y_true`, and each parameter accepts pure predictions as 1D array-like.

            For example, you can import `accuracy_score` from `sklearn.metrics`.
        metrics_kwargs : Dict[str, Dict[str, Any]]
            Each key of the dictionary represents the name of one of the functions
            indicated in the `metrics` parameter. The value is a dictionary with the
            arguments of thath function.

            For example, if your `metrics` parameter has the function `f1_score` from
            `sklearn.metrics` and you want to use the parameter `average` with `micro` value,
            your kwargs will be:

                .. code-block:: python

                metrics_kwargs = {"f1_score": {"average": "micro"}}

        num_workers : int, default=2
            Subprocesses to use for data loading.
        data_transform : torchvision.transforms.Compose, default=None
        batch_size : int, default=8

        Returns
        -------
        evaluation_results : Dict[str, Any]
            The resulting dictionary has a key for each function in the `metrics` parameter.
            The values are the results of each function.
        """
        for metric in metrics:
            func_params = inspect.signature(metric).parameters
            if not all(
                required_param in func_params
                for required_param in [
                    "y_true",
                    "y_pred",
                ]
            ):
                raise TypeError(
                    f"Parameters y_true and y_pred are required in function {metric.__name__}",
                )

        predictions = self.predict(data, num_workers, data_transform, batch_size)
        predictions = pd.DataFrame(predictions, columns=["prediction", "real_label"])

        if self.task == "single_label":
            y_pred = predictions["prediction"].astype("int")
            y_true = predictions["real_label"].astype("int")
        else:
            y_pred = np.concatenate(predictions["prediction"].to_numpy())
            y_true = np.concatenate(predictions["real_label"].to_numpy())

        return {
            metric.__name__: metric(
                y_pred=y_pred,
                y_true=y_true,
                **metrics_kwargs[metric.__name__],
            )
            if metric.__name__ in metrics_kwargs
            else metric(y_pred=y_pred, y_true=y_true)
            for metric in metrics
        }
