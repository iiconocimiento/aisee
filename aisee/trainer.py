import copy
import logging
import sys
import time
from typing import TypeVar, Union

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import ChainedScheduler
from tqdm import tqdm

from .utils import get_data_split, get_n_classes, get_n_classes_multilabel
from .vision_classifier import VisionClassifier

Loss = TypeVar("Loss")
Optimizer = TypeVar("Optimizer")
lr_scheduler = TypeVar("lr_scheduler")


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))

class ParamFilter(logging.Filter):
    """
    Add disable parameter for LOGGER
    """
    def filter(self, record):
        if hasattr(record, 'disable') and record.disable:
            return False
        return True

param_filter = ParamFilter()
LOGGER.addFilter(param_filter)


loss_functions = {"single_label": torch.nn.CrossEntropyLoss(),
                  "multi_label": torch.nn.BCELoss()}
class Trainer:
    """
    Train a model from VisionClassifier.

    The Trainer class allows you to take a VisionClassifier class and train the
    model with the data passed to the Trainer and save the best model weights
    based on the selected checkpoint_metric.

    Parameters
    ----------
    base_model : VisionClassifier
        An instance of VisionClassifier.

    data : pandas.DataFrame or str (only for `single_label` task)
        A DataFrame or a string which contains the training data:

        - If it is a dataframe:
            - If it is a multiclass problem: the dataframe must contain a `path`
            column with the full path of the image, a `label` column with
            the label assigned to the image and a `fold` column that indicates the
            'train' and 'val' samples.

            Example:

            +--------------------+-----------+-----------+
            |        path        |   label   |   fold    |
            +====================+===========+===========+
            | "sample/cat1.png"  |   "cat"   |  "train"  |
            +--------------------+-----------+-----------+
            | "sample/cat2.png"  |   "cat"   |  "val"    |
            +--------------------+-----------+-----------+
            | "sample/dog1.png"  |   "dog"   |  "train"  |
            +--------------------+-----------+-----------+
            | "sample/cat3.png"  |   "cat"   |  "val"    |
            +--------------------+-----------+-----------+

            - If it is a multilabel problem: the dataframe must contain a "path"
            column with the full path of the image, one column for each class in
            the problem and a `fold` column . The classes that belong to that
            image will be indicated with a "1" and those that do not with a "0".

            Example:

            +-------------------------+---------+---------------+---------+----------+
            |          path           |   car   |   motorbike   |   bus   |   fold   |
            +=========================+=========+===============+=========+==========+
            | "sample/vehicles1.png"  |    1    |       1       |    0    |  "train" |
            +-------------------------+---------+---------------+---------+----------+
            | "sample/vehicles2.png"  |    0    |       0       |    1    |  "train" |
            +-------------------------+---------+---------------+---------+----------+
            | "sample/vehicles3.png"  |    1    |       0       |    1    |  "val"   |
            +-------------------------+---------+---------------+---------+----------+
            | "sample/vehicles4.png"  |    1    |       0       |    0    |  "val"   |
            +-------------------------+---------+---------------+---------+----------+

        - If it is a string, it must be a directory which should contain subfolders
        with training ('train') and validation ('val') samples and second
        subfolders with labels.

        Example:

        .. code-block:: console

            └── animals
                ├── train
                |   ├── cat
                |   │  ├── cat1.jpg
                |   │  └── cat2.jpg
                |   └── dog
                |       ├── dog1.jpg
                |       └── dog2.jpg
                └── val
                    ├── cat
                    │  ├── cat3.jpg
                    │  └── cat4.jpg
                    └── dog
                        ├── dog3.jpg
                        └── dog4.jpg

    output_dir : str, default=None
        File where the weights of the neural network will be saved.
        If None output_dir = 'weights_model_name_time.pt'

    lr : float, default=0.001
        Learning rate used by the torch optimizer.

    batch_size : int, default=16
        Number of training samples.

    num_epochs : int, default=5
        Number of training epochs.

    checkpointing_metric : str, default="loss"
        Metric with which the best model will be saved. Possible values:

        - "loss"
        - "acc"
        - "f1"

        F1 is calculated as 'macro-averaged F1 score'

    verbose : int, default=3
        Controls the verbosity: the higher, the more messages.

    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting.

    num_workers : int, default=2
        How many subprocesses to use for data loading.
        ``0`` means that the data will be loaded in the main process.

    criterion : torch.nn, default= CrossEntropyLoss for single_label
                          default= BCELoss for multi_label
        A loss function from pytorch.
        This criterion computes loss between input logits and target.

    dic_data_transforms = dict with 'train' and 'val' image transformations, default=None
        A function/transform that  takes in an PIL image and returns a transformed version.
        If None for train: resize, horizontal flip and normalize
        val: resize and normalize

    optimizer : torch.optim, default=None
        Add an optimizer from pytorch. If None Adam will be used.

    optimer_kwargs : dict, default=None
        Optimizer parameters.

    schedulers : list[(scheduler, dict_scheduler_parameters)], default=None
        Add lr_schedulers from pytorch.
        Must be a list of tuples with two elements, the scheduler and a dictionary with
        scheduler parameters with values without include optimizer.
        For ReduceLROnPlateau the dictionary of parameters must include the metric
        [f1, loss, acc].
    """

    def __init__(
        self,
        base_model: VisionClassifier,
        data: Union[pd.DataFrame, str],
        output_dir: str = None,
        lr: float = 0.001,
        batch_size: int = 8,
        num_epochs: int = 5,
        checkpointing_metric: str = "loss",
        verbose: int = 3,
        shuffle: bool = True,
        num_workers: int = 2,
        dict_data_transforms: dict = None,
        criterion: type[Loss] = None,
        optimizer: type[Optimizer] = None,
        optimer_kwargs: dict = None,
        schedulers: list[lr_scheduler, dict] = None,
    ) -> None:
        self.base_model = base_model
        self.output_dir = (
            output_dir
            or
            "weights_"
            + base_model.model_name
            + "_"
            + time.strftime("%Y%m%d_%H%M%S")
            + ".pt"
        )
        self.data = data
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        if checkpointing_metric not in ["loss", "acc", "f1"]:
            raise ValueError(
                f"checkpointing_metric must be in ['loss', 'acc, 'f1'] not {checkpointing_metric}.",
            )
        self.checkpointing_metric = checkpointing_metric

        self.verbose = verbose
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.dict_data_transforms = dict_data_transforms or self.base_model.create_default_transform()

        self.criterion = criterion or loss_functions[self.base_model.task]

        self.optimizer = optimizer or torch.optim.Adam

        self.optimer_kwargs = optimer_kwargs or {}

        if schedulers:
            if not isinstance(schedulers, list):
                raise ValueError(
                    "shedulers must be a list [(lr_scheduler, dict_schd_params)]")

            check_schd = [not (isinstance(tpl, tuple) and isinstance(tpl[0], type)
            and isinstance(tpl[1], dict) and len(tpl[0].__module__.split(".")) > 2
            and tpl[0].__module__.split(".")[2] == "lr_scheduler") for tpl in schedulers]
            self.schedulers = schedulers
            if any(check_schd):
                raise ValueError("Check elements of schedulers list, see documentation.")

            for tpl in schedulers:
                if tpl[0].__name__ == "ReduceLROnPlateau":
                    if "metric" not in tpl[1]:
                        raise ValueError("No metric parameter for ReduceLROnPlateau , see documentation.")
                    if tpl[1]["metric"] not in ["loss", "acc", "f1"]:
                        raise ValueError(
                                "For ReduceLROnPlateau metric must be: ['loss', 'acc, 'f1'].",
                                )

        self.schedulers = schedulers

    def load_data_dict(self) -> dict[str, torch.utils.data.DataLoader]:
        """
        Create training and validation dataloaders.

        Returns
        -------
        dataloaders_dict : dict [str: torch.utils.data.DataLoader]
            dict with "train" and "val" torch class dataloader.
        """
        if self.base_model.task == "multi_label":
            n_real_classes = get_n_classes_multilabel(self.data)
        else:
            n_real_classes = get_n_classes(self.data)
        if self.base_model.num_classes != n_real_classes:
            raise ValueError("Differences between declared num_classes and data.")

        return {
            split: self.base_model.create_dataloader(
                get_data_split(self.data, split),
                self.num_workers,
                self.dict_data_transforms[split],
                self.batch_size,
                self.shuffle,
            )
            for split in ["train", "val"]
        }

    def train(self):
        """
        Train the base_model.

        Train the base_model and updates the weights based on the best on
        the checkpointing_metric.

        Create the hist attribute with the training history.

        Returns
        -------
        self
        """
        LOGGER.info("Initializing Dataloaders...",
                    extra={'disable': self.verbose < 2})
        LOGGER.info("\n", extra={'disable': self.verbose < 2})

        X = self.load_data_dict()

        self.base_model.class_to_idx = X["train"].dataset.class_to_idx

        params_name = []
        params_to_update = []
        for name, param in self.base_model.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                params_name.append(name)

        LOGGER.info("Params to learn:", extra={'disable': self.verbose < 1})
        LOGGER.info("\n".join(params_name) + "\n", 
                    extra={'disable': self.verbose < 1})

        self.optimizer_up = self.optimizer(
            params_to_update,
            lr=self.lr,
            **self.optimer_kwargs,
        )

        self._train_model()

        torch.save(self.base_model.model.state_dict(), self.output_dir)

        return self

    def _train_model(self):
        """
        Train model.

        Returns
        -------
        m : trained model.
        h: list of dict, metric for each epoch
        """
        dataloaders = self.load_data_dict()
        is_inception = False
        multilabel = self.base_model.task == "multi_label"

        since = time.time()

        best_epoch = 0
        best_model_wts = copy.deepcopy(self.base_model.model.state_dict())
        best_sm = -100.0 if self.checkpointing_metric == "loss" else 0.0
        factor = -1 if self.checkpointing_metric == "loss" else 1
        self.hist = []
        last_lr = self.lr

        scheduler = None
        scheduler_rlrop = None
        if self.schedulers:
            list_schedulers = []
            for sch in self.schedulers:
                func = sch[0]
                params = sch[1]
                if func.__name__ == "ReduceLROnPlateau":
                    params2 = params.copy()
                    schd_metric = params2.pop('metric')
                    scheduler_rlrop = func(self.optimizer_up, **params2)
                else:
                    list_schedulers.append(func(self.optimizer_up, **params))
                if len(list_schedulers) > 1:
                    scheduler = ChainedScheduler(list_schedulers)
                elif len(list_schedulers) == 1:
                    scheduler = list_schedulers[0]

        for epoch in range(1, self.num_epochs + 1):
            LOGGER.info(f"Epoch {epoch}/{self.num_epochs}",
                        extra={'disable': self.verbose < 2})
            LOGGER.info("-" * 10, extra={'disable': self.verbose < 2})

            self.hist.append({"epoch": epoch})

            for phase in ["train", "val"]:
                if phase == "train":
                    self.base_model.model.train()
                else:
                    self.base_model.model.eval()

                running_loss = 0.0
                epoch_outputs = torch.empty(0, device=self.base_model._available_device)
                epoch_labels = torch.empty(0, device=self.base_model._available_device)

                for batch in tqdm(
                    dataloaders[phase],
                    desc=f"{phase} batches",
                    disable=self.verbose < 3,
                    total=int(
                        len(dataloaders[phase].dataset) / dataloaders[phase].batch_size
                        + 0.5,
                    ),
                ):
                    inputs, labels, _ = batch
                    inputs = inputs.to(self.base_model._available_device)
                    labels = labels.to(self.base_model._available_device)

                    self.optimizer_up.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output.
                        # In train mode we calculate the loss by summing the final output and the
                        # auxiliary output, but in testing we only consider the final output.
                        if is_inception and phase == "train":
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = self.base_model.model(inputs)
                            if multilabel:
                                outputs = torch.sigmoid(outputs)
                                aux_outputs = torch.sigmoid(aux_outputs)
                            loss1 = self.criterion(outputs, labels)
                            loss2 = self.criterion(aux_outputs, labels)
                            loss = loss1 + 0.4 * loss2
                        else:
                            outputs = self.base_model.model(inputs)
                            if multilabel:
                                outputs = torch.sigmoid(outputs)
                            loss = self.criterion(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            self.optimizer_up.step()

                    running_loss += loss.item() * inputs.size(0)
                    epoch_outputs = torch.cat((epoch_outputs, outputs), 0)
                    epoch_labels = torch.cat((epoch_labels, labels), 0)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                if multilabel:
                    epoch_outputs = epoch_outputs.detach()
                    epoch_preds = epoch_outputs.round()
                    epoch_preds = epoch_preds.numpy(force=True)
                else:
                    _, epoch_preds = torch.max(epoch_outputs, 1)
                    epoch_preds = epoch_preds.numpy(force=True)
                epoch_labels = epoch_labels.numpy(force=True)
                epoch_acc = accuracy_score(epoch_labels, epoch_preds)
                epoch_f1 = f1_score(epoch_labels, epoch_preds, average="macro")
                epoch_lr = self.optimizer_up.param_groups[0]["lr"]

                LOGGER.info(f"New lr ={epoch_lr}",
                            extra={'disable': (self.verbose < 2
                                                    or last_lr == epoch_lr
                                                    or phase != "train")})
                last_lr = epoch_lr
                LOGGER.info(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}  F1: {epoch_f1:.4f}",
                            extra={'disable': self.verbose < 2})
                LOGGER.info("\n", extra={'disable': self.verbose < 2})

                indx = epoch - 1
                self.hist[indx][phase + "_loss"] = epoch_loss
                self.hist[indx][phase + "_acc"] = epoch_acc
                self.hist[indx][phase + "_f1"] = epoch_f1
                self.hist[indx]["lr"] = epoch_lr
                self.hist[indx]["time"] = time.time() - since
                epoch_sm = self.hist[indx][phase + "_" + self.checkpointing_metric] * factor

                if phase == "val" and epoch_sm > best_sm:
                    best_sm = epoch_sm
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(self.base_model.model.state_dict())
                    torch.save(self.base_model.model.state_dict(), self.output_dir)

            if scheduler:
                scheduler.step()
            if scheduler_rlrop:
                value_metric = self.hist[indx]["val_" + schd_metric]
                scheduler_rlrop.step(value_metric)

        best_sm = best_sm * factor
        time_elapsed = time.time() - since

        LOGGER.info(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s",
                    extra={'disable': self.verbose < 1})
        LOGGER.info(f"Best val {self.checkpointing_metric}: {best_sm:4f} in epoch {best_epoch}",
                    extra={'disable': self.verbose < 1})

        self.base_model.model.load_state_dict(best_model_wts)
