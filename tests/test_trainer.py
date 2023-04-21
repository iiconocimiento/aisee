from pathlib import Path

import pandas as pd
import pytest
import torch
from huggingface_hub import snapshot_download

from aisee import Trainer, VisionClassifier

TEST_PATH = Path(__file__).resolve().parent
if not Path(TEST_PATH, "resources").exists():
    snapshot_download(
        repo_id="IIC/aisee_resources", local_dir=f"{TEST_PATH}/resources/",
    )

SINGLE_LABEL_DATAFRAME = pd.DataFrame(
    [
        (f"{TEST_PATH}/resources/images/train/cat/cat1.jpg", "cat", "train"),
        (f"{TEST_PATH}/resources/images/train/cat/cat2.jpg", "cat", "train"),
        (f"{TEST_PATH}/resources/images/val/cat/cat3.jpg", "cat", "val"),
        (f"{TEST_PATH}/resources/images/train/dog/dog1.jpg", "dog", "train"),
        (f"{TEST_PATH}/resources/images/train/dog/dog2.jpg", "dog", "train"),
        (f"{TEST_PATH}/resources/images/val/dog/dog3.jpg", "dog", "val"),
    ],
    columns=["path", "label", "fold"],
)

MULTI_LABEL_DATAFRAME = pd.DataFrame(
    [
        (f"{TEST_PATH}/resources/images/train/cat/cat1.jpg", 1, 0, "train"),
        (f"{TEST_PATH}/resources/images/train/cat/cat2.jpg", 1, 0, "train"),
        (f"{TEST_PATH}/resources/images/val/cat/cat3.jpg", 1, 0, "val"),
        (f"{TEST_PATH}/resources/images/train/dog/dog1.jpg", 0, 1, "train"),
        (f"{TEST_PATH}/resources/images/train/dog/dog2.jpg", 0, 1, "train"),
        (f"{TEST_PATH}/resources/images/val/dog/dog3.jpg", 0, 1, "val"),
    ],
    columns=["path", "cat", "dog", "fold"],
)

MODEL_TEST = "mobilenetv2_050"


@pytest.mark.parametrize("checkpointing_metric", ["loss", "acc", "f1"])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("batch_size", [8, 20])
@pytest.mark.parametrize("criterion", [None, torch.nn.CrossEntropyLoss()])
@pytest.mark.parametrize("optimizer", [None, torch.optim.AdamW])
def test_train_path_model(
    checkpointing_metric,
    shuffle,
    batch_size,
    criterion,
    optimizer,
):
    """Check that Trainer is instantiated and train a model with path."""
    classf = VisionClassifier(model_name=MODEL_TEST, num_classes=2)
    trainer = Trainer(
        base_model=classf,
        data=f"{TEST_PATH}/resources/images",
        output_dir="test_trainer.pt",
        batch_size=batch_size,
        checkpointing_metric=checkpointing_metric,
        criterion=criterion,
        optimizer=optimizer,
        shuffle=shuffle,
    )
    trainer.train()

    assert isinstance(trainer.hist, list)


def test_train_df_model():
    """Check that Trainer is instantiated and train a model with df."""
    classf = VisionClassifier(model_name=MODEL_TEST, num_classes=2)
    trainer = Trainer(
        output_dir="test_trainer.pt",
        base_model=classf,
        data=SINGLE_LABEL_DATAFRAME,
    )
    trainer.train()
    assert isinstance(trainer.hist, list)


def test_train_multi_model():
    """Check that Trainer is instantiated and train a model with df."""
    classf = VisionClassifier(model_name=MODEL_TEST, num_classes=2, task="multi_label")
    trainer = Trainer(
        output_dir="test_trainer.pt",
        base_model=classf,
        data=MULTI_LABEL_DATAFRAME,
    )
    trainer.train()
    assert isinstance(trainer.hist, list)
