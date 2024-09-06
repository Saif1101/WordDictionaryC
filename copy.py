import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from anomalib.data import FolderDataset
from anomalib.models import Patchcore
from anomalib.utils.callbacks import ImageLogger
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from anomalib.config import get_configurable_parameters

# Load configuration
config = get_configurable_parameters(config_path="patchcore_config.yaml")

# Create train and test datasets
train_dataset = FolderDataset(
    root=config["dataset"]["path"],
    split="train",  # 'train' folder containing normal images
    transform=None,  # Apply any necessary transformations here
)

test_dataset = FolderDataset(
    root=config["dataset"]["path"],
    split="test",  # 'test' folder containing both normal and anomalous images
    transform=None,
)

# Create DataLoaders
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
)

# Initialize PatchCore model
model = Patchcore(
    backbone=config["model"]["backbone"],
    layers=config["model"]["layers"],
    input_size=(config["dataset"]["image_size"], config["dataset"]["image_size"])
)

# Callbacks for logging and checkpoints
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="checkpoints/",
    filename="patchcore-{epoch:02d}-{val_loss:.2f}",
    save_top_k=3,
    mode="min"
)

lr_monitor = LearningRateMonitor(logging_interval="epoch")
image_logger = ImageLogger()
logger = TensorBoardLogger("logs/", name="patchcore")

# Initialize Trainer
trainer = Trainer(
    max_epochs=config["trainer"]["max_epochs"],
    accelerator=config["trainer"]["accelerator"],
    gpus=config["trainer"]["gpus"],
    callbacks=[checkpoint_callback, lr_monitor, image_logger],
    logger=logger,
    precision=config["trainer"].get("precision", 32),
)

# Train the model
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)

# Test the model
trainer.test(model, test_dataloaders=test_loader)
