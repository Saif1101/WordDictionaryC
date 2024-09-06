import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from anomalib.models import Patchcore
import albumentations as A
from PIL import Image
import optuna

# Custom dataset class for loading images from folders
class CustomFolderDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.normal_dir = os.path.join(self.root_dir, split, "normal")
        self.anomalous_dir = os.path.join(self.root_dir, split, "anomalous")
        self.normal_images = [os.path.join(self.normal_dir, img) for img in os.listdir(self.normal_dir)]
        if split == "test":
            self.anomalous_images = [os.path.join(self.anomalous_dir, img) for img in os.listdir(self.anomalous_dir)]
            self.image_paths = self.normal_images + self.anomalous_images
        else:
            self.image_paths = self.normal_images
        self.labels = [0] * len(self.normal_images)
        if split == "test":
            self.labels += [1] * len(self.anomalous_images)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label

# Preprocessing and augmentation transformations
def get_preprocessing(image_size=640):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

def get_augmentation(image_size=640):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

# Optuna objective function for hyperparameter tuning
def objective(trial):
    patch_size = trial.suggest_categorical("patch_size", [16, 32, 64])
    stride = trial.suggest_categorical("stride", [8, 16, 32])
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    memory_bank_size = trial.suggest_uniform("memory_bank_size", 0.2, 0.8)
    threshold = trial.suggest_uniform("threshold", 0.5, 0.9)

    train_dataset = CustomFolderDataset(
        root_dir="./dataset",
        split="train",
        transform=get_augmentation(image_size=640)
    )

    test_dataset = CustomFolderDataset(
        root_dir="./dataset",
        split="test",
        transform=get_preprocessing(image_size=640)
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    model = Patchcore(
        backbone="resnet50",
        layers=["layer2", "layer3"],
        input_size=(640, 640),
        patch_size=patch_size,
        stride=stride,
        num_neighbors=9,
        memory_bank_size=memory_bank_size
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dir
