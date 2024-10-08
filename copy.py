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
        
        try:
            # Open the image
            image = Image.open(image_path).convert("RGB")
            image = np.array(image)
        except Exception as e:
            print(f"Error loading image at {image_path}: {e}")
            return None, None
        
        # Apply transformations (if any)
        if self.transform:
            try:
                image = self.transform(image=image)["image"]
            except Exception as e:
                print(f"Error applying transformations to image at {image_path}: {e}")
                return None, None
        
        return image, label

# Custom collate function to handle empty batches
def custom_collate_fn(batch):
    # Remove any samples that have None as image or label
    batch = [item for item in batch if item[0] is not None]
    
    if len(batch) == 0:
        raise RuntimeError("Empty batch encountered")
    
    return torch.utils.data.dataloader.default_collate(batch)

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

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate_fn
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn
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
        dirpath="checkpoints/",
        save_top_k=1,
        mode="min"
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        max_epochs=100,
        accelerator="gpu",
        gpus=1,
        precision=16,
        callbacks=[checkpoint_callback, lr_monitor]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    result = trainer.test(model, test_dataloaders=test_loader)

    val_loss = result[0]['val_loss']
    return val_loss

# Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

print("Best hyperparameters: ", study.best_params)
