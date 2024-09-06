import optuna
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from anomalib.models import Patchcore
from anomalib.data import FolderDataset
from anomalib.utils.callbacks import ImageLogger
from torch.utils.data import DataLoader
import albumentations as A
import cv2
import numpy as np
from anomalib.deploy import TorchInferencer

# Define a preprocessing function
def preprocess_image(image, image_size):
    # Apply resizing and normalization
    preprocess = A.Compose([
        A.Resize(image_size, image_size),  # Resize image to the required input size
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # ImageNet normalization
    ])
    return preprocess(image=image)["image"]

# Define augmentation function for training (optional)
def augment_image(image, image_size):
    augmentations = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),  # Random horizontal flip
        A.RandomBrightnessContrast(p=0.2),  # Random brightness and contrast
        A.GaussNoise(p=0.2),  # Add some random noise to make the model more robust
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return augmentations(image=image)["image"]

# Custom Dataset class with preprocessing
class CustomFolderDataset(FolderDataset):
    def __init__(self, root, split, image_size=640, augment=False):
        super().__init__(root=root, split=split)
        self.image_size = image_size
        self.augment = augment  # Use augmentation for training, not for validation or testing

    def __getitem__(self, index):
        # Load image using super class method
        image, label = super().__getitem__(index)
        # Convert image from PIL to numpy
        image = np.array(image)

        # Apply preprocessing or augmentation
        if self.augment:
            image = augment_image(image, self.image_size)
        else:
            image = preprocess_image(image, self.image_size)

        return image, label

# Define the objective function for Optuna
def objective(trial):
    # Define hyperparameter search space
    patch_size = trial.suggest_categorical("patch_size", [16, 32, 64])
    stride = trial.suggest_categorical("stride", [8, 16, 32])
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    memory_bank_size = trial.suggest_uniform("memory_bank_size", 0.2, 0.8)
    threshold = trial.suggest_uniform("threshold", 0.5, 0.9)
    
    # Load the dataset with preprocessing
    train_dataset = CustomFolderDataset(
        root="./dataset",
        split="train",
        image_size=640,
        augment=True  # Apply augmentations during training
    )

    test_dataset = CustomFolderDataset(
        root="./dataset",
        split="test",
        image_size=640,
        augment=False  # Do not apply augmentations during validation/testing
    )

    # Create DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,  # You can tune batch size separately
        shuffle=True,
        num_workers=4,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
    )

    # Define the PatchCore model
    model = Patchcore(
        backbone="resnet50",  # Backbone model
        layers=["layer2", "layer3"],  # Layers for feature extraction
        input_size=(640, 640),  # Image size
        patch_size=patch_size,  # Patch size from Optuna
        stride=stride,  # Stride from Optuna
        num_neighbors=9,  # Fixed number of neighbors for anomaly comparison
        memory_bank_size=memory_bank_size  # Memory bank size from Optuna
    )

    # Callbacks for model checkpointing and logging
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # Monitor validation loss
        dirpath="checkpoints/",
        filename="patchcore-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,  # Save only the best model
        mode="min"  # Minimize validation loss
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    image_logger = ImageLogger()

    # Trainer definition with precision and GPU acceleration
    trainer = Trainer(
        max_epochs=100,  # Number of epochs
        accelerator="gpu",
        gpus=1,
        precision=16,  # Mixed precision for faster training
        callbacks=[checkpoint_callback, lr_monitor, image_logger]
    )

    # Set the optimizer manually with the suggested learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Train the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)

    # Test the model
    result = trainer.test(model, test_dataloaders=test_loader)

    # Get validation loss from the result
    val_loss = result[0]['val_loss']

    # Return the validation loss as the metric to minimize
    return val_loss

# Create the Optuna study for hyperparameter optimization
study = optuna.create_study(direction="minimize")  # We aim to minimize the validation loss

# Run optimization
study.optimize(objective, n_trials=20)  # Run 20 trials

# Output the best hyperparameters after the optimization
print("Best hyperparameters: ", study.best_params)

# Now we can take the best parameters and retrain the model or use them for inference
best_params = study.best_params

# Initialize the trained model for inference using the best hyperparameters
inferencer = TorchInferencer(
    model_config_path="patchcore_config.yaml",  # Load the config file used for training
    model_weight_path="checkpoints/patchcore-best.ckpt"  # Use the best checkpoint for inference
)

# Example: Running inference on a new image
image_path = "path_to_image.png"
prediction = inferencer.predict(image_path)

# Extract the anomaly map and anomaly score
anomaly_map = prediction.anomaly_map
anomaly_score = prediction.anomaly_score

# Display anomaly score
print(f"Anomaly score: {anomaly_score}")
