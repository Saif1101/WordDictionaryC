import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.data.utils import InputNormalizationMethod, Split
from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.utils.callbacks import LoadModelCallback, ExportCallback, ModelCheckpoint

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths
dataset_path = Path("/path/to/your/dataset")
results_path = Path("./results")
export_path = Path("./exported_model")

# Ensure the results and export directories exist
results_path.mkdir(exist_ok=True, parents=True)
export_path.mkdir(exist_ok=True, parents=True)

# Create the dataset
dataset = Folder(
    root=dataset_path,
    normal_dir="train/good",
    abnormal_dir="test/abnormal",
    normal_test_dir="test/good",
    task="segmentation",  # Change to "classification" if needed
    split=Split.NONE,
    image_size=(640, 640),  # Resize images to 640x640
)

# Create data loaders
train_dataloader = DataLoader(dataset.train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(dataset.test_data, batch_size=32, shuffle=False)

# Initialize the PatchCore model
model = Patchcore(
    input_size=(640, 640),
    backbone="wide_resnet50_2",
    layers_to_extract_from=["layer2", "layer3"],
    pre_trained=True,
    num_neighbors=9,
)

# Configure the engine
engine = Engine(
    task="segmentation",  # Change to "classification" if needed
    input_size=(640, 640),
    center_crop=None,
    normalization=InputNormalizationMethod.NONE,
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    metrics=["F1Score", "AUROC"],
    device=device,
    callbacks=[
        LoadModelCallback(weights_path=None),
        ModelCheckpoint(mode="max", metric="image_F1Score"),
        ExportCallback(
            export_root=export_path,
            onnx=True,  # Export to ONNX format
            openvino=False,  # Set to True if you want OpenVINO format as well
        ),
    ],
)

# Train the model
engine.train()

# Test the model
engine.test()

print("Training and testing completed. Model exported.")
