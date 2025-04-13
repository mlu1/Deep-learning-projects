import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os
import re
from tqdm import tqdm
import lightning as L
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision.transforms import v2
from torch import nn
import timm  # Pretrained models library

# Define directory paths for training and testing images
TRAIN_DIR = Path("train")
TEST_DIR = Path("test")

# Load CSV files into dataframes
train_data = pd.read_csv("Train.csv")
test_data = pd.read_csv("Test.csv")
sample_submission = pd.read_csv("Sample_Submission.csv")

# Open a sample image for testing
sample_path = "train/A2miww5mfx/A2miww5mfx_L_014.png"
sample_img = Image.open(sample_path)

# Initialize segmentation models with different checkpoints
seg_models = {
    "full": YOLO("Models/best_full.pt"),
    "early": YOLO("Models/best_early.pt"),
    "late": YOLO("Models/best_late.pt")
}

# Run a sample inference using the "full" model and print bounding box details
current_model = seg_models["full"]
detection_results = current_model(sample_path)

for detection in detection_results:
    for bbox in detection.boxes.xywh:
        x_center, y_center, width_box, height_box = bbox  # x_center, y_center are center coordinates
        print(f"Bounding Box - X: {x_center}, Y: {y_center}, Width: {width_box}, Height: {height_box}")

# Function to extract segmented images from given image paths
def process_image_segments(img_paths, display_image=False):
    """
    Extracts and merges segments from images, returning only images with detections.
    """
    # Try models one by one until detections are found
    for key in seg_models.keys():
        model_instance = seg_models[key]
        detection_results = model_instance(img_paths, verbose=False)
        if len(detection_results[0].boxes.xyxy) != 0:
            break

    # If no detections, simply return the original images
    if len(detection_results[0].boxes.xyxy) == 0:
        return [Image.open(path) for path in img_paths]

    merged_imgs = []
    for path, detection in zip(img_paths, detection_results):
        orig_img = Image.open(path)
        combined_img = Image.new("RGBA", orig_img.size, (0, 0, 0, 0))

        # Skip image if there are no bounding boxes detected
        if len(detection.boxes.xyxy) == 0:
            continue

        # Crop and paste each detected segment onto a blank canvas
        for bbox in detection.boxes.xyxy:
            x1, y1, x2, y2 = map(int, bbox.tolist())
            segment = orig_img.crop((x1, y1, x2, y2))
            combined_img.paste(segment, (x1, y1))
        merged_imgs.append(combined_img)

    # Optionally display the segmented images
    if display_image and merged_imgs:
        fig, axs = plt.subplots(1, len(merged_imgs), figsize=(15, 10))
        if len(merged_imgs) == 1:
            axs = [axs]
        for ax, img in zip(axs, merged_imgs):
            ax.imshow(img)
            ax.axis("off")
        plt.show()

    return merged_imgs

# Display segmented output for the sample image
process_image_segments([sample_path], display_image=True)

# Function to retrieve image paths based on folder, scan side, and layer range
def fetch_images_in_range(base_dir: Path, folder_name: str, scan_side: str, start_layer: int, end_layer: int) -> list[Path]:
    """
    Retrieve images from a folder that match the specified side (L/R) and layer range.
    """
    target_folder = base_dir / folder_name

    try:
        file_names = os.listdir(target_folder)
    except FileNotFoundError:
        return []

    pattern = re.compile(r'_([LR])_(\d{3})\.png$')
    selected_paths = []

    for fname in file_names:
        match = pattern.search(fname)
        if match:
            side_char = match.group(1)
            layer_num = int(match.group(2))
            if side_char == scan_side and start_layer <= layer_num <= end_layer:
                selected_paths.append(target_folder / fname)

    return selected_paths

# Example usage of fetch_images_in_range function
sample_imgs_range = fetch_images_in_range(TRAIN_DIR, "Ypktwvqjbn", "L", 33, 41)
print(sample_imgs_range)

# Function to merge segmented images across a given layer range
def combine_segments(base_dir: Path, folder_name: str, scan_side: str, start_layer: int, end_layer: int):
    images_list = fetch_images_in_range(base_dir, folder_name, scan_side, start_layer, end_layer)
    segmented_imgs = process_image_segments(images_list)

    # Calculate the dimensions for the merged image
    merged_width = sum(img.width for img in segmented_imgs)
    merged_height = max(img.height for img in segmented_imgs)

    # Create a blank canvas and paste each segmented image side by side
    combined_img = Image.new("RGBA", (merged_width, merged_height), (0, 0, 0, 0))
    offset_x = 0
    for img in segmented_imgs:
        combined_img.paste(img, (offset_x, 0), img)
        offset_x += img.width

    return combined_img

# Merge segments for an example folder and layer range
updated_merged_img = combine_segments(TRAIN_DIR, "Ox18ob0syv", "R", 21, 28)

# Create directories for saving merged images
MERGED_DIR = Path("merged_images/")
os.makedirs(MERGED_DIR, exist_ok=True)

MERGED_TRAIN_DIR = MERGED_DIR / "Train"
MERGED_TEST_DIR = MERGED_DIR / "Test"

os.makedirs(MERGED_TRAIN_DIR, exist_ok=True)
os.makedirs(MERGED_TEST_DIR, exist_ok=True)

# Function to generate and save merged images based on dataframe info
def create_merged_images(dataframe: pd.DataFrame, out_dir: Path, in_dir: Path):
    merged_paths = []
    
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Merging Images"):
        merged_img = combine_segments(
            base_dir=in_dir,
            folder_name=row["FolderName"],
            scan_side=row["Side"],
            start_layer=row["Start"],
            end_layer=row["End"]
        )
        img_out_path = out_dir / f"{row['ID']}.png"
        merged_img.save(img_out_path)
        merged_paths.append(img_out_path)

    # Update the dataframe with the path to the merged image
    dataframe['merged_path'] = merged_paths
    return dataframe

updated_train_data = create_merged_images(train_data, MERGED_TRAIN_DIR, TRAIN_DIR)
updated_test_data = create_merged_images(test_data, MERGED_TEST_DIR, TEST_DIR)

# Define transformations for training and testing images
train_transform = v2.Compose([
    v2.Resize(size=(128, 128), antialias=True),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5], std=[0.5])
])

test_transform = v2.Compose([
    v2.Resize(size=(128, 128), antialias=True),
    v2.RandomRotation(degrees=10),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5], std=[0.5])
])

# Test the transformation on a sample image from the merged training data
sample_img = Image.open(updated_train_data['merged_path'].iloc[6])
train_transform(sample_img)

# Custom Dataset for RootVolume regression
class VolumeDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None, training_mode=True):
        super().__init__()
        self.dataframe = dataframe
        self.transform = transform
        self.training_mode = training_mode

    def __getitem__(self, idx):
        img = Image.open(self.dataframe['merged_path'].iloc[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        if self.training_mode:
            target = self.dataframe['RootVolume'].iloc[idx]
            return img, torch.tensor(target, dtype=torch.float32)

        return img

    def __len__(self):
        return len(self.dataframe)

# Set random seeds for reproducibility
def initialize_seed(seed_val):
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    L.pytorch.seed_everything(seed_val, workers=True)
    
initialize_seed(42)

# Create datasets and dataloaders for training and testing
train_dataset = VolumeDataset(updated_train_data, train_transform)
test_dataset = VolumeDataset(updated_test_data, test_transform, training_mode=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# Define the model for volume regression using a pretrained ResNet34
class VolumeRegressor(L.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate

        # Pretrained feature extractor
        self.feature_net = timm.create_model('resnet34', pretrained=True, num_classes=0)
        self.feature_net.global_pool = nn.AdaptiveAvgPool2d(1)

        # Regression head
        self.regressor_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.feature_net(x)
        x = self.regressor_head(x)
        return x

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        predictions = self(imgs).squeeze()
        loss = self.criterion(predictions, targets)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        predictions = self(imgs).squeeze()
        loss = self.criterion(predictions, targets)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)

model = VolumeRegressor()

# Train the model using a Lightning Trainer
model_trainer = L.Trainer(max_epochs=33)
model_trainer.fit(model, train_loader)

# Function to obtain predictions and ground truth from a dataloader
def evaluate_model_predictions(model, loader, device="cuda" if torch.cuda.is_available() else "cpu"):
    predictions, ground_truth = [], []
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in loader:
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            predictions.extend(outputs.cpu().numpy().flatten())
            ground_truth.extend(labels.cpu().numpy().flatten())

    return np.array(predictions), np.array(ground_truth)

train_predictions, train_targets = evaluate_model_predictions(model, train_loader)

def compute_rmse(predictions, targets):
    predictions = np.array(predictions) if not isinstance(predictions, np.ndarray) else predictions
    targets = np.array(targets) if not isinstance(targets, np.ndarray) else targets
    return np.sqrt(np.mean((predictions - targets) ** 2))

print(compute_rmse(train_predictions, train_targets))

# Function to get predictions from the test dataloader
def generate_test_predictions(model, loader, device="cuda" if torch.cuda.is_available() else "cpu"):
    predictions = []
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in loader:
            imgs = batch if isinstance(batch, torch.Tensor) else batch[0]
            imgs = imgs.to(device)
            outputs = model(imgs)
            predictions.extend(outputs.cpu().numpy().flatten())
    return np.array(predictions)

test_predictions = generate_test_predictions(model, test_loader)
test_data['RootVolume'] = test_predictions

# Create submission file
submission = test_data[['ID', 'RootVolume']]
submission.to_csv("submission.csv", index=False)

