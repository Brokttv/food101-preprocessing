import pathlib
import shutil
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# ===============================
# 1. Define paths for your dataset
# ===============================
data_dir = pathlib.Path("processed_images/")      # Folder with processed images per class
new_folder = pathlib.Path("split_dataset/")       # Destination folder for train/val/test splits
new_folder.mkdir(parents=True, exist_ok=True)     # Create split folder if not exists

# ===============================
# 2. Split each class folder into train/val/test sets
# ===============================
for class_dir in data_dir.iterdir():
    if class_dir.is_dir():
        images = list(class_dir.iterdir())
        np.random.shuffle(images)                  # Shuffle images randomly

        n_train = int(len(images) * 0.8)           # 80% train
        n_val = int(len(images) * 0.1)             # 10% val
        n_test = len(images) - n_train - n_val     # Remaining 10% test

        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]

        # Helper function to copy images to their split folder
        def copy_images(images, split_name):
            split_folder = new_folder / split_name / class_dir.name
            split_folder.mkdir(parents=True, exist_ok=True)
            for img in images:
                shutil.copy2(img, split_folder)

        # Copy images to respective folders
        copy_images(train_images, "train")
        copy_images(val_images, "val")
        copy_images(test_images, "test")

# ===============================
# 3. Setup image transformations for model input
# ===============================
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),                # Resize images to 224x224
    transforms.ToTensor(),                         # Convert PIL image to tensor
    transforms.Normalize([0.5], [0.5])             # Normalize with mean=0.5, std=0.5
])

# ===============================
# 4. Create datasets using ImageFolder
# ===============================
train_dir = new_folder / "train"
val_dir = new_folder / "val"
test_dir = new_folder / "test"

train_data = datasets.ImageFolder(root=str(train_dir), transform=image_transform)
val_data = datasets.ImageFolder(root=str(val_dir), transform=image_transform)
test_data = datasets.ImageFolder(root=str(test_dir), transform=image_transform)

# ===============================
# 5. Create DataLoaders for batching and shuffling
# ===============================
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=os.cpu_count())
val_dataloader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=os.cpu_count())
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=os.cpu_count())

# ===============================
# 6. Print dataset info
# ===============================
print(f"Number of classes: {len(train_data.classes)}")
print(f"Number of samples - train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}")
print(f"Class to index mapping: {train_data.class_to_idx}")
