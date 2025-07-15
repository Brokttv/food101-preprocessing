import os
import pathlib
import shutil
import random
import pandas as pd
import numpy as np
from PIL import Image
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Adjust this to match actual extracted folder
data_dir = pathlib.Path("raw_food_dataset/images/")
cleaned_data_dir = pathlib.Path("cleaned_food_dataset/")
cleaned_data_dir.mkdir(parents=True, exist_ok=True)

filtered_data_dir = pathlib.Path("filtered_food_dataset/")
filtered_data_dir.mkdir(parents=True, exist_ok=True)

data_splits = ["pizza", "sushi", "burger", "ramen", "hot_dog"]

train_csv = pathlib.Path("train.csv")
val_csv = pathlib.Path("val.csv")
test_csv = pathlib.Path("test.csv")

np.random.seed(35)

# Resize all images and save to cleaned folder
for class_dir in data_dir.iterdir():
    if class_dir.is_dir():
        class_out_dir = cleaned_data_dir / class_dir.name
        class_out_dir.mkdir(parents=True, exist_ok=True)
        for img_pth in class_dir.iterdir():
            with Image.open(img_pth) as img:
                img = img.convert("RGB")
                resized_img = img.resize((512, 512))
                new_pth = class_out_dir / (img_pth.stem + ".jpg")
                resized_img.save(new_pth)

# Filter selected classes and sample 300 images each
for class_dir in cleaned_data_dir.iterdir():
    if class_dir.is_dir() and class_dir.name in data_splits:
        new_dir = filtered_data_dir / class_dir.name
        new_dir.mkdir(parents=True, exist_ok=True)
        images = list(class_dir.iterdir())
        filtered_samples = random.sample(images, 300)
        for img_pth in filtered_samples:
            with Image.open(img_pth) as img:
                img.save(new_dir / img_pth.name)

# Create CSV annotations
annots = []
for class_dir in filtered_data_dir.iterdir():
    if class_dir.is_dir():
        for img_pth in class_dir.iterdir():
            image_name = img_pth.relative_to(filtered_data_dir)
            label = class_dir.name
            annots.append((str(image_name), label))

np.random.shuffle(annots)
n_train = int(len(annots) * 0.8)
n_val = int(len(annots) * 0.1)
n_test = len(annots) - n_train - n_val

train_images = annots[:n_train]
val_images = annots[n_train:n_train + n_val]
test_images = annots[n_train + n_val:]

def make_csv(split, csv_path):
    df = pd.DataFrame(split, columns=["path", "label"])
    df.to_csv(csv_path, index=False)

make_csv(train_images, train_csv)
make_csv(val_images, val_csv)
make_csv(test_images, test_csv)

# Optional: copy CSV files and zip them
csv_files = [train_csv, val_csv, test_csv]
csv_dir = pathlib.Path("train_val_test_csv_files/")
csv_dir.mkdir(parents=True, exist_ok=True)

for f in csv_files:
    shutil.copy2(f, csv_dir)

shutil.make_archive("zipped_csv_files", format="zip", root_dir=csv_dir)

# Custom Dataset Class
class CustomImageDataset(Dataset):
    def __init__(self, dataset_dir, csv_path, transform=None, target_transform=None):
        self.annotations = pd.read_csv(csv_path)
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_pth = os.path.join(self.dataset_dir, self.annotations.iloc[idx, 0])
        image = read_image(img_pth)
        label = self.annotations.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# DataLoader utility
def load_csv_dataset(csv_path, dataset_dir, transform, batch_size=32, shuffle=False):
    dataset = CustomImageDataset(dataset_dir=dataset_dir, csv_path=csv_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=os.cpu_count())

# Example transforms and usage
transform = transforms.Compose([
    transforms.Resize((68, 68)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_loader = load_csv_dataset("train.csv", "filtered_food_dataset", transform=transform, shuffle=True)
val_loader   = load_csv_dataset("val.csv", "filtered_food_dataset", transform=transform)
test_loader  = load_csv_dataset("test.csv", "filtered_food_dataset", transform=transform)











