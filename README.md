# 🍽️ Food101 Data Preprocessing

This repository contains a full image preprocessing pipeline for a subset of the [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/). It includes image resizing, filtering specific classes, random sampling, train/val/test splitting, and support for both `ImageFolder` and custom CSV-based PyTorch datasets.

## 📂 Project Structure

raw_food_dataset/
└── class_name/
└── image_1.jpg
└── image_2.jpg
...

cleaned_food_dataset/ ← All images resized to 512x512 JPG
filtered_food_dataset/ ← 5 filtered classes, 300 samples each
train.csv, val.csv, test.csv ← CSV files for custom Dataset



## ✅ Features

- Resize and convert all images to `.jpg` (512×512, RGB)
- Filter only selected classes (e.g., pizza, sushi, burger, etc.)
- Uniform random sampling (300 samples per class)
- Generate `train.csv`, `val.csv`, and `test.csv` with labels
- Supports:
  - `torchvision.datasets.ImageFolder`
  - Custom `torch.utils.data.Dataset` using CSV annotations

## 📦 Requirements

```bash
pip install -r requirements.txt

requirements.txt:

torch
torchvision
pillow
numpy
pandas
