# Food101 Dataset Preprocessing

This project preprocesses the Food101 dataset for image classification. It includes:

- Resize and convert all images to 512×512 RGB `.jpg` format  
- Filter selected classes (e.g., pizza, sushi, burger, etc.)  
- Uniform random sampling (300 samples per class)  
- Generate `train.csv`, `val.csv`, and `test.csv` with image paths and labels  
- Support for `torchvision.datasets.ImageFolder` and custom PyTorch `Dataset` using CSV annotations

## Contents

- Preprocessing scripts and notebooks  
- Generated CSV annotation files (`train.csv`, `val.csv`, `test.csv`)  
- Resized images directory

## Important Note

This repository **does not include code to download the Food101 dataset**. The preprocessing workflow is a virtual exercise designed for practice.

### How to Download the Food101 Dataset

Download the official Food101 dataset here:

- [Food101 dataset (official)](https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz)

After downloading and extracting the dataset, place the images directory in the expected folder structure before running preprocessing.

---

# requirements.txt

- pandas
- numpy
- pillow
- torch
- torchvision

## Setup

Install required dependencies:

```bash
pip install pandas numpy pillow torch torchvision
