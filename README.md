# Food101 Dataset Preprocessing

This project preprocesses the Food101 dataset for image classification. It includes:

- Resize and convert all images to 512×512 RGB `.jpg` format  
- Filter selected classes (e.g., pizza, sushi, burger, etc.)  
- Uniform random sampling (300 samples per class)  
- Generate `train.csv`, `val.csv`, and `test.csv` with image paths and labels  
- Support for `torchvision.datasets.ImageFolder` and custom PyTorch `Dataset` using CSV annotations

## Content

- `CustomDataset_Loader.py`: Resize raw Food101 images to 512×512 RGB, filter selected classes with uniform sampling, create train/val/test CSV annotations, optionally zip CSVs, and provide a PyTorch Dataset class for CSV-based loading with transforms and DataLoaders.
  
- `ImageFolder_Loader.py`: Split a folder of class-labeled images into train/val/test folders with an 80/10/10 ratio, apply image transforms, create PyTorch ImageFolder datasets and DataLoaders, and print dataset statistics.

- `projects_structure.dm`: A visualization of each project directory structure.










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
