# Food101 Dataset Preprocessing

This repository contains code to preprocess the Food101 dataset for image classification tasks. The preprocessing pipeline includes resizing images, filtering selected classes, uniform sampling, and generating CSV annotation files for easy dataset loading.

---

## What Was Done

- **Resize and Convert Images:**  
  All images were resized to 512×512 pixels and converted to RGB `.jpg` format for uniformity.

- **Class Filtering:**  
  Only a subset of selected classes (e.g., pizza, sushi, burger, etc.) were retained.

- **Uniform Random Sampling:**  
  For each selected class, 300 samples were randomly sampled to ensure balanced class representation.

- **CSV Annotation Files:**  
  Dataset splits were created as `train.csv`, `val.csv`, and `test.csv`, containing image paths and corresponding labels.

- **Dataset Compatibility:**  
  The processed dataset supports loading via:
  - `torchvision.datasets.ImageFolder`  
  - Custom PyTorch `Dataset` classes using CSV annotations.

---

## Important Note

This repository **does not include code to download the Food101 dataset**. The preprocessing workflow is a virtual exercise designed for practice.

### How to Download the Food101 Dataset

Download the official Food101 dataset here:

- [Food101 dataset (official)](https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz)

After downloading and extracting the dataset, place the images directory in the expected folder structure before running preprocessing.

---

## Used Libraries

The preprocessing code uses the following Python libraries:

```python
import pathlib
import shutil
import random
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

---

## Requirements

The following Python packages are required to run the preprocessing code:

- `pathlib`
- `shutil`
- `random`
- `pandas`
- `numpy`
- `Pillow`
- `torch`
- `torchvision`

You can install the required packages (except standard libraries like `pathlib`, `shutil`, and `random`) using:

```bash
pip install pandas numpy pillow torch torchvision




