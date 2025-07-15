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

## Setup

Install required dependencies:

```bash
pip install pandas numpy pillow torch torchvision
