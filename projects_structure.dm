#Directory structure CustomDataset_Loader.py

project_root/
├── raw_food_dataset/
│   └── images/
│       ├── pizza/
│       ├── sushi/
│       ├── burger/
│       ├── ramen/
│       └── hot_dog/
├── cleaned_food_dataset/               # resized 512x512 RGB images for all classes
│   ├── pizza/
│   ├── sushi/
│   ├── burger/
│   ├── ramen/
│   └── hot_dog/
├── filtered_food_dataset/              # sampled 300 images per selected class
│   ├── pizza/
│   ├── sushi/
│   ├── burger/
│   ├── ramen/
│   └── hot_dog/
├── train.csv                          # CSV annotations for training set
├── val.csv                            # CSV annotations for validation set
├── test.csv                           # CSV annotations for test set
├── train_val_test_csv_files/          # copied CSV files for zipping
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── zipped_csv_files.zip                # zipped CSV folder archive


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

#Directory structure for ImageFolder_Loader.py

project_root/
├── processed_images/                   # folder with processed images organized by class
│   ├── pizza/
│   ├── sushi/
│   ├── burger/
│   ├── ramen/
│   └── hot_dog/
└── split_dataset/                     # folder containing train/val/test splits by class
    ├── train/
    │   ├── pizza/
    │   ├── sushi/
    │   ├── burger/
    │   ├── ramen/
    │   └── hot_dog/
    ├── val/
    │   ├── pizza/
    │   ├── sushi/
    │   ├── burger/
    │   ├── ramen/
    │   └── hot_dog/
    └── test/
        ├── pizza/
        ├── sushi/
        ├── burger/
        ├── ramen/
        └── hot_dog/
