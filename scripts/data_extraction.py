import os
import zipfile
import shutil
import random

DATA_ZIP_PATH = "data/raw/Plant_leaf_diseases_dataset_without_augmentation.zip"
EXTRACTED_PATH = "data/raw/Plant_leaf_diseases_dataset_without_augmentation"
OUTPUT_DIR = "data/processed"
SPLIT_RATIOS = (0.7, 0.2, 0.1)  # train, val, test

def unzip_dataset(zip_path, extract_to):
    if not os.path.exists(extract_to):
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    else:
        print("Dataset already extracted.")

def split_dataset():
    print("Splitting dataset into train, val, test...")

    data_dir = os.path.join(EXTRACTED_PATH, "Plant_leave_diseases_dataset_without_augmentation")
    categories = os.listdir(data_dir)

    for category in categories:
        class_path = os.path.join(data_dir, category)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        random.shuffle(images)

        train_size = int(SPLIT_RATIOS[0] * len(images))
        val_size = int(SPLIT_RATIOS[1] * len(images))

        train_images = images[:train_size]
        val_images = images[train_size:train_size + val_size]
        test_images = images[train_size + val_size:]

        for split_name, split_images in zip(['train', 'val', 'test'], [train_images, val_images, test_images]):
            split_dir = os.path.join(OUTPUT_DIR, split_name, category)
            os.makedirs(split_dir, exist_ok=True)
            for img_name in split_images:
                src = os.path.join(class_path, img_name)
                dst = os.path.join(split_dir, img_name)
                shutil.copyfile(src, dst)

    print("Splitting completed.")

if __name__ == "__main__":
    unzip_dataset(DATA_ZIP_PATH, EXTRACTED_PATH)
    split_dataset()
    print("Data extraction and splitting completed.")