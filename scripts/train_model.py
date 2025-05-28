import os
import argparse
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(input_dir, output_dir, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1):
    subdirs = ['train', 'valid', 'test']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            if not images:
                print(f"Skipping empty class: {class_name}")
                continue

            train_imgs, temp_imgs = train_test_split(images, train_size=train_ratio, random_state=42)
            valid_imgs, test_imgs = train_test_split(temp_imgs, test_size=test_ratio / (valid_ratio + test_ratio), random_state=42)

            for subset, subset_imgs in zip(subdirs, [train_imgs, valid_imgs, test_imgs]):
                output_path = os.path.join(output_dir, subset, class_name)
                os.makedirs(output_path, exist_ok=True)
                for img in subset_imgs:
                    shutil.copy(os.path.join(class_path, img), os.path.join(output_path, img))
            print(f"{class_name}: {len(train_imgs)} train, {len(valid_imgs)} valid, {len(test_imgs)} test")

def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/valid/test folders.")
    parser.add_argument("input_dir", help="Path to input dataset folder")
    parser.add_argument("output_dir", help="Path to save processed dataset")
    args = parser.parse_args()

    split_dataset(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
