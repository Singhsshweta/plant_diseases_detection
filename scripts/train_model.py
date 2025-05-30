# scripts/train_model.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.base_model import build_model

DATA_DIR = "data/processed"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
MODEL_SAVE_PATH = "models/trained_model.h5"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15  # increase epochs to benefit from augmentation

def train():
    print("Preparing data generators with augmentation...")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    print("Building model...")
    model = build_model(input_shape=(128, 128, 3), num_classes=len(train_generator.class_indices))

    print("Starting training...")
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator
    )

    print(f"Saving trained model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)

    print("Training complete.")

if __name__ == "__main__":
    train()
