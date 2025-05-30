# scripts/evaluate_model.py

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

TEST_DIR = "data/processed/test"
MODEL_PATH = "models/trained_model.h5"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

def evaluate():
    datagen = ImageDataGenerator(rescale=1./255)

    test_generator = datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    model = load_model(MODEL_PATH)

    print("Predicting on test data...")
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")

    class_labels = list(test_generator.class_indices.keys())
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

if __name__ == "__main__":
    evaluate()
