# scripts/demo.py

import sys
import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocessing import preprocess_image

MODEL_PATH = "models/trained_model.h5"

def predict(image_path):
    model = load_model(MODEL_PATH)
    img = preprocess_image(image_path)
    
    predictions = model.predict(img)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    
    # Load class labels from the training data directory structure
    import os
    train_dir = "data/processed/train"
    class_labels = sorted(os.listdir(train_dir))
    
    predicted_class_label = class_labels[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx]
    
    print(f"Predicted class: {predicted_class_label} (confidence: {confidence:.2f})")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/demo.py path_to_image")
        sys.exit(1)
    image_path = sys.argv[1]
    predict(image_path)
