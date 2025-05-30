# utils/preprocessing.py

import numpy as np
from tensorflow.keras.preprocessing import image

def preprocess_image(img_path, target_size=(128, 128)):
    """
    Loads and preprocesses an image for prediction.

    Parameters:
        img_path (str): Path to the image file.
        target_size (tuple): Desired image size (width, height).

    Returns:
        np.array: Preprocessed image ready for model prediction.
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array
