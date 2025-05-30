# app.py

import os
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from utils.preprocessing import preprocess_image
import numpy as np
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH = "models/trained_model.h5"
model = load_model(MODEL_PATH)
train_dir = "data/processed/train"
class_labels = sorted(os.listdir(train_dir))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message="No file part")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message="No selected file")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            img = preprocess_image(file_path)
            preds = model.predict(img)
            idx = np.argmax(preds, axis=1)[0]
            label = class_labels[idx]
            confidence = preds[0][idx]

            return render_template('index.html', filename=filename, label=label, confidence=confidence)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(debug=True)
