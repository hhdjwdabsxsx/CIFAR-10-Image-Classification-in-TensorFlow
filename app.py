import os

# Suppress oneDNN and TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow info/warning logs

from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the CIFAR-10 model
MODEL_PATH = 'model.h5'
model = load_model(MODEL_PATH)

# Optionally recompile the model to remove metric warnings
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# CIFAR-10 Class Labels
CLASS_LABELS = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer", 
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded!", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected!", 400

    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Preprocess the image
        img = image.load_img(file_path, target_size=(32, 32))  # Adjust to CIFAR-10
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize

        # Prediction
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions, axis=1)[0]
        class_label = CLASS_LABELS[class_index]
        confidence = predictions[0][class_index]

        return jsonify({
            "class": class_label,
            "confidence": float(confidence)
        })

if __name__ == "__main__":
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)






















