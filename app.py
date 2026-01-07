import os
from io import BytesIO
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import requests

MODEL_PATH = "horse_model.h5"
MODEL_URL = "https://huggingface.co/Zam09ash/kuda-model-dataset/resolve/main/horse_model.h5"

IMG_SIZE = (224, 224)

# Flask app
app = Flask(__name__)

# Global variable model
model = None

# Label map
label_map = {
    "01": "Akhal-Teke",
    "02": "Appaloosa",
    "03": "Orlov Trotter",
    "04": "Vladimir Heavy Draft",
    "05": "Percheron",
    "06": "Arabian",
    "07": "Friesian"
}
labels = list(label_map.values())

# Lazy load model
def get_model():
    global model
    if model is None:
        # Download model jika belum ada
        if not os.path.exists(MODEL_PATH):
            print("Downloading model from HuggingFace...")
            with requests.get(MODEL_URL, stream=True) as r:
                r.raise_for_status()
                with open(MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            print("Model downloaded!")

        print("Loading model...")
        model = load_model(MODEL_PATH)
        print("Model loaded!")
    return model

# Predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image found"}), 400

    file = request.files['image']
    img_bytes = file.read()
    img = load_img(BytesIO(img_bytes), target_size=IMG_SIZE)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    model_instance = get_model()
    pred = model_instance.predict(img)
    idx = np.argmax(pred)

    return jsonify({
        "breed": labels[idx],
        "confidence": float(pred[0][idx])
    })

# Run
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
