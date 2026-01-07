import os
from io import BytesIO
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from huggingface_hub import hf_hub_download

MODEL_FILENAME = "horse_model.h5"
MODEL_REPO = "Zam09ash/kuda-model-dataset"

model = None  # jangan load di init
IMG_SIZE = (224, 224)

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

app = Flask(__name__)

def get_model():
    global model
    if model is None:
        # Download dari HuggingFace jika belum ada
        if not os.path.exists(MODEL_FILENAME):
            print("Downloading model from HuggingFace...")
            MODEL_PATH = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
        else:
            MODEL_PATH = MODEL_FILENAME
        model = load_model(MODEL_PATH, compile=False)
        print("Model loaded!")
    return model

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image found"}), 400

    file = request.files['image']
    img_bytes = file.read()
    img = load_img(BytesIO(img_bytes), target_size=IMG_SIZE)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    model = get_model()
    pred = model.predict(img)
    idx = np.argmax(pred)

    return jsonify({
        "breed": labels[idx],
        "confidence": float(pred[0][idx])
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
