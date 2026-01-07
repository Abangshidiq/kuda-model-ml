# app.py
import os
from io import BytesIO
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf

# ==========================
# Konfigurasi Model
# ==========================
MODEL_PATH = "horse_model.tflite"  # File TFLite
IMG_SIZE = (224, 224)

# ==========================
# Label Map
# ==========================
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

# ==========================
# Load TFLite Interpreter
# ==========================
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==========================
# Flask App
# ==========================
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # Validasi file
    if "image" not in request.files:
        return jsonify({"error": "No image found"}), 400

    file = request.files["image"]
    img_bytes = file.read()

    # Preprocessing
    img = load_img(BytesIO(img_bytes), target_size=IMG_SIZE)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # Predict dengan TFLite
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])
    idx = int(np.argmax(pred))

    return jsonify({
        "breed": labels[idx],
        "confidence": float(pred[0][idx])
    })

# ==========================
# Jalankan app
# ==========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Railway biasanya pakai env PORT
    app.run(host="0.0.0.0", port=port)
