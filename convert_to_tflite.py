import tensorflow as tf

# Load model Keras .h5
model = tf.keras.models.load_model("horse_model.h5", compile=False)

# Convert ke TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Simpan file .tflite
with open("horse_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model berhasil dikonversi ke .tflite")
