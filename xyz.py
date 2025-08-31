import tensorflow as tf
from tensorflow.keras.models import model_from_json, Sequential

# Load architecture from JSON
with open("audio_emotion_model.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json, custom_objects={"Sequential": Sequential})

# Load weights
model.load_weights("audio_emotion_model.weights.h5")

# ✅ Export SavedModel for TFLite (Keras 3 way)
model.export("audio_model_saved")  # ⬅️ This creates a SavedModel directory

# ✅ Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("audio_model_saved")
tflite_model = converter.convert()

# ✅ Save the TFLite model
with open("audio_emotion_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Successfully converted to audio_emotion_model.tflite")
