import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load test data
X_test = np.load("X_face_test.npy")
y_test = np.load("y_face_test.npy")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="facial_emotion_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_tflite(sample):
    sample = sample.astype(np.float32)
    sample = np.expand_dims(sample, axis=0)
    interpreter.set_tensor(input_details[0]['index'], sample)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output)

# Predict labels
y_pred_emotion = np.array([predict_tflite(x) for x in X_test])

# Map to sentiment classes: 0 - Negative, 1 - Neutral, 2 - Positive
emotion_to_sentiment = {
    0: 0,  # angry
    1: 0,  # disgust
    2: 0,  # fear
    3: 2,  # happy
    4: 1,  # neutral
    5: 0,  # sad
    6: 2   # surprise
}

y_test_sentiment = np.array([emotion_to_sentiment[y] for y in y_test])
y_pred_sentiment = np.array([emotion_to_sentiment[y] for y in y_pred_emotion])

# Compute normalized confusion matrix
cm = confusion_matrix(y_test_sentiment, y_pred_sentiment, normalize='true')
labels = ["Negative", "Neutral", "Positive"]

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.title("Our model's Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
