import cv2
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import threading
from collections import deque

# Constants
SAMPLE_RATE = 22050
DURATION = 2
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emotion_history = deque(maxlen=5)
audio_probs = np.zeros((1, 7))
audio_thread_running = False

# Load Face Model (TFLite)
face_interpreter = tf.lite.Interpreter(model_path="facial_emotion_model.tflite")
face_interpreter.allocate_tensors()
face_input_details = face_interpreter.get_input_details()
face_output_details = face_interpreter.get_output_details()

# Load Audio Model (TFLite)
audio_interpreter = tf.lite.Interpreter(model_path="audio_emotion_model.tflite")
audio_interpreter.allocate_tensors()
audio_input_details = audio_interpreter.get_input_details()
audio_output_details = audio_interpreter.get_output_details()

# ---------------- Transformer Fusion Block ----------------
class TransformerFusion(tf.keras.Model):
    def __init__(self, num_heads=4, embed_dim=128, ff_dim=256, num_classes=7):
        super(TransformerFusion, self).__init__()
        self.embedding_face = tf.keras.layers.Dense(embed_dim)   # face projection
        self.embedding_audio = tf.keras.layers.Dense(embed_dim)  # audio projection
        
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.classifier = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, face_vec, audio_vec):
        # Project inputs
        face_emb = self.embedding_face(face_vec)     # (batch, embed_dim)
        audio_emb = self.embedding_audio(audio_vec)  # (batch, embed_dim)

        # Stack into sequence [face, audio]
        tokens = tf.stack([face_emb, audio_emb], axis=1)  # (batch, 2, embed_dim)

        # Self-attention
        attn_output = self.attention(tokens, tokens)
        out1 = self.norm1(tokens + attn_output)

        # Feed-forward
        ffn_output = self.ffn(out1)
        out2 = self.norm2(out1 + ffn_output)

        # Pool sequence → classification
        fused_rep = tf.reduce_mean(out2, axis=1)
        return self.classifier(fused_rep)

# Instantiate fusion model
transformer_fusion = TransformerFusion(num_classes=len(EMOTION_LABELS))

# ---------------- Helper Functions ----------------
def extract_face_features(img_gray):
    roi = cv2.resize(img_gray, (48, 48)).astype(np.float32)
    roi = roi.reshape(1, 48, 48, 1) / 255.0
    return roi

def predict_face_emotion(roi):
    face_interpreter.set_tensor(face_input_details[0]['index'], roi)
    face_interpreter.invoke()
    return face_interpreter.get_tensor(face_output_details[0]['index'])

def predict_audio_emotion(audio_input):
    audio_interpreter.set_tensor(audio_input_details[0]['index'], audio_input.astype(np.float32))
    audio_interpreter.invoke()
    return audio_interpreter.get_tensor(audio_output_details[0]['index'])

def record_audio_and_predict():
    global audio_probs, audio_thread_running
    if audio_thread_running:
        return
    audio_thread_running = True
    try:
        audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()
        audio = audio.flatten()
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40)
        mfcc_processed = np.mean(mfcc.T, axis=0).reshape(1, -1).astype(np.float32)
        audio_probs = predict_audio_emotion(mfcc_processed)
    except Exception as e:
        print("Audio Error:", e)
        audio_probs = np.zeros((1, 7))
    audio_thread_running = False

# ---------------- Transformer Fusion Wrapper ----------------
def transformer_fuse(face_probs, audio_probs):
    # Convert to tensors
    face_vec = tf.convert_to_tensor(face_probs, dtype=tf.float32)
    audio_vec = tf.convert_to_tensor(audio_probs, dtype=tf.float32)

    # Pass through Transformer fusion
    return transformer_fusion(face_vec, audio_vec).numpy()

# ---------------- Real-Time Webcam ----------------
cap = cv2.VideoCapture(0)
threading.Thread(target=record_audio_and_predict).start()

print("[INFO] Press 'a' to capture audio | 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_probs = np.zeros((1, 7))

    # Haar face detection
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        roi = extract_face_features(gray[y:y+h, x:x+w])
        face_probs = predict_face_emotion(roi)
        face_label = EMOTION_LABELS[np.argmax(face_probs)]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Face: {face_label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Transformer Fusion
    final_probs = transformer_fuse(face_probs, audio_probs)
    emotion_history.append(final_probs)
    smoothed_probs = np.mean(emotion_history, axis=0)
    fused_emotion = EMOTION_LABELS[np.argmax(smoothed_probs)]

    cv2.putText(frame, f"Fused: {fused_emotion}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.imshow("Multimodal Real-Time Emotion Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):
        threading.Thread(target=record_audio_and_predict).start()
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
