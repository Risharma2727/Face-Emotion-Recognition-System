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

# Helper Functions
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

def hybrid_fusion(face_probs, audio_probs):
    fusion = 0.6 * face_probs + 0.4 * audio_probs
    return tf.nn.softmax(fusion).numpy()

# Real-Time Webcam
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

    # Fusion
    final_probs = hybrid_fusion(face_probs, audio_probs)
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
