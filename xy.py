import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import threading
import mediapipe as mp
from tensorflow.keras.models import model_from_json, Sequential

with open("facialemotionmodel.json", "r") as f:
    face_model_json = f.read()
face_model = model_from_json(face_model_json, custom_objects={"Sequential": Sequential})
face_model.load_weights("facialemotionmodel.h5")

with open("audio_emotion_model.json", "r") as f:
    audio_model_json = f.read()
audio_model = model_from_json(audio_model_json, custom_objects={"Sequential": Sequential})
audio_model.load_weights("audio_emotion_model.weights.h5")

face_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
audio_labels = {0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad', 4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'}
fusion_labels = list(face_labels.values())

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

SAMPLE_RATE = 22050
DURATION = 2

audio_probs = np.zeros((1, 7))

def extract_face_features(image):
    feature = np.array(image).reshape(1, 48, 48, 1)
    return feature / 255.0

def record_audio_and_predict():
    global audio_probs
    audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    audio = audio.flatten()
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40)
        mfcc_processed = np.mean(mfcc.T, axis=0)
        audio_input = np.expand_dims(mfcc_processed, axis=0)
        pred = audio_model.predict(audio_input)
        audio_probs = pred[:, :7]
    except Exception as e:
        audio_probs = np.zeros((1, 7))

webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)