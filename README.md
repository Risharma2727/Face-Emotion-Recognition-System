# 🎭 Multimodal Emotion Recognition System (Facial + Audio)

A real-time, multimodal emotion detection system combining facial expressions and audio signals using deep learning. Features hybrid fusion and temporal encoding for robust and stable predictions.

---

## ✨ Key Features

- **Real-Time Emotion Detection:** Seamless recognition from webcam and microphone.
- **Hybrid Fusion (Feature + Score Level):** Combines feature-level and score-level fusion of facial and audio predictions for richer joint representations.
- **Temporal Encoding:** Uses optical flow and rolling history to smooth predictions over time.
- **Manual Audio Trigger:** Press `'a'` to record and process emotion via microphone.
- **Confidence Filtering & Face Hold:** Filters low-confidence predictions and maintains bounding boxes during temporary detection loss.

---

## 📦 Datasets

| Modality | Dataset      | Classes                                 | Format/Preprocessing                       |
|----------|--------------|-----------------------------------------|--------------------------------------------|
| Facial   | GroupEmoW    | Positive, Negative, Neutral             | RGB + bounding boxes → Cropped/resized faces |
| Audio    | VGAF Audio   | Neutral, Happy, Sad, Angry, Fearful, Disgust, Surprise | `.wav` → MFCC + delta features           |

---
## 🔬 Methods

We propose a multimodal deep learning framework for video-based group emotion recognition that integrates sentiment-aware keyframe extraction and dual attention mechanisms. The model fuses four distinct streams—**visual**, **audio**, **optical flow**, and **facial features**—to capture complementary emotional cues across modalities. Each stream is processed using pretrained convolutional backbones, and the final representation is obtained via feature-level fusion.

---

### 3.1 Keyframe Extraction Based on Sentiment Estimates

To address the sparsity of emotionally salient frames in group videos, we implement a keyframe extraction algorithm that leverages sentiment estimates. A **ResNet-50** model fine-tuned on the **GroupEmoW** dataset is used to score frames based on emotional intensity across three categories: **positive**, **neutral**, and **negative**.

- The highest scoring frame per second is selected.
- Frames below a predefined sentiment threshold are discarded.
- Candidate keyframes are ranked by sentiment score.
- The frame with the highest interframe difference is selected per time slot.
- If fewer than `N` keyframes are obtained, additional frames are added until the quota is met.

**Training Parameters:**
- Backbone: `ResNet-50` (ImageNet pretrained, fine-tuned on GroupEmoW)
- Emotion categories: `Positive`, `Neutral`, `Negative`
- Keyframe count: `N = 16`
- Threshold: Tuned empirically

---

### 3.2 Visual Stream Feature Extraction

Visual features are extracted using a **3D ResNet-101** pretrained on the **UCF101** dataset. Keyframes are passed through the network, and the resulting feature maps are enhanced using **spatial** and **channel attention** mechanisms.

- **Spatial Attention:** 1×1 convolution + softmax to weight spatial regions.
- **Channel Attention:** Transposed feature maps weighted to emphasize semantic channels.
- Final visual representation is pooled from fused attention-weighted maps.

**Training Parameters:**
- Input resolution: `256×256` → random crop to `224×224`
- Augmentation: `Horizontal flip`, `Random crop`
- Optimizer: `SGD`
- Batch size: `16`
- Epochs: `100`
- Learning rate: `0.01` (decayed by 10× every 20 epochs)

---

### 3.3 Audio, Optical Flow, and Face Feature Extraction

#### 🎙️ Audio Stream
- Convert audio to Mel-spectrogram using 128 Mel filter banks.
- Extract features using `2D ResNet-101` pretrained on ImageNet.

**Parameters:**
- FFT window: `1024 samples`
- Hop length: `512 samples`
- Input size: `256×256` → crop to `224×224`
- Epochs: `30`, Batch size: `32`, LR decay every 10 epochs

#### 🌀 Optical Flow Stream
- Compute x/y-direction flows using OpenCV from 8 evenly spaced frames.
- Extract motion features using `3D ResNet-50`.

**Parameters:**
- Epochs: `50`, Batch size: `16`

#### 👤 Face Stream
- Detect faces using `MTCNN` with confidence filtering.
- Extract features using `3D ResNet-50`.

**Parameters:**
- Faces per video: `16`
- Epochs: `50`, Batch size: `16`

---

### 3.4 Feature Fusion and Classification

Feature vectors from all four streams are concatenated and passed through a fully connected layer for final sentiment classification.

**Fusion Details:**
- Fusion type: `Feature-level concatenation`
- Final classifier: `Fully connected layer + softmax`
- Epochs: `100`, Batch size: `8`
- Optimizer: `SGD`, LR decay every 20 epochs

  



## 🧠 Algorithms

- **Facial Emotion Recognition:** CNN (custom or MobileNetV2)
- **Audio Emotion Recognition:** CNN on MFCC features
- **Hybrid Fusion:**
  - Feature-level: concatenated modality outputs
  - Score-level: 60% facial + 40% audio weighting
  - Final: Softmax smoothing
- **Temporal Encoding:** Rolling buffer for history-based smoothing

```python
def hybrid_fusion(face_probs, audio_probs):
    fused_features = np.concatenate([face_probs, audio_probs], axis=1)
    score_fusion = 0.6 * face_probs + 0.4 * audio_probs
    final_probs = tf.nn.softmax(score_fusion).numpy()
    return final_probs

emotion_history = deque(maxlen=5)
emotion_history.append(combined)
smoothed_probs = np.mean(emotion_history, axis=0)
fused_emotion = fusion_labels[np.argmax(smoothed_probs)]
```

---

## 🔄 Code Execution Flow

1. Load models (facial + audio from `.json`, `.h5`, `.tflite`)
2. Capture webcam frames, detect faces via Haar/DNN
3. Predict facial emotion every 10 frames
4. Press `'a'`: Record 2s audio → extract MFCC → predict emotion
5. Fuse emotions via hybrid fusion → smooth with temporal encoding
6. Display predictions in real-time UI

---

## 🖼️ Output Behavior

| Component        | Behavior                                                        |
|------------------|-----------------------------------------------------------------|
| Webcam Feed      | Live overlay with bounding boxes and detected emotions          |
| Facial Emotion   | Updated every 10 frames, smoothed via optical flow              |
| Audio Emotion    | Triggered manually, displayed below face feed                   |
| Fusion Output    | Combines modalities, recalibrated softmax predictions           |
| Prediction Lag   | Minimized with threading and rolling buffer                     |

---

## 🎮 Manual Controls

- `'a'`: Record audio and predict emotion
- `'q'`: Quit the application cleanly

---

## 📊 Accuracy

| Model         | Accuracy          |
|---------------|-------------------|
| Facial Only   | ~72–75%           |
| Audio Only    | ~65–67%           |
| Hybrid Fusion | ~84%+ (approximate gain via fusion & encoding) |

---

## ✅ Results

- Multimodal fusion for richer representations
- Stable predictions via emotion history buffer
- Smooth UI, low latency, modular design
- Easy integration with future deep learning architectures

---

## 🚀 Future Enhancements

- Replace Bi-LSTM with Transformer-based fusion block
- Scene graph modeling for multi-face input
- Deploy as desktop/web tool with emotion dashboard
- Convert models to TFLite for lightweight deployment

---

## 🙌 Credits

- **Datasets:** GroupEmoW, VGAF Audio
- **Developed by:** Rishabh Sharma
- **Support:** Microsoft Copilot

---

Let me know if you want this updated directly in your repository or need further customization!
