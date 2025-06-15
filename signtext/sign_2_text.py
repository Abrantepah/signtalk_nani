import os
import cv2
import numpy as np
import mediapipe as mp
from transformers import T5Tokenizer, T5ForConditionalGeneration

# === Constants ===
POSE_COUNT, FACE_COUNT, HAND_COUNT = 33, 468, 21
FRAME_COUNT = 10
FPS = 20
RECORD_SECONDS = 11
RESOLUTION = (640, 480)
VECTOR_SIZE = (POSE_COUNT * 4 + FACE_COUNT * 3 + HAND_COUNT * 3 * 2)
TOTAL_VECTOR_LENGTH = FRAME_COUNT * VECTOR_SIZE

# === Load model & tokenizer ===
model = T5ForConditionalGeneration.from_pretrained("sign2text_model4")
tokenizer = T5Tokenizer.from_pretrained("sign2text_model4")

# === Setup MediaPipe ===
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=True,
    model_complexity=1,
    enable_segmentation=False,
    refine_face_landmarks=True,
    min_detection_confidence=0.5
)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(POSE_COUNT * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(FACE_COUNT * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(HAND_COUNT * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(HAND_COUNT * 3)
    return np.concatenate([pose, face, lh, rh])

def extract_vector_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, FRAME_COUNT, dtype=int)

    sequence = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

    cap.release()

    if len(sequence) == 0:
        print("⚠️ No valid frames found.")
        return " ".join(["0.00000"] * TOTAL_VECTOR_LENGTH)

    while len(sequence) < FRAME_COUNT:
        sequence.append(np.zeros_like(sequence[0]))

    vector = np.concatenate(sequence)
    return " ".join([f"{v:.5f}" for v in vector])

def predict_translation_from_video(video_path):
    input_vector = extract_vector_from_video(video_path)
    inputs = tokenizer(input_vector, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
