import cv2
import numpy as np
import pickle
import os
import sys
import joblib

# -------------------------------------------------
# Fix project path
# -------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# realtime → src → ML_Project
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

sys.path.append(PROJECT_ROOT)

from src.features.extract_features import extract_feature
from src.models.predict_with_unknown import predict_with_unknown_frame

# -------------------------------------------------
# Load trained SVM + scaler + label map
# -------------------------------------------------
MODEL_PATH = os.path.join(PROJECT_ROOT, "svm_model_fast.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "svm_scaler.pkl")
LABEL_MAP_PATH = os.path.join(PROJECT_ROOT, "data", "processed_features", "label_map.pkl")

svm_model = joblib.load(os.path.join(PROJECT_ROOT, "svm_model_fast.pkl"))
svm_scaler = joblib.load(os.path.join(PROJECT_ROOT, "svm_scaler.pkl"))

with open(LABEL_MAP_PATH, "rb") as f:
    label_map = pickle.load(f)

# Reverse label map
inv_label_map = {v: k for k, v in label_map.items()}

# -------------------------------------------------
# Camera
# -------------------------------------------------
cap = cv2.VideoCapture(0)
print("📸 Live camera started. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict
    label_id, conf = predict_with_unknown_frame(frame)
    if label_id == "Unknown":
        label = "Unknown"
        color = (0, 0, 255)
    else:
        label = inv_label_map[label_id]
        color = (0, 255, 0)

    # Display result
    cv2.putText(
    frame,
    f"{label} ({conf:.2f})",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    color,
    2
)
    cv2.imshow("Material Stream Identification - SVM", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    

