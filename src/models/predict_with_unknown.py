import cv2
import numpy as np
import joblib
import os
import json
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(PROJECT_ROOT)
from src.features.extract_features import extract_feature

MODEL_DIR = PROJECT_ROOT

# -------------------------------------------------
# Load models & thresholds
# -------------------------------------------------
try:
    svm_model = joblib.load(os.path.join(MODEL_DIR, "svm_model_fast.pkl"))
    svm_scaler = joblib.load(os.path.join(MODEL_DIR, "svm_scaler.pkl"))
    knn_model = joblib.load(os.path.join(MODEL_DIR, "knn_model.pkl"))
    knn_scaler = joblib.load(os.path.join(MODEL_DIR, "knn_scaler.pkl"))

    with open(os.path.join(MODEL_DIR, "thresholds.json"), "r") as f:
        thresholds = json.load(f)

    svm_threshold = thresholds["svm_threshold"]
    knn_threshold = thresholds["knn_threshold"]

except Exception as e:
    print(f"FATAL ERROR LOADING COMPONENTS: {e}")
    exit()

print("[INFO] Thresholds loaded:")
print("  SVM threshold =", svm_threshold)
print("  KNN threshold =", knn_threshold)

# -------------------------------------------------
# Predict with Unknown (IMPROVED)
# -------------------------------------------------
def predict_with_unknown_frame(frame, debug=False):

    vector = extract_feature(frame).reshape(1, -1)

    # ========== SVM ==========
    vec_svm = svm_scaler.transform(vector)
    svm_scores = svm_model.decision_function(vec_svm)

    # margin
    if svm_scores.ndim > 1:
        svm_margin = np.max(np.abs(svm_scores))
    else:
        svm_margin = abs(svm_scores[0])

    # normalize margin (important)
    svm_margin_norm = svm_margin / (svm_margin + 1e-6)
    svm_known = svm_margin_norm > svm_threshold

    # ========== KNN ==========
    vec_knn = knn_scaler.transform(vector)
    dists, _ = knn_model.kneighbors(vec_knn, n_neighbors=1)
    knn_dist = dists[0][0]
    knn_known = knn_dist < knn_threshold

    # ========== Debug ==========
    if debug:
        print(f"SVM margin: {svm_margin:.4f} | norm: {svm_margin_norm:.4f}")
        print(f"KNN dist: {knn_dist:.4f}")
        print(f"SVM known: {svm_known} | KNN known: {knn_known}")

    # ========== Unknown decision (LESS STRICT) ==========
    if not (svm_known or knn_known):
        return "Unknown", 0.0

    # ========== Final prediction ==========
    class_id = svm_model.predict(vec_svm)[0]
    confidence = svm_margin_norm

    return class_id, confidence
