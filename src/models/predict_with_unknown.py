import cv2
import numpy as np
import joblib
import os
import json
from src.features.extract_features import extract_feature

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
MODEL_DIR = os.path.join(PROJECT_ROOT, "src", "models")


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



def predict_with_unknown(img_path):

    img = cv2.imread(img_path)
    if img is None:
        print("Error loading image:", img_path)
        return "Invalid"



    vector = extract_feature(img).reshape(1, -1)


    vec_svm = svm_scaler.transform(vector)
    svm_scores = svm_model.decision_function(vec_svm)

    if len(svm_scores.shape) > 1:
        svm_margin = np.max(np.abs(svm_scores))
    else:
        svm_margin = abs(svm_scores[0])

    svm_known = svm_margin > svm_threshold

    vec_knn = knn_scaler.transform(vector)
    dists, _ = knn_model.kneighbors(vec_knn, n_neighbors=1)
    knn_dist = dists[0][0]

    knn_known = knn_dist < knn_threshold

    if not (svm_known and knn_known):
        return "Unknown"

    final_class_id = svm_model.predict(vec_svm)[0]
    return final_class_id


