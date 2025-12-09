import pickle
import numpy as np
import joblib
import os

print("\n--- UNKNOWN THRESHOLD CALCULATOR ---")
import os
import pickle

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))


VAL_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed_features')


X_val = pickle.load(open(os.path.join(VAL_DIR, "X_test.pkl"), "rb"))
y_val = pickle.load(open(os.path.join(VAL_DIR, "Y_test.pkl"), "rb"))


MODEL_DIR = os.path.join(PROJECT_ROOT, 'src', 'models')


print(f"[INFO] Loaded validation set (using test set): {X_val.shape} samples")

svm_model = joblib.load(os.path.join(MODEL_DIR, "svm_model_fast.pkl"))
svm_scaler = joblib.load(os.path.join(MODEL_DIR, "svm_scaler.pkl"))

knn_model = joblib.load(os.path.join(MODEL_DIR, "knn_model.pkl"))
knn_scaler = joblib.load(os.path.join(MODEL_DIR, "knn_scaler.pkl"))

print("[INFO] Models and scalers loaded successfully.")


print("\n[CALC] Computing SVM threshold...")

X_val_s_svm = svm_scaler.transform(X_val)
svm_scores = svm_model.decision_function(X_val_s_svm)


if len(svm_scores.shape) > 1:

    abs_margins = np.max(np.abs(svm_scores), axis=1)
else:

    abs_margins = np.abs(svm_scores)


svm_threshold = np.percentile(abs_margins, 5)
print(f"[RESULT] SVM threshold (5th percentile, ABSOLUTE): {svm_threshold:.4f}")


print("\n[CALC] Computing KNN threshold...")

X_val_s_knn = knn_scaler.transform(X_val)
distances, _ = knn_model.kneighbors(X_val_s_knn, n_neighbors=1, return_distance=True)

distances = distances[:, 0]
knn_threshold = np.percentile(distances, 95)

print(f"[RESULT] KNN threshold (95th percentile): {knn_threshold:.4f}")


import json

output = {
    "svm_threshold": float(svm_threshold),
    "knn_threshold": float(knn_threshold),
    "svm_percentile": 5,
    "knn_percentile": 95
}


with open(os.path.join(MODEL_DIR, "thresholds.json"), "w") as f:
    json.dump(output, f, indent=4)

print("\n[SAVED] thresholds saved to src/models/thresholds.json")
print("\n--- DONE! ---\n")
