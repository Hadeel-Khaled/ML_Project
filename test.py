import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import joblib
import os
import json
import pickle 
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

LABEL_MAP_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed_features", "label_map.pkl"
)

with open(LABEL_MAP_PATH, "rb") as f:
    label_map = pickle.load(f)
inv_label_map = {v: k for k, v in label_map.items()}


# -------------------------------------------------
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)    
model = Model(inputs=base_model.input, outputs=x)

# Freeze CNN weights
for layer in model.layers:
    layer.trainable = False

print("ResNet50 feature extractor loaded (2048 features)")

# -------------------------------------------------
# Feature extraction function
# -------------------------------------------------
def extract_feature(img):

    # BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img = cv2.resize(img, (224, 224))

    # Convert to array
    img = np.array(img, dtype=np.float32)

    # Expand dims → (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)

    # Preprocess for ResNet50
    img = preprocess_input(img)

    # Extract features
    features = model.predict(img, verbose=0)

    # Flatten → (2048,)
    return features.flatten()


# -------------------------------------------------
# REQUIRED FUNCTION
# -------------------------------------------------
def predict(dataFilePath, bestModelPath):

    # Load models
    svm_model = joblib.load(bestModelPath)
    svm_scaler = joblib.load("svm_scaler.pkl")
    knn_model = joblib.load("knn_model.pkl")
    knn_scaler = joblib.load("knn_scaler.pkl")

    with open("thresholds.json", "r") as f:
        thresholds = json.load(f)

    svm_threshold = thresholds["svm_threshold"]
    knn_threshold = thresholds["knn_threshold"]

    predictions = []

    image_files = sorted([
        f for f in os.listdir(dataFilePath)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    for img_name in image_files:
        img_path = os.path.join(dataFilePath, img_name)
        img = cv2.imread(img_path)

        if img is None:
            predictions.append({"id": -1, "label": "Invalid"})
            continue

        # ---------- Feature ----------
        vector = extract_feature(img).reshape(1, -1)

        # ---------- SVM ----------
        vec_svm = svm_scaler.transform(vector)
        svm_scores = svm_model.decision_function(vec_svm)

        if svm_scores.ndim > 1:
            svm_margin = np.max(np.abs(svm_scores))
        else:
            svm_margin = abs(svm_scores[0])

        svm_conf = 1 / (1 + np.exp(-svm_margin))
        svm_known = svm_conf > svm_threshold
        svm_pred = svm_model.predict(vec_svm)[0]

        # ---------- KNN ----------
        vec_knn = knn_scaler.transform(vector)
        dists, _ = knn_model.kneighbors(vec_knn, n_neighbors=1)
        knn_dist = dists[0][0]
        knn_known = knn_dist < knn_threshold

        # ---------- Decision ----------
        if not (svm_known or knn_known):
            predictions.append({
                "id": 6,
                "label": "Unknown"
            })
        else:
            predictions.append({
                "id": int(svm_pred),
                "label": inv_label_map[int(svm_pred)]
            })

    return predictions

if __name__ == "__main__":
    test_folder = "test"          
    model_path = "svm_model_fast.pkl"    

    results = predict(test_folder, model_path)

image_names = sorted([
    f for f in os.listdir(test_folder)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
])

rows = []
for img_name, pred in zip(image_names, results):
    rows.append({
        "ImageName": img_name,                
        "predictedlabel": pred["label"]     
    })

    df = pd.DataFrame(rows)

    output_path = "output.xlsx"
    df.to_excel(output_path, index=False)

for r in results:
    print(r)
