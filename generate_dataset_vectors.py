from sklearn.model_selection import train_test_split
import os
import pickle
import cv2
import numpy as np
from src.features.extract_features import extract_feature

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = BASE_DIR

AUGMENTED_PATH = os.path.join(PROJECT_ROOT, "data", "augmented")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed_features")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES_FILE = os.path.join(OUTPUT_DIR, "X_features.pkl")
LABELS_FILE   = os.path.join(OUTPUT_DIR, "Y_labels.pkl")
MAP_FILE      = os.path.join(OUTPUT_DIR, "label_map.pkl")
X_TRAIN_FILE  = os.path.join(OUTPUT_DIR, "X_train.pkl")
Y_TRAIN_FILE  = os.path.join(OUTPUT_DIR, "Y_train.pkl")
X_TEST_FILE   = os.path.join(OUTPUT_DIR, "X_test.pkl")
Y_TEST_FILE   = os.path.join(OUTPUT_DIR, "Y_test.pkl")

# ================== FIXED OFFICIAL LABEL MAP ==================
label_map = {
    "glass": 0,
    "paper": 1,
    "cardboard": 2,
    "plastic": 3,
    "metal": 4,
    "trash": 5
}

all_features = []
all_labels = []

print(f"--- Starting feature extraction from: {AUGMENTED_PATH} ---")

for class_name in os.listdir(AUGMENTED_PATH):
    class_path = os.path.join(AUGMENTED_PATH, class_name)

    if not os.path.isdir(class_path):
        continue

    if class_name not in label_map:
        continue   # ignore unknown folders

    label_id = label_map[class_name]
    img_count = 0

    print(f">> Processing {class_name} (ID={label_id})")

    for img_name in os.listdir(class_path):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        feat = extract_feature(img)
        all_features.append(feat)
        all_labels.append(label_id)
        img_count += 1

    print(f"   ✓ {img_count} images processed")

X = np.array(all_features)
Y = np.array(all_labels)

print("\nFeature matrix shape:", X.shape)
print("Labels shape:", Y.shape)

with open(FEATURES_FILE, "wb") as f:
    pickle.dump(X, f)

with open(LABELS_FILE, "wb") as f:
    pickle.dump(Y, f)

with open(MAP_FILE, "wb") as f:
    pickle.dump(label_map, f)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42, stratify=Y
)

pickle.dump(X_train, open(X_TRAIN_FILE, "wb"))
pickle.dump(Y_train, open(Y_TRAIN_FILE, "wb"))
pickle.dump(X_test,  open(X_TEST_FILE,  "wb"))
pickle.dump(Y_test,  open(Y_TEST_FILE,  "wb"))

print("\n Dataset ready")
print(" Unknown class reserved as ID = 6")
