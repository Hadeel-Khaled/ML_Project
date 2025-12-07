# train knn placeholder


import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
import os

# -----------------------------
# Load Pickled Features
# -----------------------------


CURRENT_DIR = os.path.dirname(os.path.abspath(_file_))

PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

FEATURES_ROOT = os.path.join(PROJECT_ROOT, 'data', 'processed_features')


X_TRAIN_FILE = os.path.join(FEATURES_ROOT, 'X_train.pkl')
Y_TRAIN_FILE = os.path.join(FEATURES_ROOT, 'Y_train.pkl')
X_TEST_FILE = os.path.join(FEATURES_ROOT, 'X_test.pkl')
Y_TEST_FILE = os.path.join(FEATURES_ROOT, 'Y_test.pkl')

with open(X_TEST_FILE, 'rb') as f:
    x_test = pickle.load(f)
with open(Y_TEST_FILE, 'rb') as f:
    y_test = pickle.load(f)

with open(X_TRAIN_FILE, 'rb') as f:
    x_train = pickle.load(f)

with open(Y_TRAIN_FILE, 'rb') as f:
    y_train = pickle.load(f)

# -----------------------------
# Scale the Features
# -----------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(x_train)
X_test_s  = scaler.transform(x_test)

dump(scaler, "knn_scaler.pkl")
print("Scaler saved.")

# -----------------------------
# Train KNN
# -----------------------------
K = 3
WEIGHTS = "distance"  # or "uniform"

knn = KNeighborsClassifier(
    n_neighbors=K,
    weights=WEIGHTS,
)

knn.fit(X_train_s, y_train)
dump(knn, "knn_model.pkl")

print("KNN model saved as knn_model.pkl")

# -----------------------------
# Evaluate
# -----------------------------
y_pred = knn.predict(X_test_s)

acc = accuracy_score(y_test, y_pred)
print("\nAccuracy:", acc)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))