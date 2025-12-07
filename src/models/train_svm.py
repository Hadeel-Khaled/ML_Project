import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
import os

# -----------------------------
# Paths
# -----------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(_file_))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
FEATURES_ROOT = os.path.join(PROJECT_ROOT, 'data', 'processed_features')

X_TRAIN_FILE = os.path.join(FEATURES_ROOT, 'X_train.pkl')
Y_TRAIN_FILE = os.path.join(FEATURES_ROOT, 'Y_train.pkl')
X_TEST_FILE = os.path.join(FEATURES_ROOT, 'X_test.pkl')
Y_TEST_FILE = os.path.join(FEATURES_ROOT, 'Y_test.pkl')

# -----------------------------
# Load data
# -----------------------------
with open(X_TRAIN_FILE, 'rb') as f:
    x_train = pickle.load(f)
with open(Y_TRAIN_FILE, 'rb') as f:
    y_train = pickle.load(f)
with open(X_TEST_FILE, 'rb') as f:
    x_test = pickle.load(f)
with open(Y_TEST_FILE, 'rb') as f:
    y_test = pickle.load(f)

# -----------------------------
# Scale features
# -----------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(x_train)
X_test_s  = scaler.transform(x_test)
dump(scaler, "svm_scaler.pkl")
print("Scaler saved.")

# -----------------------------
# Train fast Linear SVM
# -----------------------------
svm = SGDClassifier(
    loss="hinge",    # SVM hinge loss
    max_iter=1000,   # number of passes over data
    tol=1e-3,        # stopping tolerance
    penalty="l2",    # regularization
    random_state=42
)

svm.fit(X_train_s, y_train)
dump(svm, "svm_model_fast.pkl")
print("Fast SVM model saved as svm_model_fast.pkl")

# -----------------------------
# Evaluate
# -----------------------------
y_pred = svm.predict(X_test_s)

acc = accuracy_score(y_test, y_pred)
print("\nAccuracy:", acc)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))