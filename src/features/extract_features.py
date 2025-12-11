import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

# ---------------------------------------------
# Load pretrained CNN (VGG16) without top layers
# ---------------------------------------------
base_model = VGG16(weights="imagenet", include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

# ---------------------------------------------
# Feature extraction function
# ---------------------------------------------
def extract_feature(img):
    # Convert BGR → RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to VGG expected size
    img_resized = cv2.resize(img_rgb, (224, 224))

    # Convert to array
    img_arr = img_to_array(img_resized)

    # Expand dims → (1, 224, 224, 3)
    img_arr = np.expand_dims(img_arr, axis=0)

    # Preprocess for VGG
    img_arr = preprocess_input(img_arr)

    # Extract CNN features
    features = model.predict(img_arr)

    # Flatten to 1D vector for SVM/KNN
    features = features.flatten()

    return features
