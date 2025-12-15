import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# -------------------------------------------------
# Load ResNet50 pretrained model (feature extractor)
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

print("✅ ResNet50 feature extractor loaded (2048 features)")

# -------------------------------------------------
# Feature extraction function
# -------------------------------------------------
def extract_feature(img):
    """
    Extract CNN features from a single image
    Output: 1D numpy vector (2048,)
    """

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
