import cv2
import numpy as np
import os
import pickle
from skimage.feature import hog
from sklearn.model_selection import train_test_split

def feature_vector(img):
    #return a 1D vector represents the pic
    #convert the pic to gray using cvtColor
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #HOG for picture edges اتجاهات and divide the picture into cells, extract histogram for each cell, then collect all histograms into one vector
    hog_feat = hog(
        gray,
        orientations=9, #directions
        pixels_per_cell=(16, 16), # divide each cell into 16*16 pixel (جواها اتجاهات هنجمعها و نعملها هستوجرام )
        cells_per_block=(2, 2),
        feature_vector=True
    )

    #color information (HSV) H = Hue → S = Saturation → V = Value
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2],
        None, [8, 8, 8],
        [0, 180, 0, 256, 0, 256]
    ).flatten() # converting from 3D -> 8×8×8 = 512 ، to 1D -> 512

    features = np.concatenate([hog_feat, hist]) #عشان نجمع الشكل واللون
    return features


# BASE_DIR is the 'features' folder path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT is the 'ML_Project' folder path
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

# Target data directory (ML_Project/data/augmented)
AUGMENTED_PATH = os.path.join(PROJECT_ROOT, "data", "augmented")

# Output directory for saving features (within the 'features' folder)
OUTPUT_DIR = BASE_DIR
FEATURES_FILE = os.path.join(OUTPUT_DIR, 'X_features.pkl')
LABELS_FILE = os.path.join(OUTPUT_DIR, 'Y_labels.pkl')
MAP_FILE = os.path.join(OUTPUT_DIR, 'label_map.pkl')
X_TRAIN_FILE = os.path.join(OUTPUT_DIR, 'X_train.pkl')
Y_TRAIN_FILE = os.path.join(OUTPUT_DIR, 'Y_train.pkl')
X_TEST_FILE = os.path.join(OUTPUT_DIR, 'X_test.pkl')
Y_TEST_FILE = os.path.join(OUTPUT_DIR, 'Y_test.pkl')

all_features = [] # (X) - List to store feature vectors
all_labels = []   # (Y) - List to store corresponding labels (IDs)
label_map = {}    # Dictionary to map class names to numerical IDs (e.g., 'cat': 0)
current_label_id = 1

print(f"--- Starting image processing from: {AUGMENTED_PATH} ---")

# Iterate over subfolders (classes) in the augmented data path
for class_name in os.listdir(AUGMENTED_PATH):
    class_path = os.path.join(AUGMENTED_PATH, class_name)
    
    if os.path.isdir(class_path):
        
        # Assign a unique numerical ID to the class name
        if class_name not in label_map:
            label_map[class_name] = current_label_id
            current_label_id += 1
        
        
        label_id = label_map[class_name]
        image_count = 0
        
        print(f">> Processing class: {class_name} (ID: {label_id})")
        
        # Iterate over image files within the class folder
        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            
            # Check for common image extensions
            if os.path.isfile(image_path) and image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                
                # 1. Read the image
                img = cv2.imread(image_path)
                
                # 2. Extract features (Calling the function)
                features = feature_vector(img)
                
                # 3. Store results if extraction was successful
                if features is not None:
                    all_features.append(features)
                    all_labels.append(label_id)
                    image_count += 1
                     
        print(f"   Successfully extracted features for {image_count} images.")


# Convert lists to final NumPy arrays
X = np.array(all_features)
Y = np.array(all_labels)

print("\n-----------------------------------------------------")
print("Feature Extraction Completed.")
print(f"Total processed samples: {X.shape[0]}")
print(f"Shape of Features Matrix (X): {X.shape}")
print(f"Final Label Map: {label_map}")
print("-----------------------------------------------------")


print(f"Saving Features (X) and Labels (Y) to: {OUTPUT_DIR}")
with open(FEATURES_FILE, 'wb') as f:
    pickle.dump(X, f)
with open(LABELS_FILE, 'wb') as f:
    pickle.dump(Y, f)
with open(MAP_FILE, 'wb') as f:
    pickle.dump(label_map, f)

# Split the dataset into training and testing sets 
X_train, X_test, Y_train, Y_test = train_test_split(
    X, 
    Y,   
    test_size=0.25, 
    random_state=42, 
    stratify=Y 
)

with open(X_TRAIN_FILE, 'wb') as f:
    pickle.dump(X_train, f)
with open(Y_TRAIN_FILE, 'wb') as f:
    pickle.dump(Y_train, f)

with open(X_TEST_FILE, 'wb') as f:
    pickle.dump(X_test, f)
with open(Y_TEST_FILE, 'wb') as f:
    pickle.dump(Y_test, f)



print("\n-----------------------------------------------------")
print("Data Splitting Completed Successfully:")
print(f"- Training Set Size (X_train): {X_train.shape}")
print(f"- Testing Set Size (X_test): {X_test.shape}")
print("-----------------------------------------------------")


