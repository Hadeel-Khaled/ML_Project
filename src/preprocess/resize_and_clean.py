#cleaning data
import os
import cv2
from PIL import Image



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
CLEANED_DATA_DIR =  os.path.join(PROJECT_ROOT, "data", "cleaned")
tsize = (224, 224)


def corrupted(image_path):
    try:
        img = Image.open(image_path)
        img.verify()
        return False
    except:
        return True


def clean_and_resize_images():
    if not os.path.exists(CLEANED_DATA_DIR):
        os.makedirs(CLEANED_DATA_DIR)

    for classname in os.listdir(RAW_DATA_DIR):
        class_path = os.path.join(RAW_DATA_DIR, classname)
        save_path = os.path.join(CLEANED_DATA_DIR, classname)

        if os.path.isdir(class_path) == False: continue

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        print(f" cleaning for {classname}")

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            img = cv2.imread(img_path)
            if img is None or corrupted(img_path):
                print(f" skip  {img_path}")
                continue

            resizedimage = cv2.resize(img, tsize)
            cv2.imwrite(os.path.join(save_path, img_name), resizedimage)

        print(f" cleaning done {classname}")

if __name__ == "__main__":
    clean_and_resize_images()
