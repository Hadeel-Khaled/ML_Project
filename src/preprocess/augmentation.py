# Augmentation
import os
import cv2
import random
import albumentations as A

base_dir = os.path.dirname(os.path.abspath(__file__))
projectroot = os.path.dirname(os.path.dirname(base_dir))

cleanpath = os.path.join(projectroot, "data", "cleaned")
augmpath = os.path.join(projectroot, "data", "augmented")

t_count = 500


def create_output_dirs():
    for class_name in os.listdir(cleanpath):
        path = os.path.join(augmpath, class_name)
        os.makedirs(path, exist_ok=True)


# ---------------------------------------
# Augmentation pipeline (ResNet-friendly)
# ---------------------------------------
augmentation_seq = A.Compose([

    # Geometry
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.3),

    # Color & Lighting (important for ResNet)
    A.RandomBrightnessContrast(
        brightness_limit=0.15,
        contrast_limit=0.15,
        p=0.4
    ),
    A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=10,
        val_shift_limit=10,
        p=0.3
    ),

    # Light blur only
    A.GaussianBlur(blur_limit=(3, 3), p=0.1),

    # ImageNet-style resize & crop
    A.Resize(256, 256),
    A.RandomCrop(224, 224)
])


def augment_image(img):
    return augmentation_seq(image=img)["image"]


def augment_dataset():
    create_output_dirs()

    for class_name in os.listdir(cleanpath):
        clean_path = os.path.join(cleanpath, class_name)
        aug_path = os.path.join(augmpath, class_name)

        image_list = sorted(os.listdir(clean_path))
        original_count = len(image_list)

        print(f"[{class_name}] original = {original_count}, target = {t_count}")

        idx = 0
        remaining_images = image_list.copy()

        while len(os.listdir(aug_path)) < t_count:
            if remaining_images:
                img_name = remaining_images.pop(0)
            else:
                img_name = random.choice(image_list)

            img_path = os.path.join(clean_path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                print(f"error reading {img_path}")
                continue

            aug_img = augment_image(img)

            save_name = f"aug_{os.path.splitext(img_name)[0]}_{idx}.jpg"
            save_path = os.path.join(aug_path, save_name)

            cv2.imwrite(save_path, aug_img)
            idx += 1

        print(f"augmentation done for {class_name}")


if __name__ == "__main__":
    augment_dataset()