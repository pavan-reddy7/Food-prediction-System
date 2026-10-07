import os
import cv2
import numpy as np
import pandas as pd

from skimage.feature import hog, graycomatrix, graycoprops


# Paths
INPUT_DIR = r"C:\ML_project\processed_images"
OUTPUT_FILE = r"C:\ML_project\food_features_v2.csv"

features_list = []

# Process every food class
for food_class in sorted(os.listdir(INPUT_DIR)):

    class_path = os.path.join(INPUT_DIR, food_class)

    if not os.path.isdir(class_path):
        continue

    for image_name in os.listdir(class_path):

        image_path = os.path.join(class_path, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        # Convert BGR -> RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # =====================================================
        # 1. COLOR FEATURES
        # =====================================================

        # RGB histograms
        rgb_features = []

        for channel in range(3):
            hist = cv2.calcHist(
                [image_rgb],
                [channel],
                None,
                [16],
                [0, 256]
            )

            hist = cv2.normalize(hist, hist).flatten()
            rgb_features.extend(hist)

        # HSV histograms
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

        hsv_features = []

        for channel in range(3):
            hist = cv2.calcHist(
                [hsv],
                [channel],
                None,
                [16],
                [0, 256]
            )

            hist = cv2.normalize(hist, hist).flatten()
            hsv_features.extend(hist)

        # =====================================================
        # 2. TEXTURE FEATURES
        # =====================================================

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # Reduce gray levels for GLCM
        gray_small = (gray // 32).astype(np.uint8)

        glcm = graycomatrix(
            gray_small,
            distances=[1],
            angles=[0],
            levels=8,
            symmetric=True,
            normed=True
        )

        texture_features = [
            graycoprops(glcm, "contrast")[0, 0],
            graycoprops(glcm, "correlation")[0, 0],
            graycoprops(glcm, "energy")[0, 0],
            graycoprops(glcm, "homogeneity")[0, 0]
        ]

        # =====================================================
        # 3. SMALLER HOG FEATURES
        # =====================================================

        hog_features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(32, 32),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )

        # =====================================================
        # COMBINE ALL FEATURES
        # =====================================================

        combined_features = (
            rgb_features
            + hsv_features
            + texture_features
            + hog_features.tolist()
        )

        row = combined_features + [food_class, image_name]

        features_list.append(row)

    print(f"Processed: {food_class}")


# =========================================================
# CREATE DATAFRAME
# =========================================================

num_features = len(features_list[0]) - 2

columns = (
    [f"feature_{i}" for i in range(num_features)]
    + ["food_class", "image_name"]
)

df = pd.DataFrame(features_list, columns=columns)

df.to_csv(OUTPUT_FILE, index=False)

print("\n===================================")
print("Feature extraction completed!")
print("Dataset shape:", df.shape)
print("Number of features:", num_features)
print("Output:", OUTPUT_FILE)
print("===================================")