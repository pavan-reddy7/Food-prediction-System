import os
import cv2
import numpy as np
import pandas as pd

from skimage.feature import graycomatrix, graycoprops, hog


# --------------------------------------------------
# PATHS
# --------------------------------------------------

dataset_path = r"C:\ML_project\processed_images"
output_file = r"C:\ML_project\food_features.csv"


# --------------------------------------------------
# FEATURE EXTRACTION FUNCTION
# --------------------------------------------------

def extract_features(image_path):

    image = cv2.imread(image_path)

    # Convert BGR → RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    features = []

    # --------------------------------------------------
    # COLOR FEATURES
    # --------------------------------------------------

    # RGB: mean + standard deviation
    for channel in range(3):
        features.append(np.mean(image_rgb[:, :, channel]))
        features.append(np.std(image_rgb[:, :, channel]))

    # HSV: mean + standard deviation
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    for channel in range(3):
        features.append(np.mean(image_hsv[:, :, channel]))
        features.append(np.std(image_hsv[:, :, channel]))

    # --------------------------------------------------
    # TEXTURE FEATURES - GLCM
    # --------------------------------------------------

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Reduce gray levels from 256 → 16
    gray_reduced = (gray // 16).astype(np.uint8)

    glcm = graycomatrix(
        gray_reduced,
        distances=[1],
        angles=[0],
        levels=16,
        symmetric=True,
        normed=True
    )

    features.append(graycoprops(glcm, "contrast")[0, 0])
    features.append(graycoprops(glcm, "correlation")[0, 0])
    features.append(graycoprops(glcm, "energy")[0, 0])
    features.append(graycoprops(glcm, "homogeneity")[0, 0])

    # --------------------------------------------------
    # HOG FEATURES
    # --------------------------------------------------

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True
    )

    features.extend(hog_features)

    return features


# --------------------------------------------------
# PROCESS ALL IMAGES
# --------------------------------------------------

all_features = []
labels = []
image_names = []

total = 0
failed = 0

classes = sorted(os.listdir(dataset_path))

print("Number of classes:", len(classes))
print("Starting feature extraction...\n")

for food_class in classes:

    folder_path = os.path.join(dataset_path, food_class)

    if not os.path.isdir(folder_path):
        continue

    print("Processing:", food_class)

    for filename in os.listdir(folder_path):

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(folder_path, filename)

        try:

            features = extract_features(image_path)

            all_features.append(features)
            labels.append(food_class)
            image_names.append(filename)

            total += 1

        except Exception as e:

            print("Failed:", image_path)
            print(e)

            failed += 1


# --------------------------------------------------
# CREATE DATAFRAME
# --------------------------------------------------

print("\nCreating feature dataset...")

feature_columns = [
    f"feature_{i+1}"
    for i in range(len(all_features[0]))
]

df = pd.DataFrame(
    all_features,
    columns=feature_columns
)

df["food_class"] = labels
df["image_name"] = image_names


# --------------------------------------------------
# SAVE
# --------------------------------------------------

df.to_csv(output_file, index=False)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n----------------------------")
print("Feature extraction complete!")
print("Images processed:", total)
print("Images failed:", failed)
print("Number of features:", len(feature_columns))
print("Dataset shape:", df.shape)
print("Saved to:", output_file)
print("----------------------------")