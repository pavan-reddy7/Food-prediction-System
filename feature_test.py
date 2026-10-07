import os
import cv2
import numpy as np

from skimage.feature import graycomatrix, graycoprops, hog


# Path to our preprocessed dataset
dataset_path = r"C:\ML_project\processed_images"


# --------------------------------------------------
# 1. Find one image
# --------------------------------------------------

food_class = os.listdir(dataset_path)[0]
food_folder = os.path.join(dataset_path, food_class)

image_file = os.listdir(food_folder)[0]
image_path = os.path.join(food_folder, image_file)

print("Food class:", food_class)
print("Image:", image_file)


# --------------------------------------------------
# 2. Read image
# --------------------------------------------------

image = cv2.imread(image_path)

# OpenCV reads images as BGR
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("Image shape:", image_rgb.shape)


# --------------------------------------------------
# 3. COLOR FEATURES
# --------------------------------------------------

# RGB statistics
color_features = []

for channel in range(3):
    mean = np.mean(image_rgb[:, :, channel])
    std = np.std(image_rgb[:, :, channel])

    color_features.extend([mean, std])


# Convert RGB → HSV
image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

for channel in range(3):
    mean = np.mean(image_hsv[:, :, channel])
    std = np.std(image_hsv[:, :, channel])

    color_features.extend([mean, std])


print("\nColor features:")
print(color_features)


# --------------------------------------------------
# 4. TEXTURE FEATURES — GLCM
# --------------------------------------------------

gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

# Reduce gray levels to make GLCM smaller
gray_reduced = (gray // 16).astype(np.uint8)

glcm = graycomatrix(
    gray_reduced,
    distances=[1],
    angles=[0],
    levels=16,
    symmetric=True,
    normed=True
)

texture_features = [
    graycoprops(glcm, "contrast")[0, 0],
    graycoprops(glcm, "correlation")[0, 0],
    graycoprops(glcm, "energy")[0, 0],
    graycoprops(glcm, "homogeneity")[0, 0]
]

print("\nTexture features:")
print(texture_features)


# --------------------------------------------------
# 5. HOG FEATURES
# --------------------------------------------------

hog_features = hog(
    gray,
    orientations=9,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    feature_vector=True
)

print("\nHOG feature count:", len(hog_features))


# --------------------------------------------------
# 6. COMBINE EVERYTHING
# --------------------------------------------------

final_features = np.concatenate([
    np.array(color_features),
    np.array(texture_features),
    hog_features
])

print("\n----------------------------")
print("Total features:", len(final_features))
print("Feature vector shape:", final_features.shape)
print("----------------------------")