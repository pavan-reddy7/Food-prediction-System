import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import models, transforms


# =========================================================
# PATHS
# =========================================================

INPUT_DIR = r"C:\ML_project\processed_images"
OUTPUT_FILE = r"C:\ML_project\cnn_features.csv"


# =========================================================
# DEVICE
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# =========================================================
# PREPROCESSING FOR RESNET50
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# LOAD PRETRAINED RESNET50
# =========================================================

print("\nLoading pretrained ResNet50...")

weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)

# Remove final classification layer
model.fc = torch.nn.Identity()

model = model.to(device)
model.eval()

print("ResNet50 loaded!")
print("Feature size: 2048")


# =========================================================
# EXTRACT FEATURES
# =========================================================

features = []
labels = []
image_names = []

total = 0

for food_class in sorted(os.listdir(INPUT_DIR)):

    class_path = os.path.join(INPUT_DIR, food_class)

    if not os.path.isdir(class_path):
        continue

    for image_name in sorted(os.listdir(class_path)):

        image_path = os.path.join(class_path, image_name)

        try:
            image = Image.open(image_path).convert("RGB")

            image_tensor = transform(image)
            image_tensor = image_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                feature_vector = model(image_tensor)

            feature_vector = feature_vector.cpu().numpy().flatten()

            features.append(feature_vector)
            labels.append(food_class)
            image_names.append(image_name)

            total += 1

            if total % 100 == 0:
                print(f"Processed {total}/4000 images")

        except Exception as e:
            print("Error:", image_path)
            print(e)


# =========================================================
# SAVE FEATURES
# =========================================================

features = np.array(features)

print("\nFeature extraction completed!")
print("Feature matrix shape:", features.shape)

df = pd.DataFrame(
    features,
    columns=[f"cnn_feature_{i}" for i in range(features.shape[1])]
)

df["food_class"] = labels
df["image_name"] = image_names

df.to_csv(OUTPUT_FILE, index=False)

print("Saved to:", OUTPUT_FILE)
print("Total images:", len(df))