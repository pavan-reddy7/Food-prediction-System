import os
import pandas as pd
import torch
from torchvision import models, transforms
from PIL import Image


# ==============================
# Paths
# ==============================
DATASET_PATH = r"C:\Users\pavan\Downloads\archive\Indian Food Images\Indian Food Images"
OUTPUT_PATH = r"C:\ML_project\cnn_features_v2.csv"


# ==============================
# Device
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ==============================
# Load pretrained ResNet50
# ==============================
weights = models.ResNet50_Weights.DEFAULT

model = models.resnet50(weights=weights)

# Remove final classification layer
model.fc = torch.nn.Identity()

model = model.to(device)
model.eval()


# ==============================
# Image preprocessing
# ==============================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==============================
# Extract features
# ==============================
features = []
labels = []
image_names = []

class_folders = sorted([
    folder for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
])

print("Total classes:", len(class_folders))

for class_name in class_folders:

    class_path = os.path.join(DATASET_PATH, class_name)

    image_files = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Processing {class_name}: {len(image_files)} images")

    for image_file in image_files:

        image_path = os.path.join(class_path, image_file)

        try:
            image = Image.open(image_path).convert("RGB")

            image_tensor = transform(image)
            image_tensor = image_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                feature = model(image_tensor)

            feature = feature.cpu().numpy().flatten()

            features.append(feature)
            labels.append(class_name)
            image_names.append(image_file)

        except Exception as e:
            print("Error:", image_path, e)


# ==============================
# Save CSV
# ==============================
features_df = pd.DataFrame(features)

features_df["food_class"] = labels
features_df["image_name"] = image_names

features_df.to_csv(OUTPUT_PATH, index=False)

print("\n================================")
print("FEATURE EXTRACTION COMPLETED")
print("================================")
print("Feature matrix shape:", features_df.shape)
print("Total images:", len(features_df))
print("Feature dimensions:", len(features[0]))
print("Saved to:", OUTPUT_PATH)