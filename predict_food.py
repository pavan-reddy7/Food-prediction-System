import os
import tkinter as tk
from tkinter import filedialog

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = r"C:\ML_project\best_resnet50_food_gpu_v2_resumed.pth"


# ============================================================
# SELECT IMAGE
# ============================================================

root = tk.Tk()
root.withdraw()

IMAGE_PATH = filedialog.askopenfilename(
    title="Select Food Image",
    filetypes=[
        ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
        ("All files", "*.*")
    ]
)

root.destroy()

if not IMAGE_PATH:
    print("❌ No image selected.")
    exit()

print("\nSelected image:")
print(IMAGE_PATH)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDevice:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\nLoading model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

classes = checkpoint["classes"]

print("Classes:", len(classes))


# ============================================================
# CREATE RESNET50
# ============================================================

model = models.resnet50(weights=None)

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(
        model.fc.in_features,
        len(classes)
    )
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()

print("Model loaded successfully!")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD IMAGE
# ============================================================

print("\nLoading image...")

image = Image.open(
    IMAGE_PATH
).convert("RGB")

image_tensor = transform(
    image
).unsqueeze(0)

image_tensor = image_tensor.to(device)


# ============================================================
# PREDICTION
# ============================================================

print("Predicting...")

with torch.no_grad():

    outputs = model(
        image_tensor
    )

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    confidence, predicted = torch.max(
        probabilities,
        dim=1
    )


predicted_class = classes[
    predicted.item()
]

confidence_value = (
    confidence.item() * 100
)


# ============================================================
# TOP 5 PREDICTIONS
# ============================================================

top5_probabilities, top5_indices = torch.topk(
    probabilities,
    5
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print()
print("=" * 55)
print("       🇮🇳 INDIAN FOOD RECOGNITION")
print("=" * 55)

print(
    f"\n🍛 Predicted Food : {predicted_class}"
)

# print(
#     f"📊 Confidence     : {confidence_value:.2f}%"
# )

# print("\nTop 5 Predictions:")

# for i in range(5):

#     class_name = classes[
#         top5_indices[0][i].item()
#     ]

#     probability = (
#         top5_probabilities[0][i].item()
#         * 100
#     )

#     print(
#         f"{i + 1}. "
#         f"{class_name:<30} "
#         f"{probability:.2f}%"
#     )

# print("\n" + "=" * 55)