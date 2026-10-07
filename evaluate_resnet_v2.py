import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = r"C:\Users\pavan\Downloads\archive\Indian Food Images\Indian Food Images"

MODEL_PATH = r"C:\ML_project\best_resnet50_food_gpu_v2_resumed.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 32
SEED = 42

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("================================")
print("RESNET50 V2 MODEL EVALUATION")
print("================================")
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ============================================================
# CLASSES
# ============================================================

classes = sorted([
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
])

class_to_idx = {c: i for i, c in enumerate(classes)}

# ============================================================
# COLLECT IMAGES
# ============================================================

paths = []
labels = []

for class_name in classes:

    class_dir = os.path.join(
        DATASET_PATH,
        class_name
    )

    for file_name in os.listdir(class_dir):

        if file_name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            paths.append(
                os.path.join(
                    class_dir,
                    file_name
                )
            )

            labels.append(
                class_to_idx[class_name]
            )

print()
print("Classes:", len(classes))
print("Total images:", len(paths))

# ============================================================
# SAME 80/20 SPLIT USED DURING TRAINING
# ============================================================

_, val_paths, _, val_labels = train_test_split(
    paths,
    labels,
    test_size=0.20,
    stratify=labels,
    random_state=SEED
)

print("Validation images:", len(val_paths))

# ============================================================
# VALIDATION TRANSFORM
# ============================================================

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ============================================================
# DATASET
# ============================================================

class FoodDataset(Dataset):

    def __init__(self, paths, labels, transform):

        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        image = Image.open(
            self.paths[index]
        ).convert("RGB")

        image = self.transform(image)

        return image, self.labels[index]


val_dataset = FoodDataset(
    val_paths,
    val_labels,
    val_transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading model:")
print(MODEL_PATH)

model = models.resnet50(
    weights=None
)

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(
        model.fc.in_features,
        len(classes)
    )
)

model = model.to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded successfully!")

# ============================================================
# PREDICTION
# ============================================================

all_predictions = []
all_labels = []

print()
print("Evaluating...")

with torch.no_grad():

    for images, labels_batch in val_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels_batch.numpy()
        )

# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print()
print("================================")
print("RESULTS")
print("================================")

print(
    f"Validation Accuracy: "
    f"{accuracy * 100:.2f}%"
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("================================")
print("CLASSIFICATION REPORT")
print("================================")

report = classification_report(
    all_labels,
    all_predictions,
    target_names=classes,
    digits=3,
    zero_division=0
)

print(report)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

plt.figure(figsize=(22, 22))

plt.imshow(cm)

plt.title(
    "Indian Food Classification - Confusion Matrix"
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    range(len(classes)),
    classes,
    rotation=90,
    fontsize=6
)

plt.yticks(
    range(len(classes)),
    classes,
    fontsize=6
)

plt.colorbar()

plt.tight_layout()

output_path = r"C:\ML_project\confusion_matrix_v2.png"

plt.savefig(
    output_path,
    dpi=200
)

plt.show()

print()
print("Confusion matrix saved to:")
print(output_path)