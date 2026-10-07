import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split

# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = r"C:\Users\pavan\Downloads\archive\Indian Food Images\Indian Food Images"
CHECKPOINT = r"C:\ML_project\best_resnet50_food_gpu_v2.pth"
OUTPUT_MODEL = r"C:\ML_project\best_resnet50_food_gpu_v2_resumed.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 32

# We already completed Phase 2 epochs 1-10.
# Continue with epochs 11-20.
REMAINING_EPOCHS = 10

LR = 5e-5
WEIGHT_DECAY = 0.01

SEED = 42

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("================================")
print("RESUMING RESNET50 V2")
print("================================")
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU Memory:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), "GB")

# ============================================================
# DATASET
# ============================================================

classes = sorted([
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
])

class_to_idx = {c: i for i, c in enumerate(classes)}

paths = []
labels = []

for class_name in classes:
    class_dir = os.path.join(DATASET_PATH, class_name)

    for file_name in os.listdir(class_dir):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            paths.append(os.path.join(class_dir, file_name))
            labels.append(class_to_idx[class_name])

print()
print("Classes:", len(classes))
print("Total images:", len(paths))

train_paths, val_paths, train_labels, val_labels = train_test_split(
    paths,
    labels,
    test_size=0.20,
    stratify=labels,
    random_state=SEED
)

print("Training:", len(train_paths))
print("Validation:", len(val_paths))

# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.80, 1.0)
    ),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15
    ),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
    transforms.RandomErasing(
        p=0.25,
        scale=(0.02, 0.15)
    )
])

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
# DATASET CLASS
# ============================================================

class FoodDataset(Dataset):

    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        image = self.transform(image)

        return image, self.labels[index]


train_dataset = FoodDataset(
    train_paths,
    train_labels,
    train_transform
)

val_dataset = FoodDataset(
    val_paths,
    val_labels,
    val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# ============================================================
# MODEL
# ============================================================

print()
print("Loading ResNet50...")

model = models.resnet50(
    weights=models.ResNet50_Weights.DEFAULT
)

# Same architecture as V2
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(
        model.fc.in_features,
        len(classes)
    )
)

model = model.to(device)

# ============================================================
# LOAD 62.50% CHECKPOINT
# ============================================================

print()
print("Loading saved checkpoint:")
print(CHECKPOINT)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

# Handle either checkpoint format
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

print("Checkpoint loaded successfully!")
print("Starting from best saved model: 62.50% validation accuracy")

# ============================================================
# FREEZE EVERYTHING EXCEPT LAYER4 + FC
# ============================================================

for param in model.parameters():
    param.requires_grad = False

for param in model.layer4.parameters():
    param.requires_grad = True

for param in model.fc.parameters():
    param.requires_grad = True

# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1
)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=REMAINING_EPOCHS
)

scaler = torch.amp.GradScaler("cuda")

# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_epoch():

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        running_loss / total,
        100.0 * correct / total
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate():

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in val_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        running_loss / total,
        100.0 * correct / total
    )


# ============================================================
# TRAINING
# ============================================================

best_accuracy = 62.50

print()
print("================================")
print("PHASE 2 — RESUMING")
print("Epochs 11 → 20")
print("================================")

for epoch in range(1, REMAINING_EPOCHS + 1):

    train_loss, train_acc = train_epoch()

    val_loss, val_acc = validate()

    scheduler.step()

    print(
        f"  Epoch {epoch + 10:02d}/20 | "
        f"Train: {train_acc:.2f}%  "
        f"Loss: {train_loss:.4f} | "
        f"Val: {val_acc:.2f}%  "
        f"Loss: {val_loss:.4f}"
    )

    if val_acc > best_accuracy:

        best_accuracy = val_acc

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "accuracy": best_accuracy
            },
            OUTPUT_MODEL
        )

        print(
            f"    💾 NEW BEST SAVED — "
            f"Val Acc: {best_accuracy:.2f}%"
        )

# ============================================================
# FINISHED
# ============================================================

print()
print("================================")
print("RESUMED TRAINING FINISHED")
print("================================")

print(
    f"BEST VALIDATION ACCURACY: "
    f"{best_accuracy:.2f}%"
)

print("Model saved to:")
print(OUTPUT_MODEL)

if torch.cuda.is_available():

    print(
        "GPU memory used:",
        round(
            torch.cuda.max_memory_allocated() / 1024**3,
            2
        ),
        "GB"
    )