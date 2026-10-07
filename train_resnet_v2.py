import os
import copy
import random
import numpy as np
import torch

from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = r"C:\Users\pavan\Downloads\archive\Indian Food Images\Indian Food Images"
MODEL_PATH   = r"C:\ML_project\best_resnet50_food_gpu_v2.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 32

# Phase 1 — warm up head only
PHASE1_EPOCHS = 5
PHASE1_LR     = 1e-3

# Phase 2 — unfreeze layer4 + head
PHASE2_EPOCHS = 20
PHASE2_LR     = 5e-5

# Phase 3 — unfreeze layer3 + layer4 + head (deeper fine-tune)
PHASE3_EPOCHS = 20
PHASE3_LR     = 1e-5

WEIGHT_DECAY  = 0.01
PATIENCE      = 8
RANDOM_STATE  = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


# ============================================================
# GPU
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("================================")
print("GPU TRAINING — V2")
print("================================")
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU Memory:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB"
    )


# ============================================================
# DATASET
# ============================================================

classes = sorted([
    folder
    for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
])

class_to_idx = {name: i for i, name in enumerate(classes)}
num_classes  = len(classes)

samples = []
for class_name in classes:
    class_path = os.path.join(DATASET_PATH, class_name)
    for filename in os.listdir(class_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            samples.append((
                os.path.join(class_path, filename),
                class_to_idx[class_name]
            ))

print("\nClasses:", num_classes)
print("Total images:", len(samples))


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

paths  = [x[0] for x in samples]
labels = [x[1] for x in samples]

train_paths, val_paths, train_labels, val_labels = train_test_split(
    paths,
    labels,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=labels
)

print("Training:", len(train_paths))
print("Validation:", len(val_paths))


# ============================================================
# TRANSFORMS
# ============================================================

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.80, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.05
    ),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.20))
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


# ============================================================
# DATASET CLASS
# ============================================================

class FoodDataset(Dataset):

    def __init__(self, paths, labels, transform):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        image = self.transform(image)
        return image, self.labels[index]


train_dataset = FoodDataset(train_paths, train_labels, train_transform)
val_dataset   = FoodDataset(val_paths,   val_labels,   val_transform)


# ============================================================
# DATALOADERS  (num_workers=0 — safe on Windows)
# ============================================================

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
# MODEL — ResNet50 with Dropout head
# ============================================================

print("\nLoading pretrained ResNet50...")

weights = models.ResNet50_Weights.DEFAULT
model   = models.resnet50(weights=weights)

# Replace the classifier with Dropout(0.5) + Linear
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(in_features, num_classes)
)

model = model.to(device)


# ============================================================
# LOSS  (label smoothing = 0.1)
# ============================================================

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


# ============================================================
# MIXED PRECISION SCALER
# ============================================================

scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))


# ============================================================
# HELPERS
# ============================================================

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(device.type == "cuda")
        ):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


def validate(model, loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(device.type == "cuda")
            ):
                outputs = model(images)
                loss    = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += labels.size(0)

    return total_loss / total, correct / total


def freeze_all(m):
    for p in m.parameters():
        p.requires_grad = False


def unfreeze_modules(m, *module_names):
    for name in module_names:
        for p in getattr(m, name).parameters():
            p.requires_grad = True


def save_best(m, accuracy, path, cls_list):
    torch.save(
        {
            "model_state_dict": m.state_dict(),
            "classes":          cls_list,
            "accuracy":         accuracy
        },
        path
    )
    print(f"    💾 Saved — Val Acc: {accuracy*100:.2f}%")


# ============================================================
# PHASE 1 — CLASSIFIER HEAD ONLY
# ============================================================

print("\n================================")
print("PHASE 1 — CLASSIFIER HEAD (5 epochs)")
print("================================")

freeze_all(model)
unfreeze_modules(model, "fc")

optimizer_p1 = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=PHASE1_LR,
    weight_decay=WEIGHT_DECAY
)

best_accuracy          = 0.0
epochs_without_improve = 0

for epoch in range(PHASE1_EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer_p1)
    val_loss,   val_acc   = validate(model, val_loader)

    print(
        f"  Epoch {epoch+1:02d}/{PHASE1_EPOCHS} | "
        f"Train: {train_acc*100:.2f}%  Loss: {train_loss:.4f} | "
        f"Val: {val_acc*100:.2f}%  Loss: {val_loss:.4f}"
    )

    if val_acc > best_accuracy:
        best_accuracy          = val_acc
        epochs_without_improve = 0
        save_best(model, best_accuracy, MODEL_PATH, classes)
    else:
        epochs_without_improve += 1


# ============================================================
# PHASE 2 — LAYER4 + CLASSIFIER
# ============================================================

print("\n================================")
print("PHASE 2 — LAYER4 + CLASSIFIER (up to 20 epochs, LR=5e-5)")
print("================================")

freeze_all(model)
unfreeze_modules(model, "layer4", "fc")

optimizer_p2 = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=PHASE2_LR,
    weight_decay=WEIGHT_DECAY
)

scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_p2,
    T_max=PHASE2_EPOCHS,
    eta_min=1e-6
)

epochs_without_improve = 0

for epoch in range(PHASE2_EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer_p2)
    val_loss,   val_acc   = validate(model, val_loader)
    scheduler_p2.step()

    print(
        f"  Epoch {epoch+1:02d}/{PHASE2_EPOCHS} | "
        f"Train: {train_acc*100:.2f}%  Loss: {train_loss:.4f} | "
        f"Val: {val_acc*100:.2f}%  Loss: {val_loss:.4f}"
    )

    if val_acc > best_accuracy:
        best_accuracy          = val_acc
        epochs_without_improve = 0
        save_best(model, best_accuracy, MODEL_PATH, classes)
        print(f"    🔥 NEW BEST: {best_accuracy*100:.2f}%")
    else:
        epochs_without_improve += 1
        if epochs_without_improve >= PATIENCE:
            print("\n  Early stopping triggered in Phase 2.")
            break


# ============================================================
# PHASE 3 — LAYER3 + LAYER4 + CLASSIFIER
# ============================================================

print("\n================================")
print("PHASE 3 — LAYER3 + LAYER4 + CLASSIFIER (up to 20 epochs, LR=1e-5)")
print("================================")

freeze_all(model)
unfreeze_modules(model, "layer3", "layer4", "fc")

optimizer_p3 = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=PHASE3_LR,
    weight_decay=WEIGHT_DECAY
)

scheduler_p3 = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_p3,
    T_max=PHASE3_EPOCHS,
    eta_min=1e-7
)

epochs_without_improve = 0

for epoch in range(PHASE3_EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer_p3)
    val_loss,   val_acc   = validate(model, val_loader)
    scheduler_p3.step()

    print(
        f"  Epoch {epoch+1:02d}/{PHASE3_EPOCHS} | "
        f"Train: {train_acc*100:.2f}%  Loss: {train_loss:.4f} | "
        f"Val: {val_acc*100:.2f}%  Loss: {val_loss:.4f}"
    )

    if val_acc > best_accuracy:
        best_accuracy          = val_acc
        epochs_without_improve = 0
        save_best(model, best_accuracy, MODEL_PATH, classes)
        print(f"    🔥 NEW BEST: {best_accuracy*100:.2f}%")
    else:
        epochs_without_improve += 1
        if epochs_without_improve >= PATIENCE:
            print("\n  Early stopping triggered in Phase 3.")
            break


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n================================")
print("TRAINING FINISHED")
print("================================")
print("BEST VALIDATION ACCURACY:", round(best_accuracy * 100, 2), "%")
print("Model saved:", MODEL_PATH)

if device.type == "cuda":
    print(
        "\nPeak GPU memory used:",
        round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "GB"
    )
