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
MODEL_PATH = r"C:\ML_project\best_resnet50_food_gpu.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 32

HEAD_EPOCHS = 2
FINETUNE_EPOCHS = 12

HEAD_LR = 0.001
FINETUNE_LR = 0.0001

RANDOM_STATE = 42


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

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("================================")
print("GPU TRAINING")
print("================================")
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU Memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
            2
        ),
        "GB"
    )


# ============================================================
# DATASET
# ============================================================

classes = sorted([
    folder
    for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(
        os.path.join(DATASET_PATH, folder)
    )
])

class_to_idx = {
    name: i
    for i, name in enumerate(classes)
}

num_classes = len(classes)

samples = []

for class_name in classes:

    class_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    for filename in os.listdir(class_path):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            samples.append(
                (
                    os.path.join(
                        class_path,
                        filename
                    ),
                    class_to_idx[class_name]
                )
            )

print("\nClasses:", num_classes)
print("Total images:", len(samples))


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

paths = [x[0] for x in samples]
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
std = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([

    transforms.Resize(256),

    transforms.RandomResizedCrop(
        224,
        scale=(0.80, 1.0)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(mean, std)
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

    def __init__(
        self,
        paths,
        labels,
        transform
    ):

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

        label = self.labels[index]

        return image, label


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


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    persistent_workers=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    persistent_workers=False
)


# ============================================================
# RESNET50
# ============================================================

print("\nLoading pretrained ResNet50...")

weights = models.ResNet50_Weights.DEFAULT

model = models.resnet50(
    weights=weights
)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

model = model.to(device)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# MIXED PRECISION
# ============================================================

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(device.type == "cuda")
)


# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_epoch(model, loader, optimizer):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(device.type == "cuda")
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        total_loss += (
            loss.item() *
            images.size(0)
        )

        predictions = outputs.argmax(1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        total_loss / total,
        correct / total
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(model, loader):

    model.eval()

    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():

        for images, labels in loader:

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
                dtype=torch.float16,
                enabled=(device.type == "cuda")
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

            total_loss += (
                loss.item() *
                images.size(0)
            )

            predictions = outputs.argmax(1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return (
        total_loss / total,
        correct / total
    )


# ============================================================
# PHASE 1 — CLASSIFIER HEAD
# ============================================================

print("\n================================")
print("PHASE 1 — CLASSIFIER HEAD")
print("================================")


for parameter in model.parameters():

    parameter.requires_grad = False


for parameter in model.fc.parameters():

    parameter.requires_grad = True


optimizer = optim.AdamW(
    model.fc.parameters(),
    lr=HEAD_LR,
    weight_decay=1e-4
)


best_accuracy = 0

for epoch in range(HEAD_EPOCHS):

    train_loss, train_acc = train_epoch(
        model,
        train_loader,
        optimizer
    )

    val_loss, val_acc = validate(
        model,
        val_loader
    )

    print(
        f"Epoch {epoch+1}/{HEAD_EPOCHS} | "
        f"Train: {train_acc*100:.2f}% | "
        f"Val: {val_acc*100:.2f}%"
    )

    if val_acc > best_accuracy:

        best_accuracy = val_acc

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "accuracy": best_accuracy
            },
            MODEL_PATH
        )

        print(
            f"🔥 BEST: {best_accuracy*100:.2f}%"
        )


# ============================================================
# PHASE 2 — FINE-TUNE LAYER4 + FC
# ============================================================

print("\n================================")
print("PHASE 2 — FINE-TUNING")
print("================================")


for parameter in model.parameters():

    parameter.requires_grad = False


for parameter in model.layer4.parameters():

    parameter.requires_grad = True


for parameter in model.fc.parameters():

    parameter.requires_grad = True


optimizer = optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=FINETUNE_LR,
    weight_decay=1e-4
)


scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


epochs_without_improvement = 0

PATIENCE = 4


for epoch in range(FINETUNE_EPOCHS):

    train_loss, train_acc = train_epoch(
        model,
        train_loader,
        optimizer
    )

    val_loss, val_acc = validate(
        model,
        val_loader
    )

    scheduler.step(val_acc)

    print(
        f"Epoch {epoch+1}/{FINETUNE_EPOCHS} | "
        f"Train: {train_acc*100:.2f}% | "
        f"Val: {val_acc*100:.2f}%"
    )

    if val_acc > best_accuracy:

        best_accuracy = val_acc

        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "accuracy": best_accuracy
            },
            MODEL_PATH
        )

        print(
            f"🔥 NEW BEST: "
            f"{best_accuracy*100:.2f}%"
        )

    else:

        epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:

            print(
                "\nEarly stopping triggered."
            )

            break


# ============================================================
# FINAL
# ============================================================

print("\n================================")
print("TRAINING FINISHED")
print("================================")

print(
    "BEST VALIDATION ACCURACY:",
    round(best_accuracy * 100, 2),
    "%"
)

print(
    "Model saved:",
    MODEL_PATH
)

print(
    "\nGPU memory used:",
    round(
        torch.cuda.max_memory_allocated() / 1024**3,
        2
    ),
    "GB"
)