import os

dataset_path = r"C:\Users\pavan\Downloads\archive\Indian Food Images\Indian Food Images"

classes = []
total_images = 0

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)

    if os.path.isdir(folder_path):
        classes.append(folder)

        images = [
            file for file in os.listdir(folder_path)
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ]

        print(f"{folder}: {len(images)} images")
        total_images += len(images)

print("\n----------------------------")
print(f"Total classes: {len(classes)}")
print(f"Total images: {total_images}")
print("----------------------------")