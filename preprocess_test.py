import os
from PIL import Image
import matplotlib.pyplot as plt

dataset_path = r"C:\Users\pavan\Downloads\archive\Indian Food Images\Indian Food Images"

# Take the first food folder
food_class = os.listdir(dataset_path)[0]
food_folder = os.path.join(dataset_path, food_class)

# Take the first image
image_file = os.listdir(food_folder)[0]
image_path = os.path.join(food_folder, image_file)

# Open image
original = Image.open(image_path).convert("RGB")

# Resize
resized = original.resize((128, 128))

print("Food class:", food_class)
print("Image file:", image_file)
print("Original size:", original.size)
print("Resized size:", resized.size)

# Display
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(original)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(resized)
plt.title("Resized 128 × 128")
plt.axis("off")

plt.tight_layout()
plt.show()