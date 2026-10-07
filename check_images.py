import os
from PIL import Image

dataset_path = r"C:\Users\pavan\Downloads\archive\Indian Food Images\Indian Food Images"

total = 0
formats = {}
sizes = {}

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)

    if not os.path.isdir(folder_path):
        continue

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        try:
            with Image.open(file_path) as img:
                total += 1

                # File format
                image_format = img.format
                formats[image_format] = formats.get(image_format, 0) + 1

                # Image dimensions
                size = img.size
                sizes[size] = sizes.get(size, 0) + 1

        except Exception as e:
            print("Problem with:", file_path)
            print(e)

print("\n----------------------------")
print("Total images:", total)

print("\nImage formats:")
for fmt, count in formats.items():
    print(fmt, ":", count)

print("\nDifferent image sizes:", len(sizes))

print("\nMost common image sizes:")
for size, count in sorted(sizes.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(size, ":", count)

print("----------------------------")