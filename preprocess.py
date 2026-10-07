import os
from PIL import Image, ImageOps

dataset_path = r"C:\Users\pavan\Downloads\archive\Indian Food Images\Indian Food Images"

output_path = r"C:\ML_project\processed_images"

image_size = (128, 128)

os.makedirs(output_path, exist_ok=True)

processed = 0
failed = 0

for food_class in os.listdir(dataset_path):

    input_folder = os.path.join(dataset_path, food_class)

    if not os.path.isdir(input_folder):
        continue

    output_folder = os.path.join(output_path, food_class)
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, filename)

        try:
            image = Image.open(input_file).convert("RGB")

            # Resize while preserving aspect ratio
            image.thumbnail(image_size, Image.Resampling.LANCZOS)

            # Add padding to make exactly 128 × 128
            processed_image = ImageOps.pad(
                image,
                image_size,
                method=Image.Resampling.LANCZOS,
                color=(0, 0, 0),
                centering=(0.5, 0.5)
            )

            processed_image.save(output_file, "JPEG", quality=95)

            processed += 1

        except Exception as e:
            print("Failed:", input_file)
            print(e)
            failed += 1

print("\n----------------------------")
print("Preprocessing complete!")
print("Processed:", processed)
print("Failed:", failed)
print("Output folder:", output_path)
print("----------------------------")