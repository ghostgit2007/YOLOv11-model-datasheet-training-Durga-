from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(r"C:\UGV_Project\V9\datasets\raw\Offroad\Offroad-Dataset-II.v1i.png-mask-semantic")
OUTPUT = Path(r"C:\UGV_Project\V9\results\offroad_mask_inspection")

OUTPUT.mkdir(parents=True, exist_ok=True)

# Find first image and matching mask
image_files = list(ROOT.rglob("*.jpg"))

if not image_files:
    print("No JPG images found.")
    raise SystemExit

image_path = image_files[0]
mask_path = image_path.with_name(image_path.stem + "_mask.png")

if not mask_path.exists():
    print("Matching mask not found:")
    print(mask_path)
    raise SystemExit

print("Image:")
print(image_path)

print("\nMask:")
print(mask_path)

# Load
image = Image.open(image_path).convert("RGB")
mask = Image.open(mask_path).convert("L")

# Resize mask if necessary
if image.size != mask.size:
    mask = mask.resize(image.size, Image.Resampling.NEAREST)

image_np = np.array(image)
mask_np = np.array(mask)

# Create colored mask
colored = np.zeros(
    (mask_np.shape[0], mask_np.shape[1], 3),
    dtype=np.uint8
)

# Generate distinct colors automatically
colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (128, 255, 0),
    (255, 0, 128),
    (128, 128, 255),
    (255, 128, 128),
    (128, 255, 128),
    (128, 128, 128),
]

for value in range(15):
    colored[mask_np == value] = colors[value]

colored_image = Image.fromarray(colored)

# Blend original image and mask
overlay = Image.blend(image, colored_image, 0.5)

# Save
image.save(OUTPUT / "original.jpg")
mask.save(OUTPUT / "mask.png")
colored_image.save(OUTPUT / "mask_colored.png")
overlay.save(OUTPUT / "overlay.jpg")

# Print values
values = sorted(np.unique(mask_np).tolist())

print("\nMask values:")
print(values)

print("\nSaved:")
print(OUTPUT / "original.jpg")
print(OUTPUT / "mask.png")
print(OUTPUT / "mask_colored.png")
print(OUTPUT / "overlay.jpg")

print("\nDone.")