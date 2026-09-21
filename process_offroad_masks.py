from pathlib import Path
from PIL import Image
import numpy as np
import shutil
from collections import Counter

# ============================================================
# V9 OFFROAD DATASET MASK PROCESSOR
# ============================================================

ROOT = Path(
    r"C:\UGV_Project\V9\datasets\raw\Offroad\Offroad-Dataset-II.v1i.png-mask-semantic"
)

OUTPUT = Path(
    r"C:\UGV_Project\V9\datasets\processed\Offroad"
)

IMAGES_OUT = OUTPUT / "images"
MASKS_OUT = OUTPUT / "masks"

REPORT = OUTPUT / "V9_offroad_processing_report.txt"

# ------------------------------------------------------------
# Create output folders
# ------------------------------------------------------------

IMAGES_OUT.mkdir(parents=True, exist_ok=True)
MASKS_OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Counters
# ------------------------------------------------------------

image_count = 0
mask_count = 0
converted_pixels = 0
invalid_masks = 0

original_values = Counter()
processed_values = Counter()

# ------------------------------------------------------------
# Process train / valid / test
# ------------------------------------------------------------

splits = ["train", "valid", "test"]

report_lines = []

report_lines.append("V9 OFFROAD DATASET PROCESSING REPORT")
report_lines.append("=" * 60)
report_lines.append("")
report_lines.append("Source:")
report_lines.append(str(ROOT))
report_lines.append("")
report_lines.append("Output:")
report_lines.append(str(OUTPUT))
report_lines.append("")
report_lines.append("Processing:")
report_lines.append("Pixel value 1 -> 0")
report_lines.append("Pixel value 0 remains 0")
report_lines.append("All other values remain unchanged")
report_lines.append("")

for split in splits:

    split_root = ROOT / split

    if not split_root.exists():
        report_lines.append(f"{split}: NOT FOUND")
        continue

    split_images_out = IMAGES_OUT / split
    split_masks_out = MASKS_OUT / split

    split_images_out.mkdir(parents=True, exist_ok=True)
    split_masks_out.mkdir(parents=True, exist_ok=True)

    split_images = 0
    split_masks = 0
    split_converted = 0

    print()
    print("=" * 60)
    print(f"Processing: {split}")
    print("=" * 60)

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    image_files = []

    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_files.extend(split_root.glob(ext))

    # Exclude mask files
    image_files = [
        p for p in image_files
        if not p.name.endswith("_mask.png")
    ]

    for image_path in image_files:

        destination = split_images_out / image_path.name

        shutil.copy2(image_path, destination)

        image_count += 1
        split_images += 1

    # --------------------------------------------------------
    # Masks
    # --------------------------------------------------------

    mask_files = list(split_root.glob("*_mask.png"))

    for mask_path in mask_files:

        try:

            image = Image.open(mask_path).convert("L")
            array = np.array(image, dtype=np.uint8)

            unique_values = np.unique(array)

            # Record original values
            for value in unique_values:
                original_values[int(value)] += 1

            # Validate original values
            if any(value > 14 for value in unique_values):
                print(
                    f"WARNING: Invalid value in {mask_path.name}: "
                    f"{unique_values}"
                )
                invalid_masks += 1

            # ------------------------------------------------
            # Merge background values
            #
            # 0 = background
            # 1 = background
            #
            # Convert 1 -> 0
            # ------------------------------------------------

            count_value_1 = np.count_nonzero(array == 1)

            if count_value_1 > 0:
                array[array == 1] = 0
                converted_pixels += count_value_1
                split_converted += count_value_1

            # Record processed values
            for value in np.unique(array):
                processed_values[int(value)] += 1

            # Save processed mask
            destination = split_masks_out / mask_path.name

            Image.fromarray(array, mode="L").save(destination)

            mask_count += 1
            split_masks += 1

        except Exception as e:

            print(f"ERROR processing {mask_path}")
            print(e)

            invalid_masks += 1

    print(f"Images processed : {split_images}")
    print(f"Masks processed  : {split_masks}")
    print(f"Pixels 1 -> 0    : {split_converted:,}")

    report_lines.append(f"{split.upper()}")
    report_lines.append("-" * 40)
    report_lines.append(f"Images: {split_images}")
    report_lines.append(f"Masks: {split_masks}")
    report_lines.append(f"Pixels converted 1 -> 0: {split_converted:,}")
    report_lines.append("")

# ------------------------------------------------------------
# Final validation
# ------------------------------------------------------------

report_lines.append("=" * 60)
report_lines.append("FINAL RESULTS")
report_lines.append("=" * 60)

report_lines.append(f"Total images processed: {image_count}")
report_lines.append(f"Total masks processed : {mask_count}")
report_lines.append(f"Total pixels 1 -> 0    : {converted_pixels:,}")
report_lines.append(f"Invalid masks          : {invalid_masks}")
report_lines.append("")

report_lines.append("Original mask values:")
for value in sorted(original_values):
    report_lines.append(
        f"{value}: {original_values[value]} mask files"
    )

report_lines.append("")
report_lines.append("Processed mask values:")
for value in sorted(processed_values):
    report_lines.append(
        f"{value}: {processed_values[value]} mask files"
    )

report_lines.append("")
report_lines.append("Expected processed values:")
report_lines.append("0  = background")
report_lines.append("2  = dense-vegetation")
report_lines.append("3  = grass")
report_lines.append("4  = high_vegetation")
report_lines.append("5  = non_traversable_low_vegetation")
report_lines.append("6  = object")
report_lines.append("7  = obstacle")
report_lines.append("8  = path")
report_lines.append("9  = puddle")
report_lines.append("10 = rough_trail")
report_lines.append("11 = sky")
report_lines.append("12 = smooth_trali")
report_lines.append("13 = traversable_grass")
report_lines.append("14 = vegetation")
report_lines.append("")
report_lines.append("Value 1 should no longer exist.")
report_lines.append("")
report_lines.append("Processing completed.")

REPORT.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

# ------------------------------------------------------------
# Console summary
# ------------------------------------------------------------

print()
print("=" * 60)
print("V9 OFFROAD PROCESSING COMPLETE")
print("=" * 60)

print(f"Images processed : {image_count}")
print(f"Masks processed  : {mask_count}")
print(f"Pixels 1 -> 0    : {converted_pixels:,}")
print(f"Invalid masks    : {invalid_masks}")

print()
print("Output:")
print(OUTPUT)

print()
print("Report:")
print(REPORT)

print()
print("Expected final mask values:")
print("[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]")