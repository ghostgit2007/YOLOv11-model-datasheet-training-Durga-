from pathlib import Path
from PIL import Image
import numpy as np

# ============================================================
# V9 OFFROAD PROCESSED DATASET VALIDATOR
# ============================================================

ROOT = Path(
    r"C:\UGV_Project\V9\datasets\processed\Offroad"
)

IMAGES = ROOT / "images"
MASKS = ROOT / "masks"

REPORT = ROOT / "V9_offroad_validation_report.txt"

# Expected processed mask values
EXPECTED_VALUES = {
    0,   # background
    2,   # dense-vegetation
    3,   # grass
    4,   # high_vegetation
    5,   # non_traversable_low_vegetation
    6,   # object
    7,   # obstacle
    8,   # path
    9,   # puddle
    10,  # rough_trail
    11,  # sky
    12,  # smooth_trali
    13,  # traversable_grass
    14   # vegetation
}

SPLITS = ["train", "valid", "test"]

total_images = 0
total_masks = 0
missing_masks = []
missing_images = []
invalid_masks = []
wrong_dimensions = []

report = []

report.append("V9 OFFROAD PROCESSED DATASET VALIDATION")
report.append("=" * 65)
report.append("")

# ============================================================
# Validate each split
# ============================================================

for split in SPLITS:

    image_dir = IMAGES / split
    mask_dir = MASKS / split

    print()
    print("=" * 65)
    print(f"VALIDATING: {split}")
    print("=" * 65)

    if not image_dir.exists():
        print(f"ERROR: Image directory missing: {image_dir}")
        report.append(f"{split}: IMAGE DIRECTORY MISSING")
        continue

    if not mask_dir.exists():
        print(f"ERROR: Mask directory missing: {mask_dir}")
        report.append(f"{split}: MASK DIRECTORY MISSING")
        continue

    # --------------------------------------------------------
    # Get images
    # --------------------------------------------------------

    image_files = []

    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_files.extend(image_dir.glob(ext))

    # --------------------------------------------------------
    # Get masks
    # --------------------------------------------------------

    mask_files = list(mask_dir.glob("*_mask.png"))

    image_names = {p.stem for p in image_files}

    mask_names = {
        p.name[:-9]  # remove "_mask.png"
        for p in mask_files
    }

    # --------------------------------------------------------
    # Check matching images -> masks
    # --------------------------------------------------------

    for image_name in sorted(image_names):

        if image_name not in mask_names:
            missing_masks.append(
                f"{split}: {image_name}"
            )

    # --------------------------------------------------------
    # Check matching masks -> images
    # --------------------------------------------------------

    for mask_name in sorted(mask_names):

        if mask_name not in image_names:
            missing_images.append(
                f"{split}: {mask_name}"
            )

    # --------------------------------------------------------
    # Validate mask contents
    # --------------------------------------------------------

    split_invalid = 0
    split_wrong_dimensions = 0

    for mask_path in mask_files:

        try:

            mask = Image.open(mask_path)

            array = np.array(mask)

            values = set(np.unique(array).tolist())

            # Check pixel values
            invalid_values = values - EXPECTED_VALUES

            if invalid_values:

                invalid_masks.append(
                    f"{split}: {mask_path.name} -> "
                    f"invalid values {sorted(invalid_values)}"
                )

                split_invalid += 1

            # Check grayscale
            if mask.mode != "L":

                invalid_masks.append(
                    f"{split}: {mask_path.name} -> "
                    f"mode={mask.mode}, expected L"
                )

                split_invalid += 1

            # Check dimensions
            if array.ndim != 2:

                wrong_dimensions.append(
                    f"{split}: {mask_path.name} -> "
                    f"shape={array.shape}"
                )

                split_wrong_dimensions += 1

        except Exception as e:

            invalid_masks.append(
                f"{split}: {mask_path.name} -> ERROR: {e}"
            )

            split_invalid += 1

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total_images += len(image_files)
    total_masks += len(mask_files)

    # --------------------------------------------------------
    # Console result
    # --------------------------------------------------------

    print(f"Images : {len(image_files)}")
    print(f"Masks  : {len(mask_files)}")

    print(
        f"Missing masks : "
        f"{sum(1 for x in missing_masks if x.startswith(split + ':'))}"
    )

    print(
        f"Missing images: "
        f"{sum(1 for x in missing_images if x.startswith(split + ':'))}"
    )

    print(f"Invalid masks : {split_invalid}")
    print(f"Wrong shapes  : {split_wrong_dimensions}")

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report.append(split.upper())
    report.append("-" * 45)
    report.append(f"Images: {len(image_files)}")
    report.append(f"Masks: {len(mask_files)}")

    report.append(
        "Missing masks: "
        + str(sum(1 for x in missing_masks if x.startswith(split + ":")))
    )

    report.append(
        "Missing images: "
        + str(sum(1 for x in missing_images if x.startswith(split + ":")))
    )

    report.append(f"Invalid masks: {split_invalid}")
    report.append(f"Wrong dimensions: {split_wrong_dimensions}")
    report.append("")

# ============================================================
# Final result
# ============================================================

report.append("=" * 65)
report.append("FINAL VALIDATION")
report.append("=" * 65)

report.append(f"Total images: {total_images}")
report.append(f"Total masks: {total_masks}")
report.append(f"Missing masks: {len(missing_masks)}")
report.append(f"Missing images: {len(missing_images)}")
report.append(f"Invalid masks: {len(invalid_masks)}")
report.append(f"Wrong dimensions: {len(wrong_dimensions)}")
report.append("")

# ============================================================
# Detailed errors
# ============================================================

if missing_masks:

    report.append("MISSING MASKS")
    report.append("-" * 45)

    report.extend(missing_masks)

    report.append("")


if missing_images:

    report.append("MISSING IMAGES")
    report.append("-" * 45)

    report.extend(missing_images)

    report.append("")


if invalid_masks:

    report.append("INVALID MASKS")
    report.append("-" * 45)

    report.extend(invalid_masks)

    report.append("")


if wrong_dimensions:

    report.append("WRONG DIMENSIONS")
    report.append("-" * 45)

    report.extend(wrong_dimensions)

    report.append("")


# ============================================================
# Overall status
# ============================================================

if (
    total_images == 3462
    and total_masks == 3462
    and len(missing_masks) == 0
    and len(missing_images) == 0
    and len(invalid_masks) == 0
    and len(wrong_dimensions) == 0
):

    status = "PASS"

else:

    status = "FAIL"

report.append("=" * 65)
report.append(f"VALIDATION STATUS: {status}")
report.append("=" * 65)

REPORT.write_text(
    "\n".join(report),
    encoding="utf-8"
)

# ============================================================
# Console final output
# ============================================================

print()
print("=" * 65)
print("V9 OFFROAD VALIDATION COMPLETE")
print("=" * 65)

print(f"Total images    : {total_images}")
print(f"Total masks     : {total_masks}")
print(f"Missing masks   : {len(missing_masks)}")
print(f"Missing images  : {len(missing_images)}")
print(f"Invalid masks   : {len(invalid_masks)}")
print(f"Wrong dimensions: {len(wrong_dimensions)}")

print()
print(f"VALIDATION STATUS: {status}")

print()
print("Report:")
print(REPORT)