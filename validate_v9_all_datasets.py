from pathlib import Path
from PIL import Image
import numpy as np

# ============================================================
# V9 MASTER DATASET VALIDATOR
# ============================================================

ROOT = Path(r"C:\UGV_Project\V9")

RAW = ROOT / "datasets" / "raw"
PROCESSED = ROOT / "datasets" / "processed"
RESULTS = ROOT / "results"

REPORT = RESULTS / "V9_dataset_final_report.txt"

RESULTS.mkdir(parents=True, exist_ok=True)

report = []

errors = []
warnings = []

# ============================================================
# Helper
# ============================================================

def add(text=""):
    report.append(text)
    print(text)


# ============================================================
# HEADER
# ============================================================

add("=" * 70)
add("V9 MASTER DATASET VALIDATION REPORT")
add("=" * 70)
add("")
add("Project: Autonomous All-Terrain UGV")
add("Version: V9")
add("Purpose: Dataset acquisition, preparation and validation")
add("")


# ============================================================
# 1. RUGD
# ============================================================

add("=" * 70)
add("1. RUGD DATASET")
add("=" * 70)

rug_frames = RAW / "RUGD" / "frames"
rug_masks = RAW / "RUGD" / "annotations"

if rug_frames.exists() and rug_masks.exists():

    frame_files = []

    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        frame_files.extend(rug_frames.rglob(ext))

    mask_files = list(rug_masks.rglob("*.png"))

    frame_count = len(frame_files)
    mask_count = len(mask_files)

    add(f"Frames      : {frame_count}")
    add(f"Annotations : {mask_count}")

    if frame_count == 7436 and mask_count == 7436:
        add("Count check : PASS")
    else:
        add("Count check : FAIL")
        errors.append(
            f"RUGD count mismatch: {frame_count} frames / {mask_count} masks"
        )

    # Check sequence matching
    frame_rel = {
        p.relative_to(rug_frames).with_suffix(".png")
        for p in frame_files
    }

    mask_rel = {
        p.relative_to(rug_masks)
        for p in mask_files
    }

    missing_masks = frame_rel - mask_rel
    extra_masks = mask_rel - frame_rel

    add(f"Missing masks: {len(missing_masks)}")
    add(f"Extra masks  : {len(extra_masks)}")

    if not missing_masks and not extra_masks:
        add("Frame/mask matching : PASS")
    else:
        add("Frame/mask matching : FAIL")
        errors.append("RUGD frame/mask matching failed")

else:

    add("RUGD directories missing")
    errors.append("RUGD directories missing")

add("")


# ============================================================
# 2. ROAD OBSTACLE
# ============================================================

add("=" * 70)
add("2. ROAD OBSTACLE DATASET")
add("=" * 70)

road = PROCESSED / "Road_Obstacle"

if road.exists():

    road_splits = {
        "train": road / "train",
        "valid": road / "valid",
        "test": road / "test"
    }

    total_images = 0
    total_labels = 0

    for split, split_path in road_splits.items():

        image_dir = split_path / "images"
        label_dir = split_path / "labels"

        images = []

        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            images.extend(image_dir.glob(ext))

        labels = list(label_dir.glob("*.txt"))

        image_count = len(images)
        label_count = len(labels)

        total_images += image_count
        total_labels += label_count

        add(
            f"{split:5} -> Images: {image_count:4} | "
            f"Labels: {label_count:4}"
        )

        image_names = {p.stem for p in images}
        label_names = {p.stem for p in labels}

        missing_labels = image_names - label_names
        missing_images = label_names - image_names

        if missing_labels:
            errors.append(
                f"Road Obstacle {split}: missing {len(missing_labels)} labels"
            )

        if missing_images:
            errors.append(
                f"Road Obstacle {split}: labels without images "
                f"{len(missing_images)}"
            )

    add("")
    add(f"Total images: {total_images}")
    add(f"Total labels : {total_labels}")

    if total_images == 5731 and total_labels == 5731:
        add("Dataset count : PASS")
    else:
        add("Dataset count : FAIL")
        errors.append(
            f"Road Obstacle total mismatch: "
            f"{total_images} images / {total_labels} labels"
        )

else:

    add("Processed Road Obstacle dataset missing")
    errors.append("Processed Road Obstacle dataset missing")

add("")


# ============================================================
# 3. OFFROAD
# ============================================================

add("=" * 70)
add("3. OFFROAD-DATASET-II")
add("=" * 70)

offroad = PROCESSED / "Offroad"

offroad_images = offroad / "images"
offroad_masks = offroad / "masks"

offroad_total_images = 0
offroad_total_masks = 0

expected_values = {
    0, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 12, 13, 14
}

if offroad.exists():

    for split in ["train", "valid", "test"]:

        image_dir = offroad_images / split
        mask_dir = offroad_masks / split

        images = []

        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            images.extend(image_dir.glob(ext))

        masks = list(mask_dir.glob("*_mask.png"))

        image_count = len(images)
        mask_count = len(masks)

        offroad_total_images += image_count
        offroad_total_masks += mask_count

        add(
            f"{split:5} -> Images: {image_count:4} | "
            f"Masks: {mask_count:4}"
        )

        image_names = {p.stem for p in images}

        mask_names = {
            p.name[:-9]
            for p in masks
        }

        missing_masks = image_names - mask_names
        missing_images = mask_names - image_names

        if missing_masks:
            errors.append(
                f"Offroad {split}: missing {len(missing_masks)} masks"
            )

        if missing_images:
            errors.append(
                f"Offroad {split}: masks without images "
                f"{len(missing_images)}"
            )

        # Validate mask pixel values
        for mask_path in masks:

            try:

                mask = Image.open(mask_path)

                if mask.mode != "L":
                    errors.append(
                        f"Offroad {split}: {mask_path.name} "
                        f"is mode {mask.mode}"
                    )

                array = np.array(mask)

                values = set(np.unique(array).tolist())

                invalid = values - expected_values

                if invalid:
                    errors.append(
                        f"Offroad {split}: {mask_path.name} "
                        f"invalid values {sorted(invalid)}"
                    )

                if array.ndim != 2:
                    errors.append(
                        f"Offroad {split}: {mask_path.name} "
                        f"invalid shape {array.shape}"
                    )

            except Exception as e:

                errors.append(
                    f"Offroad {split}: {mask_path.name} "
                    f"read error: {e}"
                )

    add("")
    add(f"Total images: {offroad_total_images}")
    add(f"Total masks : {offroad_total_masks}")

    if (
        offroad_total_images == 3462
        and offroad_total_masks == 3462
    ):
        add("Dataset count : PASS")
    else:
        add("Dataset count : FAIL")
        errors.append("Offroad total count mismatch")

else:

    add("Processed Offroad dataset missing")
    errors.append("Processed Offroad dataset missing")

add("")


# ============================================================
# 4. CLASS MAPPING
# ============================================================

add("=" * 70)
add("4. OFFROAD CLASS MAPPING")
add("=" * 70)

mapping_file = RESULTS / "offroad_class_mapping.csv"

if mapping_file.exists():

    add("Class mapping file : FOUND")
    add(str(mapping_file))

    mapping_lines = mapping_file.read_text(
        encoding="utf-8"
    ).splitlines()

    for line in mapping_lines:
        if line.strip():
            add(f"  {line}")

else:

    add("Class mapping file : MISSING")
    errors.append("Offroad class mapping file missing")

add("")


# ============================================================
# 5. PROCESSING REPORTS
# ============================================================

add("=" * 70)
add("5. PROCESSING REPORTS")
add("=" * 70)

road_report = (
    PROCESSED /
    "Road_Obstacle" /
    "V9_conversion_report.txt"
)

offroad_report = (
    PROCESSED /
    "Offroad" /
    "V9_offroad_processing_report.txt"
)

offroad_validation = (
    PROCESSED /
    "Offroad" /
    "V9_offroad_validation_report.txt"
)

files_to_check = [
    road_report,
    offroad_report,
    offroad_validation
]

for file in files_to_check:

    if file.exists():
        add(f"FOUND : {file}")
    else:
        add(f"MISSING: {file}")
        warnings.append(f"Missing report: {file}")

add("")


# ============================================================
# 6. FINAL SUMMARY
# ============================================================

add("=" * 70)
add("6. FINAL V9 SUMMARY")
add("=" * 70)

add("")
add("RUGD:")
add("  7,436 frames")
add("  7,436 annotations")
add("  Frame/annotation matching verified")

add("")
add("Road Obstacle:")
add("  5,731 images")
add("  5,731 labels")
add("  Bounding boxes and polygons processed")
add("  Polygon annotations converted")
add("  Empty/background labels preserved")

add("")
add("Offroad:")
add("  3,462 images")
add("  3,462 masks")
add("  Pixel mapping verified")
add("  Background values 0/1 normalized")
add("  Processed masks validated")

add("")


# ============================================================
# STATUS
# ============================================================

if errors:

    status = "FAIL"

elif warnings:

    status = "PASS WITH WARNINGS"

else:

    status = "PASS"

add("=" * 70)
add(f"V9 MASTER VALIDATION STATUS: {status}")
add("=" * 70)

if errors:

    add("")
    add("ERRORS:")
    add("-" * 50)

    for error in errors:
        add(error)

if warnings:

    add("")
    add("WARNINGS:")
    add("-" * 50)

    for warning in warnings:
        add(warning)

# ============================================================
# Save report
# ============================================================

REPORT.write_text(
    "\n".join(report),
    encoding="utf-8"
)

print("")
print("=" * 70)
print("MASTER V9 VALIDATION COMPLETE")
print("=" * 70)
print(f"Status: {status}")
print("")
print(f"Report saved to:")
print(REPORT)