from pathlib import Path
import shutil


# ============================================================
# V10.8.2 FINAL TERRAIN DATASET PREPARATION
# ============================================================

V10 = Path(r"C:\UGV_Project\V10")
V9 = Path(r"C:\UGV_Project\V9")

# Source datasets
RUGD_SOURCE = V9 / "datasets" / "raw" / "RUGD"
OFFROAD_SOURCE = V9 / "datasets" / "processed" / "Offroad"

# Final destination
FINAL_ROOT = V10 / "datasets" / "final" / "Terrain"

RUGD_FRAMES_SOURCE = RUGD_SOURCE / "frames"
RUGD_ANNOTATIONS_SOURCE = RUGD_SOURCE / "annotations"

RUGD_FINAL = FINAL_ROOT / "RUGD"
RUGD_FRAMES_FINAL = RUGD_FINAL / "frames"
RUGD_ANNOTATIONS_FINAL = RUGD_FINAL / "annotations"

OFFROAD_FINAL = FINAL_ROOT / "Offroad"

print("=" * 70)
print("V10.8.2 FINAL TERRAIN DATASET PREPARATION")
print("=" * 70)

# ============================================================
# REMOVE PREVIOUS FINAL DATASET
# ============================================================

if FINAL_ROOT.exists():
    print()
    print("Removing previous final terrain dataset...")
    shutil.rmtree(FINAL_ROOT)

FINAL_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# RUGD
# ============================================================

print()
print("-" * 70)
print("RUGD")
print("-" * 70)

RUGD_FRAMES_FINAL.mkdir(parents=True, exist_ok=True)
RUGD_ANNOTATIONS_FINAL.mkdir(parents=True, exist_ok=True)

frames_copied = 0
annotations_copied = 0

frames_without_annotation = []
annotations_without_frame = []

# ------------------------------------------------------------
# COPY FRAMES
# ------------------------------------------------------------

for sequence in sorted(RUGD_FRAMES_SOURCE.iterdir()):

    if not sequence.is_dir():
        continue

    destination_sequence = RUGD_FRAMES_FINAL / sequence.name
    destination_sequence.mkdir(parents=True, exist_ok=True)

    annotation_sequence = RUGD_ANNOTATIONS_SOURCE / sequence.name

    for frame in sorted(sequence.iterdir()):

        if not frame.is_file():
            continue

        # Only consider actual image frames
        if frame.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        matching_annotation = annotation_sequence / frame.name

        if not matching_annotation.exists():
            frames_without_annotation.append(
                (sequence.name, frame.name)
            )
            continue

        shutil.copy2(
            frame,
            destination_sequence / frame.name
        )

        frames_copied += 1


# ------------------------------------------------------------
# COPY ONLY MATCHING ANNOTATIONS
# ------------------------------------------------------------

for sequence in sorted(RUGD_ANNOTATIONS_SOURCE.iterdir()):

    if not sequence.is_dir():
        continue

    frame_sequence = RUGD_FRAMES_SOURCE / sequence.name

    destination_sequence = (
        RUGD_ANNOTATIONS_FINAL / sequence.name
    )

    destination_sequence.mkdir(
        parents=True,
        exist_ok=True
    )

    for annotation in sorted(sequence.iterdir()):

        if not annotation.is_file():
            continue

        # ----------------------------------------------------
        # IMPORTANT:
        # Ignore metadata such as *_filelist
        # ----------------------------------------------------

        if annotation.suffix.lower() not in [
            ".jpg",
            ".jpeg",
            ".png"
        ]:
            continue

        matching_frame = frame_sequence / annotation.name

        # Only copy annotation if matching frame exists
        if not matching_frame.exists():
            annotations_without_frame.append(
                (sequence.name, annotation.name)
            )
            continue

        shutil.copy2(
            annotation,
            destination_sequence / annotation.name
        )

        annotations_copied += 1


# ============================================================
# OFFROAD-DATASET-II
# ============================================================

print()
print("-" * 70)
print("OFFROAD-DATASET-II")
print("-" * 70)

offroad_images_total = 0
offroad_masks_total = 0

for split in ["train", "valid", "test"]:

    print()
    print(split.upper())

    source_images = (
        OFFROAD_SOURCE
        / "images"
        / split
    )

    source_masks = (
        OFFROAD_SOURCE
        / "masks"
        / split
    )

    final_images = (
        OFFROAD_FINAL
        / "images"
        / split
    )

    final_masks = (
        OFFROAD_FINAL
        / "masks"
        / split
    )

    final_images.mkdir(
        parents=True,
        exist_ok=True
    )

    final_masks.mkdir(
        parents=True,
        exist_ok=True
    )

    image_count = 0
    mask_count = 0

    # --------------------------------------------------------
    # COPY IMAGES
    # --------------------------------------------------------

    for image in sorted(source_images.iterdir()):

        if not image.is_file():
            continue

        shutil.copy2(
            image,
            final_images / image.name
        )

        image_count += 1

    # --------------------------------------------------------
    # COPY MASKS
    # --------------------------------------------------------

    for mask in sorted(source_masks.iterdir()):

        if not mask.is_file():
            continue

        shutil.copy2(
            mask,
            final_masks / mask.name
        )

        mask_count += 1

    offroad_images_total += image_count
    offroad_masks_total += mask_count

    print(f"Images : {image_count}")
    print(f"Masks  : {mask_count}")


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 70)
print("V10.8.2 VALIDATION")
print("=" * 70)

print()
print(f"RUGD frames copied              : {frames_copied}")
print(f"RUGD annotations copied         : {annotations_copied}")
print(f"RUGD frames without annotation : {len(frames_without_annotation)}")
print(f"RUGD annotations without frame : {len(annotations_without_frame)}")

print()
print(f"Offroad images                  : {offroad_images_total}")
print(f"Offroad masks                   : {offroad_masks_total}")


# ============================================================
# WRITE REPORT
# ============================================================

report_path = (
    FINAL_ROOT /
    "V10.8.2_terrain_dataset_report.txt"
)

with open(report_path, "w", encoding="utf-8") as report:

    report.write("=" * 70 + "\n")
    report.write("V10.8.2 FINAL TERRAIN DATASET REPORT\n")
    report.write("=" * 70 + "\n\n")

    report.write("RUGD\n")
    report.write("-" * 70 + "\n")
    report.write(
        f"Frames copied              : {frames_copied}\n"
    )
    report.write(
        f"Annotations copied         : {annotations_copied}\n"
    )
    report.write(
        f"Frames without annotation  : "
        f"{len(frames_without_annotation)}\n"
    )
    report.write(
        f"Annotations without frame  : "
        f"{len(annotations_without_frame)}\n"
    )

    if frames_without_annotation:

        report.write("\nFrames without annotation:\n")

        for sequence, filename in frames_without_annotation:
            report.write(
                f"{sequence} / {filename}\n"
            )

    if annotations_without_frame:

        report.write("\nAnnotations without frame:\n")

        for sequence, filename in annotations_without_frame:
            report.write(
                f"{sequence} / {filename}\n"
            )

    report.write("\n")
    report.write("OFFROAD-DATASET-II\n")
    report.write("-" * 70 + "\n")
    report.write(
        f"Images : {offroad_images_total}\n"
    )
    report.write(
        f"Masks  : {offroad_masks_total}\n"
    )


# ============================================================
# FINAL STATUS
# ============================================================

print()
print(f"Final terrain dataset:")
print(FINAL_ROOT)

print()
print(f"Report:")
print(report_path)

print()

if (
    frames_copied == 7436
    and annotations_copied == 7436
    and len(frames_without_annotation) == 0
    and len(annotations_without_frame) == 0
    and offroad_images_total == 3462
    and offroad_masks_total == 3462
):

    print("V10.8.2 STATUS: PASS")

else:

    print("V10.8.2 STATUS: CHECK REQUIRED")

print("=" * 70)