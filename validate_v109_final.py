from pathlib import Path
from PIL import Image
import yaml


# ============================================================
# V10.9 FINAL MASTER VALIDATION
# ============================================================

V10 = Path(r"C:\UGV_Project\V10")

ROAD_ROOT = V10 / "datasets" / "final" / "Road_Obstacle"
TERRAIN_ROOT = V10 / "datasets" / "final" / "Terrain"

REPORT_DIR = V10 / "results"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT = REPORT_DIR / "V10.9_final_validation_report.txt"


print("=" * 70)
print("V10.9 FINAL MASTER VALIDATION")
print("=" * 70)

results = []
errors = []


def status(name, passed, detail=""):
    mark = "PASS" if passed else "FAIL"

    results.append((name, passed, detail))

    print()
    print(f"[{mark}] {name}")

    if detail:
        print(f"       {detail}")

    if not passed:
        errors.append(name)


# ============================================================
# ROAD OBSTACLE
# ============================================================

print()
print("-" * 70)
print("V10.8.3 — ROAD OBSTACLE INTEGRITY")
print("-" * 70)

road_expected = {
    "train": 4445,
    "valid": 818,
    "test": 468
}

road_classes = [
    "animal",
    "barrier",
    "fallen_tree",
    "pothole",
    "road_debris",
    "traffic_cone"
]

road_yaml = ROAD_ROOT / "data.yaml"

yaml_ok = False

if road_yaml.exists():

    try:

        with open(road_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        yaml_nc = data.get("nc")
        yaml_names = data.get("names")

        # Handle YAML names as either list or dictionary
        if isinstance(yaml_names, dict):
            ordered_names = [
                yaml_names[k]
                for k in sorted(
                    yaml_names,
                    key=lambda x: int(x)
                )
            ]
        else:
            ordered_names = list(yaml_names)

        yaml_ok = (
            yaml_nc == 6
            and ordered_names == road_classes
        )

    except Exception as e:

        print(f"YAML check error: {e}")
        yaml_ok = False


status(
    "Road Obstacle data.yaml",
    yaml_ok,
    "6 classes and class order verified"
)


road_total_images = 0
road_total_labels = 0
road_total_boxes = 0
road_empty = 0
road_invalid = 0


for split, expected_count in road_expected.items():

    images_dir = ROAD_ROOT / split / "images"
    labels_dir = ROAD_ROOT / split / "labels"

    images = sorted(
        p for p in images_dir.iterdir()
        if p.is_file()
    )

    labels = sorted(
        p for p in labels_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    )

    split_boxes = 0
    split_empty = 0
    split_invalid = 0

    for image in images:

        label_file = labels_dir / f"{image.stem}.txt"

        if not label_file.exists():

            split_invalid += 1
            continue

        try:

            with open(label_file, "r", encoding="utf-8") as f:

                lines = [
                    line.strip()
                    for line in f
                    if line.strip()
                ]

            if not lines:
                split_empty += 1

            for line in lines:

                parts = line.split()

                if len(parts) != 5:

                    split_invalid += 1
                    continue

                try:

                    class_id = int(parts[0])

                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                except ValueError:

                    split_invalid += 1
                    continue

                if not (0 <= class_id < 6):

                    split_invalid += 1
                    continue

                if not (
                    0 <= xc <= 1
                    and 0 <= yc <= 1
                    and 0 < w <= 1
                    and 0 < h <= 1
                ):

                    split_invalid += 1
                    continue

                split_boxes += 1

        except Exception:

            split_invalid += 1

    print()
    print(split.upper())
    print(f"Images       : {len(images)}")
    print(f"Labels       : {len(labels)}")
    print(f"Valid boxes  : {split_boxes}")
    print(f"Empty labels : {split_empty}")
    print(f"Invalid      : {split_invalid}")

    split_ok = (
        len(images) == expected_count
        and len(labels) == expected_count
        and split_invalid == 0
    )

    status(
        f"Road Obstacle {split}",
        split_ok,
        f"Expected {expected_count} images/labels"
    )

    road_total_images += len(images)
    road_total_labels += len(labels)
    road_total_boxes += split_boxes
    road_empty += split_empty
    road_invalid += split_invalid


status(
    "Road Obstacle total",
    (
        road_total_images == 5731
        and road_total_labels == 5731
        and road_invalid == 0
    ),
    (
        f"Images={road_total_images}, "
        f"Labels={road_total_labels}, "
        f"Boxes={road_total_boxes}, "
        f"Empty={road_empty}"
    )
)


# ============================================================
# RUGD
# ============================================================

print()
print("-" * 70)
print("V10.8.3 — RUGD INTEGRITY")
print("-" * 70)

RUGD_FRAMES = TERRAIN_ROOT / "RUGD" / "frames"
RUGD_ANNOTATIONS = TERRAIN_ROOT / "RUGD" / "annotations"

rug_frames = 0
rug_annotations = 0

rug_missing_annotations = []
rug_missing_frames = []
rug_corrupt_frames = []
rug_corrupt_annotations = []


for sequence in sorted(RUGD_FRAMES.iterdir()):

    if not sequence.is_dir():
        continue

    annotation_sequence = RUGD_ANNOTATIONS / sequence.name

    for frame in sequence.iterdir():

        if not frame.is_file():
            continue

        if frame.suffix.lower() not in [
            ".jpg",
            ".jpeg",
            ".png"
        ]:
            continue

        rug_frames += 1

        annotation = annotation_sequence / frame.name

        if not annotation.exists():

            rug_missing_annotations.append(
                f"{sequence.name}/{frame.name}"
            )

            continue

        try:

            with Image.open(frame) as img:
                img.verify()

        except Exception:

            rug_corrupt_frames.append(
                f"{sequence.name}/{frame.name}"
            )


for sequence in sorted(RUGD_ANNOTATIONS.iterdir()):

    if not sequence.is_dir():
        continue

    frame_sequence = RUGD_FRAMES / sequence.name

    for annotation in sequence.iterdir():

        if not annotation.is_file():
            continue

        if annotation.suffix.lower() not in [
            ".jpg",
            ".jpeg",
            ".png"
        ]:
            continue

        rug_annotations += 1

        frame = frame_sequence / annotation.name

        if not frame.exists():

            rug_missing_frames.append(
                f"{sequence.name}/{annotation.name}"
            )

            continue

        try:

            with Image.open(annotation) as img:
                img.verify()

        except Exception:

            rug_corrupt_annotations.append(
                f"{sequence.name}/{annotation.name}"
            )


print()
print(f"Frames              : {rug_frames}")
print(f"Annotations         : {rug_annotations}")
print(f"Missing annotations : {len(rug_missing_annotations)}")
print(f"Missing frames      : {len(rug_missing_frames)}")
print(f"Corrupt frames      : {len(rug_corrupt_frames)}")
print(f"Corrupt annotations : {len(rug_corrupt_annotations)}")


rug_ok = (
    rug_frames == 7436
    and rug_annotations == 7436
    and len(rug_missing_annotations) == 0
    and len(rug_missing_frames) == 0
    and len(rug_corrupt_frames) == 0
    and len(rug_corrupt_annotations) == 0
)


status(
    "RUGD final integrity",
    rug_ok,
    "7436 frames matched with 7436 annotations"
)


# ============================================================
# OFFROAD
# ============================================================

print()
print("-" * 70)
print("V10.8.3 — OFFROAD-DATASET-II INTEGRITY")
print("-" * 70)

OFFROAD = TERRAIN_ROOT / "Offroad"

offroad_expected = {
    "train": 3030,
    "valid": 288,
    "test": 144
}

offroad_total_images = 0
offroad_total_masks = 0
offroad_invalid = 0


for split, expected_count in offroad_expected.items():

    images_dir = OFFROAD / "images" / split
    masks_dir = OFFROAD / "masks" / split

    images = sorted(
        p for p in images_dir.iterdir()
        if p.is_file()
    )

    masks = sorted(
        p for p in masks_dir.iterdir()
        if p.is_file()
    )

    image_stems = {
        image.stem
        for image in images
    }

    # Masks use the pattern:
    #
    # image_name_mask.png
    #
    # Therefore:
    #
    # mask.stem = image_name_mask
    #
    # Remove "_mask" to obtain the image stem.

    mask_image_stems = {
        mask.stem[:-5]
        if mask.stem.endswith("_mask")
        else mask.stem
        for mask in masks
    }

    missing_masks = len(
        image_stems - mask_image_stems
    )

    missing_images = len(
        mask_image_stems - image_stems
    )

    corrupt_images = 0
    corrupt_masks = 0
    invalid_mask_values = 0

    # --------------------------------------------------------
    # IMAGE CHECK
    # --------------------------------------------------------

    for image in images:

        try:

            with Image.open(image) as img:
                img.verify()

        except Exception:

            corrupt_images += 1

    # --------------------------------------------------------
    # MASK CHECK
    # --------------------------------------------------------

    for mask in masks:

        try:

            with Image.open(mask) as img:

                if img.mode != "L":

                    invalid_mask_values += 1

                    continue

                img.load()

                values = set(img.getdata())

                if any(
                    value < 0 or value > 14
                    for value in values
                ):

                    invalid_mask_values += 1

        except Exception:

            corrupt_masks += 1


    split_invalid = (
        missing_masks
        + missing_images
        + corrupt_images
        + corrupt_masks
        + invalid_mask_values
    )

    offroad_invalid += split_invalid

    print()
    print(split.upper())
    print(f"Images              : {len(images)}")
    print(f"Masks               : {len(masks)}")
    print(f"Missing masks       : {missing_masks}")
    print(f"Missing images      : {missing_images}")
    print(f"Corrupt images      : {corrupt_images}")
    print(f"Corrupt masks       : {corrupt_masks}")
    print(f"Invalid mask values : {invalid_mask_values}")

    split_ok = (
        len(images) == expected_count
        and len(masks) == expected_count
        and split_invalid == 0
    )

    status(
        f"Offroad {split}",
        split_ok,
        f"Expected {expected_count} image/mask pairs"
    )

    offroad_total_images += len(images)
    offroad_total_masks += len(masks)


status(
    "Offroad total",
    (
        offroad_total_images == 3462
        and offroad_total_masks == 3462
        and offroad_invalid == 0
    ),
    (
        f"Images={offroad_total_images}, "
        f"Masks={offroad_total_masks}"
    )
)


# ============================================================
# FINAL DIRECTORY STRUCTURE
# ============================================================

print()
print("-" * 70)
print("FINAL DIRECTORY CHECK")
print("-" * 70)

required_paths = [

    ROAD_ROOT / "train" / "images",
    ROAD_ROOT / "train" / "labels",

    ROAD_ROOT / "valid" / "images",
    ROAD_ROOT / "valid" / "labels",

    ROAD_ROOT / "test" / "images",
    ROAD_ROOT / "test" / "labels",

    ROAD_ROOT / "data.yaml",

    TERRAIN_ROOT / "RUGD" / "frames",
    TERRAIN_ROOT / "RUGD" / "annotations",

    TERRAIN_ROOT / "Offroad" / "images" / "train",
    TERRAIN_ROOT / "Offroad" / "images" / "valid",
    TERRAIN_ROOT / "Offroad" / "images" / "test",

    TERRAIN_ROOT / "Offroad" / "masks" / "train",
    TERRAIN_ROOT / "Offroad" / "masks" / "valid",
    TERRAIN_ROOT / "Offroad" / "masks" / "test"
]

missing_paths = [
    str(path)
    for path in required_paths
    if not path.exists()
]

status(
    "Final directory structure",
    len(missing_paths) == 0,
    f"Missing paths: {len(missing_paths)}"
)


# ============================================================
# FINAL RESULT
# ============================================================

all_pass = all(
    passed
    for _, passed, _ in results
)


print()
print("=" * 70)

if all_pass:

    print("V10.9 STATUS: PASS")
    print()
    print("V10 DATASET PHASE: COMPLETE")
    print()
    print("READY FOR:")
    print("V11 — YOLO26 TRAINING")

else:

    print("V10.9 STATUS: FAIL")
    print()
    print("Problems detected:")

    for error in errors:
        print(f" - {error}")

print("=" * 70)


# ============================================================
# SAVE REPORT
# ============================================================

with open(REPORT, "w", encoding="utf-8") as f:

    f.write("=" * 70 + "\n")
    f.write("V10.9 FINAL MASTER VALIDATION REPORT\n")
    f.write("=" * 70 + "\n\n")

    for name, passed, detail in results:

        mark = "PASS" if passed else "FAIL"

        f.write(f"[{mark}] {name}\n")

        if detail:
            f.write(f"      {detail}\n")

        f.write("\n")

    f.write("=" * 70 + "\n")

    if all_pass:

        f.write("V10.9 STATUS: PASS\n")
        f.write("V10 DATASET PHASE: COMPLETE\n")
        f.write("NEXT PHASE: V11 YOLO26 TRAINING\n")

    else:

        f.write("V10.9 STATUS: FAIL\n")
        f.write("V10 DATASET PHASE: NOT COMPLETE\n")

    f.write("=" * 70 + "\n")


print()
print("Report saved to:")
print(REPORT)