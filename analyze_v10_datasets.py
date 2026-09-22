from pathlib import Path
from collections import Counter
import hashlib
import yaml
from PIL import Image


# ============================================================
# V10.1 DATASET QUALITY ANALYSIS
# UGV PROJECT
#
# READ-ONLY ANALYSIS
# This script does NOT delete, move, rename, or modify datasets.
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\UGV_Project")

V9_ROOT = PROJECT_ROOT / "V9"
V10_ROOT = PROJECT_ROOT / "V10"

RAW_ROOT = V9_ROOT / "datasets" / "raw"
PROCESSED_ROOT = V9_ROOT / "datasets" / "processed"

RESULTS_ROOT = V10_ROOT / "results"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

REPORT_FILE = RESULTS_ROOT / "V10_dataset_analysis.txt"


# ============================================================
# DATASET PATHS
# ============================================================

ROAD_ROOT = (
    RAW_ROOT
    / "Road_Obstacle"
    / "road-obstacle-detection.v1i.yolo26"
)

OFFROAD_ROOT = PROCESSED_ROOT / "Offroad"

RUGD_ROOT = RAW_ROOT / "RUGD"


# ============================================================
# GLOBAL REPORT
# ============================================================

report_lines = []


def log(text=""):
    print(text)
    report_lines.append(str(text))


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# HASH IMAGE
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


# ============================================================
# ROAD OBSTACLE ANALYSIS
# ============================================================

def analyze_road_obstacle():

    log("\n" + "=" * 70)
    log("ROAD OBSTACLE DATASET ANALYSIS")
    log("=" * 70)

    if not ROAD_ROOT.exists():

        log("ERROR: Road Obstacle dataset missing.")
        log(str(ROAD_ROOT))

        return False

    yaml_file = ROAD_ROOT / "data.yaml"

    if not yaml_file.exists():

        log("ERROR: data.yaml missing.")
        log(str(yaml_file))

        return False

    # --------------------------------------------------------
    # READ YAML
    # --------------------------------------------------------

    with open(yaml_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names", [])

    log("\nClasses:")

    for index, name in enumerate(names):
        log(f"  {index}: {name}")

    # --------------------------------------------------------
    # SPLITS
    # --------------------------------------------------------

    split_names = [
        "train",
        "valid",
        "test",
    ]

    total_images = 0
    total_labels = 0

    total_bbox = 0
    total_polygons = 0
    total_empty = 0
    total_invalid = 0

    class_counter = Counter()

    resolution_counter = Counter()

    duplicate_hashes = {}

    corrupt_images = []

    # --------------------------------------------------------
    # PROCESS SPLITS
    # --------------------------------------------------------

    for split in split_names:

        images_dir = ROAD_ROOT / split / "images"
        labels_dir = ROAD_ROOT / split / "labels"

        log(f"\n--- {split.upper()} ---")

        if not images_dir.exists():

            log("Images directory missing.")
            continue

        if not labels_dir.exists():

            log("Labels directory missing.")
            continue

        image_files = [
            p
            for p in images_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        label_files = [
            p
            for p in labels_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".txt"
        ]

        log(f"Images : {len(image_files)}")
        log(f"Labels : {len(label_files)}")

        total_images += len(image_files)
        total_labels += len(label_files)

        # ----------------------------------------------------
        # IMAGE QUALITY
        # ----------------------------------------------------

        for image_path in image_files:

            try:

                with Image.open(image_path) as img:

                    img.verify()

                with Image.open(image_path) as img:

                    resolution_counter[img.size] += 1

            except Exception:

                corrupt_images.append(str(image_path))

            # ------------------------------------------------
            # EXACT DUPLICATE DETECTION
            # ------------------------------------------------

            try:

                file_hash = sha256_file(image_path)

                duplicate_hashes.setdefault(
                    file_hash,
                    []
                ).append(str(image_path))

            except Exception:

                pass

        # ----------------------------------------------------
        # LABEL ANALYSIS
        # ----------------------------------------------------

        for label_path in label_files:

            try:

                text = label_path.read_text(
                    encoding="utf-8"
                ).strip()

            except Exception:

                total_invalid += 1
                continue

            # Empty label
            if not text:

                total_empty += 1
                continue

            lines = text.splitlines()

            for line in lines:

                parts = line.split()

                if len(parts) < 5:

                    total_invalid += 1
                    continue

                try:

                    class_id = int(float(parts[0]))

                except Exception:

                    total_invalid += 1
                    continue

                if class_id < 0 or class_id >= len(names):

                    total_invalid += 1
                    continue

                # ------------------------------------------------
                # STANDARD YOLO BBOX
                # ------------------------------------------------

                if len(parts) == 5:

                    try:

                        values = [
                            float(x)
                            for x in parts[1:5]
                        ]

                        if not all(
                            0.0 <= x <= 1.0
                            for x in values
                        ):

                            total_invalid += 1
                            continue

                        total_bbox += 1

                        class_counter[class_id] += 1

                    except Exception:

                        total_invalid += 1

                # ------------------------------------------------
                # POLYGON
                # ------------------------------------------------

                elif len(parts) > 5:

                    coordinate_values = parts[1:]

                    if len(coordinate_values) % 2 != 0:

                        total_invalid += 1
                        continue

                    try:

                        values = [
                            float(x)
                            for x in coordinate_values
                        ]

                        if not all(
                            0.0 <= x <= 1.0
                            for x in values
                        ):

                            total_invalid += 1
                            continue

                        total_polygons += 1

                        class_counter[class_id] += 1

                    except Exception:

                        total_invalid += 1

    # ========================================================
    # RESULTS
    # ========================================================

    log("\nROAD OBSTACLE SUMMARY")

    log(f"Total images       : {total_images}")
    log(f"Total labels       : {total_labels}")

    log(f"Bounding boxes     : {total_bbox}")
    log(f"Polygon lines      : {total_polygons}")
    log(f"Empty labels       : {total_empty}")
    log(f"Invalid lines      : {total_invalid}")
    log(f"Corrupt images     : {len(corrupt_images)}")

    # --------------------------------------------------------
    # CLASS COUNTS
    # --------------------------------------------------------

    log("\nClass distribution:")

    for class_id, name in enumerate(names):

        count = class_counter[class_id]

        log(
            f"  {class_id:2d} "
            f"{name:20s} : {count}"
        )

    # --------------------------------------------------------
    # RESOLUTIONS
    # --------------------------------------------------------

    log("\nImage resolutions:")

    for resolution, count in resolution_counter.most_common():

        log(
            f"  {resolution[0]}x{resolution[1]} : {count}"
        )

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    duplicate_groups = [
        paths
        for paths in duplicate_hashes.values()
        if len(paths) > 1
    ]

    log(
        f"\nExact duplicate image groups : "
        f"{len(duplicate_groups)}"
    )

    if duplicate_groups:

        for index, group in enumerate(
            duplicate_groups,
            start=1
        ):

            log(f"\nDuplicate group {index}:")

            for path in group:

                log(f"  {path}")

    # --------------------------------------------------------
    # CORRUPT IMAGES
    # --------------------------------------------------------

    if corrupt_images:

        log("\nCorrupt images:")

        for path in corrupt_images:

            log(f"  {path}")

    return True


# ============================================================
# OFFROAD DATASET ANALYSIS
# ============================================================

def analyze_offroad():

    log("\n" + "=" * 70)
    log("OFFROAD DATASET ANALYSIS")
    log("=" * 70)

    if not OFFROAD_ROOT.exists():

        log("ERROR: Offroad processed dataset missing.")
        log(str(OFFROAD_ROOT))

        return False

    images_root = OFFROAD_ROOT / "images"
    masks_root = OFFROAD_ROOT / "masks"

    split_names = [
        "train",
        "valid",
        "test",
    ]

    # --------------------------------------------------------
    # CLASS MAPPING
    # --------------------------------------------------------

    class_names = {
        0: "background",
        2: "dense-vegetation",
        3: "grass",
        4: "high_vegetation",
        5: "non_traversable_low_vegetation",
        6: "object",
        7: "obstacle",
        8: "path",
        9: "puddle",
        10: "rough_trail",
        11: "sky",
        12: "smooth_trali",
        13: "traversable_grass",
        14: "vegetation",
    }

    allowed_values = set(class_names.keys())

    total_images = 0
    total_masks = 0

    total_missing_masks = 0
    total_missing_images = 0
    total_invalid_masks = 0
    total_wrong_dimensions = 0

    pixel_counter = Counter()

    # --------------------------------------------------------
    # SPLITS
    # --------------------------------------------------------

    for split in split_names:

        image_dir = images_root / split
        mask_dir = masks_root / split

        log(f"\n--- {split.upper()} ---")

        if not image_dir.exists():

            log("Images directory missing.")
            continue

        if not mask_dir.exists():

            log("Masks directory missing.")
            continue

        image_files = [
            p
            for p in image_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        mask_files = [
            p
            for p in mask_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".png"
        ]

        log(f"Images : {len(image_files)}")
        log(f"Masks  : {len(mask_files)}")

        total_images += len(image_files)
        total_masks += len(mask_files)

        # ----------------------------------------------------
        # BUILD MASK MAP
        # ----------------------------------------------------

        mask_map = {}

        for mask_path in mask_files:

            name = mask_path.stem

            if name.endswith("_mask"):

                name = name[:-5]

            mask_map[name] = mask_path

        # ----------------------------------------------------
        # CHECK EACH IMAGE
        # ----------------------------------------------------

        for image_path in image_files:

            stem = image_path.stem

            mask_path = mask_map.get(stem)

            if mask_path is None:

                total_missing_masks += 1
                continue

            try:

                with Image.open(mask_path) as mask:

                    # ----------------------------------------
                    # MODE
                    # ----------------------------------------

                    if mask.mode != "L":

                        total_invalid_masks += 1
                        continue

                    # ----------------------------------------
                    # DIMENSION
                    # ----------------------------------------

                    if len(mask.size) != 2:

                        total_invalid_masks += 1
                        continue

                    # ----------------------------------------
                    # PIXEL VALUES
                    # ----------------------------------------

                    pixels = list(mask.getdata())

                    unique_values = set(pixels)

                    invalid_values = (
                        unique_values
                        - allowed_values
                    )

                    if invalid_values:

                        total_invalid_masks += 1
                        continue

                    pixel_counter.update(pixels)

            except Exception:

                total_invalid_masks += 1

        # ----------------------------------------------------
        # CHECK MASKS WITHOUT IMAGE
        # ----------------------------------------------------

        image_stems = {
            p.stem
            for p in image_files
        }

        for mask_stem in mask_map:

            if mask_stem not in image_stems:

                total_missing_images += 1

    # ========================================================
    # RESULTS
    # ========================================================

    log("\nOFFROAD SUMMARY")

    log(f"Total images          : {total_images}")
    log(f"Total masks           : {total_masks}")
    log(f"Missing masks         : {total_missing_masks}")
    log(f"Missing images        : {total_missing_images}")
    log(f"Invalid masks         : {total_invalid_masks}")
    log(f"Wrong dimensions      : {total_wrong_dimensions}")

    # --------------------------------------------------------
    # PIXEL DISTRIBUTION
    # --------------------------------------------------------

    total_pixels = sum(
        pixel_counter.values()
    )

    log("\nPixel / class distribution:")

    for class_id in sorted(class_names):

        count = pixel_counter[class_id]

        if total_pixels > 0:

            percentage = (
                count / total_pixels
            ) * 100

        else:

            percentage = 0.0

        log(
            f"  {class_id:2d} "
            f"{class_names[class_id]:35s} "
            f": {count:12d} "
            f"({percentage:6.2f}%)"
        )

    return True


# ============================================================
# RUGD DATASET ANALYSIS
# ============================================================

def analyze_rugd():

    log("\n" + "=" * 70)
    log("RUGD DATASET ANALYSIS")
    log("=" * 70)

    # ========================================================
    # CORRECT RUGD STRUCTURE
    #
    # C:\UGV_Project\V9\datasets\raw\RUGD\
    # ├── frames
    # └── annotations
    # ========================================================

    frames_root = RUGD_ROOT / "frames"
    annotations_root = RUGD_ROOT / "annotations"

    # --------------------------------------------------------
    # CHECK DIRECTORIES
    # --------------------------------------------------------

    if not frames_root.exists():

        log("ERROR: RUGD frames directory missing.")
        log(str(frames_root))

        return False

    if not annotations_root.exists():

        log("ERROR: RUGD annotations directory missing.")
        log(str(annotations_root))

        return False

    # --------------------------------------------------------
    # FIND SEQUENCES
    # --------------------------------------------------------

    frame_sequences = [
        p
        for p in frames_root.iterdir()
        if p.is_dir()
    ]

    annotation_sequences = [
        p
        for p in annotations_root.iterdir()
        if p.is_dir()
    ]

    log(
        f"\nFrame sequences      : "
        f"{len(frame_sequences)}"
    )

    log(
        f"Annotation sequences : "
        f"{len(annotation_sequences)}"
    )

    # --------------------------------------------------------
    # TOTAL COUNTS
    # --------------------------------------------------------

    total_frames = 0
    total_annotations = 0

    frames_without_annotation = []
    annotations_without_frame = []

    frame_resolution_counter = Counter()

    # ========================================================
    # PROCESS EACH FRAME SEQUENCE
    # ========================================================

    for sequence_dir in sorted(
        frame_sequences,
        key=lambda p: p.name
    ):

        sequence_name = sequence_dir.name

        annotation_dir = (
            annotations_root
            / sequence_name
        )

        frame_files = [
            p
            for p in sequence_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        annotation_files = []

        if annotation_dir.exists():

            annotation_files = [
                p
                for p in annotation_dir.iterdir()
                if p.is_file()
                and p.suffix.lower() == ".png"
            ]

        log(
            f"  {sequence_name:12s} "
            f"frames={len(frame_files):4d} "
            f"annotations={len(annotation_files):4d}"
        )

        total_frames += len(frame_files)
        total_annotations += len(annotation_files)

        # ----------------------------------------------------
        # RESOLUTION
        # ----------------------------------------------------

        for frame_path in frame_files:

            try:

                with Image.open(frame_path) as img:

                    frame_resolution_counter[
                        img.size
                    ] += 1

            except Exception:

                pass

        # ----------------------------------------------------
        # FRAME / ANNOTATION MATCHING
        # ----------------------------------------------------

        frame_stems = {
            p.stem
            for p in frame_files
        }

        annotation_stems = {
            p.stem
            for p in annotation_files
        }

        missing_annotations = (
            frame_stems
            - annotation_stems
        )

        missing_frames = (
            annotation_stems
            - frame_stems
        )

        for stem in sorted(
            missing_annotations
        ):

            frames_without_annotation.append(
                f"{sequence_name}/{stem}"
            )

        for stem in sorted(
            missing_frames
        ):

            annotations_without_frame.append(
                f"{sequence_name}/{stem}"
            )

    # ========================================================
    # ANNOTATION SEQUENCES WITHOUT FRAMES
    # ========================================================

    frame_sequence_names = {
        p.name
        for p in frame_sequences
    }

    for annotation_dir in annotation_sequences:

        if annotation_dir.name in frame_sequence_names:

            continue

        annotation_files = [
            p
            for p in annotation_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".png"
        ]

        for annotation_path in annotation_files:

            annotations_without_frame.append(
                f"{annotation_dir.name}/{annotation_path.stem}"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    log("\nRUGD SUMMARY")

    log(
        f"Total frames              : "
        f"{total_frames}"
    )

    log(
        f"Total annotations         : "
        f"{total_annotations}"
    )

    log(
        f"Frames without annotation : "
        f"{len(frames_without_annotation)}"
    )

    log(
        f"Annotations without frame : "
        f"{len(annotations_without_frame)}"
    )

    # --------------------------------------------------------
    # RESOLUTIONS
    # --------------------------------------------------------

    log("\nFrame resolutions:")

    for resolution, count in (
        frame_resolution_counter.most_common()
    ):

        log(
            f"  {resolution[0]}x{resolution[1]} "
            f": {count}"
        )

    # --------------------------------------------------------
    # MISSING ANNOTATIONS
    # --------------------------------------------------------

    if frames_without_annotation:

        log("\nFrames without annotations:")

        for item in frames_without_annotation:

            log(f"  {item}")

    # --------------------------------------------------------
    # MISSING FRAMES
    # --------------------------------------------------------

    if annotations_without_frame:

        log("\nAnnotations without frames:")

        for item in annotations_without_frame:

            log(f"  {item}")

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    log("=" * 70)
    log("V10.1 DATASET QUALITY ANALYSIS")
    log("UGV PROJECT")
    log("=" * 70)

    log("\nAnalysis mode: READ-ONLY")
    log(
        "No dataset files will be deleted, moved, "
        "renamed, or modified."
    )

    road_ok = analyze_road_obstacle()

    offroad_ok = analyze_offroad()

    rugd_ok = analyze_rugd()

    # ========================================================
    # FINAL STATUS
    # ========================================================

    log("\n" + "=" * 70)
    log("V10.1 ANALYSIS STATUS")
    log("=" * 70)

    log(
        f"Road Obstacle : "
        f"{'PASS' if road_ok else 'FAIL'}"
    )

    log(
        f"Offroad       : "
        f"{'PASS' if offroad_ok else 'FAIL'}"
    )

    log(
        f"RUGD          : "
        f"{'PASS' if rugd_ok else 'FAIL'}"
    )

    overall_ok = (
        road_ok
        and offroad_ok
        and rugd_ok
    )

    log("\nOverall V10.1 status:")

    if overall_ok:

        log("V10.1 DATASET QUALITY ANALYSIS: PASS")

    else:

        log("V10.1 DATASET QUALITY ANALYSIS: CHECK REQUIRED")

    # ========================================================
    # SAVE REPORT
    # ========================================================

    REPORT_FILE.write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    log(
        f"\nReport saved to:\n"
        f"{REPORT_FILE}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()