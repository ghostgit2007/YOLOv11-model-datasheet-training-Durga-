from pathlib import Path
from collections import Counter
import yaml
from PIL import Image


# ============================================================
# V10.2 CLASS BALANCE ANALYSIS
# UGV PROJECT
#
# READ-ONLY
# No files are deleted, moved, renamed, or modified.
# ============================================================


PROJECT_ROOT = Path(r"C:\UGV_Project")

V9_ROOT = PROJECT_ROOT / "V9"
V10_ROOT = PROJECT_ROOT / "V10"

RAW_ROOT = V9_ROOT / "datasets" / "raw"
PROCESSED_ROOT = V9_ROOT / "datasets" / "processed"

RESULTS_ROOT = V10_ROOT / "results"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

REPORT_FILE = RESULTS_ROOT / "V10_class_balance_report.txt"


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


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


report = []


def log(text=""):
    print(text)
    report.append(str(text))


# ============================================================
# ROAD OBSTACLE CLASS BALANCE
# ============================================================

def analyze_road_obstacle():

    log("\n" + "=" * 70)
    log("V10.2 ROAD OBSTACLE CLASS BALANCE")
    log("=" * 70)

    yaml_file = ROAD_ROOT / "data.yaml"

    if not yaml_file.exists():

        log("ERROR: data.yaml not found.")
        return False

    with open(
        yaml_file,
        "r",
        encoding="utf-8"
    ) as f:

        data = yaml.safe_load(f)

    names = data.get("names", [])

    class_counter = Counter()

    total_images = 0
    total_images_with_objects = 0
    total_empty_images = 0
    total_objects = 0

    split_names = [
        "train",
        "valid",
        "test",
    ]

    for split in split_names:

        images_dir = ROAD_ROOT / split / "images"
        labels_dir = ROAD_ROOT / split / "labels"

        if not images_dir.exists():
            continue

        if not labels_dir.exists():
            continue

        image_files = [
            p
            for p in images_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        log(f"\n--- {split.upper()} ---")
        log(f"Images: {len(image_files)}")

        for image_path in image_files:

            total_images += 1

            label_path = (
                labels_dir
                / f"{image_path.stem}.txt"
            )

            if not label_path.exists():

                total_empty_images += 1
                continue

            try:

                text = label_path.read_text(
                    encoding="utf-8"
                ).strip()

            except Exception:

                total_empty_images += 1
                continue

            if not text:

                total_empty_images += 1
                continue

            image_has_object = False

            for line in text.splitlines():

                parts = line.split()

                if len(parts) < 5:
                    continue

                try:

                    class_id = int(float(parts[0]))

                except Exception:

                    continue

                if (
                    class_id < 0
                    or class_id >= len(names)
                ):
                    continue

                class_counter[class_id] += 1
                total_objects += 1
                image_has_object = True

            if image_has_object:

                total_images_with_objects += 1

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    log("\nROAD OBSTACLE SUMMARY")

    log(
        f"Total images             : "
        f"{total_images}"
    )

    log(
        f"Images with objects      : "
        f"{total_images_with_objects}"
    )

    log(
        f"Empty/background images  : "
        f"{total_empty_images}"
    )

    log(
        f"Total annotated objects  : "
        f"{total_objects}"
    )

    if total_images > 0:

        empty_percentage = (
            total_empty_images
            / total_images
        ) * 100

        object_image_percentage = (
            total_images_with_objects
            / total_images
        ) * 100

    else:

        empty_percentage = 0
        object_image_percentage = 0

    log(
        f"Object-image percentage  : "
        f"{object_image_percentage:.2f}%"
    )

    log(
        f"Empty-image percentage   : "
        f"{empty_percentage:.2f}%"
    )

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    log("\nOBJECT CLASS DISTRIBUTION")

    if total_objects > 0:

        for class_id, name in enumerate(names):

            count = class_counter[class_id]

            percentage = (
                count
                / total_objects
            ) * 100

            log(
                f"{class_id:2d} "
                f"{name:20s} "
                f": {count:6d} "
                f"({percentage:6.2f}%)"
            )

    # --------------------------------------------------------
    # BALANCE RATIO
    # --------------------------------------------------------

    nonzero_counts = [
        count
        for count in class_counter.values()
        if count > 0
    ]

    if nonzero_counts:

        largest = max(nonzero_counts)
        smallest = min(nonzero_counts)

        ratio = (
            largest / smallest
            if smallest > 0
            else 0
        )

        log(
            f"\nLargest/smallest class "
            f"ratio : {ratio:.2f}:1"
        )

    # --------------------------------------------------------
    # RANKING FOR ANALYSIS ONLY
    # --------------------------------------------------------

    log("\nCLASS COUNTS FROM HIGHEST TO LOWEST")

    for class_id, count in (
        class_counter.most_common()
    ):

        log(
            f"  {names[class_id]:20s} "
            f": {count}"
        )

    return True


# ============================================================
# OFFROAD TERRAIN CLASS BALANCE
# ============================================================

def analyze_offroad():

    log("\n" + "=" * 70)
    log("V10.2 OFFROAD TERRAIN CLASS BALANCE")
    log("=" * 70)

    images_root = OFFROAD_ROOT / "images"
    masks_root = OFFROAD_ROOT / "masks"

    if not images_root.exists():
        log("ERROR: Offroad images directory missing.")
        return False

    if not masks_root.exists():
        log("ERROR: Offroad masks directory missing.")
        return False

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

    pixel_counter = Counter()

    total_images = 0
    total_masks = 0

    split_names = [
        "train",
        "valid",
        "test",
    ]

    for split in split_names:

        image_dir = images_root / split
        mask_dir = masks_root / split

        if not image_dir.exists():
            continue

        if not mask_dir.exists():
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

        total_images += len(image_files)
        total_masks += len(mask_files)

        log(
            f"\n{split.upper()}: "
            f"{len(image_files)} images, "
            f"{len(mask_files)} masks"
        )

        for mask_path in mask_files:

            try:

                with Image.open(mask_path) as mask:

                    if mask.mode != "L":
                        continue

                    pixel_counter.update(
                        mask.getdata()
                    )

            except Exception:

                continue

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    total_pixels = sum(
        pixel_counter.values()
    )

    log("\nOFFROAD SUMMARY")

    log(
        f"Total images : "
        f"{total_images}"
    )

    log(
        f"Total masks  : "
        f"{total_masks}"
    )

    log(
        f"Total pixels : "
        f"{total_pixels}"
    )

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    log("\nTERRAIN PIXEL DISTRIBUTION")

    for class_id in sorted(class_names):

        count = pixel_counter[class_id]

        percentage = (
            count / total_pixels * 100
            if total_pixels > 0
            else 0
        )

        log(
            f"{class_id:2d} "
            f"{class_names[class_id]:35s} "
            f": {count:12d} "
            f"({percentage:6.2f}%)"
        )

    # --------------------------------------------------------
    # NONZERO CLASS BALANCE
    # --------------------------------------------------------

    nonzero_counts = [
        pixel_counter[class_id]
        for class_id in class_names
        if pixel_counter[class_id] > 0
    ]

    if nonzero_counts:

        largest = max(nonzero_counts)
        smallest = min(nonzero_counts)

        ratio = (
            largest / smallest
            if smallest > 0
            else 0
        )

        log(
            f"\nTerrain largest/smallest "
            f"ratio : {ratio:.2f}:1"
        )

    return True


# ============================================================
# RUGD SEQUENCE BALANCE
# ============================================================

def analyze_rugd():

    log("\n" + "=" * 70)
    log("V10.2 RUGD SEQUENCE BALANCE")
    log("=" * 70)

    frames_root = RUGD_ROOT / "frames"
    annotations_root = RUGD_ROOT / "annotations"

    if not frames_root.exists():

        log("ERROR: RUGD frames directory missing.")
        return False

    if not annotations_root.exists():

        log("ERROR: RUGD annotations directory missing.")
        return False

    sequence_counts = {}

    total_frames = 0
    total_annotations = 0

    frame_sequences = [
        p
        for p in frames_root.iterdir()
        if p.is_dir()
    ]

    for sequence_dir in sorted(
        frame_sequences,
        key=lambda p: p.name
    ):

        sequence_name = sequence_dir.name

        annotation_dir = (
            annotations_root
            / sequence_name
        )

        frames = [
            p
            for p in sequence_dir.iterdir()
            if p.is_file()
            and p.suffix.lower()
            in IMAGE_EXTENSIONS
        ]

        annotations = []

        if annotation_dir.exists():

            annotations = [
                p
                for p in annotation_dir.iterdir()
                if p.is_file()
                and p.suffix.lower() == ".png"
            ]

        frame_count = len(frames)
        annotation_count = len(annotations)

        sequence_counts[
            sequence_name
        ] = frame_count

        total_frames += frame_count
        total_annotations += annotation_count

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    log("\nRUGD SUMMARY")

    log(
        f"Total frames      : "
        f"{total_frames}"
    )

    log(
        f"Total annotations : "
        f"{total_annotations}"
    )

    log("\nSEQUENCE DISTRIBUTION")

    for sequence_name, count in sorted(
        sequence_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        percentage = (
            count / total_frames * 100
            if total_frames > 0
            else 0
        )

        log(
            f"  {sequence_name:12s} "
            f": {count:4d} "
            f"({percentage:6.2f}%)"
        )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    log("=" * 70)
    log("V10.2 CLASS BALANCE ANALYSIS")
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
    log("V10.2 ANALYSIS STATUS")
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

    if overall_ok:

        log(
            "\nV10.2 CLASS BALANCE ANALYSIS: PASS"
        )

    else:

        log(
            "\nV10.2 CLASS BALANCE ANALYSIS: "
            "CHECK REQUIRED"
        )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    REPORT_FILE.write_text(
        "\n".join(report),
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