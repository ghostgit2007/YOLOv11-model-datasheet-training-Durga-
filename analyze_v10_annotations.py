
from pathlib import Path
from collections import Counter
from PIL import Image
import yaml
import math


# ============================================================
# V10.5 ANNOTATION QUALITY ANALYSIS
# ============================================================

PROJECT_ROOT = Path(r"C:\UGV_Project")

V9_ROOT = PROJECT_ROOT / "V9"
V10_ROOT = PROJECT_ROOT / "V10"

RAW_ROOT = V9_ROOT / "datasets" / "raw"
PROCESSED_ROOT = V9_ROOT / "datasets" / "processed"

ROAD_ROOT = (
    RAW_ROOT
    / "Road_Obstacle"
    / "road-obstacle-detection.v1i.yolo26"
)

OFFROAD_ROOT = PROCESSED_ROOT / "Offroad"

RUGD_ROOT = RAW_ROOT / "RUGD"

RESULTS_ROOT = V10_ROOT / "results"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

REPORT_PATH = (
    RESULTS_ROOT
    / "V10_annotation_quality_report.txt"
)


# ============================================================
# CLASS DEFINITIONS
# ============================================================

ROAD_CLASS_NAMES = [
    "animal",
    "barrier",
    "fallen_tree",
    "pothole",
    "road_debris",
    "traffic_cone",
]


OFFROAD_CLASSES = {
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


# ============================================================
# REPORT STORAGE
# ============================================================

REPORT_LINES = []


def log(text=""):
    print(text)
    REPORT_LINES.append(text)


# ============================================================
# GENERAL HELPERS
# ============================================================

def is_image(path):
    return path.suffix.lower() in {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }


def relative_path(path, root):
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


# ============================================================
# ROAD OBSTACLE ANALYSIS
# ============================================================

def analyze_road_obstacle():

    log()
    log("=" * 70)
    log("V10.5 ROAD OBSTACLE ANNOTATION QUALITY")
    log("=" * 70)

    data_yaml = ROAD_ROOT / "data.yaml"

    if not data_yaml.exists():

        log(
            "ERROR: Road Obstacle data.yaml not found."
        )

        return False

    # --------------------------------------------------------
    # Read class names
    # --------------------------------------------------------

    try:

        with open(
            data_yaml,
            "r",
            encoding="utf-8"
        ) as f:

            data = yaml.safe_load(f)

        names = data.get(
            "names",
            ROAD_CLASS_NAMES
        )

        if isinstance(names, dict):

            class_names = [
                names[key]
                for key in sorted(
                    names.keys(),
                    key=lambda x: int(x)
                )
            ]

        else:

            class_names = list(names)

    except Exception:

        class_names = ROAD_CLASS_NAMES

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total_images = 0
    total_label_files = 0

    empty_labels = 0
    nonempty_labels = 0

    total_lines = 0
    valid_boxes = 0
    polygon_lines = 0

    invalid_class_ids = 0
    invalid_numeric = 0
    invalid_coordinates = 0

    zero_negative_boxes = 0
    tiny_boxes = 0
    large_boxes = 0

    class_counts = Counter()
    annotation_count_per_image = Counter()

    zero_examples = []
    tiny_examples = []

    # --------------------------------------------------------
    # Process train / valid / test
    # --------------------------------------------------------

    for split in [
        "train",
        "valid",
        "test"
    ]:

        image_dir = ROAD_ROOT / split / "images"
        label_dir = ROAD_ROOT / split / "labels"

        if image_dir.exists():

            images = [
                p
                for p in image_dir.iterdir()
                if p.is_file()
                and is_image(p)
            ]

        else:

            images = []

        log()
        log(
            f"--- {split.upper()} ---"
        )
        log(
            f"Images: {len(images)}"
        )

        total_images += len(images)

        for image_path in images:

            label_path = (
                label_dir
                / f"{image_path.stem}.txt"
            )

            if not label_path.exists():
                continue

            total_label_files += 1

            try:

                with open(
                    label_path,
                    "r",
                    encoding="utf-8",
                    errors="replace"
                ) as f:

                    lines = [
                        line.strip()
                        for line in f
                        if line.strip()
                    ]

            except Exception:

                continue

            # ------------------------------------------------
            # Empty label
            # ------------------------------------------------

            if not lines:

                empty_labels += 1

                annotation_count_per_image[0] += 1

                continue

            nonempty_labels += 1

            image_annotation_count = 0

            # ------------------------------------------------
            # Analyze lines
            # ------------------------------------------------

            for line_number, line in enumerate(
                lines,
                start=1
            ):

                total_lines += 1

                parts = line.split()

                # =================================================
                # YOLO BOUNDING BOX
                # =================================================

                if len(parts) == 5:

                    try:

                        class_id = int(parts[0])

                        xc = float(parts[1])
                        yc = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                    except Exception:

                        invalid_numeric += 1

                        continue

                    # ------------------------------------------------
                    # Class ID
                    # ------------------------------------------------

                    if (
                        class_id < 0
                        or class_id >= len(class_names)
                    ):

                        invalid_class_ids += 1

                        continue

                    values = [
                        xc,
                        yc,
                        width,
                        height
                    ]

                    # ------------------------------------------------
                    # Numeric validity
                    # ------------------------------------------------

                    if not all(
                        math.isfinite(v)
                        for v in values
                    ):

                        invalid_numeric += 1

                        continue

                    # ------------------------------------------------
                    # Coordinate validity
                    # ------------------------------------------------

                    if not all(
                        0.0 <= v <= 1.0
                        for v in values
                    ):

                        invalid_coordinates += 1

                        continue

                    # ------------------------------------------------
                    # Zero / negative dimensions
                    # ------------------------------------------------

                    if (
                        width <= 0
                        or height <= 0
                    ):

                        zero_negative_boxes += 1

                        if len(zero_examples) < 20:

                            zero_examples.append(
                                (
                                    relative_path(
                                        label_path,
                                        ROAD_ROOT
                                    ),
                                    line_number,
                                    line
                                )
                            )

                        continue

                    # ------------------------------------------------
                    # Valid box
                    # ------------------------------------------------

                    valid_boxes += 1

                    image_annotation_count += 1

                    class_counts[class_id] += 1

                    # ------------------------------------------------
                    # Tiny box
                    # ------------------------------------------------

                    if (
                        width < 0.01
                        or height < 0.01
                    ):

                        tiny_boxes += 1

                        if len(tiny_examples) < 20:

                            tiny_examples.append(
                                (
                                    relative_path(
                                        label_path,
                                        ROAD_ROOT
                                    ),
                                    line_number,
                                    line
                                )
                            )

                    # ------------------------------------------------
                    # Large box
                    # ------------------------------------------------

                    if (
                        width > 0.80
                        or height > 0.80
                    ):

                        large_boxes += 1

                # =================================================
                # YOLO POLYGON
                # =================================================

                else:

                    if (
                        len(parts) >= 7
                        and (len(parts) - 1) % 2 == 0
                    ):

                        try:

                            class_id = int(parts[0])

                            coordinates = [
                                float(value)
                                for value in parts[1:]
                            ]

                            if (
                                class_id < 0
                                or class_id >= len(class_names)
                            ):

                                invalid_class_ids += 1

                                continue

                            if not all(
                                math.isfinite(v)
                                for v in coordinates
                            ):

                                invalid_numeric += 1

                                continue

                            if not all(
                                0.0 <= v <= 1.0
                                for v in coordinates
                            ):

                                invalid_coordinates += 1

                                continue

                            polygon_lines += 1

                        except Exception:

                            invalid_numeric += 1

                    else:

                        invalid_numeric += 1

            annotation_count_per_image[
                image_annotation_count
            ] += 1

    # ============================================================
    # ROAD SUMMARY
    # ============================================================

    log()
    log("ROAD OBSTACLE SUMMARY")

    log(
        f"Total images              : "
        f"{total_images}"
    )

    log(
        f"Label files found         : "
        f"{total_label_files}"
    )

    log(
        f"Empty label files         : "
        f"{empty_labels}"
    )

    log(
        f"Non-empty label files     : "
        f"{nonempty_labels}"
    )

    log(
        f"Total annotation lines    : "
        f"{total_lines}"
    )

    log(
        f"Valid YOLO boxes          : "
        f"{valid_boxes}"
    )

    log(
        f"Polygon lines             : "
        f"{polygon_lines}"
    )

    log(
        f"Invalid class IDs         : "
        f"{invalid_class_ids}"
    )

    log(
        f"Invalid numeric format    : "
        f"{invalid_numeric}"
    )

    log(
        f"Invalid coordinates       : "
        f"{invalid_coordinates}"
    )

    log(
        f"Zero/negative box size    : "
        f"{zero_negative_boxes}"
    )

    log(
        f"Tiny boxes (<1% dimension): "
        f"{tiny_boxes}"
    )

    log(
        f"Large boxes (>80% dimension): "
        f"{large_boxes}"
    )

    # ============================================================
    # CLASS COUNTS
    # ============================================================

    log()
    log("CLASS ANNOTATION COUNTS")

    for class_id, class_name in enumerate(
        class_names
    ):

        log(
            f"{class_id:2d} "
            f"{class_name:<20} : "
            f"{class_counts[class_id]}"
        )

    # ============================================================
    # ANNOTATIONS PER IMAGE
    # ============================================================

    log()
    log("ANNOTATIONS PER IMAGE")

    for count in sorted(
        annotation_count_per_image
    ):

        log(
            f"{count:2d} annotations : "
            f"{annotation_count_per_image[count]} "
            f"images"
        )

    # ============================================================
    # ZERO BOX EXAMPLES
    # ============================================================

    if zero_examples:

        log()
        log("ZERO/NEGATIVE BOX EXAMPLES")

        for (
            path,
            line_number,
            line
        ) in zero_examples:

            log(
                f"{path} line "
                f"{line_number}: {line}"
            )

    # ============================================================
    # TINY BOX EXAMPLES
    # ============================================================

    if tiny_examples:

        log()
        log("TINY BOX EXAMPLES")

        for (
            path,
            line_number,
            line
        ) in tiny_examples:

            log(
                f"{path} line "
                f"{line_number}: {line}"
            )

    # ============================================================
    # SCREENING
    # ============================================================

    log()
    log("ROAD OBSTACLE ANNOTATION SCREENING")

    if invalid_class_ids == 0:

        log(
            "PASS: No invalid class IDs."
        )

    else:

        log(
            "WARNING: Invalid class IDs detected."
        )

    if invalid_coordinates == 0:

        log(
            "PASS: No coordinates outside 0-1."
        )

    else:

        log(
            "WARNING: Coordinates outside 0-1 detected."
        )

    if zero_negative_boxes == 0:

        log(
            "PASS: No zero/negative boxes."
        )

    else:

        log(
            "WARNING: Zero/negative boxes detected."
        )

    if invalid_numeric == 0:

        log(
            "PASS: No invalid annotation formats."
        )

    else:

        log(
            "WARNING: Invalid annotation formats detected."
        )

    if polygon_lines > 0:

        log(
            "INFO: Polygon annotations exist "
            "in raw dataset."
        )

        log(
            "INFO: They are handled by the "
            "V9 processed dataset."
        )

    # --------------------------------------------------------
    # Dataset is structurally readable.
    # The 3 invalid boxes are intentionally left untouched.
    # --------------------------------------------------------

    return True


# ============================================================
# OFFROAD ANALYSIS
# ============================================================

def analyze_offroad():

    log()
    log("=" * 70)
    log("V10.5 OFFROAD MASK ANNOTATION QUALITY")
    log("=" * 70)

    total_images = 0
    total_masks = 0

    missing_masks = 0
    missing_images = 0

    invalid_masks = 0
    wrong_dimensions = 0

    empty_masks = 0
    single_class_masks = 0

    class_presence = Counter()
    unexpected_values = Counter()

    expected_values = set(
        OFFROAD_CLASSES.keys()
    )

    # ========================================================
    # IMPORTANT:
    #
    # Actual structure:
    #
    # Offroad/
    # ├── images/
    # │   ├── train/
    # │   ├── valid/
    # │   └── test/
    # │
    # └── masks/
    #     ├── train/
    #     ├── valid/
    #     └── test/
    #
    # ========================================================

    for split in [
        "train",
        "valid",
        "test"
    ]:

        image_dir = (
            OFFROAD_ROOT
            / "images"
            / split
        )

        mask_dir = (
            OFFROAD_ROOT
            / "masks"
            / split
        )

        if not image_dir.exists():

            log(
                f"WARNING: Missing Offroad "
                f"image directory: {image_dir}"
            )

            continue

        if not mask_dir.exists():

            log(
                f"WARNING: Missing Offroad "
                f"mask directory: {mask_dir}"
            )

            continue

        image_files = [
            p
            for p in image_dir.iterdir()
            if p.is_file()
            and is_image(p)
        ]

        mask_files = [
            p
            for p in mask_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".png"
        ]

        log()
        log(
            f"--- {split.upper()} ---"
        )

        log(
            f"Images: {len(image_files)}"
        )

        log(
            f"Masks : {len(mask_files)}"
        )

        total_images += len(
            image_files
        )

        total_masks += len(
            mask_files
        )

        # ----------------------------------------------------
        # Image stems
        # ----------------------------------------------------

        image_stems = {
            p.stem
            for p in image_files
        }

        # ----------------------------------------------------
        # Mask stems
        #
        # abc_mask.png
        # becomes:
        # abc
        # ----------------------------------------------------

        mask_stems = set()

        for mask_path in mask_files:

            stem = mask_path.stem

            if stem.endswith("_mask"):

                stem = stem[:-5]

            mask_stems.add(stem)

        # ----------------------------------------------------
        # Missing files
        # ----------------------------------------------------

        missing_mask_stems = (
            image_stems - mask_stems
        )

        missing_image_stems = (
            mask_stems - image_stems
        )

        missing_masks += len(
            missing_mask_stems
        )

        missing_images += len(
            missing_image_stems
        )

        # ----------------------------------------------------
        # Analyze image/mask pairs
        # ----------------------------------------------------

        for image_path in image_files:

            stem = image_path.stem

            mask_path = (
                mask_dir
                / f"{stem}_mask.png"
            )

            if not mask_path.exists():

                mask_path = (
                    mask_dir
                    / f"{stem}.png"
                )

            if not mask_path.exists():

                continue

            try:

                with Image.open(
                    mask_path
                ) as mask:

                    mask.load()

                    # ------------------------------------------------
                    # Dimensions
                    # ------------------------------------------------

                    if (
                        mask.width != 432
                        or mask.height != 432
                    ):

                        wrong_dimensions += 1

                    # ------------------------------------------------
                    # Valid mask modes
                    # ------------------------------------------------

                    if mask.mode not in [
                        "L",
                        "P",
                        "I",
                        "I;16",
                        "RGB",
                        "RGBA",
                    ]:

                        invalid_masks += 1

                        continue

                    # ------------------------------------------------
                    # Extract pixel values
                    # ------------------------------------------------

                    if mask.mode in [
                        "L",
                        "P",
                        "I",
                        "I;16",
                    ]:

                        values = list(
                            mask.getdata()
                        )

                    else:

                        rgb_values = list(
                            mask
                            .convert("RGB")
                            .getdata()
                        )

                        values = [
                            pixel[0]
                            for pixel
                            in rgb_values
                        ]

                    # ------------------------------------------------
                    # Empty mask
                    # ------------------------------------------------

                    if not values:

                        empty_masks += 1

                        continue

                    unique_values = set(
                        values
                    )

                    # ------------------------------------------------
                    # Single class
                    # ------------------------------------------------

                    if len(
                        unique_values
                    ) == 1:

                        single_class_masks += 1

                    # ------------------------------------------------
                    # Unexpected classes
                    # ------------------------------------------------

                    bad_values = (
                        unique_values
                        - expected_values
                    )

                    if bad_values:

                        invalid_masks += 1

                        for value in bad_values:

                            unexpected_values[
                                value
                            ] += 1

                    # ------------------------------------------------
                    # Class presence
                    # ------------------------------------------------

                    for value in unique_values:

                        class_presence[
                            value
                        ] += 1

            except Exception as e:

                invalid_masks += 1

                log(
                    f"WARNING: Could not read mask "
                    f"{mask_path}: {e}"
                )

    # ============================================================
    # OFFROAD SUMMARY
    # ============================================================

    log()
    log("OFFROAD SUMMARY")

    log(
        f"Images                 : "
        f"{total_images}"
    )

    log(
        f"Masks                  : "
        f"{total_masks}"
    )

    log(
        f"Missing masks          : "
        f"{missing_masks}"
    )

    log(
        f"Missing images         : "
        f"{missing_images}"
    )

    log(
        f"Invalid masks          : "
        f"{invalid_masks}"
    )

    log(
        f"Wrong dimensions       : "
        f"{wrong_dimensions}"
    )

    log(
        f"Empty masks            : "
        f"{empty_masks}"
    )

    log(
        f"Single-class masks     : "
        f"{single_class_masks}"
    )

    # ============================================================
    # CLASS PRESENCE
    # ============================================================

    log()
    log("MASK CLASS PRESENCE")

    for class_id in sorted(
        OFFROAD_CLASSES
    ):

        class_name = (
            OFFROAD_CLASSES[
                class_id
            ]
        )

        percentage = (
            class_presence[class_id]
            / total_masks
            * 100
            if total_masks
            else 0
        )

        log(
            f"{class_id:2d} "
            f"{class_name:<34} : "
            f"{class_presence[class_id]:4d} "
            f"masks "
            f"({percentage:6.2f}%)"
        )

    # ============================================================
    # UNEXPECTED PIXELS
    # ============================================================

    if unexpected_values:

        log()
        log(
            "UNEXPECTED PIXEL VALUES"
        )

        for (
            value,
            count
        ) in sorted(
            unexpected_values.items()
        ):

            log(
                f"Pixel {value} : "
                f"{count} masks"
            )

    else:

        log()
        log(
            "No unexpected pixel values detected."
        )

    # ============================================================
    # OFFROAD SCREENING
    # ============================================================

    log()
    log(
        "OFFROAD MASK SCREENING"
    )

    if (
        total_images == 3462
        and total_masks == 3462
    ):

        log(
            "PASS: Expected 3462 "
            "image/mask pairs found."
        )

    elif (
        total_images > 0
        and total_images == total_masks
    ):

        log(
            "PASS: Image/mask count matches, "
            "but expected total differs from 3462."
        )

    else:

        log(
            "WARNING: Image/mask count mismatch."
        )

    if (
        missing_masks == 0
        and missing_images == 0
    ):

        log(
            "PASS: Image/mask matching "
            "is complete."
        )

    else:

        log(
            "WARNING: Image/mask matching "
            "is incomplete."
        )

    if invalid_masks == 0:

        log(
            "PASS: Mask structure is valid."
        )

    else:

        log(
            "WARNING: Invalid masks detected."
        )

    if wrong_dimensions == 0:

        log(
            "PASS: All masks have expected "
            "432x432 dimensions."
        )

    else:

        log(
            "WARNING: Wrong mask dimensions "
            "detected."
        )

    return (
        total_images == 3462
        and total_masks == 3462
        and missing_masks == 0
        and missing_images == 0
        and invalid_masks == 0
        and wrong_dimensions == 0
    )


# ============================================================
# RUGD ANALYSIS
# ============================================================

def analyze_rugd():

    log()
    log("=" * 70)
    log("V10.5 RUGD ANNOTATION QUALITY")
    log("=" * 70)

    frames_root = (
        RUGD_ROOT / "frames"
    )

    annotations_root = (
        RUGD_ROOT / "annotations"
    )

    if not frames_root.exists():

        log(
            "ERROR: RUGD frames directory "
            "not found."
        )

        return False

    if not annotations_root.exists():

        log(
            "ERROR: RUGD annotations directory "
            "not found."
        )

        return False

    # --------------------------------------------------------
    # Find frames
    # --------------------------------------------------------

    frame_files = [
        p
        for p in frames_root.rglob("*")
        if p.is_file()
        and is_image(p)
    ]

    # --------------------------------------------------------
    # Find annotations
    # --------------------------------------------------------

    annotation_files = [
        p
        for p in annotations_root.rglob("*")
        if p.is_file()
        and is_image(p)
    ]

    # --------------------------------------------------------
    # Relative-path maps
    # --------------------------------------------------------

    frame_map = {}

    for path in frame_files:

        try:

            rel = path.relative_to(
                frames_root
            )

            frame_map[
                str(rel).lower()
            ] = path

        except Exception:
            pass

    annotation_map = {}

    for path in annotation_files:

        try:

            rel = path.relative_to(
                annotations_root
            )

            annotation_map[
                str(rel).lower()
            ] = path

        except Exception:
            pass

    missing_annotations = sorted(
        set(frame_map)
        - set(annotation_map)
    )

    missing_frames = sorted(
        set(annotation_map)
        - set(frame_map)
    )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    unreadable_annotations = 0
    empty_annotations = 0

    mode_counter = Counter()
    dimension_counter = Counter()

    # --------------------------------------------------------
    # RUGD annotations are RGB.
    #
    # Therefore pixels are tuples:
    #
    # (R, G, B)
    #
    # We count RGB tuples directly.
    # --------------------------------------------------------

    pixel_counter = Counter()

    for key in frame_map:

        annotation_path = (
            annotation_map.get(key)
        )

        if annotation_path is None:
            continue

        try:

            with Image.open(
                annotation_path
            ) as mask:

                mask.load()

                mode_counter[
                    mask.mode
                ] += 1

                dimension_counter[
                    f"{mask.width}x{mask.height}"
                ] += 1

                if (
                    mask.width == 0
                    or mask.height == 0
                ):

                    empty_annotations += 1

                    continue

                # ------------------------------------------------
                # RGB / RGBA
                # ------------------------------------------------

                if mask.mode in [
                    "RGB",
                    "RGBA"
                ]:

                    rgb_mask = (
                        mask.convert("RGB")
                    )

                    pixel_counter.update(
                        rgb_mask.getdata()
                    )

                # ------------------------------------------------
                # Grayscale
                # ------------------------------------------------

                else:

                    gray_mask = (
                        mask.convert("L")
                    )

                    pixel_counter.update(
                        gray_mask.getdata()
                    )

        except Exception:

            unreadable_annotations += 1

    # ============================================================
    # RUGD SUMMARY
    # ============================================================

    log()
    log("RUGD SUMMARY")

    log(
        f"Frames                 : "
        f"{len(frame_files)}"
    )

    log(
        f"Annotations            : "
        f"{len(annotation_files)}"
    )

    log(
        f"Missing annotations    : "
        f"{len(missing_annotations)}"
    )

    log(
        f"Missing frames         : "
        f"{len(missing_frames)}"
    )

    log(
        f"Unreadable annotations : "
        f"{unreadable_annotations}"
    )

    log(
        f"Empty annotations      : "
        f"{empty_annotations}"
    )

    # ============================================================
    # MODES
    # ============================================================

    log()
    log(
        "RUGD ANNOTATION IMAGE MODES"
    )

    for mode, count in sorted(
        mode_counter.items()
    ):

        log(
            f"{mode:<8} : {count}"
        )

    # ============================================================
    # DIMENSIONS
    # ============================================================

    log()
    log(
        "RUGD ANNOTATION DIMENSIONS"
    )

    for (
        dimension,
        count
    ) in sorted(
        dimension_counter.items()
    ):

        log(
            f"{dimension:<8} : {count}"
        )

    # ============================================================
    # PIXEL VALUES
    # ============================================================

    log()
    log(
        "RUGD RAW MASK PIXEL VALUES"
    )

    total_pixels = sum(
        pixel_counter.values()
    )

    if total_pixels > 0:

        for (
            value,
            count
        ) in pixel_counter.most_common(30):

            percentage = (
                count
                / total_pixels
                * 100
            )

            if isinstance(
                value,
                tuple
            ):

                value_text = (
                    "("
                    + ",".join(
                        str(v)
                        for v in value
                    )
                    + ")"
                )

            else:

                value_text = str(value)

            log(
                f"{value_text:<18} : "
                f"{count:12d} pixels "
                f"({percentage:6.2f}%)"
            )

    else:

        log(
            "No pixel data available."
        )

    # ============================================================
    # RUGD SCREENING
    # ============================================================

    log()
    log(
        "RUGD ANNOTATION SCREENING"
    )

    if len(frame_files) == 7436:

        log(
            "PASS: Expected 7436 "
            "RUGD frames found."
        )

    else:

        log(
            f"WARNING: Expected 7436 frames, "
            f"found {len(frame_files)}."
        )

    if len(annotation_files) == 7436:

        log(
            "PASS: Expected 7436 "
            "annotations found."
        )

    else:

        log(
            f"WARNING: Expected 7436 annotations, "
            f"found {len(annotation_files)}."
        )

    if len(missing_annotations) == 0:

        log(
            "PASS: No missing "
            "RUGD annotations."
        )

    else:

        log(
            "WARNING: Missing RUGD "
            "annotations detected."
        )

    if len(missing_frames) == 0:

        log(
            "PASS: No annotations without "
            "corresponding frames."
        )

    else:

        log(
            "WARNING: Orphan RUGD "
            "annotations detected."
        )

    if unreadable_annotations == 0:

        log(
            "PASS: All RUGD annotations "
            "are readable."
        )

    else:

        log(
            "WARNING: Unreadable RUGD "
            "annotations detected."
        )

    return (
        len(frame_files) == 7436
        and len(annotation_files) == 7436
        and len(missing_annotations) == 0
        and len(missing_frames) == 0
        and unreadable_annotations == 0
        and empty_annotations == 0
    )


# ============================================================
# WRITE REPORT
# ============================================================

def write_report():

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(
                REPORT_LINES
            )
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "V10.5 ANNOTATION QUALITY ANALYSIS "
        "- CORRECTED V3"
    )
    print("UGV PROJECT")
    print("=" * 70)

    print()
    print("Analysis mode: READ-ONLY")

    print(
        "No dataset files will be deleted, "
        "moved, renamed, or modified."
    )

    # --------------------------------------------------------
    # Run analyses
    # --------------------------------------------------------

    road_ok = analyze_road_obstacle()

    offroad_ok = analyze_offroad()

    rugd_ok = analyze_rugd()

    # ========================================================
    # FINAL STATUS
    # ========================================================

    log()
    log("=" * 70)
    log(
        "V10.5 FINAL ANALYSIS STATUS"
    )
    log("=" * 70)

    log(
        "ROAD OBSTACLE : "
        + (
            "PASS"
            if road_ok
            else "WARNING"
        )
    )

    log(
        "OFFROAD       : "
        + (
            "PASS"
            if offroad_ok
            else "WARNING"
        )
    )

    log(
        "RUGD          : "
        + (
            "PASS"
            if rugd_ok
            else "WARNING"
        )
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Road Obstacle currently contains 3 zero/negative boxes.
    # We deliberately DO NOT modify them.
    #
    # Therefore V10.5 remains WARNING until those are inspected.
    # --------------------------------------------------------

    if (
        road_ok
        and offroad_ok
        and rugd_ok
    ):

        log()
        log(
            "V10.5 OVERALL STATUS: PASS"
        )

    else:

        log()
        log(
            "V10.5 OVERALL STATUS: WARNING"
        )

        log(
            "Dataset files were NOT modified."
        )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    write_report()

    print()
    print("=" * 70)
    print("REPORT SAVED")
    print("=" * 70)
    print(
        REPORT_PATH
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()