from pathlib import Path
from collections import Counter
from PIL import Image, ImageStat


# ============================================================
# V10.4 IMAGE QUALITY ANALYSIS
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

REPORT_FILE = (
    RESULTS_ROOT
    / "V10_image_quality_report.txt"
)


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
# DATASET PATHS
# ============================================================

ROAD_ROOT = (
    RAW_ROOT
    / "Road_Obstacle"
    / "road-obstacle-detection.v1i.yolo26"
)

OFFROAD_IMAGES = (
    PROCESSED_ROOT
    / "Offroad"
    / "images"
)

RUGD_FRAMES = (
    RAW_ROOT
    / "RUGD"
    / "frames"
)


report = []


def log(text=""):
    print(text)
    report.append(str(text))


# ============================================================
# IMAGE QUALITY THRESHOLDS
# ============================================================

# These are screening thresholds, NOT automatic deletion rules.

VERY_DARK_BRIGHTNESS = 25
VERY_BRIGHT_BRIGHTNESS = 230

LOW_CONTRAST_STD = 15

SMALL_IMAGE_WIDTH = 320
SMALL_IMAGE_HEIGHT = 240

LARGE_IMAGE_WIDTH = 3000
LARGE_IMAGE_HEIGHT = 3000


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(path):

    try:

        with Image.open(path) as img:

            width, height = img.size

            # Convert to grayscale for brightness/contrast
            gray = img.convert("L")

            stat = ImageStat.Stat(gray)

            brightness = stat.mean[0]
            contrast = stat.stddev[0]

            return {
                "width": width,
                "height": height,
                "brightness": brightness,
                "contrast": contrast,
            }

    except Exception:

        return None


# ============================================================
# DATASET QUALITY ANALYSIS
# ============================================================

def analyze_dataset(
    dataset_name,
    root,
    recursive=True
):

    log("\n" + "=" * 70)
    log(f"{dataset_name} - IMAGE QUALITY ANALYSIS")
    log("=" * 70)

    if not root.exists():

        log(
            f"ERROR: Directory does not exist:\n"
            f"{root}"
        )

        return False

    if recursive:

        files = [
            p
            for p in root.rglob("*")
            if p.is_file()
            and p.suffix.lower()
            in IMAGE_EXTENSIONS
        ]

    else:

        files = [
            p
            for p in root.iterdir()
            if p.is_file()
            and p.suffix.lower()
            in IMAGE_EXTENSIONS
        ]

    log(
        f"Images found: {len(files)}"
    )

    if not files:

        log("No images found.")

        return False

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    resolution_counter = Counter()

    corrupt_images = []

    very_dark = []
    very_bright = []
    low_contrast = []

    small_images = []
    large_images = []

    brightness_values = []
    contrast_values = []

    total_pixels = 0

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for index, path in enumerate(files, start=1):

        result = analyze_image(path)

        if result is None:

            corrupt_images.append(path)
            continue

        width = result["width"]
        height = result["height"]

        brightness = result["brightness"]
        contrast = result["contrast"]

        resolution_counter[
            (width, height)
        ] += 1

        brightness_values.append(
            brightness
        )

        contrast_values.append(
            contrast
        )

        total_pixels += (
            width * height
        )

        # ----------------------------------------------------
        # Brightness
        # ----------------------------------------------------

        if brightness <= VERY_DARK_BRIGHTNESS:

            very_dark.append(path)

        if brightness >= VERY_BRIGHT_BRIGHTNESS:

            very_bright.append(path)

        # ----------------------------------------------------
        # Contrast
        # ----------------------------------------------------

        if contrast <= LOW_CONTRAST_STD:

            low_contrast.append(path)

        # ----------------------------------------------------
        # Resolution
        # ----------------------------------------------------

        if (
            width < SMALL_IMAGE_WIDTH
            or height < SMALL_IMAGE_HEIGHT
        ):

            small_images.append(path)

        if (
            width >= LARGE_IMAGE_WIDTH
            or height >= LARGE_IMAGE_HEIGHT
        ):

            large_images.append(path)

    # ========================================================
    # SUMMARY
    # ========================================================

    valid_images = (
        len(files)
        - len(corrupt_images)
    )

    log("\nSUMMARY")

    log(
        f"Total images      : "
        f"{len(files)}"
    )

    log(
        f"Valid images      : "
        f"{valid_images}"
    )

    log(
        f"Corrupt/unreadable: "
        f"{len(corrupt_images)}"
    )

    # --------------------------------------------------------
    # Percent helper
    # --------------------------------------------------------

    def percentage(count):

        if valid_images == 0:
            return 0.0

        return (
            count
            / valid_images
            * 100
        )

    # --------------------------------------------------------
    # Brightness
    # --------------------------------------------------------

    log("\nBRIGHTNESS")

    log(
        f"Very dark images   : "
        f"{len(very_dark)} "
        f"({percentage(len(very_dark)):.2f}%)"
    )

    log(
        f"Very bright images : "
        f"{len(very_bright)} "
        f"({percentage(len(very_bright)):.2f}%)"
    )

    if brightness_values:

        log(
            f"Minimum brightness : "
            f"{min(brightness_values):.2f}"
        )

        log(
            f"Maximum brightness : "
            f"{max(brightness_values):.2f}"
        )

        log(
            f"Average brightness : "
            f"{sum(brightness_values) / len(brightness_values):.2f}"
        )

    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    log("\nCONTRAST")

    log(
        f"Low contrast images : "
        f"{len(low_contrast)} "
        f"({percentage(len(low_contrast)):.2f}%)"
    )

    if contrast_values:

        log(
            f"Minimum contrast : "
            f"{min(contrast_values):.2f}"
        )

        log(
            f"Maximum contrast : "
            f"{max(contrast_values):.2f}"
        )

        log(
            f"Average contrast : "
            f"{sum(contrast_values) / len(contrast_values):.2f}"
        )

    # --------------------------------------------------------
    # Resolution
    # --------------------------------------------------------

    log("\nRESOLUTION")

    log(
        f"Small images (< "
        f"{SMALL_IMAGE_WIDTH}x"
        f"{SMALL_IMAGE_HEIGHT}) : "
        f"{len(small_images)} "
        f"({percentage(len(small_images)):.2f}%)"
    )

    log(
        f"Large images (>= "
        f"{LARGE_IMAGE_WIDTH}x"
        f"{LARGE_IMAGE_HEIGHT}) : "
        f"{len(large_images)} "
        f"({percentage(len(large_images)):.2f}%)"
    )

    log("\nTOP RESOLUTIONS")

    for (
        resolution,
        count
    ) in resolution_counter.most_common(20):

        width, height = resolution

        log(
            f"  {width:5d} x "
            f"{height:<5d} : "
            f"{count}"
        )

    # --------------------------------------------------------
    # Problem image examples
    # --------------------------------------------------------

    def show_examples(
        title,
        paths,
        limit=20
    ):

        log(f"\n{title}")

        if not paths:

            log("  None detected.")

            return

        for path in paths[:limit]:

            log(
                f"  {path}"
            )

        if len(paths) > limit:

            log(
                f"  ... and "
                f"{len(paths) - limit} more"
            )

    show_examples(
        "CORRUPT / UNREADABLE EXAMPLES",
        corrupt_images
    )

    show_examples(
        "VERY DARK IMAGE EXAMPLES",
        very_dark
    )

    show_examples(
        "VERY BRIGHT IMAGE EXAMPLES",
        very_bright
    )

    show_examples(
        "LOW CONTRAST IMAGE EXAMPLES",
        low_contrast
    )

    show_examples(
        "SMALL IMAGE EXAMPLES",
        small_images
    )

    show_examples(
        "LARGE IMAGE EXAMPLES",
        large_images
    )

    # --------------------------------------------------------
    # Screening interpretation
    # --------------------------------------------------------

    log("\nQUALITY SCREENING")

    if corrupt_images:

        log(
            "WARNING: Corrupt/unreadable "
            "images detected."
        )

    else:

        log(
            "PASS: No corrupt/unreadable "
            "images detected."
        )

    if very_dark:

        log(
            "INFO: Very dark images detected "
            "for review."
        )

    else:

        log(
            "PASS: No extremely dark images "
            "detected."
        )

    if very_bright:

        log(
            "INFO: Very bright images detected "
            "for review."
        )

    else:

        log(
            "PASS: No extremely bright images "
            "detected."
        )

    if low_contrast:

        log(
            "INFO: Low-contrast images detected "
            "for review."
        )

    else:

        log(
            "PASS: No low-contrast images "
            "detected under the screening threshold."
        )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    log("=" * 70)
    log("V10.4 IMAGE QUALITY ANALYSIS")
    log("UGV PROJECT")
    log("=" * 70)

    log(
        "\nAnalysis mode: READ-ONLY"
    )

    log(
        "No dataset files will be deleted, "
        "moved, renamed, or modified."
    )

    log(
        "\nScreening thresholds:"
    )

    log(
        f"Very dark brightness <= "
        f"{VERY_DARK_BRIGHTNESS}"
    )

    log(
        f"Very bright brightness >= "
        f"{VERY_BRIGHT_BRIGHTNESS}"
    )

    log(
        f"Low contrast <= "
        f"{LOW_CONTRAST_STD}"
    )

    log(
        f"Small image < "
        f"{SMALL_IMAGE_WIDTH}x"
        f"{SMALL_IMAGE_HEIGHT}"
    )

    log(
        f"Large image >= "
        f"{LARGE_IMAGE_WIDTH}x"
        f"{LARGE_IMAGE_HEIGHT}"
    )

    # ========================================================
    # ROAD OBSTACLE
    # ========================================================

    road_ok = analyze_dataset(
        "ROAD OBSTACLE",
        ROAD_ROOT
    )

    # ========================================================
    # OFFROAD
    # ========================================================

    offroad_ok = analyze_dataset(
        "OFFROAD IMAGES",
        OFFROAD_IMAGES
    )

    # ========================================================
    # RUGD
    # ========================================================

    rugd_ok = analyze_dataset(
        "RUGD FRAMES",
        RUGD_FRAMES
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    log("\n" + "=" * 70)
    log("V10.4 ANALYSIS STATUS")
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

    if (
        road_ok
        and offroad_ok
        and rugd_ok
    ):

        log(
            "\nV10.4 IMAGE QUALITY ANALYSIS: PASS"
        )

    else:

        log(
            "\nV10.4 IMAGE QUALITY ANALYSIS: "
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