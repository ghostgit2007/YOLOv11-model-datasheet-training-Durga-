from pathlib import Path
import shutil

# ============================================================
# UGV V9 - Road Obstacle Dataset Converter
# Converts polygon annotations to YOLO bounding boxes.
# RAW DATASET IS NEVER MODIFIED.
# ============================================================

PROJECT_ROOT = Path(r"C:\UGV_Project\V9")

RAW = PROJECT_ROOT / "datasets" / "raw" / "Road_Obstacle" / "road-obstacle-detection.v1i.yolo26"
OUT = PROJECT_ROOT / "datasets" / "processed" / "Road_Obstacle"

CLASS_COUNT = 6

BOX_LINES = 0
POLYGON_LINES = 0
EMPTY_FILES = 0
CONVERTED_POLYGONS = 0
INVALID_LINES = 0
IMAGE_COUNT = 0


def polygon_to_box(class_id, coords):
    """Convert normalized polygon coordinates to YOLO bbox."""
    xs = coords[0::2]
    ys = coords[1::2]

    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)

    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin

    return (
        f"{class_id} "
        f"{x_center:.10f} "
        f"{y_center:.10f} "
        f"{width:.10f} "
        f"{height:.10f}"
    )


def process_label(src_label, dst_label):
    global BOX_LINES, POLYGON_LINES, EMPTY_FILES
    global CONVERTED_POLYGONS, INVALID_LINES

    lines = src_label.read_text(encoding="utf-8").splitlines()

    # Preserve empty label files.
    if not any(line.strip() for line in lines):
        dst_label.write_text("", encoding="utf-8")
        EMPTY_FILES += 1
        return

    output_lines = []

    for line_number, line in enumerate(lines, start=1):

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        try:
            class_id = int(parts[0])
            values = list(map(float, parts[1:]))
        except ValueError:
            print(f"INVALID: {src_label} line {line_number}")
            INVALID_LINES += 1
            continue

        # Validate class.
        if class_id < 0 or class_id >= CLASS_COUNT:
            print(
                f"INVALID CLASS: {src_label} "
                f"line {line_number}: {class_id}"
            )
            INVALID_LINES += 1
            continue

        # Standard YOLO detection:
        # class x_center y_center width height
        if len(values) == 4:

            if any(v < 0 or v > 1 for v in values):
                print(f"INVALID COORDINATES: {src_label} line {line_number}")
                INVALID_LINES += 1
                continue

            output_lines.append(
                f"{class_id} "
                f"{values[0]:.10f} "
                f"{values[1]:.10f} "
                f"{values[2]:.10f} "
                f"{values[3]:.10f}"
            )

            BOX_LINES += 1

        # Polygon:
        # class x1 y1 x2 y2 x3 y3 ...
        elif len(values) >= 6 and len(values) % 2 == 0:

            if any(v < 0 or v > 1 for v in values):
                print(f"INVALID POLYGON: {src_label} line {line_number}")
                INVALID_LINES += 1
                continue

            converted = polygon_to_box(class_id, values)
            output_lines.append(converted)

            POLYGON_LINES += 1
            CONVERTED_POLYGONS += 1

        else:
            print(
                f"INVALID FORMAT: {src_label} "
                f"line {line_number}: {len(values)} coordinates"
            )
            INVALID_LINES += 1

    dst_label.write_text(
        "\n".join(output_lines) + ("\n" if output_lines else ""),
        encoding="utf-8"
    )


def main():

    global IMAGE_COUNT

    print("=" * 65)
    print("UGV V9 - ROAD OBSTACLE DATASET PROCESSING")
    print("=" * 65)

    if not RAW.exists():
        raise FileNotFoundError(f"Raw dataset not found:\n{RAW}")

    print(f"\nRAW : {RAW}")
    print(f"OUT : {OUT}")

    # Prevent accidental mixing with an older processed result.
    if OUT.exists():
        print("\nProcessed dataset already exists.")
        print("Deleting ONLY the processed output...")
        shutil.rmtree(OUT)

    OUT.mkdir(parents=True, exist_ok=True)

    for split in ["train", "valid", "test"]:

        src_images = RAW / split / "images"
        src_labels = RAW / split / "labels"

        dst_images = OUT / split / "images"
        dst_labels = OUT / split / "labels"

        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing: {split}")

        # Copy images.
        images = [
            p for p in src_images.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]

        for image in images:
            shutil.copy2(image, dst_images / image.name)

        IMAGE_COUNT += len(images)

        # Process labels.
        labels = list(src_labels.glob("*.txt"))

        for label in labels:
            process_label(
                label,
                dst_labels / label.name
            )

        print(f"  Images : {len(images)}")
        print(f"  Labels : {len(labels)}")

    # Copy YAML and README files.
    for filename in [
        "data.yaml",
        "README.dataset.txt",
        "README.roboflow.txt"
    ]:
        src = RAW / filename
        if src.exists():
            shutil.copy2(src, OUT / filename)

    # Create report.
    report = OUT / "V9_conversion_report.txt"

    report.write_text(
        "\n".join([
            "UGV V9 ROAD OBSTACLE DATASET CONVERSION REPORT",
            "=" * 55,
            "",
            f"Source: {RAW}",
            f"Output: {OUT}",
            "",
            f"Images copied: {IMAGE_COUNT}",
            f"Bounding-box annotations preserved: {BOX_LINES}",
            f"Polygon annotations found: {POLYGON_LINES}",
            f"Polygon annotations converted: {CONVERTED_POLYGONS}",
            f"Empty label files preserved: {EMPTY_FILES}",
            f"Invalid annotations: {INVALID_LINES}",
            "",
            "Conversion completed.",
            "RAW dataset was not modified.",
        ]),
        encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("CONVERSION COMPLETE")
    print("=" * 65)

    print(f"\nImages copied              : {IMAGE_COUNT}")
    print(f"Bounding boxes preserved   : {BOX_LINES}")
    print(f"Polygons found             : {POLYGON_LINES}")
    print(f"Polygons converted         : {CONVERTED_POLYGONS}")
    print(f"Empty labels preserved     : {EMPTY_FILES}")
    print(f"Invalid annotations        : {INVALID_LINES}")

    print(f"\nProcessed dataset:")
    print(OUT)

    print(f"\nReport:")
    print(report)


if __name__ == "__main__":
    main()