from pathlib import Path
from PIL import Image, ImageDraw

# ============================================================
# V10.7.1 - INSPECT ZERO/NEGATIVE ROAD OBSTACLE BOXES
# READ-ONLY
# ============================================================

PROJECT_ROOT = Path(r"C:\UGV_Project")

DATASET_ROOT = (
    PROJECT_ROOT
    / "V9"
    / "datasets"
    / "processed"
    / "Road_Obstacle"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "V10"
    / "results"
    / "bad_box_inspection"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SUSPICIOUS ANNOTATIONS FOUND IN V10.5
# ============================================================

BAD_FILES = [
    (
        "train",
        "161_jpeg_jpg.rf.7982c35c593848fd1641c9aca1dbd537.txt",
        3,
    ),
    (
        "train",
        "162_jpeg_jpg.rf.f92816a6923e18eb11cb931623ddba0e.txt",
        9,
    ),
    (
        "test",
        "img-415_jpg.rf.32ffdebc3339e86346bf8ac39b895415.txt",
        3,
    ),
]


# ============================================================
# HELPERS
# ============================================================

def find_image(split, label_name):
    """
    Find the image corresponding to a label file.
    """

    stem = Path(label_name).stem

    image_dir = DATASET_ROOT / split / "images"

    extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    ]

    for ext in extensions:
        candidate = image_dir / (stem + ext)

        if candidate.exists():
            return candidate

    # Fallback search
    matches = list(image_dir.glob(stem + ".*"))

    for match in matches:
        if match.suffix.lower() in extensions:
            return match

    return None


def parse_label_line(line):
    parts = line.strip().split()

    if len(parts) < 5:
        return None

    try:
        class_id = int(float(parts[0]))
        xc = float(parts[1])
        yc = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

        return class_id, xc, yc, w, h

    except Exception:
        return None


# ============================================================
# START
# ============================================================

print("=" * 70)
print("V10.7.1 BAD BOX INSPECTION")
print("UGV PROJECT")
print("=" * 70)

print()
print("MODE: READ-ONLY")
print("No source dataset files will be modified.")
print()

report = []

report.append("=" * 70)
report.append("V10.7.1 BAD BOX INSPECTION")
report.append("=" * 70)


for index, (split, label_name, target_line) in enumerate(BAD_FILES, start=1):

    print()
    print("=" * 70)
    print(f"SAMPLE {index}/3")
    print("=" * 70)

    label_path = DATASET_ROOT / split / "labels" / label_name

    print(f"Split       : {split}")
    print(f"Label       : {label_name}")
    print(f"Label path  : {label_path}")

    report.append("")
    report.append("=" * 70)
    report.append(f"SAMPLE {index}/3")
    report.append("=" * 70)
    report.append(f"Split: {split}")
    report.append(f"Label: {label_name}")

    if not label_path.exists():

        print("ERROR: Label file not found.")
        report.append("ERROR: Label file not found.")
        continue

    image_path = find_image(split, label_name)

    print(f"Image       : {image_path}")

    report.append(f"Image: {image_path}")

    if image_path is None:

        print("ERROR: Corresponding image not found.")
        report.append("ERROR: Corresponding image not found.")
        continue


    # --------------------------------------------------------
    # READ LABELS
    # --------------------------------------------------------

    lines = label_path.read_text(
        encoding="utf-8",
        errors="replace"
    ).splitlines()

    print()
    print("ANNOTATIONS:")

    report.append("")
    report.append("ANNOTATIONS:")

    parsed_annotations = []

    for line_number, line in enumerate(lines, start=1):

        parsed = parse_label_line(line)

        print(f"Line {line_number}: {line}")

        report.append(f"Line {line_number}: {line}")

        if parsed:
            parsed_annotations.append(
                (line_number, parsed)
            )


    # --------------------------------------------------------
    # OPEN IMAGE
    # --------------------------------------------------------

    try:
        image = Image.open(image_path).convert("RGB")

    except Exception as e:

        print(f"ERROR opening image: {e}")
        report.append(f"ERROR opening image: {e}")
        continue

    width, height = image.size

    print()
    print(f"Image size  : {width} x {height}")

    report.append(f"Image size: {width} x {height}")


    # --------------------------------------------------------
    # DRAW ALL VALID BOXES
    # --------------------------------------------------------

    preview = image.copy()

    draw = ImageDraw.Draw(preview)

    for line_number, (class_id, xc, yc, w, h) in parsed_annotations:

        x1 = int((xc - w / 2) * width)
        y1 = int((yc - h / 2) * height)

        x2 = int((xc + w / 2) * width)
        y2 = int((yc + h / 2) * height)

        # Clamp only for visualization.
        vx1 = max(0, min(width - 1, x1))
        vy1 = max(0, min(height - 1, y1))
        vx2 = max(0, min(width - 1, x2))
        vy2 = max(0, min(height - 1, y2))

        # Draw suspicious boxes differently.
        if line_number == target_line:

            draw.rectangle(
                [vx1, vy1, vx2, vy2],
                outline="red",
                width=4
            )

            draw.text(
                (vx1, max(0, vy1 - 15)),
                f"BAD LINE {line_number} CLASS {class_id}",
                fill="red"
            )

        else:

            draw.rectangle(
                [vx1, vy1, vx2, vy2],
                outline="yellow",
                width=2
            )

            draw.text(
                (vx1, max(0, vy1 - 15)),
                f"line {line_number} class {class_id}",
                fill="yellow"
            )


    # --------------------------------------------------------
    # SAVE INSPECTION COPY
    # --------------------------------------------------------

    output_name = (
        f"{index}_{split}_{Path(label_name).stem}_inspection.jpg"
    )

    output_path = OUTPUT_DIR / output_name

    preview.save(output_path, quality=95)

    print()
    print(f"Inspection image saved:")
    print(output_path)

    report.append(f"Inspection image: {output_path}")


# ============================================================
# FINAL
# ============================================================

report_path = OUTPUT_DIR / "inspection_report.txt"

report_path.write_text(
    "\n".join(report),
    encoding="utf-8"
)

print()
print("=" * 70)
print("V10.7.1 INSPECTION COMPLETE")
print("=" * 70)

print()
print("Inspection directory:")
print(OUTPUT_DIR)

print()
print("Report:")
print(report_path)

print()
print("IMPORTANT:")
print("No original dataset files were modified.")