from pathlib import Path

ROOT = Path(r"C:\UGV_Project\V9\datasets\raw\Road_Obstacle")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Read class count from data.yaml if available
yaml_files = list(ROOT.rglob("data.yaml"))

NUM_CLASSES = None
CLASS_NAMES = []

if yaml_files:
    yaml_path = yaml_files[0]

    print(f"Using YAML: {yaml_path}")

    text = yaml_path.read_text(encoding="utf-8", errors="ignore")

    for line in text.splitlines():
        line = line.strip()

        if line.startswith("nc:"):
            try:
                NUM_CLASSES = int(line.split(":", 1)[1].strip())
            except:
                pass

        if line.startswith("names:"):
            names_text = line.split(":", 1)[1].strip()

            if names_text.startswith("[") and names_text.endswith("]"):
                CLASS_NAMES = [
                    x.strip().strip("'\"")
                    for x in names_text[1:-1].split(",")
                ]


images = [
    p for p in ROOT.rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]

labels = [
    p for p in ROOT.rglob("*.txt")
    if p.is_file()
    and p.name not in {"README.dataset.txt", "README.roboflow.txt"}
]

image_stems = {p.stem for p in images}
label_stems = {p.stem for p in labels}

missing_labels = []
missing_images = []

invalid_lines = []
invalid_classes = []
invalid_coordinates = []
empty_labels = []

total_objects = 0

# ---------------------------------------------------------
# Images without labels
# ---------------------------------------------------------

for image in images:
    if image.stem not in label_stems:
        missing_labels.append(str(image))

# ---------------------------------------------------------
# Labels without images
# ---------------------------------------------------------

for label in labels:
    if label.stem not in image_stems:
        missing_images.append(str(label))

# ---------------------------------------------------------
# Validate YOLO label contents
# ---------------------------------------------------------

for label in labels:

    try:
        lines = label.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines()
    except Exception:
        invalid_lines.append(f"{label} : cannot read file")
        continue

    valid_content = False

    for line_number, line in enumerate(lines, start=1):

        line = line.strip()

        if not line:
            continue

        valid_content = True

        parts = line.split()

        # YOLO detection format:
        # class x_center y_center width height

        if len(parts) != 5:
            invalid_lines.append(
                f"{label} : line {line_number} -> {line}"
            )
            continue

        try:
            class_id = int(float(parts[0]))

            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

        except ValueError:
            invalid_lines.append(
                f"{label} : line {line_number} -> {line}"
            )
            continue

        total_objects += 1

        # Class validation

        if class_id < 0:
            invalid_classes.append(
                f"{label} : line {line_number} -> class {class_id}"
            )

        if NUM_CLASSES is not None and class_id >= NUM_CLASSES:
            invalid_classes.append(
                f"{label} : line {line_number} -> class {class_id}"
            )

        # Coordinate validation

        values = [x, y, w, h]

        if any(v < 0 or v > 1 for v in values):
            invalid_coordinates.append(
                f"{label} : line {line_number} -> {line}"
            )

    if not valid_content:
        empty_labels.append(str(label))


# ---------------------------------------------------------
# Report
# ---------------------------------------------------------

print()
print("=" * 70)
print("ROAD OBSTACLE YOLO LABEL VALIDATION")
print("=" * 70)

print(f"Images:                 {len(images)}")
print(f"YOLO label files:       {len(labels)}")
print(f"Total objects:          {total_objects}")

if NUM_CLASSES is not None:
    print(f"Classes from YAML:      {NUM_CLASSES}")

if CLASS_NAMES:
    print(f"Class names:            {CLASS_NAMES}")

print()
print(f"Images without labels:  {len(missing_labels)}")
print(f"Labels without images:  {len(missing_images)}")
print(f"Invalid label lines:    {len(invalid_lines)}")
print(f"Invalid class IDs:      {len(invalid_classes)}")
print(f"Invalid coordinates:    {len(invalid_coordinates)}")
print(f"Empty label files:      {len(empty_labels)}")

print()
print("=" * 70)

if (
    len(missing_labels) == 0
    and len(missing_images) == 0
    and len(invalid_lines) == 0
    and len(invalid_classes) == 0
    and len(invalid_coordinates) == 0
):
    print("RESULT: YOLO DATASET VALIDATION PASSED")
else:
    print("RESULT: ISSUES FOUND — DO NOT MODIFY DATASET YET")

print("=" * 70)

# ---------------------------------------------------------
# Save report
# ---------------------------------------------------------

RESULTS = Path(r"C:\UGV_Project\V9\results")
RESULTS.mkdir(parents=True, exist_ok=True)

report = RESULTS / "road_obstacle_label_validation.txt"

with report.open("w", encoding="utf-8") as f:

    f.write("ROAD OBSTACLE YOLO LABEL VALIDATION\n")
    f.write("=" * 70 + "\n")
    f.write(f"Images: {len(images)}\n")
    f.write(f"YOLO labels: {len(labels)}\n")
    f.write(f"Total objects: {total_objects}\n")
    f.write(f"Images without labels: {len(missing_labels)}\n")
    f.write(f"Labels without images: {len(missing_images)}\n")
    f.write(f"Invalid lines: {len(invalid_lines)}\n")
    f.write(f"Invalid classes: {len(invalid_classes)}\n")
    f.write(f"Invalid coordinates: {len(invalid_coordinates)}\n")
    f.write(f"Empty labels: {len(empty_labels)}\n")

    if invalid_lines:
        f.write("\nINVALID LINES\n")
        for item in invalid_lines:
            f.write(item + "\n")

    if invalid_classes:
        f.write("\nINVALID CLASSES\n")
        for item in invalid_classes:
            f.write(item + "\n")

    if invalid_coordinates:
        f.write("\nINVALID COORDINATES\n")
        for item in invalid_coordinates:
            f.write(item + "\n")

print()
print(f"Report saved to: {report}")