from pathlib import Path
import shutil

# ============================================================
# V10.7.2 - CLEAN MALFORMED ROAD OBSTACLE BOXES
# ============================================================

PROJECT_ROOT = Path(r"C:\UGV_Project")

SOURCE = (
    PROJECT_ROOT
    / "V9"
    / "datasets"
    / "processed"
    / "Road_Obstacle"
)

DESTINATION = (
    PROJECT_ROOT
    / "V10"
    / "datasets"
    / "improved"
    / "Road_Obstacle"
)

BAD_LINES = {
    (
        "train",
        "161_jpeg_jpg.rf.7982c35c593848fd1641c9aca1dbd537.txt",
    ): {3},

    (
        "train",
        "162_jpeg_jpg.rf.f92816a6923e18eb11cb931623ddba0e.txt",
    ): {9},

    (
        "test",
        "img-415_jpg.rf.32ffdebc3339e86346bf8ac39b895415.txt",
    ): {3},
}


print("=" * 70)
print("V10.7.2 ROAD OBSTACLE DATASET CLEANING")
print("=" * 70)

print()
print("SOURCE:")
print(SOURCE)

print()
print("DESTINATION:")
print(DESTINATION)

print()
print("MODE: CREATE CLEAN COPY")
print("Original V9 dataset will NOT be modified.")

# ------------------------------------------------------------
# Check source
# ------------------------------------------------------------

if not SOURCE.exists():
    raise FileNotFoundError(
        f"Source dataset not found:\n{SOURCE}"
    )

# ------------------------------------------------------------
# Create destination
# ------------------------------------------------------------

if DESTINATION.exists():
    print()
    print("WARNING: Destination already exists.")
    print("Deleting previous V10.7 output...")

    shutil.rmtree(DESTINATION)

DESTINATION.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Copy complete dataset
# ------------------------------------------------------------

print()
print("Copying dataset...")

shutil.copytree(
    SOURCE,
    DESTINATION,
    dirs_exist_ok=True
)

# ------------------------------------------------------------
# Remove malformed annotation lines
# ------------------------------------------------------------

removed = 0

print()
print("REMOVING MALFORMED BOXES")
print("-" * 70)

for (split, filename), lines_to_remove in BAD_LINES.items():

    label_path = (
        DESTINATION
        / split
        / "labels"
        / filename
    )

    if not label_path.exists():
        print(f"WARNING: Missing label: {filename}")
        continue

    lines = label_path.read_text(
        encoding="utf-8",
        errors="replace"
    ).splitlines()

    new_lines = []

    for line_number, line in enumerate(lines, start=1):

        if line_number in lines_to_remove:

            print(
                f"REMOVED: {split}\\labels\\{filename} "
                f"line {line_number}"
            )

            print(f"         {line}")

            removed += 1

        else:
            new_lines.append(line)

    label_path.write_text(
        "\n".join(new_lines) + "\n",
        encoding="utf-8"
    )

# ------------------------------------------------------------
# Save cleaning report
# ------------------------------------------------------------

report_path = (
    DESTINATION
    / "V10.7_cleaning_report.txt"
)

report = f"""
V10.7.2 ROAD OBSTACLE CLEANING REPORT

Original dataset:
{SOURCE}

Improved dataset:
{DESTINATION}

Malformed annotation lines removed:
{removed}

Removed annotations:

1. train/labels/161_jpeg_jpg.rf.7982c35c593848fd1641c9aca1dbd537.txt
   Line 3
   Class: road_debris
   Problem: height = 0

2. train/labels/162_jpeg_jpg.rf.f92816a6923e18eb11cb931623ddba0e.txt
   Line 9
   Class: road_debris
   Problem: width = 0

3. test/labels/img-415_jpg.rf.32ffdebc3339e86346bf8ac39b895415.txt
   Line 3
   Class: pothole
   Problem: width = 0

Original V9 dataset was not modified.
"""

report_path.write_text(
    report.strip() + "\n",
    encoding="utf-8"
)

# ------------------------------------------------------------
# Final
# ------------------------------------------------------------

print()
print("=" * 70)
print("V10.7.2 CLEANING COMPLETE")
print("=" * 70)

print()
print(f"Malformed boxes removed : {removed}")

print()
print("Improved dataset:")
print(DESTINATION)

print()
print("Report:")
print(report_path)

print()
print("Original V9 dataset remains unchanged.")