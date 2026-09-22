from pathlib import Path

ROOT = Path(r"C:\UGV_Project\V10\datasets\improved\Road_Obstacle")

splits = ["train", "valid", "test"]

total_images = 0
total_labels = 0
total_boxes = 0
zero_negative = 0
invalid = 0

print("=" * 70)
print("V10.7.3 CLEANED ROAD OBSTACLE VERIFICATION")
print("=" * 70)

for split in splits:

    image_dir = ROOT / split / "images"
    label_dir = ROOT / split / "labels"

    images = list(image_dir.glob("*"))
    labels = list(label_dir.glob("*.txt"))

    image_count = len([
        p for p in images
        if p.suffix.lower() in
        [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    ])

    label_count = len(labels)

    total_images += image_count
    total_labels += label_count

    print()
    print(f"--- {split.upper()} ---")
    print(f"Images : {image_count}")
    print(f"Labels : {label_count}")

    for label_path in labels:

        lines = label_path.read_text(
            encoding="utf-8",
            errors="replace"
        ).splitlines()

        for line_number, line in enumerate(lines, start=1):

            if not line.strip():
                continue

            parts = line.split()

            if len(parts) < 5:
                invalid += 1
                continue

            try:
                class_id = int(float(parts[0]))
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

            except ValueError:
                invalid += 1
                continue

            total_boxes += 1

            if w <= 0 or h <= 0:
                zero_negative += 1

                print(
                    f"BAD BOX: {split}\\labels\\"
                    f"{label_path.name} line {line_number}"
                )

            if not (
                0 <= xc <= 1
                and 0 <= yc <= 1
                and 0 <= w <= 1
                and 0 <= h <= 1
            ):
                invalid += 1

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Total images          : {total_images}")
print(f"Total label files     : {total_labels}")
print(f"Total valid boxes     : {total_boxes}")
print(f"Zero/negative boxes   : {zero_negative}")
print(f"Invalid annotations   : {invalid}")

print()
print("=" * 70)

if (
    total_images == 5731
    and total_labels == 5731
    and zero_negative == 0
    and invalid == 0
):
    print("V10.7.3 STATUS: PASS")
else:
    print("V10.7.3 STATUS: WARNING")

print("=" * 70)