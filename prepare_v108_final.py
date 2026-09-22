from pathlib import Path
import shutil

SOURCE = Path(r"C:\UGV_Project\V10\datasets\improved\Road_Obstacle")
FINAL = Path(r"C:\UGV_Project\V10\datasets\final\Road_Obstacle")

CLASSES = [
    "animal",
    "barrier",
    "fallen_tree",
    "pothole",
    "road_debris",
    "traffic_cone",
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def find_images(folder):
    return {
        p.stem: p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }


print("=" * 70)
print("V10.8.1 FINAL ROAD OBSTACLE DATASET PREPARATION")
print("=" * 70)

# Remove previous final dataset if it exists.
if FINAL.exists():
    print("\nRemoving previous V10 final dataset...")
    shutil.rmtree(FINAL)

FINAL.mkdir(parents=True, exist_ok=True)

total_images = 0
total_labels = 0
total_boxes = 0
total_empty_labels = 0
total_invalid = 0

for split in ["train", "valid", "test"]:

    print(f"\n--- {split.upper()} ---")

    source_images = SOURCE / split / "images"
    source_labels = SOURCE / split / "labels"

    final_images = FINAL / split / "images"
    final_labels = FINAL / split / "labels"

    final_images.mkdir(parents=True, exist_ok=True)
    final_labels.mkdir(parents=True, exist_ok=True)

    images = find_images(source_images)

    split_images = 0
    split_labels = 0
    split_boxes = 0
    split_empty = 0
    split_invalid = 0

    for stem, image_path in sorted(images.items()):

        label_path = source_labels / f"{stem}.txt"

        if not label_path.exists():
            print(f"WARNING: Missing label: {label_path}")
            continue

        destination_image = final_images / image_path.name
        destination_label = final_labels / label_path.name

        shutil.copy2(image_path, destination_image)
        shutil.copy2(label_path, destination_label)

        split_images += 1
        split_labels += 1

        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        valid_lines = 0

        for line in lines:

            line = line.strip()

            if not line:
                continue

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

            if not (0 <= class_id < len(CLASSES)):
                split_invalid += 1
                continue

            if not all(
                0 <= value <= 1
                for value in [xc, yc, w, h]
            ):
                split_invalid += 1
                continue

            if w <= 0 or h <= 0:
                split_invalid += 1
                continue

            valid_lines += 1

        split_boxes += valid_lines

        if valid_lines == 0:
            split_empty += 1

    print(f"Images       : {split_images}")
    print(f"Labels       : {split_labels}")
    print(f"Valid boxes  : {split_boxes}")
    print(f"Empty labels : {split_empty}")
    print(f"Invalid      : {split_invalid}")

    total_images += split_images
    total_labels += split_labels
    total_boxes += split_boxes
    total_empty_labels += split_empty
    total_invalid += split_invalid


# Create YOLO data.yaml
yaml_path = FINAL / "data.yaml"

yaml_content = f"""path: {FINAL.as_posix()}
train: train/images
val: valid/images
test: test/images

nc: {len(CLASSES)}

names:
"""

for index, class_name in enumerate(CLASSES):
    yaml_content += f"  {index}: {class_name}\n"

with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)


# Create preparation report
report_path = FINAL / "V10.8.1_final_dataset_report.txt"

report = f"""
V10.8.1 FINAL ROAD OBSTACLE DATASET REPORT
==========================================

Source:
{SOURCE}

Final dataset:
{FINAL}

Total images          : {total_images}
Total label files     : {total_labels}
Total valid boxes     : {total_boxes}
Empty label files     : {total_empty_labels}
Invalid annotations  : {total_invalid}

Classes:
0 animal
1 barrier
2 fallen_tree
3 pothole
4 road_debris
5 traffic_cone

YOLO data.yaml:
{yaml_path}

Dataset status:
{"PASS" if total_invalid == 0 else "FAIL"}
"""

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

print("\n" + "=" * 70)
print("V10.8.1 COMPLETE")
print("=" * 70)

print(f"Total images         : {total_images}")
print(f"Total labels         : {total_labels}")
print(f"Total valid boxes    : {total_boxes}")
print(f"Empty labels         : {total_empty_labels}")
print(f"Invalid annotations : {total_invalid}")

print(f"\nFinal dataset:")
print(FINAL)

print(f"\ndata.yaml:")
print(yaml_path)

print(f"\nReport:")
print(report_path)

if total_invalid == 0:
    print("\nV10.8.1 STATUS: PASS")
else:
    print("\nV10.8.1 STATUS: FAIL")

print("=" * 70)