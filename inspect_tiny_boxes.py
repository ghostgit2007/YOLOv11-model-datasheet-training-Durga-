from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(r"C:\UGV_Project\V10\datasets\improved\Road_Obstacle")
RESULTS = Path(r"C:\UGV_Project\V10\results\tiny_box_inspection")

RESULTS.mkdir(parents=True, exist_ok=True)

# Tiny-box threshold from V10.5
THRESHOLD = 0.01

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

tiny_boxes = []

for split in ["train", "valid", "test"]:
    label_dir = ROOT / split / "labels"
    image_dir = ROOT / split / "images"

    for label_file in sorted(label_dir.glob("*.txt")):

        with open(label_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line_number, line in enumerate(lines, start=1):

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            try:
                class_id = int(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                continue

            if w < THRESHOLD or h < THRESHOLD:

                image_file = None

                for ext in IMAGE_EXTENSIONS:
                    candidate = image_dir / (label_file.stem + ext)
                    if candidate.exists():
                        image_file = candidate
                        break

                tiny_boxes.append({
                    "split": split,
                    "label_file": label_file,
                    "line": line_number,
                    "class_id": class_id,
                    "xc": xc,
                    "yc": yc,
                    "w": w,
                    "h": h,
                    "image": image_file
                })


# Class names
class_names = {
    0: "animal",
    1: "barrier",
    2: "fallen_tree",
    3: "pothole",
    4: "road_debris",
    5: "traffic_cone"
}


print("=" * 70)
print("V10.7.4 TINY-BOX INSPECTION")
print("=" * 70)

print(f"\nTiny-box threshold : < {THRESHOLD * 100:.2f}%")
print(f"Tiny boxes found   : {len(tiny_boxes)}")

report_lines = []

for index, item in enumerate(tiny_boxes, start=1):

    class_name = class_names.get(
        item["class_id"],
        f"class_{item['class_id']}"
    )

    image_path = item["image"]

    print("\n" + "-" * 70)
    print(f"TINY BOX #{index}")
    print("-" * 70)

    print(f"Split       : {item['split']}")
    print(f"Image       : {image_path}")
    print(f"Label       : {item['label_file']}")
    print(f"Line        : {item['line']}")
    print(f"Class       : {item['class_id']} ({class_name})")
    print(f"Center      : ({item['xc']:.6f}, {item['yc']:.6f})")
    print(f"Width       : {item['w']:.6f} ({item['w'] * 100:.3f}%)")
    print(f"Height      : {item['h']:.6f} ({item['h'] * 100:.3f}%)")

    report_lines.append(
        f"""
TINY BOX #{index}
Split       : {item['split']}
Image       : {image_path}
Label       : {item['label_file']}
Line        : {item['line']}
Class       : {item['class_id']} ({class_name})
Center      : ({item['xc']:.6f}, {item['yc']:.6f})
Width       : {item['w']:.6f} ({item['w'] * 100:.3f}%)
Height      : {item['h']:.6f} ({item['h'] * 100:.3f}%)
"""
    )

    # Create visual inspection image
    if image_path and image_path.exists():

        try:
            image = Image.open(image_path).convert("RGB")

            draw = ImageDraw.Draw(image)

            img_w, img_h = image.size

            # Convert YOLO normalized coordinates
            box_w = item["w"] * img_w
            box_h = item["h"] * img_h

            center_x = item["xc"] * img_w
            center_y = item["yc"] * img_h

            x1 = center_x - box_w / 2
            y1 = center_y - box_h / 2
            x2 = center_x + box_w / 2
            y2 = center_y + box_h / 2

            # Make the tiny box visible
            draw.rectangle(
                [x1, y1, x2, y2],
                outline="red",
                width=3
            )

            # Draw a larger inspection region around it
            margin = max(50, int(max(box_w, box_h) * 10))

            crop_x1 = max(0, int(center_x - margin))
            crop_y1 = max(0, int(center_y - margin))
            crop_x2 = min(img_w, int(center_x + margin))
            crop_y2 = min(img_h, int(center_y + margin))

            crop = image.crop(
                (crop_x1, crop_y1, crop_x2, crop_y2)
            )

            output_name = (
    f"{index:02d}_{item['split']}_"
    f"{item['label_file'].stem}.jpg"
)

            output_path = RESULTS / output_name

            crop.save(output_path, quality=95)

            print(f"Inspection image: {output_path}")

        except Exception as e:
            print(f"Could not create inspection image: {e}")


report_path = RESULTS / "V10.7.4_tiny_box_report.txt"

with open(report_path, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("V10.7.4 TINY-BOX INSPECTION REPORT\n")
    f.write("=" * 70 + "\n")
    f.write(f"\nTiny-box threshold : < {THRESHOLD * 100:.2f}%\n")
    f.write(f"Tiny boxes found   : {len(tiny_boxes)}\n")
    f.write("\n")
    f.write("\n".join(report_lines))

print("\n" + "=" * 70)
print("V10.7.4 INSPECTION COMPLETE")
print("=" * 70)
print(f"Tiny boxes found : {len(tiny_boxes)}")
print(f"Report           : {report_path}")
print(f"Images           : {RESULTS}")
print("=" * 70)