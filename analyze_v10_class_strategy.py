from pathlib import Path
from collections import Counter
from PIL import Image

# ============================================================
# V10.6 UGV CLASS STRATEGY ANALYSIS
# UGV PROJECT
# ============================================================

PROJECT_ROOT = Path(r"C:\UGV_Project")

RUGD_ROOT = PROJECT_ROOT / "V9" / "datasets" / "raw" / "RUGD"
ROAD_ROOT = PROJECT_ROOT / "V9" / "datasets" / "processed" / "Road_Obstacle"
OFFROAD_ROOT = PROJECT_ROOT / "V9" / "datasets" / "processed" / "Offroad"

RESULTS_DIR = PROJECT_ROOT / "V10" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REPORT = RESULTS_DIR / "V10_class_strategy_report.txt"


# ============================================================
# KNOWN ROAD OBSTACLE CLASSES
# ============================================================

ROAD_CLASSES = {
    0: "animal",
    1: "barrier",
    2: "fallen_tree",
    3: "pothole",
    4: "road_debris",
    5: "traffic_cone",
}


# ============================================================
# RUGD OFFICIAL CLASS LIST
# ============================================================

RUGD_CLASSES = [
    "dirt",
    "sand",
    "grass",
    "tree",
    "pole",
    "water",
    "sky",
    "vehicle",
    "object",
    "asphalt",
    "gravel",
    "mulch",
    "rock-bed",
    "log",
    "bicycle",
    "person",
    "fence",
    "bush",
    "sign",
    "rock",
    "bridge",
    "concrete",
    "picnic-table",
    "building",
]


# ============================================================
# OFFROAD CLASSES
# ============================================================

OFFROAD_CLASSES = {
    0: "background",
    2: "dense-vegetation",
    3: "grass",
    4: "high-vegetation",
    5: "non_traversable_low_vegetation",
    6: "object",
    7: "obstacle",
    8: "path",
    9: "puddle",
    10: "rough_trail",
    11: "sky",
    12: "smooth_trail",
    13: "traversable_grass",
    14: "vegetation",
}


# ============================================================
# PROPOSED UGV OBJECT CLASSES
# ============================================================

PROPOSED_OBJECT_CLASSES = [
    "person",
    "vehicle",
    "animal",
    "bicycle",
    "rock",
    "tree",
    "fallen_tree",
    "pothole",
    "road_debris",
    "barrier",
    "traffic_cone",
    "other_obstacle",
]


# ============================================================
# PROPOSED TERRAIN CLASSES
# ============================================================

PROPOSED_TERRAIN_CLASSES = [
    "grass",
    "dirt",
    "gravel",
    "sand",
    "rocky_ground",
    "mud",
    "water",
    "puddle",
    "vegetation",
    "traversable",
    "non_traversable",
    "unknown",
]


# ============================================================
# HELPERS
# ============================================================

def write(text=""):
    print(text)
    report_lines.append(text)


def find_rugd_annotation_colors():
    """
    Count the RGB annotation colors appearing in RUGD.

    This does NOT modify the dataset.
    """

    annotations_root = RUGD_ROOT / "annotations"

    if not annotations_root.exists():
        write("WARNING: RUGD annotation directory not found.")
        return Counter()

    counter = Counter()

    image_paths = list(annotations_root.rglob("*.png"))

    write(f"RUGD annotation files found: {len(image_paths)}")

    for i, path in enumerate(image_paths, start=1):

        try:
            with Image.open(path) as img:
                rgb = img.convert("RGB")

                counter.update(rgb.getdata())

        except Exception as e:
            write(f"WARNING: Could not read {path}: {e}")

        if i % 500 == 0:
            print(f"Processed RUGD masks: {i}/{len(image_paths)}")

    return counter


# ============================================================
# START
# ============================================================

report_lines = []

write("=" * 70)
write("V10.6 UGV CLASS STRATEGY ANALYSIS")
write("UGV PROJECT")
write("=" * 70)

write()
write("Analysis mode: READ-ONLY")
write("No dataset files will be modified.")

# ============================================================
# ROAD OBSTACLE
# ============================================================

write()
write("=" * 70)
write("1. ROAD OBSTACLE CLASSES")
write("=" * 70)

for class_id, name in ROAD_CLASSES.items():
    write(f"{class_id:2d} -> {name}")

write()
write("Purpose:")
write("Road Obstacle provides object-detection annotations.")
write("These classes are candidates for the YOLO26 object detector.")


# ============================================================
# RUGD
# ============================================================

write()
write("=" * 70)
write("2. RUGD CLASSES")
write("=" * 70)

for i, name in enumerate(RUGD_CLASSES):
    write(f"{i:2d} -> {name}")

write()
write("RUGD contains semantic segmentation masks.")
write("It should primarily support terrain/environment understanding.")


# ============================================================
# OFFROAD
# ============================================================

write()
write("=" * 70)
write("3. OFFROAD-DATASET-II CLASSES")
write("=" * 70)

for class_id, name in OFFROAD_CLASSES.items():
    write(f"{class_id:2d} -> {name}")

write()
write("Offroad-Dataset-II will support terrain/traversability segmentation.")


# ============================================================
# PROPOSED OBJECT STRATEGY
# ============================================================

write()
write("=" * 70)
write("4. PROPOSED UGV OBJECT-DETECTION CLASSES")
write("=" * 70)

for i, name in enumerate(PROPOSED_OBJECT_CLASSES):
    write(f"{i:2d} -> {name}")

write()
write("IMPORTANT:")
write("These are PROPOSED classes only.")
write("They are NOT yet the final YOLO26 training classes.")
write("Final mapping requires checking which source datasets actually")
write("contain reliable annotations for each class.")


# ============================================================
# OBJECT SOURCE MAPPING
# ============================================================

write()
write("=" * 70)
write("5. OBJECT CLASS SOURCE MAPPING")
write("=" * 70)

object_sources = {
    "person": "RUGD",
    "vehicle": "RUGD",
    "animal": "Road Obstacle",
    "bicycle": "RUGD",
    "rock": "RUGD",
    "tree": "RUGD",
    "fallen_tree": "Road Obstacle",
    "pothole": "Road Obstacle",
    "road_debris": "Road Obstacle",
    "barrier": "Road Obstacle",
    "traffic_cone": "Road Obstacle",
    "other_obstacle": "Offroad / RUGD",
}

for name, source in object_sources.items():
    write(f"{name:20s} -> {source}")


# ============================================================
# TERRAIN STRATEGY
# ============================================================

write()
write("=" * 70)
write("6. PROPOSED TERRAIN / TRAVERSABILITY CLASSES")
write("=" * 70)

for i, name in enumerate(PROPOSED_TERRAIN_CLASSES):
    write(f"{i:2d} -> {name}")

write()
write("Terrain segmentation remains separate from YOLO object detection.")


# ============================================================
# RUGD COLOR ANALYSIS
# ============================================================

write()
write("=" * 70)
write("7. RUGD ANNOTATION COLOR ANALYSIS")
write("=" * 70)

rug_colors = find_rugd_annotation_colors()

write()
write(f"Unique RGB colors found: {len(rug_colors)}")

write()
write("Top annotation colors:")

for color, count in rug_colors.most_common(30):
    percentage = 0.0

    total_pixels = sum(rug_colors.values())

    if total_pixels:
        percentage = count / total_pixels * 100

    write(
        f"{str(color):18s} : "
        f"{count:12d} pixels "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# FINAL STRATEGY
# ============================================================

write()
write("=" * 70)
write("8. V10.6 CURRENT STRATEGY")
write("=" * 70)

write()
write("OBJECT DETECTION")
write("----------------")
write("Road Obstacle -> primary object detection source")
write("RUGD         -> additional semantic information")
write("Offroad-II   -> obstacle/terrain semantic information")

write()
write("TERRAIN / TRAVERSABILITY")
write("-------------------------")
write("RUGD         -> off-road semantic segmentation")
write("Offroad-II   -> terrain/traversability segmentation")

write()
write("YOLO26")
write("------")
write("YOLO26 will be used for object detection in V11.")
write("Terrain segmentation will remain a separate perception component.")

write()
write("IMPORTANT:")
write("V10.6 does not train any model.")
write("V10.6 only defines and validates the class strategy.")


# ============================================================
# SAVE REPORT
# ============================================================

REPORT.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

write()
write("=" * 70)
write("V10.6 CLASS STRATEGY ANALYSIS COMPLETE")
write("=" * 70)

write()
write("REPORT SAVED:")
write(str(REPORT))