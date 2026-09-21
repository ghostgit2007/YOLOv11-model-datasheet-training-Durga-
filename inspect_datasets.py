from pathlib import Path
from collections import Counter
from PIL import Image

ROOT = Path(r"C:\UGV_Project\V9\datasets\raw")
RESULTS = Path(r"C:\UGV_Project\V9\results")

RESULTS.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXTENSIONS = {".txt", ".xml", ".json"}

report = []

def add(text=""):
    print(text)
    report.append(text)


def inspect_images(folder):
    files = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    resolutions = Counter()
    corrupt = []

    for path in files:
        try:
            with Image.open(path) as img:
                img.verify()

            with Image.open(path) as img:
                resolutions[img.size] += 1

        except Exception:
            corrupt.append(str(path))

    return files, resolutions, corrupt


def inspect_labels(folder):
    files = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in LABEL_EXTENSIONS
    ]

    extensions = Counter(p.suffix.lower() for p in files)

    return files, extensions


# ==========================================================
# HEADER
# ==========================================================

add("=" * 70)
add("UGV V9 DATASET INSPECTION")
add("=" * 70)
add(f"Dataset root: {ROOT}")
add("")


# ==========================================================
# DATASETS
# ==========================================================

datasets = [
    "Road_Obstacle",
    "Offroad",
    "RUGD"
]


for dataset_name in datasets:

    dataset_path = ROOT / dataset_name

    add("")
    add("=" * 70)
    add(f"DATASET: {dataset_name}")
    add("=" * 70)

    if not dataset_path.exists():
        add("STATUS: DATASET FOLDER NOT FOUND")
        continue

    # ------------------------------------------------------
    # Images
    # ------------------------------------------------------

    image_files, resolutions, corrupt = inspect_images(dataset_path)

    add(f"Image files: {len(image_files)}")

    add("")
    add("Common image resolutions:")

    for resolution, count in resolutions.most_common(10):
        add(f"  {resolution[0]} x {resolution[1]} : {count}")

    # ------------------------------------------------------
    # Corrupt images
    # ------------------------------------------------------

    add("")
    add(f"Corrupt/unreadable images: {len(corrupt)}")

    if corrupt:
        add("First 20 corrupt files:")

        for path in corrupt[:20]:
            add(f"  {path}")

    # ------------------------------------------------------
    # Labels
    # ------------------------------------------------------

    label_files, label_extensions = inspect_labels(dataset_path)

    add("")
    add(f"Annotation/label files: {len(label_files)}")

    if label_extensions:
        add("Annotation formats:")

        for ext, count in label_extensions.items():
            add(f"  {ext} : {count}")

    # ------------------------------------------------------
    # Directory structure
    # ------------------------------------------------------

    add("")
    add("Top-level folders:")

    folders = [
        p for p in dataset_path.iterdir()
        if p.is_dir()
    ]

    for folder in folders:
        add(f"  {folder.name}")


# ==========================================================
# RUGD SPECIFIC CHECK
# ==========================================================

rugd_frames = ROOT / "RUGD" / "frames"
rugd_annotations = ROOT / "RUGD" / "annotations"

add("")
add("=" * 70)
add("RUGD FRAME / ANNOTATION CHECK")
add("=" * 70)

if rugd_frames.exists() and rugd_annotations.exists():

    frame_files = {
        p.relative_to(rugd_frames).as_posix()
        for p in rugd_frames.rglob("*.png")
    }

    annotation_files = {
        p.relative_to(rugd_annotations).as_posix()
        for p in rugd_annotations.rglob("*.png")
    }

    matching = frame_files & annotation_files
    missing_annotations = frame_files - annotation_files
    missing_frames = annotation_files - frame_files

    add(f"RUGD frames: {len(frame_files)}")
    add(f"RUGD annotations: {len(annotation_files)}")
    add(f"Matching pairs: {len(matching)}")
    add(f"Missing annotations: {len(missing_annotations)}")
    add(f"Missing frames: {len(missing_frames)}")

else:
    add("RUGD frames/annotations folders not found.")


# ==========================================================
# SAVE REPORT
# ==========================================================

report_path = RESULTS / "v9_dataset_report.txt"

report_path.write_text(
    "\n".join(report),
    encoding="utf-8"
)

add("")
add("=" * 70)
add("INSPECTION COMPLETE")
add("=" * 70)
add(f"Report saved to: {report_path}")