from pathlib import Path

RUGD = Path(r"C:\UGV_Project\V9\datasets\raw\RUGD")

frames_root = RUGD / "frames"
annotations_root = RUGD / "annotations"

print("=" * 70)
print("RUGD UNMATCHED ANNOTATION CHECK")
print("=" * 70)

unmatched_annotations = []

for sequence in sorted(annotations_root.iterdir()):

    if not sequence.is_dir():
        continue

    frame_sequence = frames_root / sequence.name

    for annotation in sequence.iterdir():

    if not annotation.is_file():
        continue

    matching_frame = (
        rug_frames_source
        / sequence.name
        / annotation.name
    )

    if not matching_frame.exists():
        continue

    shutil.copy2(
        annotation,
        destination_sequence / annotation.name
    )

    rug_annotation_count += 1