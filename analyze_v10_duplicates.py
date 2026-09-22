from pathlib import Path
from collections import defaultdict
from PIL import Image
import hashlib
import math


# ============================================================
# V10.3 DUPLICATE & SIMILAR IMAGE ANALYSIS
# UGV PROJECT
#
# READ-ONLY
# No files are deleted, moved, renamed, or modified.
# ============================================================


PROJECT_ROOT = Path(r"C:\UGV_Project")

V9_ROOT = PROJECT_ROOT / "V9"
V10_ROOT = PROJECT_ROOT / "V10"

RAW_ROOT = V9_ROOT / "datasets" / "raw"
PROCESSED_ROOT = V9_ROOT / "datasets" / "processed"

RESULTS_ROOT = V10_ROOT / "results"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

REPORT_FILE = (
    RESULTS_ROOT
    / "V10_duplicate_similarity_report.txt"
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# DATASET PATHS
# ============================================================

ROAD_ROOT = (
    RAW_ROOT
    / "Road_Obstacle"
    / "road-obstacle-detection.v1i.yolo26"
)

OFFROAD_ROOT = PROCESSED_ROOT / "Offroad"

RUGD_ROOT = RAW_ROOT / "RUGD"


report = []


def log(text=""):
    print(text)
    report.append(str(text))


# ============================================================
# FILE HASH
# ============================================================

def sha256_file(path):

    hasher = hashlib.sha256()

    try:

        with open(path, "rb") as f:

            while True:

                chunk = f.read(1024 * 1024)

                if not chunk:
                    break

                hasher.update(chunk)

        return hasher.hexdigest()

    except Exception:

        return None


# ============================================================
# EXACT DUPLICATE ANALYSIS
# ============================================================

def analyze_exact_duplicates(
    dataset_name,
    root,
    recursive=True
):

    log("\n" + "=" * 70)
    log(
        f"{dataset_name} - EXACT DUPLICATE ANALYSIS"
    )
    log("=" * 70)

    if not root.exists():

        log(f"ERROR: Directory missing: {root}")
        return False

    if recursive:

        files = [
            p
            for p in root.rglob("*")
            if p.is_file()
            and p.suffix.lower()
            in IMAGE_EXTENSIONS
        ]

    else:

        files = [
            p
            for p in root.iterdir()
            if p.is_file()
            and p.suffix.lower()
            in IMAGE_EXTENSIONS
        ]

    log(
        f"Images scanned: {len(files)}"
    )

    hashes = defaultdict(list)

    processed = 0
    failed = 0

    for image_path in files:

        digest = sha256_file(image_path)

        if digest is None:

            failed += 1
            continue

        hashes[digest].append(
            image_path
        )

        processed += 1

    duplicate_groups = {
        digest: paths
        for digest, paths in hashes.items()
        if len(paths) > 1
    }

    duplicate_files = sum(
        len(paths)
        for paths in duplicate_groups.values()
    )

    extra_duplicate_files = sum(
        len(paths) - 1
        for paths in duplicate_groups.values()
    )

    log(
        f"Successfully hashed : "
        f"{processed}"
    )

    log(
        f"Hash failures       : "
        f"{failed}"
    )

    log(
        f"Duplicate groups     : "
        f"{len(duplicate_groups)}"
    )

    log(
        f"Files in duplicate groups : "
        f"{duplicate_files}"
    )

    log(
        f"Extra duplicate copies    : "
        f"{extra_duplicate_files}"
    )

    if duplicate_groups:

        log("\nDUPLICATE GROUPS")

        group_number = 1

        for digest, paths in duplicate_groups.items():

            log(
                f"\nGroup {group_number}"
            )

            for path in paths:

                log(
                    f"  {path}"
                )

            group_number += 1

    else:

        log(
            "\nNo exact duplicate image groups found."
        )

    return True


# ============================================================
# IMAGE RESIZE FOR PERCEPTUAL COMPARISON
# ============================================================

def perceptual_signature(
    image_path,
    size=32
):

    try:

        with Image.open(image_path) as img:

            img = img.convert("L")
            img = img.resize(
                (size, size)
            )

            pixels = list(
                img.getdata()
            )

            if not pixels:
                return None

            average = (
                sum(pixels)
                / len(pixels)
            )

            bits = [
                1 if p >= average else 0
                for p in pixels
            ]

            return bits

    except Exception:

        return None


# ============================================================
# HAMMING DISTANCE
# ============================================================

def hamming_distance(
    signature_a,
    signature_b
):

    if (
        signature_a is None
        or signature_b is None
    ):

        return None

    if len(signature_a) != len(signature_b):

        return None

    return sum(
        a != b
        for a, b in zip(
            signature_a,
            signature_b
        )
    )


# ============================================================
# PERCEPTUAL HASH ANALYSIS
# ============================================================

def analyze_perceptual_similarity(
    dataset_name,
    root,
    threshold=3,
    max_comparisons=150000
):

    log("\n" + "=" * 70)
    log(
        f"{dataset_name} - NEAR-DUPLICATE ANALYSIS"
    )
    log("=" * 70)

    if not root.exists():

        log(
            f"ERROR: Directory missing: {root}"
        )

        return False

    files = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower()
        in IMAGE_EXTENSIONS
    ]

    log(
        f"Images found: {len(files)}"
    )

    # --------------------------------------------------------
    # For very large datasets, limit the number of comparisons.
    #
    # This avoids an enormous O(n²) operation.
    # --------------------------------------------------------

    if len(files) > 2500:

        log(
            "\nDataset is large."
        )

        log(
            "Using deterministic sampling "
            "for near-duplicate analysis."
        )

        step = max(
            1,
            len(files) // 2500
        )

        files = files[::step]

        log(
            f"Images sampled: {len(files)}"
        )

    # --------------------------------------------------------
    # Generate signatures
    # --------------------------------------------------------

    signatures = []

    failed = 0

    for path in files:

        signature = perceptual_signature(
            path
        )

        if signature is None:

            failed += 1
            continue

        signatures.append(
            (path, signature)
        )

    log(
        f"Valid signatures: "
        f"{len(signatures)}"
    )

    log(
        f"Signature failures: "
        f"{failed}"
    )

    # --------------------------------------------------------
    # Compare signatures
    # --------------------------------------------------------

    total_comparisons = 0
    similar_pairs = []

    n = len(signatures)

    for i in range(n):

        path_a, sig_a = signatures[i]

        for j in range(i + 1, n):

            if (
                total_comparisons
                >= max_comparisons
            ):

                break

            path_b, sig_b = signatures[j]

            distance = hamming_distance(
                sig_a,
                sig_b
            )

            total_comparisons += 1

            if (
                distance is not None
                and distance <= threshold
            ):

                similar_pairs.append(
                    (
                        distance,
                        path_a,
                        path_b
                    )
                )

        if (
            total_comparisons
            >= max_comparisons
        ):

            break

    similar_pairs.sort(
        key=lambda x: x[0]
    )

    log(
        f"\nComparisons performed: "
        f"{total_comparisons}"
    )

    log(
        f"Near-duplicate pairs "
        f"(threshold <= {threshold}): "
        f"{len(similar_pairs)}"
    )

    if total_comparisons >= max_comparisons:

        log(
            "NOTE: Comparison limit reached."
        )

        log(
            "This is a similarity screening, "
            "not an exhaustive pairwise scan."
        )

    if similar_pairs:

        log(
            "\nCLOSEST SIMILAR IMAGE PAIRS"
        )

        display_limit = min(
            100,
            len(similar_pairs)
        )

        for (
            distance,
            path_a,
            path_b
        ) in similar_pairs[
            :display_limit
        ]:

            log(
                f"\nDistance: {distance}"
            )

            log(
                f"  A: {path_a}"
            )

            log(
                f"  B: {path_b}"
            )

    else:

        log(
            "\nNo near-duplicate pairs "
            "were detected under the threshold."
        )

    return True


# ============================================================
# RUGD SEQUENCE-LOCAL SIMILARITY
# ============================================================

def analyze_rugd_sequences():

    log("\n" + "=" * 70)
    log("RUGD SEQUENCE-LOCAL SIMILARITY ANALYSIS")
    log("=" * 70)

    frames_root = RUGD_ROOT / "frames"

    if not frames_root.exists():

        log(
            f"ERROR: RUGD frames missing: "
            f"{frames_root}"
        )

        return False

    sequence_dirs = [
        p
        for p in frames_root.iterdir()
        if p.is_dir()
    ]

    total_pairs = 0
    similar_pairs = 0

    threshold = 3

    log(
        "Checking consecutive frames within "
        "each RUGD sequence."
    )

    log(
        "This is intentionally sequence-local "
        "because consecutive video frames "
        "are naturally similar."
    )

    for sequence_dir in sorted(
        sequence_dirs,
        key=lambda p: p.name
    ):

        frames = sorted(
            [
                p
                for p in sequence_dir.iterdir()
                if p.is_file()
                and p.suffix.lower()
                in IMAGE_EXTENSIONS
            ],
            key=lambda p: p.name
        )

        previous_signature = None

        sequence_similar = 0
        sequence_pairs = 0

        for frame_path in frames:

            signature = perceptual_signature(
                frame_path
            )

            if (
                previous_signature
                is not None
                and signature
                is not None
            ):

                distance = hamming_distance(
                    previous_signature,
                    signature
                )

                sequence_pairs += 1
                total_pairs += 1

                if (
                    distance is not None
                    and distance <= threshold
                ):

                    sequence_similar += 1
                    similar_pairs += 1

            previous_signature = signature

        percentage = (
            sequence_similar
            / sequence_pairs
            * 100
            if sequence_pairs > 0
            else 0
        )

        log(
            f"{sequence_dir.name:12s} "
            f"pairs={sequence_pairs:4d} "
            f"similar={sequence_similar:4d} "
            f"({percentage:6.2f}%)"
        )

    log(
        f"\nTotal consecutive pairs: "
        f"{total_pairs}"
    )

    log(
        f"Similar consecutive pairs: "
        f"{similar_pairs}"
    )

    percentage = (
        similar_pairs
        / total_pairs
        * 100
        if total_pairs > 0
        else 0
    )

    log(
        f"Similarity percentage: "
        f"{percentage:.2f}%"
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    log("=" * 70)
    log("V10.3 DUPLICATE & SIMILARITY ANALYSIS")
    log("UGV PROJECT")
    log("=" * 70)

    log(
        "\nAnalysis mode: READ-ONLY"
    )

    log(
        "No dataset files will be deleted, "
        "moved, renamed, or modified."
    )

    # --------------------------------------------------------
    # ROAD OBSTACLE
    # --------------------------------------------------------

    road_exact = analyze_exact_duplicates(
        "ROAD OBSTACLE",
        ROAD_ROOT
    )

    road_near = analyze_perceptual_similarity(
        "ROAD OBSTACLE",
        ROAD_ROOT,
        threshold=3
    )

    # --------------------------------------------------------
    # OFFROAD
    #
    # Only analyze images, not masks.
    # --------------------------------------------------------

    offroad_images = (
        OFFROAD_ROOT / "images"
    )

    offroad_exact = analyze_exact_duplicates(
        "OFFROAD IMAGES",
        offroad_images
    )

    offroad_near = analyze_perceptual_similarity(
        "OFFROAD IMAGES",
        offroad_images,
        threshold=3
    )

    # --------------------------------------------------------
    # RUGD
    #
    # Exact duplicates across frames.
    # --------------------------------------------------------

    rugd_exact = analyze_exact_duplicates(
        "RUGD FRAMES",
        RUGD_ROOT / "frames"
    )

    # --------------------------------------------------------
    # RUGD consecutive-frame similarity
    # --------------------------------------------------------

    rugd_sequence = (
        analyze_rugd_sequences()
    )

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    log("\n" + "=" * 70)
    log("V10.3 ANALYSIS STATUS")
    log("=" * 70)

    checks = [
        road_exact,
        road_near,
        offroad_exact,
        offroad_near,
        rugd_exact,
        rugd_sequence,
    ]

    if all(checks):

        log(
            "V10.3 DUPLICATE & SIMILARITY "
            "ANALYSIS: PASS"
        )

    else:

        log(
            "V10.3 DUPLICATE & SIMILARITY "
            "ANALYSIS: CHECK REQUIRED"
        )

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    REPORT_FILE.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    log(
        f"\nReport saved to:\n"
        f"{REPORT_FILE}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()